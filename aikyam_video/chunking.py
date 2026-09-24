"""Long-video analysis (README Sec.8's known limitation: `process` used to decode the whole file
at once -- multi-hour videos need a big machine). For a source >= `o.long_video_threshold_s`,
`pipeline.run` calls `run_chunked` here instead of analyzing the whole file directly.

Strategy, deliberately the SAME one `live.py` already uses for streams (not the transcript-only
chunker some reference implementations use): split into overlapping PHYSICAL sub-clip files, run
the existing, unmodified single-file pipeline (`pipeline.run(..., upto="highlights", ...)`) on each
chunk independently, then merge the per-chunk artifacts into one global-timeline result. Everything
downstream of this module (story planner, render, QC) is unmodified and unaware chunking happened --
`stages.plan`/`stages.render` still run once, on the merged artifacts, against the ORIGINAL file.

Timestamp discipline (GLOBAL_SOURCE_TIME / CHUNK_LOCAL_TIME): each chunk's stage functions see ONLY
chunk-local time (t=0..chunk_duration, the small physical sub-clip file). This module is the ONLY
place chunk-local time is converted to global source time -- exactly once, right after that chunk's
stages finish, before anything is written to the merged output. Nothing downstream ever sees a
chunk-local timestamp.

Overlap handling: consecutive chunks share `chunk_overlap_seconds` of real footage so nothing near a
seam is missed by either neighbour's analysis. Each chunk then contributes only its OWN, non-overlapping
half of that shared footage to the merge (first/last chunk keep their outer edge in full) -- so the
overlap region is analyzed twice but used once, never duplicated in the merged output. A final
moment-level dedupe (identical rule to live.py's) is a second safety net at the seams.
"""
from __future__ import annotations
import json, os, tempfile
from typing import Dict, List
import numpy as np
from . import pipeline, stages
from .audio import AudioProfile
from .live import _overlap, grab_chunk
from .options import Options


def _keep(items: List[dict], overlap_s: float, is_first: bool, is_last: bool, local_dur: float, key: str = "start") -> List[dict]:
    """This chunk's OWNED, non-overlapping slice of chunk-local time: [0, dur) for the first chunk's
    start and the last chunk's end (nothing beyond the real video to hand off to), otherwise trimmed
    by half the overlap on each side (the other half is the neighbouring chunk's job)."""
    lo = 0.0 if is_first else overlap_s / 2
    hi = local_dur if is_last else local_dur - overlap_s / 2
    return [x for x in items if lo <= x[key] < hi]


def _offset(items: List[dict], by: float, keys=("start", "end")) -> List[dict]:
    out = []
    for x in items:
        y = dict(x)
        for k in keys:
            if k in y and y[k] is not None:
                y[k] = round(y[k] + by, 4)
        out.append(y)
    return out


def run_chunked(path: str, out: str, o: Options, log, duration: float) -> None:
    chunk_s, overlap_s = o.chunk_seconds, o.chunk_overlap_seconds
    step = max(1.0, chunk_s - overlap_s)
    starts: List[float] = []
    t = 0.0
    while t < duration:
        starts.append(t); t += step
    log.info(f"long video ({duration:.0f}s): {len(starts)} chunks of ~{chunk_s:.0f}s (overlap {overlap_s:.0f}s)", extra={"stage": "chunking"})

    all_segs, all_scenes, all_vision, all_black, all_moments = [], [], [], [], []
    all_entities: Dict[str, list] = {}
    all_embeddings: Dict[str, list] = {}
    audio_parts = {"rms_db": [], "silence": [], "peaks": [], "tonal": []}
    events_parts: Dict[str, List[np.ndarray]] = {}
    hop, language = None, "en"

    with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(out)) or None) as tmp:
        for i, cstart in enumerate(starts):
            is_first, is_last = i == 0, i == len(starts) - 1
            cdur = min(chunk_s, duration - cstart)
            chunk_file = os.path.join(tmp, f"chunk_{i:04d}.mp4")
            got = grab_chunk(path, cstart, cdur, chunk_file, realtime=False)
            if got < max(1.0, o.min_source_seconds):
                continue                                              # too little real footage in this chunk to analyze
            cdir = os.path.join(tmp, f"chunk_{i:04d}")
            # a chunk can itself be >= long_video_threshold_s (chunk_seconds often IS >= the threshold
            # that triggered chunking in the first place) -- without this it recurses into chunking
            # its own chunks, forever, since chunk_seconds/overlap never shrink between levels.
            co = Options(**{**o.__dict__, "on_stage": None, "long_video_threshold_s": float("inf")})
            pipeline.run(chunk_file, cdir, "highlights", co, log=log)  # the EXISTING single-file pipeline, unmodified, run ONCE per physical chunk

            tr = json.load(open(os.path.join(cdir, stages.TRANSCRIPT)))
            language = tr.get("language", language)
            all_segs += _offset(_keep(tr.get("segments", []), overlap_s, is_first, is_last, got), cstart)

            scenes = _keep(json.load(open(os.path.join(cdir, stages.SCENES))), overlap_s, is_first, is_last, got)
            id_map = {}
            for s in scenes:
                new_id = f"c{i}_{s['sceneId']}"; id_map[s["sceneId"]] = new_id; s["sceneId"] = new_id
            all_scenes += _offset(scenes, cstart)

            vision = [v for v in json.load(open(os.path.join(cdir, stages.VISION))) if v["sceneId"] in id_map]
            for v in vision:
                v["sceneId"] = id_map[v["sceneId"]]
            all_vision += _offset(vision, cstart)

            emb_path = os.path.join(cdir, stages.EMBEDDINGS)
            if os.path.exists(emb_path):
                emb = json.load(open(emb_path))
                all_embeddings.update({new: emb[old] for old, new in id_map.items() if old in emb})

            black = _keep([{"start": b[0], "end": b[1]} for b in json.load(open(os.path.join(cdir, stages.BLACK)))],
                          overlap_s, is_first, is_last, got)
            all_black += [[b["start"] + cstart, b["end"] + cstart] for b in black]

            ents = json.load(open(os.path.join(cdir, stages.ENTITIES)))
            for etype, refs in ents.items():
                all_entities.setdefault(etype, []).extend(refs)

            # moment.sceneIds is write-only downstream (nothing resolves it by exact id -- vision/scene
            # lookups elsewhere are all by TIME overlap), so it's left as this chunk's own local ids
            # rather than remapped: a real gap if that ever changes, harmless as long as it doesn't.
            moms = _keep(json.load(open(os.path.join(cdir, stages.MOMENTS)))["moments"], overlap_s, is_first, is_last, got)
            for m in moms:
                m["momentId"] = f"c{i}_{m['momentId']}"
            all_moments += _offset(moms, cstart)

            ap = AudioProfile.load(os.path.join(cdir, stages.AUDIO))
            hop = hop or ap.hop
            lo_i = 0 if is_first else int(round((overlap_s / 2) / ap.hop))
            hi_i = len(ap.rms_db) if is_last else len(ap.rms_db) - int(round((overlap_s / 2) / ap.hop))
            for name, arr in (("rms_db", ap.rms_db), ("silence", ap.silence), ("peaks", ap.peaks), ("tonal", ap.tonal)):
                audio_parts[name].append(arr[lo_i:hi_i])
            for name, arr in ap.events.items():
                events_parts.setdefault(name, []).append(arr[lo_i:hi_i])

    all_moments.sort(key=lambda m: -m["score"])                       # same IoU dedupe rule as live.py: a second safety net at the seams
    kept: List[dict] = []
    for m in all_moments:
        if not any(_overlap((m["start"], m["end"]), (k["start"], k["end"])) > 0.3 for k in kept):
            kept.append(m)
    kept.sort(key=lambda m: m["start"])

    json.dump({"language": language, "language_probability": 0.0, "segments": sorted(all_segs, key=lambda s: s["start"])},
              open(os.path.join(out, stages.TRANSCRIPT), "w"), indent=2, ensure_ascii=False)
    json.dump(sorted(all_scenes, key=lambda s: s["start"]), open(os.path.join(out, stages.SCENES), "w"), indent=2)
    json.dump(sorted(all_vision, key=lambda v: v["start"]), open(os.path.join(out, stages.VISION), "w"), indent=2)
    json.dump(all_embeddings, open(os.path.join(out, stages.EMBEDDINGS), "w"))
    json.dump(all_black, open(os.path.join(out, stages.BLACK), "w"))
    json.dump(all_entities, open(os.path.join(out, stages.ENTITIES), "w"), indent=2, ensure_ascii=False)
    json.dump({"moments": kept, "rejected": []}, open(os.path.join(out, stages.MOMENTS), "w"), indent=2, ensure_ascii=False)
    AudioProfile(hop or 0.5, *(np.concatenate(audio_parts[k]) if audio_parts[k] else np.array([]) for k in ("rms_db", "silence", "peaks", "tonal")),
                {n: np.concatenate(a) for n, a in events_parts.items()}).save(os.path.join(out, stages.AUDIO))
    log.info(f"long video: merged {len(kept)} moments from {len(starts)} chunks", extra={"stage": "chunking"})
