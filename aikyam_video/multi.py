"""Mixed inputs: any number of videos + images (+ optional user music) -> ONE reel.

Every video runs the normal analysis stages in its own folder (assets/<id>/), every image is measured and embedded once; the creative engine then
pools all the shots, so the story may cut between a long recording, short clips and photographs. Same code path as the single-video engine."""
from __future__ import annotations
import os
from typing import Dict, List, Optional, Tuple
from . import ffmpeg as ff, stages
from .audio import AudioProfile
from .creative import engine as E
from .creative.shots import image_shot
from .models import EntityRef, Moment, Transcript
from .options import Options

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}


def classify(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXT: return "image"
    if ext in AUDIO_EXT: return "audio"
    return "video"


def _merge_entities(all_: List[Dict[str, List[EntityRef]]]) -> Dict[str, List[EntityRef]]:
    out: Dict[str, Dict[str, EntityRef]] = {}
    for d in all_:
        for typ, refs in d.items():
            for r in refs:
                cur = out.setdefault(typ, {}).get(r.entityId)
                if cur is None or r.confidence > cur.confidence: out[typ][r.entityId] = r
    return {t: sorted(v.values(), key=lambda r: -r.confidence) for t, v in out.items()}


def ingest(inputs: List[str], art: str, o: Options) -> Tuple[List[E.Source], List[str]]:
    """Analyse every input. Returns (sources in input order, audio inputs). Ids: v1.. for videos, i1.. for images."""
    srcs, audios, nv, ni = [], [], 0, 0
    for p in inputs:
        if not os.path.isfile(p): raise FileNotFoundError(p)
        kind = classify(p)
        if kind == "audio": raise ValueError(f"{p}: pass music with --music-file (and --i-own-the-music-rights), not as an input")
        if kind == "image":
            ni += 1; srcs.append(E.Source(f"i{ni}", p, "image", None, shot=image_shot(p, f"i{ni}", stages._provider("vision", o.vision)))); continue
        nv += 1; aid = f"v{nv}"; d = os.path.join(art, "assets", aid); os.makedirs(d, exist_ok=True)
        stamp, sf = {"path": os.path.abspath(p), "size": os.path.getsize(p), "mtime": int(os.path.getmtime(p)), "allow_silent": o.allow_silent}, os.path.join(d, "source.json")
        if not (os.path.exists(sf) and stages._load(d, "source.json") == stamp and os.path.exists(os.path.join(d, stages.MOMENTS))):   # analysis of an unchanged file is reused
            stages.transcription(p, d, o); stages.scene_analysis(p, d, o); stages.highlights(p, d, o); stages._dump(sf, stamp)
        _, vis, audio, _ = stages._load_analysis(d)
        srcs.append(E.Source(aid, p, "video", ff.probe(p), [Moment(**m) for m in stages._load(d, stages.MOMENTS)["moments"]], vis, audio,
                             Transcript(**stages._load(d, stages.TRANSCRIPT))))
    if not any(s.kind == "video" for s in srcs): raise ValueError("need at least one video input")
    return srcs, audios


def plan(inputs: List[str], art: str, o: Options) -> dict:
    srcs, audios = ingest(inputs, art, o)
    import sys
    print("assets: " + ", ".join(f"{x.id}={os.path.basename(x.path)}" for x in srcs), file=sys.stderr)
    vids = [s for s in srcs if s.kind == "video"]
    ents = _merge_entities([stages._load_entities(os.path.join(art, "assets", s.id)) for s in vids])
    stages._dump(os.path.join(art, stages.ENTITIES), {k: [r.model_dump() for r in v] for k, v in ents.items()})
    ctx = E.Ctx(vids[0].path, vids[0].info, vids[0].moments, vids[0].vision, vids[0].audio, vids[0].transcript, ents, o, o.video_id or "mix", sources=srcs)
    plans = E.build_plans(ctx)
    for k, pl in enumerate(plans):
        stages._dump(os.path.join(art, stages.PLAN if k == 0 else f"edit-plan-{k + 1}.json"), pl)
    return plans[0]
