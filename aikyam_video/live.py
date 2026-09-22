"""Live stream -> rolling highlight detection (Phase 2).

Near-real-time by chunking: pull `chunk_s` seconds of the stream (HLS/RTMP/file), run analysis + highlight ranking on that
chunk, emit moments with STREAM-GLOBAL timestamps to a JSONL feed, and (optionally) render a short reel for each strong
moment. Latency ~= chunk length + processing time. Chunks overlap by `overlap_s` so a ritual straddling a boundary is seen
whole; moments overlapping an already-emitted one are dropped (dedupe by IoU).
ponytail: sequential chunks in one process; run one process per stream, scale by streams."""
from __future__ import annotations
import json, os, subprocess, time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional
from . import ffmpeg as ff, pipeline, stages
from .options import Options


@dataclass
class LiveConfig:
    chunk_s: float = 60.0
    overlap_s: float = 10.0
    min_score: float = 0.4            # emit threshold
    render: bool = True               # render a short reel for each emitted moment
    max_chunks: Optional[int] = None
    dedupe_overlap: float = 0.3     # drop a moment overlapping an emitted one by more than this fraction of the shorter clip


def _overlap(a, b):
    """Overlap as a fraction of the SHORTER clip (IoU lets a clip that is 40% covered by a neighbour through)."""
    i = max(0, min(a[1], b[1]) - max(a[0], b[0])); m = min(a[1] - a[0], b[1] - b[0])
    return i / m if m > 0 else 0


def grab_chunk(source: str, start: float, dur: float, out: str, realtime: bool) -> float:
    """Extract [start, start+dur) into `out`. For a live URL (realtime) `start` is ignored: we record the next `dur` seconds.
    Returns the actual duration captured (0 = stream ended)."""
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if not realtime:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", source, "-t", f"{dur:.3f}", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-c:a", "aac", out]
    subprocess.run(cmd, capture_output=True)
    try:
        return ff.probe(out).duration
    except Exception:
        return 0.0


def run_live(source: str, out_dir: str, o: Optional[Options] = None, cfg: Optional[LiveConfig] = None,
             on_moment: Optional[Callable[[dict], None]] = None) -> List[dict]:
    o, cfg = o or Options(), cfg or LiveConfig()
    realtime = "://" in source and not source.startswith("file://")
    os.makedirs(out_dir, exist_ok=True)
    feed = os.path.join(out_dir, "live_highlights.jsonl")
    emitted: List[dict] = []
    pos, i = 0.0, 0                      # pos = stream-global time of the current chunk's start
    while cfg.max_chunks is None or i < cfg.max_chunks:
        d = os.path.join(out_dir, f"chunk_{i:04d}"); os.makedirs(d, exist_ok=True)
        chunk = os.path.join(d, "chunk.mp4")
        t0 = time.time()
        got = grab_chunk(source, pos, cfg.chunk_s, chunk, realtime)
        if got < o.min_source_seconds:
            break                        # stream ended (or nothing usable left)
        co = Options(**{**o.__dict__, "on_stage": None, "formats": ["reel"]})
        try:
            pipeline.run(chunk, d, "highlights", co)
            moments = json.load(open(os.path.join(d, stages.MOMENTS)))["moments"]
        except Exception as e:           # a bad chunk must not kill the stream
            with open(os.path.join(d, "error.txt"), "w") as f: f.write(str(e))
            moments = []
        for m in moments:
            g = {"chunk": i, "momentId": m["momentId"], "start": round(pos + m["start"], 2), "end": round(pos + m["end"], 2),
                 "score": m["score"], "reason": m["reason"], "entities": m["entities"], "detectedAfterSeconds": round(time.time() - t0, 1)}
            if m["score"] < cfg.min_score or any(_overlap((g["start"], g["end"]), (e["start"], e["end"])) > cfg.dedupe_overlap for e in emitted):
                continue
            if cfg.render:
                try:                     # plan + render just this moment out of the chunk
                    json.dump({"moments": [m], "rejected": []}, open(os.path.join(d, stages.MOMENTS), "w"))
                    c2 = Options(**{**co.__dict__, "target_seconds": min(30.0, m["end"] - m["start"])})
                    stages.plan(chunk, d, c2); g["reel"] = stages.render(chunk, d, c2).get("reel")
                except Exception as e:   # noqa: BLE001
                    g["renderError"] = str(e)[:200]
            emitted.append(g)
            with open(feed, "a", encoding="utf-8") as f:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
            if on_moment: on_moment(g)
        i += 1
        if not realtime and got < cfg.chunk_s - 0.5:
            break                        # end of file
        pos += got - (0.0 if realtime else cfg.overlap_s)   # a live stream can't be re-read, so no overlap there
    return emitted
