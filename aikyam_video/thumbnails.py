"""Thumbnail candidates + scoring (section 15). All candidates are kept for later experimentation."""
from __future__ import annotations
import json, os
from typing import List
import cv2
from . import ffmpeg as ff
from .plan import seg_source
from .providers import VisionProvider
from .scoring import DEVOTIONAL

W = {"face": 0.30, "clarity": 0.25, "semantic": 0.20, "brightness": 0.15, "composition": 0.10}


def _score(v, plan_entities: bool) -> dict:
    face = min(1.0, sum(f[2] * f[3] for f in v.faces) * 8)
    deity = max([DEVOTIONAL.get(k, 0) * c for k, c in v.labels.items() if k in ("deity", "idol", "priest")] or [0])
    sem = max([DEVOTIONAL.get(k, 0) * c for k, c in v.labels.items()] or [0])
    comp = 0.5
    if v.faces:  # rule of thirds / centre: reward faces near x in {1/3, 1/2, 2/3}
        f = max(v.faces, key=lambda f: f[2] * f[3]); cx = f[0] + f[2] / 2
        comp = 1 - min(abs(cx - t) for t in (1 / 3, 0.5, 2 / 3)) * 3
    return {"face": max(face, deity), "clarity": v.sharpness, "semantic": sem,
            "brightness": max(0.0, 1 - abs(v.brightness - 0.55) * 2.2), "composition": max(0.0, comp)}


def generate_thumbnails(src: str, plan: dict, vision: VisionProvider, out_dir: str, n: int = 12) -> dict:
    tdir = os.path.join(out_dir, "thumbnails"); os.makedirs(tdir, exist_ok=True)
    times: List[tuple] = []                                    # (file, time) over every VIDEO segment; stills are not thumbnail candidates
    for s in plan["segments"]:
        f = seg_source(plan, s, src)
        if f is None:
            continue
        L = s["end"] - s["start"]
        k = max(1, round(n * L / max(plan["durationSeconds"], 1e-6)))
        times += [(f, s["start"] + L * (j + 1) / (k + 1)) for j in range(k)]
    cands = []
    for i, (f, t) in enumerate(times):
        p = os.path.join(tdir, f"thumb_{i}.jpg")
        ff.frame_at(f, t, p)
        img = cv2.imread(p)
        if img is None:
            continue
        comps = _score(vision.analyze(img), True)
        cands.append({"file": p, "timestamp": round(t, 2), "score": round(sum(W[k] * comps[k] for k in W), 4),
                      "components": {k: round(v, 3) for k, v in comps.items()}})
    if not cands:
        raise RuntimeError("no thumbnail candidates could be extracted")
    best = max(cands, key=lambda c: c["score"])
    final = os.path.join(out_dir, "thumbnail.jpg")
    cv2.imwrite(final, cv2.imread(best["file"]), [cv2.IMWRITE_JPEG_QUALITY, 92])
    res = {"selected": best["file"], "timestamp": best["timestamp"], "weights": W, "candidates": cands}
    json.dump(res, open(os.path.join(tdir, "scores.json"), "w"), indent=2)
    return res
