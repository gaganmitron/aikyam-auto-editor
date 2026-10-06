"""Layer 1: run vision over scene keyframes + sub-samples."""
from __future__ import annotations
from typing import List
import cv2
import numpy as np
from .models import Scene, SceneVision, VisionResult
from .providers import VisionProvider
from . import ffmpeg as ff
import os, tempfile


def _merge(rs: List[VisionResult]) -> VisionResult:
    labels = {}
    for r in rs:
        for k, v in r.labels.items():
            labels[k] = max(labels.get(k, 0), v)
    n = len(rs)
    faces = max((r.faces for r in rs), key=len)
    xs = sorted(r.saliency_x for r in rs)
    mod, deities = {}, {}
    for r in rs:
        for k, v in r.moderation.items(): mod[k] = max(mod.get(k, 0), v)
        for k, v in r.deities.items(): deities[k] = max(deities.get(k, 0), v)
    embs = [r.embedding for r in rs if r.embedding]
    emb = None
    if embs:
        e = np.mean(embs, axis=0); emb = (e / np.linalg.norm(e)).tolist()
    return VisionResult(labels=labels, faces=faces, saliency_x=xs[n // 2],
                        brightness=sum(r.brightness for r in rs) / n, sharpness=sum(r.sharpness for r in rs) / n,
                        provider=rs[0].provider, moderation=mod, deities=deities, embedding=emb,
                        aesthetic=(sum(r.aesthetic for r in rs) / n) if all(r.aesthetic is not None for r in rs) else None,
                        no_text=(sum(r.no_text for r in rs) / n) if all(r.no_text is not None for r in rs) else None,
                        beats=({k: sum(r.beats[k] for r in rs) / n for k in rs[0].beats} if all(r.beats for r in rs) else None),
                        shot_size=({k: sum(r.shot_size[k] for r in rs) / n for k in rs[0].shot_size} if all(r.shot_size for r in rs) else None))


def analyze_scenes(path: str, scenes: List[Scene], vision: VisionProvider, samples: int = 3) -> List[SceneVision]:
    """Analyse `samples` frames per scene (25/50/75%) and merge: label = max, saliency = median."""
    out = []
    with tempfile.TemporaryDirectory() as td:
        for s in scenes:
            rs = []
            for j in range(samples):
                t = s.start + (s.end - s.start) * (j + 1) / (samples + 1)
                f = os.path.join(td, "f.jpg"); ff.frame_at(path, t, f)
                img = cv2.imread(f)
                if img is not None:
                    rs.append(vision.analyze(img))
            if rs:
                out.append(SceneVision(sceneId=s.sceneId, start=s.start, end=s.end, vision=_merge(rs)))
    return out


def text_track(path: str, vision: VisionProvider, step: float = 2.0) -> List[List[float]]:
    """[[t, no_text]] every `step` s over the WHOLE file. The per-scene keyframes (3 per scene, averaged) miss a graphic that sits for a few seconds inside a long scene
    (a 56 s drone scene hid a "Leave a like!" animation). [] when the provider has no no_text score (heuristic vision)."""
    score = getattr(vision, "no_text_score", None)
    if score is None:
        return []
    with tempfile.TemporaryDirectory() as td:
        ff.run(["-i", path, "-vf", f"fps=1/{step},scale=384:-2", "-an", "-q:v", "4", os.path.join(td, "%06d.jpg")])
        names = sorted(os.listdir(td)); out = []
        for i, n in enumerate(names):
            img = cv2.imread(os.path.join(td, n))
            if img is not None: out.append([round(i * step + step / 2, 2), round(score(img), 3)])
    return out
