"""Temporal event-boundary detection from scene embeddings (EXP-004,
docs/research/experiment_matrix.md): consecutive scenes whose SigLIP embedding
similarity drops sharply are candidate event boundaries. Validated on real footage
(Diwali: the clip's real day/night event transition ranked at the 6.7th percentile
of within-clip consecutive-scene similarities; 2-means clustering on the same
embeddings scored 95% purity against hand-verified ground truth).

NOT wired into the story planner yet -- detection only. See the experiment log
before connecting this to creative/story.py.
"""
from __future__ import annotations
from typing import List, Sequence, Tuple
import numpy as np
from ..models import SceneVision


def scene_similarities(vision: Sequence[SceneVision]) -> List[Tuple[float, float]]:
    """Cosine similarity between each pair of temporally consecutive scenes that have an embedding.
    Returns (boundary_time, similarity) pairs; boundary_time is the later scene's start. Scenes
    without an embedding are skipped (their neighbours are compared directly across the gap)."""
    pts = [(v.start, np.asarray(v.vision.embedding, dtype=float)) for v in sorted(vision, key=lambda v: v.start) if v.vision.embedding]
    out = []
    for (_, a), (t1, b) in zip(pts, pts[1:]):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            continue
        out.append((t1, float((a / na) @ (b / nb))))
    return out


def event_boundaries(vision: Sequence[SceneVision], percentile: float = 15.0) -> List[float]:
    """Candidate event-boundary timestamps: scene transitions whose similarity falls in the
    bottom `percentile` of THIS clip's own consecutive-scene similarities. Self-relative, not a
    fixed cosine cutoff -- one real clip isn't enough to trust a magic absolute constant, but a
    clip's own similarity distribution adapts to its content automatically. Too few scenes (<4
    transitions) to make a percentile meaningful -> no boundaries reported, not a guess."""
    sims = scene_similarities(vision)
    if len(sims) < 4:
        return []
    thresh = float(np.percentile([s for _, s in sims], percentile))
    return [t for t, s in sims if s <= thresh]


def event_windows(vision: Sequence[SceneVision], source_duration: float, percentile: float = 15.0) -> List[Tuple[float, float]]:
    """Boundaries turned into contiguous (start, end) windows spanning the whole source."""
    edges = [0.0] + sorted(event_boundaries(vision, percentile)) + [source_duration]
    return [(a, b) for a, b in zip(edges, edges[1:]) if b > a]
