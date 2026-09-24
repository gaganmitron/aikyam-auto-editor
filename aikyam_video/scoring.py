"""Pluggable moment scoring (section Layer 3). Weights come from config; scorers are registered functions."""
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
import numpy as np

DEFAULT_WEIGHTS: Dict[str, float] = {
    "visualImportance": 0.25, "devotionalRelevance": 0.20, "audioImportance": 0.15,
    "semanticImportance": 0.15, "novelty": 0.10, "completeness": 0.10, "temporalImportance": 0.05,
    "aesthetic": 0.0,   # EXP-012: opt-in until measured end to end
}
DEVOTIONAL = {"deity": 1.0, "idol": 0.9, "aarti": 1.0, "abhishekam": 1.0, "priest": 0.7, "procession": 0.8,
              "lamps": 0.7, "flowers": 0.5, "devotees": 0.5, "temple_architecture": 0.4, "decorations": 0.4, "crowd": 0.4,
              "sanctum_view": 0.8, "ritual_hands": 0.7, "offerings": 0.5, "incense_smoke": 0.5, "temple_bell": 0.5,
              "devotees_walking": 0.5, "food_service": 0.5, "architectural_detail": 0.4, "temple_tank": 0.4, "temple_night": 0.4}

Scorer = Callable[[object, float, float], float]   # (ctx, start, end) -> 0..1
SCORERS: Dict[str, Scorer] = {}


def scorer(name: str):
    def deco(fn: Scorer) -> Scorer:
        SCORERS[name] = fn; return fn
    return deco


def _overlap_scenes(ctx, a, b):
    return [sv for sv in ctx.vision if sv.end > a and sv.start < b]


def _dominant_labels(ctx, a, b) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for sv in _overlap_scenes(ctx, a, b):
        for k, v in sv.vision.labels.items():
            out[k] = max(out.get(k, 0), v)
    return out


@scorer("visualImportance")
def visual(ctx, a, b):
    svs = _overlap_scenes(ctx, a, b)
    if not svs:
        return 0.0
    vals = []
    for sv in svs:
        v = sv.vision
        face = min(1.0, sum(f[2] * f[3] for f in v.faces) * 8)      # face area, ~12% of frame saturates
        light = v.labels.get("lamps", 0)
        clarity = v.sharpness * (1 - abs(v.brightness - 0.5) * 1.2)
        vals.append(0.4 * face + 0.25 * light + 0.35 * max(0.0, clarity))
    return float(np.clip(max(vals), 0, 1))


@scorer("aesthetic")
def aesthetic(ctx, a, b):
    """Look quality of the window: overlap-weighted WITHIN-VIDEO percentile of its scenes' zero-shot aesthetic score (percentile, because only the
    ordering inside one video is meaningful -- tools/bench_labels.py). Neutral 0.5 with no image-text model or fewer than 3 scenes."""
    raw = [sv.vision.aesthetic for sv in ctx.vision]
    if len(raw) < 3 or any(r is None for r in raw):
        return 0.5
    rank = getattr(ctx, "_aes_rank", None)
    if rank is None:
        arr = np.asarray(raw); rank = ctx._aes_rank = [float(((arr < v).sum() + 0.5 * (arr == v).sum()) / len(arr)) for v in arr]
    w = [(rank[i], min(b, sv.end) - max(a, sv.start)) for i, sv in enumerate(ctx.vision) if sv.end > a and sv.start < b]
    return float(sum(r * o for r, o in w) / sum(o for _, o in w)) if w else 0.5


@scorer("devotionalRelevance")
def devotional(ctx, a, b):
    labs = _dominant_labels(ctx, a, b)
    return float(max([DEVOTIONAL.get(k, 0) * v for k, v in labs.items()] or [0]))


DEVOTIONAL_SOUNDS = ("bell", "conch", "chant", "bhajan", "drums", "applause")


@scorer("audioImportance")
def audio(ctx, a, b):
    p = ctx.audio
    if not len(p.rms_db):
        return 0.0
    s = p.slice(a, b)
    if s.start >= len(p.rms_db):
        return 0.0
    peaks = min(1.0, float(p.peaks[s].sum()) / max(1.0, (b - a) / 5))   # ~1 onset per 5 s saturates
    tonal = float(p.tonal[s].mean())
    live = 1.0 - p.silence_ratio(a, b)
    if p.events:   # tagger available: devotional sound events carry most of the signal
        ev = max(p.event_score(k, a, b) for k in DEVOTIONAL_SOUNDS if k in p.events)
        return float(np.clip(0.2 * peaks + 0.1 * tonal + 0.25 * live + 0.45 * ev, 0, 1))
    return float(np.clip(0.4 * peaks + 0.3 * tonal + 0.3 * live, 0, 1))


@scorer("semanticImportance")
def semantic(ctx, a, b):
    hits = set()
    for seg, refs in ctx.segment_entities:
        if seg.end > a and seg.start < b:
            hits.update(r.entityId for r in refs)
    words = sum(len(seg.text.split()) for seg in ctx.transcript.segments if seg.end > a and seg.start < b)
    return float(np.clip(0.25 * len(hits) + min(0.25, words / 120), 0, 1))


@scorer("novelty")
def novelty(ctx, a, b):
    """Rare content relative to the whole recording: 1 - mean prevalence of this window's labels."""
    labs = _dominant_labels(ctx, a, b)
    if not labs or not ctx.label_prevalence:
        return 0.5
    return float(np.clip(1 - np.mean([ctx.label_prevalence.get(k, 0) for k in labs]), 0, 1))


@scorer("completeness")
def completeness(ctx, a, b):
    """Starts/ends near a scene boundary or speech boundary, and not mid-word."""
    bounds = [x for sv in ctx.vision for x in (sv.start, sv.end)] + \
             [x for s in ctx.transcript.segments for x in (s.start, s.end)]
    if not bounds:
        return 0.5
    d = lambda t: min(abs(t - x) for x in bounds)
    return float(np.clip(1 - (d(a) + d(b)) / 2.0, 0, 1))          # within 0s => 1, >=2s away => 0


@scorer("temporalImportance")
def temporal(ctx, a, b):
    dur = max(ctx.info.duration, 1e-6)
    m = (a + b) / 2 / dur
    return float(max(0.0, 1 - min(m, 1 - m) / 0.15))               # opening/closing 15% of the event


@dataclass
class ScoringConfig:
    weights: Dict[str, float]

    @classmethod
    def load(cls, path: Optional[str] = None) -> "ScoringConfig":
        w = dict(DEFAULT_WEIGHTS)
        if path:
            w.update(json.load(open(path)))
        unknown = set(w) - set(SCORERS)
        if unknown:
            raise ValueError(f"weights for unregistered scorers: {sorted(unknown)}")
        return cls(w)


def score(ctx, a: float, b: float, cfg: ScoringConfig):
    comps = {k: SCORERS[k](ctx, a, b) for k in cfg.weights}
    total = sum(cfg.weights[k] * comps[k] for k in comps) / (sum(cfg.weights.values()) or 1)
    return round(float(total), 4), {k: round(v, 3) for k, v in comps.items()}
