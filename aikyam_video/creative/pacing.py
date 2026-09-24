"""Pacing: choose a profile from what the footage IS (semantic labels + sound), and adapt shot length to its energy."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence
import numpy as np
from .model import Shot

_DATA = json.loads((Path(__file__).parent.parent / "data" / "pacing.json").read_text())
FESTIVE = {"procession", "ritual_dance", "fireworks", "crowd", "decorations"}
CALM = {"deity", "idol", "priest", "abhishekam", "aarti", "lamps", "devotees", "incense_smoke", "offerings", "sanctum_view", "ritual_hands"}


@dataclass(frozen=True)
class PacingProfile:
    name: str
    base_shot: float
    min_shot: float
    max_shot: float
    target_min: float
    target_max: float
    opening: str
    cut_bias: float          # 0 = always blend, 1 = always cut
    base_blend: float        # nominal crossfade seconds


def profile(name: str) -> PacingProfile:
    p = _DATA["profiles"][name]
    return PacingProfile(name, **{k: v for k, v in p.items() if k != "why"})


def choose_profile(shots: Sequence[Shot], override: Optional[str] = None) -> PacingProfile:
    """festive / contemplative / devotional from the score-weighted label mix and drums/applause vs bhajan/chant/bell sound."""
    if override:
        return profile(override)
    if not shots:
        return profile("devotional")
    w = np.array([max(s.score, 1e-3) for s in shots]); w = w / w.sum()
    fest = sum(wi * max([s.labels.get(k, 0.0) for k in FESTIVE] or [0]) for wi, s in zip(w, shots))
    calm = sum(wi * max([s.labels.get(k, 0.0) for k in CALM] or [0]) for wi, s in zip(w, shots))
    fest += 0.5 * sum(wi * max(s.events.get("drums", 0.0), s.events.get("applause", 0.0)) for wi, s in zip(w, shots))
    calm += 0.5 * sum(wi * max(s.events.get("bell", 0.0), s.events.get("chant", 0.0), s.events.get("bhajan", 0.0)) for wi, s in zip(w, shots))
    if fest > calm * 1.15:
        return profile("festive")
    if calm > fest * 1.6:
        return profile("contemplative")
    return profile("devotional")


def shot_length(p: PacingProfile, energy: float) -> float:
    """Higher energy -> shorter shots: base * (1 - 0.4*(energy-0.5)*2)... energy in [0,1], clamped to the profile's bounds."""
    return float(min(max(p.base_shot * (1.0 - 0.4 * (2 * energy - 1)), p.min_shot), p.max_shot))


def target_duration(p: PacingProfile, available: float) -> float:
    """Reel length: the profile's upper target when there is plenty of footage, shrinking toward its lower target (never below) when short."""
    return float(min(p.target_max, max(min(p.target_min, available), 0.85 * available)))
