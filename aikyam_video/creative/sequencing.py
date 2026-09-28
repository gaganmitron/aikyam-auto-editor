"""Transition costs for the story beam (EXP-026 / E4, docs/research/SEQUENCE_SEARCH.md §4.3): how a shot reads AFTER the one before it.

Each term is in 0..1 and is computed from the shot cards only (shot size, energy, brightness, embedding); every weight defaults to 0, so with no weights the planner is byte-identical to before.
Only signals that were measured reliable are used: shot size is trusted for "wide vs tighter" (EXP-025: medium vs close is at chance), so `size_repeat` never separates medium from close.

  size_repeat       the same coarse size twice in a row (wide-wide 1.0; two tighter shots 0.5: they may or may not be the same size)
  energy_jump       a large change of energy between neighbours, except into a CLIMAX (the arc asks for it there)
  motion_unrelated  two energetic shots from DIFFERENT videos whose pictures are unrelated (FilmGPT's failure: "cuts on action between two unrelated videos")
  luma_jump         a large brightness change at the cut (a flash)
  same_source_repeat  two clips in a row from the SAME video, in a reel that draws on several videos (EXP-026 lead / E4b: the user preferred reels with fewer of them; 5 of 10 pairs, not significant)"""
from __future__ import annotations
from typing import Callable, Dict, Optional
import numpy as np

TERMS = ("size_repeat", "energy_jump", "motion_unrelated", "luma_jump", "same_source_repeat")
ENERGY_FREE = 0.35          # energy differences up to this are normal
UNRELATED_SIM = 0.6         # embedding similarity below this counts as "unrelated" (1 - sim / this)
LUMA_FREE = 0.15            # brightness differences up to this are normal


def _wide(s) -> Optional[bool]:
    from ..shotsize import tightness
    return None if not getattr(s, "shot_size", None) else tightness(s.shot_size) < 0.5


def _luma(s) -> Optional[float]:
    l = getattr(getattr(s, "slots", None), "luma", None)
    return float(np.mean(l)) if l is not None and len(l) else None


def terms(prev, s, role: str, energy: Dict[str, float], similarity: Callable, multi: bool = True) -> Dict[str, float]:
    """The four terms for cutting from `prev` to `s` (both Shots), each 0..1; an unknown signal gives 0 (never a penalty on missing data)."""
    out = {k: 0.0 for k in TERMS}
    wp, wc = _wide(prev), _wide(s)
    if wp is not None and wc is not None and wp == wc:
        out["size_repeat"] = 1.0 if wc else 0.5
    ep, ec = energy.get(prev.id), energy.get(s.id)
    if ep is not None and ec is not None and role != "CLIMAX":
        out["energy_jump"] = float(np.clip((abs(ep - ec) - ENERGY_FREE) / (1.0 - ENERGY_FREE), 0.0, 1.0))
    if ep is not None and ec is not None and prev.asset_id != s.asset_id and prev.kind == "video" and s.kind == "video":
        out["motion_unrelated"] = float(min(ep, ec) * np.clip(1.0 - similarity(prev, s) / UNRELATED_SIM, 0.0, 1.0))
    if multi and prev.kind == "video" and s.kind == "video" and prev.asset_id == s.asset_id:            # only when the reel CAN vary its source: in a one-video reel every cut would carry the same constant
        out["same_source_repeat"] = 1.0
    lp, lc = _luma(prev), _luma(s)
    if lp is not None and lc is not None:
        out["luma_jump"] = float(np.clip((abs(lp - lc) - LUMA_FREE) / 0.3, 0.0, 1.0))
    return out


def cost(prev, s, role: str, energy: Dict[str, float], similarity: Callable, weights: Dict[str, float], multi: bool = True) -> float:
    """Weighted sum of the terms; 0 when no weight is set."""
    if not weights or not any(weights.get(k, 0.0) for k in TERMS):
        return 0.0
    t = terms(prev, s, role, energy, similarity, multi)
    return float(sum(weights.get(k, 0.0) * v for k, v in t.items()))
