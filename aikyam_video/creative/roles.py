"""Narrative roles: how well does each candidate shot serve OPENING / BUILDUP / RITUAL / REVEAL / CLIMAX / CLOSING?"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Sequence
import numpy as np
from .model import ROLES, Shot
from .shots import Population

CFG = json.loads((Path(__file__).parent.parent / "data" / "roles.json").read_text())
L = CFG["labels"]


def _lab(shot: Shot, group: str) -> float:
    return max([shot.labels.get(k, 0.0) for k in L[group]] or [0.0])


def aggregates(shot: Shot, pop: Population, position: float) -> Dict[str, float]:
    """Scalar descriptors of a shot, all in 0..1. `position` = where the shot sits in the source (0 start .. 1 end)."""
    sl = shot.slots; e = pop.energy(sl); q = pop.slot_quality(sl)
    trend = float(np.polyfit(np.arange(len(e)), e, 1)[0] * len(e)) if len(e) > 2 else 0.0        # rise across the shot
    peak_ev = max([shot.events.get(k, 0.0) for k in CFG["audio_peak_events"]] or [0.0])
    return {"quality": float(q.mean()), "energy": float(e.mean()), "energy_peak": float(e.max()), "rising": float(np.clip(0.5 + trend, 0, 1)),
            "closeness": float(pop.rank("concentration", np.asarray(sl.concentration)).mean()),
            "face": float(min(1.0, 8 * np.max(sl.face))) if len(sl.face) else 0.0, "position": position,
            "audio_peak": float(max(peak_ev, pop.rank("rms_db", np.asarray(sl.rms_db)).max() * 0.5))}


def affinities(shot: Shot, pop: Population, position: float, opening: str = "establish") -> Dict[str, float]:
    a = aggregates(shot, pop, position); W = CFG["weights"]
    ritual, devotion, setting, deity = _lab(shot, "ritual"), _lab(shot, "devotion"), _lab(shot, "setting"), _lab(shot, "deity")
    calm, wide = 1 - a["energy"], 1 - a["closeness"]
    d = {
        "OPENING": (lambda w: w["setting"] * setting + w["wide"] * wide + w["early"] * (1 - position) + w["calm"] * calm)(W["OPENING_establish"]) if opening == "establish"
        else (lambda w: w["quality"] * a["quality"] + w["focus"] * max(ritual, deity) + w["closeness"] * a["closeness"] + w.get("energy", 0.0) * a["energy"])(W["OPENING_hook"]),
        "BUILDUP": (lambda w: w["devotion"] * devotion + w["mid_energy"] * (1 - abs(a["energy"] - 0.5) * 2) + w["rising"] * a["rising"] + w["early"] * (1 - position))(W["BUILDUP"]),
        "RITUAL": (lambda w: w["ritual"] * ritual + w["energy"] * a["energy"] + w["quality"] * a["quality"])(W["RITUAL"]),
        "REVEAL": (lambda w: w["deity"] * deity + w["closeness"] * a["closeness"] + w["calm"] * calm + w["face"] * a["face"])(W["REVEAL"]),
        "CLIMAX": (lambda w: w["ritual"] * ritual + w["energy_peak"] * a["energy_peak"] + w["audio_peak"] * a["audio_peak"] + w["quality"] * a["quality"])(W["CLIMAX"]),
        "CLOSING": (lambda w: w["calm"] * calm + w["late"] * position + w["devotion_or_setting"] * max(devotion, setting) + w["wide"] * wide)(W["CLOSING"]),
    }
    return {k: float(np.clip(v, 0, 1)) for k, v in d.items()}
