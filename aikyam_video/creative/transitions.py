"""Per-edge transition decisions from the actual pair of shots: visual similarity, exposure jump, energy at the boundary, sound continuity, pacing.
Every decision records its signals and a reason (Timeline.decisions), so an edit can be audited and re-planned by QC."""
from __future__ import annotations
from typing import Dict, List, Optional, Sequence
import numpy as np
from .model import Clip, Timeline
from .pacing import PacingProfile
from .shots import HOP, Population
from .story import similarity

LUMA_JUMP = 0.35         # >= 35% of the luma range: a visible exposure jump (night <-> lit interior); a dissolve alone shows a pumping flash
LEAD = 0.3               # J/L-cut audio lead in seconds: the classic 6-8 frame overlap
PRESENCE = 0.5           # CLAP devotional/speech score at which a sound counts as "there"


def _slot(clip: Clip, t: float) -> int:
    n = len(clip.shot.slots.luma)
    return int(np.clip((t - clip.shot.start) / HOP, 0, n - 1))


def edge_signals(a: Clip, b: Clip, pop: Population) -> Dict[str, float]:
    ia, ib = _slot(a, a.end - 1e-3), _slot(b, b.start)
    ea, eb = pop.energy(a.shot.slots)[ia], pop.energy(b.shot.slots)[ib]
    return {"similarity": similarity(a.shot, b.shot), "luma_gap": abs(a.shot.slots.luma[ia] - b.shot.slots.luma[ib]),
            "energy_out": float(ea), "energy_in": float(eb), "presence_out": float(a.shot.slots.presence[ia]), "presence_in": float(b.shot.slots.presence[ib])}


def decide(a: Clip, b: Clip, sig: Dict[str, float], profile: PacingProfile, sim_floor: float, beat: Optional[float] = None) -> dict:
    """One edge. Order of the rules matters: continuity first, then exposure, then energy/pacing, else a blend scaled to how different the shots are."""
    cap = min(a.length, b.length) / 3.0                                             # never let a transition eat its clips
    if sig["similarity"] >= max(sim_floor, 0.9) and a.shot.id != b.shot.id and sig["luma_gap"] < LUMA_JUMP:
        return {"type": "cut", "durationSeconds": 0.0, "reason": f"near-identical shots (sim {sig['similarity']:.2f}): a dissolve would ghost, a straight cut reads as continuity"}
    if sig["luma_gap"] >= LUMA_JUMP:
        d = round(min(max(profile.base_blend * 1.2, 0.3), 0.7, cap), 2)
        return {"type": "dip_black", "durationSeconds": d, "reason": f"exposure jump {sig['luma_gap']:.2f}: dipping through black hides the flash"}
    edge_energy = 0.5 * (sig["energy_out"] + sig["energy_in"])
    cut_score = profile.cut_bias * (0.5 + 0.5 * edge_energy) - 0.25 * (1 - sig["similarity"]) * (1 - profile.cut_bias)
    if cut_score >= 0.35 and min(sig["energy_out"], sig["energy_in"]) >= 0.4:
        return {"type": "cut", "durationSeconds": 0.0, "reason": f"energetic boundary (energy {edge_energy:.2f}) at a {profile.name} pace: cut keeps momentum"}
    d = profile.base_blend * (0.75 + 0.5 * (1 - min(sig["similarity"], 1.0))) * (1.15 - 0.3 * edge_energy)
    if beat:                                                                         # snap the blend to a half-beat when that moves it by < 30%
        half = beat / 2.0; k = max(1, round(d / half))
        if abs(k * half - d) <= 0.3 * d:
            d = k * half
    d = round(float(min(max(d, 0.25), 0.9, cap)), 2)
    return {"type": "crossfade", "durationSeconds": d, "reason": f"different shots (sim {sig['similarity']:.2f}) at calm energy {edge_energy:.2f}: a {d}s dissolve"}


def plan_transitions(tl: Timeline, pop: Population, profile: PacingProfile, beat: Optional[float] = None, source_duration=1e9) -> None:
    """Fills clip.transition_in and clip.audio_lead in place and appends to tl.decisions."""
    sim_floor = tl.meta.get("sim_floor", 1.0)
    for i in range(1, len(tl.clips)):
        a, b = tl.clips[i - 1], tl.clips[i]
        sig = edge_signals(a, b, pop); t = decide(a, b, sig, profile, sim_floor, beat)
        b.transition_in = t
        lead = 0.0
        if sig["presence_in"] >= PRESENCE and b.start - LEAD >= 0:                    # J-cut: the next clip's sound (chant, bhajan, bell) arrives just before its picture
            lead = LEAD
        elif sig["presence_out"] >= PRESENCE and a.shot.kind == "video" and a.end + LEAD <= (source_duration.get(a.shot.asset_id, 1e9) if isinstance(source_duration, dict) else source_duration) and sig["presence_in"] < PRESENCE:   # L-cut: the outgoing sound rings into the next picture
            lead = -LEAD
        b.audio_lead = lead
        tl.decisions.append({"type": "transition", "edge": i, "choice": t["type"], "seconds": t["durationSeconds"], "audioLead": lead,
                             "reason": t["reason"], "signals": {k: round(v, 3) for k, v in sig.items()}})
