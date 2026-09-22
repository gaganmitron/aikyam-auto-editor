"""Deterministic re-edit from QC findings. Each failing check maps to ONE targeted fix; the same report always yields the same fix (no randomness).
Fixes are recorded in plan.creative.qc.history so an edit can be audited. Returns (new plan or None if nothing can be done, description of the fixes, remix params)."""
from __future__ import annotations
import copy
from typing import Dict, List, Optional, Tuple
from ..plan import PlanError, edge_overlaps, output_duration, validate_plan
from .qc import QCReport

MIN_CLIP = 1.5            # never trim a clip below this


def _rebuild(plan: dict, segs: List[dict], src_dur: float) -> Optional[dict]:
    """Recompute the plan's totals/cues after the segment list changed. None if the result is invalid."""
    from ..captions import build_cues
    p = copy.deepcopy(plan); p["segments"] = segs
    if segs: segs[0].pop("transitionIn", None)
    p["durationSeconds"] = round(output_duration(segs, p["transitions"]), 2)
    return p


def _drop(plan: dict, i: int) -> Optional[dict]:
    segs = [s for k, s in enumerate(plan["segments"]) if k != i]
    if not segs:
        return None
    return _rebuild(plan, copy.deepcopy(segs), plan["source"]["durationSeconds"])


def _set_edge(plan: dict, edge_into: int, kind: str, d: float, reason: str) -> Optional[dict]:
    p = copy.deepcopy(plan)
    if not (1 <= edge_into < len(p["segments"])):
        return None
    p["segments"][edge_into]["transitionIn"] = {"type": kind, "durationSeconds": d, "reason": reason}
    p["durationSeconds"] = round(output_duration(p["segments"], p["transitions"]), 2)
    return p


def apply_fixes(plan: dict, report: QCReport, transcript=None, entities=None, retimer=None) -> Tuple[Optional[dict], List[str], Dict]:
    """(plan', fixes, remix). `remix` = parameters for re-mixing audio only (volume, drop music, ...)."""
    p = copy.deepcopy(plan); fixes: List[str] = []; remix: Dict = {}
    seen = set(); drop_clip: Optional[int] = None; drop_why = ""          # structural drop is applied LAST: it shifts every clip index
    for c in report.failed():
        key = (c.name, c.clip)
        if key in seen:
            continue
        seen.add(key)
        if c.name in ("loudness", "true_peak", "clipping"):
            remix["retarget"] = True; fixes.append(f"{c.name}: re-master with a lower ceiling / corrected gain")
        elif c.name in ("music_balance", "music_overpowers"):
            m = p["audio"].get("music", {})
            if m.get("enabled"):
                v = float(m.get("volume", 0.5))
                if v > 0.2: m["volume"] = round(v * 0.6, 3); fixes.append(f"{c.name}: music volume {v} -> {m['volume']}")
                else: p["audio"]["music"] = {"enabled": False, "reason": "QC: music could not be balanced under the live sound"}; fixes.append(f"{c.name}: music removed")
        elif c.name in ("black_frames", "frozen_frames", "duplicate_shots", "weak_shots", "silence") and c.clip is not None:
            if drop_clip is None:                                         # one drop per pass, deferred to the end
                drop_clip, drop_why = c.clip, c.name
        elif c.name in ("bad_crop", "subject_in_frame") and c.clip is not None:
            s = p["segments"][c.clip]; s["layout"] = {"mode": "fit_blur", "pushIn": 0.0, "reason": f"QC: {c.name}: the crop lost the subject"}; fixes.append(f"{c.name}: clip {c.clip} shown whole over blurred background")
        elif c.name == "abrupt_transition" and c.clip is not None:
            e = p["segments"][c.clip].get("transitionIn") or p["transitions"]
            if e["type"] == "cut":
                q = _set_edge(p, c.clip, "crossfade", 0.4, "QC: the hard cut was audibly/visibly abrupt")
                if q: p = q; fixes.append(f"abrupt_transition: edge {c.clip} cut -> 0.4s crossfade")
            else:
                q = _set_edge(p, c.clip, "cut", 0.0, "QC: the blend jumped, a clean cut reads better")
                if q: p = q; fixes.append(f"abrupt_transition: edge {c.clip} blend -> cut")
        elif c.name in ("caption_overflow", "caption_overlay_collision", "caption_overlap", "caption_reading_speed"):
            st = p["captions"].setdefault("style", {}); fs = float(st.get("fontSize", 0.032))
            if fs > 0.024:
                st["fontSize"] = round(fs * 0.85, 4); fixes.append(f"{c.name}: caption font {fs} -> {st['fontSize']}")
            else:
                p["captions"]["enabled"] = False; fixes.append(f"{c.name}: captions turned off (cannot be made to fit)")
            if retimer: p["captions"]["cues"] = retimer(p["captions"].get("cues", []))
        elif c.name == "duration":
            fixes.append("duration: cannot be repaired by editing; reported")
    if drop_clip is not None:
        q = _drop(p, drop_clip)
        if q is not None and q["durationSeconds"] >= 5.0:
            p = q; fixes.append(f"{drop_why}: dropped clip {drop_clip}")
    if not fixes:
        return None, [], remix
    try:
        validate_plan(p, plan["source"]["durationSeconds"])
    except PlanError:
        return None, fixes, remix
    hist = p.setdefault("creative", {}).setdefault("qc", {}).setdefault("history", [])
    hist.append({"failed": [c.name for c in report.failed()], "fixes": fixes})
    return p, fixes, remix
