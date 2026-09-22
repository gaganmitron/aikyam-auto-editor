"""Composition per clip and output format: crop (subject-following window) vs fit-with-blurred-background, plus an optional slow push-in.

Cropping 16:9 to 9:16 keeps only ~32% of the width. If the subject (a procession, a crowd, a wide facade) is wider than that window, the crop amputates it:
then the whole frame is shown fitted to the width over a blurred, darkened copy of itself (the standard broadcast/social answer). Decided from MEASURED subject extent.
"""
from __future__ import annotations
from typing import Dict, Optional, Sequence
import numpy as np

CORE_MARGIN = 1.25      # the window holds the core of the action plus 25% margin
WINDOW_MAX = 0.6        # never show more than 60% of the source width: keeps >= ~half of a 9:16 frame real picture (the rest is blurred fill)
MAX_THIN_BAR = 0.15     # blurred fill thinner than this share of the frame height is not worth showing
PUSH_IN = 0.05          # 5% slow zoom over the clip: perceptible as movement, invisible as a crop
PUSH_ROLES = {"REVEAL", "OPENING", "CLOSING"}


def crop_fraction(src_w: int, src_h: int, target_aspect: float) -> float:
    """Width share of the source that a pure crop of the target aspect keeps."""
    return float(min(1.0, target_aspect * src_h / src_w))


def decide(seg: dict, src_w: int, src_h: int, target_aspect: float) -> Dict:
    """{'mode': 'crop'|'fit_blur', 'window': width fraction shown, 'pushIn', 'reason'} for ONE segment and ONE format.
    window = clip(1.25 * core extent of the action, pure-crop width, 0.6): a narrow action gets a tight crop; a wide one gets a wider window whose leftover
    frame height is filled with a blurred copy; the window never exceeds 60% so the picture stays large. A plan-level override (QC/replan) wins."""
    forced = seg.get("layout") or {}
    cw = crop_fraction(src_w, src_h, target_aspect)
    if forced.get("mode"):
        mode, win, why = forced["mode"], float(forced.get("window", 1.0 if forced["mode"] == "fit_blur" else cw)), forced.get("reason", "set by plan")
    elif cw >= 0.92:
        mode, win, why = "crop", cw, "source already has (nearly) the target aspect"
    else:
        core = float(seg.get("subjectSpread", 0.0)); win = float(np.clip(CORE_MARGIN * core, cw, max(cw, WINDOW_MAX)))
        if win <= cw / (1 - MAX_THIN_BAR):                  # bars would be a thin sliver (<15% of the height): a clean crop looks better than a stripe of blur
            mode, win, why = "crop", cw, f"the core of the action ({core:.0%}) fits the {cw:.0%} crop window"
        else:
            mode, why = "fit_blur", f"the core of the action spans {core:.0%}: window widened to {win:.0%} (pure crop would hold {cw:.0%}), the rest of the height is blurred fill"
    push = float(forced["pushIn"]) if "pushIn" in forced else (PUSH_IN if (mode == "crop" and seg.get("role") in PUSH_ROLES and seg.get("calm")) else 0.0)
    return {"mode": mode, "window": win, "pushIn": push, "reason": why}
