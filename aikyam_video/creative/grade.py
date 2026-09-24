"""Colour match across a reel's clips -- the light grade a human editor puts on a cut so consecutive clips sit in the same world.

Measured on a real 6-clip reel (11-min Golden Temple source): brightness varied by 0.18, saturation by 0.15, warmth (R-B) by 0.16 between clips, with
two clips visibly darker and cooler than the rest. Each clip is nudged PART of the way toward the reel's median look (mood survives: a night aarti stays
darker than a daylight exterior, just less jarringly) and every correction is capped, so the result reads as an edit, not a filter."""
from __future__ import annotations
import os, tempfile
from typing import Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np
from .. import ffmpeg as ff

PULL = 0.6            # fraction of the gap to the reel median that is corrected
MAX_BRIGHT = 0.08     # eq brightness offset cap (0..1 scale)
MAX_SAT = 0.20        # saturation multiplier stays within 1 +- this
MAX_WARM = 0.05       # red/blue channel gain cap (+-5%: R gains, B loses)
MAX_TINT = 0.06       # green channel gain cap (+-6%): the yellow-green cast of fluorescent / aged footage
NEUTRAL = (0.012, 0.03, 0.006, 0.006)   # corrections smaller than this (brightness, saturation, warmth, tint) are not worth a filter


def measure(src: str, t: float) -> Optional[Tuple[float, float, float, float]]:
    """(mean luma, mean saturation, warmth = mean(R)-mean(B), tint = mean(G)-(mean(R)+mean(B))/2), each 0..1, from one frame of the SOURCE at time t; None if the frame can't be read."""
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "f.jpg")
        try:
            ff.frame_at(src, t, f)
        except Exception:                                                        # noqa: BLE001 -- grading is cosmetic: never fail a reel over it
            return None
        im = cv2.imread(f)
    if im is None:
        return None
    im = cv2.resize(im, (160, 90), interpolation=cv2.INTER_AREA)
    b, g, r = (im[..., k].astype(float).mean() / 255 for k in range(3))
    return float(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).mean() / 255), float(cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[..., 1].mean() / 255), float(r - b), float(g - (r + b) / 2)


def match(stats: Sequence[Optional[Tuple[float, float, float, float]]]) -> List[Optional[dict]]:
    """Per-clip corrections {brightness, saturation, warmth} toward the median of the measurable clips; None = leave that clip alone."""
    ok = [s for s in stats if s is not None]
    if len(ok) < 2:
        return [None] * len(stats)
    tl, ts, tw, tt = (float(np.median([s[k] for s in ok])) for k in range(4))
    out: List[Optional[dict]] = []
    for s in stats:
        if s is None:
            out.append(None); continue
        b = float(np.clip(PULL * (tl - s[0]), -MAX_BRIGHT, MAX_BRIGHT))
        k = float(np.clip(1.0 + PULL * (ts / max(s[1], 1e-3) - 1.0), 1 - MAX_SAT, 1 + MAX_SAT))
        w = float(np.clip(1.25 * PULL * (tw - s[2]), -MAX_WARM, MAX_WARM))       # a gain g moves R-B by ~2*g*mean(level) ~ 0.8*g at typical levels
        t = float(np.clip(1.25 * PULL * (tt - s[3]), -MAX_TINT, MAX_TINT))              # too green (s[3] > median) -> negative: green gain drops
        g = {"brightness": round(b, 3), "saturation": round(k, 3), "warmth": round(w, 3), "tint": round(t, 3)}
        out.append(g if (abs(b) >= NEUTRAL[0] or abs(k - 1) >= NEUTRAL[1] or abs(w) >= NEUTRAL[2] or abs(t) >= NEUTRAL[3]) else None)
    return out


def filter_expr(g: Dict[str, float]) -> str:
    """The ffmpeg filter for one clip's correction ('' when there is nothing to do)."""
    if not g:
        return ""
    parts = []
    if abs(g.get("brightness", 0)) >= NEUTRAL[0] or abs(g.get("saturation", 1) - 1) >= NEUTRAL[1]:
        parts.append(f"eq=brightness={g.get('brightness', 0):.3f}:saturation={g.get('saturation', 1):.3f}")
    w, t = g.get("warmth", 0.0), g.get("tint", 0.0)
    if abs(w) >= NEUTRAL[2] or abs(t) >= NEUTRAL[3]:
        parts.append(f"colorchannelmixer=rr={1 + w:.3f}:gg={1 + t:.3f}:bb={1 - w:.3f}")     # linear, predictable (ffmpeg's colorbalance lost its effect after an eq)
    return ",".join(parts)


def apply(segs: List[dict], probes: Sequence[Tuple[str, float, bool]]) -> int:
    """probes[i] = (source path, time to look at, gradable) for segs[i]. Writes seg['grade'] where a correction is worth making; returns how many clips got one."""
    stats = [measure(p, t) if ok else None for p, t, ok in probes]
    n = 0
    for s, g in zip(segs, match(stats)):
        if g:
            s["grade"] = g; n += 1
    return n
