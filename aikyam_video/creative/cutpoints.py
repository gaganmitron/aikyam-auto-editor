"""Where a clip starts and stops. Every in/out point is chosen from REAL boundary candidates, scored together, instead of the quality grid alone.

  in  candidates : hard cuts, speech-phrase starts, pause ends (sound resumes), motion onsets (the action begins)
  out candidates : hard cuts, speech-phrase ends, pause starts (sound stops), motion lulls (the action has settled)
  A boundary that falls INSIDE a spoken phrase is penalised: nobody cuts a sentence in half.
  The reel's FIRST clip must begin on something (cut / onset / phrase) and its LAST clip must end resolved (lull / phrase end / pause): that is what makes a reel read as started and finished.
Generalises hypecut's snap rules (edges.snap_window, kept for ablation): cut > pause there, here every kind has a weight and the speech/motion evidence competes with travel distance."""
from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from .shots import HOP

W = {"cut": 1.0, "phrase": 0.9, "pause": 0.7, "onset": 0.55, "lull": 0.55, "grid": 0.0}
W_FIRST = {"cut": 1.0, "onset": 1.0, "phrase": 0.9, "pause": 0.4, "lull": 0.0, "grid": 0.0}       # an opening starts ON something
W_LAST = {"lull": 1.0, "phrase": 1.0, "pause": 0.9, "cut": 0.8, "onset": 0.0, "grid": 0.0}         # an ending has settled
TRAVEL = 0.30            # score lost per second an edge moves from the quality-optimal spot
MID_SPEECH = 1.2         # score lost for an edge inside a spoken phrase
MIN_PHRASE_GAP = 0.35    # a silence this long between words ends a phrase


def phrases(transcript, gap: float = MIN_PHRASE_GAP) -> List[Tuple[float, float]]:
    """Speech phrases (start, end) in SOURCE seconds: words grouped until a silence >= gap; a segment without word times counts as one phrase."""
    out: List[Tuple[float, float]] = []
    for s in getattr(transcript, "segments", []) or []:
        ws = [w for w in (s.words or []) if w.text]
        if not ws:
            out.append((s.start, s.end)); continue
        a = ws[0].start; prev = ws[0].end
        for w in ws[1:]:
            if w.start - prev >= gap: out.append((a, prev)); a = w.start
            prev = w.end
        out.append((a, prev))
    return out


def motion_points(shot, pct_lo: float = 40, pct_hi: float = 60) -> Tuple[List[float], List[float]]:
    """(onsets, lulls): absolute times on the 0.5 s slot grid. Onset = calm slot -> active slot; lull = the slot before the time is calm. Relative to THIS shot's own motion."""
    sl = getattr(shot, "slots", None)
    if sl is None or len(sl.motion) < 3: return [], []
    m = np.asarray(sl.motion, float); lo, hi = np.percentile(m, pct_lo), np.percentile(m, pct_hi)
    if hi - lo < 1e-6: return [], []
    ons = [shot.start + i * HOP for i in range(1, len(m)) if m[i] >= hi and m[i - 1] <= lo]
    lul = [shot.start + i * HOP for i in range(1, len(m) + 1) if m[i - 1] <= lo]
    return ons, lul


def candidates(shot, cuts: Sequence[float], pauses: Sequence[Tuple[float, float]], phr: Sequence[Tuple[float, float]], pad_in: float = 0.05, pad_out: float = 0.12) -> Tuple[List[Tuple[float, str]], List[Tuple[float, str]]]:
    """(in candidates, out candidates) as (time, kind), clipped to the shot."""
    ons, lul = motion_points(shot)
    ins = [(c, "cut") for c in cuts] + [(max(shot.start, s - pad_in), "phrase") for s, _ in phr] + [(max(shot.start, e - pad_in), "pause") for _, e in pauses] + [(t, "onset") for t in ons]
    outs = [(c, "cut") for c in cuts] + [(e + pad_out, "phrase") for _, e in phr] + [(s + pad_out, "pause") for s, _ in pauses] + [(t, "lull") for t in lul]
    ok = lambda x: shot.start <= x[0] <= shot.end
    return [x for x in ins if ok(x)], [x for x in outs if ok(x)]


def _inside(t: float, phr: Sequence[Tuple[float, float]], margin: float = 0.08) -> bool:
    return any(s + margin < t < e - margin for s, e in phr)


def choose_edges(a: float, b: float, shot_start: float, shot_end: float, ins: Sequence[Tuple[float, str]], outs: Sequence[Tuple[float, str]], phr: Sequence[Tuple[float, float]],
                 min_len: float, max_len: float, window: float = 1.5, first: bool = False, last: bool = False,
                 refine=None) -> Tuple[float, float, Dict]:
    """Best (a', b') within `window` s of the quality-optimal (a, b). Exhaustive over the (few) candidates. Returns (a', b', info) with info like edges.snap_window:
    {"start": (shift, kind), "end": (shift, kind)} for every edge that MOVED to a real boundary."""
    wi, wo = (W_FIRST if first else W), (W_LAST if last else W)
    cin = [(a, "grid")] + [(t, k) for t, k in ins if abs(t - a) <= window]
    cout = [(b, "grid")] + [(t, k) for t, k in outs if abs(t - b) <= window]
    best = (-9.0, a, b, "grid", "grid")
    for ta, ka in cin:
        pa = wi[ka] - TRAVEL * abs(ta - a) - (MID_SPEECH if _inside(ta, phr) else 0.0)
        for tb, kb in cout:
            L = tb - ta
            if not (min_len - 1e-6 <= L <= max_len + 1e-6) or ta < shot_start - 1e-6 or tb > shot_end + 1e-6:
                continue
            s = pa + wo[kb] - TRAVEL * abs(tb - b) - (MID_SPEECH if _inside(tb, phr) else 0.0)
            if s > best[0] + 1e-9: best = (s, ta, tb, ka, kb)
    _, ta, tb, ka, kb = best; info: Dict = {}
    if refine and ka == "cut": ta = refine(ta)
    if refine and kb == "cut": tb = refine(tb)
    if ka != "grid" and abs(ta - a) > 1e-6: info["start"] = (round(ta - a, 3), ka)
    elif ka != "grid": info["start"] = (0.0, ka)               # the quality-optimal edge already sat on a real boundary
    if kb != "grid" and abs(tb - b) > 1e-6: info["end"] = (round(tb - b, 3), kb)
    elif kb != "grid": info["end"] = (0.0, kb)
    return round(ta, 3), round(tb, 3), info
