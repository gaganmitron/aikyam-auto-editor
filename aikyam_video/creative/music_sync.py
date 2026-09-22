"""Music <-> edit synchronisation.

  align_cuts        : bar-aware dynamic programme over ALL cut times at once (BEAT-style: many-to-one beat matching, shot-length deviation vs beat error, downbeat bonus).
  align_cuts_greedy : the earlier one-edge-at-a-time nudge, kept for ablation (tools/eval).

  offset : where in the track to start so its energy contour follows the edit's energy arc (peak under the climax), on a beat when the track has a grid.
  align  : nudge each cut (or the centre of each blend) onto the nearest beat by trimming/extending the outgoing clip, within what the footage allows.
Beatless tracks (drones) get an offset by energy only and NO beat alignment.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from .model import Clip, Timeline
from .musicdna import MusicAnalysis
from .shots import HOP, Population

TOL_BEAT = 0.35          # a cut may move by up to 35% of a beat interval (at ~100 BPM: ~0.2 s)


def edit_energy(tl: Timeline, pop: Population, T: float, overlaps: Sequence[float]) -> np.ndarray:
    """Energy of the edit on the OUTPUT timeline at 0.5 s resolution (0..1)."""
    n = max(1, int(np.ceil(T / HOP))); out = np.zeros(n); t = 0.0
    for i, c in enumerate(tl.clips):
        e = pop.energy(c.shot.slots); s0 = int((c.start - c.shot.start) / HOP); vals = e[s0: s0 + max(1, int(round(c.length / HOP)))]
        k0 = int(t / HOP)
        for j, v in enumerate(vals):
            if k0 + j < n: out[k0 + j] = v
        t += c.length - (overlaps[i] if i < len(overlaps) else 0.0)
    return out


def pick_offset(analysis: MusicAnalysis, edit_e: np.ndarray, climax_t: Optional[float], T: float) -> Tuple[float, float]:
    """(offset seconds, match score). Slides the track's energy curve under the edit's and keeps the best correlation + a bonus for a music peak at the climax."""
    me = np.asarray(analysis.energy); n = len(edit_e)
    if len(me) <= n + 1:
        return 0.0, 0.0
    best, bo = -9.0, 0
    grid = analysis.downbeats or analysis.beats                                     # bar starts when Beat This! provides them
    for o in range(0, len(me) - n):
        seg = me[o:o + n]
        c = float(np.corrcoef(seg, edit_e)[0, 1]) if seg.std() > 1e-6 and edit_e.std() > 1e-6 else 0.0
        if climax_t is not None:
            c += 0.25 * float(seg[min(int(climax_t / HOP), n - 1)])
        if c > best + 1e-9: best, bo = c, o
    off = bo * HOP
    if grid:                                                                       # start ON a beat so the bar structure lines up with the edit
        off = min(grid, key=lambda b: abs(b - off)); off = float(min(off, max(analysis.duration - T, 0.0)))
    return round(off, 3), round(best, 3)


def align_cuts_greedy(tl: Timeline, analysis: MusicAnalysis, offset: float, overlaps_fn) -> Dict[str, float]:
    """Move edge times onto beats. Edge time = cut instant (or blend centre). Adjusts the OUTGOING clip's out-point, in order, within its shot; returns a report."""
    if not analysis.bpm or len(analysis.beats) < 4 or len(tl.clips) < 2:
        return {"beats": 0, "aligned": 0, "on_beat_before": 0.0, "on_beat_after": 0.0, "mean_shift_s": 0.0}
    beat = 60.0 / analysis.bpm; beats = np.asarray(analysis.beats) - offset

    def edge_times():
        ov = overlaps_fn(tl); t, out = 0.0, []
        for i, c in enumerate(tl.clips[:-1]):
            t += c.length - ov[i]; out.append(t + ov[i] / 2.0)
        return out
    def frac_on(ts): return float(np.mean([np.min(np.abs(beats - t)) <= 0.04 for t in ts])) if ts else 0.0
    before = frac_on(edge_times()); shifts = []
    for i in range(len(tl.clips) - 1):
        ts = edge_times(); t = ts[i]; b = float(beats[np.argmin(np.abs(beats - t))]); d = b - t
        if abs(d) > TOL_BEAT * beat or abs(d) < 1e-3:
            continue
        c = tl.clips[i]; new_end = c.end + d
        lo = c.start + 1.5; hi = c.shot.end
        if not (lo <= new_end <= hi):
            continue
        c.end = round(new_end, 3); shifts.append(abs(d))
    after = frac_on(edge_times())
    return {"beats": len(analysis.beats), "aligned": len(shifts), "on_beat_before": before, "on_beat_after": after, "mean_shift_s": float(np.mean(shifts)) if shifts else 0.0}


# ---------------------------------------------------------------- bar-level dynamic programme
W_LEN = 4.0          # cost of stretching/shrinking a clip by 100% of its length (quadratic): beat error competes with shot-length deviation
W_OFF = 3.0          # cost of a cut that is off the beat grid, per beat of distance (max 0.5 beat)
BONUS_BAR = 0.6      # reward for a cut on a downbeat (bar start)
BONUS_KEY = 0.6      # extra reward when the cut leads INTO a key moment (REVEAL / CLIMAX): those land on the bar
REACH = 1.25         # candidate beats: within +-1.25 beat intervals of where the edge would fall unchanged
MIN_CLIP = 1.5       # seconds


def align_cuts(tl: Timeline, analysis: MusicAnalysis, offset: float, overlaps_fn, max_first: Optional[float] = None, max_len: Optional[float] = None) -> Dict[str, float]:
    """Choose every cut time TOGETHER (Viterbi over edges). Edge time = cut instant (or blend centre); clip i's out point moves so that its length becomes
    (t_i - t_{i-1}) + half of the overlaps at its two ends. Candidates per edge: the beats near where it would fall, plus 'keep the nominal length'.
    Cost = length deviation + off-beat distance - downbeat bonuses; clips must stay inside their shot and >= 1.5 s. Returns a report (same keys as the greedy version + downbeat share)."""
    empty = {"beats": 0, "aligned": 0, "on_beat_before": 0.0, "on_beat_after": 0.0, "on_downbeat_after": 0.0, "mean_shift_s": 0.0, "method": "dp"}
    n = len(tl.clips)
    if not analysis.bpm or len(analysis.beats) < 4 or n < 2:
        return empty
    beat = 60.0 / analysis.bpm; beats = np.asarray(analysis.beats, float) - offset; downs = np.asarray(analysis.downbeats, float) - offset if analysis.downbeats else np.zeros(0)
    ov = list(overlaps_fn(tl)); L0 = [c.length for c in tl.clips]
    def dist_beat(t): return float(np.min(np.abs(beats - t)))
    def on_down(t): return bool(len(downs) and np.min(np.abs(downs - t)) <= 0.04)
    def edge_times(L):
        t, out = 0.0, []
        for i in range(n - 1): t += L[i] - ov[i]; out.append(t + ov[i] / 2.0)
        return out
    nominal = edge_times(L0); before = float(np.mean([dist_beat(t) <= 0.04 for t in nominal]))
    hi = [c.shot.end - c.start for c in tl.clips]; lo = [min(MIN_CLIP, h) for h in hi]
    for i, c in enumerate(tl.clips):
        if getattr(c, "locked_end", False): lo[i] = hi[i] = L0[i]                                      # an out point on a REAL cut wins over the beat (hypecut rule: a cut is evidence, a beat a preference)
    if max_len is not None: hi = [max(min(h, max_len), min(l0, h)) for h, l0 in zip(hi, L0)]           # alignment may not push a clip past the pacing profile's longest shot (a clip already longer keeps its length)
    if max_first is not None: hi[0] = max(min(hi[0], max_first), min(lo[0], hi[0]))                    # a hook opening stays short (story.HOOK_MAX): alignment may not stretch it
    key = [tl.clips[i + 1].role in ("REVEAL", "CLIMAX") for i in range(n - 1)]
    # layer i (edge i): list of (time, back-pointer, accumulated cost)
    layers: List[List[Tuple[float, int, float]]] = []
    for i in range(n - 1):
        prev_t = [(0.0 - ov[0] * 0.0, -1, 0.0)] if i == 0 else layers[i - 1]                   # virtual "edge -1": clip 0 starts at 0
        half_prev = 0.0 if i == 0 else ov[i - 1] / 2.0
        cands: Dict[Tuple[int, int], Tuple[float, int, float]] = {}
        for pi, (tp, _, cp) in enumerate(prev_t):
            # length of clip i if its edge lands at time t:  L' = (t - tp) + ov[i]/2 + half_prev  (tp = previous edge time; for i == 0 tp = 0 and half_prev = 0)
            nominal_t = tp + L0[i] - half_prev - ov[i] / 2.0
            near = beats[np.abs(beats - nominal_t) <= REACH * beat]
            for t in [*near.tolist(), nominal_t]:
                Lp = (t - tp) + ov[i] / 2.0 + half_prev
                if not (lo[i] - 1e-6 <= Lp <= hi[i] + 1e-6):
                    continue
                dev = (Lp - L0[i]) / max(L0[i], 1e-6); c = cp + W_LEN * dev * dev
                d = dist_beat(t)
                if d > 0.04: c += W_OFF * min(d / beat, 0.5)
                elif on_down(t): c -= BONUS_BAR + (BONUS_KEY if key[i] else 0.0)
                k = (pi, int(round(t * 1000)))
                if k not in cands or c < cands[k][2]: cands[k] = (t, pi, c)
        if not cands:                                                                                   # nothing feasible from here on: leave the rest as it is
            return empty
        layers.append(sorted(cands.values(), key=lambda x: (x[2], x[0])))
    # the last clip keeps its length: pick the cheapest end state and walk back
    best = min(range(len(layers[-1])), key=lambda j: (layers[-1][j][2], layers[-1][j][0])); times = [0.0] * (n - 1); j = best
    for i in range(n - 2, -1, -1):
        times[i] = layers[i][j][0]; j = layers[i][j][1]
    shifts = []; prev = 0.0; half_prev = 0.0
    for i in range(n - 1):
        Lp = (times[i] - prev) + ov[i] / 2.0 + half_prev; c = tl.clips[i]
        if abs(Lp - L0[i]) > 1e-3: shifts.append(abs(Lp - L0[i])); c.end = round(c.start + Lp, 3)
        prev, half_prev = times[i], ov[i] / 2.0
    after_t = edge_times([c.length for c in tl.clips])
    return {"beats": len(analysis.beats), "aligned": len(shifts), "on_beat_before": before, "on_beat_after": float(np.mean([dist_beat(t) <= 0.04 for t in after_t])),
            "on_downbeat_after": float(np.mean([on_down(t) for t in after_t])) if len(downs) else 0.0, "mean_shift_s": float(np.mean(shifts)) if shifts else 0.0, "method": "dp"}


def align_end(tl: Timeline, analysis: MusicAnalysis, offset: float, overlaps_fn, tol_beats: float = 2.0) -> Dict[str, float]:
    """Make the reel END on a bar line (downbeat, else beat) so the music resolves with the picture: the last clip's out point moves by at most `tol_beats` beats, inside its shot,
    never below 1.5 s, never if it sits on a real cut/phrase (locked_end). Returns {"moved": seconds, "on": "downbeat"|"beat"|None}."""
    grid = analysis.downbeats or analysis.beats
    if not analysis.bpm or len(grid) < 2 or not tl.clips or getattr(tl.clips[-1], "locked_end", False):
        return {"moved": 0.0, "on": None}
    c = tl.clips[-1]; beat = 60.0 / analysis.bpm; T = sum(k.length for k in tl.clips) - sum(overlaps_fn(tl)); g = np.asarray(grid, float) - offset
    for label, gr in (("downbeat" if analysis.downbeats else "beat", g), ("beat", np.asarray(analysis.beats, float) - offset)):
        near = gr[(np.abs(gr - T) <= tol_beats * beat) & (gr > 0)]
        for t in sorted(near.tolist(), key=lambda x: abs(x - T)):
            new_end = c.end + (t - T)
            if c.start + 1.5 <= new_end <= c.shot.end + 1e-6:
                moved = new_end - c.end; c.end = round(new_end, 3); return {"moved": round(moved, 3), "on": label}
    return {"moved": 0.0, "on": None}
