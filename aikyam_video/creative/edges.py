"""Where a clip starts and stops: real shot boundaries and pauses, and what a shot MOVES like.

Adapted from open-source projects (see NOTICE, docs/OSS_AUDIT.md):
  * hypecut (Apache-2.0)  src/hypecut/snapping.py   boundary_strength, find_boundaries, refine_boundary, snap_segments (edge rules)
                          src/hypecut/trimming.py   silence_mask, find_pause (pause landing)
                          src/hypecut/refine/similarity.py  descriptor, cosine (motion signature)
  * auto-editor (Unlicense) src/lib/editutil.nim    smoothing (mincut / minclip) on a boolean mask

The single most visible difference between an auto-cut reel and a hand-cut one is where the edges land: a clip that opens three frames into a continuous shot looks
sliced, the same clip opened on the cut the camera operator already made looks edited. Rules (unchanged from the source design): a real cut always beats a pause; an edge may
not enter the protected core of the window; a snap that would break the length budget is rejected, not clamped.
"""
from __future__ import annotations
import subprocess
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

GRID_FPS = 10                                                                # analysis grid of the boundary detector (96x54 gray)
SIG_ROWS, SIG_COLS, SIG_MIN_MOTION = 6, 8, 0.1                               # motion signature grid; absolute floor in luma levels (noise is not motion)
MOTION_SAME = 0.85                                                           # cosine at which two signatures describe the same movement


# ------------------------------------------------------------------ shot boundaries (hypecut snapping.py)
def boundary_strength(gray: np.ndarray) -> np.ndarray:
    """Per-frame cut likelihood: frame difference over its LOCAL baseline (median over ~4 s). Raw difference is useless as an absolute measure."""
    if gray is None or gray.shape[0] < 3:
        return np.zeros(0)
    f = gray.astype(np.float32, copy=False); diff = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2)).astype(np.float64); diff = np.concatenate([diff[:1], diff])
    w = min(diff.size, 41)
    if w < 3:
        return diff / max(float(np.median(diff)), 1e-6)
    pad = w // 2; base = np.median(np.lib.stride_tricks.sliding_window_view(np.pad(diff, (pad, pad), mode="edge"), w), axis=1)[:diff.size]
    return diff / np.maximum(base, 1e-6)


def find_boundaries(gray: np.ndarray, fps: float, ratio: float = 2.5, min_gap: float = 0.5, min_diff: float = 1.5) -> np.ndarray:
    """Times (s, relative to the first frame) of likely hard cuts. `ratio` x its local baseline AND at least `min_diff` luma levels: on footage that holds perfectly
    still the baseline collapses to ~0 and codec flicker would otherwise look like a cut. Strongest peak per `min_gap` cluster."""
    s = boundary_strength(gray)
    if s.size == 0:
        return np.zeros(0)
    raw = np.concatenate([np.abs(np.diff(gray.astype(np.float32), axis=0)).mean(axis=(1, 2))[:1], np.abs(np.diff(gray.astype(np.float32), axis=0)).mean(axis=(1, 2))]).astype(np.float64)
    peaks = np.flatnonzero((s >= ratio) & (raw >= min_diff) & (s >= np.concatenate([s[:1], s[:-1]])) & (s >= np.concatenate([s[1:], s[-1:]])))
    kept: List[int] = []; gap = max(1, int(round(min_gap * fps)))
    for i in peaks[np.argsort(s[peaks])[::-1]]:
        if all(abs(int(i) - k) >= gap for k in kept): kept.append(int(i))
    return np.sort(np.asarray(kept, float)) / fps


def gray_frames(path: str, start: float, dur: float, fps: float = GRID_FPS, w: int = 96, h: int = 54) -> np.ndarray:
    p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(start, 0):.3f}", "-t", f"{dur:.3f}", "-i", path, "-an", "-vf", f"fps={fps},scale={w}:{h}:flags=area,format=gray", "-f", "rawvideo", "-"], capture_output=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, h, w)


def detect_cuts(path: str, start: float, end: float) -> List[float]:
    """Absolute source times of hard cuts in [start, end] (coarse: +-0.1 s)."""
    fr = gray_frames(path, start, end - start)
    return [float(start + t) for t in find_boundaries(fr, GRID_FPS)]


def refine_boundary(path: str, at: float, fps: float, window: float = 0.5) -> float:
    """Re-locate a cut at the SOURCE frame rate within +-window: the frame AFTER the change (one frame late is invisible, one early flashes the old shot)."""
    fps = fps if fps and fps > 0 else 30.0; start = max(0.0, at - window)
    fr = gray_frames(path, start, window * 2, fps=fps)
    if fr.shape[0] < 3:
        return at
    d = np.abs(np.diff(fr.astype(np.float32), axis=0)).mean(axis=(1, 2)); return float(start + (int(np.argmax(d)) + 1) / fps)


# ------------------------------------------------------------------ pauses (hypecut trimming.py + auto-editor smoothing)
def smooth_mask(mask: np.ndarray, mincut: int, minclip: int) -> np.ndarray:
    """auto-editor `smoothing`: runs of True shorter than `minclip` are dropped, runs of False shorter than `mincut` are filled; iterated to a fixed point."""
    val = np.asarray(mask, bool).copy(); prev = prev2 = None
    def runs(x, v):
        out, i = [], 0
        while i < len(x):
            if x[i] == v:
                j = i
                while j < len(x) and x[j] == v: j += 1
                out.append((i, j)); i = j
            else: i += 1
        return out
    while prev is None or (not np.array_equal(prev, val) and (prev2 is None or not np.array_equal(prev2, val))):
        prev2, prev = prev, val.copy()
        for a, b in runs(prev, True):
            if b - a < minclip: val[a:b] = False
        for a, b in runs(val.copy(), False):
            if b - a < mincut: val[a:b] = True
    return val


def pause_runs(rms_db: np.ndarray, hop: float, t0: float, t1: float, drop_db: float = 14.0, min_silence: float = 0.3) -> List[Tuple[float, float]]:
    """Absolute (start, end) of pauses inside [t0, t1]. The threshold is relative to THIS clip's own speech level (a whisper and a shout both have a ~14 dB gap between
    sound and pause). No usable contrast (continuous sound or continuous quiet) -> no pauses: guessing would move edges for nothing."""
    i0, i1 = int(max(0, t0 / hop)), int(min(len(rms_db), t1 / hop) + 1); w = np.asarray(rms_db[i0:i1], float)
    if w.size < 4:
        return []
    level = float(np.percentile(w, 90)); mask = w < level - drop_db
    if mask.mean() < 0.05 or mask.mean() > 0.8:
        return []
    mask = smooth_mask(mask, mincut=1, minclip=max(1, int(round(min_silence / hop))))
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]: j += 1
            out.append(((i0 + i) * hop, (i0 + j) * hop)); i = j
        else: i += 1
    return out


# ------------------------------------------------------------------ motion signature (hypecut refine/similarity.py)
def motion_signature(frames: np.ndarray) -> Optional[List[float]]:
    """Where a shot MOVES, pooled to a coarse grid, zero-meaned and normalised. Appearance describes the venue (a locked camera gives cosine > 0.99 for every clip);
    averaged frame differences cancel the static background and leave a map of where things happened. None when the shot barely moves (noise would correlate with everything)."""
    if frames is None or frames.shape[0] < 2:
        return None
    moved = np.abs(np.diff(frames.astype(np.float32), axis=0)).mean(axis=0); h, w = moved.shape; r, c = SIG_ROWS * (h // SIG_ROWS), SIG_COLS * (w // SIG_COLS)
    if r == 0 or c == 0:
        return None
    v = moved[:r, :c].reshape(SIG_ROWS, r // SIG_ROWS, SIG_COLS, c // SIG_COLS).mean(axis=(1, 3)).ravel()
    if float(v.mean()) < SIG_MIN_MOTION:
        return None
    v = v - v.mean(); n = float(np.linalg.norm(v))
    return None if n < 1e-6 else (v / n).tolist()


def motion_differs(a: Optional[List[float]], b: Optional[List[float]]) -> bool:
    """True only when both signatures exist and clearly differ (cosine < MOTION_SAME): the same venue, a different happening."""
    return a is not None and b is not None and float(np.dot(a, b)) < MOTION_SAME


# ------------------------------------------------------------------ edge snapping (hypecut snap_segments rules, on a Clip window inside a Shot)
def snap_window(a: float, b: float, shot_start: float, shot_end: float, cuts: List[float], pauses: List[Tuple[float, float]], min_len: float, max_len: float,
                window: float = 1.5, core: float = 0.25, refine: Optional[Callable[[float], float]] = None, pad_in: float = 0.05, pad_out: float = 0.12) -> Tuple[float, float, Dict]:
    """Move [a, b) onto real cuts (preferred) or pauses. Rules: an edge travels at most `window` s; it may not enter the protected core (the middle 1 - 2*core of the window);
    it stays inside the shot; a snap that would take the length outside [min_len, max_len] is REJECTED. Returns (a', b', info)."""
    info: Dict = {}; L = b - a; lo_in, hi_in = max(shot_start, a - window), a + core * L; lo_out, hi_out = b - core * L, min(shot_end, b + window)
    def nearest(cands, t, lo, hi):
        c = [x for x in cands if lo <= x <= hi]
        return min(c, key=lambda x: abs(x - t)) if c else None
    na, nb = a, b
    t = nearest(cuts, a, lo_in, hi_in); kind = "cut"
    if t is None:
        t = nearest([e + 0.0 for _, e in pauses], a, lo_in, hi_in); kind = "pause"; t = None if t is None else max(shot_start, t - pad_in)
    if t is not None:
        t = refine(t) if (kind == "cut" and refine) else t
        if min_len <= b - t <= max_len and lo_in <= t <= hi_in and t < b:
            info["start"] = (round(t - a, 3), kind); na = t
    t = nearest(cuts, b, lo_out, hi_out); kind = "cut"
    if t is None:
        t = nearest([s for s, _ in pauses], b, lo_out, hi_out); kind = "pause"; t = None if t is None else min(shot_end, t + pad_out)
    if t is not None:
        t = refine(t) if (kind == "cut" and refine) else t
        if min_len <= t - na <= max_len and lo_out <= t <= hi_out and t > na:
            info["end"] = (round(t - b, 3), kind); nb = t
    return round(na, 3), round(nb, 3), info
