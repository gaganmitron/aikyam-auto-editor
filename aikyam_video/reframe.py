"""Subject-aware reframing (section 14).

Per plan segment we sample frames (default 2/s) and estimate the subject's x-centre with this priority:
  1. largest face (priest / speaker)            2. centroid of MOTION (ritual action: the lamp, the dancers)
  3. centroid of image detail (central action)
then smooth with an EMA and cap the pan speed so the virtual camera glides instead of jittering.
Not implemented: localising a *deity* specifically (needs a detector/segmenter; the label model is image-level).
"""
from __future__ import annotations
import os, subprocess, tempfile
from typing import List, Tuple
import cv2
import numpy as np
from .models import SceneVision


def crop_box(src_w: int, src_h: int, target_aspect: float, subject_x: float = 0.5) -> Tuple[int, int, int, int]:
    """Largest crop of the target aspect (w/h) that fits the source, centred on subject_x (0..1) horizontally,
    clamped to the frame. Returns even (x, y, w, h)."""
    if target_aspect <= src_w / src_h:
        ch = src_h; cw = int(round(ch * target_aspect)) // 2 * 2
        x = int(round(min(max(subject_x * src_w - cw / 2, 0), src_w - cw))) // 2 * 2
        return x, 0, cw, ch
    cw = src_w; ch = int(round(cw / target_aspect)) // 2 * 2
    return 0, (src_h - ch) // 2 // 2 * 2, cw, ch


def subject_x(vision: List[SceneVision], start: float, end: float) -> float:
    """Overlap-weighted median of scene saliency over [start, end]; 0.5 when nothing is known."""
    pts = [(sv.vision.saliency_x, min(end, sv.end) - max(start, sv.start)) for sv in vision
           if sv.end > start and sv.start < end]
    if not pts:
        return 0.5
    pts.sort()
    half, acc = sum(w for _, w in pts) / 2, 0.0
    for x, w in pts:
        acc += w
        if acc >= half:
            return x
    return pts[-1][0]


ALPHA, MAX_PAN = 0.5, 0.25      # EMA weight; max virtual-camera speed (fraction of frame width per second)


def smooth_path(raw: List[Tuple[float, float]], alpha: float = ALPHA, max_speed: float = MAX_PAN) -> List[Tuple[float, float]]:
    """EMA + pan-speed limit (fraction of frame width per second). raw = [(t, x)] ascending in t."""
    if not raw:
        return []
    out = [raw[0]]
    for t, x in raw[1:]:
        prev_t, prev_x = out[-1]
        target = prev_x + alpha * (x - prev_x)
        lim = max_speed * max(t - prev_t, 1e-6)
        out.append((t, prev_x + float(np.clip(target - prev_x, -lim, lim))))
    return out


def _extent(col: np.ndarray) -> float:
    """Horizontal extent of the CORE of the action as a fraction of the frame width: the middle 50% of the energy (25th..75th percentile of its CDF).
    Outliers (a flag at the edge, a far-off person) must not decide the composition."""
    tot = col.sum()
    if tot <= 0:
        return 0.0
    cdf = np.cumsum(col) / tot
    return float((np.searchsorted(cdf, 0.75) - np.searchsorted(cdf, 0.25)) / len(col))


def track_subject_ex(src: str, start: float, end: float, fps: float = 2.0, face_cascade=None) -> List[Tuple[float, float, float]]:
    """[(t_relative, x centre 0..1, horizontal extent 0..1)] for the segment. Extent tells the layout stage whether a narrow crop can hold the subject."""
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", src,
                        "-vf", f"fps={fps},scale=320:-2", "-q:v", "3", os.path.join(td, "f_%04d.jpg")], check=True)
        files = sorted(os.listdir(td))
        face = face_cascade or cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
        raw, ext, prev = [], [], None
        for i, f in enumerate(files):
            g = cv2.GaussianBlur(cv2.cvtColor(cv2.imread(os.path.join(td, f)), cv2.COLOR_BGR2GRAY), (5, 5), 0)
            h, w = g.shape
            x = spread = None
            fr = face.detectMultiScale(g, 1.15, 5, minSize=(16, 16))
            if len(fr):
                fx, fy, fw, fh = max(fr, key=lambda r: r[2] * r[3]); x = (fx + fw / 2) / w; spread = fw / w
            if x is None and prev is not None:
                m = cv2.absdiff(g, prev).astype(np.float32); m[m < 12] = 0
                col = m.sum(axis=0)
                if col.sum() > 0.004 * 255 * w * h:                       # enough real motion to trust
                    x = float((col * np.arange(w)).sum() / col.sum() / w); spread = _extent(col)
            if x is None:
                mag = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1))
                col = np.convolve(mag.sum(axis=0) + 1e-6, np.ones(9) / 9, mode="same")
                x = float((col * np.arange(w)).sum() / col.sum() / w); spread = _extent(col)
            raw.append((i / fps, x)); ext.append(spread); prev = g
    if len(raw) > 1:      # frame 0 has no motion reference: start the camera where the first real estimate says the subject is
        raw[0] = (raw[0][0], raw[1][1]); ext[0] = ext[1]
    return [(t, x, e) for (t, x), e in zip(smooth_path(raw), ext)]


def track_subject(src: str, start: float, end: float, fps: float = 2.0, face_cascade=None) -> List[Tuple[float, float]]:
    """[(t_relative_to_start, x in 0..1)] for the segment."""
    return [(t, x) for t, x, _ in track_subject_ex(src, start, end, fps, face_cascade)]


def path_expr(path: List[Tuple[float, float]], src_w: int, cw: int) -> str:
    """ffmpeg crop `x` expression (pixels, evaluated per frame with local time t) interpolating the tracked path."""
    px = [(t, int(min(max(x * src_w - cw / 2, 0), src_w - cw))) for t, x in path]
    if len(px) == 1:
        return str(px[0][1])
    e = str(px[-1][1])
    for (t0, x0), (t1, x1) in reversed(list(zip(px[:-1], px[1:]))):
        e = f"if(lt(t\\,{t1:.3f})\\,{x0}+({x1}-{x0})*(t-{t0:.3f})/{max(t1 - t0, 1e-3):.3f}\\,{e})"
    return e


# ---------------------------------------------------------------- virtual camera (AutoFlip-style modes)
INNER = 0.6         # a stationary camera is chosen when the subject's whole range fits the inner 60% of the window
PAN_FIT = 0.08      # a linear pan is chosen when the tracked x stays within 8% of the window from a straight line
PAN_MIN = 0.06      # ... and the subject really moves (>= 6% of the window over the clip): else it is stationary anyway
H_MIN = 0.03        # half-extent floor of a subject (a face box is small; a person is wider)


def _cx(x, win):
    return np.clip(x, win / 2, 1 - win / 2)


def camera_path(track: List[Tuple[float, float, float]], win: float) -> dict:
    """Choose ONE camera behaviour for a clip from the tracked subject [(t, x, extent)] and the crop window width `win` (fraction of the source width):
      stationary  the subject's whole range fits the inner 60% of the window: a locked-off frame (no gimbal drift on a steady scene)
      pan         the subject moves along a line: one smooth linear move
      track       anything else: a low-order polynomial fit of x(t) (smooth by construction), speed-limited
    Returns {mode, path [(t, x)] ([] for stationary), subjectX, inFrame (share of samples whose subject core lies inside the window), reason}."""
    if not track:
        return {"mode": "stationary", "path": [], "subjectX": 0.5, "inFrame": 1.0, "reason": "no tracking data"}
    t = np.array([a for a, _, _ in track], float); x = np.array([b for _, b, _ in track], float); h = np.maximum(np.array([c for _, _, c in track], float), H_MIN) / 2
    lo, hi = float((x - h).min()), float((x + h).max()); T = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    if win >= 0.999 or hi - lo <= INNER * win:
        cx = float(_cx((lo + hi) / 2, win)); path: List[Tuple[float, float]] = []; mode = "stationary"
        why = f"the subject stays within {hi - lo:.0%} of the frame, inside the {INNER:.0%} core of the {win:.0%} window" if win < 0.999 else "the window is the whole frame"
        f = lambda tt: np.full_like(np.asarray(tt, float), cx)
    else:
        b, a = np.polyfit(t, x, 1) if len(t) > 2 else (0.0, float(x.mean()))
        resid = float(np.sqrt(np.mean((x - (a + b * t)) ** 2)))
        if resid <= PAN_FIT * win and abs(b * T) >= PAN_MIN * win:
            b = float(np.clip(b, -MAX_PAN, MAX_PAN)); x0, x1 = float(_cx(a + b * t[0], win)), float(_cx(a + b * t[-1], win))
            path = [(float(t[0]), x0), (float(t[-1]), x1)]; mode = "pan"; why = f"the subject moves along a line ({abs(b * T):.0%} of the frame in {T:.1f}s, fit error {resid:.1%})"
            f = lambda tt: np.interp(tt, [t[0], t[-1]], [x0, x1])
        else:
            deg = int(min(3, len(t) - 1)); coef = np.polyfit(t, x, deg) if deg >= 1 else [float(x[0])]
            ts = np.arange(t[0], t[-1] + 1e-6, 0.5) if T > 0.5 else np.array([t[0], t[-1]])
            raw = [(float(tt), float(_cx(np.polyval(coef, tt), win))) for tt in ts]
            path = smooth_path(raw, alpha=1.0); mode = "track"; why = f"the subject moves irregularly (fit error {resid:.1%}): a smooth degree-{deg} path"
            pt = np.array([p for p, _ in path]); px = np.array([q for _, q in path]); f = lambda tt: np.interp(tt, pt, px)
        cx = float(np.median([q for _, q in path])) if path else 0.5
    c = _cx(f(t), win); inside = (x - h >= c - win / 2 - 1e-6) & (x + h <= c + win / 2 + 1e-6)
    return {"mode": mode, "path": path, "subjectX": cx, "inFrame": float(inside.mean()), "reason": why}


SCAN_SPAN_MAX = 0.6     # a scan never travels past the 60% of the source width that layout.WINDOW_MAX allows


def scan_path(track: List[Tuple[float, float, float]], win: float, length: float, spread: float) -> dict:
    """Full-bleed answer to a subject WIDER than the crop window: instead of fitting the whole frame over blurred fill, a pure-crop window (`win`) glides across the
    wide subject for the whole clip (a slow, steady pan: the classic way an editor shows a facade, a crowd or a procession in 9:16).
    Direction follows the tracked subject's drift when it has one, else left to right. Travel = the subject span (1.25 x extent, <= 60% of the width) minus the window.
    Same return shape as camera_path (inFrame here = share of samples whose subject CENTRE is inside the moving window, since the whole subject is wider than the window by construction; a pan that misses the subject is still caught)."""
    span = float(np.clip(1.25 * spread, win, max(win, SCAN_SPAN_MAX))); half = (span - win) / 2
    cx = float(np.median([x for _, x, _ in track])) if track else 0.5
    cx = float(np.clip(cx, span / 2, 1 - span / 2)) if span < 1 else 0.5
    sign = 1.0
    if len(track) > 2:
        t = np.array([a for a, _, _ in track], float); x = np.array([b for _, b, _ in track], float)
        if np.ptp(t) > 0 and abs(np.polyfit(t, x, 1)[0]) > 1e-3: sign = float(np.sign(np.polyfit(t, x, 1)[0]))
    x0, x1 = cx - sign * half, cx + sign * half
    path = [(0.0, float(_cx(x0, win))), (float(max(length, 0.1)), float(_cx(x1, win)))]
    inside = 1.0
    if track:
        t = np.array([a for a, _, _ in track], float); x = np.array([b for _, b, _ in track], float)
        c = _cx(np.interp(t, [path[0][0], path[1][0]], [path[0][1], path[1][1]]), win)
        inside = float((np.abs(x - c) <= win / 2 + 1e-6).mean())          # the subject's CENTRE stays in the window: its whole extent is wider than the window by construction (that is why it pans)
    return {"mode": "pan", "path": path, "subjectX": cx, "inFrame": inside, "reason": f"subject spans ~{spread:.0%} (wider than the {win:.0%} window): slow pan across it instead of blurred fill"}
