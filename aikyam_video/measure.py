"""Objective measurements of rendered media (used by the baseline evaluation and by the QC stage).
Standards where they exist: loudness = ITU-R BS.1770-4 (pyloudnorm) with EBU R128-style gating; true peak = 4x oversampled.
Everything else is reported as raw numbers; QC decides pass/fail against adaptive per-video statistics or documented limits."""
from __future__ import annotations
import re, subprocess
from typing import Dict, List, Optional, Tuple
import numpy as np


def decode_audio(path: str, sr: int = 48000) -> np.ndarray:
    """Stereo float32 (n, 2). Video-only files return zeros(0, 2)."""
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "2", "-ar", str(sr), "-f", "f32le", "-"], capture_output=True)
    if p.returncode or not p.stdout:
        return np.zeros((0, 2), np.float32)
    return np.frombuffer(p.stdout, np.float32).reshape(-1, 2)


def decode_audio_range(path: str, start: float, dur: float, sr: int = 48000) -> np.ndarray:
    """Stereo float32 of [start, start+dur) decoded by ffmpeg (fast seek); shorter than requested near the end of the file."""
    p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.4f}", "-t", f"{dur:.4f}", "-i", path, "-vn", "-ac", "2", "-ar", str(sr), "-f", "f32le", "-"], capture_output=True)
    return np.frombuffer(p.stdout, np.float32).reshape(-1, 2) if p.stdout else np.zeros((0, 2), np.float32)


def loudness(x: np.ndarray, sr: int = 48000) -> Dict[str, float]:
    """Integrated LUFS (BS.1770-4), loudness range (EBU R128 style, gated short-term), true peak (dBTP), clipping."""
    import pyloudnorm as pyln
    from scipy.signal import resample_poly
    if len(x) < int(0.4 * sr):
        return {"lufs": float("-inf"), "lra": 0.0, "true_peak_db": float("-inf"), "sample_peak": 0.0, "clipped": 0}
    m = pyln.Meter(sr)
    lufs = float(m.integrated_loudness(x.astype(np.float64)))
    st = []                                                                   # short-term (3 s) loudness every 1 s
    for i in range(0, max(1, len(x) - 3 * sr + 1), sr):
        seg = x[i:i + 3 * sr]
        if len(seg) >= 3 * sr:
            st.append(m.integrated_loudness(seg.astype(np.float64)))
    st = np.array([v for v in st if np.isfinite(v) and v > -70.0])            # absolute gate
    if len(st) >= 2:
        st = st[st > 10 * np.log10(np.mean(10 ** (st / 10))) - 20.0]         # relative gate: 20 LU below the mean (EBU Tech 3342)
    lra = float(np.percentile(st, 95) - np.percentile(st, 10)) if len(st) >= 2 else 0.0
    tp = float(np.abs(resample_poly(x, 4, 1, axis=0)).max())
    return {"lufs": lufs, "lra": lra, "true_peak_db": 20 * np.log10(tp + 1e-12), "sample_peak": float(np.abs(x).max()),
            "clipped": int((np.abs(x) >= 0.999).sum())}


def rms_db(x: np.ndarray) -> float:
    return float(10 * np.log10((x.astype(np.float64) ** 2).mean() + 1e-12)) if len(x) else -120.0


def silence_stats(x: np.ndarray, sr: int = 48000, thresh_db: float = -50.0, win: float = 0.05) -> Dict[str, float]:
    if not len(x):
        return {"ratio": 1.0, "longest_s": 0.0}
    n = int(sr * win); m = x[: len(x) // n * n].mean(axis=1).reshape(-1, n)
    quiet = 10 * np.log10((m ** 2).mean(axis=1) + 1e-12) < thresh_db
    longest = run = 0
    for q in quiet:
        run = run + 1 if q else 0; longest = max(longest, run)
    return {"ratio": float(quiet.mean()), "longest_s": longest * win}


def frame_series(path: str, fps: float = 10.0, size: Tuple[int, int] = (96, 54)) -> np.ndarray:
    """Gray low-res frames (N, h, w) uint8 sampled at `fps`."""
    w, h = size
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-an", "-vf", f"fps={fps},scale={w}:{h}:flags=area,format=gray", "-f", "rawvideo", "-"], capture_output=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, h, w)


def frame_diffs(frames: np.ndarray) -> np.ndarray:
    """Mean absolute luma change between consecutive sampled frames (0..255); length N-1."""
    return np.abs(np.diff(frames.astype(np.int16), axis=0)).mean(axis=(1, 2)) if len(frames) > 1 else np.zeros(0)


def robust_z(v: float, series: np.ndarray) -> float:
    """How many robust standard deviations (median/MAD) `v` is above the series' typical value: adaptive, per-video."""
    if not len(series):
        return 0.0
    med = float(np.median(series)); mad = float(np.median(np.abs(series - med))) * 1.4826
    return (v - med) / max(mad, 0.25)                                     # floor: 0.25 luma levels (sensor-noise scale)


def _ffmpeg_filter_events(path: str, vf: str, start_re: str, end_re: str) -> List[Tuple[float, float]]:
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-vf", vf, "-an", "-f", "null", "-"], capture_output=True)
    err = p.stderr.decode(errors="replace")
    s, e = re.findall(start_re, err), re.findall(end_re, err)
    dur = _duration(path)
    return [(float(a), float(e[i]) if i < len(e) else dur) for i, a in enumerate(s)]


def _duration(path: str) -> float:
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path], capture_output=True, text=True)
    return float(p.stdout.strip() or 0)


def black_intervals(path: str, min_len: float = 0.25) -> List[Tuple[float, float]]:
    """Studio-black (luma <= 16/255 on >= 98% pixels) runs of >= min_len seconds."""
    return _ffmpeg_filter_events(path, f"blackdetect=d={min_len}:pix_th=0.06:pic_th=0.98", r"black_start:([\d.]+)", r"black_end:([\d.]+)")


def freeze_intervals(path: str, min_len: float = 0.6) -> List[Tuple[float, float]]:
    """Frozen picture (frame-to-frame change below -60 dB) for >= min_len seconds."""
    return _ffmpeg_filter_events(path, f"freezedetect=n=-60dB:d={min_len}", r"freeze_start: ([\d.]+)", r"freeze_end: ([\d.]+)")


def audio_jump_db(x: np.ndarray, t: float, sr: int = 48000, win: float = 0.12) -> float:
    """Level change across time t (RMS after minus RMS before, dB): large values are audible discontinuities."""
    i = int(t * sr); w = int(win * sr)
    if i - w < 0 or i + w > len(x):
        return 0.0
    return rms_db(x[i:i + w]) - rms_db(x[i - w:i])
