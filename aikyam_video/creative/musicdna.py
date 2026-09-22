"""Music analysis: tempo, beat grid, energy curve and character of a track. Cached next to the file (<track>.analysis.json).

Tempo: onset-strength envelope (positive log-spectral flux) -> autocorrelation -> peak under a log-normal tempo prior (people cut and dance around 100 BPM;
the prior only resolves the octave ambiguity 52/104/208). Confidence = how much the winning lag stands out of the autocorrelation; below CONF the track is
treated as beatless (a tanpura drone, a free flute) and NO beat sync is attempted: better none than a wrong grid.
"""
from __future__ import annotations
import hashlib, json, os
from dataclasses import dataclass, asdict, field
from typing import List, Optional
import numpy as np
from .. import ffmpeg as ff

SR, HOP, NFFT = 22050, 512, 1024
MIN_REL_FLUX = 0.01    # onset strength relative to the spectral level: a steady tone is 0.001, every real signal measured (music, noise, plucks) >= 0.09
MIN_ONSET_RATE = 0.9  # strong onsets per second: below this (a drone, a pad) there is nothing to lock a grid to
CONF = 3.0          # peak-to-median ratio of the autocorrelation above which a steady beat is claimed


@dataclass
class MusicAnalysis:
    duration: float
    bpm: Optional[float]
    beat_conf: float
    beats: List[float] = field(default_factory=list)
    energy: List[float] = field(default_factory=list)         # per 0.5 s, 0..1 (rank within the track)
    kind: str = "melodic"                                     # rhythmic | drone | melodic
    onset_rate: float = 0.0                                   # strong onsets per second
    downbeats: List[float] = field(default_factory=list)      # bar starts (Beat This! only)
    tracker: str = "autocorr"                                 # autocorr | beat_this


def _onset_env(x: np.ndarray) -> np.ndarray:
    win = np.hanning(NFFT); n = 1 + (len(x) - NFFT) // HOP
    fr = np.lib.stride_tricks.as_strided(x, (n, NFFT), (x.strides[0] * HOP, x.strides[0])) * win
    mag = np.abs(np.fft.rfft(fr, axis=1)); mel = np.log1p(30 * mag[:, 2:200])                  # low-mid emphasis: kick / dhol / plucks
    flux = np.maximum(0, np.diff(mel, axis=0)).sum(axis=1)
    return np.concatenate([[0], flux]), float(flux.mean() / (mel.sum(axis=1).mean() + 1e-9))


def analyze_samples(x: np.ndarray) -> MusicAnalysis:
    dur = len(x) / SR
    env, rel_flux = _onset_env(x.astype(np.float32)); env = env - np.convolve(env, np.ones(43) / 43, mode="same")      # remove the slow trend (~1 s)
    env = np.maximum(env, 0); fps = SR / HOP
    ac = np.correlate(env, env, "full")[len(env) - 1:]; ac = ac / (ac[0] + 1e-9)
    lo, hi = int(fps * 60 / 180), int(fps * 60 / 60)                                              # 60..180 BPM
    lags = np.arange(lo, hi); bpm_l = 60 * fps / lags
    prior = np.exp(-0.5 * (np.log2(bpm_l / 100.0) / 0.6) ** 2)
    sc = ac[lo:hi] * prior; k = int(np.argmax(sc)); lag = float(lags[k])
    if 0 < k < len(sc) - 1:                                                                     # parabolic interpolation: sub-frame lag => tempo error << 1 BPM
        y0, y1, y2 = ac[lo + k - 1], ac[lo + k], ac[lo + k + 1]; den = y0 - 2 * y1 + y2
        if abs(den) > 1e-12: lag += float(np.clip(0.5 * (y0 - y2) / den, -1, 1))
    conf = float(ac[int(round(lag))] / (np.median(np.abs(ac[lo:hi])) + 1e-6))
    thr = env.mean() + 1.5 * env.std(); onsets = int(((env[1:-1] > thr) & (env[1:-1] > env[:-2]) & (env[1:-1] >= env[2:])).sum())
    rate = onsets / max(dur, 1e-9)
    # significance: real periodicity must beat what the SAME envelope gives with its time order destroyed (deterministic shuffles) -> no magic threshold
    rng = np.random.default_rng(0); nulls = []
    for _ in range(24):
        sh = rng.permutation(env); a2 = np.correlate(sh, sh, "full")[len(sh) - 1:]; a2 = a2 / (a2[0] + 1e-9)
        nulls.append(float(a2[lo:hi].max() / (np.median(np.abs(a2[lo:hi])) + 1e-6)))
    significant = conf >= 1.5 * float(np.max(nulls))
    bpm = float(60 * fps / lag) if (conf >= CONF and rate >= MIN_ONSET_RATE and significant and rel_flux >= MIN_REL_FLUX) else None     # needs onsets AND periodicity beyond chance
    beats: List[float] = []
    if bpm:
        p = 60.0 / bpm * fps                                                                     # beat period in frames (float)
        best, ph = -1.0, 0.0
        for cand in np.arange(0, p, 0.5):
            idx = np.arange(cand, len(env) - 1, p).astype(int); s = env[idx].sum()
            if s > best: best, ph = s, cand
        for t in np.arange(ph, len(env) - 1, p):                                                # snap each grid point to the local onset peak (+-15% of a beat)
            w = int(0.15 * p); i = int(round(t)); seg = env[max(0, i - w): i + w + 1]
            beats.append(float((max(0, i - w) + int(np.argmax(seg))) / fps))
    hop = int(0.5 * SR); n = len(x) // hop
    e = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1)); en = (np.argsort(np.argsort(e)) / max(n - 1, 1)).tolist()
    kind = "rhythmic" if bpm else ("drone" if rate < 0.9 else "melodic")
    return MusicAnalysis(dur, bpm, conf, beats, en, kind, rate)


BT_CV, BT_MIN_BEATS = 0.1, 8     # Beat This! beats count as a steady grid when their intervals vary < 10% (std/median): drones and free flutes get scattered beats (CV 0.3-0.6)


def _with_beat_this(a: MusicAnalysis, bt: dict) -> MusicAnalysis:
    """Replace tempo/beats by Beat This!'s when it found a steady grid; otherwise the track is beatless (its scattered beats are not a grid to cut to)."""
    b = np.asarray(bt["beats"], float); iv = np.diff(b)
    if len(b) >= BT_MIN_BEATS and float(iv.std() / np.median(iv)) < BT_CV and 40 <= 60 / float(np.median(iv)) <= 220:
        bpm = float(60 / np.median(iv))
        return MusicAnalysis(a.duration, bpm, float(1 / max(iv.std() / np.median(iv), 1e-3)), b.tolist(), a.energy, "rhythmic", a.onset_rate, list(bt["downbeats"]), "beat_this")
    return MusicAnalysis(a.duration, None, 0.0, [], a.energy, "drone" if a.onset_rate < MIN_ONSET_RATE else "melodic", a.onset_rate, [], "beat_this")


def analyze_track(path: str, use_cache: bool = True) -> MusicAnalysis:
    from . import beats as _beats
    st = os.stat(path); key = hashlib.sha1(f"{st.st_size}:{int(st.st_mtime)}:{'bt' if _beats.python_exe() else 'ac'}".encode()).hexdigest()[:12]
    cache = path + ".analysis.json"
    if use_cache and os.path.exists(cache):
        d = json.load(open(cache))
        if d.get("_key") == key:
            return MusicAnalysis(**{k: v for k, v in d.items() if k != "_key"})
    x = ff.extract_audio_pcm(path, SR)
    a = analyze_samples(x)
    bt = _beats.track(path, use_cache)
    if bt is not None:
        a = _with_beat_this(a, bt)
    if use_cache:
        try: json.dump({**asdict(a), "_key": key}, open(cache, "w"))
        except OSError: pass
    return a
