"""Audio engine: builds the reel's final sound deterministically in numpy and returns the STEMS as well, so QC can measure the mix, not guess it.

  live bed  : each clip's source audio, high-passed (rumble/wind), level-matched to its neighbours, joined with equal-power crossfades whose
              windows follow the plan's transitions and J/L-cut leads (sound arrives before / lingers after the picture).
  music bed : one licensed track from the chosen offset, gain-AUTOMATED so it sits below the live sound by a margin that grows with how devotional
              the live sound is (ambience-only 8 dB, chant/speech/bhajan 16 dB), with fast attack / slow release so it never pumps or overpowers.
  master    : integrated loudness to the target (BS.1770-4), then a look-ahead true-peak limiter (ceiling in dBTP).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import numpy as np
from scipy.signal import butter, resample_poly, sosfilt
from .. import ffmpeg as ff, measure as M
from ..plan import edge_transitions, edge_overlaps, seg_source, timeline

SR = 48000
TARGET_LUFS = -16.0        # short-form social feeds normalise around -14..-16 LUFS; -16 leaves headroom for their own gain
CEILING_DB = -1.5          # true-peak ceiling: AAC encoding can overshoot the limiter by ~1 dB, and -1.0 dBTP is the common delivery limit
MICRO_FADE = 0.02          # hard cuts still get a 20 ms equal-power fade: no clicks
FRAME = 0.05               # gain-automation resolution (50 ms)


@dataclass
class MixResult:
    pcm: np.ndarray                                   # (n, 2) float32 final mix
    live: np.ndarray                                  # live stem after master gain/limiter, same length
    music: np.ndarray                                 # music stem after gain automation + master gain
    report: Dict[str, float] = field(default_factory=dict)
    music_gain_db: Optional[np.ndarray] = None        # per FRAME
    live_db: Optional[np.ndarray] = None
    music_db: Optional[np.ndarray] = None
    presence: Optional[np.ndarray] = None


def decode_range(path: str, start: float, dur: float, sr: int = SR) -> np.ndarray:
    """Stereo float32 of [start, start+dur); zero-padded if the source is shorter (or start < 0)."""
    n = int(round(dur * sr))
    if dur <= 0:
        return np.zeros((0, 2), np.float32)
    s0 = max(0.0, start); pad_l = int(round((s0 - start) * sr))
    x = M.decode_audio_range(path, s0, dur - (s0 - start), sr) if pad_l < n else np.zeros((0, 2), np.float32)
    out = np.zeros((n, 2), np.float32); m = min(len(x), n - pad_l); out[pad_l:pad_l + m] = x[:m]
    return out


def _hp(x: np.ndarray, hz: float = 70.0) -> np.ndarray:
    return sosfilt(butter(2, hz, "highpass", fs=SR, output="sos"), x, axis=0).astype(np.float32) if len(x) else x


def _fade(n: int, up: bool) -> np.ndarray:
    """Equal-power (sine) fade of n samples: sum of an up and a down fade keeps constant power across a crossfade."""
    if n <= 0:
        return np.zeros(0, np.float32)
    t = np.linspace(0, np.pi / 2, n, dtype=np.float32)
    return np.sin(t) if up else np.cos(t)


def _win_db(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    """RMS (dB) of mono-summed x in windows of `win` samples every `hop`."""
    m = x.mean(axis=1) if x.ndim == 2 else x
    n = max(1, (len(m) - win) // hop + 1)
    c = np.concatenate([[0], np.cumsum(m.astype(np.float64) ** 2)])
    idx = np.arange(n) * hop
    return 10 * np.log10((c[np.minimum(idx + win, len(m))] - c[idx]) / win + 1e-12)


def _smooth_db(g: np.ndarray, attack: float, release: float) -> np.ndarray:
    """Asymmetric one-pole on a dB curve sampled every FRAME: drops fast (attack), recovers slowly (release)."""
    ka, kr = 1 - np.exp(-FRAME / attack), 1 - np.exp(-FRAME / release); out = np.empty_like(g); y = g[0]
    for i, v in enumerate(g):
        y += (ka if v < y else kr) * (v - y); out[i] = y
    return out


def _presence(fn, s: dict, t: float) -> float:
    """presence_fn is either one callable t -> score, or {assetId: callable} for multi-asset plans."""
    f = fn.get(s.get("assetId")) if isinstance(fn, dict) else fn
    return f(t) if f else 0.0


def _decode_seg(plan: dict, s: dict, src: str, start: float, dur: float) -> np.ndarray:
    p = seg_source(plan, s, src)
    return decode_range(p, start, dur) if p else np.zeros((int(round(max(dur, 0) * SR)), 2), np.float32)     # a still is silent


def build_live(plan: dict, src: str, presence_fn=None) -> Dict:
    """Join the clips' audio on the output timeline. Returns {'live', 'starts', 'presence' (per FRAME on the output timeline), 'gains'}."""
    segs, tr = plan["segments"], plan["transitions"]
    tl = timeline(segs, tr); ovs = edge_overlaps(segs, tr); total = int(round(plan["durationSeconds"] * SR))
    leads = [0.0] + [float(s.get("audioLead", 0.0)) for s in segs[1:]]
    clips, levels = [], []
    for i, s in enumerate(segs):
        lead_in = max(leads[i], 0.0) if i > 0 else 0.0                                   # J-cut: pre-roll before this clip's picture
        lead_out = max(-leads[i + 1], 0.0) if i + 1 < len(segs) else 0.0                 # L-cut: this clip's sound lingers into the next
        ov_in = ovs[i - 1] if i > 0 else 0.0; ov_out = ovs[i] if i < len(ovs) else 0.0
        pre, post = lead_in, lead_out
        a = _hp(_decode_seg(plan, s, src, 0.0 if s.get("kind") == "image" else s["start"] - pre, (s["end"] - s["start"]) + pre + post))
        gain_db = float(s.get("liveGainDb", 0.0)); a = a * (10 ** (gain_db / 20))
        # fade windows (seconds): in-window covers [start-lead_in, start+ov_in], out-window covers [end-ov_out, end+lead_out]
        fin = int(round(max(lead_in + ov_in, MICRO_FADE if i > 0 else 0.005) * SR)); fout = int(round(max(ov_out + lead_out, MICRO_FADE if i + 1 < len(segs) else 0.0) * SR))
        env = np.ones(len(a), np.float32); fin = min(fin, len(a)); fout = min(fout, len(a))
        env[:fin] *= _fade(fin, True); env[len(a) - fout:] *= _fade(fout, False)
        clips.append((int(round((tl[i]["start"] - pre) * SR)), a * env[:, None]))
    live = np.zeros((total + 4 * SR, 2), np.float32)
    for pos, a in clips:
        lo = max(pos, 0); hi = min(pos + len(a), len(live)); live[lo:hi] += a[lo - pos: hi - pos]
    live = live[:total]
    n_frames = int(np.ceil(plan["durationSeconds"] / FRAME)); pres = np.zeros(n_frames)
    if presence_fn:
        for i, s in enumerate(segs):
            f0, f1 = int(tl[i]["start"] / FRAME), min(n_frames, int(np.ceil(tl[i]["end"] / FRAME)))
            for f in range(f0, f1): pres[f] = max(pres[f], _presence(presence_fn, s, s["start"] + (f * FRAME - tl[i]["start"])))
    return {"live": live, "presence": pres, "timeline": tl}


def level_match(plan: dict, src: str, max_db: float = 6.0) -> List[float]:
    """Per-clip gain (dB) pulling every clip's live loudness toward the median (ambience differs a lot between shots): |gain| <= max_db."""
    lv = []
    for s in plan["segments"]:
        p = seg_source(plan, s, src)
        lv.append(M.rms_db(_hp(decode_range(p, s["start"], min(s["end"] - s["start"], 6.0)))) if p else -90.0)      # stills carry no live sound
    lv = np.array(lv); med = float(np.median(lv[lv > -60])) if (lv > -60).any() else 0.0
    return [float(np.clip(med - v, -max_db, max_db)) if v > -60 else 0.0 for v in lv]


def true_peak_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.abs(resample_poly(x, 4, 1, axis=0)).max() + 1e-12)) if len(x) else -120.0


def limit(x: np.ndarray, ceiling_db: float = CEILING_DB, lookahead: float = 0.005, release: float = 0.12):
    """Look-ahead true-peak limiter. Returns (limited signal, per-sample gain curve) so the SAME gain can be applied to the stems.
    Gain is lowered ahead of each (4x oversampled) peak and recovers exponentially: no clipping, no zipper noise."""
    from numpy.lib.stride_tricks import sliding_window_view
    ceil = 10 ** (ceiling_db / 20); la = max(1, int(lookahead * SR)); rel = np.exp(-1.0 / (release * SR))
    total = np.ones(len(x), np.float32)
    for _ in range(3):                                                             # a pass can leave inter-sample overs: verify and repeat
        up = np.abs(resample_poly(x, 4, 1, axis=0)).max(axis=1)
        need = np.minimum(1.0, ceil / np.maximum(up, 1e-9)).reshape(-1, 4).min(axis=1)[:len(x)]
        if need.min() >= 0.9999:
            break
        gm = sliding_window_view(np.pad(need, (0, la), constant_values=1.0), la + 1).min(axis=1)   # min over [i, i+la]: down BEFORE the peak arrives
        out = np.empty_like(gm); y = 1.0
        for i in range(len(gm)):
            y = gm[i] if gm[i] < y else y + (gm[i] - y) * (1 - rel)
            out[i] = y
        g = out.astype(np.float32); x = (x * g[:, None]).astype(np.float32); total *= g
    return x, total


def render_audio(plan: dict, src: str, music_path: Optional[str] = None, music_offset: float = 0.0, presence_fn=None,
                 volume: float = 0.5, target_lufs: float = TARGET_LUFS, ceiling_db: float = CEILING_DB, match_levels: bool = True, live_on: bool = True) -> MixResult:
    """live_on=False: the recorded sound is dropped entirely -- the reel's sound is the music alone (nothing to duck under, nothing to cut through)."""
    if match_levels and live_on:
        gains = level_match(plan, src)
        for s, g in zip(plan["segments"], gains): s.setdefault("liveGainDb", g)
    if live_on:
        bed = build_live(plan, src, presence_fn); live = bed["live"]; total = len(live); presence_in = bed["presence"]
    else:
        total = int(round(plan["durationSeconds"] * SR)); live = np.zeros((total, 2), np.float32); presence_in = np.zeros(int(np.ceil(plan["durationSeconds"] / FRAME)))
    T = plan["durationSeconds"]
    n_frames = int(np.ceil(T / FRAME)); presence = presence_in
    L = _win_db(live, int(0.4 * SR), int(FRAME * SR)); L = np.concatenate([L, np.full(max(0, n_frames - len(L)), L[-1] if len(L) else -90.0)])[:n_frames]
    music = np.zeros_like(live); gain_db = np.zeros(n_frames); M0 = np.full(n_frames, -90.0)
    if music_path and total:
        mus = decode_range(music_path, music_offset, T)
        m_all = _win_db(mus, int(0.4 * SR), int(FRAME * SR)); M0 = np.concatenate([m_all, np.full(max(0, n_frames - len(m_all)), m_all[-1])])[:n_frames]
        if live_on:                                                                     # duck under the recorded sound; with none there is nothing to duck under (gain_db stays 0 dB)
            active = L > max(-55.0, float(np.median(L)) - 25.0)                              # the live sound is there (steady chanting has no quiet frames, so no percentile floor)
            live_ref = float(np.median(L[active])) if active.any() else float(np.median(L))
            base = live_ref - 8.0 + 20 * np.log10(max(volume, 1e-3))                          # music bed level when the live sound is quiet
            margin = 8.0 + 8.0 * np.clip(presence, 0, 1)                                      # 8 dB (ambience) .. 16 dB (chant/speech/bhajan)
            tgt = np.where(active, np.minimum(base, L - margin), base)
            g = np.clip(tgt - M0, -40.0, 12.0); g = _smooth_db(g, attack=0.08, release=0.5)
            # smoothing lags by up to the release time: never let it push the music ABOVE the ceiling implied by the margin
            g = np.minimum(g, np.clip(np.where(active, L - margin, base) - M0, -40.0, 12.0) + 3.0)
            gain_db = g
        env = np.interp(np.arange(total) / SR, np.arange(n_frames) * FRAME, gain_db)
        music = (mus[:total] * (10 ** (env / 20))[:, None]).astype(np.float32)
        fi, fo = int(0.8 * SR), int(1.5 * SR)
        music[:fi] *= _fade(fi, True)[:, None]; music[-fo:] *= _fade(fo, False)[:, None]
    mix = live + music
    st = M.loudness(mix, SR); gain = 10 ** ((target_lufs - st["lufs"]) / 20) if np.isfinite(st["lufs"]) else 1.0
    mix, live, music = mix * gain, live * gain, music * gain
    lim, gcurve = limit(mix, ceiling_db)
    live, music = live * gcurve[:, None], music * gcurve[:, None]                       # the very same gain on the stems
    fin = M.loudness(lim, SR)
    rep = {"lufs": fin["lufs"], "lra": fin["lra"], "true_peak_db": true_peak_db(lim), "clipped": fin["clipped"], "master_gain_db": float(20 * np.log10(gain + 1e-12)),
           "music_used": bool(music_path), "music_offset": music_offset}
    return MixResult(lim, live, music, rep, gain_db, L, M0 + gain_db, presence)
