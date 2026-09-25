"""Source-audio BED (--source-audio bed): the video's OWN soundtrack -- e.g. the devotional music and chanting recorded at Tirumala -- used as ONE continuous track
under a reel that is cut from anywhere in the footage. Cutting each clip's own audio chops the music at every transition; a bed keeps it whole, which is how an editor
uses concert or festival footage. No music-library licence gate is involved: it is the source's audio, exactly what the default mode already keeps.

The window is chosen from the audio alone: the T-second stretch that sounds most like sustained devotional music/chant (music-like sound events, steady level, tonal),
without speech or silence, away from the very start and end of the recording."""
from __future__ import annotations
import subprocess
from typing import Tuple
import numpy as np

MUSICAL = ("chant", "bhajan", "bell", "conch", "drums")       # sound events that mean devotional music/chanting
EDGE = 0.05                                                   # the first/last 5% of a recording is intros and fade-outs


def pick_window(audio, T: float) -> Tuple[float, float, str]:
    """(start seconds, score, why) of the best contiguous T-second stretch of `audio` (an AudioProfile)."""
    D = float(len(audio.rms_db)) * audio.hop
    if D <= T + 1.0:
        return 0.0, 0.0, "the recording is no longer than the reel: whole audio used"
    lo, hi = EDGE * D, max(EDGE * D, D * (1 - EDGE) - T)
    if hi <= lo:
        lo, hi = 0.0, D - T
    best, bs = -9.0, lo
    for s in np.arange(lo, hi + 1e-6, 1.0):
        a, b = float(s), float(s) + T
        sl = audio.slice(a, b); rms = audio.rms_db[sl]
        if not len(rms):
            continue
        mus = max([audio.event_score(k, a, b) for k in MUSICAL if k in audio.events] or [0.0])
        speech = audio.event_score("speech", a, b) if "speech" in audio.events else 0.0
        steady = 1.0 - float(np.clip(np.std(rms) / 6.0, 0.0, 1.0)); tonal = float(np.mean(audio.tonal[sl])) if len(audio.tonal) else 0.0
        sc = mus - 0.6 * speech + 0.3 * steady + 0.2 * tonal - 1.0 * audio.silence_ratio(a, b)
        if sc > best + 1e-9:
            best, bs = sc, float(s)
    return round(bs, 2), round(best, 3), f"most music-like {T:.0f}s of the source audio (score {best:.2f}) starting at {bs:.0f}s"


def extract(src: str, start: float, seconds: float, out_wav: str) -> str:
    """The chosen stretch as a 48 kHz stereo wav (the mixer's music path decodes any audio file)."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{seconds:.3f}", "-i", src, "-vn", "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", out_wav], check=True)
    return out_wav
