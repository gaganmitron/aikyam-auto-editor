"""Voiceover (--voiceover): reads the plan's own caption text aloud via espeak-ng (offline TTS, no
API key or network needed) and layers it onto the finished mix, ducking the rest of the audio
briefly under each spoken cue. Opt-in, off by default, and a no-op with no captions -- this reads
what's already being shown as captions, it does not invent new narration."""
from __future__ import annotations
import subprocess, tempfile
from dataclasses import replace
from typing import List
import numpy as np
from scipy.io import wavfile
from .. import measure as M
from . import mixer

SR = mixer.SR


def _synthesize(text: str, sr: int = SR) -> np.ndarray:
    """One cue's text -> mono float32 PCM at `sr`, via espeak-ng. Empty text or any synthesis failure
    -> silence (best-effort narration, never a reason to fail the render)."""
    if not text.strip():
        return np.zeros(0, dtype=np.float32)
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        try:
            subprocess.run(["espeak-ng", "-s", "165", "-w", tmp.name, text], check=True, capture_output=True, timeout=20)
            rate, data = wavfile.read(tmp.name)
        except Exception:                                     # noqa: BLE001
            return np.zeros(0, dtype=np.float32)
    data = data.astype(np.float32) / 32768.0
    if data.ndim > 1:
        data = data.mean(axis=1)
    if rate != sr and len(data):                               # ponytail: linear resample, good enough for speech; swap for resample_poly if quality matters later
        n = int(round(len(data) * sr / rate))
        data = np.interp(np.linspace(0, len(data) - 1, n), np.arange(len(data)), data).astype(np.float32)
    return data


def _voice_track(cues: List[dict], duration_s: float, sr: int = SR) -> np.ndarray:
    """A full-length mono buffer, silent except where each cue's synthesized speech plays, starting at
    the cue's own start time. Speech may run past the cue's caption `end` into the next cue's window
    (espeak's rate roughly matches caption pacing, so overflow is small); not time-stretched to fit."""
    out = np.zeros(int(round(duration_s * sr)), dtype=np.float32)
    for cue in cues:
        speech = _synthesize(cue.get("text", ""), sr)
        if not len(speech):
            continue
        start = int(round(cue["start"] * sr))
        if start >= len(out):
            continue
        end = min(start + len(speech), len(out))
        out[start:end] += speech[:end - start] * 0.9
    return out


def apply_voiceover(mix, cues: List[dict], duration_s: float, ceiling_db: float = mixer.CEILING_DB, duck_db: float = -8.0):
    """Layer narration onto a finished MixResult: duck the mix under active narration, add the voice,
    then re-normalize loudness and re-limit true peak so QC measures a correctly-mastered file."""
    voice = _voice_track(cues, duration_s)
    if not np.any(voice):
        return mix
    active = (np.abs(voice) > 1e-4).astype(np.float32)
    win = max(1, int(0.05 * SR))
    mask = np.convolve(active, np.ones(win) / win, mode="same") if win > 1 else active
    gain = 1.0 + mask * (10 ** (duck_db / 20) - 1.0)
    n = min(len(mix.pcm), len(voice))
    pcm = mix.pcm.copy()
    pcm[:n] = mix.pcm[:n] * gain[:n, None] + voice[:n, None]
    st = M.loudness(pcm, SR)
    g = 10 ** ((mixer.TARGET_LUFS - st["lufs"]) / 20) if np.isfinite(st["lufs"]) else 1.0
    pcm = pcm * g
    lim, _ = mixer.limit(pcm, ceiling_db)
    fin = M.loudness(lim, SR)
    report = {**mix.report, "voiceover": True, "lufs": fin["lufs"], "true_peak_db": mixer.true_peak_db(lim), "clipped": fin["clipped"]}
    return replace(mix, pcm=lim, report=report)
