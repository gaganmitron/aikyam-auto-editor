"""Audio analysis (silence, onsets, devotional sound events) + AudioProcessor (normalize/duck/mix), section 16.

Sound events (bell, conch, chant, bhajan, applause, ...) come from an AudioTagger: default is zero-shot CLAP
(laion/clap-htsat-unfused via transformers, the `[clip]` extra) scoring 5 s windows against text prompts, softmaxed
against background classes. I tried spectral heuristics first: they fired "chant" on every clip, so they were removed.
Without the extra, `events` is empty (silence/onset/tonality are still measured).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict
import numpy as np
from . import ffmpeg as ff

HOP = 0.5  # seconds per analysis frame


@dataclass
class AudioProfile:
    hop: float
    rms_db: np.ndarray          # per-hop dBFS
    silence: np.ndarray         # bool per hop
    peaks: np.ndarray           # bool per hop: sudden energy rise
    tonal: np.ndarray           # per-hop 0..1 harmonic-ness (sustained tone vs noise)
    events: Dict[str, np.ndarray] = field(default_factory=dict)

    def slice(self, start: float, end: float) -> slice:
        a = max(0, int(start / self.hop)); b = max(a + 1, int(np.ceil(end / self.hop)))
        return slice(a, b)

    def silence_ratio(self, start: float, end: float) -> float:
        s = self.silence[self.slice(start, end)]
        return float(s.mean()) if len(s) else 1.0

    def event_score(self, name: str, start: float, end: float) -> float:
        v = self.events.get(name)
        if v is None or not len(v):
            return 0.0
        s = v[self.slice(start, end)]
        return float(s.mean()) if len(s) else 0.0

    def top_event(self, start: float, end: float, min_score: float = 0.35):
        best = max(((self.event_score(n, start, end), n) for n in self.events), default=(0.0, None))
        return best[1] if best[0] >= min_score else None

    def save(self, path: str) -> None:
        np.savez_compressed(path, hop=self.hop, rms_db=self.rms_db, silence=self.silence, peaks=self.peaks,
                            tonal=self.tonal, **{f"ev_{k}": v for k, v in self.events.items()})

    @classmethod
    def load(cls, path: str) -> "AudioProfile":
        z = np.load(path)
        return cls(float(z["hop"]), z["rms_db"], z["silence"], z["peaks"], z["tonal"],
                   {k[3:]: z[k] for k in z.files if k.startswith("ev_")})


def _smooth_min(x: np.ndarray, k: int) -> np.ndarray:
    """Rolling min over k hops (a sound is 'sustained' only if all k hops are high)."""
    if len(x) < k:
        return np.zeros_like(x)
    w = np.lib.stride_tricks.sliding_window_view(np.pad(x, (k // 2, k - 1 - k // 2), mode="edge"), k)
    return w.min(axis=1)


def profile(path: str, sr: int = 16000, silence_db: float = -45.0, tagger=None) -> AudioProfile:
    x = ff.extract_audio_pcm(path, sr)
    n = int(sr * HOP)
    frames = len(x) // n
    if frames == 0:
        z = np.zeros(0)
        return AudioProfile(HOP, z, z.astype(bool), z.astype(bool), z, {})
    w = x[: frames * n].reshape(frames, n)
    rms = np.sqrt((w ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    silence = db < silence_db
    jump = np.diff(db, prepend=db[0])
    peaks = (jump >= 9.0) & (db > silence_db + 10)

    spec = np.abs(np.fft.rfft(w * np.hanning(n), axis=1)) + 1e-9
    freqs = np.fft.rfftfreq(n, 1 / sr)
    pw = spec ** 2
    tot = pw.sum(axis=1)
    band = lambda lo, hi: pw[:, (freqs >= lo) & (freqs < hi)].sum(axis=1) / tot
    centroid = (pw * freqs).sum(axis=1) / tot
    flat = np.exp(np.log(spec).mean(axis=1)) / spec.mean(axis=1)
    tonal = np.clip(1.0 - flat * 4, 0, 1) * (~silence)

    ev: Dict[str, np.ndarray] = {}
    if tagger is not None:
        ev = tagger.tag(path, frames, HOP)
    return AudioProfile(HOP, db, silence, peaks, tonal, ev)


class AudioProcessor:
    """ffmpeg filter-graph builders; render.py wires them (loudnorm always, ducking when music is enabled)."""

    @staticmethod
    def normalize(target_lufs: float = -16.0) -> str:
        return f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11"

    @staticmethod
    def duck(voice: str = "0:a", music: str = "1:a", out: str = "ducked", threshold: float = 0.05, ratio: float = 5.0) -> str:
        """Compress the music under the voice/original audio (sidechain)."""
        return f"[{music}][{voice}]sidechaincompress=threshold={threshold}:ratio={ratio}:attack=20:release=400[{out}]"

    @staticmethod
    def mix(inputs: int = 2) -> str:
        return f"amix=inputs={inputs}:duration=first:normalize=0"


# label -> phrasings. The FIRST group is what we want to detect; BACKGROUND competes in the softmax.
SOUND_PROMPTS: Dict[str, list] = {
    "bell": ["a temple bell ringing", "a hand bell being rung"],
    "conch": ["a conch shell being blown", "a shankh blowing sound"],
    "chant": ["people chanting mantras in unison", "a priest reciting Sanskrit shlokas"],
    "bhajan": ["devotional singing with harmonium and tabla", "a group singing bhajans with instruments"],
    "drums": ["loud drumming, dhol and nagara drums"],
    "applause": ["a crowd clapping and cheering"],
    "speech": ["a person speaking"],
}
BACKGROUND = ["crowd noise and chatter", "wind noise", "traffic and vehicles", "silence", "fireworks and explosions", "electrical hum"]


class ClapAudioTagger:
    """Zero-shot audio tagging with CLAP. 5 s windows every 2.5 s -> per-0.5 s-hop scores (softmax vs background)."""
    name = "clap"

    def __init__(self, model="laion/clap-htsat-unfused", win=5.0, step=2.5):
        import torch
        from transformers import ClapModel, ClapProcessor
        torch.set_num_threads(max(1, (__import__("os").cpu_count() or 2) // 2))
        self.torch, self.win, self.step = torch, win, step
        self.model = ClapModel.from_pretrained(model).eval(); self.proc = ClapProcessor.from_pretrained(model)
        self.keys = list(SOUND_PROMPTS)
        flat = [(k, p) for k, ps in SOUND_PROMPTS.items() for p in ps]
        self.owner = np.array([self.keys.index(k) for k, _ in flat])
        texts = [p for _, p in flat] + BACKGROUND
        with torch.no_grad():
            t = self.model.get_text_features(**self.proc(text=texts, return_tensors="pt", padding=True))
        self.txt = (t / t.norm(dim=-1, keepdim=True)).numpy()

    def tag(self, path: str, frames: int, hop: float) -> Dict[str, np.ndarray]:
        x = ff.extract_audio_pcm(path, 48000)
        W, S = int(self.win * 48000), int(self.step * 48000)
        starts = list(range(0, max(1, len(x) - W // 2), S)) or [0]
        scale = float(self.model.logit_scale_a.exp())
        out = {k: np.zeros(len(starts)) for k in self.keys}
        for i in range(0, len(starts), 8):
            chunk = [x[s:s + W] for s in starts[i:i + 8]]
            with self.torch.no_grad():
                a = self.model.get_audio_features(**self.proc(audios=chunk, sampling_rate=48000, return_tensors="pt"))
            a = (a / a.norm(dim=-1, keepdim=True)).numpy()
            p = np.exp(scale * (a @ self.txt.T)); p /= p.sum(axis=1, keepdims=True)   # softmax over phrasings+background
            for j in range(len(chunk)):
                for ki, k in enumerate(self.keys):
                    out[k][i + j] = p[j, :len(self.owner)][self.owner == ki].sum()
        centers = np.array([(s + W / 2) / 48000 for s in starts])
        hop_t = (np.arange(frames) + 0.5) * hop
        idx = np.abs(hop_t[:, None] - centers[None, :]).argmin(axis=1)
        return {k: v[idx] for k, v in out.items()}
