"""FFmpeg/ffprobe helpers (packages/ffmpeg)."""
from __future__ import annotations
import json, subprocess
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


class FFmpegError(RuntimeError):
    pass


def run(args: List[str]) -> subprocess.CompletedProcess:
    p = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
                       capture_output=True)
    if p.returncode:
        raise FFmpegError(p.stderr.decode(errors="replace")[-2000:])
    return p


HDR_TRANSFERS = {"arib-std-b67", "smpte2084"}


@dataclass
class MediaInfo:
    duration: float
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: Optional[str]
    size_bytes: int
    color_transfer: Optional[str] = None      # arib-std-b67 (HLG) / smpte2084 (PQ) = HDR

    @property
    def hdr(self) -> bool:
        return self.color_transfer in HDR_TRANSFERS

    @property
    def has_audio(self) -> bool:
        return self.audio_codec is not None


def probe(path: str) -> MediaInfo:
    p = subprocess.run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", path],
                       capture_output=True)
    if p.returncode:
        raise FFmpegError(f"ffprobe failed for {path}: {p.stderr.decode(errors='replace')}")
    d = json.loads(p.stdout)
    v = next((s for s in d["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in d["streams"] if s["codec_type"] == "audio"), None)
    if v is None:
        raise FFmpegError(f"no video stream in {path}")
    num, den = (v.get("avg_frame_rate") or "0/1").split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    try:
        dur = float(d["format"].get("duration") or v.get("duration") or 0)
    except ValueError:                       # still images report "N/A"
        dur = 0.0
    return MediaInfo(dur, int(v["width"]), int(v["height"]), fps, v["codec_name"],
                     a["codec_name"] if a else None, int(d["format"].get("size", 0)), v.get("color_transfer"))


def has_audio_stream(path: str) -> bool:
    """True if the file has an audio stream (works for audio-only files, unlike probe() which needs video)."""
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", path], capture_output=True)
    return bool(p.stdout.strip())


def extract_audio_pcm(path: str, sr: int = 16000) -> np.ndarray:
    """Mono float32 samples in [-1, 1]; empty array if no audio."""
    if not has_audio_stream(path):                             # picture-only file: ffmpeg would fail with "no stream"
        return np.zeros(0, np.float32)
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr),
                        "-f", "s16le", "-"], capture_output=True)
    if p.returncode:
        raise FFmpegError(p.stderr.decode(errors="replace")[-2000:])
    return np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def frame_at(path: str, t: float, out_jpg: str) -> None:
    run(["-ss", f"{t:.3f}", "-i", path, "-frames:v", "1", "-q:v", "2", out_jpg])
