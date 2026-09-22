"""Local voice denoise (adapted from claude-youtube-editor clean_voice.py, MIT; see NOTICE). RNNoise via ffmpeg arnndn, then RMS-match back to the source so levels are preserved. Speech model: do NOT run it on chant/music."""
from __future__ import annotations
import os, re, subprocess, tempfile
from . import ffmpeg as ff

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "rnnoise", "sh.rnnn")
CEIL_DB = -1.0


def _run(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode: raise RuntimeError(f"{' '.join(cmd[:3])} failed:\n{r.stdout[-1500:]}")
    return r.stdout


def _stat(path: str, label: str):
    m = re.search(rf"{label}:\s*(-?[\d.]+)", _run(["ffmpeg", "-hide_banner", "-i", path, "-af", "astats=metadata=1", "-f", "null", os.devnull]))
    return float(m.group(1)) if m else None


def denoise(src: str, dst: str, model: str = MODEL, preserve_loudness: bool = True) -> dict:
    """src (video or audio) -> dst with the voice denoised. Video is stream-copied. Returns {gain_db, src_rms, clean_rms}."""
    if not ff.has_audio_stream(src): raise ValueError(f"{src} has no audio stream")
    has_v = "video" in _run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", src])
    with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(dst))) as w:
        a_in, a_cl = os.path.join(w, "in.wav"), os.path.join(w, "clean.wav")
        _run(["ffmpeg", "-y", "-hide_banner", "-i", src, "-vn", "-ac", "1", "-ar", "44100", "-c:a", "pcm_s16le", a_in])
        # the model path sits inside the filtergraph, so it must be relative to cwd (no ':' or '\\' escapes)
        r = subprocess.run(["ffmpeg", "-y", "-hide_banner", "-i", "in.wav", "-af", f"highpass=f=90,arnndn=m={os.path.relpath(model, w)}", "-ar", "44100", "clean.wav"],
                           cwd=w, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if r.returncode: raise RuntimeError(f"arnndn failed:\n{r.stdout[-1500:]}")
        s_rms, c_rms, c_pk = _stat(a_in, "RMS level dB"), _stat(a_cl, "RMS level dB"), _stat(a_cl, "Peak level dB")
        gain = 0.0
        if preserve_loudness and None not in (s_rms, c_rms, c_pk):
            gain = min(s_rms - c_rms, CEIL_DB - c_pk)          # flat gain, capped so the peak never passes the ceiling
        if has_v:
            _run(["ffmpeg", "-y", "-hide_banner", "-i", src, "-i", a_cl, "-filter_complex", f"[1:a]volume={gain:.2f}dB,apad[a]", "-map", "0:v:0", "-map", "[a]", "-shortest", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", dst])
        else:
            _run(["ffmpeg", "-y", "-hide_banner", "-i", a_cl, "-af", f"volume={gain:.2f}dB", dst])
    return {"gain_db": gain, "src_rms": s_rms, "clean_rms": c_rms}
