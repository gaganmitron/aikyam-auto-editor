import subprocess, numpy as np
from aikyam_video.audio_clean import denoise
from aikyam_video.creative.verify import verify_render
from aikyam_video.models import Transcript, TranscriptSegment, Word
from aikyam_video import measure as M


def tr(words, off=0.0, conf=0.9):
    ws = [Word(start=i + off, end=i + off + .5, text=t, confidence=conf) for i, t in enumerate(words)]
    return Transcript(language="en", segments=[TranscriptSegment(start=ws[0].start, end=ws[-1].end, text=" ".join(words), words=ws)])


PLAN = {"segments": [{"start": 0.0, "end": 6.0, "assetId": "v1"}], "transitions": {}}


def by(cs): return {c.name: c for c in cs}


def test_verify_clean_match():
    c = by(verify_render(PLAN, {"v1": tr("om namah shivaya om namah shivaya".split())}, tr("om namah shivaya om namah shivaya".split())))
    assert all(x.status == "pass" for x in c.values()), c


def test_verify_flags_ghost_missing_and_drift():
    src = {"v1": tr("one two three four five".split())}
    c = by(verify_render(PLAN, src, tr("one ghost ghost two three four".split())))
    assert c["verify_extra_words"].value == 2 and c["verify_extra_words"].status == "warn"
    c = by(verify_render(PLAN, src, tr("one two".split())))
    assert c["verify_missing_words"].value == 3 and c["verify_missing_words"].status == "warn"
    c = by(verify_render(PLAN, src, tr("one two three four five".split(), off=1.0)))
    assert c["verify_av_drift"].status == "pass" and abs(c["verify_av_drift"].value - 1.0) < 1e-6   # constant offset is reported, not per-word drift


def test_verify_no_speech():
    assert verify_render(PLAN, {}, tr(["x"]))[0].status == "pass"


def test_denoise_reduces_noise_and_keeps_video(tmp_path):
    src, out = str(tmp_path / "a.mp4"), str(tmp_path / "o.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=gray:s=64x64:d=2", "-f", "lavfi", "-i", "anoisesrc=d=2:c=white:a=0.2", "-shortest", "-c:v", "libx264", "-c:a", "aac", src], check=True)
    r = denoise(src, out)
    assert r["clean_rms"] < r["src_rms"] and r["gain_db"] > 0          # noise-only input: model removes energy, gain restores level (peak-capped)
    assert "video" in subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", out], capture_output=True, text=True).stdout
    assert len(M.decode_audio(out)) > 0
