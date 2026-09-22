"""Regression on REAL temple clips (samples/*.webm, licences in samples/LICENSES.md). Skipped if a clip is missing.
Assertions are deliberately coarse: they encode the failures we actually hit (fireworks read as aarti, processions unseen)."""
import glob, os, subprocess
import cv2, pytest
from aikyam_video import ffmpeg as ff

S = os.path.join(os.path.dirname(__file__), "..", "samples")


def frames(clip, n=6):
    d = ff.probe(clip).duration; out = []
    for j in range(n):
        f = f"/tmp/_t_{os.getpid()}_{j}.jpg"; ff.frame_at(clip, d * (j + .5) / n, f); out.append(cv2.imread(f))
    return out


@pytest.fixture(scope="module")
def vp():
    pytest.importorskip("open_clip")
    from aikyam_video.vision import ClipVision
    return ClipVision()


def top_labels(vp, clip):
    if not os.path.exists(clip): pytest.skip(f"{clip} missing")
    agg = {}
    for im in frames(clip):
        for k, v in vp.analyze(im).labels.items(): agg[k] = max(agg.get(k, 0), v)
    return agg


@pytest.mark.parametrize("clip,must,must_not", [
    ("ganga-aarti-haridwar.webm", ["aarti"], ["abhishekam"]),        # a priest waving a burning multi-tier lamp at dusk (not fireworks, not abhishekam)
])
def test_vision_labels_on_real_footage(vp, clip, must, must_not):
    labs = top_labels(vp, os.path.join(S, clip))
    for k in must: assert labs.get(k, 0) >= 0.5, (k, labs)
    for k in must_not: assert labs.get(k, 0) < 0.5, (k, labs)


def test_no_real_clip_is_flagged_by_moderation(vp):
    for clip in glob.glob(os.path.join(S, "*.webm")):
        for im in frames(clip, 3):
            assert max(vp.analyze(im).moderation.values(), default=0) < 0.6, clip      # shirtless priests/dancers must not trip it


def test_deity_guess_is_rare_and_never_overrides(vp):
    """Zero-shot deity identity is weak: it may abstain, and when it answers it must be a KG id."""
    import json
    from aikyam_video.entities import SEED
    ids = {e["id"] for e in json.load(open(SEED, encoding="utf-8"))["entities"]}
    for clip in glob.glob(os.path.join(S, "*.webm")):
        for im in frames(clip, 3):
            assert set(vp.analyze(im).deities) <= ids


def test_clap_audio_tags_real_devotional_sound():
    pytest.importorskip("transformers")
    clip = os.path.join(S, "ganga-aarti-haridwar.webm")
    if not os.path.exists(clip): pytest.skip("clip missing")
    from aikyam_video.audio import ClapAudioTagger, profile
    ev = profile(clip, tagger=ClapAudioTagger()).events
    assert ev["bhajan"].mean() + ev["chant"].mean() > 0.4 and ev["bell"].mean() < 0.2 and ev["applause"].mean() < 0.3


def test_live_rolling_detection_on_real_clip(tmp_path):
    from aikyam_video.live import LiveConfig, run_live
    from aikyam_video.options import Options
    clip = os.path.join(S, "ganga-aarti-haridwar.webm")            # 28 s: chunks of 12 s with 2 s overlap
    if not os.path.exists(clip): pytest.skip("clip missing")
    got = run_live(clip, str(tmp_path / "live"), Options(formats=["reel"], engine="classic"), LiveConfig(chunk_s=12, overlap_s=2, min_score=0.2, render=True, max_chunks=3))
    assert got and len({g["chunk"] for g in got}) >= 1
    d = ff.probe(clip).duration
    for g in got:
        assert 0 <= g["start"] < g["end"] <= d + 0.5                           # STREAM-global timestamps, inside the source
        assert "reel" in g and os.path.getsize(g["reel"]) > 0
    for a in got:                                                              # rolling dedupe: no duplicate overlapping highlights
        for b in got:
            if a is not b: assert min(a["end"], b["end"]) - max(a["start"], b["start"]) < 0.3 * min(a["end"] - a["start"], b["end"] - b["start"])
    lines = open(tmp_path / "live" / "live_highlights.jsonl").read().splitlines()
    assert len(lines) == len(got)
