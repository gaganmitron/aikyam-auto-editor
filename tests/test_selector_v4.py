"""V4 selector (EXP-V4-001): quality -> diversity -> target count -> chronology.

No ML models, no story engine. Checks the four steps in isolation, that the plan is legal, that the
selector is deterministic, and that its output really renders (contract metric: render reliability).
"""
import subprocess

import numpy as np
import pytest

from aikyam_video import ffmpeg as ff
from aikyam_video.creative import selector_v4 as V4
from aikyam_video.creative.selector_v4 import TARGET_CLIPS
from aikyam_video.models import Moment, Transcript
from aikyam_video.plan import validate_plan
from aikyam_video.scoring import DEFAULT_WEIGHTS, V4_WEIGHTS


def moments(spec):
    """spec: [(start, end, score)] -> Moments, ids numbered in SOURCE order."""
    return [Moment(momentId=f"m{i}", start=a, end=b, reason=f"why{i}", score=s) for i, (a, b, s) in enumerate(spec)]


def v4(spec, **kw):
    return V4.plan_edit("v", "/x", 600, moments(spec), Transcript(language="en"), [], {}, **kw)


# ------------------------------------------------------------------ frozen EXP-V4-001 configuration
def test_v4_weights_are_the_frozen_four_signals_and_v3_is_untouched():
    assert V4_WEIGHTS == {"visualImportance": 0.35, "devotionalRelevance": 0.30,
                          "semanticImportance": 0.20, "audioImportance": 0.15,
                          "novelty": 0.0, "completeness": 0.0, "temporalImportance": 0.0,
                          "aesthetic": 0.0, "learned": 0.0}
    assert TARGET_CLIPS == 5 and V4.ORDERING == "chronological" and V4.DIVERSITY == "no-overlap"


# ------------------------------------------------------------------ step 1: quality
def test_takes_the_highest_scoring_moments_not_the_earliest():
    p = v4([(0, 20, 0.30), (100, 120, 0.90), (200, 220, 0.60), (300, 320, 0.80), (400, 420, 0.50), (500, 520, 0.95)])
    assert set(s["momentId"] for s in p["segments"]) == {"m1", "m2", "m3", "m4", "m5"}   # top 5 by score; m0 (0.30) loses
    assert [s["start"] for s in p["segments"]] == [100.0, 200.0, 300.0, 400.0, 500.0]   # ...played in source order


def test_equal_scores_resolve_the_same_way_every_time():
    """Determinism: with every score tied, the same Moment objects in a different input order must give the
    same reel -- the tie-break is (start, momentId), never dict/set or input order."""
    spec = [(10 * i, 10 * i + 12, 0.5) for i in range(9)]
    ms = moments(spec)
    tr = Transcript(language="en")
    a = V4.plan_edit("v", "/x", 600, ms, tr, [], {})
    b = V4.plan_edit("v", "/x", 600, list(reversed(ms)), tr, [], {})
    assert [s["momentId"] for s in a["segments"]] == [s["momentId"] for s in b["segments"]] == ["m0", "m2", "m4", "m6", "m8"]
    assert a == b


# ------------------------------------------------------------------ step 2: diversity
def test_overlapping_candidates_are_skipped_and_the_slot_is_backfilled():
    """The second candidate overlaps the first, so its slot goes to the NEXT candidate, not to a hole."""
    p = v4([(0, 60, 0.90), (5, 70, 0.85), (30, 90, 0.80), (60, 120, 0.70), (150, 180, 0.60)])
    assert [s["momentId"] for s in p["segments"]] == ["m0", "m2", "m3", "m4"]
    rej = {d["momentId"]: d["why"] for d in p["creative"]["decisions"] if d["type"] == "reject"}
    assert "overlaps" in rej["m1"] and len(p["segments"]) == 4
    for i, x in enumerate(p["segments"]):                                    # nothing is ever used twice
        for y in p["segments"][i + 1:]:
            assert x["end"] <= y["start"] or y["end"] <= x["start"]


def test_clips_shorter_than_two_seconds_are_dropped():
    p = v4([(0, 1.5, 0.9), (10, 30, 0.8), (40, 60, 0.7)])
    assert [s["momentId"] for s in p["segments"]] == ["m1", "m2"]
    assert any(d["type"] == "reject" and "shorter" in d["why"] for d in p["creative"]["decisions"])


# ------------------------------------------------------------------ step 3: target count
def test_never_more_than_the_target_number_of_clips():
    assert len(v4([(10 * i, 10 * i + 12, 1.0 - i * 0.01) for i in range(30)])["segments"]) == TARGET_CLIPS
    assert len(v4([(10 * i, 10 * i + 12, 1.0 - i * 0.01) for i in range(30)], target_clips=2)["segments"]) == 2
    assert len(v4([(0, 20, 0.9), (100, 120, 0.8)])["segments"]) == 2         # never pads a short reel


# ------------------------------------------------------------------ step 4: chronology, and the plan contract
def test_segments_are_in_source_order_and_the_plan_validates():
    p = v4([(0, 20, 0.5), (300, 320, 0.9), (100, 120, 0.7), (500, 520, 0.95), (200, 220, 0.8), (400, 420, 0.6)])
    starts = [s["start"] for s in p["segments"]]
    assert starts == sorted(starts) and len(p["segments"]) == TARGET_CLIPS
    validate_plan(p, 600)
    assert p["creative"]["engine"] == "v4" and p["creative"]["order"] == "source"
    assert p["creative"]["selector"] == {"targetClips": 5, "ordering": "chronological", "diversity": "no-overlap",
                                         "maxSegSeconds": 12.0, "transition": "crossfade", "transitionSeconds": 0.5}


def test_the_reel_never_exceeds_the_plan_s_length_cap():
    """MAX_REEL_S is a hard contract invariant, not a preference: 5 clips of a full 15 s (75 s) is an illegal
    plan. The clip cap is the budget per target clip, so TARGET_CLIPS and the cap hold together."""
    from aikyam_video.plan import MAX_REEL_S
    long_enough = [(10 * i, 10 * i + 900, 1.0 - i * 0.01) for i in range(30)]          # every moment 15 minutes long
    p = v4(long_enough)
    assert len(p["segments"]) == TARGET_CLIPS
    assert sum(s["end"] - s["start"] for s in p["segments"]) <= MAX_REEL_S
    validate_plan(p, 4000)
    assert all(s["end"] - s["start"] <= MAX_REEL_S / TARGET_CLIPS + 1e-6 for s in p["segments"])


def test_a_source_with_too_few_candidates_says_so_in_the_trace():
    p = v4([(0, 20, 0.5), (100, 120, 0.8)])
    assert len(p["segments"]) == 2
    short = [d for d in p["creative"]["decisions"] if d["type"] == "short"]
    assert short and short[0]["clips"] == 2 and short[0]["target"] == TARGET_CLIPS   # never padded with junk
    validate_plan(p, 600)


def test_a_clip_is_trimmed_to_the_cap_and_renders_to_that_length(tmp_path):
    """Render reliability + the trim rule, checked against a real encoded file (FFmpeg, no models)."""
    src = str(tmp_path / "s.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=25:d=120",
                    "-f", "lavfi", "-i", "sine=f=440:d=120", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", src], check=True)
    spec = [(0, 60, 0.5), (30, 90, 0.9), (60, 120, 0.7)]
    p = v4(spec, aspect="9:16", fmt="REEL", target_clips=2)
    assert all(s["end"] - s["start"] <= 15.0 for s in p["segments"])            # trimmed
    p["audio"]["preserveOriginal"] = True
    out = str(tmp_path / "r.mp4")
    from aikyam_video.render import render_format
    render_format(p, src, out, "reel", str(tmp_path))
    got = ff.probe(out).duration
    n = len(p["segments"]); d = p["transitions"]["durationSeconds"]
    assert abs(got - (sum(s["end"] - s["start"] for s in p["segments"]) - (n - 1) * d)) < 0.3
    assert 1.0 < got < 40.0


# ------------------------------------------------------------------ boundary against V3
def test_v4_differs_from_the_classic_planner_but_reuses_the_same_moments():
    """V4 is independent of creative/story.py: same Moments in, a different (count-ordered) plan out.
    And it never reads the V3-only scoring signals, which stay on the Moment untouched."""
    from aikyam_video.plan import plan_edit
    ms = moments([(0, 20, 0.30), (100, 120, 0.90), (200, 220, 0.60), (300, 320, 0.80), (400, 420, 0.50), (500, 520, 0.95)])
    ms[1].signals = {"novelty": 0.99, "completeness": 0.99, "temporalImportance": 0.99}
    tr = Transcript(language="en")
    v3 = plan_edit("v", "/x", 600, ms, tr, [], {}, target_s=45)
    v4p = V4.plan_edit("v", "/x", 600, ms, tr, [], {})
    assert len(v4p["segments"]) == TARGET_CLIPS and len(v3["segments"]) != TARGET_CLIPS
    assert v4p["segments"] != v3["segments"]
    assert [s["momentId"] for s in v4p["segments"]] == ["m1", "m2", "m3", "m4", "m5"]   # score only, then source order
    assert ms[1].signals["novelty"] == 0.99                                            # V3 signals preserved, not consumed


def test_weights_come_from_config_not_from_the_engine_name():
    from aikyam_video.scoring import ScoringConfig
    assert ScoringConfig.load(engine="v4").weights == V4_WEIGHTS
    assert ScoringConfig.load().weights == DEFAULT_WEIGHTS                          # the V3 default is unchanged
    up = ScoringConfig.load(engine="v4"); up.weights = {**up.weights, "audioImportance": 0.5}
    assert up.weights["audioImportance"] == 0.5 and DEFAULT_WEIGHTS["audioImportance"] == 0.15


def test_unknown_scorer_in_a_weight_file_is_still_an_error():
    from aikyam_video.scoring import ScoringConfig
    with pytest.raises(ValueError, match="unregistered scorers"):
        ScoringConfig.load(engine="v4", path=_weights_file())


def _weights_file():
    import json, tempfile, os
    fd, p = tempfile.mkstemp(suffix=".json"); os.write(fd, json.dumps({"not_a_scorer": 1.0}).encode()); os.close(fd)
    return p

# ------------------------------------------------------------------ graphic guard on the dense track (a "Leave a like!" animation inside a long scene)
def test_dense_text_track_rejects_a_window_over_a_dip_under_the_videos_own_median(monkeypatch):
    from aikyam_video import highlights as H
    from aikyam_video.audio import AudioProfile
    from types import SimpleNamespace as N
    ctx = H.Ctx("x", N(duration=400.0, has_audio=False), [], [], Transcript(language="en"), AudioProfile.__new__(AudioProfile), allow_silent=True,
                text_frames=[[float(t), 14.0] for t in range(1, 400, 2)])
    ctx.text_frames[179] = [359.0, 6.4]                                    # graphic at ~358-360 s: far above the fixed cut-off 3.0, far below this video's median 14
    import aikyam_video.vision as V
    monkeypatch.setattr(V, "model_name", lambda: "hf-hub:timm/ViT-B-16-SigLIP")
    assert H.validate_clip(ctx, 350, 362, []) == "graphic_or_text_overlay"
    assert H.validate_clip(ctx, 300, 312, []) is None
