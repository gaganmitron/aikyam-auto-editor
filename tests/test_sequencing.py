"""Transition costs for the story beam (EXP-026 / E4): off = identical to before; each term has its stated direction; switching one on reduces exactly what it penalises."""
import numpy as np
from aikyam_video.creative import pacing, sequencing, story
from aikyam_video.creative.shots import Population
from tests.test_creative_story import shot, e, temple_pool

WIDE = {"wide": 1.0, "medium": 0.0, "close": 0.0, "detail": 0.0}
MED = {"wide": 0.0, "medium": 1.0, "close": 0.0, "detail": 0.0}
SIM = lambda a, b: float(np.dot(a.embedding, b.embedding))


def _plan(pool, w=None):
    return story.plan_story(pool, Population(pool), pacing.profile("devotional"), 400.0, 45.0, transition=w)


def _seq(tl): return [(c.role, c.shot.id) for c in tl.clips]


def test_no_weights_is_byte_identical_to_the_planner_without_the_feature():
    base = _seq(_plan(temple_pool()))
    assert _seq(_plan(temple_pool(), {})) == base
    assert _seq(_plan(temple_pool(), {k: 0.0 for k in sequencing.TERMS})) == base
    assert not any(d["type"] == "transition_cost" for d in _plan(temple_pool()).decisions)


def test_size_repeat_wide_wide_is_worst_two_tighter_is_half_and_unknown_is_free():
    a, b = shot(0, 0, 10, {"aarti": 0.9}, e(1, 0, 0, 0)), shot(1, 20, 10, {"aarti": 0.9}, e(0, 1, 0, 0))
    f = lambda sa, sb: (setattr(a, "shot_size", sa), setattr(b, "shot_size", sb), sequencing.terms(a, b, "RITUAL", {}, SIM)["size_repeat"])[-1]
    assert f(WIDE, WIDE) == 1.0 and f(MED, MED) == 0.5 and f(WIDE, MED) == 0.0 and f({}, WIDE) == 0.0


def test_energy_jump_is_free_when_small_and_exempt_into_a_climax():
    a, b = shot(0, 0, 10, {"aarti": 0.9}, e(1, 0, 0, 0)), shot(1, 20, 10, {"aarti": 0.9}, e(0, 1, 0, 0))
    t = lambda role, ea, eb: sequencing.terms(a, b, role, {a.id: ea, b.id: eb}, SIM)["energy_jump"]
    assert t("RITUAL", 0.5, 0.7) == 0.0 and t("RITUAL", 0.05, 0.95) > 0.5 and t("CLIMAX", 0.05, 0.95) == 0.0


def test_motion_unrelated_needs_different_videos_high_energy_and_unrelated_pictures():
    a, b = shot(0, 0, 10, {"aarti": 0.9}, e(1, 0, 0, 0)), shot(1, 20, 10, {"aarti": 0.9}, e(0, 1, 0, 0))
    en = {a.id: 0.9, b.id: 0.9}; m = lambda: sequencing.terms(a, b, "RITUAL", en, SIM)["motion_unrelated"]
    a.asset_id, b.asset_id = "v1", "v1"; assert m() == 0.0                                  # same video: never
    b.asset_id = "v2"; assert m() >= 0.9 - 1e-9                                                   # different videos, both energetic, orthogonal pictures
    en[b.id] = 0.1; assert m() < 0.15                                                        # one of them calm
    en[b.id] = 0.9; b.embedding = list(a.embedding); assert m() == 0.0                       # related pictures


def test_luma_jump_is_free_within_the_normal_range_and_grows_beyond_it():
    a, b = shot(0, 0, 10, {"aarti": 0.9}, e(1, 0, 0, 0), luma=0.30), shot(1, 20, 10, {"aarti": 0.9}, e(0, 1, 0, 0), luma=0.40)
    assert sequencing.terms(a, b, "RITUAL", {}, SIM)["luma_jump"] == 0.0
    b.slots.luma = [0.85] * len(b.slots.luma); assert sequencing.terms(a, b, "RITUAL", {}, SIM)["luma_jump"] > 0.9


def test_turning_size_repeat_on_reduces_repeats_and_is_recorded_in_the_trace():
    def repeats(tl): return sum(sequencing.terms(x.shot, y.shot, y.role, {}, SIM)["size_repeat"] for x, y in zip(tl.clips, tl.clips[1:]))
    pool = temple_pool()
    for s, k in zip(pool, "wwwwmm"): s.shot_size = dict(WIDE if k == "w" else MED)
    before, after = _plan(pool), _plan(pool, {"size_repeat": 3.0})
    assert _seq(before) != _seq(after) and repeats(after) < repeats(before)
    tc = next(d for d in after.decisions if d["type"] == "transition_cost")
    assert tc["weights"] == {"size_repeat": 3.0} and len(tc["pairs"]) == len(after.clips) - 1


def test_same_source_repeat_needs_a_multi_video_pool_and_two_clips_of_one_video():
    a, b = shot(0, 0, 10, {"aarti": 0.9}, e(1, 0, 0, 0)), shot(1, 20, 10, {"aarti": 0.9}, e(0, 1, 0, 0))
    a.asset_id, b.asset_id = "v1", "v1"; f = lambda multi=True: sequencing.terms(a, b, "RITUAL", {}, SIM, multi)["same_source_repeat"]
    assert f() == 1.0 and f(multi=False) == 0.0                                            # one-video reels: every cut would carry the same constant, so it is off
    b.asset_id = "v2"; assert f() == 0.0                                                     # different videos: no penalty


def test_spread_sources_leaves_a_one_video_reel_untouched_and_reduces_repeats_across_videos():
    base = _seq(_plan(temple_pool()))
    assert _seq(_plan(temple_pool(), {"same_source_repeat": 3.0})) == base                  # single-asset pool (all "a1"): identical plan, including the reel length
    def repeats(tl): return sum(1 for x, y in zip(tl.clips, tl.clips[1:]) if x.shot.asset_id == y.shot.asset_id)
    pool = temple_pool()
    for s, k in zip(pool, "aaaabb"): s.asset_id = "v1" if k == "a" else "v2"
    before, after = _plan(pool), _plan(pool, {"same_source_repeat": 3.0})
    assert repeats(after) < repeats(before) and len(after.clips) >= len(before.clips) - 1
