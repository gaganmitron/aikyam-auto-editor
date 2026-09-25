"""A blocked lens / whip inside one continuous take must never be inside a clip (real Tirumala closing: 1.5 s of smudge at 306.8 s in a 9.4 s scene)."""
from aikyam_video.creative.shots import Population, best_window
from tests.test_creative_story import shot, e


def _closing(sharp):
    sh = shot(0, 100, 9.4, {"temple_tank": 0.9}, e(1, 0, 0, 0), 0.5)
    sh.slots.sharpness = sharp
    return sh


def test_window_shrinks_to_the_clean_stretch_instead_of_including_the_dead_slots():
    sh = _closing([3000.0] * 7 + [30.0, 100.0, 475.0] + [1700.0] * 8)              # 18 slots: 3.5 s clean, 1.5 s dead, 4.5 s clean
    a, b, _ = best_window(sh, Population([sh]), 8.0)
    assert a >= 100 + 5.0 - 1e-6 and b <= sh.end                                   # entirely after the dead run
    assert b - a >= 3.0


def test_short_clean_stretches_do_not_shrink_the_clip_and_evenly_sharp_shots_are_untouched():
    dead_heavy = _closing([1700.0, 30.0] * 9)                          # no clean stretch reaches 3 s -> old behaviour
    assert best_window(dead_heavy, Population([dead_heavy]), 8.0)[1] - best_window(dead_heavy, Population([dead_heavy]), 8.0)[0] >= 4.0
    even = _closing([1700.0] * 18)
    a, b, _ = best_window(even, Population([even]), 8.0); assert b - a == 8.0


def test_edge_snapping_may_not_pull_a_clip_back_onto_dead_footage():
    from aikyam_video.creative import story, pacing
    from tests.test_creative_story import temple_pool
    pool = temple_pool(); last = pool[-1]
    n = len(last.slots.sharpness); last.slots.sharpness = [3000.0] * 5 + [30.0] * 3 + [1700.0] * (n - 8)
    pop = Population(pool); dead = (last.start + 2.5, last.start + 4.0)
    edger = lambda s, a, b, lo, hi, first=False, last=False: ((s.start, b, {"start": (0.5, "cut")}) if s is pool[-1] else (a, b, {}))   # a "cut" at the shot start, i.e. before the dead run
    tl = story.plan_story(pool, pop, pacing.profile("devotional"), 400.0, 45.0, edger=edger)
    for c in tl.clips:
        if c.shot is last:
            assert not (c.start < dead[1] and c.end > dead[0]), (c.start, c.end)
