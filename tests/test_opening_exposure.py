"""A dark shot is a worse OPENING (EXP-017): the real reel opened on luma 39-65/255 and QC, which only flags near-black, let it through."""
import pytest
from aikyam_video.creative import roles
from aikyam_video.creative.shots import Population
from tests.test_creative_story import shot, e


def test_dark_shot_is_a_worse_opening_and_only_the_opening_is_affected():
    bright = shot(0, 0, 10, {"temple_architecture": 0.9, "crowd": 0.4}, e(1, 0, 0, 0), 0.5, luma=0.45)
    dark = shot(1, 20, 10, {"temple_architecture": 0.9, "crowd": 0.4}, e(0, 1, 0, 0), 0.5, luma=0.12)
    pop = Population([bright, dark]); ab, ad = roles.affinities(bright, pop, 0.1), roles.affinities(dark, pop, 0.1)
    assert ad["OPENING"] < 0.5 * ab["OPENING"]
    for r in ("BUILDUP", "RITUAL", "REVEAL", "CLIMAX", "CLOSING"):
        assert ad[r] == pytest.approx(ab[r])


def test_exposure_factor_is_smooth_and_bounded():
    f = lambda l: roles.opening_exposure(shot(0, 0, 10, {"aarti": 0.9}, e(1, 0, 0, 0), 0.5, luma=l))
    assert f(0.5) == 1.0 and f(0.30) == pytest.approx(1.0) and 0.10 <= f(0.2) <= 0.6 and f(0.02) == pytest.approx(0.10) and f(0.15) < f(0.2) < f(0.3)


def test_an_opening_window_avoids_the_dark_seconds_of_an_otherwise_fine_shot():
    """Real case: the chosen opening moment averaged bright (mid-frame 0.48) but the 3 s used were dark (~0.2). Quality alone cannot tell."""
    from aikyam_video.creative.shots import best_window
    sh = shot(0, 0, 12, {"aarti": 0.9}, e(1, 0, 0, 0), 0.5)
    sh.slots.luma = [0.10] * 12 + [0.50] * 12                                  # first half dark, second half bright; identical sharpness/steadiness everywhere
    pop = Population([sh])
    a, b, _ = best_window(sh, pop, 3.0); assert a == 0.0                       # without the brightness term it lands on the first (dark) half
    a, b, _ = best_window(sh, pop, 3.0, lit=True); assert a >= 6.0             # with it, on the bright half
