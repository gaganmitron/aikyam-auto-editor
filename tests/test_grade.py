"""Colour match across clips (creative/grade.py)."""
import subprocess
import cv2
import numpy as np
from aikyam_video.creative import grade


def test_match_pulls_outliers_part_of_the_way_and_never_past_the_caps():
    stats = [(0.42, 0.53, 0.25, 0.02), (0.47, 0.49, 0.22, 0.02), (0.39, 0.51, 0.23, 0.01), (0.37, 0.45, 0.09, 0.03), (0.29, 0.38, 0.11, 0.02), (0.45, 0.52, 0.23, 0.02)]      # the real 6-clip reel
    g = grade.match(stats)
    dark = g[4]; assert dark["brightness"] > 0 and dark["saturation"] > 1 and dark["warmth"] > 0                                          # the dark, cool, dull clip is lifted
    assert all(abs(x["brightness"]) <= grade.MAX_BRIGHT + 1e-9 and abs(x["warmth"]) <= grade.MAX_WARM + 1e-9 for x in g if x)
    assert all(1 - grade.MAX_SAT - 1e-9 <= x["saturation"] <= 1 + grade.MAX_SAT + 1e-9 for x in g if x)
    med = np.median([s[0] for s in stats]); assert abs((stats[4][0] + dark["brightness"]) - med) > 0.02      # a PART of the gap: mood survives, no full normalisation


def test_clips_already_close_are_left_alone_and_unmeasurable_ones_are_skipped():
    assert grade.match([(0.4, 0.5, 0.2, 0.02), (0.405, 0.505, 0.201, 0.02), (0.4, 0.5, 0.2, 0.02)]) == [None, None, None]
    assert grade.match([(0.3, 0.5, 0.2, 0.02), None, (0.5, 0.5, 0.2, 0.02)])[1] is None
    assert grade.match([(0.3, 0.5, 0.2, 0.02), None, None]) == [None, None, None]                                  # nothing to match against


def test_filter_is_valid_ffmpeg_and_actually_brightens_and_warms(tmp_path):
    src = str(tmp_path / "in.png"); cv2.imwrite(src, np.full((72, 128, 3), (90, 90, 90), np.uint8))          # neutral dark grey
    g = {"brightness": 0.06, "saturation": 1.1, "warmth": 0.05}; out = str(tmp_path / "out.png")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-vf", grade.filter_expr(g), out], check=True)
    a, b = cv2.imread(src).astype(float), cv2.imread(out).astype(float)
    assert b.mean() > a.mean() + 5                                                                          # brighter
    assert b[..., 2].mean() - b[..., 0].mean() > 6                                                          # warmer: red above blue (both filters chained)
    assert grade.filter_expr({"brightness": 0.0, "saturation": 1.0, "warmth": 0.0}) == ""


def test_a_green_cast_clip_gets_its_green_pulled_down_within_the_cap():
    stats = [(0.4, 0.5, 0.2, 0.01), (0.4, 0.5, 0.2, 0.01), (0.4, 0.5, 0.2, 0.01), (0.4, 0.5, 0.2, 0.11)]          # the last clip is yellow-green (tint 0.11 vs 0.01)
    g = grade.match(stats)[3]
    assert g and g["tint"] < 0 and abs(g["tint"]) <= grade.MAX_TINT + 1e-9
    assert all(x is None for x in grade.match(stats)[:3])


def test_tint_filter_lowers_green_relative_to_red_and_blue(tmp_path):
    src = str(tmp_path / "in.png"); cv2.imwrite(src, np.full((72, 128, 3), (90, 140, 120), np.uint8))               # BGR: green-heavy
    out = str(tmp_path / "out.png"); subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-vf", grade.filter_expr({"brightness": 0.0, "saturation": 1.0, "warmth": 0.0, "tint": -0.06}), out], check=True)
    a, b = cv2.imread(src).astype(float), cv2.imread(out).astype(float)
    assert b[..., 1].mean() < a[..., 1].mean() - 4 and abs(b[..., 0].mean() - a[..., 0].mean()) < 1.5              # green down, blue untouched
