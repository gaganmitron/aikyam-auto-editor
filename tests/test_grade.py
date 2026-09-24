"""Colour match across clips (creative/grade.py)."""
import subprocess
import cv2
import numpy as np
from aikyam_video.creative import grade


def test_match_pulls_outliers_part_of_the_way_and_never_past_the_caps():
    stats = [(0.42, 0.53, 0.25), (0.47, 0.49, 0.22), (0.39, 0.51, 0.23), (0.37, 0.45, 0.09), (0.29, 0.38, 0.11), (0.45, 0.52, 0.23)]      # the real 6-clip reel
    g = grade.match(stats)
    dark = g[4]; assert dark["brightness"] > 0 and dark["saturation"] > 1 and dark["warmth"] > 0                                          # the dark, cool, dull clip is lifted
    assert all(abs(x["brightness"]) <= grade.MAX_BRIGHT + 1e-9 and abs(x["warmth"]) <= grade.MAX_WARM + 1e-9 for x in g if x)
    assert all(1 - grade.MAX_SAT - 1e-9 <= x["saturation"] <= 1 + grade.MAX_SAT + 1e-9 for x in g if x)
    med = np.median([s[0] for s in stats]); assert abs((stats[4][0] + dark["brightness"]) - med) > 0.02      # a PART of the gap: mood survives, no full normalisation


def test_clips_already_close_are_left_alone_and_unmeasurable_ones_are_skipped():
    assert grade.match([(0.4, 0.5, 0.2), (0.405, 0.505, 0.201), (0.4, 0.5, 0.2)]) == [None, None, None]
    assert grade.match([(0.3, 0.5, 0.2), None, (0.5, 0.5, 0.2)])[1] is None
    assert grade.match([(0.3, 0.5, 0.2), None, None]) == [None, None, None]                                  # nothing to match against


def test_filter_is_valid_ffmpeg_and_actually_brightens_and_warms(tmp_path):
    src = str(tmp_path / "in.png"); cv2.imwrite(src, np.full((72, 128, 3), (90, 90, 90), np.uint8))          # neutral dark grey
    g = {"brightness": 0.06, "saturation": 1.1, "warmth": 0.05}; out = str(tmp_path / "out.png")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-vf", grade.filter_expr(g), out], check=True)
    a, b = cv2.imread(src).astype(float), cv2.imread(out).astype(float)
    assert b.mean() > a.mean() + 5                                                                          # brighter
    assert b[..., 2].mean() - b[..., 0].mean() > 6                                                          # warmer: red above blue (both filters chained)
    assert grade.filter_expr({"brightness": 0.0, "saturation": 1.0, "warmth": 0.0}) == ""
