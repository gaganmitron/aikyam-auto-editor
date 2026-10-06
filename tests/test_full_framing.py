from aikyam_video.reframe import scan_path


def test_scan_pans_across_a_wide_subject_and_follows_its_drift():
    track = [(t, 0.3 + 0.05 * t, 0.5) for t in range(5)]                 # wide subject (50% extent) drifting right
    cam = scan_path(track, 0.316, 4.0, 0.5)
    (t0, x0), (t1, x1) = cam["path"]
    assert cam["mode"] == "pan" and t0 == 0.0 and t1 == 4.0
    assert x1 > x0 + 0.1                                                   # real travel, rightwards (the drift)
    assert 0.158 <= x0 and x1 <= 1 - 0.158                                 # window centre never leaves the frame
    assert scan_path(track[::-1] if False else [(t, 0.7 - 0.05 * t, 0.5) for t in range(5)], 0.316, 4.0, 0.5)["path"][1][1] < 0.7   # leftwards drift pans left


def test_scan_without_tracking_is_left_to_right_and_honest():
    cam = scan_path([], 0.316, 3.0, 0.6)
    assert cam["path"][1][1] > cam["path"][0][1] and cam["inFrame"] == 1.0
