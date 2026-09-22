"""Virtual camera modes (stationary / pan / track), subject-in-frame measurement, its QC check and the replan fix."""
import numpy as np, pytest
from aikyam_video.creative import qc as QC, replan
from aikyam_video.reframe import MAX_PAN, camera_path

WIN = 0.316                                     # a 9:16 crop of 16:9 footage holds 31.6% of the width


def trk(xs, e=0.06, dt=0.5): return [(i * dt, float(x), e) for i, x in enumerate(xs)]


def test_a_steady_subject_gets_a_locked_off_camera():
    rng = np.random.RandomState(0); c = camera_path(trk(0.55 + rng.randn(16) * 0.01), WIN)
    assert c["mode"] == "stationary" and c["path"] == [] and abs(c["subjectX"] - 0.55) < 0.03 and c["inFrame"] == 1.0


def test_a_subject_moving_along_a_line_gets_one_smooth_pan():
    c = camera_path(trk(np.linspace(0.3, 0.6, 16)), WIN)
    assert c["mode"] == "pan" and len(c["path"]) == 2 and c["path"][0][1] < c["path"][1][1] and c["inFrame"] >= 0.95
    (t0, x0), (t1, x1) = c["path"]; assert abs(x1 - x0) / (t1 - t0) <= MAX_PAN + 1e-6                # never faster than the speed limit


def test_an_irregular_subject_gets_a_smooth_polynomial_track_with_far_less_wobble_than_the_raw_path():
    rng = np.random.RandomState(1); x = 0.5 + 0.18 * np.sin(np.linspace(0, 3.2, 20)) + rng.randn(20) * 0.03
    c = camera_path(trk(x), WIN); assert c["mode"] == "track" and len(c["path"]) >= 5
    xs = np.array([p for _, p in c["path"]]); assert np.abs(np.diff(xs, 2)).max() < 0.5 * np.abs(np.diff(x, 2)).max()   # second difference: acceleration
    assert all(WIN / 2 - 1e-6 <= p <= 1 - WIN / 2 + 1e-6 for p in xs)                                # the window stays inside the frame


def test_a_subject_wider_than_the_window_is_reported_as_cut_off():
    c = camera_path(trk([0.5] * 10, e=0.6), WIN); assert c["inFrame"] == 0.0                          # 60% wide subject in a 31.6% window
    c2 = camera_path(trk([0.1] * 6), WIN); assert c2["inFrame"] == 1.0 and c2["subjectX"] >= WIN / 2 - 1e-6   # subject at the edge: window clamped, still in frame


def test_whole_frame_window_and_empty_tracks_are_stationary():
    assert camera_path(trk(np.linspace(0.1, 0.9, 8)), 1.0)["mode"] == "stationary" and camera_path([], WIN)["mode"] == "stationary"


def plan(v):
    seg = lambda i, val: {"assetId": "a", "kind": "video", "start": 10 * i, "end": 10 * i + 5, "reason": "x", "score": .5, "subjectInFrame": val}
    return {"segments": [seg(i, x) for i, x in enumerate(v)]}


def tl(n): return [{"start": 5.0 * i, "end": 5.0 * i + 5} for i in range(n)]


def test_qc_flags_a_clip_that_loses_its_subject_and_replan_shows_it_whole():
    ok = QC.check_subject_in_frame(plan([1.0, 0.95]), tl(2)); assert ok[0].status == "pass"
    warn = QC.check_subject_in_frame(plan([1.0, 0.7]), tl(2)); assert warn[0].status == "warn" and warn[0].clip == 1
    bad = QC.check_subject_in_frame(plan([1.0, 0.2]), tl(2))[0]; assert bad.status == "fail" and bad.clip == 1
    full = {"schemaVersion": 1, "source": {"videoId": "v", "path": "/x", "durationSeconds": 100}, "outputFormat": "REEL", "aspectRatio": "9:16", "durationSeconds": 10,
            "segments": plan([1.0, 0.2])["segments"], "captions": {"enabled": False, "language": "en"}, "overlays": {}, "transitions": {"type": "cut", "durationSeconds": 0},
            "audio": {"preserveOriginal": True}, "thumbnail": {}}
    for s in full["segments"]: s.pop("assetId")
    new, fixes, _ = replan.apply_fixes(full, QC.QCReport([bad], {}))
    assert new["segments"][1]["layout"]["mode"] == "fit_blur" and "subject_in_frame" in fixes[0]
    assert QC.check_subject_in_frame(new, tl(2))[0].status == "pass"                                   # fit_blur clips are not judged by the crop rule
