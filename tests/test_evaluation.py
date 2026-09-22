"""Evaluation harness on a synthetic but REAL render: every metric is computed from the encoded file / plan / qc report."""
import json, subprocess
from pathlib import Path
from types import SimpleNamespace as N
import numpy as np, pytest
from aikyam_video import evaluation as EV
from aikyam_video.render import render_format
from tests.test_multi_asset import A, I, V, image, plan


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("ev"); mk = lambda n, e: subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"{e}=s=1280x720:r=25", "-f", "lavfi", "-i", "sine=f=300:d=6", "-t", "6", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(d / n)], check=True)
    mk("a.mp4", "testsrc2"); mk("b.mp4", "mandelbrot"); image(str(d / "i.png"), w=1600, h=1000)
    assets = [A("a1", "video", str(d / "a.mp4"), 6), A("a2", "video", str(d / "b.mp4"), 6), A("i1", "image", str(d / "i.png"))]
    p = plan(assets, [V("a1", 0, 3), I("i1", 2.5, zoom=[1.0, 1.2], center=[[0.5, 0.5], [0.7, 0.4]]), V("a2", 1, 4.5)], [("cut", 0), ("crossfade", 0.5)])
    p["segments"][0]["subjectInFrame"] = 0.9; p["segments"][2]["layout"] = {"mode": "fit_blur", "window": 0.6, "pushIn": 0}; p["creative"] = {"outro": {"fadeSeconds": 0.4}, "decisions": []}
    (d / "w").mkdir(); render_format(p, "u", str(d / "reel-9x16.mp4"), "reel", str(d / "w"), mix=N(pcm=(0.05 * np.random.RandomState(0).randn(int(p["durationSeconds"] * 48000), 2)).astype(np.float32)))
    json.dump(p, open(d / "edit-plan.json", "w")); json.dump({"status": "warn", "stats": {"music_margin_db_median": 14.2, "music_margin_db_p5": 11.0}, "attempts": [{"status": "fail", "failed": ["duplicate_shots"]}, {"status": "warn", "failed": []}],
                                                            "checks": [{"name": "duration", "status": "warn", "value": 8, "where": []}, {"name": "bad_crop", "status": "fail", "value": 3, "where": [1, 2]}]}, open(d / "qc.json", "w"))
    return d, p


def test_all_metrics_are_measured_and_in_range(run_dir):
    d, p = run_dir; m = EV.evaluate(str(d), grid=(np.arange(0, 20, 0.5), np.arange(0, 20, 2.0)))
    assert m["clips"] == 3 and abs(m["duration_s"] - p["durationSeconds"]) < 1e-6
    assert abs(m["shot_mean_s"] - np.mean([3, 2.5, 3.5])) < 1e-6 and m["shot_min_s"] == 2.5 and m["shot_max_s"] == 3.5
    assert m["blend_pct"] == 0.5 and m["cut_pct"] == 0.5                                                # one cut + one crossfade
    assert 0.0 <= m["hook_score"] <= 1.0 and m["first_clip_s"] == 3.0 and m["hook_short"] and not m["hook_dark"]
    assert 0.0 <= m["redundancy_max"] <= 1.0 and m["redundancy_mean"] <= m["redundancy_max"]
    assert np.isfinite(m["lufs"]) and m["true_peak_db"] < 0
    assert m["subject_in_frame"] == 0.9 and m["fit_blur_pct"] == 0.5 and m["deity_cutoffs"] is None


def test_cut_to_beat_error_is_measured_against_the_grid_and_missing_grid_is_reported_as_none(run_dir):
    d, _ = run_dir
    tight = EV.evaluate(str(d), grid=(np.array([3.0, 5.5, 8.0]), np.array([3.0]))); assert tight["cut_to_beat_mean_ms"] is not None
    off = EV.evaluate(str(d), grid=(np.array([0.0, 100.0]), np.array([0.0]))); assert off["cut_to_beat_mean_ms"] > 500 and off["cuts_on_beat_pct"] == 0.0
    assert EV.evaluate(str(d))["cut_to_beat_mean_ms"] is None                                            # no music decision in this plan


def test_qc_history_and_cutoffs_come_from_the_qc_report(run_dir):
    d, _ = run_dir; m = EV.evaluate(str(d))
    assert m["qc_status"] == "warn" and m["qc_first_attempt"] == "fail" and m["auto_reedits"] == 1 and m["face_cutoffs"] == 3
    assert m["music_margin_db_median"] == 14.2 and m["qc_warn_or_fail"] == ["duration", "bad_crop"]


def test_hook_score_separates_a_sharp_opening_from_a_dark_one(tmp_path):
    f = str(tmp_path / "v.mp4"); subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=320x568:r=25:d=3", "-f", "lavfi", "-i", "testsrc2=s=320x568:r=25:d=5", "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0", "-pix_fmt", "yuv420p", f], check=True)
    h = EV.hook_quality(f, 3.0); assert h["hook_dark"] and h["hook_score"] < 0.3                          # a black opening ranks at the bottom of the reel
    g = str(tmp_path / "w.mp4"); subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x568:r=25:d=3", "-f", "lavfi", "-i", "color=c=gray:s=320x568:r=25:d=5", "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0", "-pix_fmt", "yuv420p", g], check=True)
    hg = EV.hook_quality(g, 3.0); assert hg["hook_score"] > 0.55 and hg["hook_score"] > h["hook_score"] + 0.3 and not hg["hook_dark"]


def test_compare_table_lists_every_run():
    rows = {"a": {"duration_s": 30.0, "clips": 4, "qc_status": "pass"}, "b": {"duration_s": 20.0, "clips": 3, "qc_status": "warn", "cuts_on_beat_pct": 1.0}}
    t = EV.compare(rows); assert "a" in t.splitlines()[0] and "b" in t.splitlines()[0] and "100%" in t and "n/a" in t
