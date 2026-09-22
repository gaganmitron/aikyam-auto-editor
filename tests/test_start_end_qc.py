import subprocess
from aikyam_video.creative.qc import check_start_end


def mk(path, vf):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=160x90:r=25:d=8", "-vf", vf, "-pix_fmt", "yuv420p", path], check=True)


def by(cs): return {c.name: c for c in cs}


def plan(): return {"durationSeconds": 8.0, "creative": {"outro": {"fadeSeconds": 0.8}}}


def test_settled_ending_and_clean_opening_pass(tmp_path):
    f = str(tmp_path / "a.mp4"); mk(f, "trim=duration=5.5,tpad=stop_mode=clone:stop_duration=2.5")   # motion, then the last frame is held
    c = by(check_start_end(plan(), f, [{}, {}]))
    assert c["ending_resolved"].status == "pass" and c["ending_resolved"].value < 0.5 and c["opening_strong"].status == "pass"


def test_moving_end_and_black_start_warn(tmp_path):
    f = str(tmp_path / "b.mp4"); mk(f, "fade=t=in:d=0.6")
    c = by(check_start_end(plan(), f, [{}, {}]))
    assert c["opening_strong"].status == "warn"
    # a still start with a busy finish: static 5 s then motion
    g = str(tmp_path / "c.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=160x90:r=25:d=5", "-f", "lavfi", "-i", "testsrc2=s=160x90:r=25:d=3", "-filter_complex", "[0][1]concat=n=2:v=1:a=0", "-pix_fmt", "yuv420p", g], check=True)
    assert by(check_start_end(plan(), g, [{}, {}]))["ending_resolved"].status == "warn"
