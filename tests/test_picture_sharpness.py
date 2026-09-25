"""weak_shots must measure the PICTURE, not the blurred fill of a fitted clip (17 real reels: 8 of 10 warnings vanished once it did), and still catch a truly soft clip."""
import subprocess
from types import SimpleNamespace
from aikyam_video.creative.qc import _picture_sharpness
from tests.test_creative_shots import make, SHARP, BLUR


def _out_video(path, band_sharp: bool):
    """360x640 'reel' with a detailed picture in the middle 53% and heavy blur above/below, like a fit_blur render."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=25:d=2", "-filter_complex",
                    f"[0:v]{'null' if band_sharp else 'gblur=sigma=9'},scale=360:-2[f];[0:v]scale=360:640:force_original_aspect_ratio=increase,crop=360:640,gblur=sigma=25[b];[b][f]overlay=(W-w)/2:(H-h)/2,format=yuv420p[v]",
                    "-map", "[v]", "-c:v", "libx264", path], check=True)


def test_sharp_and_soft_pictures_are_told_apart_on_a_fitted_layout(tmp_path):
    src = str(tmp_path / "src.mp4"); make(src, [(2, SHARP)])
    sharp, soft = str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4"); _out_video(sharp, True); _out_video(soft, False)
    info = SimpleNamespace(width=360, height=640); seg = {"subjectSpread": 0.5, "kind": "video"}; clip = {"start": 0.0, "end": 2.0}
    a = _picture_sharpness(sharp, clip, seg, info, {None: src}, {}); b = _picture_sharpness(soft, clip, seg, info, {None: src}, {})
    assert a > 4 * b                                                                                  # a truly soft picture is far below 25% of a sharp one


def test_blurred_fill_does_not_make_a_sharp_picture_look_weak(tmp_path):
    src = str(tmp_path / "src.mp4"); make(src, [(2, SHARP)]); out = str(tmp_path / "o.mp4"); _out_video(out, True)
    info = SimpleNamespace(width=360, height=640); clip = {"start": 0.0, "end": 2.0}
    on_picture = _picture_sharpness(out, clip, {"subjectSpread": 0.5}, info, {None: src}, {})
    whole = _picture_sharpness(out, clip, {"subjectSpread": 0.5}, info, {}, {})                        # no source known: whole frame, fill included
    assert on_picture > 1.5 * whole
