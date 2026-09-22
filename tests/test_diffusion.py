"""Third renderer (diffusionstudio runtime in headless Chromium + FFmpeg finishing): composition contents (pure) and a real render compared with the FFmpeg renderer on the SAME plan."""
import re, shutil, subprocess, time
from pathlib import Path
from types import SimpleNamespace as N
import numpy as np, pytest
from aikyam_video import ffmpeg as ff
from aikyam_video.render import render_format as ffmpeg_render
from aikyam_video import render_diffusion as D
from tests.test_multi_asset import A, I, V, image, plan, video

BUILT = (D.HARNESS / "dist" / "harness.js").is_file() and shutil.which("node")


def mix(seconds): return N(pcm=np.zeros((int(seconds * 48000), 2), np.float32))


def info(w, h): return N(width=w, height=h)


# ---------------------------------------------------------------- composition (no browser)
def comp(segs, edges, assets, infos, W=1080, H=1920):
    p = plan(assets, segs, edges); return D.build_composition(p, W, H, infos=infos)


def test_composition_places_clips_on_the_plan_timeline_with_overlaps_and_source_trims():
    assets = [A("a1", "video", "/x/a.mp4", 10), A("a2", "video", "/x/b.mp4", 10)]
    t = comp([V("a1", 1, 5), V("a2", 2, 6)], [("crossfade", 0.5)], assets, {"a1": info(1920, 1080), "a2": info(1920, 1080)})
    starts = re.findall(r"start=\{([\d.]+)\} end=\{([\d.]+)\} sourceIn=\{([\d.]+)\}", t)
    assert starts == [("0", "4", "1"), ("3.5", "7.5", "2")]                                      # second clip starts 0.5 s before the first ends (the overlap)
    assert 'property="opacity"' in t and t.count("<Video") == 2                                    # the dissolve is an opacity fade-in on the incoming clip


def test_landscape_crop_is_positioned_on_the_subject_and_clamped_inside_the_frame():
    assets = [A("a1", "video", "/x/a.mp4", 10)]; s = V("a1", 0, 4); s["subjectX"] = 0.9
    t = comp([s], None, assets, {"a1": info(1920, 1080)})
    Wb = 1920 * 1920 / 1080; x = float(re.search(r' x=\{(-?[\d.]+)\}', t).group(1))
    assert abs(x - (1080 - Wb)) < 0.01                                                            # subject at the right edge: the box is pushed as far left as the frame allows
    s["subjectX"] = 0.5; x = float(re.search(r' x=\{(-?[\d.]+)\}', comp([s], None, assets, {"a1": info(1920, 1080)})).group(1)); assert abs(x - (1080 / 2 - 0.5 * Wb)) < 0.01


def test_tracked_subject_becomes_keyframes_in_source_time():
    assets = [A("a1", "video", "/x/a.mp4", 20)]; s = V("a1", 5, 9); s["subjectPath"] = [[0, 0.3], [2, 0.5], [4, 0.6]]
    t = comp([s], None, assets, {"a1": info(1920, 1080)})
    assert 'property="x"' in t and [float(m) for m in re.findall(r'Keyframe time=\{([\d.]+)\}', t.split('property="x"')[1].split("</KeyframeTrack>")[0])] == [5, 7, 9]    # source-local: sourceIn + t


def test_fit_blur_uses_a_blurred_background_copy_and_a_fitted_foreground():
    assets = [A("a1", "video", "/x/a.mp4", 10)]; s = V("a1", 0, 4); s["layout"] = {"mode": "fit_blur", "window": 0.6, "pushIn": 0}
    t = comp([s], None, assets, {"a1": info(1920, 1080)})
    assert t.count("<Video") == 2 and '<Effect type="blur"' in t


def test_image_gets_ken_burns_keyframes_and_dips_get_a_colour_rect():
    assets = [A("a1", "video", "/x/a.mp4", 10), A("i1", "image", "/x/i.jpg")]
    t = comp([V("a1", 0, 4), I("i1", 3, zoom=[1.0, 1.4], center=[[0.5, 0.5], [0.7, 0.4]])], [("dip_black", 0.6)], assets, {"a1": info(1920, 1080), "i1": info(3000, 2000)})
    img = t.split("<Image")[1]
    assert img.count('property="scale"') == 1 and 'value={1.4}' in img and '<Rect x={0} y={0}' in t and 'fill="#000000"' in t


# ---------------------------------------------------------------- real render vs FFmpeg
def psnr(a, b): return 10 * np.log10(255 ** 2 / max(float(((a.astype(float) - b.astype(float)) ** 2).mean()), 1e-9))


def frame(path, t):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", path, "-frames:v", "1", "-vf", "scale=270:480", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(480, 270, 3)


@pytest.mark.skipif(not BUILT, reason="diffusion harness not built (renderer/diffusion/setup.sh + node build.mjs)")
def test_diffusion_render_matches_the_ffmpeg_render_of_the_same_plan(tmp_path):
    def mk(name, expr):
        f = str(tmp_path / name); subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"{expr}=s=1280x720:r=25", "-t", "5", "-c:v", "libx264", "-pix_fmt", "yuv420p", f], check=True); return f
    v1, v2, im = mk("a.mp4", "testsrc2"), mk("b.mp4", "mandelbrot"), image(str(tmp_path / "i.png"), w=1600, h=1000)
    assets = [A("a1", "video", v1, 5), A("a2", "video", v2, 5), A("i1", "image", im)]
    p = plan(assets, [V("a1", 0, 3), I("i1", 2.5, zoom=[1.0, 1.3], center=[[0.5, 0.5], [0.7, 0.4]]), V("a2", 1, 4)], [("cut", 0), ("crossfade", 0.5)])
    p["creative"] = {"outro": {"fadeSeconds": 0.4}}; T = p["durationSeconds"]
    Path(tmp_path / "w1").mkdir(); t0 = time.perf_counter(); a = ffmpeg_render(p, "u", str(tmp_path / "ff.mp4"), "reel", str(tmp_path / "w1"), mix=mix(T)); t_ff = time.perf_counter() - t0
    Path(tmp_path / "w2").mkdir(); t0 = time.perf_counter(); b = D.render_format(p, "u", str(tmp_path / "df.mp4"), "reel", str(tmp_path / "w2"), mix=mix(T)); t_df = time.perf_counter() - t0
    ia, ib = ff.probe(a), ff.probe(b)
    assert (ib.width, ib.height) == (1080, 1920) and abs(ib.duration - ia.duration) < 0.15
    scores = {t: psnr(frame(a, t), frame(b, t)) for t in (0.5, 2.0, 3.0, 4.0, 5.0)}
    print(f"\nffmpeg {t_ff:.1f}s, diffusion {t_df:.1f}s, PSNR(dB) vs ffmpeg: {({k: round(v, 1) for k, v in scores.items()})}")
    assert min(scores.values()) > 15, scores                                                        # the same picture in the same places (different scalers, so not identical)


@pytest.mark.skipif(not BUILT, reason="diffusion harness not built")
def test_relative_work_directories_work(tmp_path, monkeypatch):
    """The CLI passes a relative -o folder; the node harness runs from its own directory."""
    f = str(tmp_path / "a.mp4"); subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=25", "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p", f], check=True)
    p = plan([A("a1", "video", f, 2)], [V("a1", 0, 2)]); monkeypatch.chdir(tmp_path)
    out = D.render_format(p, "u", "o.mp4", "reel", "work", mix=mix(2)); assert abs(ff.probe(out).duration - 2.0) < 0.15


@pytest.mark.skipif(not BUILT, reason="diffusion harness not built")
def test_a_clip_larger_than_chromiums_blob_limit_in_tmp_still_renders(tmp_path):
    """Regression: fetch().blob() of a >14 MB file failed with the default /tmp profile, so big 4K clips rendered as a placeholder colour (QC: frozen_frames)."""
    f = str(tmp_path / "big.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=1920x1080:r=25", "-f", "lavfi", "-i", "anoisesrc=d=8", "-t", "8", "-c:v", "libx264", "-b:v", "20M", "-pix_fmt", "yuv420p", f], check=True)
    assert Path(f).stat().st_size > 16e6
    p = plan([A("a1", "video", f, 8)], [V("a1", 0, 3)]); Path(tmp_path / "w").mkdir()
    out = D.render_format(p, "u", str(tmp_path / "o.mp4"), "reel", str(tmp_path / "w"), mix=mix(3))
    a, b = frame(out, 0.2), frame(out, 2.5); assert np.abs(a.astype(int) - b.astype(int)).mean() > 2 and a.std() > 10          # real, moving picture (not the flat placeholder)
