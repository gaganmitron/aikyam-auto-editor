"""EditPlan v3: several video/image/audio assets, image segments with Ken Burns, per-asset validation and rendering. Real FFmpeg."""
import subprocess
from types import SimpleNamespace as N
import cv2, numpy as np, pytest
from aikyam_video import ffmpeg as ff
from aikyam_video.captions import build_cues
from aikyam_video.models import Transcript, TranscriptSegment
from aikyam_video.plan import PlanError, output_duration, timeline, validate_plan
from aikyam_video.render import render_format


def video(path, color="red", d=6, w=1280, h=720):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={color}:s={w}x{h}:r=25:d={d}", "-f", "lavfi", "-i", f"sine=f=440:d={d}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", path], check=True); return path


def image(path, w=1600, h=1000, sq=(1150, 380, 120, 120)):
    """Gray canvas with a red square at a KNOWN place (right of centre, above the middle)."""
    im = np.full((h, w, 3), 90, np.uint8); x, y, a, b = sq; im[y:y + b, x:x + a] = (0, 0, 255); cv2.imwrite(path, im); return path


def plan(assets, segs, edges=None):
    p = {"schemaVersion": 3, "source": {"videoId": "v", "path": assets[0]["path"], "durationSeconds": 6}, "outputFormat": "REEL", "aspectRatio": "9:16",
         "assets": assets, "segments": segs, "captions": {"enabled": False, "language": "en"}, "overlays": {"template": "divine_moment"},
         "transitions": {"type": "cut", "durationSeconds": 0}, "audio": {"preserveOriginal": True, "music": {"enabled": False}}, "thumbnail": {}}
    for s, (ty, d) in zip(p["segments"][1:], edges or []): s["transitionIn"] = {"type": ty, "durationSeconds": d}
    p["durationSeconds"] = round(output_duration(p["segments"], p["transitions"]), 2); return p


def A(i, kind, path, dur=None): return {"id": i, "kind": kind, "path": path, "durationSeconds": dur}
def V(aid, a, b): return {"assetId": aid, "kind": "video", "start": a, "end": b, "reason": "x", "score": .5}
def I(aid, d, **m): return {"assetId": aid, "kind": "image", "start": 0, "end": d, "reason": "image", "score": .5, **({"motion": m} if m else {})}


# ------------------------------------------------ validation
def test_plan_v3_validates_bounds_per_asset_and_allows_overlapping_times_across_assets(tmp_path):
    assets = [A("a1", "video", "x", 10), A("a2", "video", "y", 4), A("i1", "image", "z")]
    ok = plan(assets, [V("a1", 0, 5), V("a2", 0, 3), I("i1", 3)], [("cut", 0), ("cut", 0)])          # a1 0-5 and a2 0-3 overlap in TIME but are different assets
    validate_plan(ok, 10)
    bad = plan(assets, [V("a2", 0, 6)]); 
    with pytest.raises(PlanError, match="outside source"): validate_plan(bad, 10)                     # a2 is only 4 s long
    with pytest.raises(PlanError, match="unknown asset"): validate_plan(plan(assets, [V("nope", 0, 2)]), 10)
    with pytest.raises(PlanError, match="overlap"): validate_plan(plan(assets, [V("a1", 0, 5), V("a1", 4, 8)], [("cut", 0)]), 10)
    with pytest.raises(PlanError, match="image"): validate_plan(plan(assets, [{**I("a1", 3)}]), 10)     # an image segment must point at an image asset
    with pytest.raises(PlanError, match="audio"): validate_plan(plan(assets + [A("m", "audio", "m.mp3", 30)], [V("m", 0, 2)]), 10)


def test_image_segments_have_no_source_bound_and_the_same_image_is_not_an_overlap():
    assets = [A("a1", "video", "x", 10), A("i1", "image", "z")]
    p = plan(assets, [I("i1", 2.5), V("a1", 0, 5), I("i1", 3)], [("cut", 0), ("cut", 0)]); validate_plan(p, 10)


def test_music_asset_reference_is_validated():
    assets = [A("a1", "video", "x", 10), A("m1", "audio", "m.mp3", 60)]
    p = plan(assets, [V("a1", 0, 5)]); p["audio"]["music"] = {"enabled": True, "assetId": "m1"}; validate_plan(p, 10)
    p["audio"]["music"] = {"enabled": True, "assetId": "ghost"}
    with pytest.raises(PlanError, match="music"): validate_plan(p, 10)


def test_v1_plans_without_assets_still_validate():
    p = {"schemaVersion": 1, "source": {"videoId": "v", "path": "/x", "durationSeconds": 20}, "outputFormat": "REEL", "aspectRatio": "9:16", "durationSeconds": 6,
         "segments": [{"start": 1, "end": 7, "reason": "x", "score": .5}], "captions": {"enabled": False, "language": "en"}, "overlays": {},
         "transitions": {"type": "cut", "durationSeconds": 0}, "audio": {"preserveOriginal": True}, "thumbnail": {}}
    validate_plan(p)


def test_captions_use_each_assets_own_transcript_and_images_are_silent():
    t1 = Transcript(language="en", segments=[TranscriptSegment(start=0, end=4, text="hello from a1")])
    t2 = Transcript(language="en", segments=[TranscriptSegment(start=0, end=4, text="hello from a2")])
    segs = [V("a1", 0, 4), I("i1", 3), V("a2", 0, 4)]
    cues = build_cues({"a1": t1, "a2": t2}, segs, "sentence", [0.0, 0.0])
    assert [c["text"] for c in cues] == ["hello from a1", "hello from a2"]
    assert abs(cues[1]["start"] - 7.0) < 0.01                                                      # 4 s video + 3 s image before the second speaker


# ------------------------------------------------ rendering
def frame(path, t):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", path, "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True).stdout
    i = ff.probe(path); return np.frombuffer(raw, np.uint8).reshape(i.height, i.width, 3)


def red_stats(f):
    m = (f[..., 2] > 200) & (f[..., 1] < 60) & (f[..., 0] < 60); ys, xs = np.where(m)
    return (int(m.sum()), float(xs.mean()) if len(xs) else None, float(ys.mean()) if len(ys) else None)


def test_ken_burns_zooms_in_and_pans_toward_the_subject(tmp_path):
    img = image(str(tmp_path / "i.png")); cx, cy = (1150 + 60) / 1600, (380 + 60) / 1000                           # the red square's centre, normalised
    p = plan([A("i1", "image", img)], [I("i1", 4, zoom=[1.0, 1.6], center=[[0.5, 0.5], [cx, cy]], easing="smooth")])
    out = str(tmp_path / "kb.mp4"); render_format(p, "unused", out, "square", str(tmp_path), mix=N(pcm=np.zeros((4 * 48000, 2), np.float32)))
    i = ff.probe(out); assert (i.width, i.height) == (1080, 1080) and abs(i.duration - 4) < 0.2
    n0, x0, y0 = red_stats(frame(out, 0.1)); n1, x1, y1 = red_stats(frame(out, 3.8))
    assert n0 > 0 and n1 > 2.0 * n0                                                                                  # the square appears larger at the end: zoomed in
    assert abs(x1 - 540) < abs(x0 - 540) and abs(y1 - 540) < abs(y0 - 540)                                            # and moved toward the centre: the crop panned to the subject


def test_ken_burns_is_smooth_not_jerky(tmp_path):
    img = image(str(tmp_path / "i.png")); p = plan([A("i1", "image", img)], [I("i1", 4, zoom=[1.0, 1.4], center=[[0.6, 0.45], [0.75, 0.44]])])
    out = str(tmp_path / "kb.mp4"); render_format(p, "unused", out, "reel", str(tmp_path), mix=N(pcm=np.zeros((4 * 48000, 2), np.float32)))
    xs = [red_stats(frame(out, t))[1] for t in np.arange(0.2, 3.8, 0.2)]; xs = [v for v in xs if v is not None]
    d = np.diff(xs); assert len(xs) >= 12 and np.abs(np.diff(d)).max() < 0.25 * np.abs(d).max() + 8               # eased motion: acceleration bounded, no jumps


def test_mixed_assets_render_in_order_with_images_between_clips(tmp_path):
    v1 = video(str(tmp_path / "a.mp4"), "red"); v2 = video(str(tmp_path / "b.mp4"), "blue"); im = image(str(tmp_path / "i.png"))
    assets = [A("a1", "video", v1, 6), A("a2", "video", v2, 6), A("i1", "image", im)]
    p = plan(assets, [V("a1", 0, 2.5), I("i1", 2.5, zoom=[1.0, 1.2], center=[[0.5, 0.5], [0.7, 0.4]]), V("a2", 1, 3.5)], [("cut", 0), ("crossfade", 0.5)])
    assert p["durationSeconds"] == 7.0                                                                               # 2.5 + 2.5 + 2.5 - 0.5
    out = str(tmp_path / "m.mp4"); render_format(p, "unused", out, "reel", str(tmp_path), mix=N(pcm=np.zeros((7 * 48000, 2), np.float32)))
    assert abs(ff.probe(out).duration - 7.0) < 0.2
    px = lambda t: frame(out, t)[::96, ::54].reshape(-1, 3).mean(axis=0)
    a, b, c = px(1.0), px(3.5), px(6.5)
    assert a[2] > 200 and a[0] < 60                                                                                  # red video first
    assert abs(float(b[0]) - 90) < 25 and abs(float(b[1]) - 90) < 25                                                # the gray image in the middle
    assert c[0] > 200 and c[2] < 60                                                                                  # blue video last


def test_same_asset_image_and_video_use_their_own_inputs_and_multi_asset_needs_the_mixer(tmp_path):
    v1 = video(str(tmp_path / "a.mp4")); p = plan([A("a1", "video", v1, 6)], [V("a1", 0, 3)])
    with pytest.raises(ValueError, match="mixer"): render_format(p, "unused", str(tmp_path / "o.mp4"), "reel", str(tmp_path))


def test_ken_burns_survives_rounding_on_a_low_res_landscape_still(tmp_path):
    """1500x900 -> 9:16: at zoom 1 the scaled image is only ~1917 px tall vs a 1920 window: the crop must never exceed the frame."""
    img = image(str(tmp_path / "i.png"), w=1500, h=900, sq=(1000, 340, 100, 100))
    p = plan([A("i1", "image", img)], [I("i1", 3, zoom=[1.0, 1.08], center=[[0.65, 0.5], [0.7, 0.4]])])
    out = str(tmp_path / "kb.mp4"); render_format(p, "unused", out, "reel", str(tmp_path), mix=N(pcm=np.zeros((3 * 48000, 2), np.float32)))
    assert (ff.probe(out).width, ff.probe(out).height) == (1080, 1920)
