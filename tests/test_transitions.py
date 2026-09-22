"""Crossfade between moments: the plan accounts for the overlap, captions follow it, and BOTH renderers really blend picture and sound."""
import subprocess
import numpy as np, pytest
from aikyam_video import ffmpeg as ff
from aikyam_video.models import Moment, Transcript, TranscriptSegment
from aikyam_video.plan import PlanError, output_duration, plan_edit, validate_plan
from aikyam_video.stages import quiet_snap
from aikyam_video.audio import AudioProfile


def two_color_video(path, w=640, h=360):
    """3 s pure red + 440 Hz, then 3 s pure blue + 880 Hz: a blend is detectable in colour AND spectrum."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=red:s={w}x{h}:d=3:r=25", "-f", "lavfi", "-i", f"color=c=blue:s={w}x{h}:d=3:r=25",
                    "-f", "lavfi", "-i", "sine=f=440:d=3", "-f", "lavfi", "-i", "sine=f=880:d=3",
                    "-filter_complex", "[0][1]concat=n=2:v=1:a=0[v];[2][3]concat=n=2:v=0:a=1[a]", "-map", "[v]", "-map", "[a]",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", path], check=True)


def plan(transition, segs=((0.0, 3.0), (3.0, 6.0)), dur_s=1.0, captions=False):
    p = {"schemaVersion": 1, "source": {"videoId": "v", "path": "/x", "durationSeconds": 6}, "outputFormat": "REEL", "aspectRatio": "9:16",
         "segments": [{"start": a, "end": b, "reason": "x", "score": .9} for a, b in segs],
         "captions": {"enabled": False, "language": "en"}, "overlays": {"template": "divine_moment"},
         "transitions": {"type": transition, "durationSeconds": dur_s if transition != "cut" else 0},
         "audio": {"preserveOriginal": True, "normalize": False}, "thumbnail": {}}
    p["durationSeconds"] = round(output_duration(p["segments"], p["transitions"]), 2)
    return p


# ------------------------------------------------------------------ plan accounting
def test_crossfade_shortens_output_by_the_overlaps():
    p = plan("crossfade", segs=((0, 10), (20, 30), (40, 50)), dur_s=0.5)
    assert p["durationSeconds"] == 29.0                       # 30 - 2 * 0.5
    validate_plan({**p, "source": {**p["source"], "durationSeconds": 60}}, 60)
    assert plan("cut", segs=((0, 10), (20, 30)))["durationSeconds"] == 20.0


def test_validate_rejects_wrong_duration_or_oversized_crossfade():
    p = plan("crossfade", segs=((0, 10), (20, 30)), dur_s=0.5)
    with pytest.raises(PlanError, match="rendered length"):
        validate_plan({**p, "durationSeconds": 20.0, "source": {**p["source"], "durationSeconds": 60}}, 60)     # forgot the overlap
    big = plan("crossfade", segs=((0, 2), (5, 7)), dur_s=1.5)                                                   # > half of a 2 s segment
    with pytest.raises(PlanError, match="longer than half of segment"):
        validate_plan({**big, "source": {**big["source"], "durationSeconds": 60}}, 60)


def _moments(n=3, length=12.0):
    return [Moment(momentId=f"m{i}", start=i * 40.0, end=i * 40.0 + length, reason="aarti", score=0.9 - i * 0.1) for i in range(n)]


def test_planner_defaults_to_crossfade_and_single_segment_stays_cut():
    tr = Transcript(language="en")
    p = plan_edit("v", "/x", 200, _moments(3), tr, [], {}, target_s=45)
    assert p["transitions"]["type"] == "crossfade" and 0 < p["transitions"]["durationSeconds"] <= 0.5
    n = len(p["segments"]); assert n >= 2
    assert abs(p["durationSeconds"] - (sum(s["end"] - s["start"] for s in p["segments"]) - (n - 1) * p["transitions"]["durationSeconds"])) < 0.02
    one = plan_edit("v", "/x", 200, _moments(1), tr, [], {}, target_s=45)
    assert one["transitions"]["type"] == "cut"                                 # nothing to blend
    assert plan_edit("v", "/x", 200, _moments(3), tr, [], {}, target_s=45, transition="cut")["transitions"]["type"] == "cut"


def test_captions_follow_overlapped_timeline_and_do_not_stack():
    w = lambda s, e, t: TranscriptSegment(start=s, end=e, text=t)
    tr = Transcript(language="en", segments=[w(0, 10, "first moment words"), w(40, 50, "second moment words")])
    moments = [Moment(momentId="a", start=0, end=10, reason="x", score=.9), Moment(momentId="b", start=40, end=50, reason="x", score=.8)]
    p = plan_edit("v", "/x", 100, moments, tr, [], {}, target_s=45, transition_s=1.0)
    d = p["transitions"]["durationSeconds"]; cues = p["captions"]["cues"]
    assert len(cues) == 2 and abs(cues[1]["start"] - (10 - d)) < 0.01           # 2nd cue starts when segment 2 starts (overlapped)
    assert cues[0]["end"] <= cues[1]["start"] + 1e-6                            # first cue cut before the second appears
    assert cues[1]["end"] <= p["durationSeconds"] + 0.01


def test_snap_moves_a_trimmed_end_to_a_quiet_moment():
    n = 60; db = np.full(n, -20.0); db[20] = -60.0                             # one quiet half-second at t = 10.0-10.5
    a = AudioProfile(0.5, db, db < -45, np.zeros(n, bool), np.zeros(n), {})
    assert abs(quiet_snap(a)(10.4) - 10.25) < 0.3 and quiet_snap(a)(25.0) == 25.0   # flat loud audio, no quiet spot: cut is left exactly where it was
    m = [Moment(momentId="m0", start=0, end=60, reason="x", score=.9)]         # a 60 s moment trimmed to 15 s: its END should land on the dip
    db2 = np.full(120, -20.0); db2[29] = -60.0                                 # dip at 14.5-15.0
    a2 = AudioProfile(0.5, db2, db2 < -45, np.zeros(120, bool), np.zeros(120), {})
    p = plan_edit("v", "/x", 60, m, Transcript(language="en"), [], {}, target_s=45, max_seg_s=15, snap=quiet_snap(a2))
    assert 14.0 <= p["segments"][0]["end"] <= 15.5 and abs(p["segments"][0]["end"] - 14.75) < 0.3


# ------------------------------------------------------------------ FFmpeg renderer: the blend is real
def _frame_rgb(path, t):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", path, "-frames:v", "1", "-vf", "scale=32:32,format=rgb24", "-f", "rawvideo", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, 3).mean(axis=0)


def _tone_energy(path, t0, t1, f):
    pcm = ff.extract_audio_pcm(path, 16000)[int(t0 * 16000): int(t1 * 16000)]
    spec = np.abs(np.fft.rfft(pcm * np.hanning(len(pcm)))); fr = np.fft.rfftfreq(len(pcm), 1 / 16000)
    return spec[(fr > f - 15) & (fr < f + 15)].sum() / (spec.sum() + 1e-9)


def test_ffmpeg_crossfade_blends_picture_and_sound(tmp_path):
    from aikyam_video.render import render_format
    src = str(tmp_path / "rb.mp4"); two_color_video(src)
    out = str(tmp_path / "x.mp4"); render_format(plan("crossfade", dur_s=1.0), src, out, "square", str(tmp_path))
    i = ff.probe(out); assert abs(i.duration - 5.0) < 0.15 and i.has_audio            # 3 + 3 - 1
    before, mid, after = _frame_rgb(out, 1.0), _frame_rgb(out, 2.5), _frame_rgb(out, 4.0)
    assert before[0] > 200 and before[2] < 60                                          # pure red before
    assert after[2] > 200 and after[0] < 60                                            # pure blue after
    assert mid[0] > 60 and mid[2] > 60                                                 # mid-transition: BOTH red and blue present
    assert mid.max() > 100                                                             # ... and not a dip to black
    a440, a880 = _tone_energy(out, 2.3, 2.7, 440), _tone_energy(out, 2.3, 2.7, 880)
    assert a440 > 0.05 and a880 > 0.05                                                 # both tones audible during the blend (audio crossfade)
    assert _tone_energy(out, 0.5, 1.5, 880) < 0.02 and _tone_energy(out, 3.6, 4.6, 440) < 0.02   # outside the blend: one tone only


def test_ffmpeg_hard_cut_still_available(tmp_path):
    from aikyam_video.render import render_format
    src = str(tmp_path / "rb.mp4"); two_color_video(src)
    out = str(tmp_path / "c.mp4"); render_format(plan("cut"), src, out, "square", str(tmp_path))
    assert abs(ff.probe(out).duration - 6.0) < 0.15
    a, b = _frame_rgb(out, 2.9), _frame_rgb(out, 3.1)
    assert a[0] > 200 and a[2] < 60 and b[2] > 200 and b[0] < 60                       # instant switch, no blend


def test_ffmpeg_three_segments_and_captions_stay_in_sync(tmp_path):
    from aikyam_video.render import render_format
    src = str(tmp_path / "rb.mp4"); two_color_video(src)
    p = plan("crossfade", segs=((0.0, 2.0), (2.0, 4.0), (4.0, 6.0)), dur_s=0.5)
    p["captions"] = {"enabled": True, "language": "en", "mode": "sentence", "style": {}, "cues": [{"start": 3.0, "end": 4.5, "text": "HELLO", "words": []}]}
    out = str(tmp_path / "3.mp4"); render_format(p, src, out, "reel", str(tmp_path))
    assert abs(ff.probe(out).duration - 5.0) < 0.15                                    # 6 - 2 * 0.5
    def band(t):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", out, "-frames:v", "1", "-vf", "crop=iw:ih*0.14:0:ih*0.68,format=gray", "-f", "rawvideo", "-"], capture_output=True).stdout
        return (np.frombuffer(raw, np.uint8) > 240).sum()
    assert band(3.7) > 300 and band(1.0) < 50                                          # caption present when it should be, absent otherwise


# ------------------------------------------------------------------ per-edge transitions, narrative order (plan schema v2)
def v2(segs, edges, order=None):
    """segs: [(start, end)]; edges: transitionIn for segment 1..n-1 as (type, seconds)."""
    p = plan("cut", segs=segs)
    for i, (ty, d) in enumerate(edges, start=1):
        p["segments"][i]["transitionIn"] = {"type": ty, "durationSeconds": d}
    if order: p["creative"] = {"version": 2, "order": order}
    p["durationSeconds"] = round(output_duration(p["segments"], p["transitions"]), 2)
    return p


def test_per_edge_transitions_set_the_timeline():
    from aikyam_video.plan import edge_overlaps, timeline
    p = v2(((0, 10), (20, 30), (40, 50)), (("crossfade", 1.0), ("cut", 0)))
    assert edge_overlaps(p["segments"], p["transitions"]) == [1.0, 0.0]
    assert p["durationSeconds"] == 29.0 and [round(t["start"], 2) for t in timeline(p["segments"], p["transitions"])] == [0.0, 9.0, 19.0]
    validate_plan({**p, "source": {**p["source"], "durationSeconds": 60}}, 60)
    q = v2(((0, 10), (20, 30), (40, 50)), (("dip_black", 0.6), ("dip_white", 0.4)))
    assert q["durationSeconds"] == 29.0                                       # every blending type overlaps; only cut does not


def test_narrative_order_allowed_only_when_declared_and_never_overlapping():
    src = lambda p: {**p, "source": {**p["source"], "durationSeconds": 60}}
    out_of_order = v2(((30, 40), (0, 10)), (("cut", 0),))
    with pytest.raises(PlanError, match="out of order"): validate_plan(src(out_of_order), 60)
    validate_plan(src(v2(((30, 40), (0, 10)), (("cut", 0),), order="narrative")), 60)    # declared: hook-first / reordered story is fine
    with pytest.raises(PlanError, match="overlap in the source"): validate_plan(src(v2(((30, 42), (40, 50)), (("cut", 0),), order="narrative")), 60)


def test_transitions_cannot_swallow_a_clip():
    p = v2(((0, 4), (10, 14), (20, 24)), (("crossfade", 1.9), ("crossfade", 1.9)))            # middle clip: 3.8 of 4.0 s consumed
    with pytest.raises(PlanError, match="80%"): validate_plan({**p, "source": {**p["source"], "durationSeconds": 60}}, 60)


def test_ffmpeg_mixed_edges_render_with_correct_length_and_blend(tmp_path):
    """cut then crossfade: instant switch at the cut, blend at the crossfade, length = sum - overlaps."""
    from aikyam_video.render import render_format
    src = str(tmp_path / "rb.mp4"); two_color_video(src)
    # red 0-2 | blue 3-5 (cut) | red... reuse: segments (0,2) red, (4,6) blue, (0,2)?? sources may not overlap -> use (0,1.5), (3.5,5), (1.6,3)
    p = v2(((0.0, 1.5), (3.5, 5.0), (1.6, 3.0)), (("cut", 0), ("crossfade", 0.6)), order="narrative")
    out = str(tmp_path / "m.mp4"); render_format(p, src, out, "square", str(tmp_path))
    assert abs(ff.probe(out).duration - (1.5 + 1.5 + 1.4 - 0.6)) < 0.2 and ff.probe(out).has_audio
    a, b = _frame_rgb(out, 1.4), _frame_rgb(out, 1.6)                                          # the hard cut at t = 1.5: red -> blue
    assert a[0] > 200 and b[2] > 200
    mid = _frame_rgb(out, 2.7)                                                                  # crossfade blue -> red spans 2.4-3.0
    assert mid[0] > 60 and mid[2] > 60
