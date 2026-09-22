"""Edge placement + motion signatures adapted from hypecut (Apache-2.0) / auto-editor (Unlicense): unit behaviour, real-FFmpeg edge landing, duplicate rule, beat-lock."""
import subprocess
from types import SimpleNamespace as N
import numpy as np, pytest
from aikyam_video.creative import edges as ED, music_sync as MS, story
from aikyam_video.creative.model import Clip, Timeline
from aikyam_video.creative.musicdna import MusicAnalysis
from tests.test_creative_story import e, shot


def scene_frames(spec, fps=10, h=54, w=96, seed=0):
    """spec = [(seconds, level)]: flat gray scenes with slight noise, hard cuts between them."""
    r = np.random.RandomState(seed); out = []
    for sec, lv in spec: out += [np.clip(lv + r.randn(h, w) * 1.5, 0, 255).astype(np.uint8) for _ in range(int(sec * fps))]
    return np.stack(out)


# ---------------------------------------------------------------- boundaries
def test_hard_cuts_are_found_where_they_are_and_flicker_on_still_footage_is_not_a_cut():
    fr = scene_frames([(3, 40), (3, 200), (2, 90)]); cuts = ED.find_boundaries(fr, 10)
    assert len(cuts) == 2 and abs(cuts[0] - 3.0) < 0.15 and abs(cuts[1] - 6.0) < 0.15
    still = scene_frames([(8, 120)]); assert len(ED.find_boundaries(still, 10)) == 0                  # codec noise on a still frame: the absolute floor says no


def test_a_cut_is_still_found_inside_busy_footage_thanks_to_the_local_baseline():
    r = np.random.RandomState(2); busy = [np.clip(120 + r.randn(54, 96) * 30, 0, 255).astype(np.uint8) for _ in range(60)]                # chaotic frames differ a lot every step
    dark = [np.clip(20 + r.randn(54, 96) * 2, 0, 255).astype(np.uint8) for _ in range(30)]
    cuts = ED.find_boundaries(np.stack(busy[:30] + dark + busy[30:]), 10); assert any(abs(c - 3.0) < 0.15 for c in cuts) or any(abs(c - 6.0) < 0.15 for c in cuts)


# ---------------------------------------------------------------- smoothing (auto-editor) + pauses (hypecut)
def test_smoothing_drops_short_runs_fills_short_gaps_and_terminates():
    m = np.array([0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0], bool)
    s = ED.smooth_mask(m, mincut=2, minclip=2)                                                          # the lone 1 goes, the single-step 0 gap between runs is filled
    assert not s[2] and s[10] and s[6:15].all()
    assert np.array_equal(ED.smooth_mask(s, 2, 2), s)                                                   # a fixed point


def test_pauses_are_relative_to_the_clips_own_level():
    for base in (-20.0, -50.0):                                                                          # a shout and a whisper both have gaps 14+ dB down
        rms = np.full(100, base); rms[30:38] = base - 25; rms[60:66] = base - 25
        p = ED.pause_runs(rms, 0.1, 0.0, 10.0); assert len(p) == 2 and abs(p[0][0] - 3.0) < 0.11 and abs(p[1][0] - 6.0) < 0.11
    assert ED.pause_runs(np.full(100, -30.0), 0.1, 0, 10) == []                                          # continuous sound: nothing to land on
    assert ED.pause_runs(np.full(100, -80.0), 0.1, 0, 10) == []


# ---------------------------------------------------------------- snapping rules
def test_edges_land_on_real_cuts_within_the_window_and_the_core_is_protected():
    a, b, info = ED.snap_window(10.3, 16.2, 8.0, 20.0, cuts=[10.0, 16.0], pauses=[], min_len=4.0, max_len=9.0)
    assert (a, b) == (10.0, 16.0) and info["start"][1] == "cut" and info["end"][1] == "cut"
    a, b, info = ED.snap_window(10.0, 16.0, 8.0, 20.0, cuts=[12.5], pauses=[], min_len=3.0, max_len=9.0)      # a cut in the middle of the window: inside the protected core
    assert (a, b) == (10.0, 16.0) and not info


def test_a_snap_that_breaks_the_length_budget_is_rejected_not_clamped():
    a, b, info = ED.snap_window(10.0, 14.0, 8.0, 20.0, cuts=[10.9], pauses=[], min_len=3.8, max_len=5.0)
    assert (a, b) == (10.0, 14.0) and "start" not in info                                                 # snapping the start to 10.9 would leave 3.1 s < 3.8 s


def test_a_real_cut_wins_over_a_pause_and_pauses_are_used_when_there_is_no_cut():
    a, b, info = ED.snap_window(10.4, 16.0, 8.0, 20.0, cuts=[10.0], pauses=[(10.5, 10.9)], min_len=4.0, max_len=9.0)
    assert a == 10.0 and info["start"][1] == "cut"
    a, b, info = ED.snap_window(10.4, 16.3, 8.0, 20.0, cuts=[], pauses=[(9.6, 10.2), (16.0, 16.2)], min_len=4.0, max_len=9.0)
    assert info["start"][1] == "pause" and a < 10.4 and abs(a - (10.2 - 0.05)) < 1e-6 and info["end"][1] == "pause" and abs(b - (16.0 + 0.12)) < 1e-6
    a, b, info = ED.snap_window(10.0, 16.0, 9.9, 16.05, cuts=[16.9], pauses=[], min_len=4.0, max_len=9.0); assert b <= 16.05                # never leaves the shot


# ---------------------------------------------------------------- motion signature
def moving(x0, x1, n=30, h=54, w=96):
    """Static textured background + a bright block sweeping from x0 to x1: same 'venue', different movement."""
    bg = (np.random.RandomState(0).rand(h, w) * 60 + 80).astype(np.uint8); out = []
    for i in range(n):
        f = bg.copy(); x = int(x0 + (x1 - x0) * i / (n - 1)); f[20:34, max(0, x):x + 12] = 250; out.append(f)
    return np.stack(out)


def test_motion_signatures_separate_different_movements_in_the_same_venue():
    left, right, left2 = ED.motion_signature(moving(2, 30)), ED.motion_signature(moving(60, 82)), ED.motion_signature(moving(3, 31))
    assert left is not None and float(np.dot(left, left2)) > 0.85 and float(np.dot(left, right)) < 0.5
    assert ED.motion_differs(left, right) and not ED.motion_differs(left, left2) and not ED.motion_differs(left, None)
    assert ED.motion_signature(np.stack([moving(5, 5)[0]] * 20)) is None                                 # a still frame: no signature (noise must not correlate)


def slots_with_sig(sig):
    s = shot(0, 0, 10, {"aarti": 0.9}, e(1, 0, 0, 0)); s.slots.motion_sig = sig; return s


def test_identical_looking_shots_are_only_duplicates_when_they_also_move_alike():
    A = ED.motion_signature(moving(2, 30)); B = ED.motion_signature(moving(60, 82))
    x, y = slots_with_sig(A), slots_with_sig(B); y.id = "m1"
    assert story.is_duplicate(x, y, motion_dedupe=False)                                                  # appearance alone: the old rule
    assert not story.is_duplicate(x, y, motion_dedupe=True)                                               # a different happening in the same venue
    z = slots_with_sig(A); z.id = "m2"; assert story.is_duplicate(x, z, motion_dedupe=True)               # the same movement twice really is a repeat
    n = slots_with_sig(None); n.id = "m3"; assert story.is_duplicate(x, n, motion_dedupe=True)            # unknown motion: fall back to the old rule


def test_same_venue_shots_can_form_a_longer_story_with_motion_dedupe():
    """Regression for the 28 s single-scene recording that produced a 10 s one-clip reel."""
    from aikyam_video.creative import pacing
    from aikyam_video.creative.shots import Population
    sigs = [ED.motion_signature(moving(x, x + 24)) for x in (2, 30, 60)]
    pool = []
    for i, sg in enumerate(sigs):
        s = shot(i, i * 12, 10, {"aarti": 0.9}, e(1, 0.001 * i, 0, 0), 0.5); s.slots.motion_sig = sg; pool.append(s)
    pop = Population(pool); prof = pacing.profile("devotional")
    old = story.plan_story(pool, pop, prof, 40.0, motion_dedupe=False); new = story.plan_story(pool, pop, prof, 40.0, motion_dedupe=True)
    fills = lambda tl: sum(d["type"] == "fill" for d in tl.decisions)
    assert len(new.clips) >= 2 and fills(new) == 0                                                          # motion-aware duplicates: the story stands on its own
    assert len(old.clips) - fills(old) == 1 and fills(old) >= 1                                            # appearance-only: one clip, the rest only via the top-up rule


# ---------------------------------------------------------------- real FFmpeg: a clip edge lands on the actual cut
def test_a_real_clip_edge_lands_within_a_frame_of_the_real_cut(tmp_path):
    from aikyam_video.creative import engine as E
    from aikyam_video.creative.shots import build_shots
    from aikyam_video.models import Moment, SceneVision, VisionResult
    from aikyam_video import ffmpeg as ff
    f = str(tmp_path / "cuts.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=25:d=6", "-f", "lavfi", "-i", "mandelbrot=s=320x180:r=25", "-f", "lavfi", "-i", "smptebars=s=320x180:r=25:d=6",
                    "-filter_complex", "[1:v]trim=duration=6,setpts=PTS-STARTPTS[m];[0:v][m][2:v]concat=n=3:v=1:a=0", "-pix_fmt", "yuv420p", f], check=True)
    info = ff.probe(f); assert abs(info.duration - 18.0) < 0.2                                             # cuts at 6.0 and 12.0
    mom = Moment(momentId="m0", start=5.4, end=13.0, reason="x", score=0.5)
    vis = [SceneVision(sceneId="s", start=0, end=18, vision=VisionResult(labels={"aarti": 0.8}, embedding=[1.0, 0.0]))]
    sh = build_shots(f, [mom], vis, None, asset_id="a1", duration=info.duration)[0]
    assert any(abs(c - 6.0) < 0.2 for c in sh.cuts) and any(abs(c - 12.0) < 0.2 for c in sh.cuts)
    edger = E._make_edger({"a1": N(path=f, info=info)}); a, b, inf = edger(sh, 6.3, 11.6, 4.0, 8.0)
    assert abs(a - 6.0) <= 1.5 / 25 and abs(b - 12.0) <= 1.5 / 25 and inf["start"][1] == "cut" and inf["end"][1] == "cut"           # within one frame of the true cuts (6.0 / 12.0)


# ---------------------------------------------------------------- an out point on a real cut is not moved by the beat aligner
def test_beat_alignment_leaves_a_clip_that_ends_on_a_real_cut_alone():
    b = 0.6; beats = [i * b for i in range(120)]; an = MusicAnalysis(72, 100, 9.0, beats, [0.5] * 144, "rhythmic", 2.0, beats[::4], "beat_this")
    clips = []
    for i, L in enumerate([4.1, 5.3, 3.7]):
        s = shot(i, i * 30, 20.0, {"aarti": 0.8}); clips.append(Clip(s, s.start, s.start + L, "RITUAL", locked_end=(i == 1)))
    tl = Timeline(clips, "devotional", ["RITUAL"] * 3); MS.align_cuts(tl, an, 0.0, lambda t: [0.0, 0.0])
    assert abs(tl.clips[1].length - 5.3) < 1e-6


# ---------------------------------------------------------------- QC duplicate check must not flag two DIFFERENT movements in the same venue
def _write(path, frames):
    import cv2
    h, w = frames[0].shape[:2]; p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", "25", "-i", "-", "-vf", "scale=270:480,format=yuv420p", "-c:v", "libx264", path], stdin=subprocess.PIPE)
    p.stdin.write(np.stack(frames).tobytes()); p.stdin.close(); p.wait()


def test_qc_duplicate_check_needs_matching_movement_not_just_matching_looks(tmp_path):
    from aikyam_video.creative import qc as QC
    def clip(x0, x1, n=100): return [cv2.resize(f, (192, 108), interpolation=cv2.INTER_NEAREST) for f in moving(x0, x1, n=n)]
    import cv2
    tl = [{"start": 0.0, "end": 4.0}, {"start": 4.0, "end": 8.0}]
    plan = {"creative": {"motionDedupe": True}, "segments": []}
    different = str(tmp_path / "d.mp4"); _write(different, clip(2, 30) + clip(60, 82))                       # same background, block sweeps left vs right
    same = str(tmp_path / "s.mp4"); _write(same, clip(2, 30) + clip(2, 30))                                   # the same movement twice
    d = {c.name: c for c in QC.check_duplicates_and_weak(plan, different, tl)}; s = {c.name: c for c in QC.check_duplicates_and_weak(plan, same, tl)}
    assert d["duplicate_shots"].status == "pass" and s["duplicate_shots"].status == "fail"


# ---------------------------------------------------------------- disjoint windows from overlapping moments, and a length target relative to the footage
def test_overlapping_moments_become_disjoint_windows_covering_the_footage():
    from aikyam_video.creative.shots import partition_moments
    from aikyam_video.models import Moment
    ms = [Moment(momentId="a", start=0, end=20, reason="x", score=0.27), Moment(momentId="b", start=10, end=28.15, reason="x", score=0.26), Moment(momentId="c", start=20, end=28.15, reason="x", score=0.25)]
    out = partition_moments(ms)                                                                          # the Ganga clip: three overlapping windows of ONE take
    spans = sorted((m.start, m.end) for m in out)
    assert all(b1 <= a2 + 1e-6 for (_, b1), (a2, _) in zip(spans, spans[1:]))                              # disjoint
    assert abs(sum(b - a for a, b in spans) - 28.15) < 0.01                                                  # the whole take is covered, nothing counted twice
    long = partition_moments([Moment(momentId="l", start=0, end=28.0, reason="x", score=0.5)]); assert len(long) == 3 and max(m.end - m.start for m in long) <= 12.0 + 1e-6   # a 28 s take is split
    assert len(partition_moments([Moment(momentId="s", start=0, end=19.6, reason="x", score=0.5)])) == 1      # a 19.6 s take is not (two halves would be near-duplicates)
    assert len({m.momentId for m in out}) == len(out)                                                      # unique ids
    assert [(m.start, m.end) for m in partition_moments([Moment(momentId="a", start=2, end=9, reason="x", score=.5)])] == [(2, 9)]   # a single short moment is untouched


def test_qc_duration_window_shrinks_with_the_footage_available(tmp_path):
    from aikyam_video.creative import qc as QC
    f = str(tmp_path / "v.mp4"); subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=160x90:r=25:d=18", "-pix_fmt", "yuv420p", f], check=True)
    strict = {"durationSeconds": 18.0, "creative": {"targetRange": [28, 40]}}; relative = {"durationSeconds": 18.0, "creative": {"targetRange": [23.8, 40]}}
    assert QC.check_duration(strict, f)[0].status == "warn" and QC.check_duration(relative, f)[0].status == "pass"


def test_a_reel_is_topped_up_to_15_seconds_from_unused_footage_even_when_the_duplicate_rule_removed_it():
    from aikyam_video.creative import pacing
    from aikyam_video.creative.shots import Population
    a, b = ED.motion_signature(moving(2, 30)), ED.motion_signature(moving(60, 82))
    pool = []
    for i, sg in enumerate([a, a, a]):                                           # three 6 s shots of ONE repetitive ritual: every pair is a duplicate, the story alone can only use one
        s = shot(i, i * 12, 6, {"aarti": 0.9}, e(1, 0.001 * i, 0, 0), 0.5); s.slots.motion_sig = sg; pool.append(s)
    pop = Population(pool); prof = pacing.profile("devotional")
    tl = story.plan_story(pool, pop, prof, 40.0)
    total = sum(c.length for c in tl.clips); assert total >= 14.0 and any(d["type"] == "fill" for d in tl.decisions)      # 6 s from the story, the rest from the top-up
    ids = [c.shot.id for c in tl.clips]; assert len(ids) == len(set(ids)) and all(not story._clash(x.shot, y.shot) for i, x in enumerate(tl.clips) for y in tl.clips[i + 1:])
    rich = [shot(i, i * 30, 12, {"aarti": 0.9, "deity": 0.3 * i}, e(*np.eye(6)[i]), 0.5) for i in range(6)]                # plenty of distinct footage: no top-up needed
    assert not any(d["type"] == "fill" for d in story.plan_story(rich, Population(rich), prof, 200.0).decisions)
