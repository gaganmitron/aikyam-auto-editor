from types import SimpleNamespace as N
from aikyam_video.creative import cutpoints as cp
from aikyam_video.creative.model import SlotFeatures, Shot
from aikyam_video.models import Transcript, TranscriptSegment, Word


def shot(motion, start=10.0):
    n = len(motion); z = [0.0] * n
    return Shot("s", start, start + n * 0.5, "m", 0.5, {}, [], None, SlotFeatures(0.5, [1.0] * n, z, list(motion), [.5] * n, [.5] * n, z, z, z, z, z))


def tr(words):
    ws = [Word(start=s, end=e, text=t) for t, s, e in words]; return Transcript(language="en", segments=[TranscriptSegment(start=ws[0].start, end=ws[-1].end, text="x", words=ws)])


def test_phrases_split_on_silence():
    p = cp.phrases(tr([("om", 10.0, 10.4), ("namah", 10.5, 11.0), ("shivaya", 12.0, 12.6)]))
    assert p == [(10.0, 11.0), (12.0, 12.6)]


def test_edge_never_lands_inside_a_spoken_phrase():
    phr = [(10.0, 13.05), (14.0, 16.0)]                                        # the quality-optimal out point (13.0) is mid-sentence
    a, b, info = cp.choose_edges(10.0, 13.0, 8.0, 20.0, [], [(13.05 + 0.12, "phrase")], phr, 2.0, 6.0)
    assert b > 13.05 and info["end"][1] == "phrase"


def test_last_clip_ends_settled_first_clip_starts_on_action():
    m = [0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1]                      # calm, action from 1.0 s to 3.0 s, calm again
    s = shot(m); ons, lul = cp.motion_points(s); assert 11.0 in ons and 13.5 in lul and 13.0 not in lul
    ins, outs = cp.candidates(s, [], [], [])
    a, b, info = cp.choose_edges(10.5, 12.5, 10.0, 15.0, ins, outs, [], 1.5, 4.0, first=True, last=True)
    assert info["start"][1] == "onset" and a == 11.0                              # opens where the action begins
    assert info["end"][1] == "lull" and b >= 13.5                                 # ends after the action has settled, not mid-motion


def test_real_cut_beats_everything_and_no_candidates_changes_nothing():
    a, b, info = cp.choose_edges(10.3, 16.2, 8.0, 20.0, [(10.0, "cut")], [(16.0, "cut")], [], 4.0, 9.0)
    assert (a, b) == (10.0, 16.0) and info["start"][1] == "cut" and info["end"][1] == "cut"
    assert cp.choose_edges(10.3, 16.2, 8.0, 20.0, [], [], [], 4.0, 9.0) == (10.3, 16.2, {})


def test_length_budget_is_respected():
    a, b, _ = cp.choose_edges(10.0, 14.0, 8.0, 20.0, [(10.9, "cut")], [], [], 3.8, 5.0)      # snapping to 10.9 would make it 3.1 s: rejected
    assert (a, b) == (10.0, 14.0)


def test_reel_end_lands_on_a_bar_line():
    from aikyam_video.creative import music_sync as MS
    from aikyam_video.creative.model import Clip, Timeline
    from aikyam_video.creative.musicdna import MusicAnalysis
    b = 0.6; beats = [i * b for i in range(120)]; an = MusicAnalysis(72, 100, 9.0, beats, [0.5] * 144, "rhythmic", 2.0, beats[::4], "beat_this")
    s = shot([0.5] * 40, start=0.0); tl = Timeline([Clip(s, 0.0, 9.0, "RITUAL")], "devotional", ["RITUAL"])          # 9.0 s: bars end every 2.4 s -> 9.6 is the nearest downbeat
    r = MS.align_end(tl, an, 0.0, lambda t: []); assert r["on"] == "downbeat" and abs(tl.clips[0].end - 9.6) < 1e-6
    tl.clips[0].locked_end = True; tl.clips[0].end = 9.0; assert MS.align_end(tl, an, 0.0, lambda t: [])["on"] is None and tl.clips[0].end == 9.0


def test_neural_models_find_real_cuts_and_speech(tmp_path):
    import os, shutil, subprocess, pytest
    from aikyam_video.creative import mlx
    if mlx.python_exe() is None: pytest.skip("no .venv-beat")
    f = str(tmp_path / "cuts.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=25:d=6", "-f", "lavfi", "-i", "mandelbrot=s=320x180:r=25", "-f", "lavfi", "-i", "smptebars=s=320x180:r=25:d=6",
                    "-filter_complex", "[1:v]trim=duration=6,setpts=PTS-STARTPTS[m];[0:v][m][2:v]concat=n=3:v=1:a=0", "-pix_fmt", "yuv420p", f], check=True)
    c = mlx.cuts(f); assert c is not None and len(c) == 2 and abs(c[0] - 6.0) < 0.1 and abs(c[1] - 12.0) < 0.1        # TransNetV2
    assert os.path.exists(f + ".ml.json") and mlx.cuts(f) == c                                                        # cached
    assert mlx.pauses([[0.0, 1.8], [3.2, 5.3]], 0.0, 6.0) == [[1.8, 3.2]] and mlx.speech(f) is None                    # Silero: no audio stream -> None
    if shutil.which("espeak-ng"):
        w = str(tmp_path / "s.wav"); subprocess.run(["espeak-ng", "-w", w, "om namah shivaya. jai ganga maiya"], check=True, capture_output=True)
        sp = mlx.speech(w); assert sp and 0.0 <= sp[0][0] < 0.5 and sp[-1][1] > 1.0
