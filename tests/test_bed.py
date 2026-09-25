"""--source-audio bed and --no-transcript: the recording's own music as ONE continuous soundtrack, and no speech-to-text at all."""
import json
import numpy as np
from aikyam_video.audio import AudioProfile
from aikyam_video.creative import bed, mixer
from aikyam_video.options import Options


def _profile(D=300.0, hop=0.5):
    n = int(D / hop); z = np.zeros(n)
    return AudioProfile(hop, np.full(n, -30.0), np.zeros(n, bool), np.zeros(n, bool), np.full(n, 0.5), {})


def test_picks_the_music_like_stretch_not_the_speech_or_the_silence():
    a = _profile(); n = len(a.rms_db); ev = lambda lo, hi, v: np.array([v if lo <= i * a.hop < hi else 0.0 for i in range(n)])
    a.events = {"chant": ev(100, 180, 0.9), "speech": ev(200, 260, 0.9)}
    a.silence[int(20 / a.hop):int(60 / a.hop)] = True                                  # a silent stretch early on
    start, score, why = bed.pick_window(a, 40.0)
    assert 100 <= start <= 140 and score > 0.5 and "music-like" in why                # inside the chanting, clear of the speech and the silence


def test_short_recording_and_missing_tagger_are_handled():
    a = _profile(D=30.0); assert bed.pick_window(a, 40.0)[0] == 0.0                     # shorter than the reel: use it all
    b = _profile(D=300.0); s, _, _ = bed.pick_window(b, 40.0); assert 15 <= s <= 245    # no sound events at all: still a valid window away from the edges


def test_bed_is_a_silent_source_for_selection_but_the_mix_still_has_sound(tmp_path):
    o = Options(source_audio="bed"); assert o.silent_source and Options(source_audio="off").silent_source and not Options().silent_source
    from scipy.io import wavfile
    t = np.arange(12 * 48000) / 48000; wavfile.write(str(tmp_path / "bed.wav"), 48000, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32))
    plan = {"durationSeconds": 6.0, "segments": [{"start": 0.0, "end": 3.0}, {"start": 10.0, "end": 13.0}], "transitions": {"type": "cut", "durationSeconds": 0}}
    r = mixer.render_audio(plan, "/no/such/source.mp4", str(tmp_path / "bed.wav"), 0.0, None, 0.5, live_on=False)
    assert r.report["music_used"] and float(np.abs(r.pcm).max()) > 0.01 and abs(r.report["lufs"] - mixer.TARGET_LUFS) < 1.0     # the bed is mastered like any soundtrack


def test_no_transcript_skips_whisper_and_writes_an_empty_transcript(tmp_path, monkeypatch):
    from aikyam_video import stages
    monkeypatch.setattr(stages, "_provider", lambda *a, **k: (_ for _ in ()).throw(AssertionError("Whisper must not run")))
    tr = stages.transcription("/no/such/video.mp4", str(tmp_path), Options(transcript=False))
    assert tr.segments == [] and json.load(open(tmp_path / stages.TRANSCRIPT))["segments"] == []


def test_cli_guards_for_bed(capsys):
    from aikyam_video.cli import main
    assert main(["process", "x.mp4", "--source-audio", "bed", "--engine", "classic"]) == 1
    assert "creative engine" in capsys.readouterr().err


def test_bed_is_picked_across_several_recordings_from_the_file_it_lives_in():
    poor = _profile(D=300.0); rich = _profile(D=300.0); short = _profile(D=20.0); n = len(rich.rms_db)
    rich.events = {"bhajan": np.array([0.9 if 100 <= i * rich.hop < 180 else 0.0 for i in range(n)])}
    s, sc, why, path = bed.pick_across([(poor, "a.mp4"), (rich, "b.mp4"), (short, "c.mp4")], 40.0)
    assert path == "b.mp4" and 100 <= s <= 140 and sc > 0.5                       # the chanting recording, not the first one and not the one too short for the reel
    assert bed.pick_across([(short, "c.mp4")], 40.0)[3] == "c.mp4"                # nothing better: the short one is still used (whole audio)
    assert bed.pick_across([(None, "x.mp4")], 40.0) is None
