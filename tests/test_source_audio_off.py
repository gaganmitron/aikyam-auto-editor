"""--source-audio off: the recorded sound (narration, chanting) is never heard; the reel's sound is the music alone."""
import numpy as np
from scipy.io import wavfile
from aikyam_video import music as M
from aikyam_video.creative import engine as E, mixer
from aikyam_video.options import Options


def _tone(path, seconds=12.0, hz=220.0):
    t = np.arange(int(seconds * 48000)) / 48000
    wavfile.write(str(path), 48000, (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32))
    return str(path)


PLAN = {"durationSeconds": 6.0, "segments": [{"start": 0.0, "end": 3.0}, {"start": 10.0, "end": 13.0}], "transitions": {"type": "cut", "durationSeconds": 0}}


def test_mix_is_music_only_and_mastered_to_target(tmp_path):
    r = mixer.render_audio(dict(PLAN), "/no/such/source.mp4", _tone(tmp_path / "m.wav"), 0.0, None, 0.5, live_on=False)   # the source file is never opened
    assert r.pcm.shape == (6 * mixer.SR, 2)
    assert float(np.abs(r.live).max()) == 0.0 and float(np.abs(r.music).max()) > 0.01          # no recorded sound at all
    assert abs(r.report["lufs"] - mixer.TARGET_LUFS) < 1.0 and r.report["true_peak_db"] <= mixer.CEILING_DB + 0.2
    assert r.report["music_used"] is True


def test_music_is_not_ducked_when_there_is_nothing_to_duck_under(tmp_path):
    r = mixer.render_audio(dict(PLAN), "/no/such/source.mp4", _tone(tmp_path / "m.wav"), 0.0, None, 0.5, live_on=False)
    assert float(np.ptp(r.music_gain_db[: int(4 / mixer.FRAME)])) < 1e-6                      # flat gain (fades aside): no pumping


def test_no_music_and_no_source_is_silence_not_a_crash():
    r = mixer.render_audio(dict(PLAN), "/no/such/source.mp4", None, 0.0, None, 0.5, live_on=False)
    assert float(np.abs(r.pcm).max()) == 0.0


def test_music_is_always_chosen_when_the_source_is_off(monkeypatch):
    lib = [M.Track("weak", "weak", "/x", "AIKYAM-OWNED", None, 0.5, ["deity"], [], [], [])]
    monkeypatch.setattr(M, "load_library", lambda *a, **k: (lib, [])); monkeypatch.setattr(M, "score", lambda t, p: 0.1); monkeypatch.setattr(E, "_kind", lambda t: "melodic")
    prof = E.pacing.profile("devotional"); plan_like = {"source": {}, "segments": []}
    assert E.choose_music(plan_like, None, prof, Options(source_audio="keep"))[0] is None       # normal mode: a poor match means no music
    track, why = E.choose_music(plan_like, None, prof, Options(source_audio="off"))
    assert track is not None and track.id == "weak" and "recorded sound off" in why              # silent source: the best available track, always


def test_cli_refuses_combinations_that_cannot_work(capsys):
    from aikyam_video.cli import main
    assert main(["process", "x.mp4", "--source-audio", "off", "--engine", "classic"]) == 1
    assert main(["process", "x.mp4", "--source-audio", "off", "--music", "off"]) == 1
    assert "creative engine" in capsys.readouterr().err
