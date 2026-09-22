"""Devotional music: licence gate, topic matching, 'never on top of live bhajan', and the mix in the rendered reel."""
import json, subprocess
import numpy as np, pytest
from aikyam_video import ffmpeg as ff, music
from aikyam_video.audio import AudioProfile


def write_lib(d, tracks, files=True):
    d.mkdir(exist_ok=True)
    for t in tracks:
        if files and t.get("file") and not t["file"].startswith("..") and not t["id"].startswith("bad_missing"):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=300:d=1", str(d / t["file"])], check=True)
    (d / "library.json").write_text(json.dumps({"tracks": tracks}))
    return str(d)


def trk(tid, licence="AIKYAM-OWNED", **kw):
    return {"id": tid, "title": tid, "file": f"{tid}.wav", "licence": licence, "licenceVerified": True, "energy": 0.3, "tags": {}, **kw}


# ------------------------------------------------------------------ licence gate
def test_licence_gate_accepts_only_verified_allowed_licences(tmp_path):
    d = write_lib(tmp_path / "m", [
        trk("ok_owned"), trk("ok_cc0", "CC0"), trk("ok_ccby", "CC-BY-4.0", attribution="Jane Doe, CC BY 4.0"),
        trk("bad_nc", "CC-BY-NC-4.0", attribution="x"), trk("bad_sa", "CC-BY-SA-4.0", attribution="x"), trk("bad_unknown", "ALL-RIGHTS-RESERVED"),
        trk("bad_unverified", licenceVerified=False), trk("bad_noattrib", "CC-BY-4.0"),
        {**trk("bad_missing_file"), "file": "nope.wav"}, {**trk("bad_traversal"), "file": "../escape.wav"}])
    ok, bad = music.load_library(d)
    assert {t.id for t in ok} == {"ok_owned", "ok_cc0", "ok_ccby"}
    reasons = dict(bad)
    assert "not allowed" in reasons["bad_nc"] and "not allowed" in reasons["bad_sa"] and "not allowed" in reasons["bad_unknown"]
    assert "licenceVerified" in reasons["bad_unverified"] and "attribution" in reasons["bad_noattrib"]
    assert "missing" in reasons["bad_missing_file"] and "outside" in reasons["bad_traversal"] or "missing" in reasons["bad_traversal"]


def test_get_track_refuses_unlisted_and_unlicensed(tmp_path, monkeypatch):
    d = write_lib(tmp_path / "m", [trk("good"), trk("nc", "CC-BY-NC-4.0", attribution="x")])
    (tmp_path / "m" / "bare.wav").write_bytes((tmp_path / "m" / "good.wav").read_bytes())          # a file with NO licence record
    monkeypatch.setenv("MUSIC_LIBRARY_DIR", d)
    assert music.get_track("good").licence == "AIKYAM-OWNED"
    with pytest.raises(music.LicenceError): music.get_track("nc")
    with pytest.raises(FileNotFoundError): music.get_track("bare")                                  # unlisted => refused
    for bad in ("../x", "a/b", ""):
        with pytest.raises(FileNotFoundError): music.get_track(bad)
    monkeypatch.setenv("AIKYAM_ALLOW_UNLISTED_MUSIC", "1")                                          # dev-only escape hatch
    assert music.get_track("bare").licence == "UNVERIFIED-DEV"


def test_allowed_licences_are_configurable(tmp_path, monkeypatch):
    d = write_lib(tmp_path / "m", [trk("owned"), trk("cc0", "CC0")])
    monkeypatch.setenv("AIKYAM_ALLOWED_MUSIC_LICENCES", "AIKYAM-OWNED")
    assert {t.id for t in music.load_library(d)[0]} == {"owned"}


# ------------------------------------------------------------------ matching
def lib():
    T = lambda tid, energy, labels, **tags: music.Track(tid, tid, "/x", "AIKYAM-OWNED", None, energy, labels, tags.get("deities", []), tags.get("rituals", []), tags.get("festivals", []))
    return [T("bells", 0.3, ["aarti", "lamps", "deity"], rituals=["ritual_12"]), T("calm", 0.15, ["deity", "idol", "priest"]),
            T("festive", 0.85, ["procession", "ritual_dance"], rituals=["ritual_16"], festivals=["festival_9"])]


def plan(reasons, **ids):
    return {"source": {"ritualId": None, "deityId": None, "festivalId": None, **ids}, "segments": [{"start": i * 10.0, "end": i * 10.0 + 10, "reason": r, "score": .9} for i, r in enumerate(reasons)]}


def audio_with(**ev):
    n = 40; z = np.zeros(n)
    return AudioProfile(0.5, z - 30, z.astype(bool), z.astype(bool), z, {k: z + v for k, v in ev.items()})


def test_topic_matching_picks_the_right_mood():
    t, why = music.choose(plan(["aarti", "aarti"], ritualId="ritual_12"), audio_with(bhajan=0.1), lib())
    assert t.id == "bells" and "matched" in why
    t, _ = music.choose(plan(["procession", "ritual_dance"], festivalId="festival_9"), audio_with(drums=0.4), lib())   # only a non-devotional tag present
    assert t.id == "festive"
    t, _ = music.choose(plan(["deity", "idol"]), audio_with(), lib())
    assert t.id in ("calm", "bells") and t.energy <= 0.3                                             # calm content never gets the dhol track


def test_no_music_when_nothing_matches_or_library_empty():
    t, why = music.choose(plan(["fireworks"]), audio_with(), lib()); assert t is None and "no track matches" in why
    t, why = music.choose(plan(["aarti"]), audio_with(), []); assert t is None and "no licensed music library" in why


def test_no_music_on_top_of_live_bhajan_or_chanting():
    p = plan(["aarti"], ritualId="ritual_12")
    assert music.choose(p, audio_with(bhajan=0.9), lib())[0] is None
    t, why = music.choose(p, audio_with(chant=0.7), lib()); assert t is None and "already devotional" in why
    assert music.choose(p, audio_with(bhajan=0.9), lib(), skip_threshold=0.95)[0] is not None       # threshold is tunable
    assert music.choose(p, None, lib())[0] is not None                                               # no audio tagger => can't tell; music allowed


def test_music_block_carries_credit_info():
    t = music.Track("x", "Song", "/p", "CC-BY-4.0", "Jane, CC BY 4.0", 0.3)
    b = music.music_block(t, "why"); assert b["enabled"] and b["attribution"] == "Jane, CC BY 4.0" and b["licence"] == "CC-BY-4.0"


# ------------------------------------------------------------------ the shipped starter library
def test_starter_library_is_valid_and_covers_the_main_moods(monkeypatch):
    import os
    monkeypatch.setenv("MUSIC_LIBRARY_DIR", os.path.join(os.path.dirname(__file__), "..", "music"))
    tracks, bad = music.load_library()
    assert not bad and len(tracks) >= 4 and all(t.licence == "AIKYAM-OWNED" for t in tracks)
    assert all(t.duration >= 60 for t in tracks)                                                     # longer than the 60 s reel cap: never loops
    for reasons, kw in ((["aarti"], {"ritualId": "ritual_12"}), (["procession"], {"festivalId": "festival_13"}), (["deity", "idol"], {})):
        assert music.choose(plan(reasons, **kw), audio_with(), tracks)[0] is not None


# ------------------------------------------------------------------ rendering
def test_reel_gets_licensed_music_mixed_under_original(tmp_path, monkeypatch):
    import os
    from aikyam_video.render import render_format
    from tests.test_features import moving_dot_video
    monkeypatch.setenv("MUSIC_LIBRARY_DIR", os.path.join(os.path.dirname(__file__), "..", "music"))
    src = str(tmp_path / "s.mp4"); moving_dot_video(src)                                             # original audio = 440 Hz tone
    tr = music.get_track("aikyam_tanpura_calm")
    plan_ = {"schemaVersion": 1, "source": {"videoId": "v", "path": src, "durationSeconds": 6}, "outputFormat": "REEL", "durationSeconds": 5, "aspectRatio": "9:16",
             "segments": [{"start": 0, "end": 5, "reason": "deity", "score": .9}], "captions": {"enabled": False, "language": "en"}, "overlays": {"template": "divine_moment"},
             "transitions": {"type": "cut", "durationSeconds": 0}, "audio": {"preserveOriginal": True, "normalize": True, "music": music.music_block(tr, "test", 0.6)}, "thumbnail": {}}
    out = str(tmp_path / "o.mp4"); render_format(plan_, src, out, "square", str(tmp_path))
    pcm = ff.extract_audio_pcm(out, 16000); assert abs(len(pcm) / 16000 - 5) < 0.3
    e = lambda a, b, f: (lambda sp, fr: sp[(fr > f - 20) & (fr < f + 20)].sum() / sp.sum())(np.abs(np.fft.rfft(pcm[int(a * 16000):int(b * 16000)] * np.hanning(int((b - a) * 16000)))), np.fft.rfftfreq(int((b - a) * 16000), 1 / 16000))
    assert e(1.5, 3.5, 440) > 0.05                                                                   # the ORIGINAL audio is still there
    assert e(1.5, 3.5, 260) + e(1.5, 3.5, 525) > 0.01                                                # ... and the tanpura is audible under it
    env = lambda a, b: np.sqrt((pcm[int(a * 16000):int(b * 16000)] ** 2).mean())
    assert env(0.0, 0.3) < env(1.5, 3.5) and env(4.6, 5.0) < env(1.5, 3.5)                           # music fades in at the start and out at the end


def test_publish_payload_lists_music_credit(tmp_path):
    from aikyam_video.jobs import JobStore
    from aikyam_video.publish import build_payload
    from aikyam_video.storage import LocalStorage
    st = JobStore(f"sqlite:///{tmp_path}/p.db"); sto = LocalStorage(str(tmp_path / "s"))
    st.create_media("m1", "t", "k", "temple_123")
    plan_ = {"source": {"templeId": "temple_123"}, "overlays": {"temple": "T", "ritual": "Aarti"}, "durationSeconds": 20, "captions": {"enabled": False},
             "audio": {"music": {"enabled": True, "trackId": "x", "title": "Song", "licence": "CC-BY-4.0", "attribution": "Jane, CC BY 4.0"}}}
    (tmp_path / "plan.json").write_text(json.dumps(plan_)); (tmp_path / "r.mp4").write_bytes(b"x")
    sto.upload(str(tmp_path / "plan.json"), "t/plan.json"); sto.upload(str(tmp_path / "r.mp4"), "t/reel.mp4")
    st.record_outputs("m1", {"edit_plan": "t/plan.json", "reel": "t/reel.mp4"})
    p = build_payload(st, sto, "m1")
    assert p["music"]["trackId"] == "x" and p["credits"] == ["Music: Song — Jane, CC BY 4.0"]


def test_user_music_needs_rights_confirmation_and_is_then_usable(tmp_path, monkeypatch):
    import shutil
    monkeypatch.delenv("MUSIC_LIBRARY_DIR", raising=False); monkeypatch.delenv("AIKYAM_ALLOWED_MUSIC_LICENCES", raising=False)
    f = tmp_path / "my song.mp3"; shutil.copy(next(iter(__import__("pathlib").Path("music").glob("*.mp3"))), f)
    with pytest.raises(music.LicenceError, match="rights"): music.register_user_track(str(f), str(tmp_path / "o"), False)
    tid = music.register_user_track(str(f), str(tmp_path / "o"), True)
    t = music.get_track(tid); assert t.licence == "USER-ATTESTED" and t.path.endswith(".mp3")
    with pytest.raises(FileNotFoundError): music.register_user_track(str(tmp_path / "x.txt"), str(tmp_path / "o"), True)


# ---------------------------------------------------------------- Beat This! integration (the tracker itself runs in .venv-beat)
def test_beat_this_grid_replaces_the_autocorrelation_tempo(monkeypatch, tmp_path):
    from aikyam_video.creative import beats, musicdna as D
    f = tmp_path / "t.wav"; subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=200:d=12", str(f)], check=True)
    monkeypatch.setattr(beats, "python_exe", lambda: "x")
    b = [0.1 + 0.92 * i for i in range(13)]                                                       # 65.2 BPM, perfectly steady
    monkeypatch.setattr(beats, "track", lambda p, c=True: {"beats": b, "downbeats": b[::4]})
    a = D.analyze_track(str(f), use_cache=False)
    assert a.tracker == "beat_this" and a.kind == "rhythmic" and abs(a.bpm - 60 / 0.92) < 0.01 and a.beats == b and a.downbeats == b[::4]


def test_scattered_beat_this_output_means_beatless_not_a_grid(monkeypatch, tmp_path):
    from aikyam_video.creative import beats, musicdna as D
    f = tmp_path / "t.wav"; subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=200:d=12", str(f)], check=True)
    monkeypatch.setattr(beats, "python_exe", lambda: "x")
    rng = np.random.RandomState(0); b = np.cumsum(0.4 + rng.rand(20) * 1.1).tolist()               # intervals vary a lot: a drone / free flute
    monkeypatch.setattr(beats, "track", lambda p, c=True: {"beats": b, "downbeats": []})
    a = D.analyze_track(str(f), use_cache=False); assert a.bpm is None and a.beats == [] and a.kind != "rhythmic" and a.tracker == "beat_this"


def test_without_beat_this_the_builtin_tracker_is_used(monkeypatch, tmp_path):
    from aikyam_video.creative import beats, musicdna as D
    f = tmp_path / "t.wav"; subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=200:d=12", str(f)], check=True)
    monkeypatch.setattr(beats, "python_exe", lambda: None)
    assert D.analyze_track(str(f), use_cache=False).tracker == "autocorr"


def test_real_beat_this_on_the_generated_104_bpm_track():
    from pathlib import Path
    from aikyam_video.creative import beats
    if beats.python_exe() is None: pytest.skip("no .venv-beat (Beat This!)")
    d = beats.track("music/aikyam_festive_rhythm.mp3"); iv = np.diff(d["beats"])
    assert abs(60 / np.median(iv) - 104) < 1.5 and iv.std() / np.median(iv) < 0.05           # tools/make_music.py renders 104 BPM
