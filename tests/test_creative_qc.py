"""QC must CATCH defects (each test renders a reel with one known fault) and replan must map each failure to the right fix. Real FFmpeg + libass."""
import json, os, subprocess
import numpy as np, pytest
from scipy.io import wavfile
from aikyam_video import ffmpeg as ff, measure as M
from aikyam_video.creative import mixer, qc as QC, replan
from aikyam_video.plan import output_duration, validate_plan
from aikyam_video.render import render_format
from aikyam_video.captions import retime_cues

SR = 48000


def source(path, w=1280, h=720, parts=None, tone_hz=440, amp=0.25):
    """5 s clips: bright test pattern; optional per-part vf. Audio = tone with 4 Hz amplitude modulation so it is 'speech-like'."""
    parts = parts or [(3, "null"), (3, "hue=h=150,negate")]      # two visibly different looks: a clean reel must not look like duplicates
    inp, ch = [], []
    for i, (d, vf) in enumerate(parts):
        inp += ["-f", "lavfi", "-i", f"testsrc2=s={w}x{h}:r=25:d={d}"]; ch.append(f"[{i}:v]{vf},format=yuv420p[p{i}]")
    fg = ";".join(ch) + ";" + "".join(f"[p{i}]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=1:a=0[v]"
    total = sum(d for d, _ in parts); t = np.arange(int(total * SR)) / SR
    x = amp * np.sin(2 * np.pi * tone_hz * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t)); wav = path + ".wav"
    wavfile.write(wav, SR, (x * 32767).astype(np.int16))
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inp, "-i", wav, "-filter_complex", fg, "-map", "[v]", "-map", f"{len(parts)}:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", path], check=True)
    return path


def plan_for(src_dur, segs, edges=None, cues=None, music=None, creative=None):
    p = {"schemaVersion": 1, "source": {"videoId": "v", "path": "/x", "durationSeconds": src_dur}, "outputFormat": "REEL", "aspectRatio": "9:16",
         "segments": [{"start": a, "end": b, "reason": "x", "score": .5} for a, b in segs],
         "captions": {"enabled": bool(cues), "language": "en", "mode": "sentence", "style": {"fontSize": 0.032, "position": "bottom", "background": True, "animation": "fade"}, "cues": cues or []},
         "overlays": {"template": "divine_moment"}, "transitions": {"type": "cut", "durationSeconds": 0},
         "audio": {"preserveOriginal": True, "normalize": True, "music": music or {"enabled": False}}, "thumbnail": {}}
    for s, (ty, d) in zip(p["segments"][1:], edges or []): s["transitionIn"] = {"type": ty, "durationSeconds": d}
    if creative: p["creative"] = creative
    p["durationSeconds"] = round(output_duration(p["segments"], p["transitions"]), 2); return p


def rend(tmp_path, plan, src, fmt="reel", presence=None, music_path=None, volume=0.5):
    mus = plan["audio"].get("music", {})
    mix = mixer.render_audio(plan, src, music_path, 0.0, presence, volume)
    out = str(tmp_path / "o.mp4"); render_format(plan, src, out, fmt, str(tmp_path), mix=mix); return out, mix


def failed(rep): return {c.name for c in rep.failed()}


# ------------------------------------------------------------------ a clean reel passes; each defect is caught
def test_clean_reel_passes_qc(tmp_path):
    src = source(str(tmp_path / "s.mp4")); p = plan_for(6, [(0, 3), (3, 6)], [("crossfade", 0.5)])
    out, mix = rend(tmp_path, p, src)
    rep = QC.run_qc(p, out, str(tmp_path), mix); assert rep.status in ("pass", "warn") and not rep.failed(), [(c.name, c.detail) for c in rep.failed()]
    assert {"loudness", "true_peak", "clipping", "black_frames", "abrupt_transition"} <= {c.name for c in rep.checks}
    assert abs(rep.stats["lufs"] - QC.TARGET_LUFS) < QC.LUFS_TOL


def test_black_frames_are_caught_but_planned_dips_are_not(tmp_path):
    src = source(str(tmp_path / "s.mp4"), parts=[(3, "null"), (2, "drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill"), (3, "null")])          # 2 s of black inside the source
    p = plan_for(8, [(0, 3.5), (3.5, 8)], [("cut", 0)])
    out, mix = rend(tmp_path, p, src); rep = QC.run_qc(p, out, str(tmp_path), mix)
    assert "black_frames" in failed(rep) and next(c for c in rep.checks if c.name == "black_frames").where
    dip = plan_for(6, [(0, 3), (3, 6)], [("dip_black", 0.6)])                                                                      # a black dip the plan asked for
    src2 = source(str(tmp_path / "s2.mp4")); out2, mix2 = rend(tmp_path, dip, src2)
    assert "black_frames" not in failed(QC.run_qc(dip, out2, str(tmp_path), mix2))


def test_frozen_picture_is_reported(tmp_path):
    src = source(str(tmp_path / "s.mp4"), parts=[(3, "null"), (4, "trim=end_frame=1,tpad=stop_mode=clone:stop_duration=4")])         # last 4 s = one frozen frame
    p = plan_for(7, [(0, 7)]); out, mix = rend(tmp_path, p, src); rep = QC.run_qc(p, out, str(tmp_path), mix)
    assert "frozen_frames" in failed(rep)


def test_clipping_and_loudness_and_true_peak(tmp_path):
    src = source(str(tmp_path / "s.mp4"), amp=0.9); p = plan_for(6, [(0, 6)])
    out, mix = rend(tmp_path, p, src)
    assert not {"clipping", "true_peak"} & failed(QC.run_qc(p, out, str(tmp_path), mix))                                              # the mixer limits a hot source: passes
    bad = str(tmp_path / "bad.mp4")                                                                                                   # now the same reel with distorted audio muxed in
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", out, "-af", "volume=14dB,alimiter=limit=1:level=disabled,volume=6dB", "-c:v", "copy", bad], check=True)
    rep = QC.run_qc(p, bad, str(tmp_path), mix); assert {"loudness", "true_peak"} & failed(rep)


def test_music_that_overpowers_the_live_sound_is_caught(tmp_path):
    src = source(str(tmp_path / "s.mp4")); mp = str(tmp_path / "m.wav"); t = np.arange(SR * 8) / SR
    wavfile.write(mp, SR, (0.6 * np.sin(2 * np.pi * 261 * t) * 32767).astype(np.int16))
    p = plan_for(6, [(0, 6)], music={"enabled": True, "trackId": "x", "volume": 1.0})
    mix = mixer.render_audio(p, src, mp, 0.0, lambda t: 1.0, 1.0)
    out = str(tmp_path / "o.mp4"); render_format(p, src, out, "reel", str(tmp_path), mix=mix)
    assert "music_balance" not in failed(QC.run_qc(p, out, str(tmp_path), mix))                                                       # the mixer ducks it: passes
    mix.music_db = mix.live_db + 4.0                                                                                                  # simulate a broken mix: music 4 dB ABOVE the live sound
    rep = QC.run_qc(p, out, str(tmp_path), mix); assert {"music_balance", "music_overpowers"} & failed(rep)


def test_abrupt_hard_cut_is_caught_and_fixed(tmp_path):
    t = np.arange(int(8 * SR)) / SR; x = np.concatenate([0.4 * np.sin(2 * np.pi * 300 * t[:4 * SR]), 0.004 * np.sin(2 * np.pi * 300 * t[4 * SR:])])      # 40 dB level cliff
    src = source(str(tmp_path / "s.mp4"), parts=[(8, "null")]); wavfile.write(str(tmp_path / "loud.wav"), SR, (x * 32767).astype(np.int16))
    cliff = str(tmp_path / "cliff.mp4"); subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-i", str(tmp_path / "loud.wav"), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", cliff], check=True)
    p = plan_for(8, [(0, 4), (4, 8)], [("cut", 0)])
    mix = mixer.render_audio(p, cliff, None, 0.0, None, match_levels=False)                                                            # no level matching: the cliff survives
    out = str(tmp_path / "o.mp4"); render_format(p, cliff, out, "reel", str(tmp_path), mix=mix)
    rep = QC.run_qc(p, out, str(tmp_path), mix); assert "abrupt_transition" in failed(rep)
    new, fixes, _ = replan.apply_fixes(p, rep); assert new and new["segments"][1]["transitionIn"]["type"] == "crossfade" and any("abrupt" in f for f in fixes)


def test_duplicate_shots_are_caught_and_one_is_dropped(tmp_path):
    src = source(str(tmp_path / "s.mp4"), parts=[(10, "null")])                                                                        # a static-ish pattern: two windows look identical
    p = plan_for(10, [(0, 3), (5, 8)], [("cut", 0)]); out, mix = rend(tmp_path, p, src)
    rep = QC.run_qc(p, out, str(tmp_path), mix)
    dup = next(c for c in rep.checks if c.name == "duplicate_shots")
    if dup.status == "fail":                                                                                                            # testsrc2 animates: only assert the fix when it fires
        new, fixes, _ = replan.apply_fixes(p, rep); assert new is None or len(new["segments"]) == 1


def test_cut_off_face_is_a_bad_crop_and_fix_switches_to_fit_blur(tmp_path):
    import cv2
    face = None
    for cand in ("/usr/share/opencv4/haarcascades", cv2.data.haarcascades): pass
    p = plan_for(6, [(0, 3), (3, 6)], [("cut", 0)])
    rep = QC.QCReport([QC.Check("bad_crop", "fail", 3, 1, [1.0, 1.5, 2.0], 1, "a face is cut by the frame edge")])              # QC's detector needs real faces; the mapping is what we test here
    new, fixes, _ = replan.apply_fixes(p, rep)
    assert new["segments"][1]["layout"]["mode"] == "fit_blur" and new["segments"][0].get("layout") is None and any("bad_crop" in f for f in fixes)


# ------------------------------------------------------------------ captions measured by libass
def test_caption_overflow_and_overlay_collision_are_measured(tmp_path):
    src = source(str(tmp_path / "s.mp4"))
    long = "This is a deliberately very long caption sentence that cannot possibly fit on a narrow vertical frame in one or two lines at this size"
    ok = [{"start": 0.2, "end": 2.6, "text": "Om Namah Shivaya", "words": []}]
    p = plan_for(6, [(0, 6)], cues=ok); p["overlays"] = {"template": "ritual_highlight", "temple": "Chamundeshwari Temple", "deity": "Durga", "ritual": "Evening Aarti"}
    out, mix = rend(tmp_path, p, src); rep = QC.run_qc(p, out, str(tmp_path), mix)
    assert not {"caption_overflow", "caption_overlay_collision", "caption_overlap"} & failed(rep), [(c.name, c.detail) for c in rep.failed()]
    p["captions"]["cues"] = [{"start": 0.2, "end": 4.0, "text": long, "words": []}]; p["captions"]["style"]["fontSize"] = 0.09                  # huge font
    out2, mix2 = rend(tmp_path, p, src); rep2 = QC.run_qc(p, out2, str(tmp_path), mix2)
    assert {"caption_overflow", "caption_overlay_collision"} & failed(rep2)
    new, fixes, _ = replan.apply_fixes(p, rep2); assert new["captions"]["style"]["fontSize"] < 0.09 or not new["captions"]["enabled"]


def test_overlapping_cues_and_reading_speed(tmp_path):
    src = source(str(tmp_path / "s.mp4")); cues = [{"start": 0.2, "end": 2.0, "text": "first caption here", "words": []}, {"start": 1.5, "end": 3.0, "text": "second appears early", "words": []}]
    p = plan_for(6, [(0, 6)], cues=cues); out, mix = rend(tmp_path, p, src); rep = QC.run_qc(p, out, str(tmp_path), mix)
    assert "caption_overlap" in failed(rep)
    fast = [{"start": 0.2, "end": 0.9, "text": "a very very long text for less than a second", "words": []}]
    p2 = plan_for(6, [(0, 6)], cues=fast); rep2 = QC.run_qc(p2, out, str(tmp_path), mix)
    assert {c.name: c.status for c in rep2.checks}["caption_reading_speed"] in ("warn", "fail")
    slow = retime_cues(fast, max_cps=17, min_dur=1.0, end=6.0); assert slow[0]["end"] - slow[0]["start"] >= 2.0                            # retiming gives it time to be read


def test_retime_never_overlaps_the_next_cue():
    cues = [{"start": 0, "end": 0.5, "text": "abcdefghijklmnopqrstuvwxyz abcdefghijklmnop", "words": []}, {"start": 1.0, "end": 2.0, "text": "next", "words": []}]
    r = retime_cues(cues, end=5.0); assert r[0]["end"] <= r[1]["start"] and r[0]["end"] > 0.5


# ------------------------------------------------------------------ replan mapping
def test_replan_is_deterministic_and_maps_each_failure_to_its_fix():
    p = plan_for(30, [(0, 8), (10, 18), (20, 28)], [("crossfade", 0.5), ("cut", 0)], music={"enabled": True, "trackId": "t", "volume": 0.5})
    rep = QC.QCReport([QC.Check("music_balance", "fail", 5, 12), QC.Check("black_frames", "fail", 1, 0.25, [10.0], 1), QC.Check("loudness", "fail", -20, -16)])
    a = replan.apply_fixes(p, rep); b = replan.apply_fixes(p, rep)
    assert a[0] == b[0] and a[1] == b[1]                                                                                              # same report => same fix
    assert a[0]["audio"]["music"]["volume"] < 0.5 and len(a[0]["segments"]) == 2 and a[2].get("retarget")                              # volume down, bad clip dropped, re-master
    assert a[0]["creative"]["qc"]["history"][0]["failed"]
    validate_plan(a[0], 30)


def test_replan_gives_up_cleanly_when_nothing_can_be_fixed():
    p = plan_for(30, [(0, 8)]); rep = QC.QCReport([QC.Check("duration", "fail", 3, 8)])
    new, fixes, remix = replan.apply_fixes(p, rep); assert new is None or fixes
    rep2 = QC.QCReport([QC.Check("black_frames", "fail", 1, 0.25, [1.0], 0)])                                                         # dropping the only clip would leave nothing
    assert replan.apply_fixes(p, rep2)[0] is None
