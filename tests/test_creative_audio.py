"""Music analysis + audio mixer on synthetic signals with known ground truth (levels, tempo, ducking margins, J/L-cuts, limiter)."""
import numpy as np, pytest
from scipy.io import wavfile
from aikyam_video import measure as M
from aikyam_video.creative import mixer, musicdna
from aikyam_video.creative.mixer import FRAME, SR

R = np.random.default_rng(3)


def wav(path, x, sr=SR):
    wavfile.write(path, sr, (np.clip(x, -1, 1) * 32767).astype(np.int16)); return str(path)


def tone(f, dur, amp=0.3, am=0.0):
    t = np.arange(int(dur * SR)) / SR; x = amp * np.sin(2 * np.pi * f * t)
    return x * (0.6 + 0.4 * np.sin(2 * np.pi * am * t)) if am else x


def stereo(x): return np.stack([x, x], 1)


def plan(segs, edges=None, dur=None):
    """segs: [(start, end)]; edges: [(type, seconds, lead)] for segments 1..n-1."""
    p = {"segments": [{"start": a, "end": b, "reason": "x", "score": .5} for a, b in segs], "transitions": {"type": "cut", "durationSeconds": 0}}
    for s, (ty, d, lead) in zip(p["segments"][1:], edges or [("cut", 0, 0)] * (len(segs) - 1)):
        s["transitionIn"] = {"type": ty, "durationSeconds": d}
        if lead: s["audioLead"] = lead
    from aikyam_video.plan import output_duration
    p["durationSeconds"] = round(output_duration(p["segments"], p["transitions"]), 3); return p


def db_at(x, t, w=0.2):
    i = int(t * SR); return M.rms_db(x[max(0, i - int(w * SR / 2)): i + int(w * SR / 2)])


# ------------------------------------------------------------------ tempo / beats
def clicks(bpm, dur=30, sr=22050):
    x = np.zeros(int(dur * sr)); step = 60 / bpm
    for k in np.arange(0.3, dur - 0.1, step):
        i = int(k * sr); n = int(0.05 * sr); x[i:i + n] += np.sin(2 * np.pi * 90 * np.arange(n) / sr) * np.exp(-np.arange(n) / (0.012 * sr))
    return x.astype(np.float32)


@pytest.mark.parametrize("bpm", [80, 100, 120, 140])
def test_tempo_and_beat_grid_recovered(bpm):
    a = musicdna.analyze_samples(clicks(bpm))
    assert a.bpm and a.kind == "rhythmic" and a.beat_conf >= musicdna.CONF
    assert min(abs(a.bpm - bpm), abs(a.bpm - bpm / 2)) <= 1.5                       # tempo, or its half (the octave ambiguity every tracker has)
    true = np.arange(0.3, 29.9, 60 / bpm)                                           # the real click times: what matters for editing is that beats land ON them
    hit = np.mean([np.min(np.abs(true - b)) <= 0.04 for b in a.beats]); assert hit >= 0.95 and len(a.beats) >= 0.45 * 30 * bpm / 60


def test_beatless_material_gets_no_beat_grid():
    noise = (R.normal(0, 0.05, 22050 * 30)).astype(np.float32); a = musicdna.analyze_samples(noise)
    assert a.bpm is None and a.beats == [] and a.kind != "rhythmic"
    drone = (0.3 * np.sin(2 * np.pi * 131 * np.arange(22050 * 30) / 22050)).astype(np.float32)
    assert musicdna.analyze_samples(drone).bpm is None


def test_starter_tracks_are_classified_sensibly():
    a = musicdna.analyze_track("music/aikyam_festive_rhythm.mp3", use_cache=False)
    assert a.kind == "rhythmic" and abs(a.bpm - 104) <= 2.5
    assert musicdna.analyze_track("music/aikyam_tanpura_calm.mp3", use_cache=False).bpm is None


# ------------------------------------------------------------------ live bed
def test_master_loudness_and_true_peak_meet_the_targets(tmp_path):
    src = wav(tmp_path / "s.wav", np.concatenate([tone(440, 6, 0.5, am=4), tone(660, 6, 0.05, am=4), tone(550, 6, 0.4, am=4)]))
    r = mixer.render_audio(plan([(0, 4), (6, 10), (12, 16)], [("crossfade", 0.5, 0), ("cut", 0, 0)]), src)
    assert abs(r.report["lufs"] - mixer.TARGET_LUFS) < 0.6 and r.report["true_peak_db"] <= mixer.CEILING_DB + 0.1 and r.report["clipped"] == 0


def test_clip_levels_are_matched_within_limits(tmp_path):
    src = wav(tmp_path / "s.wav", np.concatenate([tone(440, 6, 0.5, am=4), tone(440, 6, 0.04, am=4)]))         # second clip 22 dB quieter
    p = plan([(0, 4), (6, 10)], [("cut", 0, 0)]); before = abs(M.rms_db(np.array(tone(440, 4, 0.5))) - M.rms_db(np.array(tone(440, 4, 0.04))))
    r = mixer.render_audio(p, src, match_levels=True); gA, gB = db_at(r.pcm, 2.0, 1.0), db_at(r.pcm, 6.0, 1.0)
    assert abs(gA - gB) < before - 5.0                                                                       # gap shrinks by >= 5 dB
    assert all(abs(s["liveGainDb"]) <= 6.0 + 1e-6 for s in p["segments"])                                     # never more than +-6 dB


def test_hard_cut_has_no_click_and_crossfade_no_dropout(tmp_path):
    src = wav(tmp_path / "s.wav", np.concatenate([tone(440, 6, 0.3), tone(880, 6, 0.3)]))
    for edge, d in (("cut", 0), ("crossfade", 0.6)):
        r = mixer.render_audio(plan([(0, 4), (6, 10)], [(edge, d, 0)]), src, match_levels=False); t = 4.0 - d / 2
        assert np.abs(np.diff(r.pcm[:, 0])).max() < 0.25                                                      # no step discontinuity (click)
        assert db_at(r.pcm, t, 0.1) > db_at(r.pcm, 1.0, 0.5) - 3.5                                            # equal-power: no level dip at the edge


def test_j_cut_brings_the_next_sound_in_early_and_l_cut_lets_the_old_one_ring(tmp_path):
    src = wav(tmp_path / "s.wav", np.concatenate([tone(440, 6, 0.3), tone(880, 6, 0.3)]))
    def band(x, t, f, w=0.15):
        i = int(t * SR); seg = x[i - int(w * SR): i + int(w * SR), 0]; sp = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))); fr = np.fft.rfftfreq(len(seg), 1 / SR)
        return sp[(fr > f - 30) & (fr < f + 30)].sum() / (sp.sum() + 1e-9)
    j = mixer.render_audio(plan([(0, 4), (7, 11)], [("cut", 0, 0.3)]), src, match_levels=False)               # boundary at t = 4.0
    assert band(j.pcm, 3.85, 880) > 0.3 and band(j.pcm, 3.5, 880) < 0.05                                       # 880 Hz arrives ~0.3 s before the picture cut
    l = mixer.render_audio(plan([(0, 4), (7, 11)], [("cut", 0, -0.3)]), src, match_levels=False)
    assert band(l.pcm, 4.15, 440) > 0.3 and band(l.pcm, 4.5, 440) < 0.05                                       # 440 Hz lingers ~0.3 s into the next picture


# ------------------------------------------------------------------ music bed: ducking
def music_file(tmp_path, dur=20, f=261.0, amp=0.5): return wav(tmp_path / "m.wav", np.stack([tone(f, dur, amp)] * 2, 1)[:, 0])


def margins(r):
    live, mus = r.live_db, r.music_db; n = min(len(live), len(mus)); return live[:n] - mus[:n]


def test_music_sits_16_db_under_devotional_live_sound_and_8_db_under_ambience(tmp_path):
    src = wav(tmp_path / "s.wav", tone(300, 20, 0.3, am=4)); p = plan([(0, 10)]); mus = music_file(tmp_path)
    loud = mixer.render_audio(p, src, mus, presence_fn=lambda t: 1.0, volume=1.0, match_levels=False)
    settle = slice(int(1.5 / FRAME), int(9 / FRAME))                                                          # after the fade-in, before the fade-out
    assert np.percentile(margins(loud)[settle], 5) >= 15.0                                                     # chant/speech: >= 16 dB (15 with smoothing tolerance)
    amb = mixer.render_audio(p, src, mus, presence_fn=lambda t: 0.0, volume=1.0, match_levels=False)
    m = margins(amb)[settle]; assert np.percentile(m, 5) >= 7.0 and np.median(m) < 14.0                        # ambience-only: ~8 dB, music clearly present
    assert np.median(margins(amb)[settle]) < np.median(margins(loud)[settle])


def test_music_never_overpowers_at_any_volume_setting(tmp_path):
    src = wav(tmp_path / "s.wav", tone(300, 20, 0.3, am=4)); mus = music_file(tmp_path)
    for v in (0.1, 0.5, 1.0):
        r = mixer.render_audio(plan([(0, 10)]), src, mus, presence_fn=lambda t: 1.0, volume=v, match_levels=False)
        assert np.percentile(margins(r)[int(1.5 / FRAME):int(9 / FRAME)], 5) >= 15.0, v


def test_music_rises_in_silences_without_pumping(tmp_path):
    live = np.concatenate([tone(300, 4, 0.3, am=4), np.zeros(int(3 * SR)), tone(300, 4, 0.3, am=4)])
    src = wav(tmp_path / "s.wav", live); r = mixer.render_audio(plan([(0, 11)]), src, music_file(tmp_path, 15), presence_fn=lambda t: 1.0 if (t < 4 or t > 7) else 0.0, volume=1.0, match_levels=False)
    g = r.music_gain_db
    assert g[int(5.5 / FRAME)] > g[int(2.5 / FRAME)] + 5.0                                                     # the bed comes up when the live sound stops
    assert np.abs(np.diff(g)).max() < 12.0                                                                     # ... smoothly: no step > 12 dB per 50 ms frame (no pumping)


def test_stems_sum_to_the_mix_and_limiter_holds_the_ceiling(tmp_path):
    src = wav(tmp_path / "s.wav", np.concatenate([tone(300, 5, 0.9, am=3), tone(500, 5, 0.9)]))               # very hot source
    r = mixer.render_audio(plan([(0, 8)]), src, music_file(tmp_path, 10, amp=0.9), presence_fn=lambda t: 0.5, volume=1.0, match_levels=False)
    assert np.abs(r.pcm - (r.live + r.music)).max() < 1e-3                                                     # stems are exactly what was mixed
    assert r.report["true_peak_db"] <= mixer.CEILING_DB + 0.15 and np.isfinite(r.pcm).all() and np.abs(r.pcm).max() <= 1.0


def test_music_offset_selects_the_right_part_of_the_track(tmp_path):
    m = np.concatenate([tone(200, 6, 0.4), tone(1000, 6, 0.4)]); mus = wav(tmp_path / "m.wav", m)
    src = wav(tmp_path / "s.wav", np.zeros(int(6 * SR)) + R.normal(0, 1e-4, int(6 * SR)))
    r = mixer.render_audio(plan([(0, 4)]), src, mus, music_offset=6.0, volume=1.0, match_levels=False)
    x = r.music[int(1 * SR): int(3 * SR), 0]; sp = np.abs(np.fft.rfft(x)); fr = np.fft.rfftfreq(len(x), 1 / SR)
    assert abs(fr[np.argmax(sp)] - 1000) < 20                                                                  # started 6 s into the track => the 1 kHz half
