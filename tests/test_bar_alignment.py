"""Bar-level (dynamic-programming) cut alignment vs the earlier greedy nudge: on-beat and on-downbeat share, shot-length deviation, feasibility, determinism."""
import numpy as np, pytest
from aikyam_video.creative.model import Clip, Timeline
from aikyam_video.creative.musicdna import MusicAnalysis
from aikyam_video.creative import music_sync as MS
from tests.test_creative_story import shot

BEAT = 0.6                                                                          # 100 BPM


def music(bpm=100, dur=80, bar=4, jitter=0.0):
    b = 60.0 / bpm; beats = [i * b for i in range(int(dur / b))]
    return MusicAnalysis(dur, bpm, 9.0, beats, [0.5] * int(dur / 0.5), "rhythmic", 2.0, beats[::bar], "beat_this")


def timeline(lengths, roles=None, shot_len=20.0):
    clips = []
    for i, L in enumerate(lengths):
        s = shot(i, i * 30, shot_len, {"aarti": 0.8}); clips.append(Clip(s, s.start, s.start + L, (roles or ["RITUAL"] * len(lengths))[i]))
    return Timeline(clips, "devotional", [c.role for c in clips])


def edges(tl, ov):
    t, out = 0.0, []
    for i, c in enumerate(tl.clips[:-1]): t += c.length - ov[i]; out.append(t + ov[i] / 2)
    return out


def share(ts, grid, tol=0.04): return float(np.mean([np.min(np.abs(np.asarray(grid) - t)) <= tol for t in ts]))


def run(fn, lengths, ov, roles=None, an=None):
    tl = timeline(lengths, roles); an = an or music(); rep = fn(tl, an, 0.0, lambda t: list(ov)); return tl, rep, an


def test_dp_puts_every_cut_on_a_beat_when_the_footage_allows_it():
    L = [4.1, 5.3, 3.7, 4.9, 4.4]; ov = [0.0, 0.4, 0.0, 0.4]
    tl, rep, an = run(MS.align_cuts, L, ov)
    assert rep["method"] == "dp" and rep["on_beat_after"] == 1.0 and rep["on_beat_before"] < 1.0
    assert share(edges(tl, ov), an.beats) == 1.0


def test_key_moments_land_on_downbeats_and_bars_beat_plain_beats():
    L = [4.9, 5.1, 4.8, 5.0]; ov = [0.0, 0.0, 0.0]
    tl, rep, an = run(MS.align_cuts, L, ov, roles=["OPENING", "RITUAL", "REVEAL", "CLIMAX"])
    ts = edges(tl, ov); down = an.downbeats
    assert share([ts[1], ts[2]], down) == 1.0                                             # the edges into REVEAL and CLIMAX are on bar starts
    assert rep["on_downbeat_after"] >= 2 / 3


def test_clips_never_leave_their_shot_or_get_shorter_than_1_5_s_and_lengths_stay_close():
    rng = np.random.RandomState(3)
    for _ in range(25):
        L = list(np.round(rng.uniform(2.0, 9.0, 5), 2)); ov = [0.0, 0.5, 0.0, 0.4]
        tl, rep, _ = run(MS.align_cuts, L, ov)
        for c, l0 in zip(tl.clips, L):
            assert c.start >= c.shot.start - 1e-6 and c.end <= c.shot.end + 1e-6 and c.length >= min(1.5, c.shot.length) - 1e-6
            assert abs(c.length - l0) <= 1.25 * BEAT + 0.6 + 1e-6                        # bounded by the beat reach (+ overlap halves)


def test_dp_is_at_least_as_beat_accurate_as_greedy_and_better_on_bars_on_average():
    rng = np.random.RandomState(7); ds, gs, dbd, dbg, dl, gl = [], [], [], [], [], []
    for _ in range(60):
        L = list(np.round(rng.uniform(2.5, 8.0, 5), 2)); ov = [0.0, 0.5, 0.0, 0.4]; roles = ["OPENING", "BUILDUP", "REVEAL", "CLIMAX", "CLOSING"]
        a, ra, an = run(MS.align_cuts, L, ov, roles); b, rb, _ = run(MS.align_cuts_greedy, L, ov, roles)
        ds.append(ra["on_beat_after"]); gs.append(rb["on_beat_after"])
        dbd.append(share(edges(a, ov), an.downbeats, 0.04)); dbg.append(share(edges(b, ov), an.downbeats, 0.04))
        dl.append(np.mean([abs(c.length - l) / l for c, l in zip(a.clips, L)])); gl.append(np.mean([abs(c.length - l) / l for c, l in zip(b.clips, L)]))
    print(f"\non-beat  DP {np.mean(ds):.2f} greedy {np.mean(gs):.2f} | on-downbeat DP {np.mean(dbd):.2f} greedy {np.mean(dbg):.2f} | mean length change DP {np.mean(dl):.3f} greedy {np.mean(gl):.3f}")
    assert np.mean(ds) >= np.mean(gs) - 1e-9 and np.mean(dbd) > np.mean(dbg) + 0.15                     # clearly more cuts on bar starts
    assert np.mean(dl) < 0.08                                                                            # the price: shots change by < 8% on average (greedy skipped the hard edges: 1%)


def test_beatless_music_and_single_clip_are_left_alone_and_the_result_is_deterministic():
    L = [4.1, 5.3, 3.7]; ov = [0.0, 0.0]
    tl, rep, _ = run(MS.align_cuts, L, ov, an=MusicAnalysis(60, None, 0.0, [], [], "drone", 0.1))
    assert np.allclose([c.length for c in tl.clips], L) and rep["aligned"] == 0
    t1, r1, _ = run(MS.align_cuts, L, ov); t2, r2, _ = run(MS.align_cuts, L, ov); assert [c.end for c in t1.clips] == [c.end for c in t2.clips] and r1 == r2
    one, r, _ = run(MS.align_cuts, [5.0], []); assert r["aligned"] == 0


def test_offset_starts_on_a_bar_when_downbeats_exist():
    an = music(); tl = timeline([4, 5, 4]); e = np.linspace(0.2, 0.9, 26)
    off, _ = MS.pick_offset(an, e, None, 13.0); assert min(abs(off - d) for d in an.downbeats) < 1e-6


def test_a_hook_opening_is_not_stretched_past_its_cap_to_reach_a_beat():
    L = [3.3, 5.0, 5.0]; ov = [0.4, 0.4]
    tl, rep, an = run(MS.align_cuts, L, ov, roles=["OPENING", "RITUAL", "CLIMAX"]); free_first = tl.clips[0].length
    tl2 = timeline(L, ["OPENING", "RITUAL", "CLIMAX"]); MS.align_cuts(tl2, music(), 0.0, lambda t: list(ov), 3.5)
    assert tl2.clips[0].length <= 3.5 + 1e-6 and free_first >= tl2.clips[0].length - 1e-6


def test_alignment_never_pushes_a_clip_past_the_profiles_longest_shot():
    rng = np.random.RandomState(11)
    for _ in range(40):
        L = list(np.round(rng.uniform(6.0, 9.6, 4), 2)); ov = [0.4, 0.4, 0.4]; tl = timeline(L, shot_len=20.0); MS.align_cuts(tl, music(), 0.0, lambda t: list(ov), None, 9.8)
        assert all(c.length <= max(l, 9.8) + 1e-6 for c, l in zip(tl.clips, L))
