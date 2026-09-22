"""Story planner / pacing / transitions on synthetic shot pools (no video): determinism, role logic, redundancy, multi-reel, edge decisions."""
import numpy as np, pytest
from aikyam_video.creative import pacing, roles, story, transitions
from aikyam_video.creative.model import Shot, SlotFeatures, Clip
from aikyam_video.creative.shots import Population


def slots(n=12, sharp=100, luma=0.5, motion=5, rms=-30, presence=0.0, conc=0.4, face=0.0, jitter=0.5, contrast=0.5, clipped=0.0):
    f = lambda v: [float(v)] * n if np.isscalar(v) else list(np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(v)), v))   # lists are resampled to n slots
    return SlotFeatures(0.5, f(sharp), f(jitter), f(motion), f(luma), f(contrast), f(clipped), f(conc), f(face), f(rms), f(presence))


def shot(i, start, length, labels, emb=None, score=0.5, events=None, **kw):
    n = int(length / 0.5)
    return Shot(f"m{i}", start, start + length, f"m{i}", score, labels, [], emb, slots(n, **kw), events or {})


def e(*v):
    v = np.array(v, float); return (v / np.linalg.norm(v)).tolist()


def temple_pool():
    """A believable 5-minute recording: wide gopuram, priest preparing, aarti flame, deity close-up, crowd bhajan, calm ending."""
    return [
        shot(0, 5, 14, {"temple_architecture": 0.9, "crowd": 0.4}, e(1, 0, 0, 0), 0.45, motion=2, rms=-40, conc=0.2, sharp=200),
        shot(1, 40, 16, {"priest": 0.8, "lamps": 0.6, "devotees": 0.4}, e(0, 1, 0, 0), 0.5, motion=6, rms=-32, conc=0.5),
        shot(2, 90, 18, {"aarti": 0.95, "lamps": 0.8, "priest": 0.6}, e(0, 0, 1, 0), 0.6, motion=[3, 5, 8, 12, 16, 20] * 3, rms=-22, events={"bell": 0.7}, conc=0.6),
        shot(3, 140, 14, {"deity": 0.95, "idol": 0.9}, e(0, 0, 0, 1), 0.55, motion=1, rms=-36, conc=0.8, face=0.05, sharp=300),
        shot(4, 190, 20, {"devotees": 0.9, "aarti": 0.3}, e(0.3, 0.2, 0.9, 0.1), 0.4, motion=7, rms=-25, presence=0.8, events={"bhajan": 0.8}),
        shot(5, 250, 12, {"deity": 0.6, "devotees": 0.6, "decorations": 0.5}, e(0.5, 0.1, 0.1, 0.8), 0.42, motion=1, rms=-42, conc=0.25),
    ]


def plan(pool, prof=None, dur=300.0, **kw):
    pop = Population(pool); return story.plan_story(pool, pop, prof or pacing.profile("devotional"), dur, **kw), pop


def test_story_is_deterministic():
    a, _ = plan(temple_pool()); b, _ = plan(temple_pool())
    assert [(c.role, c.shot.id, c.start, c.end) for c in a.clips] == [(c.role, c.shot.id, c.start, c.end) for c in b.clips] and a.score == b.score


def test_story_follows_the_arc_and_serves_each_role_with_the_right_kind_of_shot():
    # KNOWN FAILING as of 2026-09-22 (left red on purpose, not silently patched -- see EXP-002b/EXP-003
    # in docs/research/experiment_matrix.md for the full trace before touching this):
    # temple_pool() is a deliberately thin pool -- exactly 6 shots for 6 roles, no slack. The EXP-002
    # quality gate correctly finds the true highest-scoring sequence is a SHORTER one (4 clips: OPENING,
    # BUILDUP, RITUAL, REVEAL=m3 -- CLIMAX and CLOSING legitimately skip, nothing good remains for them).
    # But a separate, pre-existing filter a few lines below in story.py (`n_lo <= len(clips) <= n_hi`, a
    # nominal clip-count band) discards that sequence as too short and falls back to a worse, longer one,
    # which reassigns REVEAL to m5 instead. Dropping n_lo fixes REVEAL but then CLIMAX goes unfilled
    # instead (verified: same class of cascade, different assertion). Two attempts, two different
    # cascades -- reverted both rather than force a third. Not fixed; tracked here.
    import dataclasses
    tl, _ = plan(temple_pool(), prof=dataclasses.replace(pacing.profile("devotional"), opening="establish"))    # the establishing opening (hook-first is tested in test_opening_order)
    roles_ = [c.role for c in tl.clips]
    from aikyam_video.creative.model import ROLES
    assert roles_ == sorted(roles_, key=ROLES.index) and len(set(roles_)) == len(roles_)        # arc order, no role twice
    by = {c.role: c.shot.id for c in tl.clips}
    assert by.get("REVEAL") == "m3"                                                              # the deity close-up is the reveal
    assert by.get("CLIMAX") in ("m2", "m4")                                                      # the peak of action + bell/bhajan
    if "OPENING" in by: assert by["OPENING"] in ("m0", "m5")                                     # establishing = the wide temple shot, not the aarti
    assert 3 <= len(tl.clips) <= 7


def test_durations_hit_the_target_and_respect_profile_bounds():
    for name in ("contemplative", "devotional", "festive"):
        prof = pacing.profile(name); tl, _ = plan(temple_pool(), prof)
        total = sum(c.length for c in tl.clips)
        assert prof.target_min - 8 <= total <= prof.target_max + 4, (name, total)
        assert all(prof.min_shot - 0.6 <= c.length <= prof.max_shot + 0.6 or c.length == c.shot.length for c in tl.clips), name
        assert all(c.shot.start <= c.start < c.end <= c.shot.end for c in tl.clips)


def test_festive_cuts_faster_than_contemplative():
    fest, _ = plan(temple_pool(), pacing.profile("festive")); calm, _ = plan(temple_pool(), pacing.profile("contemplative"))
    assert np.mean([c.length for c in fest.clips]) < np.mean([c.length for c in calm.clips])


def test_redundant_shots_are_not_both_used():
    pool = temple_pool() + [shot(6, 270, 14, {"deity": 0.95, "idol": 0.9}, e(0, 0, 0, 1.0001), 0.56, motion=1, conc=0.8, face=0.05, sharp=300)]   # a clone of the deity shot
    tl, _ = plan(pool)
    assert not ({"m3", "m6"} <= {c.shot.id for c in tl.clips})


def test_pacing_profile_follows_content():
    fest = [shot(0, 0, 10, {"procession": 0.95, "crowd": 0.7}, events={"drums": 0.8}), shot(1, 20, 10, {"ritual_dance": 0.9}, events={"drums": 0.6})]
    calm = [shot(0, 0, 10, {"deity": 0.95, "idol": 0.9}, events={"chant": 0.8}), shot(1, 20, 10, {"abhishekam": 0.9, "priest": 0.7}, events={"bell": 0.5})]
    assert pacing.choose_profile(fest).name == "festive" and pacing.choose_profile(calm).name == "contemplative"
    assert pacing.choose_profile(temple_pool()).name in ("devotional", "contemplative")
    assert pacing.choose_profile(fest, override="contemplative").name == "contemplative"
    p = pacing.profile("devotional")
    assert pacing.shot_length(p, 0.9) < pacing.shot_length(p, 0.1) and p.min_shot <= pacing.shot_length(p, 0.0) <= p.max_shot


def test_multiple_reels_from_one_source_use_disjoint_footage_and_share_it_fairly():
    pool = temple_pool() + [shot(6, 20, 14, {"aarti": 0.9, "priest": 0.7}, e(0.1, 0.9, 0.3, 0), 0.5, motion=9), shot(7, 120, 14, {"deity": 0.8, "flowers": 0.6}, e(0, 0.2, 0.2, 0.9), 0.5, motion=2)]
    pop = Population(pool); reels = story.plan_stories(pool, pop, pacing.profile("devotional"), 300.0, k=2)
    assert len(reels) == 2
    used = [{c.shot.id for c in r.clips} for r in reels]
    assert used[0].isdisjoint(used[1])                                                       # no footage reused across reels
    assert min(len(u) for u in used) >= 2 and max(len(u) for u in used) <= len(pool) // 2   # the first reel does not eat the pool
    assert len(story.plan_stories(pool, pop, pacing.profile("devotional"), 300.0, k=1)) == 1


def test_tiny_or_empty_pools_degrade_gracefully():
    assert plan([])[0].clips == []
    one, _ = plan([temple_pool()[2]]); assert len(one.clips) == 1 and one.clips[0].length > 1
    two, _ = plan(temple_pool()[:2]); assert 1 <= len(two.clips) <= 4


# ------------------------------------------------------------------ transitions
def clip_pair(sa, sb):
    a = Clip(sa, sa.start, sa.end, "RITUAL"); b = Clip(sb, sb.start, sb.end, "REVEAL"); return a, b


def edge(sa, sb, prof="devotional"):
    pop = Population([sa, sb]); a, b = clip_pair(sa, sb); sig = transitions.edge_signals(a, b, pop)
    return transitions.decide(a, b, sig, pacing.profile(prof), sim_floor=0.9), sig


def test_near_identical_shots_get_a_cut_not_a_ghosting_dissolve():
    a = shot(0, 0, 8, {"aarti": 0.9}, e(1, 0, 0, 0), luma=0.5); b = shot(1, 20, 8, {"aarti": 0.9}, e(1, 0.01, 0, 0), luma=0.5)
    t, _ = edge(a, b); assert t["type"] == "cut" and "near-identical" in t["reason"]


def test_big_exposure_jump_dips_through_black():
    a = shot(0, 0, 8, {"aarti": 0.9}, e(1, 0, 0, 0), luma=0.12); b = shot(1, 20, 8, {"deity": 0.9}, e(0, 1, 0, 0), luma=0.62)
    t, sig = edge(a, b); assert sig["luma_gap"] >= 0.35 and t["type"] == "dip_black" and 0.3 <= t["durationSeconds"] <= 0.7


def test_calm_different_shots_dissolve_and_energetic_festive_shots_cut():
    calm_a = shot(0, 0, 8, {"deity": 0.9}, e(1, 0, 0, 0), motion=1, rms=-45); calm_b = shot(1, 20, 8, {"devotees": 0.9}, e(0, 1, 0, 0), motion=1, rms=-45)
    t, _ = edge(calm_a, calm_b, "contemplative"); assert t["type"] == "crossfade" and 0.25 <= t["durationSeconds"] <= 0.9
    hot_a = shot(0, 0, 8, {"procession": 0.9}, e(1, 0, 0, 0), motion=30, rms=-15); hot_b = shot(1, 20, 8, {"ritual_dance": 0.9}, e(0, 1, 0, 0), motion=30, rms=-15)
    hp = Population([hot_a, hot_b, calm_a]); a, b = clip_pair(hot_a, hot_b); sig = transitions.edge_signals(a, b, hp)
    assert transitions.decide(a, b, sig, pacing.profile("festive"), 0.9)["type"] == "cut"


def test_transition_never_eats_a_short_clip_and_snaps_to_the_beat():
    a = shot(0, 0, 3, {"deity": 0.9}, e(1, 0, 0, 0), motion=1); b = shot(1, 20, 3, {"devotees": 0.9}, e(0, 1, 0, 0), motion=1)
    t, _ = edge(a, b, "contemplative"); assert t["durationSeconds"] <= 3 / 3 + 1e-9
    a2 = shot(0, 0, 9, {"deity": 0.9}, e(1, 0, 0, 0), motion=1); b2 = shot(1, 20, 9, {"devotees": 0.9}, e(0, 1, 0, 0), motion=1)
    pop = Population([a2, b2]); x, y = clip_pair(a2, b2); sig = transitions.edge_signals(x, y, pop)
    beat = 0.6; d = transitions.decide(x, y, sig, pacing.profile("devotional"), 0.9, beat)["durationSeconds"]
    assert abs(d / (beat / 2) - round(d / (beat / 2))) < 1e-6 or d in (0.25, 0.9)                     # a whole number of half-beats (or a clamp)


def test_plan_transitions_sets_every_edge_and_audio_leads():
    tl, pop = plan(temple_pool()); transitions.plan_transitions(tl, pop, pacing.profile("devotional"), source_duration=300.0)
    assert tl.clips[0].transition_in is None and all(c.transition_in and c.transition_in["reason"] for c in tl.clips[1:])
    assert sum(d["type"] == "transition" for d in tl.decisions) == len(tl.clips) - 1
    bhajan_in = [c for c in tl.clips[1:] if c.shot.id == "m4"]
    for c in bhajan_in: assert c.audio_lead > 0                                                       # sound arrives before the picture (J-cut)


def test_thin_pools_never_fail_while_moments_exist():
    """Regression (found on a single-scene 74 s clip): with few shots the beam used to run out of options and return NO story."""
    for n in (1, 2, 3):
        pool = temple_pool()[:n]; tl, _ = plan(pool)
        assert len(tl.clips) >= 1, n
    # one shot that clears none of the role floors (all affinities tiny): still yields a one-clip story
    weak = [shot(0, 10, 12, {"crowd": 0.1}, e(1, 0, 0, 0), 0.3, motion=1, rms=-60, conc=0.1)]
    tl, _ = plan(weak); assert len(tl.clips) == 1 and tl.clips[0].length >= 1.5
    two_same = [shot(0, 10, 12, {"devotees": 0.9}, e(1, 0, 0, 0), 0.5), shot(1, 40, 12, {"devotees": 0.9}, e(1, 0, 0, 0.0001), 0.5)]     # near-duplicates: at most one survives
    tl, _ = plan(two_same); assert 1 <= len(tl.clips) <= 4 and len({c.shot.id for c in tl.clips}) == len(tl.clips) and all(not story._clash(a.shot, b.shot) for i, a in enumerate(tl.clips) for b in tl.clips[i + 1:])     # a reel under 15 s is topped up from unused, non-overlapping footage
