"""Hook-first opening (default, configurable) and a person-fixed story order."""
import dataclasses
import pytest
from aikyam_video.creative import pacing, story
from aikyam_video.creative.shots import Population
from tests.test_creative_story import plan, temple_pool
from tests.test_multi_engine import mixctx  # noqa: F401  (fixture)


def multi_pool():
    pool = temple_pool()
    for i, s in enumerate(pool): s.asset_id = f"v{i % 3 + 1}"; s.id = f"v{i % 3 + 1}_{s.id}"; s.position = 0.5
    return pool


def test_every_profile_opens_with_a_hook_by_default():
    assert all(pacing.profile(n).opening == "hook" for n in ("contemplative", "devotional", "festive"))


def test_hook_opening_is_short_and_is_the_strongest_close_sharp_shot():
    tl, _ = plan(temple_pool())
    first = tl.clips[0]
    if first.role == "OPENING": assert first.length <= story.HOOK_MAX + 1e-6
    est = dataclasses.replace(pacing.profile("devotional"), opening="establish"); tl2, _ = plan(temple_pool(), prof=est)
    assert [(c.shot.id) for c in tl.clips] != [(c.shot.id) for c in tl2.clips] or tl.clips[0].length != tl2.clips[0].length      # the setting changes the edit


def test_opening_role_prefers_quality_and_focus_when_hook_and_setting_when_establish():
    pool = temple_pool(); pop = Population(pool)
    from aikyam_video.creative.roles import affinities
    hook = max(pool, key=lambda s: affinities(s, pop, 0.5, "hook")["OPENING"]); est = max(pool, key=lambda s: affinities(s, pop, 0.5, "establish")["OPENING"])
    assert est.id == "m0" and hook.id != "m0"                                                       # the wide gopuram establishes; a focused, sharp shot hooks


def test_forced_order_is_followed_with_roles_spread_along_the_arc():
    pool = multi_pool(); tl, _ = plan(pool, forced_order=["v3", "v1", "v2"])
    assert [c.shot.asset_id for c in tl.clips] == ["v3", "v1", "v2"]
    assert [c.role for c in tl.clips] == ["OPENING", "RITUAL", "CLOSING"] and any(d["type"] == "order" for d in tl.decisions)


def test_forced_order_picks_the_best_shot_of_each_asset_and_can_reuse_an_asset_once_per_shot():
    pool = multi_pool(); tl, _ = plan(pool, forced_order=["v1", "v1"])
    assert len({c.shot.id for c in tl.clips}) == 2                                                  # the second v1 slot gets a different shot
    with pytest.raises(ValueError, match="nothing usable"): plan(pool, forced_order=["v1", "v1", "v1"])   # v1 only has two shots


def test_engine_follows_a_forced_order_and_rejects_bad_ones(mixctx):
    import copy
    from aikyam_video.creative import engine as E
    ctx, _ = mixctx
    def go(**kw):
        c = copy.copy(ctx); c.o = copy.copy(ctx.o)
        for k, v in kw.items(): setattr(c.o, k, v)
        return E.build_plans(c)[0]
    p = go(order=["v2", "i1", "v1"]); assert [s["assetId"] for s in p["segments"]] == ["v2", "i1", "v1"] and p["segments"][1]["kind"] == "image"
    with pytest.raises(ValueError, match="unknown asset"): go(order=["v9"])
    with pytest.raises(ValueError, match="ONE reel"): go(order=["v1"], reels=2)
    est = go(opening="establish"); hook = go(); assert est["creative"]["decisions"][0]["profile"] == hook["creative"]["decisions"][0]["profile"]


def test_the_strongest_hook_shot_always_opens_the_reel_and_is_not_reused():
    from aikyam_video.creative.roles import affinities
    pool = temple_pool(); pop = Population(pool); prof = pacing.profile("devotional")
    assert prof.opening == "hook"
    tl, _ = plan(pool, prof=prof, hook_first=True)
    first = tl.clips[0]; assert first.role == "OPENING" and first.length <= story.HOOK_MAX + 1e-6
    assert any(d["type"] == "hook" and d["shot"] == first.shot.id for d in tl.decisions)
    assert sum(c.shot.id == first.shot.id for c in tl.clips) == 1                                       # the hook footage is not shown a second time
    # it is the shot with the best hook value (quality, subject focus, closeness, energy), whatever the rest of the story wants
    best = max(pool, key=lambda s: affinities(s, pop, 0.5, "hook")["OPENING"]); assert best.id in {c.shot.id for c in tl.clips}


def test_establishing_openings_forced_orders_and_directors_are_not_hook_forced():
    est = dataclasses.replace(pacing.profile("devotional"), opening="establish"); tl, _ = plan(temple_pool(), prof=est, hook_first=True)
    assert not any(d["type"] == "hook" for d in tl.decisions)
    assert not any(d["type"] == "hook" for d in plan(temple_pool())[0].decisions)                       # off by default
    pool = multi_pool(); tl2, _ = plan(pool, forced_order=["v3", "v1"]); assert not any(d["type"] == "hook" for d in tl2.decisions)


def test_a_short_hook_does_not_swallow_the_rest_of_its_shot():
    """A 3.5 s hook out of a 20 s shot leaves footage the reel can still use (before/after the hook window), never overlapping it."""
    from aikyam_video.creative.shots import carve
    pool = temple_pool(); big = max(pool, key=lambda s: s.length); parts = carve(big, big.start + 5.0, big.start + 8.5)
    assert parts[0].id == big.id and abs(parts[0].length - 3.5) < 1e-6 and {p.id.split("~")[-1] for p in parts[1:]} == {"pre", "post"}
    spans = sorted((p.start, p.end) for p in parts); assert all(b1 <= a2 + 1e-6 for (_, b1), (a2, _) in zip(spans, spans[1:]))               # disjoint
    assert abs(sum(b - a for a, b in spans) - big.length) < 1e-6 and all(len(p.slots.motion) >= 1 for p in parts)
    tl, _ = plan(pool, prof=pacing.profile("devotional"), hook_first=True); first = tl.clips[0]
    assert first.role == "OPENING" and first.length <= story.HOOK_MAX + 1e-6 and sum(c.length for c in tl.clips) > first.length + 8
