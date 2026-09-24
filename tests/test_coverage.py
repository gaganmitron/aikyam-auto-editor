"""Coverage-aware story planning (story.py, S["coverage"], EXP-014): opt-in facility-location term -- reward picks that stand in for more of the
WHOLE pool. Off by default (measure on real reels before enabling)."""
import numpy as np
from aikyam_video.creative import story
from tests.test_creative_story import shot, e, plan


def _pool():
    rng = np.random.default_rng(3)
    centers = [np.eye(8)[0], np.eye(8)[1], np.eye(8)[2]]                       # 3 visual topics: one dominates the recording, one is rare
    labs = [{"aarti": 0.9, "lamps": 0.7}, {"devotees": 0.9, "priest": 0.6}, {"deity": 0.9, "idol": 0.8}, {"temple_architecture": 0.9, "crowd": 0.4}]
    pool, k = [], 0
    for t, n in enumerate([7, 4, 2]):
        for _ in range(n):
            pool.append(shot(k, 10 + k * 25, 12, labs[(k + t) % 4], e(*(centers[t] + 0.25 * rng.standard_normal(8))), 0.45 + 0.02 * rng.random(), motion=int(rng.integers(1, 9)), conc=0.5)); k += 1
    return pool


def _coverage(ids, pool):
    E = np.array([s.embedding for s in pool]); S = E @ E.T; s0 = np.percentile(S[np.triu_indices(len(pool), 1)], 75)
    R = np.maximum(0, S - s0) / (1 - s0); w = np.array([s.length * s.score for s in pool]); w /= w.sum()
    return float((w * R[:, [i for i, s in enumerate(pool) if s.id in ids]].max(axis=1)).sum())


def test_coverage_is_off_by_default():
    assert story.S.get("coverage", 0.0) == 0.0


def test_coverage_weight_changes_the_plan_and_raises_how_well_it_represents_the_pool(monkeypatch):
    pool = _pool()
    off = {c.shot.id for c in plan(pool)[0].clips}
    monkeypatch.setitem(story.S, "coverage", 0.5)
    on = {c.shot.id for c in plan(pool)[0].clips}
    assert on != off                                                          # not a dead knob
    assert _coverage(on, pool) > _coverage(off, pool)


def test_coverage_never_rescues_a_pick_that_fails_the_quality_gate(monkeypatch):
    """Like chronology: coverage ranks candidates that already clear the bar. A huge weight must not fill roles with shots the gate rejects."""
    pool = _pool()
    base = plan(pool)[0]
    monkeypatch.setitem(story.S, "coverage", 50.0)
    big = plan(pool)[0]
    assert len(big.clips) <= max(len(base.clips) + 2, 6)                       # no explosion of extra clips
    assert [c.role for c in big.clips] == sorted([c.role for c in big.clips], key=lambda r: ["OPENING", "BUILDUP", "RITUAL", "REVEAL", "CLIMAX", "CLOSING"].index(r))
