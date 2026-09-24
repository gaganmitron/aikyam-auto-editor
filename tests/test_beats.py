"""Story beats (beats.py + roles.json "beats" + story.py variety rule): what a shot is FOR in the story steers role fit and avoids back-to-back repetition."""
import numpy as np
import pytest
from aikyam_video import beats
from aikyam_video.creative import roles, story
from aikyam_video.creative.shots import Population
from tests.test_creative_story import shot, e, plan


def test_score_is_a_distribution_and_follows_the_prompt_it_matches():
    n = sum(len(v) for v in beats.BEATS.values()); T = np.eye(n); _, owner = beats.text_embeddings(lambda ps: np.eye(len(ps)))
    emb = np.zeros(n); emb[list(owner).index(beats.KEYS.index("procession"))] = 1.0                         # an image that looks exactly like a procession prompt
    p = beats.score(emb, T, owner, scale=10.0)
    assert set(p) == set(beats.KEYS) and sum(p.values()) == pytest.approx(1.0) and max(p, key=p.get) == "procession"


def _mk(i, start, labels, emb, bts):
    s = shot(i, start, 12, labels, emb, 0.5, motion=4); s.beats = bts; return s


def test_role_fit_follows_the_beat_and_no_beats_means_no_change(monkeypatch):
    rit = _mk(0, 0, {"aarti": 0.6}, e(1, 0, 0), {"ritual_action": 0.9, "establishing": 0.05})
    est = _mk(1, 20, {"aarti": 0.6}, e(0, 1, 0), {"establishing": 0.9, "ritual_action": 0.05})
    pop = Population([rit, est]); a, b = roles.affinities(rit, pop, 0.5), roles.affinities(est, pop, 0.5)
    assert a["RITUAL"] > b["RITUAL"] and b["OPENING"] > a["OPENING"]                                        # each shot serves the role its beat is for
    bare = _mk(2, 40, {"aarti": 0.6}, e(0, 0, 1), {}); on = roles.affinities(bare, Population([bare]), 0.5)
    monkeypatch.setitem(roles.CFG["beats"], "weight", 0.0)
    assert on == roles.affinities(bare, Population([bare]), 0.5)                                            # a shot with no beats is untouched by the beat term


def test_variety_rule_stops_the_top_up_from_repeating_the_neighbouring_beat(monkeypatch):
    """The beam leaves the second ritual shot out; the top-up used to add it back (adjacent, same beat) just to reach the target length."""
    pool = [_mk(0, 5, {"temple_architecture": 0.9}, e(1, 0, 0, 0, 0, 0), {"establishing": 0.9}), _mk(1, 40, {"devotees": 0.9}, e(0, 1, 0, 0, 0, 0), {"approach": 0.8}),
            _mk(2, 90, {"aarti": 0.9, "priest": 0.6}, e(0, 0, 1, 0, 0, 0), {"ritual_action": 0.9}), _mk(3, 140, {"aarti": 0.85, "priest": 0.6}, e(0, 0, 0.9, 0.4, 0, 0), {"ritual_action": 0.9}),
            _mk(4, 190, {"procession": 0.9}, e(0, 0, 0, 0, 1, 0), {"procession": 0.9}), _mk(5, 250, {"deity": 0.9, "idol": 0.8}, e(0, 0, 0, 0, 0, 1), {"darshan": 0.9})]
    dom = lambda s: max(s.beats, key=s.beats.get)
    adjacent = lambda tl: sum(1 for a, b in zip(tl.clips, tl.clips[1:]) if dom(a.shot) == dom(b.shot))
    monkeypatch.setitem(story.S, "beat_repeat_penalty", 0.0); off = adjacent(plan(pool)[0])
    monkeypatch.setitem(story.S, "beat_repeat_penalty", 0.15); on = adjacent(plan(pool)[0])
    assert off == 1 and on == 0


def test_pool_reserves_room_for_a_beat_the_score_ranking_would_leave_out():
    from types import SimpleNamespace as N
    from aikyam_video.creative.shots import diverse_top
    ms = [N(id=f"m{i}", start=float(i), score=0.9 - 0.02 * i) for i in range(10)] + [N(id="wide", start=99.0, score=0.30)]        # a low-scoring establishing view
    bt = {m.id: ({"ritual_action": 0.8} if m.id != "wide" else {"establishing": 0.7}) for m in ms}
    pool = diverse_top(ms, lambda m: bt[m.id], top_k=6, reserve=2)
    assert "wide" in [m.id for m in pool] and len(pool) == 6                                                                        # the missing beat gets a seat, the pool size is unchanged
    assert [m.id for m in diverse_top(ms, lambda m: {}, top_k=6, reserve=2)] == [f"m{i}" for i in range(6)]                       # beats unknown: exactly the plain top-k
    weak = {**bt, "wide": {"establishing": 0.2, "detail": 0.19}}
    assert "wide" not in [m.id for m in diverse_top(ms, lambda m: weak[m.id], top_k=6, reserve=2)]                                  # a beat only counts when it clearly dominates
