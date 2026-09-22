"""LLM director on the creative engine (FakeLLM: no credentials here). The LLM chooses WHAT/ORDER/LENGTH from measured shots; the engine validates every proposal."""
import pytest
from aikyam_video.creative import pacing, story
from aikyam_video.planner_llm import DIRECTOR_SCHEMA, make_director
from tests.test_creative_story import plan, temple_pool


class FakeLLM:
    name = "fake"; model = "fake-1"
    def __init__(self, out=None, exc=None): self.out, self.exc, self.seen = out, exc, None
    def propose(self, system, user, schema):
        self.seen = (system, user, schema)
        if self.exc: raise self.exc
        return self.out


def clip(sid, role, sec, why="x"): return {"shotId": sid, "role": role, "seconds": sec, "why": why}


def run(out=None, exc=None):
    llm = FakeLLM(out, exc); tl, _ = plan(temple_pool(), director=make_director(llm), notes={"m2": "jai ganga maiya"}); return tl, llm


def kinds(tl): return [d["type"] for d in tl.decisions]


def test_valid_proposal_is_used_in_order_with_its_lengths_and_recorded():
    tl, llm = run({"clips": [clip("m0", "OPENING", 5), clip("m2", "CLIMAX", 6), clip("m3", "REVEAL", 4)], "overlayTitle": "Ganga Aarti"})
    assert [(c.shot.id, c.role) for c in tl.clips] == [("m0", "OPENING"), ("m2", "CLIMAX"), ("m3", "REVEAL")]
    assert [round(c.length) for c in tl.clips] == [4, 6, 4] and "director" in kinds(tl)          # 5 s asked for the opening: the 3.5 s hook cap applies to a director's opening too
    assert tl.clips[0].length <= story.HOOK_MAX + 1e-6
    d = next(x for x in tl.decisions if x["type"] == "director"); assert d["model"] == "fake-1" and d["title"] == "Ganga Aarti"
    assert "jai ganga maiya" in llm.seen[1] and "m5" in llm.seen[1]                     # measured facts + transcript reach the model


@pytest.mark.parametrize("bad", [
    {"clips": [clip("nope", "OPENING", 5)], "overlayTitle": ""},                          # invented shot
    {"clips": [clip("m0", "OPENING", 5), clip("m0", "RITUAL", 5)], "overlayTitle": ""},   # same shot twice
    {"clips": [], "overlayTitle": ""},                                                    # nothing
    {"clips": [clip("m0", "FINALE", 5)], "overlayTitle": ""},                             # invented role
    {"clips": [clip(f"m{i % 6}", "RITUAL", 3) for i in range(9)], "overlayTitle": ""},    # too many
])
def test_bad_proposals_fall_back_to_the_deterministic_story(bad):
    tl, _ = run(bad); base, _ = plan(temple_pool())
    assert "director-fallback" in kinds(tl) and [(c.shot.id, c.role) for c in tl.clips] == [(c.shot.id, c.role) for c in base.clips]


def test_llm_outage_falls_back_and_lengths_are_clamped_to_the_profile():
    tl, _ = run(exc=RuntimeError("API down")); assert "director-fallback" in kinds(tl) and tl.clips
    tl, _ = run({"clips": [clip("m0", "OPENING", 60), clip("m2", "CLIMAX", 0.2)], "overlayTitle": ""}); p = pacing.profile("devotional")
    assert all(p.min_shot - 1e-6 <= c.length <= p.max_shot + 1e-6 or c.length == c.shot.length for c in tl.clips)


def test_schema_only_allows_ids_roles_seconds_and_a_title():
    assert set(DIRECTOR_SCHEMA["properties"]) == {"clips", "overlayTitle"}
    assert set(DIRECTOR_SCHEMA["properties"]["clips"]["items"]["properties"]) == {"shotId", "role", "seconds", "why"}   # no timestamps, transitions or audio
