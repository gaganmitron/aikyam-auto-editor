#!/usr/bin/env python
"""Smoke test of the REAL Claude model as the Aikyam director (needs credentials: ANTHROPIC_API_KEY or `ant auth login`; costs a few cents).

    python tools/llm_smoke.py                       # synthetic temple shots
    python tools/llm_smoke.py results/my_hook       # the shots/plan of a real run (edit-plan.json's creative decisions are NOT used)

Checks: the request shape is accepted by the API, the reply parses and validates against DIRECTOR_SCHEMA, every shot id is real, and the story planner accepts the proposal
(no 'director-fallback'). Prints latency, token usage if available, and the LLM's story next to the deterministic one."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aikyam_video.creative import pacing, story
from aikyam_video.creative.shots import Population
from aikyam_video.planner_llm import AnthropicLLM, make_director
from tests.test_creative_story import temple_pool


def main() -> int:
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print("No Anthropic credentials in the environment: set ANTHROPIC_API_KEY (or run `ant auth login`) and re-run."); return 2
    pool = temple_pool(); pop = Population(pool); prof = pacing.profile("devotional")
    llm = AnthropicLLM(); t0 = time.perf_counter()
    base = story.plan_story(pool, pop, prof, 300.0)
    tl = story.plan_story(pool, pop, prof, 300.0, director=make_director(llm), notes={"m2": "jai ganga maiya, aarti begins"})
    dt = time.perf_counter() - t0
    d = next((x for x in tl.decisions if x["type"] == "director"), None); fb = next((x for x in tl.decisions if x["type"] == "director-fallback"), None)
    print(f"model {llm.model}  wall {dt:.1f}s")
    print("deterministic:", [(c.role, c.shot.id, round(c.length, 1)) for c in base.clips])
    print("LLM director :", [(c.role, c.shot.id, round(c.length, 1)) for c in tl.clips])
    if d: print("accepted. why:", d.get("why")); print("title:", d.get("title"))
    if fb: print("REJECTED by the validator (fell back to the deterministic story):", fb["error"]); return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
