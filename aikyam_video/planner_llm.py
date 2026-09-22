"""LLM edit planner (section 11). The LLM proposes WHAT (which validated moments), WHY (reason) and HOW (trim, transition,
overlay title). It never renders and never invents timestamps: every proposal is snapped into an already-validated
moment and re-validated against the source; on ANY problem we fall back to the deterministic plan."""
from __future__ import annotations
import json, os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from .models import Moment
from .plan import PlanError, plan_edit, snap_to_moments, validate_plan

PROPOSAL_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["segments", "transition", "overlayTitle"],
    "properties": {
        "segments": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                     "required": ["momentId", "start", "end", "reason"],
                     "properties": {"momentId": {"type": "string"}, "start": {"type": "number"}, "end": {"type": "number"},
                                    "reason": {"type": "string"}}}},
        "transition": {"type": "string", "enum": ["crossfade", "fade", "cut"]},
        "overlayTitle": {"type": "string"},
    },
}
SYSTEM = ("You are the edit planner for Aikyam, a devotional platform for Indian temples. Given candidate moments from a long "
          "temple recording, choose which to include in a short vertical reel, trim them, and explain why. Prefer complete, "
          "devotionally meaningful moments (darshan of the deity, aarti, abhishekam, processions, bhajans), avoid repetition, "
          "and keep the temple footage dominant. Use ONLY the momentIds given; start/end must lie inside that moment. "
          "The total duration must not exceed the target. Segments are shown in source order.")


class LLMClient(ABC):
    name = "abstract"

    @abstractmethod
    def propose(self, system: str, user: str, schema: dict) -> dict:
        """Return a dict conforming to `schema` or raise."""


class AnthropicLLM(LLMClient):
    """Claude via the official SDK. Model: AIKYAM_LLM_MODEL (default claude-opus-5). Uses adaptive thinking, structured
    output (output_config.format) and server-side refusal fallbacks. Credentials: ANTHROPIC_API_KEY / `ant auth login`."""
    name = "anthropic"

    def __init__(self, model: Optional[str] = None, client=None):
        import anthropic
        self.model = model or os.environ.get("AIKYAM_LLM_MODEL", "claude-opus-5")
        self.client = client or anthropic.Anthropic()

    def propose(self, system, user, schema):
        try:
            r = self.client.beta.messages.create(
                model=self.model, max_tokens=8000, system=system, betas=["server-side-fallback-2026-07-01"],
                extra_body={"fallbacks": "default"},
                thinking={"type": "adaptive"}, output_config={"effort": "medium", "format": {"type": "json_schema", "schema": schema}},
                messages=[{"role": "user", "content": user}])
        except Exception as e:                                                        # noqa: BLE001 -- a rejected request shape (beta/thinking/structured-output params) -> the plain call
            if type(e).__name__ not in ("BadRequestError", "NotFoundError", "UnprocessableEntityError", "TypeError"):
                raise
            import logging
            logging.getLogger("aikyam").warning(f"structured-output request rejected ({type(e).__name__}); retrying as a plain JSON request")
            r = self.client.messages.create(model=self.model, max_tokens=8000, system=system + "\nReply with ONE JSON object that matches this JSON schema, nothing else:\n" + json.dumps(schema),
                                            messages=[{"role": "user", "content": user}])
            text = next((b.text for b in r.content if b.type == "text"), "")
            a, b = text.find("{"), text.rfind("}")
            if a < 0 or b < a:
                raise RuntimeError("LLM returned no JSON")
            return json.loads(text[a:b + 1])
        if r.stop_reason == "refusal":
            raise RuntimeError(f"LLM refused: {getattr(r, 'stop_details', None)}")
        if r.stop_reason == "max_tokens":
            raise RuntimeError("LLM output truncated")
        text = next((b.text for b in r.content if b.type == "text"), None)
        if not text:
            raise RuntimeError("LLM returned no text")
        return json.loads(text)


def get_llm() -> LLMClient:
    """Provider switch. Add Gemini/Vertex/Qwen/OpenAI as new LLMClient subclasses; nothing else changes."""
    kind = os.environ.get("AIKYAM_LLM", "anthropic")
    if kind == "anthropic":
        return AnthropicLLM()
    raise ValueError(f"unknown LLM provider {kind!r}")


def _describe(moments: List[Moment], transcript) -> str:
    out = []
    for m in moments:
        text = " ".join(s.text for s in transcript.segments if s.end > m.start and s.start < m.end)[:200]
        out.append({"momentId": m.momentId, "start": m.start, "end": m.end, "score": m.score, "reason": m.reason,
                    "entities": [f"{e.entityType}:{e.name or e.text}" for e in m.entities], "transcript": text})
    return json.dumps(out, ensure_ascii=False)


def plan_with_llm(video_id, path, duration, moments, transcript, vision, entities, target_s=45.0, llm: Optional[LLMClient] = None,
                  **kw) -> dict:
    base = dict(video_id=video_id, path=path, duration=duration, moments=moments, transcript=transcript, vision=vision,
                entities=entities, target_s=target_s, **kw)
    try:
        llm = llm or get_llm()
        names = {t: [(e.name or e.text) for e in v] for t, v in entities.items()}
        user = (f"Target duration: {min(target_s, 60):.0f}s (hard max 60s). Source length: {duration:.0f}s.\n"
                f"Known entities: {json.dumps(names, ensure_ascii=False)}\nCandidate moments:\n{_describe(moments, transcript)}")
        prop = llm.propose(SYSTEM, user, PROPOSAL_SCHEMA)
        # WHAT/HOW: snap every proposed segment into a validated moment; drop anything that doesn't fit
        snapped = snap_to_moments([{"start": s["start"], "end": s["end"], "reason": s["reason"]} for s in prop["segments"]], moments)
        chosen, used = [], 0.0
        for s in snapped:
            take = min(s["end"] - s["start"], min(target_s, 60.0) - used)
            if take >= 2.0:
                chosen.append(Moment(momentId=s["momentId"], start=s["start"], end=round(s["start"] + take, 2), reason=s["reason"],
                                     score=s["score"])); used += take
        if not chosen:
            raise PlanError("LLM proposal had no usable segments after validation")
        plan = plan_edit(**{**base, "moments": chosen, "max_seg_s": 60.0, "transition": prop["transition"]})   # duration + captions follow it
        if prop.get("overlayTitle") and plan["overlays"].get("template") == "divine_moment":
            plan["overlays"]["ritual"] = prop["overlayTitle"][:60]
        plan["planner"] = {"mode": "llm", "model": getattr(llm, "model", llm.name)}
        validate_plan(plan, duration)
        return plan
    except Exception as e:   # noqa: BLE001 — any LLM/network/validation failure => deterministic plan, recorded
        import logging
        logging.getLogger("aikyam").warning(f"LLM planner failed, using deterministic plan: {e}")
        plan = plan_edit(**base)
        plan["planner"] = {"mode": "deterministic-fallback", "error": str(e)[:200]}
        return plan


# ---------------------------------------------------------------- director mode (creative engine)
DIRECTOR_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["clips", "overlayTitle"],
    "properties": {
        "clips": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["shotId", "role", "seconds", "why"],
                  "properties": {"shotId": {"type": "string"}, "role": {"type": "string", "enum": ["OPENING", "BUILDUP", "RITUAL", "REVEAL", "CLIMAX", "CLOSING"]},
                                 "seconds": {"type": "number"}, "why": {"type": "string"}}}},
        "overlayTitle": {"type": "string"},
    },
}
DIRECTOR_SYSTEM = ("You are the director of a short devotional Reel for Aikyam (Indian temples). You are given MEASURED candidate shots (labels, quality and energy ranks, "
                   "what is said, which source they come from; kind=image are stills that get a slow camera move). Choose and ORDER the shots that tell a short story "
                   "(establish -> build -> ritual -> reveal of the deity -> climax -> quiet close; any role may be skipped) and give each a role and a length in seconds. "
                   "Rules: use ONLY the shotIds given; never repeat a shot or use look-alikes back to back; stills are seasoning, never two in a row; respect the pacing "
                   "profile's shot lengths; total near the target. You choose WHAT and WHY; you never give timestamps, transitions or audio: the editor decides those.")


def make_director(llm: Optional[LLMClient] = None):
    """director(shots_info, profile, target_s) -> {'clips': [(role, shotId, seconds, why)], 'title', 'model'} ; raises on ANY problem (the engine then uses its own story)."""
    def direct(info: List[dict], profile: dict, target_s: float) -> dict:
        c = llm or get_llm()
        user = (f"Pacing profile: {json.dumps(profile)}\nTarget reel length: {target_s:.0f}s (hard max 60s).\nCandidate shots:\n{json.dumps(info, ensure_ascii=False)}")
        prop = c.propose(DIRECTOR_SYSTEM, user, DIRECTOR_SCHEMA)
        clips = [(x["role"], x["shotId"], float(x["seconds"]), x.get("why", "")[:160]) for x in prop["clips"]]
        return {"clips": clips, "title": (prop.get("overlayTitle") or "")[:60], "model": getattr(c, "model", c.name)}
    return direct
