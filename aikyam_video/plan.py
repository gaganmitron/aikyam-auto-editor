"""EditPlan schema validation + deterministic planner (sections 4, 11). Renderer-independent.

LLM planners plug in by producing a proposal of segments; `validate_plan` is the gate: every timestamp must lie in the
source, segments must not overlap, total <= 60 s, and (via `snap_to_moments`) must fall inside validated moments.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional
import jsonschema
from .captions import build_cues
from .models import EntityRef, Moment, SceneVision, Transcript
from .reframe import subject_x

SCHEMA = json.loads((Path(__file__).parent / "data" / "edit_plan.schema.json").read_text())
MAX_REEL_S = 60.0


class PlanError(ValueError):
    pass


OVERLAPPING = {"crossfade", "dip_black", "dip_white"}      # transitions whose two clips overlap in time (xfade); "cut" and legacy "fade" do not


def edge_transitions(segments: List[dict], default: dict) -> List[dict]:
    """Transition INTO segment i+1 for every edge: the segment's own `transitionIn` if set, else the plan-level default."""
    out = []
    for s in segments[1:]:
        t = s.get("transitionIn") or default
        out.append({"type": t["type"], "durationSeconds": float(t.get("durationSeconds", 0)), "reason": t.get("reason", "")})
    return out


def edge_overlaps(segments: List[dict], default: dict) -> List[float]:
    """Seconds by which segment i+1 overlaps segment i (one entry per edge). THE single source of truth for the timeline:
    validation, captions, both renderers, the audio mixer and QC all use it."""
    return [e["durationSeconds"] if e["type"] in OVERLAPPING else 0.0 for e in edge_transitions(segments, default)]


def overlap_s(transitions: dict, n: int) -> float:
    """Legacy uniform overlap (plan-level transition only)."""
    return float(transitions.get("durationSeconds", 0)) if transitions.get("type") in OVERLAPPING and n > 1 else 0.0


def output_duration(segments: List[dict], transitions: dict) -> float:
    """Length of the rendered reel: sum of segments minus every edge overlap."""
    if not segments:
        return 0.0
    return sum(s["end"] - s["start"] for s in segments) - sum(edge_overlaps(segments, transitions))


def timeline(segments: List[dict], transitions: dict) -> List[dict]:
    """Where each segment sits on the OUTPUT timeline: [{start, end}] in seconds."""
    ovs = edge_overlaps(segments, transitions); t, out = 0.0, []
    for i, s in enumerate(segments):
        L = s["end"] - s["start"]; out.append({"start": t, "end": t + L}); t += L - (ovs[i] if i < len(ovs) else 0.0)
    return out


def asset_map(plan: dict) -> Dict[str, dict]:
    return {a["id"]: a for a in plan.get("assets", [])}


def seg_source(plan: dict, s: dict, default: Optional[str] = None) -> Optional[str]:
    """File a segment's picture/sound comes from: its asset's path (None for an image: silent), else the single source `default`."""
    a = asset_map(plan).get(s.get("assetId"))
    if a is None:
        return default
    return None if (s.get("kind") or a["kind"]) == "image" else a["path"]


def validate_plan(plan: dict, source_duration: Optional[float] = None) -> None:
    try:
        jsonschema.validate(plan, SCHEMA)
    except jsonschema.ValidationError as e:
        raise PlanError(f"schema: {'/'.join(map(str, e.path))}: {e.message}")
    D = source_duration or plan["source"]["durationSeconds"]
    segs, tr = plan["segments"], plan["transitions"]
    assets = asset_map(plan)
    narrative = plan.get("creative", {}).get("order") == "narrative"        # narrative edits may reorder source material
    last_end: Dict[str, float] = {}
    for i, s in enumerate(segs):
        aid = s.get("assetId")
        if assets and aid not in assets:
            raise PlanError(f"segment {i} references unknown asset {aid!r}")
        a = assets.get(aid)
        kind = s.get("kind") or (a["kind"] if a else "video")
        if a and a["kind"] == "audio":
            raise PlanError(f"segment {i} references an audio asset")
        if kind == "image":
            if not (s["start"] >= 0 and s["end"] > s["start"]):
                raise PlanError(f"image segment {i} has an invalid duration")
            if a and a["kind"] != "image":
                raise PlanError(f"segment {i} is an image but asset {aid!r} is not")
            continue
        limit = (a.get("durationSeconds") if a else None) or D
        if not (0 <= s["start"] < s["end"] <= limit + 1e-3):
            raise PlanError(f"segment {i} [{s['start']}, {s['end']}] outside source 0..{limit}")
        if not narrative and s["start"] < last_end.get(aid or "", -1.0):
            raise PlanError(f"segment {i} overlaps or is out of order")
        last_end[aid or ""] = s["end"]
    for i, a in enumerate(segs):                                             # source ranges of the SAME video asset must never overlap, in any order
        if (a.get("kind") or "video") == "image":
            continue
        for j in range(i + 1, len(segs)):
            b = segs[j]
            if (b.get("kind") or "video") == "video" and a.get("assetId") == b.get("assetId") and a["start"] < b["end"] and b["start"] < a["end"]:
                raise PlanError(f"segments {i} and {j} overlap in the source")
    ovs = edge_overlaps(segs, tr)
    for i, s in enumerate(segs):                                             # the overlaps at both ends must leave the clip room to play
        din = ovs[i - 1] if i > 0 else 0.0; dout = ovs[i] if i < len(ovs) else 0.0
        if din + dout > 0.8 * (s["end"] - s["start"]) + 1e-6:
            raise PlanError(f"transitions consume more than 80% of segment {i} ({din + dout:.2f}s of {s['end'] - s['start']:.2f}s)")
        if din > 0.5 * (s["end"] - s["start"]) + 1e-6 or dout > 0.5 * (s["end"] - s["start"]) + 1e-6:
            raise PlanError(f"a transition is longer than half of segment {i}")
    m = plan.get("audio", {}).get("music", {})
    if m.get("assetId") and (m["assetId"] not in assets or assets[m["assetId"]]["kind"] != "audio"):
        raise PlanError(f"music references unknown/non-audio asset {m['assetId']!r}")
    total = output_duration(segs, tr)
    if abs(total - plan["durationSeconds"]) > 0.05:
        raise PlanError(f"durationSeconds {plan['durationSeconds']} != rendered length {total:.2f} (segments minus transition overlaps)")
    if total > MAX_REEL_S + 1e-6:
        raise PlanError(f"total {total:.1f}s exceeds {MAX_REEL_S}s")


def snap_to_moments(proposed: List[dict], moments: List[Moment]) -> List[dict]:
    """LLM safety net: keep only proposed segments that lie inside a validated moment (clamped to it)."""
    out = []
    for p in proposed:
        for m in moments:
            a, b = max(p["start"], m.start), min(p["end"], m.end)
            if b - a >= 2.0:
                out.append({**p, "start": a, "end": b, "momentId": m.momentId,
                            "score": m.score, "reason": p.get("reason", m.reason)}); break
    return sorted(out, key=lambda s: s["start"])


def _pick(entities: Dict[str, List[EntityRef]], typ: str) -> Optional[EntityRef]:
    l = entities.get(typ)
    return max(l, key=lambda r: r.confidence) if l else None


def plan_edit(video_id: str, path: str, duration: float, moments: List[Moment], transcript: Transcript,
              vision: List[SceneVision], entities: Dict[str, List[EntityRef]], target_s: float = 45.0,
              max_seg_s: float = 15.0, caption_lang: Optional[str] = None, caption_mode: str = "sentence",
              caption_style: Optional[dict] = None, aspect: str = "9:16", fmt: str = "REEL",
              location: Optional[str] = None, transition: str = "crossfade", transition_s: float = 0.5, snap=None) -> dict:
    """Greedy: take highest-scoring non-overlapping moments (each trimmed to max_seg_s) until target_s is filled,
    then present them in source order. Deterministic: same inputs -> same plan.
    transition: crossfade (default; picture + sound blend, segments overlap) | fade (dip to black per segment) | cut.
    snap(t) -> t': moves a TRIMMED segment end to a nearby quiet point so we don't cut through a word/chant."""
    target_s = min(target_s, MAX_REEL_S)
    chosen, used = [], 0.0
    for m in sorted(moments, key=lambda m: (-m.score, m.start)):
        if used >= target_s - 2.0:
            break
        a, b = m.start, min(m.end, m.start + max_seg_s, m.start + (target_s - used))
        if snap and b < m.end - 1e-6:                              # end was trimmed mid-moment: land on a quiet moment instead
            b = min(max(snap(b), a + 2.0), m.end, m.start + (target_s - used))
        if b - a < 2.0 or any(a < c["end"] and b > c["start"] for c in chosen):
            continue
        chosen.append({"start": max(0.0, round(a, 2)), "end": min(duration, round(b, 2)), "reason": m.reason, "score": m.score, "momentId": m.momentId})
        used += b - a
    chosen.sort(key=lambda s: s["start"])
    for s in chosen:
        s["subjectX"] = round(subject_x(vision, s["start"], s["end"]), 3)
    n = len(chosen)
    if transition == "crossfade" and n > 1:
        trans = {"type": "crossfade", "durationSeconds": round(min(transition_s, min(s["end"] - s["start"] for s in chosen) / 3), 2)}
    elif transition == "fade" and n:
        trans = {"type": "fade", "durationSeconds": round(min(transition_s, 0.4), 2)}
    else:
        trans = {"type": "cut", "durationSeconds": 0}
    return assemble_plan(video_id, path, duration, chosen, trans, transcript, entities, caption_lang, caption_mode, caption_style, aspect, fmt, location)


def assemble_plan(video_id: str, path: str, duration: float, segments: List[dict], trans: dict, transcript: Transcript,
                  entities: Dict[str, List[EntityRef]], caption_lang: Optional[str] = None, caption_mode: str = "sentence",
                  caption_style: Optional[dict] = None, aspect: str = "9:16", fmt: str = "REEL", location: Optional[str] = None,
                  creative: Optional[dict] = None, assets: Optional[List[dict]] = None, title: Optional[str] = None, subtitle: Optional[str] = None) -> dict:
    """Everything around the segments (source ids, overlays/template, captions on the overlapped timeline, audio block, validation).
    Shared by the classic planner and the creative engine so both produce the same contract."""
    chosen = segments
    used = output_duration(chosen, trans)
    lang = caption_lang or (next(iter(transcript.values())) if isinstance(transcript, dict) and transcript else transcript).language or "en"
    cues = build_cues(transcript, chosen, caption_mode, edge_overlaps(chosen, trans)) if chosen else []
    temple, deity, ritual, fest = (_pick(entities, t) for t in ("TEMPLE", "DEITY", "RITUAL", "FESTIVAL"))
    from .render import TEMPLATES
    template = (TEMPLATES.get("_festivalTemplates", {}).get(fest.entityId, "festival") if fest
                else "ritual_highlight" if (ritual and temple) else "divine_moment")
    ids = lambda e: e.entityId if e else None
    name = lambda e: e.text if e else None
    disp = lambda e: (e.name or e.text) if e else None      # canonical display name, not the matched alias text
    plan = {
        "schemaVersion": 1,
        "source": {"videoId": video_id, "path": path, "durationSeconds": duration, "templeId": ids(temple),
                   "deityId": ids(deity), "ritualId": ids(ritual), "festivalId": ids(fest)},
        "outputFormat": fmt, "durationSeconds": round(used, 2), "aspectRatio": aspect, "segments": chosen,
        "captions": {"enabled": bool(cues), "language": lang, "mode": caption_mode,
                     "style": caption_style or {"fontSize": 0.032, "position": "bottom", "background": True, "animation": "fade"},
                     "cues": cues},
        "overlays": {"template": template, "temple": disp(temple), "deity": disp(deity), "ritual": disp(ritual),
                     "festival": disp(fest), "location": location},
        "transitions": trans,
        "audio": {"preserveOriginal": True, "normalize": True, "music": {"enabled": False}},
        "thumbnail": {"timestamp": None},
    }
    if title is None and template == "divine_moment":                       # a plain template shows no text: name what is known, invent nothing
        names = [n for n in (disp(ritual), disp(fest), disp(deity), disp(temple)) if n]
        title = names[0] if names else None
        subtitle = subtitle or next((n for n in (disp(temple), location) if n and n != title), None)
    if title:
        plan["overlays"]["title"] = title[:70]
        if subtitle: plan["overlays"]["subtitle"] = subtitle[:70]
    if assets:
        plan["schemaVersion"] = 3; plan["assets"] = assets
    if creative:
        plan["creative"] = creative                          # attached BEFORE validation: `order: narrative` legalises a reordered story
    if chosen:
        validate_plan(plan, duration)
    return plan
