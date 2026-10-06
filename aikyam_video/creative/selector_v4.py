"""V4: the simple deterministic moment selector (EXP-V4-001, baseline).

The V4 hypothesis is that reels come from a SMALL NUMBER of deterministic decisions, not from
story theory. Models supply evidence (Moments, already scored upstream by scoring.V4_WEIGHTS); this
module only decides which evidence is used, in what order, and how much.

Four steps, in this order, no learned weights, no story arc, no beam search:
    1. quality    -- keep the highest-scoring Moments
    2. diversity  -- never two clips overlapping in the source
    3. target     -- take TARGET_CLIPS of them
    4. chronology -- play them back in source order

Everything else (captions, music block, subject tracking, thumbnails, render, QC) is the SAME shared
code the classic planner uses (plan.assemble_plan, stages._plan_classic), so a V3/V4 A/B compares the
SELECTOR and nothing else. It deliberately does not import creative/story.py or any other V3 creative
engine: the only shared things are the Moment/SceneVision data structures and the plan contract.

Determinism: the sort key is (-score, start, momentId), so equal scores always resolve the same way
and the same inputs always give a byte-identical plan.
"""
from __future__ import annotations
from typing import Dict, List, Optional

from ..models import EntityRef, Moment, SceneVision, Transcript
from ..plan import MAX_REEL_S, assemble_plan
from ..reframe import subject_x

# EXP-V4-001, FROZEN. Nothing here is tuned: this is the clean baseline we measure against V3.
TARGET_CLIPS = 5            # number of clips in the reel
MAX_SEG_S = 15.0            # longest single clip (see per_clip_cap below: the reel cap usually binds first)
MIN_SEG_S = 2.0             # shorter than this and a clip is dropped
ORDERING = "chronological"  # play in source order, never reorder (the one decision V3's arc makes instead)
DIVERSITY = "no-overlap"    # reject any candidate overlapping an already-picked clip
ENGINE = "v4"


def plan_edit(video_id: str, path: str, duration: float, moments: List[Moment], transcript: Transcript,
              vision: List[SceneVision], entities: Dict[str, List[EntityRef]], target_s: float = 45.0,
              max_seg_s: float = MAX_SEG_S, target_clips: int = TARGET_CLIPS,
              caption_lang: Optional[str] = None, caption_mode: str = "sentence",
              caption_style: Optional[dict] = None, aspect: str = "9:16", fmt: str = "REEL",
              location: Optional[str] = None, transition: str = "crossfade", transition_s: float = 0.5,
              snap=None) -> dict:
    """One V4 plan. `moments` are already scored (scoring.py, V4_WEIGHTS); this function only selects.

    Unknown Moment fields (the V3 scorers' signals) are left on the Moment and ignored here: V4 never
    reads them, so the same moments.json can be replayed by either engine.
    """
    ranked = sorted(moments, key=lambda m: (-m.score, m.start, m.momentId))
    # plan.validate_plan rejects any reel longer than MAX_REEL_S, so the reel length is a HARD constraint and
    # TARGET_CLIPS is a target, not a promise. Splitting the budget by the target count is what lets both frozen
    # numbers hold at once (5 x 12 s <= 60 s): taking TARGET_CLIPS clips of up to MAX_SEG_S each would emit an
    # illegal plan 3 out of 4 times. The clip cap is therefore the reel budget per target clip.
    per_clip_cap = min(max_seg_s, MAX_REEL_S / max(1, target_clips))
    decisions: List[dict] = [{"type": "candidates", "count": len(ranked),
                              "topScore": round(ranked[0].score, 4) if ranked else None,
                              "scoreRange": [round(ranked[-1].score, 4), round(ranked[0].score, 4)] if ranked else None,
                              "clipCapSeconds": round(per_clip_cap, 2)}]

    # 2 + 3. quality-ordered greedy with a hard overlap rule: the diversity rule is applied DURING the
    # walk, so a rejected clip's slot is offered to the next candidate (taking a top-N slice first and
    # then de-duplicating would leave holes).
    chosen: List[dict] = []
    rejected_overlap: List[str] = []
    rejected_short: List[str] = []
    for m in ranked:
        if len(chosen) >= target_clips:
            break
        a, b = m.start, min(m.end, m.start + per_clip_cap)
        if snap and b < m.end - 1e-6:                                   # end was trimmed mid-moment: land on a quiet moment instead
            b = min(max(snap(b), a + MIN_SEG_S), m.end, m.start + per_clip_cap)
        if b - a < MIN_SEG_S - 1e-6:
            rejected_short.append(m.momentId)
            continue
        if any(a < c["end"] and b > c["start"] for c in chosen):
            rejected_overlap.append(m.momentId)
            continue
        chosen.append({"start": max(0.0, round(a, 2)), "end": min(duration, round(b, 2)),
                       "reason": m.reason, "score": m.score, "momentId": m.momentId})
        decisions.append({"type": "select", "momentId": m.momentId, "start": round(a, 2), "end": round(b, 2),
                          "score": m.score, "why": m.reason})
    for mid in rejected_short:
        decisions.append({"type": "reject", "momentId": mid, "why": f"shorter than {MIN_SEG_S}s"})
    for mid in rejected_overlap:
        decisions.append({"type": "reject", "momentId": mid, "why": f"{DIVERSITY}: overlaps a clip already chosen"})

    # 4. chronology. Plan validation (plan.validate_plan) also requires segments in source order
    # unless a plan declares order="narrative", so this is what makes a V4 plan legal.
    chosen.sort(key=lambda s: s["start"])
    for s in chosen:
        s["subjectX"] = round(subject_x(vision, s["start"], s["end"]), 3)      # basic centre/subject crop

    n = len(chosen)
    if transition == "crossfade" and n > 1:
        trans = {"type": "crossfade", "durationSeconds": round(min(transition_s, min(s["end"] - s["start"] for s in chosen) / 3), 2)}
    elif transition == "fade" and n:
        trans = {"type": "fade", "durationSeconds": round(min(transition_s, 0.4), 2)}
    else:
        trans = {"type": "cut", "durationSeconds": 0}

    decisions.append({"type": "order", "order": ORDERING, "clips": [s["momentId"] for s in chosen]})
    if len(chosen) < target_clips:
        decisions.append({"type": "short", "clips": len(chosen), "target": target_clips,
                          "why": "not enough non-overlapping candidates in the source"})
    # `engine: "v4"` (not "creative") is what keeps stages.render on the plain FFmpeg path.
    creative = {"version": 1, "engine": ENGINE, "order": "source", "arc": [], "decisions": decisions,
                "selector": {"targetClips": target_clips, "ordering": ORDERING, "diversity": DIVERSITY,
                             "maxSegSeconds": round(per_clip_cap, 2), "transition": trans["type"],
                             "transitionSeconds": trans["durationSeconds"]}}
    return assemble_plan(video_id, path, duration, chosen, trans, transcript, entities,
                         caption_lang, caption_mode, caption_style, aspect, fmt, location, creative=creative)