"""Candidate generation -> validation -> ranking (sections 9, 10). Timestamps are ALWAYS validated here."""
from __future__ import annotations
import re, subprocess
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from .audio import AudioProfile
from .entities import GazetteerEntityExtractor
from .ffmpeg import MediaInfo
from .models import EntityRef, Moment, Rejection, Scene, SceneVision, Transcript
from .vision import MODERATION_BLOCK
from .scoring import ScoringConfig, score, DEVOTIONAL, DEVOTIONAL_SOUNDS, _dominant_labels
from .creative.events import event_boundaries


@dataclass
class Ctx:
    path: str
    info: MediaInfo
    scenes: List[Scene]
    vision: List[SceneVision]
    transcript: Transcript
    audio: AudioProfile
    segment_entities: list = field(default_factory=list)   # [(TranscriptSegment, [EntityRef])]
    label_prevalence: Dict[str, float] = field(default_factory=dict)
    black: List[Tuple[float, float]] = field(default_factory=list)
    profile: object = None   # options.Profile
    allow_silent: bool = False   # picture-only footage is acceptable (its sound comes from a separate music track)
    text_frames: List[List[float]] = field(default_factory=list)   # [[t, no_text]] dense track (analysis.text_track): catches a graphic inside a long scene
    text_floor: Optional[float] = None                             # cached by validate_clip: the no_text value under which a dense-track frame counts as a graphic
    event_bounds: List[float] = field(default_factory=list)   # EXP-004: candidate real-event boundaries (embedding-similarity dips) -- a moment shouldn't straddle two different occasions


def build_ctx(path, info, scenes, vision, transcript, audio, extractor: GazetteerEntityExtractor, black=None, profile=None, allow_silent=False, text_frames=None) -> Ctx:
    n = max(1, len(vision))
    prev: Dict[str, float] = {}
    for sv in vision:
        for k, v in sv.vision.labels.items():
            if v >= 0.3:
                prev[k] = prev.get(k, 0) + 1 / n
    return Ctx(path, info, scenes, vision, transcript, audio,
               [(s, extractor.extract(s.text, transcript.language)) for s in transcript.segments], prev,
               detect_black(path) if black is None else black, profile, allow_silent, text_frames or [], event_bounds=event_boundaries(vision))


def detect_black(path: str, min_len: float = 0.5) -> List[Tuple[float, float]]:
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-vf", f"blackdetect=d={min_len}:pix_th=0.10",
                        "-an", "-f", "null", "-"], capture_output=True)
    return [(float(a), float(b)) for a, b in
            re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", p.stderr.decode(errors="replace"))]


# Frames that are a title card, credits, a subtitle or a channel's own "LIKE / subscribe" animation are not footage (EXP-019: a reel picked a giant thumbs-up graphic as its REVEAL).
# `no_text` is a raw model logit difference, so the cut-off is calibrated PER MODEL: default B-16 SigLIP -- title card -1.6, LIKE graphic 2.2, credits 0.8, subtitled footage -4.9 / 2.4,
# against >= 4.7 for every genuine footage scene in 42 + 35 frames. A model without a calibrated cut-off gets no rule (never a wrong one).
NO_TEXT_MIN = {"hf-hub:timm/ViT-B-16-SigLIP": 3.0}
NO_TEXT_DIP = 6.0       # dense track (2 s samples): a frame this far under the video's OWN median no_text is a graphic too. Needed because a small animation scores ~6.4 on a drone video whose footage sits at ~14 (the fixed cut-off 3.0 misses it); B-16 SigLIP only
SPECIAL = {"ritual_dance": 0.9, "fireworks": 0.5}   # labels DEVOTIONAL doesn't weight but that name a moment well


# ---------------------------------------------------------------- candidate generation
def _scene_at(ctx: Ctx, t: float) -> Optional[Scene]:
    return next((s for s in ctx.scenes if s.start <= t < s.end), ctx.scenes[-1] if ctx.scenes else None)


def _reason(ctx: Ctx, a: float, b: float) -> str:
    labs: Dict[str, float] = {}
    for sv in ctx.vision:
        if sv.end > a and sv.start < b:
            for k, v in sv.vision.labels.items():
                labs[k] = max(labs.get(k, 0), v)
    ranked = sorted(labs, key=lambda k: -SPECIAL.get(k, DEVOTIONAL.get(k, 0)) * labs[k])
    if ranked and SPECIAL.get(ranked[0], DEVOTIONAL.get(ranked[0], 0)) * labs[ranked[0]] >= 0.3:
        return ranked[0]
    snd = ctx.audio.top_event(a, b)          # weak visuals: name the moment by what it sounds like
    if snd in DEVOTIONAL_SOUNDS:
        return snd
    return "speech" if any(s.end > a and s.start < b for s in ctx.transcript.segments) else "scene"


def generate_candidates(ctx: Ctx, min_len=5.0, max_len=25.0, step=10.0) -> List[Tuple[float, float]]:
    """Signal-driven windows: whole scenes (windowed if long), audio-onset windows, entity-mention windows."""
    D = ctx.info.duration
    c: List[Tuple[float, float]] = []
    for s in ctx.scenes:
        if s.end - s.start <= max_len:
            c.append((s.start, s.end))
        else:
            t = s.start
            while t < s.end - min_len:
                c.append((t, min(t + max_len * 0.8, s.end))); t += step
    p = ctx.audio
    for i in [i for i, v in enumerate(p.peaks) if v]:
        t = i * p.hop; sc = _scene_at(ctx, t)
        c.append((max(sc.start if sc else 0, t - 3), min(D, t + 12)))
    for seg, refs in ctx.segment_entities:
        if refs:
            c.append((max(0, seg.start - 2), min(D, seg.end + 4)))
    out = []
    for a, b in c:  # widen too-short windows, clamp to the source
        if b - a < min_len:
            m = (a + b) / 2; a, b = m - min_len / 2, m + min_len / 2
        a, b = max(0.0, a), min(D, b)
        out.append((round(a, 2), min(D, round(b, 2))))   # clamp AFTER rounding: 20.017 must not become 20.02
    return out


# ---------------------------------------------------------------- validation (section 10)
def _iou(a, b):
    i = max(0, min(a[1], b[1]) - max(a[0], b[0])); u = (a[1] - a[0]) + (b[1] - b[0]) - i
    return i / u if u > 0 else 0


def validate_clip(ctx: Ctx, a: float, b: float, accepted: List[Tuple[float, float]],
                  min_len=3.0, max_len=30.0, max_silence=0.6, max_black=0.3, dup_iou=0.5) -> Optional[str]:
    """Return a rejection reason, or None if the clip is good."""
    if not (a >= 0 and b <= ctx.info.duration + 1e-3 and b > a):
        return "out_of_source_bounds"
    if not (min_len <= b - a <= max_len):
        return "bad_duration"
    silent = not ctx.info.has_audio or not len(ctx.audio.rms_db)
    if silent and not ctx.allow_silent:
        return "no_audio"
    if not silent and not ctx.allow_silent and ctx.audio.silence_ratio(a, b) > max_silence:   # allow_silent: sound comes from a separate track, quiet footage is fine
        return "excessive_silence"
    blk = sum(max(0, min(b, y) - max(a, x)) for x, y in ctx.black)
    if blk / (b - a) > max_black:
        return "black_frames"
    if len([s for s in ctx.scenes if s.end - 0.25 > a and s.start + 0.25 < b]) > 2:
        return "crosses_too_many_scenes"
    from .vision import model_name
    cut = NO_TEXT_MIN.get(model_name())
    if cut is not None and any(sv.vision.no_text is not None and sv.vision.no_text < cut and min(sv.end, b) - max(sv.start, a) >= 1.0 for sv in ctx.vision):
        return "graphic_or_text_overlay"
    if cut is not None and ctx.text_frames:                         # dense track: a 2 s sample that shows a graphic was on screen for about +-1 s around it
        if ctx.text_floor is None: ctx.text_floor = max(cut, float(np.median([nt for _, nt in ctx.text_frames])) - NO_TEXT_DIP)
        if any(nt < ctx.text_floor and a - 1.0 < t < b + 1.0 for t, nt in ctx.text_frames):
            return "graphic_or_text_overlay"
    if any(a + 0.25 < t < b - 0.25 for t in ctx.event_bounds):        # EXP-004b: a candidate that straddles a detected event boundary spans two different occasions
        return "crosses_event_boundary"
    if any(sv.end > a and sv.start < b and max(sv.vision.moderation.values(), default=0) >= MODERATION_BLOCK
           for sv in ctx.vision):
        return "moderation_flagged"
    if any(_iou((a, b), x) > dup_iou for x in accepted):
        return "duplicate"
    return None


def _personalize(ctx: Ctx, a: float, b: float, total: float) -> float:
    p = ctx.profile
    if not p:
        return total
    labs = {k: v for k, v in _dominant_labels(ctx, a, b).items() if v >= 0.5}
    boost = max([p.preferred_labels.get(k, 1.0) for k in labs] or [1.0])
    if p.preferred_deities and any(d in p.preferred_deities for sv in ctx.vision if sv.end > a and sv.start < b
                                   for d in sv.vision.deities):
        boost *= 1.15
    return round(min(1.0, total * boost ** 0.5), 4)


def generate_moments(ctx: Ctx, cfg: ScoringConfig, extractor: GazetteerEntityExtractor,
                     metrics=None) -> Tuple[List[Moment], List[Rejection]]:
    scored = []
    for a, b in dict.fromkeys(generate_candidates(ctx)):
        total, comps = score(ctx, a, b, cfg)
        scored.append((_personalize(ctx, a, b, total), a, b, comps))
    scored.sort(key=lambda x: -x[0])
    moments, rejected, accepted = [], [], []
    for total, a, b, comps in scored:
        why = validate_clip(ctx, a, b, accepted)
        if why:
            rejected.append(Rejection(start=a, end=b, reason=why))
            if metrics: metrics.clips_rejected.labels(reason=why).inc()
            continue
        accepted.append((a, b))
        ents = {}
        for seg, refs in ctx.segment_entities:
            if seg.end > a and seg.start < b:
                for r in refs: ents[r.entityId] = r
        moments.append(Moment(momentId=f"moment_{len(moments)}", start=a, end=b, reason=_reason(ctx, a, b),
                              score=total, signals=comps, entities=list(ents.values()),
                              sceneIds=[s.sceneId for s in ctx.scenes if s.end > a and s.start < b]))
        if metrics: metrics.clips_generated.inc()
    return moments, rejected
