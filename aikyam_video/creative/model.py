"""Data model of the creative layer. Plain dataclasses, no I/O: every decision is inspectable and serialisable."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

ROLES = ["OPENING", "BUILDUP", "RITUAL", "REVEAL", "CLIMAX", "CLOSING"]     # narrative arc order


@dataclass
class SlotFeatures:
    """Per-0.5 s measurements of a source window (arrays share one length). Raw values: ranking against the footage happens in Population."""
    hop: float
    sharpness: List[float]        # variance of the Laplacian (higher = sharper)
    jitter: List[float]           # camera shake: frame-to-frame change of the global shift (px at 160 px width); pans are not jitter
    motion: List[float]           # mean absolute luma change between consecutive samples (scene action)
    luma: List[float]             # mean luma 0..1
    contrast: List[float]         # p95 - p5 of luma 0..1 (a dark but detailed scene still has contrast)
    clipped: List[float]          # fraction of blown-out (>=250) pixels
    concentration: List[float]    # share of edge energy in the central window (closeup ~ high, wide/crowd ~ low)
    face: List[float]             # largest face area as a fraction of the frame
    rms_db: List[float]           # live-audio level
    presence: List[float]         # strongest devotional/speech sound score 0..1 (CLAP), 0 if no tagger
    motion_sig: Optional[List[float]] = None   # where the shot MOVES (6x8 pooled, normalised); None = barely moves


@dataclass
class Shot:
    """A candidate source window (usually one validated Moment) with everything the editor needs to judge it."""
    id: str
    start: float
    end: float
    moment_id: str
    score: float                                  # the highlight-ranking score of the moment
    labels: Dict[str, float]                      # max vision-label confidence over the window
    entities: List[str]
    embedding: Optional[List[float]]              # mean scene embedding (visual similarity)
    slots: Optional[SlotFeatures] = None
    events: Dict[str, float] = field(default_factory=dict)   # mean CLAP event scores over the window
    asset_id: str = "a1"                          # which plan asset the window belongs to (time ranges of different assets never conflict)
    kind: str = "video"                           # "video" | "image" (an image is a flexible-length shot: any length up to `end`)
    position: Optional[float] = None              # 0..1 place in the story material; None = derived from time within the source
    focus: Optional[List[float]] = None           # images: [cx, cy, w, h] of the subject (normalised), for Ken Burns
    size: Optional[List[int]] = None              # images: [width, height] px
    cuts: List[float] = field(default_factory=list)                     # absolute source times of hard cuts in/around the window (coarse)
    pauses: List[List[float]] = field(default_factory=list)             # absolute [start, end] of pauses in the live sound (relative to the shot's own level)
    beats: Dict[str, float] = field(default_factory=dict)               # story-beat distribution of the window (beats.py); {} = unknown

    @property
    def length(self) -> float:
        return self.end - self.start


@dataclass
class Clip:
    """One placed clip: a refined in/out inside a Shot, its story role and how it joins the previous clip."""
    shot: Shot
    start: float
    end: float
    role: str
    transition_in: Optional[dict] = None          # {"type","durationSeconds","reason"}; None for the first clip
    layout: Optional[dict] = None                 # {"mode","pushIn","reason"}
    live_gain_db: float = 0.0
    audio_lead: float = 0.0                       # >0: J-cut (next clip's sound starts before its picture)
    locked_end: bool = False                      # the out point sits on a real cut: beat alignment must not move it
    why: str = ""

    @property
    def length(self) -> float:
        return self.end - self.start


@dataclass
class Timeline:
    clips: List[Clip]
    profile: str
    arc: List[str]
    order: str = "narrative"
    decisions: List[dict] = field(default_factory=list)      # human-readable audit trail of every creative choice
    score: float = 0.0
    meta: Dict[str, float] = field(default_factory=dict)      # pool statistics the later stages reuse (e.g. similarity quantiles)
