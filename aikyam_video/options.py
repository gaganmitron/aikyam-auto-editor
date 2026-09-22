"""Pipeline options shared by CLI, API and workers."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class Profile:
    """Personalization (Phase 2): a viewer/temple profile biases WHICH moments become the reel and how long it is."""
    preferred_labels: Dict[str, float] = field(default_factory=dict)   # e.g. {"aarti": 1.5, "procession": 0.7}
    preferred_deities: List[str] = field(default_factory=list)          # KG ids
    target_seconds: Optional[float] = None
    caption_lang: Optional[str] = None


@dataclass
class Options:
    transcription: str = "faster-whisper"
    vision: str = "auto"
    audio_tagger: str = "auto"                # auto|clap|none
    whisper_model: Optional[str] = None
    language: Optional[str] = None            # force source language (default: auto-detect)
    translate: bool = False                   # whisper "translate to English"
    caption_lang: Optional[str] = None
    caption_mode: str = "sentence"            # sentence | word
    target_seconds: Optional[float] = None    # None: the pacing profile decides (25-40 s); the classic engine falls back to 45
    scoring_config: Optional[str] = None
    formats: List[str] = field(default_factory=lambda: ["reel", "square", "landscape"])
    video_id: Optional[str] = None
    temple_id: Optional[str] = None           # authoritative hints from upload metadata, e.g. "temple_123"
    location: Optional[str] = None
    festival_id: Optional[str] = None
    music: str = "auto"                       # auto = licensed, topic-matched track when it suits | off
    music_volume: float = 0.5                 # music level before ducking (0.1 quiet .. 1.0 loud); tune by ear
    music_skip_threshold: float = 0.6         # auto: skip if the ORIGINAL audio is already this devotional (bhajan/chant/bell/conch)
    music_web: bool = False                   # find a licence-clean track on the internet (Openverse: CC0 / public domain / CC BY), ranked by CLAP against the reel's content
    music_track: Optional[str] = None         # id of an Aikyam-owned royalty-safe track in MUSIC_LIBRARY_DIR (never auto-picked)
    profile: Optional[Profile] = None
    auto_publish: bool = False                # publish unattended when publish.auto_publish_ok(plan)
    auto_publish_min_score: float = 0.4
    transition: str = "crossfade"             # crossfade | fade (dip to black) | cut
    transition_seconds: float = 0.5
    renderer: str = "ffmpeg"                  # ffmpeg (default, deterministic) | remotion (React layouts) | diffusion (diffusionstudio runtime picture + FFmpeg finishing)
    allow_silent: bool = False                # accept picture-only videos (stock footage; give it sound with --music-file)
    opening: Optional[str] = None             # hook | establish (default: the pacing profile's, hook)
    order: Optional[List[str]] = None         # force the story order by asset id, e.g. ["v3", "v1", "i2"] (one clip per listed asset)
    hook_first: bool = False                  # force the strongest opener to start the reel (measured on real footage: shorter reels, worse opener on one set -> opt-in)
    title: Optional[str] = None               # opening title text (default: the best known ritual / festival / deity / temple name, nothing if none is known)
    subtitle: Optional[str] = None
    edge_snap: bool = True                    # start/stop clips on real cuts and pauses (creative/edges.py); False = the earlier fixed-grid windows
    motion_dedupe: bool = True                # same-looking shots that MOVE differently are different moments (False = appearance-only duplicate rule)
    engine: str = "creative"                  # creative (story-aware editing engine) | classic (greedy score-ordered planner)
    pacing: Optional[str] = None              # contemplative | devotional | festive (None: chosen from the footage)
    reels: int = 1                            # reels per source video (creative engine)
    qc: bool = True                           # post-render quality control + deterministic re-edit
    qc_attempts: int = 2
    planner: str = "deterministic"            # deterministic | llm
    min_source_seconds: float = 5.0
    max_source_seconds: float = 4 * 3600.0
    on_stage: Optional[Callable[[str], None]] = None
