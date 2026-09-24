"""Shared data model (packages/media-model)."""
from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class Word(BaseModel):
    start: float
    end: float
    text: str
    confidence: float = 0.0


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str
    confidence: float = 0.0
    words: List[Word] = Field(default_factory=list)


class Transcript(BaseModel):
    language: str
    language_probability: float = 0.0
    segments: List[TranscriptSegment] = Field(default_factory=list)


class Scene(BaseModel):
    sceneId: str
    start: float
    end: float
    keyframe: str


class VisionResult(BaseModel):
    labels: Dict[str, float] = Field(default_factory=dict)  # label -> confidence 0..1
    faces: List[List[float]] = Field(default_factory=list)  # normalized [x, y, w, h]
    saliency_x: float = 0.5  # normalized x-centre of the most salient region
    brightness: float = 0.0
    sharpness: float = 0.0
    provider: str = ""
    moderation: Dict[str, float] = Field(default_factory=dict)   # unsafe-content label -> confidence
    deities: Dict[str, float] = Field(default_factory=dict)      # KG deityId -> zero-shot confidence (visual guess)
    beats: Optional[Dict[str, float]] = None                     # story-beat distribution (beats.py): what the frame is FOR in the story; None without an image-text model
    no_text: Optional[float] = None                              # zero-shot "no text overlay / title card" score (raw logit difference); low for credits and subtitles
    aesthetic: Optional[float] = None                            # zero-shot look quality (CLIP-IQA-style antonym prompts, raw logit difference); None without an image-text model
    embedding: Optional[List[float]] = Field(default=None, exclude=True)  # image embedding; stored separately


class SceneVision(BaseModel):
    sceneId: str
    start: float
    end: float
    vision: VisionResult


class EntityRef(BaseModel):
    text: str
    name: Optional[str] = None      # canonical display name (KG), e.g. "Chamundeshwari Temple"
    entityType: str
    entityId: str
    confidence: float


class Moment(BaseModel):
    momentId: str
    start: float
    end: float
    reason: str
    score: float
    signals: Dict[str, float] = Field(default_factory=dict)
    entities: List[EntityRef] = Field(default_factory=list)
    sceneIds: List[str] = Field(default_factory=list)


class Rejection(BaseModel):
    start: float
    end: float
    reason: str
