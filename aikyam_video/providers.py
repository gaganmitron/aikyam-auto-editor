"""Provider abstractions (section 27). Pipeline depends only on these; register new backends in REGISTRY."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Sequence
import numpy as np
from .models import EntityRef, Moment, Transcript, VisionResult


class TranscriptionProvider(ABC):
    name = "abstract"

    @abstractmethod
    def transcribe(self, media_path: str) -> Transcript: ...


class VisionProvider(ABC):
    name = "abstract"

    @abstractmethod
    def analyze(self, frame: np.ndarray) -> VisionResult:
        """frame: BGR uint8 image."""


class EmbeddingProvider(ABC):
    name = "abstract"

    @abstractmethod
    def embed_text(self, texts: Sequence[str]) -> np.ndarray: ...

    @abstractmethod
    def embed_image(self, frame: np.ndarray) -> np.ndarray: ...


class EntityExtractionProvider(ABC):
    name = "abstract"

    @abstractmethod
    def extract(self, text: str, language: str = "") -> List[EntityRef]: ...


class HighlightProvider(ABC):
    """Proposes candidate moments. Output is ALWAYS validated by clip_validation before use."""
    name = "abstract"

    @abstractmethod
    def propose(self, ctx) -> List[Moment]: ...


REGISTRY: Dict[str, Dict[str, Callable[[], object]]] = {
    "transcription": {}, "vision": {}, "embedding": {}, "entity": {}, "highlight": {},
}


def register(kind: str, name: str, factory: Callable[[], object]) -> None:
    REGISTRY[kind][name] = factory


def get(kind: str, name: str):
    try:
        return REGISTRY[kind][name]()
    except KeyError:
        raise ValueError(f"unknown {kind} provider {name!r}; available: {sorted(REGISTRY[kind])}")
