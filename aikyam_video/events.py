"""Event envelope (section 18). Transport lives in bus.py; handlers are transport-agnostic and idempotent."""
from __future__ import annotations
import datetime as dt, uuid
from typing import Any, Callable, Dict
from pydantic import BaseModel, Field

TOPIC = "media-processing-jobs"
TYPES = ["MediaUploaded", "TranscriptionRequested", "TranscriptionCompleted", "SceneAnalysisRequested",
         "SceneAnalysisCompleted", "HighlightGenerationRequested", "HighlightGenerationCompleted",
         "EditPlanGenerated", "RenderRequested", "RenderCompleted", "MediaPublished"]


class Event(BaseModel):
    eventId: str = Field(default_factory=lambda: uuid.uuid4().hex)
    type: str
    jobId: str
    mediaId: str
    tenantId: str
    timestamp: str = Field(default_factory=lambda: dt.datetime.utcnow().isoformat() + "Z")
    attempt: int = 1
    payload: Dict[str, Any] = Field(default_factory=dict)

    def key(self) -> str:  # dedupe key: same logical event on redelivery / retry of the same attempt
        return f"{self.type}:{self.jobId}:{self.attempt}"


def consume(store, event: Event, handler: Callable[[Event], None], role: str = "") -> bool:
    """Run handler at most once per (role, type, jobId, attempt) *successfully*. Returns False for a duplicate delivery.
    The marker is taken before the handler (so concurrent duplicates lose) and released if the handler raises (so the
    redelivery / retry runs again)."""
    if event.type not in TYPES:
        raise ValueError(f"unknown event type {event.type}")
    key = f"{role}:{event.key()}"
    if store.seen_event(key):
        return False
    try:
        handler(event)
    except BaseException:
        store.forget_event(key)
        raise
    return True
