"""Publishing + notification (Phase 1 milestone: preview -> publish; Phase 2: auto-publish, notify)."""
from __future__ import annotations
import json, os, time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional
import requests


class PublishProvider(ABC):
    name = "abstract"

    @abstractmethod
    def publish(self, payload: dict) -> dict: ...


class WebhookPublisher(PublishProvider):
    """POST the reel metadata + signed CDN/storage URLs to the Aikyam feed service.
    ASSUMED contract (no Aikyam feed API was available to read): JSON body below, `Authorization: Bearer $AIKYAM_FEED_TOKEN`,
    2xx = accepted. Adapt `payload` in one place when the real API is known."""
    name = "webhook"

    def __init__(self, url: Optional[str] = None, token: Optional[str] = None, timeout: float = 20.0):
        self.url, self.token, self.timeout = url or os.environ["AIKYAM_FEED_URL"], token or os.environ.get("AIKYAM_FEED_TOKEN"), timeout

    def publish(self, payload: dict) -> dict:
        h = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        r = requests.post(self.url, json=payload, headers=h, timeout=self.timeout)
        r.raise_for_status()
        return {"status": r.status_code, "body": r.text[:500]}


class FileFeedPublisher(PublishProvider):
    """Dev/demo feed: appends one JSON line per published reel."""
    name = "file"

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path or os.environ.get("FEED_FILE", "./feed.jsonl"))

    def publish(self, payload: dict) -> dict:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return {"file": str(self.path)}


class Notifier(ABC):
    @abstractmethod
    def notify(self, event: dict) -> None: ...


class WebhookNotifier(Notifier):
    """POST {type, mediaId, templeId, ...} to AIKYAM_NOTIFY_URL (the push/notification service subscribes to it)."""

    def __init__(self, url: Optional[str] = None):
        self.url = url or os.environ["AIKYAM_NOTIFY_URL"]

    def notify(self, event: dict) -> None:
        requests.post(self.url, json=event, timeout=10).raise_for_status()


def from_env():
    pub = WebhookPublisher() if os.environ.get("AIKYAM_FEED_URL") else FileFeedPublisher() if os.environ.get("FEED_FILE") else None
    note = WebhookNotifier() if os.environ.get("AIKYAM_NOTIFY_URL") else None
    return pub, note


def build_payload(store, storage, media_id: str, expires: int = 7 * 24 * 3600) -> dict:
    """Everything the feed needs: ids, metadata, signed URLs (never storage credentials)."""
    m = store.get_media(media_id)
    outs = store.get_outputs(media_id)
    if "reel" not in outs:
        raise ValueError("media has no rendered reel")
    with __import__("tempfile").TemporaryDirectory() as td:
        plan = json.load(open(storage.download(outs["edit_plan"], os.path.join(td, "p.json"))))
    return {"mediaId": media_id, "tenantId": m["tenant_id"], "templeId": plan["source"].get("templeId") or m["temple_id"],
            "deityId": plan["source"].get("deityId"), "ritualId": plan["source"].get("ritualId"),
            "festivalId": plan["source"].get("festivalId"), "title": " · ".join(x for x in (plan["overlays"].get("temple"), plan["overlays"].get("ritual") or plan["overlays"].get("deity")) if x),
            "music": ({k: plan["audio"]["music"].get(k) for k in ("trackId", "title", "licence", "attribution")} if plan["audio"].get("music", {}).get("enabled") else None),
            "credits": [f"Music: {plan['audio']['music'].get('title')} — {plan['audio']['music']['attribution']}"] if plan["audio"].get("music", {}).get("attribution") else [],
            "durationSeconds": plan["durationSeconds"], "captionLanguage": plan["captions"]["language"] if plan["captions"]["enabled"] else None,
            "urls": {k: storage.signed_url(v, expires) for k, v in outs.items() if k in ("reel", "square", "landscape", "thumbnail")},
            "publishedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def qc_status(store, storage, media_id: str) -> Optional[str]:
    """pass | warn | fail | None (no QC report, e.g. classic engine)."""
    key = store.get_outputs(media_id).get("qc")
    if not key:
        return None
    with __import__("tempfile").TemporaryDirectory() as td:
        return json.load(open(storage.download(key, os.path.join(td, "qc.json")))).get("status")


def auto_publish_ok(plan: dict, min_score: float = 0.4) -> bool:
    """Policy for unattended publishing: strong best-moment score, real duration. (Moderation already gates moments.)"""
    return bool(plan["segments"]) and max(s["score"] for s in plan["segments"]) >= min_score and plan["durationSeconds"] >= 5


def publish_media(store, storage, job_id: str, media_id: str, publisher: PublishProvider, notifier: Optional[Notifier] = None, bus=None, force: bool = False) -> dict:
    """READY -> PUBLISHED. Publisher failure leaves the job READY (retry-able); notifier failure never blocks publishing."""
    if store.get_job(job_id)["state"] != "READY":
        raise ValueError("job is not READY")
    qc = qc_status(store, storage, media_id)
    if qc == "fail" and not force:
        raise ValueError("quality control failed for this reel: fix the edit, or publish with force=true")
    payload = build_payload(store, storage, media_id)
    payload["qc"] = qc
    res = publisher.publish(payload)
    store.transition(job_id, "PUBLISHED")
    if bus is not None:
        from .events import Event
        j = store.get_job(job_id)
        bus.publish(Event(type="MediaPublished", jobId=job_id, mediaId=media_id, tenantId=j["tenant_id"], payload={"publisher": publisher.name}))
    if notifier:
        try:
            notifier.notify({"type": "MediaPublished", "mediaId": media_id, "templeId": payload["templeId"], "title": payload["title"]})
        except Exception as e:      # noqa: BLE001 — notification is best effort
            import logging; logging.getLogger("aikyam").warning(f"notify failed: {e}")
    return {"result": res, "payload": payload}
