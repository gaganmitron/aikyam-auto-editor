"""Prometheus metrics (section 22) + structured logging."""
from __future__ import annotations
import json, logging, sys, time
from contextlib import contextmanager
from prometheus_client import Counter, Histogram, CollectorRegistry, generate_latest

B = (1, 5, 15, 30, 60, 120, 300, 600, 1800, 3600)


class Metrics:
    def __init__(self, registry: CollectorRegistry = None):
        r = self.registry = registry or CollectorRegistry()
        h = lambda n, d: Histogram(n, d, buckets=B, registry=r)
        self.media_processing = h("media_processing_duration_seconds", "End-to-end job time")
        self.transcription = h("transcription_duration_seconds", "Transcription time")
        self.scene_detection = h("scene_detection_duration_seconds", "Scene detection time")
        self.highlight_generation = h("highlight_generation_duration_seconds", "Highlight generation time")
        self.render = h("render_duration_seconds", "Render time per output")
        self.processing_failures = Counter("media_processing_failures_total", "Failed jobs", ["stage"], registry=r)
        self.render_failures = Counter("render_failures_total", "Failed renders", registry=r)
        self.clips_generated = Counter("clips_generated_total", "Accepted moments", registry=r)
        self.clips_rejected = Counter("clips_rejected_total", "Rejected candidates", ["reason"], registry=r)
        self.source_seconds = Counter("source_video_duration_seconds", "Source seconds processed", registry=r)
        self.output_seconds = Counter("output_video_duration_seconds", "Output seconds rendered", registry=r)
        self.cost_inr = Counter("processing_cost_inr_total", "Estimated processing cost (INR)", registry=r)
        self.cpu_seconds = Counter("processing_cpu_seconds_total", "CPU seconds used", registry=r)

    def expose(self) -> bytes:
        return generate_latest(self.registry)


METRICS = Metrics()


@contextmanager
def timed(hist):
    t = time.perf_counter()
    try:
        yield
    finally:
        hist.observe(time.perf_counter() - t)


class JsonFormatter(logging.Formatter):
    def format(self, r):
        d = {"ts": self.formatTime(r), "level": r.levelname, "msg": r.getMessage()}
        for k in ("jobId", "mediaId", "templeId", "stage"):
            if hasattr(r, k): d[k] = getattr(r, k)
        return json.dumps(d, ensure_ascii=False)


def get_logger(job_id="", media_id="", temple_id="") -> logging.LoggerAdapter:
    lg = logging.getLogger("aikyam")
    if not lg.handlers:
        h = logging.StreamHandler(sys.stderr); h.setFormatter(JsonFormatter()); lg.addHandler(h); lg.setLevel(logging.INFO)
    return logging.LoggerAdapter(lg, {"jobId": job_id, "mediaId": media_id, "templeId": temple_id or ""})
