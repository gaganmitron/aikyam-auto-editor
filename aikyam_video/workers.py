"""Event-driven workers (Kafka). One process = one role = one job at a time (scale with replicas).

  API  --MediaUploaded-->  orchestrator --TranscriptionRequested / SceneAnalysisRequested--> intelligence
  intelligence --*Completed--> orchestrator --HighlightGenerationRequested--> highlight
  highlight --HighlightGenerationCompleted--> planner --EditPlanGenerated--> orchestrator --RenderRequested--> render
  render --RenderCompleted--> orchestrator (READY)      publisher --MediaPublished--> orchestrator (PUBLISHED)

Stage outputs travel through object storage under `<tenant>/<media>/<job>/artifacts/`, so replicas need no shared disk.
Failures retry (attempt+1, up to MAX_ATTEMPTS) via FAILED -> RETRYING -> PROCESSING; after that the job stays FAILED.
"""
from __future__ import annotations
import dataclasses, json, os, shutil, time
from pathlib import Path
from typing import Callable, Dict, List, Optional
from . import library, stages
from .bus import Bus
from .cost import ProcessingCost, cpu_seconds
from .events import Event, consume
from .jobs import InvalidTransition, JobStore
from .metrics import METRICS, get_logger
from .options import Options, Profile
from .storage import ObjectStorage

MAX_ATTEMPTS = int(os.environ.get("MAX_ATTEMPTS", "3"))
HANDLES: Dict[str, List[str]] = {
    "orchestrator": ["MediaUploaded", "TranscriptionCompleted", "SceneAnalysisCompleted", "HighlightGenerationCompleted",
                     "EditPlanGenerated", "RenderCompleted", "MediaPublished"],
    "intelligence": ["TranscriptionRequested", "SceneAnalysisRequested"],
    "highlight": ["HighlightGenerationRequested"],
    "planner": ["HighlightGenerationCompleted"],
    "render": ["RenderRequested"],
}


def options_to_dict(o: Options) -> dict:
    d = dataclasses.asdict(o); d.pop("on_stage", None); return d


def options_from_dict(d: dict) -> Options:
    d = dict(d or {}); p = d.pop("profile", None)
    return Options(**d, profile=Profile(**p) if p else None)


class Worker:
    def __init__(self, role: str, store: JobStore, storage: ObjectStorage, bus: Bus, workdir: str = "./work", publisher=None, notifier=None):
        assert role in HANDLES, role
        self.publisher, self.notifier = publisher, notifier
        self.role, self.store, self.storage, self.bus, self.workdir = role, store, storage, bus, Path(workdir)

    # ------------------------------------------------------------------ helpers
    def _emit(self, src: Event, type: str, **payload):
        self.bus.publish(Event(type=type, jobId=src.jobId, mediaId=src.mediaId, tenantId=src.tenantId,
                               attempt=src.attempt, payload={**src.payload, **payload}))

    def _prefix(self, ev: Event) -> str:
        return f"{ev.tenantId}/{ev.mediaId}/{ev.jobId}/artifacts/"

    def _opts(self, ev: Event) -> Options:
        return options_from_dict(ev.payload.get("options") or json.loads(self.store.get_job(ev.jobId)["options"] or "{}"))

    def _workdir(self, ev: Event) -> Path:
        d = self.workdir / ev.jobId; d.mkdir(parents=True, exist_ok=True); return d

    def _source(self, ev: Event, d: Path) -> str:
        p = d / "source.mp4"
        if not p.exists():
            self.storage.download(self.store.get_media(ev.mediaId)["storage_key"], str(p))
        return str(p)

    def _pull(self, ev: Event, d: Path, names: List[str]):
        for n in names:
            if not (d / n).exists():
                self.storage.download(self._prefix(ev) + n, str(d / n))

    def _push(self, ev: Event, d: Path, names: List[str]):
        for n in names:
            self.storage.upload(str(d / n), self._prefix(ev) + n)

    def _timed_stage(self, ev: Event, stage: str, model: str, fn):
        c0 = cpu_seconds()
        r = fn()
        self.store.record_stage_cost(ev.jobId, stage, cpu_seconds() - c0, model)
        return r

    # ------------------------------------------------------------------ handlers
    def on_MediaUploaded(self, ev):     # orchestrator
        o = self._opts(ev)
        if self.store.get_job(ev.jobId)["state"] == "UPLOADED":
            self.store.advance(ev.jobId, "PROCESSING")
        self._emit(ev, "TranscriptionRequested", options=options_to_dict(o))
        self._emit(ev, "SceneAnalysisRequested", options=options_to_dict(o))

    def _joined(self, ev, step):
        steps = self.store.mark_step(ev.jobId, step)
        if {"transcription", "scenes"} <= steps and self.store.get_job(ev.jobId)["state"] == "PROCESSING":
            self.store.advance(ev.jobId, "ANALYZED")
            self._emit(ev, "HighlightGenerationRequested")

    def on_TranscriptionCompleted(self, ev): self._joined(ev, "transcription")
    def on_SceneAnalysisCompleted(self, ev): self._joined(ev, "scenes")

    def on_HighlightGenerationCompleted(self, ev):
        if self.role == "orchestrator":
            self.store.advance(ev.jobId, "HIGHLIGHTS_READY")
            return
        d = self._workdir(ev); o = self._opts(ev)     # planner
        src = self._source(ev, d)
        self._pull(ev, d, [stages.TRANSCRIPT, *stages.ANALYSIS_ARTIFACTS, stages.ENTITIES, stages.MOMENTS])
        plan = self._timed_stage(ev, "plan", o.planner, lambda: stages.plan(src, str(d), o))
        self._push(ev, d, [stages.PLAN])
        self._emit(ev, "EditPlanGenerated", durationSeconds=plan["durationSeconds"])

    def on_EditPlanGenerated(self, ev):
        self.store.advance(ev.jobId, "EDIT_PLAN_READY")
        self._emit(ev, "RenderRequested")
        self.store.advance(ev.jobId, "RENDERING")

    def on_TranscriptionRequested(self, ev):
        d = self._workdir(ev); o = self._opts(ev)
        src = self._source(ev, d)
        self._timed_stage(ev, "transcription", f"{o.transcription}", lambda: stages.transcription(src, str(d), o))
        self._push(ev, d, [stages.TRANSCRIPT])
        self._emit(ev, "TranscriptionCompleted")

    def on_SceneAnalysisRequested(self, ev):
        d = self._workdir(ev); o = self._opts(ev)
        src = self._source(ev, d)
        self._timed_stage(ev, "scene_analysis", f"vision:{o.vision}", lambda: stages.scene_analysis(src, str(d), o))
        self._push(ev, d, stages.ANALYSIS_ARTIFACTS)
        library.index_media(self.store, ev.mediaId, ev.tenantId, str(d))
        self._emit(ev, "SceneAnalysisCompleted")

    def on_HighlightGenerationRequested(self, ev):
        d = self._workdir(ev); o = self._opts(ev)
        src = self._source(ev, d)
        self._pull(ev, d, [stages.TRANSCRIPT, *stages.ANALYSIS_ARTIFACTS])
        moments, _ = self._timed_stage(ev, "highlights", "scoring", lambda: stages.highlights(src, str(d), o))
        self._push(ev, d, [stages.ENTITIES, stages.MOMENTS])
        self._emit(ev, "HighlightGenerationCompleted", momentCount=len(moments))

    def on_RenderRequested(self, ev):
        d = self._workdir(ev); o = self._opts(ev)
        src = self._source(ev, d)
        self._pull(ev, d, [stages.TRANSCRIPT, *stages.ANALYSIS_ARTIFACTS, stages.ENTITIES, stages.MOMENTS, stages.PLAN])
        files = self._timed_stage(ev, "render", "ffmpeg", lambda: stages.render(src, str(d), o))
        keys = {"edit_plan": self._prefix(ev) + stages.PLAN}
        for name, p in files.items():
            k = f"{ev.tenantId}/{ev.mediaId}/{ev.jobId}/{Path(p).name}"
            self.storage.upload(p, k); keys[name] = k
        for n in (stages.TRANSCRIPT, stages.SCENES, stages.ENTITIES, stages.MOMENTS):
            keys[n.split(".")[0].replace("edit-plan", "edit_plan")] = self._prefix(ev) + n
        self.store.record_outputs(ev.mediaId, keys)
        plan = json.load(open(d / stages.PLAN))
        cost = self._final_cost(ev, plan, o, str(d / "source.mp4"))
        self._emit(ev, "RenderCompleted", cost=cost)
        shutil.rmtree(d, ignore_errors=True)

    def _final_cost(self, ev, plan, o, src) -> dict:
        from . import ffmpeg as ff
        rows = self.store.stage_costs(ev.jobId)
        c = ProcessingCost(sum(r["cpu_seconds"] for r in rows), sum(r["gpu_seconds"] for r in rows),
                           "+".join(r["model"] for r in rows), ff.probe(src).duration,
                           plan["durationSeconds"] * len(o.formats), outputs=len(o.formats)).finalize()
        self.store.record_cost(ev.jobId, ev.mediaId, c.dict())
        METRICS.cost_inr.inc(c.inr); METRICS.cpu_seconds.inc(c.cpuSeconds)
        return c.dict()

    def on_RenderCompleted(self, ev):
        self.store.advance(ev.jobId, "READY")
        o = self._opts(ev)
        if not o.auto_publish:
            return
        from . import publish
        pub, note = self.publisher, self.notifier
        if pub is None:
            get_logger(ev.jobId, ev.mediaId).warning("auto_publish requested but no publisher configured (AIKYAM_FEED_URL / FEED_FILE)")
            return
        outs = self.store.get_outputs(ev.mediaId)
        with __import__("tempfile").TemporaryDirectory() as td:
            plan = json.load(open(self.storage.download(outs["edit_plan"], os.path.join(td, "p.json"))))
        if publish.auto_publish_ok(plan, o.auto_publish_min_score):
            publish.publish_media(self.store, self.storage, ev.jobId, ev.mediaId, pub, note, self.bus)

    def on_MediaPublished(self, ev):
        self.store.advance(ev.jobId, "PUBLISHED")

    # ------------------------------------------------------------------ loop
    def _fail(self, ev: Event, exc: Exception):
        METRICS.processing_failures.labels(stage=ev.type).inc()
        log = get_logger(ev.jobId, ev.mediaId)
        log.error(f"{self.role} failed on {ev.type} attempt {ev.attempt}: {exc}")
        try:
            self.store.transition(ev.jobId, "FAILED", f"{ev.type}: {exc}"[:1000])
        except InvalidTransition:
            pass
        if self.role != "orchestrator" and ev.attempt < MAX_ATTEMPTS:
            try:
                self.store.transition(ev.jobId, "RETRYING"); self.store.transition(ev.jobId, "PROCESSING")
            except InvalidTransition:
                return
            self.bus.publish(ev.model_copy(update={"attempt": ev.attempt + 1, "eventId": os.urandom(8).hex()}))

    def handle(self, ev: Event) -> bool:
        if ev.type not in HANDLES[self.role]:
            return False
        try:
            return consume(self.store, ev, getattr(self, f"on_{ev.type}"), role=self.role)
        except Exception as e:
            self._fail(ev, e)
            return False

    def run(self, poll: float = 1.0, max_events: Optional[int] = None, idle_exit: bool = False, stop: Callable[[], bool] = lambda: False):
        n, idle = 0, 0
        while not stop() and (max_events is None or n < max_events):
            ev = self.bus.poll(self.role, poll)
            if ev is None:
                idle += 1
                if idle_exit and idle >= 2: break
                continue
            idle = 0
            self.handle(ev); self.bus.commit(self.role, ev); n += 1
        return n


def run_until_idle(workers: List[Worker], max_rounds: int = 1000) -> int:
    """Test/single-process driver: round-robin the roles over a MemoryBus until nothing is left to do."""
    total = 0
    for _ in range(max_rounds):
        moved = sum(w.run(poll=0.0, max_events=1, idle_exit=True) for w in workers)
        total += moved
        if not moved: break
    return total
