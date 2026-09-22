"""Job state machine + persistence (sections 18, 19). SQLAlchemy Core: Postgres in prod (DATABASE_URL), SQLite locally."""
from __future__ import annotations
import datetime as dt, json, os, uuid
from typing import Dict, Optional, Set
from sqlalchemy import LargeBinary, Column, text, DateTime, Float, Integer, MetaData, String, Table, Text, create_engine, select, update
from sqlalchemy.exc import IntegrityError

TRANSITIONS: Dict[str, Set[str]] = {
    "UPLOADED": {"PROCESSING"},
    "PROCESSING": {"ANALYZED", "FAILED"},
    "ANALYZED": {"HIGHLIGHTS_READY", "FAILED"},
    "HIGHLIGHTS_READY": {"EDIT_PLAN_READY", "FAILED"},
    "EDIT_PLAN_READY": {"RENDERING", "FAILED"},
    "RENDERING": {"READY", "FAILED"},
    "READY": {"PUBLISHED"},
    "FAILED": {"RETRYING"},
    "RETRYING": {"PROCESSING"},
    "PUBLISHED": set(),
}


class InvalidTransition(RuntimeError):
    pass


md = MetaData()
media = Table("media", md, Column("media_id", String(64), primary_key=True), Column("tenant_id", String(64)),
              Column("temple_id", String(64)), Column("storage_key", Text), Column("state", String(32)),
              Column("meta", Text), Column("created_at", DateTime))
job_steps = Table("job_steps", md, Column("job_id", String(64), primary_key=True), Column("step", String(64), primary_key=True))
cost_stage = Table("cost_stage", md, Column("job_id", String(64), primary_key=True), Column("stage", String(64), primary_key=True),
                   Column("cpu_seconds", Float), Column("gpu_seconds", Float), Column("model", String(128)))
jobs = Table("jobs", md, Column("job_id", String(64), primary_key=True), Column("media_id", String(64), index=True),
             Column("tenant_id", String(64)), Column("state", String(32)), Column("attempt", Integer, default=1),
             Column("error", Text), Column("idempotency_key", String(128), unique=True), Column("options", Text),
             Column("created_at", DateTime), Column("updated_at", DateTime))
processing_cost = Table("processing_cost", md, Column("job_id", String(64), primary_key=True), Column("media_id", String(64)),
                        Column("cpu_seconds", Float), Column("gpu_seconds", Float), Column("model", String(64)),
                        Column("input_duration_seconds", Float), Column("output_duration_seconds", Float),
                        Column("inr", Float), Column("inr_per_source_hour", Float), Column("inr_per_reel", Float))
scene_embeddings = Table("scene_embeddings", md, Column("media_id", String(64), primary_key=True), Column("scene_id", String(64), primary_key=True),
                         Column("tenant_id", String(64), index=True), Column("start", Float), Column("end", Float),
                         Column("vec", LargeBinary), Column("labels", Text))
video_embeddings = Table("video_embeddings", md, Column("media_id", String(64), primary_key=True), Column("tenant_id", String(64), index=True),
                         Column("duration", Float), Column("vec", LargeBinary), Column("duplicate_of", String(64)))
processed_events = Table("processed_events", md, Column("event_id", String(128), primary_key=True), Column("at", DateTime))
outputs = Table("outputs", md, Column("media_id", String(64), index=True), Column("name", String(64)), Column("storage_key", Text))

now = lambda: dt.datetime.utcnow()


class JobStore:
    def __init__(self, url: Optional[str] = None):
        self.e = create_engine(url or os.environ.get("DATABASE_URL", "sqlite:///aikyam.db"), future=True)
        # Many processes (API + workers) start at once: serialise schema creation with a Postgres advisory lock,
        # otherwise concurrent CREATE TABLEs race (UniqueViolation on pg_type).
        with self.e.begin() as c:
            if self.e.dialect.name == "postgresql":
                c.execute(text("SELECT pg_advisory_xact_lock(727274)"))
            md.create_all(c)

    def create_media(self, media_id, tenant_id, storage_key, temple_id=None, meta=None):
        with self.e.begin() as c:
            c.execute(media.insert().values(media_id=media_id, tenant_id=tenant_id, temple_id=temple_id,
                                            storage_key=storage_key, state="UPLOADED", meta=json.dumps(meta or {}),
                                            created_at=now()))

    def get_media(self, media_id):
        with self.e.connect() as c:
            r = c.execute(select(media).where(media.c.media_id == media_id)).mappings().first()
            return dict(r) if r else None

    def list_media(self, tenant_id: str, temple_id: Optional[str] = None, limit: int = 200) -> list:
        q = select(media).where(media.c.tenant_id == tenant_id).order_by(media.c.created_at.desc()).limit(limit)
        if temple_id:
            q = q.where(media.c.temple_id == temple_id)
        with self.e.connect() as c:
            rows = [dict(r) for r in c.execute(q).mappings()]
            for r in rows:
                jb = c.execute(select(jobs).where(jobs.c.media_id == r["media_id"]).order_by(jobs.c.created_at.desc())).mappings().first()
                r["job"] = dict(jb) if jb else None
                pc = c.execute(select(processing_cost).where(processing_cost.c.job_id == (jb["job_id"] if jb else ""))).mappings().first()
                r["cost"] = dict(pc) if pc else None
                ve = c.execute(select(video_embeddings.c.duplicate_of).where(video_embeddings.c.media_id == r["media_id"])).first()
                r["duplicate_of"] = ve[0] if ve else None
        return rows

    def create_job(self, media_id, tenant_id, idempotency_key: Optional[str] = None, options: Optional[dict] = None) -> dict:
        """Idempotent: the same key returns the existing job instead of creating a duplicate."""
        key = idempotency_key or f"process:{media_id}:1"
        jid = f"job_{uuid.uuid4().hex[:12]}"
        try:
            with self.e.begin() as c:
                c.execute(jobs.insert().values(job_id=jid, media_id=media_id, tenant_id=tenant_id, state="UPLOADED",
                                               attempt=1, idempotency_key=key, options=json.dumps(options or {}),
                                               created_at=now(), updated_at=now()))
        except IntegrityError:
            pass
        return self.get_job_by_key(key)

    def get_job_by_key(self, key):
        with self.e.connect() as c:
            return dict(c.execute(select(jobs).where(jobs.c.idempotency_key == key)).mappings().one())

    def get_job(self, job_id):
        with self.e.connect() as c:
            r = c.execute(select(jobs).where(jobs.c.job_id == job_id)).mappings().first()
            return dict(r) if r else None

    def transition(self, job_id: str, new: str, error: Optional[str] = None) -> str:
        """Compare-and-set so concurrent workers cannot both move a job."""
        with self.e.begin() as c:
            cur = c.execute(select(jobs.c.state, jobs.c.attempt).where(jobs.c.job_id == job_id)).one()
            if new not in TRANSITIONS[cur.state]:
                raise InvalidTransition(f"{cur.state} -> {new}")
            attempt = cur.attempt + 1 if new == "RETRYING" else cur.attempt
            n = c.execute(update(jobs).where(jobs.c.job_id == job_id, jobs.c.state == cur.state)
                          .values(state=new, error=error, attempt=attempt, updated_at=now())).rowcount
            if n != 1:
                raise InvalidTransition(f"concurrent update on {job_id}")
        return new

    CHAIN = ["UPLOADED", "PROCESSING", "ANALYZED", "HIGHLIGHTS_READY", "EDIT_PLAN_READY", "RENDERING", "READY", "PUBLISHED"]

    def advance(self, job_id: str, target: str) -> str:
        """Walk the happy-path chain forward to `target` (no-op if already there or beyond; never moves backwards)."""
        cur = self.get_job(job_id)["state"]
        if cur in self.CHAIN and self.CHAIN.index(cur) < self.CHAIN.index(target):
            for s in self.CHAIN[self.CHAIN.index(cur) + 1: self.CHAIN.index(target) + 1]:
                self.transition(job_id, s)
        return self.get_job(job_id)["state"]

    def mark_step(self, job_id: str, step: str) -> set:
        try:
            with self.e.begin() as c:
                c.execute(job_steps.insert().values(job_id=job_id, step=step))
        except IntegrityError:
            pass
        with self.e.connect() as c:
            return {r.step for r in c.execute(select(job_steps).where(job_steps.c.job_id == job_id))}

    def forget_event(self, event_id: str) -> None:
        with self.e.begin() as c:
            c.execute(processed_events.delete().where(processed_events.c.event_id == event_id))

    def record_stage_cost(self, job_id: str, stage: str, cpu: float, model: str, gpu: float = 0.0) -> None:
        with self.e.begin() as c:
            c.execute(cost_stage.delete().where((cost_stage.c.job_id == job_id) & (cost_stage.c.stage == stage)))
            c.execute(cost_stage.insert().values(job_id=job_id, stage=stage, cpu_seconds=cpu, gpu_seconds=gpu, model=model))

    def stage_costs(self, job_id: str) -> list:
        with self.e.connect() as c:
            return [dict(r) for r in c.execute(select(cost_stage).where(cost_stage.c.job_id == job_id)).mappings()]

    def seen_event(self, event_id: str) -> bool:
        """True if already processed (idempotent consumers); records it otherwise."""
        try:
            with self.e.begin() as c:
                c.execute(processed_events.insert().values(event_id=event_id, at=now()))
            return False
        except IntegrityError:
            return True

    def record_cost(self, job_id, media_id, cost: dict):
        with self.e.begin() as c:
            c.execute(processing_cost.delete().where(processing_cost.c.job_id == job_id))
            c.execute(processing_cost.insert().values(
                job_id=job_id, media_id=media_id, cpu_seconds=cost["cpuSeconds"], gpu_seconds=cost["gpuSeconds"],
                model=cost["model"], input_duration_seconds=cost["inputDurationSeconds"],
                output_duration_seconds=cost["outputDurationSeconds"], inr=cost["inr"],
                inr_per_source_hour=cost["inrPerSourceHour"], inr_per_reel=cost["inrPerReel"]))

    def record_outputs(self, media_id, files: Dict[str, str]):
        with self.e.begin() as c:
            c.execute(outputs.delete().where(outputs.c.media_id == media_id))
            for n, k in files.items():
                c.execute(outputs.insert().values(media_id=media_id, name=n, storage_key=k))

    def get_outputs(self, media_id) -> Dict[str, str]:
        with self.e.connect() as c:
            return {r.name: r.storage_key for r in c.execute(select(outputs).where(outputs.c.media_id == media_id))}
