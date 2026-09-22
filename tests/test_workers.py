"""Event-driven pipeline over MemoryBus with the REAL stages (the same worker code Kafka drives)."""
import json, os, shutil
import pytest
from aikyam_video import stages
from aikyam_video.bus import MemoryBus
from aikyam_video.events import Event
from aikyam_video.jobs import JobStore
from aikyam_video.options import Options
from aikyam_video.publish import FileFeedPublisher
from aikyam_video.storage import LocalStorage
from aikyam_video.workers import Worker, options_to_dict, run_until_idle

def fake_scene_analysis(src, art, o):
    """Light stand-in that still writes every artifact the real stage does (the worker uploads them)."""
    import numpy as np
    from aikyam_video.audio import AudioProfile
    z = np.zeros(4)
    for n, v in (("scenes.json", []), ("vision.json", []), ("embeddings.json", {}), ("black.json", [])): json.dump(v, open(f"{art}/{n}", "w"))
    AudioProfile(0.5, z, z.astype(bool), z.astype(bool), z, {}).save(f"{art}/audio.npz")
    return []


ROLES = ["orchestrator", "intelligence", "highlight", "planner", "render"]


def make_env(tmp_path, sample, opts, publisher=None, notifier=None):
    st = JobStore(f"sqlite:///{tmp_path}/w.db"); sto = LocalStorage(str(tmp_path / "s")); bus = MemoryBus()
    sto.upload(sample, "ten/vid1/source.mp4"); st.create_media("vid1", "ten", "ten/vid1/source.mp4", opts.temple_id)
    job = st.create_job("vid1", "ten", "k1", options_to_dict(opts))
    ws = [Worker(r, st, sto, bus, str(tmp_path / "work"), publisher, notifier) for r in ROLES]
    bus.publish(Event(type="MediaUploaded", jobId=job["job_id"], mediaId="vid1", tenantId="ten", payload={"options": options_to_dict(opts)}))
    return st, sto, bus, ws, job["job_id"]


def test_full_event_chain_publishes_and_records_everything(tmp_path, sample):
    feed = tmp_path / "feed.jsonl"; notes = []
    class Note:
        def notify(self, e): notes.append(e)
    opts = Options(temple_id="temple_123", formats=["reel"], auto_publish=True, auto_publish_min_score=0.0)
    st, sto, bus, ws, jid = make_env(tmp_path, sample, opts, FileFeedPublisher(str(feed)), Note())
    run_until_idle(ws)
    job = st.get_job(jid)
    assert job["state"] == "PUBLISHED", job
    types = [e.type for e in bus.log]
    order = [types.index(t) for t in ["MediaUploaded", "TranscriptionRequested", "HighlightGenerationRequested", "HighlightGenerationCompleted", "EditPlanGenerated", "RenderRequested", "RenderCompleted", "MediaPublished"]]
    assert order == sorted(order)                                          # causal order of the state machine
    assert all(e.jobId == jid and e.mediaId == "vid1" and e.tenantId == "ten" and e.attempt == 1 and e.timestamp for e in bus.log)   # envelope
    outs = st.get_outputs("vid1")
    assert {"reel", "thumbnail", "edit_plan", "moments", "transcript"} <= set(outs) and sto.exists(outs["reel"])
    reel_local = tmp_path / "reel.mp4"; sto.download(outs["reel"], str(reel_local))
    from aikyam_video import ffmpeg as ff
    assert ff.probe(str(reel_local)).height == 1920 and ff.probe(str(reel_local)).has_audio
    stages_seen = {r["stage"] for r in st.stage_costs(jid)}
    assert stages_seen == {"transcription", "scene_analysis", "highlights", "plan", "render"}      # cost recorded per stage
    from sqlalchemy import select
    from aikyam_video.jobs import processing_cost
    with st.e.connect() as c: pc = c.execute(select(processing_cost)).mappings().one()
    assert pc["cpu_seconds"] > 0 and pc["inr_per_reel"] >= 0 and pc["input_duration_seconds"] > 0
    line = json.loads(feed.read_text().splitlines()[0])
    assert line["templeId"] == "temple_123" and "Chamundeshwari Temple" in line["title"] and {"reel", "thumbnail"} <= set(line["urls"])
    assert notes and notes[0]["type"] == "MediaPublished"
    # library indexed at analysis time
    from aikyam_video import library
    assert library.video_vector(st, "vid1") is not None


def test_duplicate_delivery_is_processed_once(tmp_path, sample, monkeypatch):
    calls = {"t": 0}
    def fake_transcribe(src, art, o):
        calls["t"] += 1; json.dump({"language": "en", "language_probability": 1, "segments": []}, open(f"{art}/transcript.json", "w"))
    monkeypatch.setattr(stages, "transcription", fake_transcribe)
    monkeypatch.setattr(stages, "scene_analysis", fake_scene_analysis)
    st, sto, bus, ws, jid = make_env(tmp_path, sample, Options(formats=["reel"]))
    intel = next(w for w in ws if w.role == "intelligence")
    ws[0].run(poll=0, max_events=1, idle_exit=True)                         # orchestrator -> requests
    req = next(e for e in bus.log if e.type == "TranscriptionRequested")
    bus.publish(req.model_copy(update={"eventId": "redelivered"}))          # broker redelivers the same logical event
    intel.run(poll=0, idle_exit=True)
    assert calls["t"] == 1


def test_failure_retries_then_succeeds(tmp_path, sample, monkeypatch):
    calls = {"n": 0}
    def flaky(src, art, o):
        calls["n"] += 1
        if calls["n"] == 1: raise RuntimeError("transient GPU hiccup")
        json.dump({"language": "en", "language_probability": 1, "segments": []}, open(f"{art}/transcript.json", "w"))
    monkeypatch.setattr(stages, "transcription", flaky)
    monkeypatch.setattr(stages, "scene_analysis", fake_scene_analysis)
    st, sto, bus, ws, jid = make_env(tmp_path, sample, Options(formats=["reel"]))
    intel = next(w for w in ws if w.role == "intelligence")
    ws[0].run(poll=0, max_events=1, idle_exit=True); intel.run(poll=0, idle_exit=True)
    j = st.get_job(jid)
    assert calls["n"] == 2 and j["attempt"] == 2 and j["state"] == "PROCESSING"
    assert any(e.type == "TranscriptionCompleted" and e.attempt == 2 for e in bus.log)


def test_failure_gives_up_after_max_attempts(tmp_path, sample, monkeypatch):
    monkeypatch.setattr(stages, "transcription", lambda *a: (_ for _ in ()).throw(RuntimeError("model missing")))
    monkeypatch.setattr(stages, "scene_analysis", fake_scene_analysis)
    monkeypatch.setattr("aikyam_video.workers.MAX_ATTEMPTS", 3)
    st, sto, bus, ws, jid = make_env(tmp_path, sample, Options(formats=["reel"]))
    intel = next(w for w in ws if w.role == "intelligence")
    ws[0].run(poll=0, max_events=1, idle_exit=True); intel.run(poll=0, idle_exit=True)
    j = st.get_job(jid)
    assert j["state"] == "FAILED" and "model missing" in j["error"] and j["attempt"] == 3
    assert sum(e.type == "TranscriptionRequested" for e in bus.log) == 3          # 1 original + 2 retries, then stop


@pytest.mark.skipif(not os.environ.get("KAFKA_BOOTSTRAP"), reason="set KAFKA_BOOTSTRAP to test against a real broker (e.g. Redpanda)")
def test_kafka_bus_roundtrip_and_consumer_groups():
    import uuid
    from aikyam_video.bus import KafkaBus
    topic = f"test-{uuid.uuid4().hex[:8]}"
    b = KafkaBus(topic=topic, partitions=2)
    try:
        ev = Event(type="MediaUploaded", jobId="j1", mediaId="m1", tenantId="t", payload={"x": 1})
        b.publish(ev)
        got = None
        for _ in range(30):
            got = b.poll("groupA", 1.0)
            if got: break
        assert got and got.jobId == "j1" and got.payload == {"x": 1}
        b.commit("groupA", got)
        other = None                                                               # a different group sees the same event
        for _ in range(30):
            other = b.poll("groupB", 1.0)
            if other: break
        assert other and other.eventId == ev.eventId
        assert b.poll("groupA", 1.0) is None                                       # committed: not redelivered
    finally:
        b.close()


@pytest.mark.skipif(not os.environ.get("KAFKA_BOOTSTRAP"), reason="set KAFKA_BOOTSTRAP to test against a real broker (e.g. Redpanda)")
def test_kafka_end_to_end_all_roles_real_stages(tmp_path, sample):
    """The production topology: 5 roles = 5 consumer groups on ONE Kafka topic, real stages, real storage/DB, until PUBLISHED."""
    import threading, time, uuid
    from aikyam_video import bus as bus_mod
    topic = f"e2e-{uuid.uuid4().hex[:8]}"
    opts = Options(temple_id="temple_123", formats=["reel"], auto_publish=True, auto_publish_min_score=0.0)
    st = JobStore(f"sqlite:///{tmp_path}/k.db"); sto = LocalStorage(str(tmp_path / "s"))
    sto.upload(sample, "ten/vid1/source.mp4"); st.create_media("vid1", "ten", "ten/vid1/source.mp4", "temple_123")
    job = st.create_job("vid1", "ten", "k1", options_to_dict(opts)); jid = job["job_id"]
    feed = tmp_path / "feed.jsonl"
    stop = threading.Event(); workers = []
    for r in ROLES:                                               # each role: its own KafkaBus (own producer + consumer group)
        w = Worker(r, st, sto, bus_mod.KafkaBus(topic=topic), str(tmp_path / "work"), FileFeedPublisher(str(feed)))
        workers.append(w); threading.Thread(target=w.run, kwargs={"poll": 0.5, "stop": stop.is_set}, daemon=True).start()
    try:
        starter = bus_mod.KafkaBus(topic=topic)
        starter.publish(Event(type="MediaUploaded", jobId=jid, mediaId="vid1", tenantId="ten", payload={"options": options_to_dict(opts)}))
        t0 = time.time()
        while time.time() - t0 < 420 and st.get_job(jid)["state"] not in ("PUBLISHED", "FAILED"):
            time.sleep(2)
        j = st.get_job(jid)
        assert j["state"] == "PUBLISHED", j
        assert feed.exists() and json.loads(feed.read_text().splitlines()[0])["mediaId"] == "vid1"
        assert {"reel", "thumbnail"} <= set(st.get_outputs("vid1"))
    finally:
        stop.set(); time.sleep(1.5)
        for w in workers: w.bus.close()
