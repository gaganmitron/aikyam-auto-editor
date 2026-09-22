"""Engine-level behaviour on a REAL run (conftest.pipeline_out uses the default = creative engine): plan v2 contract, decisions audit trail,
QC report, creative-vs-classic switch, and the publish gate."""
import json, os
import pytest
from aikyam_video import ffmpeg as ff, pipeline
from aikyam_video.plan import edge_overlaps, timeline, validate_plan


def load(p): return json.load(open(p))


def test_creative_plan_is_v2_valid_and_explains_itself(pipeline_out, sample):
    out, _ = pipeline_out; plan = load(f"{out}/edit-plan.json"); info = ff.probe(sample)
    validate_plan(plan, info.duration)
    c = plan["creative"]
    assert c["version"] == 2 and c["engine"] == "creative" and c["profile"] in ("contemplative", "devotional", "festive") and c["arc"]
    types = {d["type"] for d in c["decisions"]}
    assert {"plan", "select", "music"} <= types                                       # every choice is on the record
    assert all(s.get("role") for s in plan["segments"]) and all("subjectPath" in s for s in plan["segments"])
    assert plan["durationSeconds"] > 0 and c["outro"]["fadeSeconds"] > 0


def test_qc_report_is_written_and_the_reel_is_not_blindly_published(pipeline_out):
    out, files = pipeline_out
    qc = load(f"{out}/qc.json"); assert qc["status"] in ("pass", "warn", "fail") and qc["checks"] and qc["attempts"]
    names = {c["name"] for c in qc["checks"]}
    assert {"duration", "loudness", "true_peak", "clipping", "black_frames", "abrupt_transition"} <= names
    assert qc["stats"]["true_peak_db"] <= -1.0 and abs(qc["stats"]["lufs"] - (-16.0)) <= 2.0
    assert "qc" in files


def test_classic_engine_is_still_available(tmp_path, sample):
    files = pipeline.run(sample, str(tmp_path / "c"), "plan", pipeline.Options(engine="classic", music="off"))
    p = load(files["edit_plan"]); assert "creative" not in p and p["segments"]


def test_publish_is_blocked_when_qc_failed(tmp_path):
    from aikyam_video.jobs import JobStore
    from aikyam_video.publish import FileFeedPublisher, publish_media
    from aikyam_video.storage import LocalStorage
    st = JobStore(f"sqlite:///{tmp_path}/p.db"); sto = LocalStorage(str(tmp_path / "s")); st.create_media("m1", "t", "k", "temple_123")
    j = st.create_job("m1", "t", "k1"); [st.transition(j["job_id"], s) for s in ("PROCESSING", "ANALYZED", "HIGHLIGHTS_READY", "EDIT_PLAN_READY", "RENDERING", "READY")]
    plan = {"source": {"templeId": "temple_123"}, "overlays": {"temple": "T"}, "durationSeconds": 20, "captions": {"enabled": False}, "audio": {}}
    (tmp_path / "plan.json").write_text(json.dumps(plan)); (tmp_path / "r.mp4").write_bytes(b"x"); (tmp_path / "qc.json").write_text(json.dumps({"status": "fail", "checks": []}))
    for n, f in (("edit_plan", "plan.json"), ("reel", "r.mp4"), ("qc", "qc.json")): sto.upload(str(tmp_path / f), f"t/{f}")
    st.record_outputs("m1", {"edit_plan": "t/plan.json", "reel": "t/r.mp4", "qc": "t/qc.json"})
    pub = FileFeedPublisher(str(tmp_path / "feed.jsonl"))
    with pytest.raises(ValueError, match="quality control failed"): publish_media(st, sto, j["job_id"], "m1", pub)
    assert not (tmp_path / "feed.jsonl").exists() and st.get_job(j["job_id"])["state"] == "READY"       # nothing published, job stays READY
    r = publish_media(st, sto, j["job_id"], "m1", pub, force=True); assert r["payload"]["qc"] == "fail" and st.get_job(j["job_id"])["state"] == "PUBLISHED"


def test_reel_audio_meets_the_loudness_target(pipeline_out):
    from aikyam_video import measure as M
    out, _ = pipeline_out; L = M.loudness(M.decode_audio(f"{out}/reel-9x16.mp4"))
    assert abs(L["lufs"] + 16.0) < 1.5 and L["true_peak_db"] <= -1.0 and L["clipped"] == 0
