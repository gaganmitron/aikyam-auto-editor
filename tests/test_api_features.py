"""API plumbing for preview/publish/dashboard/search/options. pipeline.run is replaced by a copy of a REAL run's artifacts
(conftest.pipeline_out) so this stays fast; the compute itself is covered by test_pipeline / test_workers."""
import json, os, shutil
import numpy as np, pytest
from fastapi.testclient import TestClient
from aikyam_video.api import create_app
from aikyam_video.jobs import JobStore
from aikyam_video.publish import FileFeedPublisher
from aikyam_video.storage import LocalStorage

H = {"X-API-Key": "k", "X-Tenant-Id": "t1"}


class Inline:
    def submit(self, fn, *a): fn(*a)


@pytest.fixture
def env(tmp_path, monkeypatch, pipeline_out, sample):
    out, files = pipeline_out
    monkeypatch.setenv("AIKYAM_API_KEY", "k")
    monkeypatch.setattr("aikyam_video.api.WORK", tmp_path / "work")
    seen = {}
    def fake_run(path, dest, upto, o, log=None):
        seen["options"] = o
        shutil.copytree(out, dest, dirs_exist_ok=True)
        return {k: os.path.join(dest, os.path.relpath(v, out)) for k, v in files.items()}
    monkeypatch.setattr("aikyam_video.api.pipeline.run", fake_run)
    feed = tmp_path / "feed.jsonl"
    class Emb:  # query embedding pointing at whatever the first indexed scene is
        def embed_text(self, qs): return np.array([np.array(json.load(open(f"{out}/embeddings.json"))["scene_0"])])
    st = JobStore(f"sqlite:///{tmp_path}/a.db")
    app = create_app(st, LocalStorage(str(tmp_path / "s")), Inline(), bus=None, publisher=FileFeedPublisher(str(feed)), notifier=None, embedder=Emb())
    c = TestClient(app)
    mid = c.post("/v1/media", headers=H, files={"file": ("s.mp4", open(sample, "rb"), "video/mp4")}, data={"templeId": "temple_123"}).json()["mediaId"]
    return c, mid, seen, feed, st


def test_options_validation_and_passthrough(env):
    c, mid, seen, *_ = env
    for bad in ({"music_track": "../../etc/passwd"}, {"target_seconds": 500}, {"planner": "evil"}, {"caption_lang": "kannada"}, {"formats": ["gif"]}):
        assert c.post(f"/v1/media/{mid}/process", headers=H, json=bad).status_code == 422, bad
    r = c.post(f"/v1/media/{mid}/process", headers=H, json={"target_seconds": 30, "festival_id": "festival_9", "caption_lang": "kn",
                                                              "profile": {"preferred_labels": {"aarti": 1.5}, "target_seconds": 20}})
    assert r.status_code == 200
    o = seen["options"]
    assert o.target_seconds == 30 and o.festival_id == "festival_9" and o.caption_lang == "kn" and o.profile.preferred_labels == {"aarti": 1.5} and o.temple_id == "temple_123"


def test_preview_then_publish_flow(env):
    c, mid, seen, feed, st = env
    assert c.post(f"/v1/media/{mid}/publish", headers=H).status_code == 409           # nothing READY yet
    jid = c.post(f"/v1/media/{mid}/process", headers=H, json={}).json()["jobId"]
    assert st.get_job(jid)["state"] == "READY"
    url = c.post(f"/v1/media/{mid}/preview-link", headers=H).json()["url"]
    page = c.get(url); assert page.status_code == 200 and "<video" in page.text and "Publish to Aikyam feed" in page.text
    tok = url.split("t=")[1]
    assert c.get(f"/preview/{mid}/asset/reel?t={tok}").headers["content-type"] == "video/mp4"      # playable through the token-guarded route
    assert c.get(f"/preview/{mid}?t={tok}x").status_code == 403                                     # tampered
    assert c.get(f"/preview/other_media?t={tok}").status_code == 403                                # token is scoped to one media
    assert c.post(f"/preview/{mid}/publish?t={tok}").status_code == 200
    assert st.get_job(jid)["state"] == "PUBLISHED" and json.loads(feed.read_text().splitlines()[0])["mediaId"] == mid
    assert c.post(f"/v1/media/{mid}/publish", headers=H).status_code == 409                        # already published: not READY any more


def test_expired_link_rejected(env, monkeypatch):
    c, mid, *_ = env
    c.post(f"/v1/media/{mid}/process", headers=H, json={})
    import time
    url = c.post(f"/v1/media/{mid}/preview-link", headers=H).json()["url"]
    real = time.time; monkeypatch.setattr("aikyam_video.api.time.time", lambda: real() + 3600)
    assert c.get(url).status_code == 403


def test_auto_publish(env):
    c, mid, seen, feed, st = env
    jid = c.post(f"/v1/media/{mid}/process", headers=H, json={"auto_publish": True}).json()["jobId"]
    assert st.get_job(jid)["state"] in ("PUBLISHED", "READY")                  # PUBLISHED when the best moment clears the policy
    if st.get_job(jid)["state"] == "PUBLISHED":
        assert feed.exists()


def test_dashboard_cost_and_tenant_scoping(env):
    c, mid, *_ = env
    c.post(f"/v1/media/{mid}/process", headers=H, json={})
    url = c.post("/v1/dashboard-link", headers=H).json()["url"]
    page = c.get(url).text
    assert mid in page and "READY" in page and "/ source hour" in page and "₹" in page
    other = c.post("/v1/dashboard-link", headers={**H, "X-Tenant-Id": "t2"}).json()["url"]
    assert mid not in c.get(other).text                                        # other tenant sees nothing
    assert c.get("/dashboard?t=garbage").status_code == 403


def test_search_similar_embedding_duplicate(env, sample):
    c, mid, *_ = env
    c.post(f"/v1/media/{mid}/process", headers=H, json={})
    r = c.get("/v1/search", headers=H, params={"q": "evening aarti"}).json()
    assert r["results"] and r["results"][0]["mediaId"] == mid and r["results"][0]["score"] > 0.9
    assert c.get("/v1/search", headers={**H, "X-Tenant-Id": "t2"}, params={"q": "evening aarti"}).json()["results"] == []
    assert len(c.get(f"/v1/media/{mid}/embedding", headers=H).json()["vector"]) > 100
    mid2 = c.post("/v1/media", headers=H, files={"file": ("s.mp4", open(sample, "rb"), "video/mp4")}).json()["mediaId"]   # re-upload of the same video
    c.post(f"/v1/media/{mid2}/process", headers=H, json={})
    assert c.get(f"/v1/media/{mid2}", headers=H).json()["duplicateOf"] == mid
    assert c.get(f"/v1/media/{mid}/similar", headers=H).json()["similar"][0]["mediaId"] == mid2
