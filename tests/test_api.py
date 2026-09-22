import os
import pytest
from fastapi.testclient import TestClient
from aikyam_video.api import create_app
from aikyam_video.jobs import JobStore
from aikyam_video.storage import LocalStorage


class Inline:  # run jobs synchronously in tests
    def submit(self, fn, *a): fn(*a)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("AIKYAM_API_KEY", "k")
    monkeypatch.setattr("aikyam_video.api.WORK", tmp_path / "work")
    return TestClient(create_app(JobStore(f"sqlite:///{tmp_path}/a.db"), LocalStorage(str(tmp_path / "s")), Inline()))


H = {"X-API-Key": "k", "X-Tenant-Id": "t1"}


def test_auth_required(client):
    assert client.get("/v1/media/x").status_code == 401
    assert client.get("/v1/media/x", headers={"X-API-Key": "k"}).status_code == 400
    assert client.get("/v1/media/x", headers=H).status_code == 404


def test_fail_closed_without_key(client, monkeypatch):
    monkeypatch.delenv("AIKYAM_API_KEY")
    assert client.get("/v1/media/x", headers=H).status_code == 503


def test_upload_rejects_bad_mime_and_fake_video(client):
    r = client.post("/v1/media", headers=H, files={"file": ("a.txt", b"hi", "text/plain")})
    assert r.status_code == 415
    r = client.post("/v1/media", headers=H, files={"file": ("a.mp4", b"not video", "video/mp4")})
    assert r.status_code == 422


def test_full_flow(client, sample):
    r = client.post("/v1/media", headers=H, files={"file": ("s.mp4", open(sample, "rb"), "video/mp4")}, data={"templeId": "temple_123"})
    assert r.status_code == 201; mid = r.json()["mediaId"]
    assert client.get(f"/v1/media/{mid}", headers={**H, "X-Tenant-Id": "other"}).status_code == 404   # tenant isolation
    r = client.post(f"/v1/media/{mid}/process", headers=H, json={"formats": ["reel"], "idempotencyKey": "k1"})
    assert r.json()["status"] in ("PROCESSING", "READY"); jid = r.json()["jobId"]
    assert client.post(f"/v1/media/{mid}/process", headers=H, json={"formats": ["reel"], "idempotencyKey": "k1"}).json()["jobId"] == jid
    j = client.get(f"/v1/media/{mid}", headers=H).json()["jobs"][0]
    assert j["status"] == "READY", j
    assert client.get(f"/v1/media/{mid}/edit-plan", headers=H).json()["source"]["templeId"] == "temple_123"
    assert client.get(f"/v1/media/{mid}/highlights", headers=H).json()["moments"]
    assert "transcript" in client.get(f"/v1/media/{mid}/analysis", headers=H).json()
    outs = client.get(f"/v1/media/{mid}/outputs", headers=H).json()
    assert "reel" in outs and "thumbnail" in outs
    assert b"media_processing_duration_seconds" in client.get("/metrics").content
