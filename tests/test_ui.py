"""Small reel-maker UI: upload / list / delete, the clip order becomes Options.order with stable asset ids, guards (rights, busy, file types). The heavy pipeline is replaced by a stub."""
import time
import pytest
from fastapi.testclient import TestClient
from aikyam_video import ui


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "ROOT", tmp_path / "ui"); ui.JOBS.clear(); ui._DESC.clear()
    if ui._lock.locked(): ui._lock.release()
    return TestClient(ui.create_app())


def up(c, *names):
    r = c.post("/api/upload", files=[("files", (n, b"not really media", "application/octet-stream")) for n in names]); assert r.status_code == 200, r.text; return r.json()


def test_upload_list_and_delete_and_type_checks(client):
    a = up(client, "temple aarti.mp4", "photo.jpg", "song.mp3")
    assert [x["kind"] for x in a] == ["video", "image", "music"] and all(x["id"] and "__" not in x["name"] for x in a)
    assert {f["name"] for f in client.get("/api/files").json()} == {"temple_aarti.mp4", "photo.jpg", "song.mp3"}                  # names are made safe
    assert client.post("/api/upload", files=[("files", ("notes.txt", b"x", "text/plain"))]).status_code == 400
    assert client.delete(f"/api/files/{a[1]['id']}").status_code == 200 and len(client.get("/api/files").json()) == 2
    assert client.delete("/api/files/../../etc").status_code in (404, 405, 400) and client.delete("/api/files/zzzz").status_code == 400
    assert "Aikyam reel maker" in client.get("/").text


def test_the_persons_order_becomes_option_order_with_stable_ids(client):
    a, b, c, img = up(client, "a.mp4", "b.mp4", "c.mp4", "p.jpg")
    ids = [a["id"], b["id"], c["id"]]; canon = sorted(ids); vid = {cid: f"v{i + 1}" for i, cid in enumerate(canon)}
    inputs, order, canonical = ui.plan_order([c["id"], a["id"], img["id"], b["id"]], True)
    assert canonical == sorted([a["id"], b["id"], c["id"], img["id"]]) and len(inputs) == 4
    assert order == [vid[c["id"]], vid[a["id"]], "i1", vid[b["id"]]]                                        # videos v1.., photos i1.. in canonical numbering
    _, order2, _ = ui.plan_order([b["id"], c["id"]], False); assert order2 is None                             # "let the editor choose"
    i1, o1, c1 = ui.plan_order([a["id"], b["id"]], True); i2, o2, c2 = ui.plan_order([b["id"], a["id"]], True)
    assert i1 == i2 and c1 == c2 and o1 == list(reversed(o2))                                                # re-ordering keeps the input set, so cached analysis is reused
    with pytest.raises(Exception): ui.plan_order([img["id"]], True)                                            # photos only: needs a video
    with pytest.raises(Exception): ui.plan_order(["nope"], True)


def test_render_guards_and_job_lifecycle(client, monkeypatch):
    a, b, m = up(client, "a.mp4", "b.mp4", "song.mp3")
    body = {"clips": [a["id"], b["id"]], "use_order": True}
    assert client.post("/api/render", json={**body, "music": m["id"], "rights": False}).status_code == 400              # music needs the rights tick
    assert client.post("/api/render", json={"clips": [], "use_order": True}).status_code == 400
    seen = {}
    def fake(job, req):
        try:
            seen["req"] = req; time.sleep(1.5); ui.JOBS[job].update(state="done", stage="done", reel=str(ui.ROOT / "out" / f"{job}.mp4"), seconds=20.0, qc="pass", qc_notes=[], story=[])
            (ui.ROOT / "out" / f"{job}.mp4").write_bytes(b"mp4")
        finally: ui._lock.release()
    monkeypatch.setattr(ui, "_run", fake)
    r = client.post("/api/render", json={**body, "music": m["id"], "rights": True, "title": "Ganga Aarti"}); assert r.status_code == 200
    job = r.json()["job"]; assert client.post("/api/render", json=body).status_code == 409                          # one reel at a time
    for _ in range(100):
        s = client.get(f"/api/job/{job}").json()
        if s["state"] != "running": break
        time.sleep(0.1)
    assert s["state"] == "done" and s["reel"] == f"/reel/{job}" and client.get(s["reel"]).status_code == 200 and seen["req"].title == "Ganga Aarti"
    assert client.get("/api/job/none").status_code == 404 and client.get("/reel/none").status_code == 404
    assert client.post("/api/render", json=body).status_code == 200                                                    # free again after the job
