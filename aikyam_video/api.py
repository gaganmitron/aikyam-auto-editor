"""Media API (section 23). Auth: X-API-Key (env AIKYAM_API_KEY, fail-closed) + X-Tenant-Id. Uploads are validated
(MIME, size, duration, codec) before storage; clients get signed URLs, never storage credentials.
Execution: in-process worker thread by default; with KAFKA_BOOTSTRAP the API only publishes events (workers do the work)."""
from __future__ import annotations
import base64, hashlib, hmac, html, json, os, re, tempfile, time, uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Response, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from . import bus as bus_mod, ffmpeg as ff, library, pipeline, publish as publish_mod
from .events import Event
from .jobs import InvalidTransition, JobStore
from .metrics import METRICS, get_logger
from .options import Options, Profile
from .render import FORMATS
from .storage import from_env
from .workers import options_to_dict

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 8 * 1024 ** 3))
ALLOWED_MIME = {"video/mp4", "video/quicktime", "video/x-matroska", "video/webm"}
WORK = Path(os.environ.get("WORK_DIR", "./work"))
STAGE_TO_STATE = {"transcription": "PROCESSING", "scene_detection": "PROCESSING", "highlights": "ANALYZED",
                  "edit_plan": "HIGHLIGHTS_READY", "render": "RENDERING"}
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class ProfileReq(BaseModel):
    preferred_labels: Dict[str, float] = Field(default_factory=dict)
    preferred_deities: List[str] = Field(default_factory=list)
    target_seconds: Optional[float] = Field(None, ge=5, le=60)
    caption_lang: Optional[str] = Field(None, pattern=r"^[a-z]{2,3}$")


class ProcessReq(BaseModel):
    """Client-settable options ONLY (no file paths): validated whitelist."""
    formats: Optional[List[str]] = None
    idempotencyKey: Optional[str] = Field(None, max_length=128)
    caption_lang: Optional[str] = Field(None, pattern=r"^[a-z]{2,3}$")
    caption_mode: Optional[str] = Field(None, pattern=r"^(sentence|word)$")
    target_seconds: Optional[float] = Field(None, ge=5, le=60)
    festival_id: Optional[str] = Field(None, pattern=ID_RE.pattern)
    location: Optional[str] = Field(None, max_length=120)
    music_track: Optional[str] = Field(None, pattern=ID_RE.pattern)
    music: Optional[str] = Field(None, pattern=r"^(auto|off)$")
    music_volume: Optional[float] = Field(None, ge=0.1, le=1.0)
    planner: Optional[str] = Field(None, pattern=r"^(deterministic|llm)$")
    transition: Optional[str] = Field(None, pattern=r"^(crossfade|fade|cut)$")
    transition_seconds: Optional[float] = Field(None, ge=0.2, le=1.5)
    auto_publish: bool = False
    profile: Optional[ProfileReq] = None


def _make_token(secret: str, scope: str, tenant: str, ttl: int = 900) -> str:
    body = base64.urlsafe_b64encode(json.dumps({"s": scope, "t": tenant, "e": int(time.time()) + ttl}).encode()).decode().rstrip("=")
    return f"{body}.{hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()}"


def _check_token(secret: str, token: str, scope: str) -> str:
    """Return tenant if the token is valid for `scope`, else raise 403."""
    try:
        body, sig = token.rsplit(".", 1)
        if not hmac.compare_digest(sig, hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()):
            raise ValueError
        d = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
        if d["s"] != scope or d["e"] < time.time():
            raise ValueError
        return d["t"]
    except Exception:
        raise HTTPException(403, "invalid or expired link")


def create_app(store: Optional[JobStore] = None, storage=None, executor=None, bus=None, publisher=None, notifier=None,
               embedder=None) -> FastAPI:
    app = FastAPI(title="Aikyam Media API", version="0.2.0")
    store = store or JobStore()
    storage = storage or from_env()
    bus = bus if bus is not None else bus_mod.from_env()
    if publisher is None and notifier is None:
        publisher, notifier = publish_mod.from_env()
    pool = executor or ThreadPoolExecutor(max_workers=1)  # one job at a time per worker process
    state = {"embedder": embedder}

    def secret() -> str:
        s = os.environ.get("AIKYAM_API_KEY")
        if not s:
            raise HTTPException(503, "server auth not configured")
        return s

    def auth(x_api_key: str = Header(""), x_tenant_id: str = Header("")):
        if not hmac.compare_digest(x_api_key, secret()):
            raise HTTPException(401, "invalid API key")
        if not x_tenant_id:
            raise HTTPException(400, "X-Tenant-Id required")
        return x_tenant_id

    def media_or_404(mid, tenant):
        m = store.get_media(mid)
        if not m or m["tenant_id"] != tenant:   # same 404 for other tenants' media: no existence leak
            raise HTTPException(404, "media not found")
        return m

    def to_options(m: dict, req: ProcessReq) -> Options:
        o = Options(video_id=m["media_id"], temple_id=m["temple_id"], formats=req.formats or ["reel", "square", "landscape"])
        if any(f not in FORMATS for f in o.formats):
            raise HTTPException(422, f"formats must be in {sorted(FORMATS)}")
        for k in ("caption_lang", "caption_mode", "target_seconds", "festival_id", "location", "music_track", "music", "music_volume", "planner", "transition", "transition_seconds"):
            v = getattr(req, k)
            if v is not None: setattr(o, k, v)
        o.auto_publish = req.auto_publish
        if req.profile: o.profile = Profile(**req.profile.model_dump())
        return o

    def work(job_id, m, o: Options):
        """Thread mode: whole pipeline in this process."""
        log = get_logger(job_id, m["media_id"], m["temple_id"] or "")
        out = WORK / m["media_id"] / job_id
        o.on_stage = lambda st: store.advance(job_id, STAGE_TO_STATE[st])
        try:
            local = str(WORK / m["media_id"] / "source.mp4")
            storage.download(m["storage_key"], local)
            files = pipeline.run(local, str(out), "process", o, log)
            keys = {}
            for name, p in files.items():
                key = f"{m['tenant_id']}/{m['media_id']}/{job_id}/{Path(p).name}"
                storage.upload(p, key); keys[name] = key
            for extra in ("vision.json", "embeddings.json"):
                if (out / extra).exists(): storage.upload(str(out / extra), f"{m['tenant_id']}/{m['media_id']}/{job_id}/{extra}")
            store.record_outputs(m["media_id"], keys)
            store.record_cost(job_id, m["media_id"], json.load(open(files["cost"])))
            library.index_media(store, m["media_id"], m["tenant_id"], str(out))
            store.advance(job_id, "READY")
            if o.auto_publish and publisher is not None:
                plan = json.load(open(files["edit_plan"]))
                if publish_mod.auto_publish_ok(plan, o.auto_publish_min_score):
                    publish_mod.publish_media(store, storage, job_id, m["media_id"], publisher, notifier)
        except Exception as e:
            log.error(f"job failed: {e}")
            try: store.transition(job_id, "FAILED", str(e)[:1000])
            except InvalidTransition: pass

    def start(job, m, o: Options):
        if job["state"] != "UPLOADED":       # duplicate request with the same key: job already moved on
            return
        if bus is not None:                   # event-driven: workers do the work
            bus.publish(Event(type="MediaUploaded", jobId=job["job_id"], mediaId=m["media_id"], tenantId=m["tenant_id"],
                              payload={"options": options_to_dict(o)}))
        else:
            store.transition(job["job_id"], "PROCESSING")
            pool.submit(work, job["job_id"], m, o)

    # ------------------------------------------------------------------ media
    @app.post("/v1/media", status_code=201)
    async def upload(file: UploadFile = File(...), templeId: Optional[str] = Form(None), tenant=Depends(auth)):
        if file.content_type not in ALLOWED_MIME:
            raise HTTPException(415, f"unsupported MIME {file.content_type}")
        if templeId is not None and not ID_RE.match(templeId):
            raise HTTPException(422, "invalid templeId")
        mid = f"video_{uuid.uuid4().hex[:12]}"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4"); size = 0
        try:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "file too large")
                tmp.write(chunk)
            tmp.close()
            try:   # never trust the client MIME: probe the real streams
                pipeline.validate_source(tmp.name, pipeline.Options())
            except (ValueError, ff.FFmpegError) as e:
                raise HTTPException(422, f"invalid video: {e}")
            key = f"{tenant}/{mid}/source.mp4"
            storage.upload(tmp.name, key, "video/mp4")
        finally:
            tmp.close(); os.unlink(tmp.name)
        store.create_media(mid, tenant, key, templeId)
        return {"mediaId": mid, "status": "UPLOADED"}

    @app.post("/v1/media/{mid}/process")
    def process(mid: str, req: ProcessReq = ProcessReq(), tenant=Depends(auth)):
        m = media_or_404(mid, tenant)
        o = to_options(m, req)
        job = store.create_job(mid, tenant, req.idempotencyKey, options_to_dict(o))
        start(job, m, o)
        return {"jobId": job["job_id"], "status": store.get_job(job["job_id"])["state"]}

    @app.post("/v1/media/{mid}/render")
    def render(mid: str, req: ProcessReq = ProcessReq(), tenant=Depends(auth)):
        m = media_or_404(mid, tenant)
        o = to_options(m, req)
        job = store.create_job(mid, tenant, req.idempotencyKey or f"render:{mid}:{uuid.uuid4().hex[:8]}", options_to_dict(o))
        start(job, m, o)
        return {"jobId": job["job_id"], "status": store.get_job(job["job_id"])["state"]}

    @app.post("/v1/jobs/{job_id}/retry")
    def retry(job_id: str, tenant=Depends(auth)):
        job = store.get_job(job_id)
        if not job or job["tenant_id"] != tenant:
            raise HTTPException(404, "job not found")
        m = media_or_404(job["media_id"], tenant)
        try:
            store.transition(job_id, "RETRYING"); store.transition(job_id, "PROCESSING")
        except InvalidTransition as e:
            raise HTTPException(409, str(e))
        from .workers import options_from_dict
        o = options_from_dict(json.loads(job["options"] or "{}"))
        if bus is not None:
            bus.publish(Event(type="MediaUploaded", jobId=job_id, mediaId=m["media_id"], tenantId=tenant, attempt=job["attempt"],
                              payload={"options": options_to_dict(o)}))
        else:
            pool.submit(work, job_id, m, o)
        return {"jobId": job_id, "status": "PROCESSING"}

    def _json_output(mid, tenant, name):
        media_or_404(mid, tenant)
        key = store.get_outputs(mid).get(name)
        if not key:
            raise HTTPException(409, "not ready")
        with tempfile.TemporaryDirectory() as td:
            return json.load(open(storage.download(key, os.path.join(td, "x.json"))))

    @app.get("/v1/media/{mid}")
    def get_media(mid: str, tenant=Depends(auth)):
        m = media_or_404(mid, tenant)
        with store.e.connect() as c:
            from sqlalchemy import select
            from .jobs import jobs, video_embeddings
            js = [dict(r) for r in c.execute(select(jobs).where(jobs.c.media_id == mid).order_by(jobs.c.created_at.desc())).mappings()]
            ve = c.execute(select(video_embeddings.c.duplicate_of).where(video_embeddings.c.media_id == mid)).first()
        return {"mediaId": mid, "templeId": m["temple_id"], "duplicateOf": ve[0] if ve else None,
                "jobs": [{"jobId": j["job_id"], "status": j["state"], "attempt": j["attempt"], "error": j["error"]} for j in js]}

    @app.get("/v1/media/{mid}/analysis")
    def analysis(mid: str, tenant=Depends(auth)):
        return {"transcript": _json_output(mid, tenant, "transcript"), "scenes": _json_output(mid, tenant, "scenes"),
                "entities": _json_output(mid, tenant, "entities")}

    @app.get("/v1/media/{mid}/highlights")
    def highlights(mid: str, tenant=Depends(auth)):
        return _json_output(mid, tenant, "moments")

    @app.get("/v1/media/{mid}/edit-plan")
    def edit_plan(mid: str, tenant=Depends(auth)):
        return _json_output(mid, tenant, "edit_plan")

    @app.get("/v1/media/{mid}/outputs")
    def get_outputs(mid: str, tenant=Depends(auth)):
        media_or_404(mid, tenant)
        return {n: storage.signed_url(k) for n, k in store.get_outputs(mid).items()}

    # ------------------------------------------------------------------ publish / preview / dashboard
    def _latest_ready_job(mid):
        from sqlalchemy import select
        from .jobs import jobs
        with store.e.connect() as c:
            r = c.execute(select(jobs).where(jobs.c.media_id == mid, jobs.c.state == "READY").order_by(jobs.c.created_at.desc())).mappings().first()
        if not r:
            raise HTTPException(409, "no READY job to publish")
        return dict(r)

    def _publish(mid, force=False):
        if publisher is None:
            raise HTTPException(503, "no publisher configured (AIKYAM_FEED_URL or FEED_FILE)")
        j = _latest_ready_job(mid)
        try:
            return publish_mod.publish_media(store, storage, j["job_id"], mid, publisher, notifier, bus, force)
        except ValueError as e:     # QC gate
            raise HTTPException(409, str(e))
        except Exception as e:      # publisher failure leaves the job READY, retry-able
            raise HTTPException(502, f"publish failed: {e}")

    @app.post("/v1/media/{mid}/publish")
    def publish_ep(mid: str, force: bool = False, tenant=Depends(auth)):
        media_or_404(mid, tenant)
        return _publish(mid, force)["payload"]

    @app.post("/v1/media/{mid}/preview-link")
    def preview_link(mid: str, tenant=Depends(auth)):
        media_or_404(mid, tenant)
        return {"url": f"/preview/{mid}?t={_make_token(secret(), 'preview:' + mid, tenant)}"}

    @app.get("/preview/{mid}", response_class=HTMLResponse)
    def preview(mid: str, t: str = Query(...)):
        tenant = _check_token(secret(), t, "preview:" + mid)
        media_or_404(mid, tenant)
        outs = store.get_outputs(mid)
        if "reel" not in outs:
            raise HTTPException(409, "not ready")
        def url(name):   # LocalStorage URLs are file:// (dev): stream through a token-guarded route instead
            u = storage.signed_url(outs[name])
            return f"/preview/{mid}/asset/{name}?t={t}" if u.startswith("file:") else u
        reel, thumb = url("reel"), url("thumbnail") if "thumbnail" in outs else ""
        plan = _json_output(mid, tenant, "edit_plan")
        ov = plan["overlays"]
        return HTMLResponse(f"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Preview {html.escape(mid)}</title><style>body{{font:16px system-ui;margin:0;background:#111;color:#eee;display:flex;flex-direction:column;align-items:center;gap:12px;padding:16px}}
video{{max-height:75vh;max-width:100%;border-radius:12px}}button{{font:inherit;padding:10px 24px;border:0;border-radius:8px;background:#ffb300;cursor:pointer}}small{{color:#aaa}}</style>
<video src="{html.escape(reel)}" poster="{html.escape(thumb)}" controls playsinline></video>
<div>{html.escape(str(ov.get('temple') or ''))} · {html.escape(str(ov.get('deity') or ''))} · {plan['durationSeconds']}s · captions {'on' if plan['captions']['enabled'] else 'off'}</div>
<button id=p>Publish to Aikyam feed</button><small id=m></small>
<script>p.onclick=async()=>{{p.disabled=true;const r=await fetch('/preview/{html.escape(mid)}/publish?t={html.escape(t)}',{{method:'POST'}});m.textContent=r.ok?'Published ✓':'Failed: '+await r.text()}}</script>""")

    @app.get("/preview/{mid}/asset/{name}")
    def preview_asset(mid: str, name: str, t: str = Query(...)):
        from fastapi.responses import FileResponse
        tenant = _check_token(secret(), t, "preview:" + mid)
        media_or_404(mid, tenant)
        key = store.get_outputs(mid).get(name) if name in ("reel", "thumbnail") else None
        if not key:
            raise HTTPException(404, "no such asset")
        local = WORK / "preview" / mid / Path(key).name
        storage.download(key, str(local))
        return FileResponse(str(local))

    @app.post("/preview/{mid}/publish")
    def preview_publish(mid: str, t: str = Query(...)):
        tenant = _check_token(secret(), t, "preview:" + mid)
        media_or_404(mid, tenant)
        return _publish(mid)["payload"]

    @app.post("/v1/dashboard-link")
    def dashboard_link(templeId: Optional[str] = None, tenant=Depends(auth)):
        scope = "dashboard" + (f":{templeId}" if templeId else "")
        return {"url": f"/dashboard?t={_make_token(secret(), scope, tenant)}" + (f"&templeId={templeId}" if templeId else "")}

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard(t: str = Query(...), templeId: Optional[str] = None):
        tenant = _check_token(secret(), t, "dashboard" + (f":{templeId}" if templeId else ""))
        rows = store.list_media(tenant, templeId)
        tot_cost = sum((r["cost"] or {}).get("inr", 0) or 0 for r in rows)
        tot_src = sum((r["cost"] or {}).get("input_duration_seconds", 0) or 0 for r in rows)
        reels = sum(1 for r in rows if r["cost"])
        body = "".join(
            f"<tr><td>{html.escape(r['media_id'])}</td><td>{html.escape(str(r['temple_id'] or ''))}</td>"
            f"<td>{html.escape(r['job']['state'] if r['job'] else 'UPLOADED')}</td>"
            f"<td>{(r['cost'] or {}).get('inr', 0):.2f}</td><td>{html.escape(str(r['duplicate_of'] or ''))}</td></tr>" for r in rows)
        return HTMLResponse(f"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>Aikyam dashboard</title>
<style>body{{font:15px system-ui;margin:16px}}table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #ddd;padding:6px;text-align:left}}</style>
<h2>Aikyam · {html.escape(templeId or 'all temples')}</h2>
<p>{len(rows)} videos · {reels} processed · ₹{tot_cost:.2f} total · ₹{(tot_cost / (tot_src / 3600)) if tot_src else 0:.2f} / source hour · ₹{(tot_cost / reels) if reels else 0:.2f} / job</p>
<table><tr><th>Media</th><th>Temple</th><th>Status</th><th>₹</th><th>Duplicate of</th></tr>{body}</table>""")

    # ------------------------------------------------------------------ semantic library
    def embedder_():
        if state["embedder"] is None:
            from .vision import ClipVision
            state["embedder"] = ClipVision()
        return state["embedder"]

    @app.get("/v1/search")
    def search(q: str = Query(..., min_length=2, max_length=200), k: int = Query(10, ge=1, le=50), tenant=Depends(auth)):
        return {"query": q, "results": library.search(store, embedder_(), tenant, q, k)}

    @app.get("/v1/media/{mid}/similar")
    def similar(mid: str, k: int = Query(10, ge=1, le=50), tenant=Depends(auth)):
        media_or_404(mid, tenant)
        return {"mediaId": mid, "similar": library.similar(store, mid, tenant, k)}

    @app.get("/v1/media/{mid}/embedding")
    def embedding(mid: str, tenant=Depends(auth)):
        media_or_404(mid, tenant)
        v = library.video_vector(store, mid)
        if v is None:
            raise HTTPException(409, "not indexed yet")
        return {"mediaId": mid, "dim": len(v), "vector": v}

    @app.get("/metrics")
    def metrics():
        return Response(METRICS.expose(), media_type="text/plain; version=0.0.4")

    @app.get("/healthz")
    def health():
        return {"ok": True}

    return app


def app():  # uvicorn --factory aikyam_video.api:app
    return create_app()
