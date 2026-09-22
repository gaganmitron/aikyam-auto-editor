"""Small local web UI: upload clips (+ photos) and a song, put the clips in the order you want, get the finished reel.

    python -m aikyam_video.ui            # http://127.0.0.1:8090

Everything runs on this machine (files stay under $AIKYAM_UI_DIR, default results/ui). One reel is made at a time (heavy models, small RAM).
Analysis of a set of files is cached, so re-ordering and re-rendering the same clips does not repeat it."""
from __future__ import annotations
import hashlib, os, re, shutil, subprocess, threading, time, uuid
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from . import ffmpeg as ff, multi

ROOT = Path(os.environ.get("AIKYAM_UI_DIR", "results/ui")).resolve()
CLIP_EXT = multi.IMAGE_EXT | {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}
MUSIC_EXT = multi.AUDIO_EXT
MAX_MB = int(os.environ.get("AIKYAM_UI_MAX_MB", "2000"))
_lock = threading.Lock()
JOBS: Dict[str, dict] = {}


def _dirs():
    for d in ("uploads", "thumbs", "out", "jobs"):
        (ROOT / d).mkdir(parents=True, exist_ok=True)


def _kind(name: str) -> Optional[str]:
    e = Path(name).suffix.lower()
    return "music" if e in MUSIC_EXT else ("image" if e in multi.IMAGE_EXT else ("video" if e in CLIP_EXT else None))


_DESC: Dict[tuple, dict] = {}


def _describe(path: Path) -> dict:
    """File description (kind, duration, thumbnail), cached per file so listing does not re-run ffprobe."""
    st = path.stat(); key = (str(path), st.st_mtime_ns, st.st_size)
    if key not in _DESC: _DESC[key] = _describe_uncached(path)
    return dict(_DESC[key])


def _describe_uncached(path: Path) -> dict:
    fid, _, name = path.name.partition("__"); kind = _kind(name); item = {"id": fid, "name": name, "kind": kind, "duration": None, "thumb": False, "size": path.stat().st_size, "t": path.stat().st_mtime}
    try:
        if kind == "video": item["duration"] = round(ff.probe(str(path)).duration, 1); item["audio"] = ff.has_audio_stream(str(path))
        elif kind == "music": item["duration"] = round(float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip() or 0), 1)
    except Exception:                                                              # a file ffprobe cannot read still lists (and fails clearly at render time)
        pass
    th = ROOT / "thumbs" / f"{fid}.jpg"
    if kind in ("video", "image") and not th.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-y", *(["-ss", "1"] if kind == "video" else []), "-i", str(path), "-frames:v", "1", "-vf", "scale=240:-2", "-q:v", "4", str(th)], capture_output=True)
        if not th.exists() and kind == "video": subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-frames:v", "1", "-vf", "scale=240:-2", "-q:v", "4", str(th)], capture_output=True)
    item["thumb"] = th.exists()
    return item


def list_files() -> List[dict]:
    _dirs(); return sorted((_describe(p) for p in (ROOT / "uploads").iterdir() if "__" in p.name), key=lambda i: i["t"])


def plan_order(clip_ids: List[str], use_order: bool):
    """(inputs in a CANONICAL order, Options.order). Canonical = sorted by file id, so the same files always get the same asset ids and the analysis cache is reused;
    the person's order becomes Options.order (asset ids v1.. for videos, i1.. for images)."""
    files = {i["id"]: i for i in list_files() if i["kind"] in ("video", "image")}
    missing = [c for c in clip_ids if c not in files]
    if missing or not clip_ids:
        raise HTTPException(400, f"unknown or no clips: {missing}")
    canon = sorted(set(clip_ids)); ids, nv, ni = {}, 0, 0
    for c in canon:
        if files[c]["kind"] == "video": nv += 1; ids[c] = f"v{nv}"
        else: ni += 1; ids[c] = f"i{ni}"
    if not any(files[c]["kind"] == "video" for c in canon):
        raise HTTPException(400, "add at least one video clip")
    inputs = [str(next((ROOT / "uploads").glob(f"{c}__*"))) for c in canon]
    return inputs, ([ids[c] for c in clip_ids] if use_order else None), canon


class RenderReq(BaseModel):
    clips: List[str]                    # in the order the person arranged them
    use_order: bool = True              # False = let the editor decide the story order
    music: Optional[str] = None
    rights: bool = False                # "I own this music / have a licence"
    title: Optional[str] = None
    subtitle: Optional[str] = None


def _run(job: str, req: RenderReq):
    """Background worker: analysis (cached per file set) -> plan -> render. Status in JOBS[job]."""
    from .options import Options
    from . import music as _music, stages
    j = JOBS[job]
    try:
        inputs, order, canon = plan_order(req.clips, req.use_order)
        work = ROOT / "jobs" / hashlib.sha1("|".join(canon).encode()).hexdigest()[:12]; work.mkdir(parents=True, exist_ok=True)
        o = Options(formats=["reel"], allow_silent=True, order=order, title=(req.title or None), subtitle=(req.subtitle or None))
        if req.music:
            mp = next((ROOT / "uploads").glob(f"{req.music}__*"), None)
            if mp is None: raise ValueError("music file not found")
            o.music_track = _music.register_user_track(str(mp), str(work), req.rights)
        j["stage"] = "analysing the clips (the first time for a set of files takes a few minutes)"
        multi.ingest(inputs, str(work), o)
        j["stage"] = "planning the story"; plan = multi.plan(inputs, str(work), o)
        j["stage"] = "rendering"; stages.render(plan["source"]["path"], str(work), o)
        out = ROOT / "out" / f"{job}.mp4"; shutil.copy(work / "reel-9x16.mp4", out)
        qc = {}
        try:
            import json; qc = json.load(open(work / "qc.json"))
        except Exception: pass
        j.update(state="done", stage="done", reel=str(out), seconds=plan["durationSeconds"], qc=qc.get("status"), qc_notes=[c["name"] for c in qc.get("checks", []) if c["status"] != "pass"],
                 story=[{"clip": s.get("assetId"), "role": s.get("role"), "seconds": round(s["end"] - s["start"], 1)} for s in plan["segments"]])
    except HTTPException as e:
        j.update(state="error", error=str(e.detail))
    except Exception as e:                                                          # noqa: BLE001 -- shown to the person
        j.update(state="error", error=f"{type(e).__name__}: {e}")
    finally:
        _lock.release()


def create_app() -> FastAPI:
    _dirs(); app = FastAPI(title="Aikyam reel maker")

    @app.get("/", response_class=HTMLResponse)
    def index(): return PAGE

    @app.get("/api/files")
    def files(): return list_files()

    @app.post("/api/upload")
    async def upload(files: List[UploadFile] = File(...)):
        out = []
        for f in files:
            kind = _kind(f.filename or "")
            if kind is None: raise HTTPException(400, f"unsupported file type: {f.filename}")
            fid = uuid.uuid4().hex[:10]; safe = re.sub(r"[^A-Za-z0-9._-]", "_", Path(f.filename).name)[:80]; dest = ROOT / "uploads" / f"{fid}__{safe}"; n = 0
            with open(dest, "wb") as w:
                while chunk := await f.read(1 << 20):
                    n += len(chunk)
                    if n > MAX_MB * 1024 * 1024: w.close(); dest.unlink(missing_ok=True); raise HTTPException(413, f"{f.filename} is larger than {MAX_MB} MB")
                    w.write(chunk)
            out.append(_describe(dest))
        return out

    @app.delete("/api/files/{fid}")
    def delete(fid: str):
        if not re.fullmatch(r"[0-9a-f]{10}", fid): raise HTTPException(400, "bad id")
        for p in (ROOT / "uploads").glob(f"{fid}__*"): p.unlink()
        (ROOT / "thumbs" / f"{fid}.jpg").unlink(missing_ok=True); return {"ok": True}

    @app.get("/thumb/{fid}")
    def thumb(fid: str):
        p = ROOT / "thumbs" / f"{re.sub(r'[^0-9a-f]', '', fid)}.jpg"
        if not p.exists(): raise HTTPException(404)
        return FileResponse(p)

    @app.post("/api/render")
    def render(req: RenderReq):
        if req.music and not req.rights: raise HTTPException(400, "tick the box to confirm you own the music or have a licence to use it")
        plan_order(req.clips, req.use_order)                                       # validate now: the person gets the error immediately
        if not _lock.acquire(blocking=False): raise HTTPException(409, "a reel is already being made: wait for it to finish")
        job = uuid.uuid4().hex[:10]; JOBS[job] = {"state": "running", "stage": "starting", "t0": time.time()}
        threading.Thread(target=_run, args=(job, req), daemon=True).start(); return {"job": job}

    @app.get("/api/job/{job}")
    def status(job: str):
        j = JOBS.get(job)
        if not j: raise HTTPException(404)
        return {**{k: v for k, v in j.items() if k != "reel"}, "elapsed": round(time.time() - j["t0"]), "reel": f"/reel/{job}" if j.get("reel") else None}

    @app.get("/reel/{job}")
    def reel(job: str):
        p = ROOT / "out" / f"{re.sub(r'[^0-9a-f]', '', job)}.mp4"
        if not p.exists(): raise HTTPException(404)
        return FileResponse(p, media_type="video/mp4", filename="reel.mp4")

    return app


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Aikyam reel maker</title>
<style>
:root{--bg:#faf7f2;--card:#fff;--ink:#241c15;--mute:#7a6d60;--line:#e6ddd0;--acc:#c2410c;--acc2:#9a3412}
@media (prefers-color-scheme:dark){:root{--bg:#171310;--card:#211b16;--ink:#f3ece3;--mute:#a89b8c;--line:#3a3129;--acc:#f97316;--acc2:#fb923c}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,sans-serif}
main{max-width:980px;margin:0 auto;padding:20px 16px 60px}h1{font-size:22px;margin:0 0 4px}p.sub{color:var(--mute);margin:0 0 18px}
section{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:0 0 14px}h2{font-size:15px;margin:0 0 10px}
.drop{border:2px dashed var(--line);border-radius:10px;padding:18px;text-align:center;color:var(--mute);cursor:pointer}.drop.on{border-color:var(--acc);color:var(--acc)}
button{background:var(--acc);color:#fff;border:0;border-radius:8px;padding:9px 14px;font:inherit;cursor:pointer}button:hover{background:var(--acc2)}button.g{background:none;color:var(--ink);border:1px solid var(--line)}button:disabled{opacity:.5;cursor:default}
ol{list-style:none;padding:0;margin:0;display:grid;gap:8px}li.clip{display:grid;grid-template-columns:26px 74px 1fr auto;gap:10px;align-items:center;border:1px solid var(--line);border-radius:10px;padding:8px;background:var(--bg)}
li.clip.drag{opacity:.4}.num{font-weight:700;color:var(--acc);text-align:center}.th{width:74px;height:74px;object-fit:cover;border-radius:6px;background:var(--line)}.meta{min-width:0}.meta b{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.meta span{color:var(--mute);font-size:13px}
.btns{display:flex;gap:4px}.btns button{padding:6px 9px}row,.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}input[type=text]{flex:1;min-width:180px;padding:9px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--ink);font:inherit}
.mute{color:var(--mute)}.err{color:#dc2626}video{width:100%;max-width:340px;border-radius:12px;background:#000;display:block}.res{display:grid;grid-template-columns:auto 1fr;gap:16px;align-items:start}@media(max-width:640px){.res{grid-template-columns:1fr}}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin:8px 0}.bar i{display:block;height:100%;width:30%;background:var(--acc);animation:m 1.4s infinite ease-in-out}@keyframes m{0%{margin-left:-30%}100%{margin-left:100%}}
</style></head><body><main>
<h1>Aikyam reel maker</h1><p class="sub">Add your clips and a song, put the clips in the order you want, make the reel. Everything stays on this computer.</p>
<section><h2>1. Clips and photos</h2><div class="drop" id="dz">Drop videos or photos here, or click to choose<input id="fc" type="file" multiple accept="video/*,image/*" hidden></div><p class="mute" id="up"></p></section>
<section><h2>2. Order</h2><p class="mute" style="margin-top:0">Drag a clip, or use the arrows: the reel plays them top to bottom. Untick a clip to leave it out.</p>
<ol id="list"></ol><p class="mute" id="none">No clips yet.</p>
<label class="row" style="margin-top:10px"><input type="checkbox" id="uo" checked> Use my order <span class="mute">(untick to let the editor choose the story order)</span></label></section>
<section><h2>3. Music (optional)</h2><div class="row"><input id="mc" type="file" accept="audio/*"><select id="ms"><option value="">No music</option></select></div>
<label class="row" style="margin-top:8px"><input type="checkbox" id="rt"> I own this music or have a licence to use it in published reels</label></section>
<section><h2>4. Title (optional)</h2><div class="row"><input type="text" id="ti" placeholder="Title, shown for the first seconds"><input type="text" id="st" placeholder="Second line"></div></section>
<section><button id="go">Make the reel</button> <span id="msg" class="mute"></span><div id="prog" hidden><div class="bar"><i></i></div><span class="mute" id="stage"></span></div>
<div class="res" id="res" hidden style="margin-top:14px"><video id="vid" controls playsinline></video><div><p id="info"></p><p><a id="dl" download="reel.mp4"><button class="g">Download the reel</button></a></p></div></div></section>
</main><script>
const $=s=>document.querySelector(s);let files=[],order=[],on=new Set();
async function load(){files=await (await fetch('/api/files')).json();const clips=files.filter(f=>f.kind!=='music');const ids=clips.map(c=>c.id);order=order.filter(i=>ids.includes(i));for(const i of ids)if(!order.includes(i)){order.push(i);on.add(i)}draw()}
function draw(){const L=$('#list');L.innerHTML='';$('#none').hidden=order.length>0;const by=Object.fromEntries(files.map(f=>[f.id,f]));let n=0;
order.forEach((id,ix)=>{const f=by[id];if(!f)return;const li=document.createElement('li');li.className='clip';li.draggable=true;li.dataset.id=id;
li.innerHTML=`<div class="num">${on.has(id)?++n:'–'}</div><img class="th" src="${f.thumb?'/thumb/'+id:''}" alt=""><div class="meta"><b>${f.name}</b><span>${f.kind==='video'?(f.duration?f.duration+' s':'video')+(f.audio===false?' · no sound':''):'photo'}</span></div>
<div class="btns"><label class="row" style="gap:4px"><input type="checkbox" ${on.has(id)?'checked':''}></label><button class="g" title="up">↑</button><button class="g" title="down">↓</button><button class="g" title="remove">✕</button></div>`;
const [ck,up,dn,rm]=li.querySelectorAll('input,button');ck.onchange=()=>{ck.checked?on.add(id):on.delete(id);draw()};up.onclick=()=>mv(ix,-1);dn.onclick=()=>mv(ix,1);rm.onclick=async()=>{await fetch('/api/files/'+id,{method:'DELETE'});load()};
li.ondragstart=e=>{li.classList.add('drag');e.dataTransfer.setData('t',id)};li.ondragend=()=>li.classList.remove('drag');li.ondragover=e=>e.preventDefault();
li.ondrop=e=>{e.preventDefault();const s=e.dataTransfer.getData('t');if(s&&s!==id){order.splice(order.indexOf(s),1);order.splice(order.indexOf(id),0,s);draw()}};L.append(li)});
const ms=$('#ms'),cur=ms.value;ms.innerHTML='<option value="">No music</option>'+files.filter(f=>f.kind==='music').map(f=>`<option value="${f.id}">${f.name}${f.duration?' ('+Math.round(f.duration)+' s)':''}</option>`).join('');if([...ms.options].some(o=>o.value===cur))ms.value=cur}
function mv(i,d){const j=i+d;if(j<0||j>=order.length)return;[order[i],order[j]]=[order[j],order[i]];draw()}
async function send(list){if(!list.length)return;$('#up').textContent='Uploading '+list.length+' file(s)…';const fd=new FormData();[...list].forEach(f=>fd.append('files',f));const r=await fetch('/api/upload',{method:'POST',body:fd});$('#up').textContent=r.ok?'':'Upload failed: '+(await r.json()).detail;const added=r.ok?await r.json():[];
for(const a of added){if(a.kind==='music'){$('#ms').dataset.pick=a.id}}await load();if($('#ms').dataset.pick){$('#ms').value=$('#ms').dataset.pick;$('#ms').dataset.pick=''}}
const dz=$('#dz'),fc=$('#fc');dz.onclick=()=>fc.click();fc.onchange=()=>{send(fc.files);fc.value=''};dz.ondragover=e=>{e.preventDefault();dz.classList.add('on')};dz.ondragleave=()=>dz.classList.remove('on');dz.ondrop=e=>{e.preventDefault();dz.classList.remove('on');send(e.dataTransfer.files)};
$('#mc').onchange=e=>{send(e.target.files);e.target.value=''};
$('#go').onclick=async()=>{const clips=order.filter(i=>on.has(i));$('#msg').textContent='';$('#msg').className='mute';if(!clips.length){$('#msg').textContent='Tick at least one clip.';return}
const body={clips,use_order:$('#uo').checked,music:$('#ms').value||null,rights:$('#rt').checked,title:$('#ti').value||null,subtitle:$('#st').value||null};
const r=await fetch('/api/render',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const j=await r.json();if(!r.ok){$('#msg').textContent=j.detail;$('#msg').className='err';return}
$('#go').disabled=true;$('#prog').hidden=false;$('#res').hidden=true;poll(j.job)}
async function poll(job){const s=await (await fetch('/api/job/'+job)).json();$('#stage').textContent=s.stage+' · '+s.elapsed+' s';
if(s.state==='running'){setTimeout(()=>poll(job),2000);return}$('#go').disabled=false;$('#prog').hidden=true;
if(s.state==='error'){$('#msg').textContent=s.error;$('#msg').className='err';return}
$('#res').hidden=false;$('#vid').src=s.reel;$('#dl').href=s.reel;$('#info').innerHTML=`<b>${s.seconds} s</b> · quality check: <b>${s.qc}</b>${s.qc_notes.length?' ('+s.qc_notes.join(', ')+')':''}<br><span class="mute">${s.story.map((x,i)=>(i+1)+'. '+x.clip+' '+x.role.toLowerCase()+' '+x.seconds+'s').join(' → ')}</span>`}
load();
</script></body></html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="127.0.0.1", port=int(os.environ.get("AIKYAM_UI_PORT", "8090")))
