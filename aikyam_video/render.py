"""EditPlan -> media (section Layer 5). FFmpeg does all media work; libass draws captions + template overlays."""
from __future__ import annotations
import json, os
from typing import Dict, List, Optional
from . import ffmpeg as ff
from .captions import ass_header, caption_events, caption_style, lang_cfg
from .plan import OVERLAPPING, edge_transitions, validate_plan
from .audio import AudioProcessor
from .reframe import crop_box, path_expr
from pathlib import Path

FORMATS: Dict[str, dict] = {
    "reel": {"aspect": "9:16", "w": 1080, "h": 1920, "file": "reel-9x16.mp4", "outputFormat": "REEL"},
    "square": {"aspect": "1:1", "w": 1080, "h": 1080, "file": "square-1x1.mp4", "outputFormat": "SQUARE"},
    "landscape": {"aspect": "16:9", "w": 1920, "h": 1080, "file": "landscape-16x9.mp4", "outputFormat": "LANDSCAPE"},
}
TEMPLATES = json.loads((Path(__file__).parent / "data" / "templates.json").read_text())
from .safezone import is_portrait, zones
# HDR -> SDR: linear light, BT.709 primaries, Hable tone mapping (highlights roll off, no clipping; npl=150 tuned so brightness matches Chromium's own HLG handling in the diffusion renderer), BT.709 transfer/matrix, limited range
TONEMAP = "zscale=t=linear:npl=150,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p"
LOGO = Path(__file__).parent / "assets" / "aikyam-logo.png"


def music_path(track_id: str) -> str:
    """Resolve a track through the LICENSED library (music.get_track): unlisted / unlicensed / missing => error, never a fallback."""
    from .music import get_track
    return get_track(track_id).path


def music_chain(dur: float, volume: float = 0.35, src: str = "1:a", out: str = "mus") -> str:
    """Trim the (looped) track to the reel, set its level, fade in 1 s / out 1.5 s so it never starts or stops abruptly."""
    return (f"[{src}]atrim=0:{dur},asetpts=PTS-STARTPTS,aresample=48000,volume={volume},"
            f"afade=t=in:d=1.0,afade=t=out:st={max(0.0, dur - 1.5):.2f}:d=1.5[{out}]")


def build_ass(plan: dict, w: int, h: int, text_brand: bool = True) -> str:
    cap, ov = plan["captions"], plan["overlays"]
    dur = plan["durationSeconds"]
    lang = cap["language"]
    styles = [caption_style("Cap", w, h, lang, cap.get("style", {})),
              f"Style: Ovl,Noto Sans,{int(h * 0.03)},&H00FFFFFF,&H00FFFFFF,&H00000000,&H99000000,0,0,0,0,100,100,0,0,1,3,1,8,20,20,20,1",
              f"Style: OvlBox,Noto Sans,{int(h * 0.03)},&H00FFFFFF,&H00FFFFFF,&H00000000,&HAA000000,0,0,0,0,100,100,0,0,3,10,0,8,20,20,20,1",
              f"Style: Brand,Noto Sans,{int(h * 0.028)},&HCC00D7FF,&H00FFFFFF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,2,0,3,30,30,30,1"]
    from .captions import _t, _esc
    ev: List[str] = []
    tpl = TEMPLATES.get(ov.get("template") or "divine_moment", TEMPLATES["divine_moment"])
    for ln in tpl["lines"]:
        txt = ov.get(ln["field"])
        if txt:
            style = "OvlBox" if ln.get("box") else "Ovl"
            b = "\\b1" if ln.get("bold") else ""
            if tpl.get("accent") and ln.get("bold"):
                b += "\\1c" + tpl["accent"] + "&"
            ev.append(f"Dialogue: 0,{_t(0)},{_t(dur)},{style},,0,0,0,,"
                      f"{{\\an8\\pos({w // 2},{int(h * (ln.get('yPortrait', ln['y']) if is_portrait(w, h) else ln['y']))})\\fs{int(h * ln['size'])}{b}\\fad(400,400)}}{_esc(txt)}")
    if ov.get("title"):                                                    # opening title (+ subtitle): first ~3.5 s, inside the platform safe zone, fades in and out
        from .captions import _wrap, fit_chars, lang_cfg
        z = zones(w, h); has_lines = any(ov.get(ln["field"]) for ln in tpl["lines"]); t_end = min(3.6, max(1.6, dur * 0.25))
        y = (0.30 if has_lines else z["top"] + 0.06) if is_portrait(w, h) else (0.22 if has_lines else 0.10)
        font = lang_cfg(lang)["font"] if any(ord(c) > 127 for c in ov["title"] + ov.get("subtitle", "")) else "Noto Sans"
        width = fit_chars(lang, w, h, {"fontSize": 0.044})
        ev.append(f"Dialogue: 2,{_t(0.15)},{_t(t_end)},Ovl,,0,0,0,,{{\\an8\\pos({w // 2},{int(h * y)})\\fn{font}\\fs{int(h * 0.044)}\\b1\\fad(450,500)}}{_wrap(ov['title'], width)}")
        if ov.get("subtitle"):
            n_lines = _wrap(ov["title"], width).count("\\N") + 1
            ev.append(f"Dialogue: 2,{_t(0.35)},{_t(t_end)},Ovl,,0,0,0,,{{\\an8\\pos({w // 2},{int(h * (y + 0.05 * n_lines + 0.008))})\\fn{font}\\fs{int(h * 0.028)}\\fad(450,500)}}{_wrap(ov['subtitle'], fit_chars(lang, w, h, {'fontSize': 0.028}))}")
    if tpl.get("brand") and text_brand:
        st = max(0, dur - 2.5) if tpl.get("brandOnlyAtEnd") else 0
        ev.append(f"Dialogue: 0,{_t(st)},{_t(dur)},Brand,,0,0,0,,{{\\fad(300,300)}}Aikyam")
    if cap["enabled"]:
        ev += caption_events(cap.get("cues", []), "Cap", lang, {**cap.get("style", {}), "mode": cap.get("mode")}, w, h)
    return ass_header(w, h, styles) + "\n".join(ev) + "\n"


def _esc_filter_path(p: str) -> str:
    return p.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def _ken_burns(s: dict, info, W: int, H: int, fps: int, L: float) -> str:
    """Ken Burns on a looped still, in ASPECT-INDEPENDENT terms: zoom 1 = the largest window of the output aspect that fits the image, zoom z shows 1/z of it;
    `center` (normalised image coords) moves from c0 to c1, both eased. The image is scaled once (static), then per frame (zoom) and cropped (pan)."""
    m = s.get("motion") or {}
    z0, z1 = (m.get("zoom") or [1.0, 1.1]); (cx0, cy0), (cx1, cy1) = (m.get("center") or [[0.5, 0.5], [0.5, 0.5]])
    A = W / H; w0 = info.height * A if info.width / info.height >= A else float(info.width)        # base window width in source pixels
    zmax = max(z0, z1); pre = W * zmax / w0
    u = f"min(t/{L:.3f},1)"; e = f"(3*pow({u},2)-2*pow({u},3))" if m.get("easing", "smooth") == "smooth" else u
    Z = f"({z0}+({z1 - z0})*{e})"; cx = f"({cx0}+({cx1 - cx0})*{e})"; cy = f"({cy0}+({cy1 - cy0})*{e})"
    PW, PH = int(info.width * pre) // 2 * 2, int(info.height * pre) // 2 * 2      # the pre-scaled image; crop's own iw/ih can be stale while scale eval=frame resizes, so use the known size
    sw, sh = f"({PW}*{Z}/{zmax})", f"({PH}*{Z}/{zmax})"
    return (f"scale=w={PW}:h={PH},setsar=1,fps={fps},format=yuv420p,"
            f"scale=w='max({W},trunc({sw}/2)*2)':h='max({H},trunc({sh}/2)*2)':eval=frame,"   # never smaller than the output window (rounding)
            f"crop={W}:{H}:x='max(0,min({sw}-{W},{cx}*{sw}-{W}/2))':y='max(0,min({sh}-{H},{cy}*{sh}-{H}/2))',setsar=1,format=yuv420p")


def render_format(plan: dict, src: str, out_path: str, fmt: str, workdir: str, fps: int = 30, mix=None) -> str:
    """Render one output format. `mix` (creative.mixer.MixResult) supplies the finished audio: then the graph is picture-only and the mix is muxed in.
    Without it (classic engine) the audio is built inside the filtergraph. Per clip: layout (crop | fit-with-blur, optional push-in); per edge: cut/blend."""
    from .creative import layout as _layout
    spec = FORMATS[fmt]
    validate_plan(plan)
    W, H = spec["w"], spec["h"]
    aspect = W / H
    with_audio = mix is None
    assets = plan.get("assets") or []
    if assets and with_audio:
        raise ValueError("multi-asset plans need the creative audio mixer (pass mix=)")
    # ---- inputs: one per video/image asset that a segment uses (legacy single-source plans: `src` only)
    inputs, idx, infos = [], {}, {}
    if assets:
        used = [a for a in assets if a["kind"] in ("video", "image") and any(s.get("assetId") == a["id"] for s in plan["segments"])]
        for a in used:
            idx[a["id"]] = len(idx); infos[a["id"]] = ff.probe(a["path"])
            inputs += (["-loop", "1", "-framerate", str(fps), "-i", a["path"]] if a["kind"] == "image" else ["-i", a["path"]])
    else:
        idx[None] = 0; infos[None] = ff.probe(src); inputs = ["-i", src]
    n_in = len(idx)                       # index of the next input file (logo, audio)
    kinds = {a["id"]: a["kind"] for a in assets}
    fade = plan["transitions"]["type"] == "fade" and plan["transitions"]["durationSeconds"] > 0   # legacy dip-to-black
    fd = plan["transitions"]["durationSeconds"]
    chains, labels, layouts = [], "", []
    for i, s in enumerate(plan["segments"]):
        aid = s.get("assetId"); k = idx[aid]; info = infos[aid]
        is_img = (s.get("kind") or kinds.get(aid, "video")) == "image"
        lay = _layout.decide(s, info.width, info.height, aspect); layouts.append(lay)
        L = s["end"] - s["start"]
        sp = float(s.get("speed") or 1.0) if not is_img else 1.0
        base = (f"[{k}:v]trim=end={L:.3f},setpts=PTS-STARTPTS" if is_img else
                (f"[{k}:v]trim=start={s['start']}:end={s['end']},setpts=PTS-STARTPTS" if sp >= 0.999 else
                 f"[{k}:v]trim=start={s['start']}:end={s['start'] + L * sp:.3f},setpts=(PTS-STARTPTS)/{sp},minterpolate=fps={fps}:mi_mode=blend"))      # slow motion: L*sp source seconds stretched over L, frames blended so it does not stutter
        if info.hdr and not is_img:                    # HDR (HLG / PQ) phone footage: tone-map to SDR BT.709, otherwise it comes out flat and washed out next to SDR clips
            base += "," + TONEMAP
        if s.get("grade") and not is_img:              # light colour match to the rest of the reel (creative/grade.py)
            from .creative import grade as _grade
            if _grade.filter_expr(s["grade"]): base += "," + _grade.filter_expr(s["grade"])
        if lay["mode"] == "fit_blur":       # a (tracked) window of the frame fitted to the width over a blurred, darkened copy of the whole frame
            f = min(1.0, float(lay.get("window", 1.0))); fw = int(round(f * info.width / 2)) * 2
            path = s.get("subjectPath") or []
            fx = path_expr([(t, px) for t, px in path], info.width, fw) if (f < 0.999 and len(path) > 1) else str(int(min(max((s.get("subjectX", 0.5) * info.width - fw / 2), 0), info.width - fw)) // 2 * 2)
            fg = f"crop={fw}:{info.height}:x={fx}:y=0," if f < 0.999 else ""
            v = (f"{base},split=2[bg{i}][fg{i}];"
                 f"[bg{i}]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=luma_radius=30:luma_power=2,eq=brightness=-0.08[b{i}];"
                 f"[fg{i}]{fg}scale={W}:-2:force_original_aspect_ratio=decrease,setsar=1[f{i}];"
                 f"[b{i}][f{i}]overlay=(W-w)/2:(H-h)/2,setsar=1,fps={fps},format=yuv420p")
            if is_img:                       # a fitted still still moves: gentle push on the composite
                p = 0.04
                v += (f",scale=w='trunc(iw*(1+{p}*t/{L:.3f})/2)*2':h='trunc(ih*(1+{p}*t/{L:.3f})/2)*2':eval=frame,crop={W}:{H}:x='(iw-{W})/2':y='(ih-{H})/2',setsar=1,format=yuv420p")
        elif is_img:
            v = f"{base},{_ken_burns(s, info, W, H, fps, L)}"
        else:
            x, y, cw, ch = crop_box(info.width, info.height, aspect, s.get("subjectX", 0.5))
            path = s.get("subjectPath") or []
            xe = path_expr([(t, px) for t, px in path], info.width, cw) if len(path) > 1 and cw < info.width else str(x)   # tracked subject
            v = f"{base},crop={cw}:{ch}:x={xe}:y={y},scale={W}:{H}:flags=lanczos,setsar=1,fps={fps},format=yuv420p"
            if lay["pushIn"] > 0:            # slow push-in on the framed subject: scale up over the clip, re-crop the centre
                p = lay["pushIn"]
                v += (f",scale=w='trunc(iw*(1+{p}*t/{L:.3f})/2)*2':h='trunc(ih*(1+{p}*t/{L:.3f})/2)*2':eval=frame,"
                      f"crop={W}:{H}:x='(iw-{W})/2':y='(ih-{H})/2',setsar=1,format=yuv420p")
        if fade:
            v += f",fade=t=in:d={fd},fade=t=out:st={max(0, L - fd)}:d={fd}"
        chains.append(f"{v}[v{i}]")
        if with_audio:
            a = f"[{k}:a]atrim=start={s['start']}:end={s['end']},asetpts=PTS-STARTPTS,aresample=48000"
            if fade: a += f",afade=t=in:d={fd},afade=t=out:st={max(0, L - fd)}:d={fd}"
            chains.append(f"{a}[a{i}]"); labels += f"[v{i}][a{i}]"
        else:
            labels += f"[v{i}]"
    n = len(plan["segments"])
    dur = plan["durationSeconds"]
    use_logo = LOGO.is_file()
    ass = os.path.join(workdir, f"overlay_{fmt}.ass")
    Path(ass).write_text(build_ass(plan, W, H, text_brand=not use_logo), encoding="utf-8")
    XFADE = {"crossfade": "fade", "dip_black": "fadeblack", "dip_white": "fadewhite"}
    edges = edge_transitions(plan["segments"], plan["transitions"])
    lens = [s["end"] - s["start"] for s in plan["segments"]]
    vl, al, acc, joined = "[v0]", "[a0]", lens[0], []
    if n == 1:
        joined = [f"{vl}null[vc]"] + ([f"{al}anull[ac]"] if with_audio else [])
    else:
        # per-edge joins: blend (xfade [+ acrossfade]) where the plan says so, hard cut (concat) elsewhere. `acc` = output length so far.
        for i in range(1, n):
            e = edges[i - 1]
            if e["type"] in OVERLAPPING and e["durationSeconds"] > 0:
                d = e["durationSeconds"]
                joined.append(f"{vl}[v{i}]xfade=transition={XFADE[e['type']]}:duration={d}:offset={acc - d:.3f}[xv{i}]")
                if with_audio: joined.append(f"{al}[a{i}]acrossfade=d={d}:c1=tri:c2=tri[xa{i}]")
                acc += lens[i] - d
            else:
                # concat outputs timebase 1/1000000; xfade needs every input at the segments' 1/fps: reset it with fps
                if with_audio: joined += [f"{vl}{al}[v{i}][a{i}]concat=n=2:v=1:a=1[cv{i}][xa{i}]", f"[cv{i}]fps={fps}[xv{i}]"]
                else: joined += [f"{vl}[v{i}]concat=n=2:v=1:a=0[cv{i}]", f"[cv{i}]fps={fps}[xv{i}]"]
                acc += lens[i]
            vl, al = f"[xv{i}]", f"[xa{i}]"
        joined += [f"{vl}null[vc]"] + ([f"{al}anull[ac]"] if with_audio else [])
    graph = chains + joined + [f"[vc]ass='{_esc_filter_path(ass)}'[vs]"]
    if use_logo:                                                            # logo image (top-right, 13% of the frame width)
        inputs += ["-i", str(LOGO)]
        li, n_in = n_in, n_in + 1
        graph += [f"[{li}:v]scale={int(W * 0.13)}:-1[logo]", f"[vs][logo]overlay=W-w-{int(W * (zones(W, H)['side'] if is_portrait(W, H) else 0.03))}:{int(H * (zones(W, H)['top'] + 0.005 if is_portrait(W, H) else 0.03))}[vo]"]
    else:
        graph += ["[vs]null[vo]"]
    outro = float(plan.get("creative", {}).get("outro", {}).get("fadeSeconds", 0))
    graph += [f"[vo]fade=t=out:st={max(0.0, dur - outro):.3f}:d={outro:.3f}[vout]" if outro > 0 else "[vo]null[vout]"]
    if with_audio:
        # classic path: original audio, optional licensed music ducked under it, loudness-normalised
        music = plan["audio"].get("music", {})
        norm = AudioProcessor.normalize() if plan["audio"].get("normalize") else "anull"
        if music.get("enabled"):
            inputs += ["-stream_loop", "-1", "-i", music_path(music["trackId"])]
            mi = n_in
            graph += [f"[ac]asplit=2[voice][sc]", music_chain(dur, music.get("volume", 0.5), src=f"{mi}:a"),
                      AudioProcessor.duck(voice="sc", music="mus", out="duck"), f"[voice][duck]{AudioProcessor.mix(2)}[mixed]", f"[mixed]{norm}[aout]"]
        else:
            graph += [f"[ac]{norm}[aout]"]
        amap = ["-map", "[aout]"]
    else:                                                                   # creative path: the finished mix is muxed in as-is
        from scipy.io import wavfile
        wav = os.path.join(workdir, "mix.wav"); wavfile.write(wav, 48000, mix.pcm.astype("float32"))
        inputs += ["-i", wav]; amap = ["-map", f"{n_in}:a"]
    ids = plan["source"]
    meta = json.dumps({k: ids.get(k) for k in ("videoId", "templeId", "deityId", "ritualId", "festivalId")}
                      | {"template": plan["overlays"].get("template"), "generator": "aikyam-video"})
    ff.run([*inputs, "-filter_complex", ";".join(graph), "-map", "[vout]", *amap, "-t", f"{dur}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-movflags", "+faststart",
            "-metadata", f"comment={meta}", "-metadata", f"title={plan['overlays'].get('temple') or 'Aikyam'}",
            out_path])
    return out_path
