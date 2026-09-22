"""Third renderer: EditPlan -> diffusionstudio/editor composition (MPL-2.0, renderer/diffusion/) -> picture rendered in headless Chromium -> FFmpeg finishing pass.

Division of labour (so QC and the audio guarantees stay exactly the same as for the FFmpeg renderer):
  diffusion : the PICTURE only - clip placement, crop / pan / track / push-in, fit-with-blur, Ken Burns on stills, dissolve and dip transitions.
  FFmpeg    : captions and title overlays (libass: Indic shaping, safe zones), logo, outro fade, and our finished audio mix (WebCodecs here cannot encode AAC).
Plan timing is the single source of truth (plan.timeline / edge_overlaps), same as every other renderer."""
from __future__ import annotations
import json, os, subprocess
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from . import ffmpeg as ff
from .creative import layout as _layout
from .plan import asset_map, edge_transitions, seg_source, timeline, validate_plan
from .reframe import MAX_PAN  # noqa: F401  (the tracked path is already speed-limited by the planner)
from .render import FORMATS, LOGO, _esc_filter_path, build_ass
from .safezone import is_portrait, zones

HARNESS = Path(os.environ.get("AIKYAM_DIFFUSION_DIR") or Path(__file__).resolve().parent.parent / "renderer" / "diffusion")
DIP = {"dip_black": "#000000", "dip_white": "#ffffff"}


def _n(x: float) -> str:
    return f"{x:.3f}".rstrip("0").rstrip(".") if abs(x) >= 1e-9 else "0"


def _kf(prop: str, pts: List[tuple], easing: str = "linear") -> str:
    """<keyframeTrack>: pts = [(source-local time, value)]. Times are SOURCE-local in this runtime (0 = first frame of the source, not of the clip)."""
    return (f'<KeyframeTrack property="{prop}">' + "".join(f'<Keyframe time={{{_n(t)}}} value={{{_n(v)}}} easing="{easing}" />' for t, v in pts) + "</KeyframeTrack>")


def _interp(path, t: float, default: float) -> float:
    return float(np.interp(t, [p[0] for p in path], [p[1] for p in path])) if len(path) > 1 else default


def _place(W, H, Wb, Hb, px, py, s):
    """x, y (top-left, px) of a Wb x Hb box, scaled by s ABOUT ITS CENTRE (measured against the FFmpeg render: the runtime's `scale` pivots on the box centre, not its origin),
    so that the box point (px, py) lands at the frame centre while the frame stays covered."""
    x = W / 2 - Wb / 2 - s * (px - Wb / 2); y = H / 2 - Hb / 2 - s * (py - Hb / 2)
    return float(np.clip(x, W - Wb / 2 - s * Wb / 2, (s - 1) * Wb / 2)), float(np.clip(y, H - Hb / 2 - s * Hb / 2, (s - 1) * Hb / 2))


def _fade_in(seg: dict, i: int, d: float, edges: List[dict], t_local0: float) -> Optional[str]:
    """Opacity keyframes for the incoming side of a dissolve (the clip fades in over the overlap)."""
    if i == 0 or edges[i - 1]["type"] != "crossfade" or edges[i - 1]["durationSeconds"] <= 0:
        return None
    d = edges[i - 1]["durationSeconds"]
    return _kf("opacity", [(t_local0, 0.0), (t_local0 + d, 1.0)])


def _video(seg, i, t0, t1, W, H, info, lay, edges, src) -> str:
    L = seg["end"] - seg["start"]; sa = info.width / info.height; fade = _fade_in(seg, i, 0, edges, seg["start"]); tl = "" if fade is None else fade
    path = seg.get("subjectPath") or []; cx0 = float(seg.get("subjectX", 0.5))
    ts = sorted({0.0, L, *[float(p[0]) for p in path if 0 <= p[0] <= L]}) if (len(path) > 1 or lay.get("pushIn", 0) > 0) else [0.0]
    common = f'src="{src}" start={{{_n(t0)}}} end={{{_n(t1)}}} sourceIn={{{_n(seg["start"])}}} muted'
    if lay["mode"] == "fit_blur":
        f = min(1.0, float(lay.get("window", 1.0)))
        bw, bh = (H * sa, H) if sa >= W / H else (W, W / sa)
        bx, by = (W - bw) / 2, (H - bh) / 2
        fw = (W / f) if f < 0.999 else W; fh = fw / sa
        def fg_xy(t):
            cx = _interp(path, t, cx0)
            return (float(np.clip(W / 2 - cx * fw, W - fw, 0.0)) if f < 0.999 else 0.0), (H - fh) / 2
        bg = f'<Video {common} x={{{_n(bx)}}} y={{{_n(by)}}} width={{{_n(bw)}}} height={{{_n(bh)}}} objectFit="fill"><Effect type="blur" value={{30}} />{tl}</Video>'
        if f < 0.999 and len(path) > 1:
            tr = _kf("x", [(seg["start"] + t, fg_xy(t)[0]) for t in ts])
            return bg + f'<Video {common} y={{{_n(fg_xy(0)[1])}}} width={{{_n(fw)}}} height={{{_n(fh)}}} objectFit="fill">{tr}{tl}</Video>'
        x, y = fg_xy(0)
        return bg + f'<Video {common} x={{{_n(x)}}} y={{{_n(y)}}} width={{{_n(fw)}}} height={{{_n(fh)}}} objectFit="fill">{tl}</Video>'
    Wb, Hb = (H * sa, float(H)) if sa >= W / H else (float(W), W / sa)
    def state(t):
        s = 1.0 + float(lay.get("pushIn", 0.0)) * t / max(L, 1e-6)
        px = _interp(path, t, cx0) * Wb if sa >= W / H else Wb / 2
        x, y = _place(W, H, Wb, Hb, px, Hb / 2, s)
        return x, y, s
    if len(ts) == 1:
        x, y, s = state(0.0)
        return f'<Video {common} x={{{_n(x)}}} y={{{_n(y)}}} width={{{_n(Wb)}}} height={{{_n(Hb)}}} scale={{{_n(s)}}} objectFit="fill">{tl}</Video>'
    st = [state(t) for t in ts]; x0, y0, s0 = st[0]
    tr = _kf("x", [(seg["start"] + t, v[0]) for t, v in zip(ts, st)]) + _kf("y", [(seg["start"] + t, v[1]) for t, v in zip(ts, st)]) + _kf("scale", [(seg["start"] + t, v[2]) for t, v in zip(ts, st)])
    return f'<Video {common} x={{{_n(x0)}}} y={{{_n(y0)}}} width={{{_n(Wb)}}} height={{{_n(Hb)}}} objectFit="fill">{tr}{tl}</Video>'


def _image(seg, i, t0, t1, W, H, info, edges, src) -> str:
    """Ken Burns: `motion.zoom` z (1 = the largest window of the output aspect inside the image = cover) and `motion.center` (normalised image coords), eased."""
    L = t1 - t0; m = seg.get("motion") or {"zoom": [1.0, 1.06], "center": [[0.5, 0.5], [0.5, 0.5]]}
    sa = info.width / info.height; Wb, Hb = (H * sa, float(H)) if sa >= W / H else (float(W), W / sa)
    (z0, z1), ((a0, b0), (a1, b1)) = m["zoom"], m["center"]
    x0, y0 = _place(W, H, Wb, Hb, a0 * Wb, b0 * Hb, z0); x1, y1 = _place(W, H, Wb, Hb, a1 * Wb, b1 * Hb, z1)
    ez = "easeInOut" if m.get("easing", "smooth") == "smooth" else "linear"
    fade = _fade_in(seg, i, 0, edges, 0.0) or ""
    tr = _kf("x", [(0, x0), (L, x1)], ez) + _kf("y", [(0, y0), (L, y1)], ez) + _kf("scale", [(0, z0), (L, z1)], ez)
    return f'<Image src="{src}" start={{{_n(t0)}}} end={{{_n(t1)}}} x={{{_n(x0)}}} y={{{_n(y0)}}} width={{{_n(Wb)}}} height={{{_n(Hb)}}} objectFit="fill">{tr}{fade}</Image>'


def build_composition(plan: dict, W: int, H: int, src: Optional[str] = None, infos: Optional[Dict[str, object]] = None, urls: Optional[Dict[str, str]] = None) -> str:
    """The TSX text of the plan's picture. `infos`: {assetId: MediaInfo}; `urls`: {assetId: src string} (default: the asset's path)."""
    segs, tr = plan["segments"], plan["transitions"]; tl = timeline(segs, tr); edges = edge_transitions(segs, tr); amap = asset_map(plan); parts: List[str] = []
    for i, s in enumerate(segs):
        aid = s.get("assetId"); a = amap.get(aid); kind = s.get("kind") or (a["kind"] if a else "video")
        path = a["path"] if a else src; info = infos[aid] if infos else ff.probe(path); u = (urls or {}).get(aid, path)
        if kind == "image":
            parts.append(_image(s, i, tl[i]["start"], tl[i]["end"], W, H, info, edges, u))
        else:
            lay = _layout.decide(s, info.width, info.height, W / H)
            parts.append(_video(s, i, tl[i]["start"], tl[i]["end"], W, H, info, lay, edges, u))
    for i, e in enumerate(edges):                                   # dips: a full-frame colour rect fades 0 -> 1 -> 0 over the overlap, hiding the cut at its peak
        if e["type"] in DIP and e["durationSeconds"] > 0:
            d = e["durationSeconds"]; t0 = tl[i + 1]["start"]
            parts.append(f'<Rect x={{0}} y={{0}} width={{{W}}} height={{{H}}} fill="{DIP[e["type"]]}" start={{{_n(t0)}}} end={{{_n(t0 + d)}}}>' + _kf("opacity", [(0, 0), (d / 2, 1), (d, 0)]) + "</Rect>")
    return ('import { Stage, Scene, Video, Image, Rect, Effect, KeyframeTrack, Keyframe } from "@diffusionstudio/jsx";\n'
            f'export default function Project() {{ return (<Stage><Scene width={{{W}}} height={{{H}}} fill="#000000" active>\n' + "\n".join(parts) + "\n</Scene></Stage>); }\n")


def _picture(tsx_path: str, out: str, fps: int, bitrate: int) -> None:
    if not (HARNESS / "dist" / "harness.js").is_file():
        raise RuntimeError(f"the diffusion harness is not built: run {HARNESS}/setup.sh && (cd {HARNESS} && node build.mjs)")
    tsx_path, out = os.path.abspath(tsx_path), os.path.abspath(out)          # the harness runs from its own folder
    r = subprocess.run(["node", str(HARNESS / "render.mjs"), tsx_path, out, "--fps", str(fps), "--bitrate", str(bitrate)], capture_output=True, text=True, cwd=str(HARNESS))
    if r.returncode or not os.path.isfile(out):
        raise RuntimeError("diffusion render failed: " + (r.stderr or r.stdout)[-800:])


def render_format(plan: dict, src: str, out_path: str, fmt: str, workdir: str, fps: int = 30, mix=None) -> str:
    """Same contract as render.render_format (creative path: `mix` = the finished audio)."""
    if mix is None:
        raise ValueError("the diffusion renderer needs the creative audio mixer (pass mix=)")
    validate_plan(plan); spec = FORMATS[fmt]; W, H = spec["w"], spec["h"]; dur = plan["durationSeconds"]
    amap = asset_map(plan); paths = {a["id"]: a["path"] for a in plan.get("assets", [])}
    # every distinct file becomes an absolute quoted path in the TSX; render.mjs serves it over http and rewrites the string
    infos = {aid: ff.probe(p) for aid, p in paths.items()} if amap else {None: ff.probe(src)}
    tsx = build_composition(plan, W, H, src, infos)
    os.makedirs(workdir, exist_ok=True); tsx_path = os.path.join(workdir, f"composition_{fmt}.tsx"); Path(tsx_path).write_text(tsx, encoding="utf-8")
    pic = os.path.join(workdir, f"picture_{fmt}.mp4"); _picture(tsx_path, pic, fps, 12_000_000)
    use_logo = LOGO.is_file(); ass = os.path.join(workdir, f"overlay_{fmt}.ass"); Path(ass).write_text(build_ass(plan, W, H, text_brand=not use_logo), encoding="utf-8")
    from scipy.io import wavfile
    wav = os.path.join(workdir, "mix.wav"); wavfile.write(wav, 48000, mix.pcm.astype("float32"))
    inputs = ["-i", pic]; graph = [f"[0:v]ass='{_esc_filter_path(ass)}'[vs]"]; n_in = 1
    if use_logo:
        inputs += ["-i", str(LOGO)]; n_in = 2
        side = zones(W, H)["side"] if is_portrait(W, H) else 0.03; top = zones(W, H)["top"] + 0.005 if is_portrait(W, H) else 0.03
        graph += [f"[1:v]scale={int(W * 0.13)}:-1[logo]", f"[vs][logo]overlay=W-w-{int(W * side)}:{int(H * top)}[vo]"]
    else:
        graph += ["[vs]null[vo]"]
    outro = float(plan.get("creative", {}).get("outro", {}).get("fadeSeconds", 0))
    graph += [f"[vo]fade=t=out:st={max(0.0, dur - outro):.3f}:d={outro:.3f}[vout]" if outro > 0 else "[vo]null[vout]"]
    inputs += ["-i", wav]; ids = plan["source"]
    meta = json.dumps({k: ids.get(k) for k in ("videoId", "templeId", "deityId", "ritualId", "festivalId")} | {"template": plan["overlays"].get("template"), "generator": "aikyam-video/diffusion"})
    ff.run([*inputs, "-filter_complex", ";".join(graph), "-map", "[vout]", "-map", f"{n_in}:a", "-t", f"{dur}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-movflags", "+faststart", "-metadata", f"comment={meta}", "-metadata", f"title={plan['overlays'].get('temple') or 'Aikyam'}", out_path])
    return out_path
