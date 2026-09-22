"""Remotion renderer: same EditPlan, React/Chromium rendering (richer layouts). FFmpeg stays the default renderer.

Needs Node >= 18, `npm install` in renderer/remotion, and a Chromium/Chrome binary (AIKYAM_CHROMIUM, default /usr/bin/chromium).
Remotion renders picture + the source audio; we then run ONE ffmpeg pass for loudness normalisation, optional ducked music,
and the Aikyam metadata so both renderers' outputs are equivalent for the feed."""
from __future__ import annotations
import json, os, shutil, subprocess, tempfile
from pathlib import Path
from . import ffmpeg as ff
from .audio import AudioProcessor
from .captions import lang_cfg
from .plan import validate_plan
from .render import FORMATS, LOGO, TEMPLATES, music_chain, music_path

PROJECT = Path(os.environ.get("AIKYAM_REMOTION_DIR", Path(__file__).resolve().parent.parent / "renderer" / "remotion"))


def available() -> bool:
    return (PROJECT / "node_modules" / ".bin" / "remotion").exists() and shutil.which("node") is not None \
        and Path(os.environ.get("AIKYAM_CHROMIUM", "/usr/bin/chromium")).exists()


def render_format(plan: dict, src: str, out_path: str, fmt: str, workdir: str, fps: int = 30) -> str:
    if not available():
        raise RuntimeError(f"Remotion not available: run `npm install` in {PROJECT} and install Chromium (AIKYAM_CHROMIUM)")
    spec = FORMATS[fmt]
    validate_plan(plan)
    info = ff.probe(src)
    with tempfile.TemporaryDirectory() as td:
        pub = Path(td) / "public"; pub.mkdir()
        try: os.link(src, pub / "source.mp4")
        except OSError: shutil.copyfile(src, pub / "source.mp4")
        if LOGO.is_file(): shutil.copyfile(LOGO, pub / "logo.png")
        props = {"plan": plan, "template": TEMPLATES.get(plan["overlays"].get("template") or "divine_moment", TEMPLATES["divine_moment"]),
                 "width": spec["w"], "height": spec["h"], "fps": fps, "srcWidth": info.width, "srcHeight": info.height,
                 "videoFile": "source.mp4", "logoFile": "logo.png" if LOGO.is_file() else None,
                 "captionFont": lang_cfg(plan["captions"]["language"])["font"]}
        pj = Path(td) / "props.json"; pj.write_text(json.dumps(props), encoding="utf-8")
        raw = str(Path(td) / "raw.mp4")
        p = subprocess.run(["node_modules/.bin/remotion", "render", "src/index.ts", "AikyamVideo", raw, f"--props={pj}", f"--public-dir={pub}",
                            f"--browser-executable={os.environ.get('AIKYAM_CHROMIUM', '/usr/bin/chromium')}", "--concurrency=2", "--log=error",
                            "--gl=angle"],
                           cwd=PROJECT, capture_output=True, text=True, timeout=3600)
        if p.returncode:
            raise RuntimeError(f"remotion failed: {(p.stderr or p.stdout)[-1500:]}")
        # finalize: normalise loudness (+ ducked music), stamp metadata
        norm = AudioProcessor.normalize() if plan["audio"].get("normalize") else "anull"
        music = plan["audio"].get("music", {})
        ids = plan["source"]
        meta = json.dumps({k: ids.get(k) for k in ("videoId", "templeId", "deityId", "ritualId", "festivalId")}
                          | {"template": plan["overlays"].get("template"), "generator": "aikyam-video/remotion"})
        args = ["-i", raw]
        if music.get("enabled"):
            args += ["-stream_loop", "-1", "-i", music_path(music["trackId"])]
            fg = (f"[0:a]asplit=2[voice][sc];{music_chain(plan['durationSeconds'], music.get('volume', 0.5))};"
                  f"{AudioProcessor.duck(voice='sc', music='mus', out='duck')};[voice][duck]{AudioProcessor.mix(2)}[mixed];[mixed]{norm}[aout]")
        else:
            fg = f"[0:a]{norm}[aout]"
        ff.run([*args, "-filter_complex", fg, "-map", "0:v", "-map", "[aout]", "-t", f"{plan['durationSeconds']}", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
                "-ar", "48000", "-movflags", "+faststart", "-metadata", f"comment={meta}", "-metadata", f"title={plan['overlays'].get('temple') or 'Aikyam'}", out_path])
    return out_path
