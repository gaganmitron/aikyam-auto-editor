"""In-process pipeline: video -> transcript/scenes/moments/edit-plan -> renders + thumbnail. Composes `stages`."""
from __future__ import annotations
import json, os, time
from typing import Dict, Optional
from . import ffmpeg as ff, stages
from .cost import ProcessingCost, cpu_seconds
from .metrics import METRICS, get_logger, timed
from .options import Options, Profile  # noqa: F401 (re-exported: pipeline.Options)
from .render import FORMATS

STAGES = ["analyze", "highlights", "plan", "render", "process"]
SUPPORTED_CODECS = {"h264", "hevc", "vp9", "av1", "mpeg4", "mpeg2video", "prores", "vp8"}


def validate_source(path: str, o: Options) -> ff.MediaInfo:
    """Trust boundary: reject unreadable, tiny, huge, or unsupported input before spending compute."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    info = ff.probe(path)
    if not (o.min_source_seconds <= info.duration <= o.max_source_seconds):
        raise ValueError(f"source duration {info.duration:.1f}s outside [{o.min_source_seconds}, {o.max_source_seconds}]")
    if info.video_codec not in SUPPORTED_CODECS:
        raise ValueError(f"unsupported video codec {info.video_codec}")
    return info


def run(path: str, out: str, upto: str = "process", o: Optional[Options] = None, log=None) -> Dict[str, str]:
    o = o or Options()
    assert upto in STAGES, upto
    log = log or get_logger(media_id=o.video_id or os.path.basename(path), temple_id=o.temple_id or "")
    stage = lambda n: (log.info(f"stage {n}", extra={"stage": n}), o.on_stage and o.on_stage(n))
    os.makedirs(out, exist_ok=True)
    cpu0, t0 = cpu_seconds(), time.perf_counter()
    info = validate_source(path, o)
    METRICS.source_seconds.inc(info.duration)
    files: Dict[str, str] = {}
    plan = None
    try:
        with timed(METRICS.media_processing):
            stage("transcription")
            with timed(METRICS.transcription):
                stages.transcription(path, out, o)
            stage("scene_detection")
            with timed(METRICS.scene_detection):
                stages.scene_analysis(path, out, o)
            for k, n in [("transcript", stages.TRANSCRIPT), ("scenes", stages.SCENES)]:
                files[k] = os.path.join(out, n)
            if upto == "analyze":
                with timed(METRICS.highlight_generation):   # entities need both; cheap
                    stages.highlights(path, out, o)
                files["entities"] = os.path.join(out, stages.ENTITIES)
                return files
            stage("highlights")
            with timed(METRICS.highlight_generation):
                stages.highlights(path, out, o)
            files.update(entities=os.path.join(out, stages.ENTITIES), moments=os.path.join(out, stages.MOMENTS))
            if upto == "highlights":
                return files
            stage("edit_plan")
            plan = stages.plan(path, out, o)
            files["edit_plan"] = os.path.join(out, stages.PLAN)
            if upto == "plan":
                return files
            stage("render")
            files.update(stages.render(path, out, o))
    except Exception as e:
        METRICS.processing_failures.labels(stage="pipeline").inc()
        log.error(f"pipeline failed: {e}")
        raise
    tp = stages._cache.get(f"transcription:{o.transcription}")
    vp = stages._cache.get(f"vision:{o.vision}")
    cost = ProcessingCost(cpu_seconds() - cpu0, 0.0,
                          f"{o.transcription}:{getattr(tp, 'model_name', '?')}+{getattr(vp, 'name', '?')}",
                          info.duration, plan["durationSeconds"] * len(o.formats), outputs=len(o.formats)).finalize()
    METRICS.cost_inr.inc(cost.inr); METRICS.cpu_seconds.inc(cost.cpuSeconds)
    with open(os.path.join(out, "cost.json"), "w") as f:
        json.dump({**cost.dict(), "wallSeconds": round(time.perf_counter() - t0, 2)}, f, indent=2)
    files["cost"] = os.path.join(out, "cost.json")
    with open(os.path.join(out, "metrics.prom"), "wb") as f:   # CLI runs are observable too (node_exporter textfile format)
        f.write(METRICS.expose())
    return files


def run_multi(inputs, out: str, upto: str = "process", o: Optional[Options] = None) -> Dict[str, str]:
    """Several videos / images (+ music files, see multi.py) -> one reel. Creative engine only."""
    from . import multi
    o = o or Options(); os.makedirs(out, exist_ok=True)
    if o.engine != "creative":
        raise ValueError("mixed inputs need the creative engine (--engine creative)")
    p = multi.plan(list(inputs), out, o); files = {"edit_plan": os.path.join(out, stages.PLAN)}
    if upto == "plan":
        return files
    files.update(stages.render(p["source"]["path"], out, o))
    return files
