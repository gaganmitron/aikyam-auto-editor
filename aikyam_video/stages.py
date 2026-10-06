"""Pipeline stages. Each stage reads/writes named artifacts in one directory, so the same code runs in-process (CLI)
or as separate workers (Kafka) that sync that directory through object storage."""
from __future__ import annotations
import json, os
from typing import Dict, List, Optional
import numpy as np
from . import ffmpeg as ff, kg as _kg, providers, transcribe, vision as _vision, entities as _entities  # noqa: F401 (register)
from .analysis import analyze_scenes, text_track
from .audio import AudioProfile, ClapAudioTagger, profile as audio_profile
from .highlights import build_ctx, detect_black, generate_moments
from .metrics import METRICS
from .models import EntityRef, Moment, Scene, SceneVision, Transcript
from .options import Options
from .plan import plan_edit, validate_plan
from .render import FORMATS, render_format as render_ffmpeg
from .reframe import track_subject
from .scenes import detect_scenes
from .scoring import ScoringConfig
from .thumbnails import generate_thumbnails

# artifact names (the contract between stages / workers)
TRANSCRIPT, SCENES, VISION, EMBEDDINGS = "transcript.json", "scenes.json", "vision.json", "embeddings.json"
AUDIO, BLACK, ENTITIES, MOMENTS, PLAN = "audio.npz", "black.json", "entities.json", "moments.json", "edit-plan.json"
TEXTFRAMES = "textframes.json"          # [[t, no_text]] every 2 s over the whole file (analysis.text_track); absent in older caches = no dense check
ANALYSIS_ARTIFACTS = [SCENES, VISION, EMBEDDINGS, AUDIO, BLACK]
_cache: Dict[str, object] = {}


def _dump(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _load(art, name):
    with open(os.path.join(art, name), encoding="utf-8") as f:
        return json.load(f)


def _provider(kind: str, name: str):
    """Providers are heavy (model weights): build once per process."""
    key = f"{kind}:{name}"
    if key not in _cache:
        _cache[key] = providers.get(kind, name)
    return _cache[key]


def _audio_tagger(o: Options):
    if o.audio_tagger == "none":
        return None
    if "clap" not in _cache:
        try:
            _cache["clap"] = ClapAudioTagger()
        except ImportError:
            import logging
            logging.getLogger("aikyam").warning("transformers not installed: no audio event tagging")
            _cache["clap"] = None
    return _cache["clap"]


def transcription(src: str, art: str, o: Options) -> Transcript:
    if not o.transcript:                                                       # --no-transcript: nothing is transcribed, nothing downstream reads speech
        tr = Transcript(language=o.language or "und", segments=[]); _dump(os.path.join(art, TRANSCRIPT), tr.model_dump()); return tr
    tp = _provider("transcription", o.transcription)
    if o.whisper_model: tp.model_name = o.whisper_model
    tp.language, tp.translate = o.language, o.translate
    tr = tp.transcribe(src)
    _dump(os.path.join(art, TRANSCRIPT), tr.model_dump())
    return tr


def scene_analysis(src: str, art: str, o: Options) -> List[SceneVision]:
    scenes = detect_scenes(src, art)
    vp = _provider("vision", o.vision)
    sv = analyze_scenes(src, scenes, vp)
    _dump(os.path.join(art, SCENES), [s.model_dump() for s in scenes])
    _dump(os.path.join(art, VISION), [s.model_dump() for s in sv])
    _dump(os.path.join(art, EMBEDDINGS), {s.sceneId: s.vision.embedding for s in sv if s.vision.embedding})
    audio_profile(src, tagger=_audio_tagger(o)).save(os.path.join(art, AUDIO))
    _dump(os.path.join(art, BLACK), detect_black(src))
    _dump(os.path.join(art, TEXTFRAMES), text_track(src, vp))
    return sv


def _load_analysis(art):
    scenes = [Scene(**s) for s in _load(art, SCENES)]
    emb = _load(art, EMBEDDINGS) if os.path.exists(os.path.join(art, EMBEDDINGS)) else {}
    vis = []
    for s in _load(art, VISION):
        sv = SceneVision(**s)
        if sv.sceneId in emb: sv.vision.embedding = emb[sv.sceneId]
        vis.append(sv)
    return scenes, vis, AudioProfile.load(os.path.join(art, AUDIO)), [tuple(x) for x in _load(art, BLACK)]


def highlights(src: str, art: str, o: Options):
    info = ff.probe(src)
    tr = Transcript(**_load(art, TRANSCRIPT))
    scenes, vis, audio, black = _load_analysis(art)
    ex = _provider("entity", "kg" if os.environ.get("AIKYAM_KG_URL") else "gazetteer")
    hints = {"TEMPLE": o.temple_id} if o.temple_id else {}
    if o.festival_id: hints["FESTIVAL"] = o.festival_id
    ents = _entities.resolve_media_entities(tr, vis, ex, hints)
    _dump(os.path.join(art, ENTITIES), {k: [r.model_dump() for r in v] for k, v in ents.items()})
    ctx = build_ctx(src, info, scenes, vis, tr, audio, ex, black=black, profile=o.profile, allow_silent=o.allow_silent,
                    text_frames=_load(art, TEXTFRAMES) if os.path.exists(os.path.join(art, TEXTFRAMES)) else None)
    cfg = ScoringConfig.load(o.scoring_config, engine=o.engine)
    if o.silent_source:                                                      # the recorded sound (narration, chanting) is not part of the reel: it must not decide which moments are picked
        cfg.weights = {**cfg.weights, "audioImportance": 0.0, "semanticImportance": 0.0}
    if o.experimental_selection:                                             # opt-in terms; explicit weights from --scoring-config still win
        from . import ranker
        cfg.weights = {**cfg.weights, **{k: v for k, v in {"aesthetic": 0.15, "learned": 0.15 if ranker.load() else 0.0}.items() if not cfg.weights.get(k)}}
    moments, rejected = generate_moments(ctx, cfg, ex, METRICS)
    _dump(os.path.join(art, MOMENTS), {"moments": [m.model_dump() for m in moments], "rejected": [r.model_dump() for r in rejected]})
    return moments, rejected


def _load_entities(art) -> Dict[str, List[EntityRef]]:
    return {k: [EntityRef(**r) for r in v] for k, v in _load(art, ENTITIES).items()}


def quiet_snap(audio: AudioProfile, window: float = 0.75):
    """snap(t): the quietest half-second within +-window of t, so a trimmed clip ends between sounds, not through one."""
    def snap(t: float) -> float:
        if not len(audio.rms_db):
            return t
        lo = int(max(0.0, t - window) / audio.hop); seg = audio.rms_db[lo: int((t + window) / audio.hop) + 1]
        if not len(seg) or seg.min() > np.median(seg) - 6.0:      # no genuinely quieter spot nearby (>= 6 dB): leave the cut alone
            return t
        return (lo + int(np.argmin(seg))) * audio.hop + audio.hop / 2
    return snap


def plan(src: str, art: str, o: Options) -> dict:
    info = ff.probe(src)
    tr = Transcript(**_load(art, TRANSCRIPT))
    _, vis, audio, _ = _load_analysis(art)
    moments = [Moment(**m) for m in _load(art, MOMENTS)["moments"]]
    if not moments:
        raise RuntimeError("no valid highlight found in source")
    if o.engine == "creative":
        from .creative import engine as _eng
        vid = o.video_id or os.path.splitext(os.path.basename(src))[0]
        ctx = _eng.Ctx(src, info, moments, vis, audio, tr, _load_entities(art), o, vid)
        plans = _eng.build_plans(ctx)
        for k, pl in enumerate(plans):
            _dump(os.path.join(art, PLAN if k == 0 else f"edit-plan-{k + 1}.json"), pl)
        return plans[0]
    return _plan_classic(src, art, o, info, tr, vis, audio, moments)


def _plan_classic(src, art, o, info, tr, vis, audio, moments) -> dict:
    spec = FORMATS[o.formats[0]]
    target = (o.profile.target_seconds if o.profile and o.profile.target_seconds else (o.target_seconds or 45.0))
    clang = o.caption_lang or (o.profile.caption_lang if o.profile else None)
    kw = dict(video_id=o.video_id or os.path.splitext(os.path.basename(src))[0], path=os.path.abspath(src), duration=info.duration,
              moments=moments, transcript=tr, vision=vis, entities=_load_entities(art), target_s=target, caption_lang=clang,
              caption_mode=o.caption_mode, aspect=spec["aspect"], fmt=spec["outputFormat"], location=o.location,
              transition=o.transition, transition_s=o.transition_seconds, snap=quiet_snap(audio))
    if o.engine == "v4":
        from .creative import selector_v4
        p = selector_v4.plan_edit(**kw)
    elif o.planner == "llm":
        from .planner_llm import plan_with_llm
        p = plan_with_llm(**kw)
    else:
        p = plan_edit(**kw)
    p["source"]["festivalId"] = p["source"].get("festivalId") or o.festival_id
    from . import music as _music
    if o.music_track:                          # explicit choice: must still be a licensed library track
        p["audio"]["music"] = _music.music_block(_music.get_track(o.music_track), "explicitly requested", o.music_volume)
    elif o.music == "auto":
        track, why = _music.choose(p, audio, _music.load_library()[0], o.music_skip_threshold)
        p["audio"]["music"] = _music.music_block(track, why, o.music_volume) if track else {"enabled": False, "reason": why}
    else:
        p["audio"]["music"] = {"enabled": False, "reason": "music off"}
    for s in p["segments"]:      # subject tracking: a smoothed x-path per segment (renderer-independent, in the plan)
        s["subjectPath"] = [[round(t, 2), round(x, 3)] for t, x in track_subject(src, s["start"], s["end"])]
    if not o.captions:
        p["captions"]["cues"] = []; p["captions"]["enabled"] = False
    if o.engine == "v4" and o.silent_source:                                  # own audio only: the recorded sound is never cut per clip; V4 then renders through the shared mixer (see render)
        from .creative import bed
        p["audio"]["preserveOriginal"] = False; p["audio"]["sourceAudio"] = o.source_audio
        if o.source_audio == "bed":
            pick = bed.pick_across([(audio, src)], p["durationSeconds"] + 1.0)
            if pick: p["audio"]["bed"] = {"start": pick[0], "score": pick[1], "why": pick[2], "path": os.path.abspath(pick[3])}
    validate_plan(p, info.duration)
    _dump(os.path.join(art, PLAN), p)
    return p




def render(src: str, art: str, o: Options) -> Dict[str, str]:
    p = _load(art, PLAN)
    files = {}
    if p.get("creative", {}).get("engine") == "creative" or p["audio"].get("sourceAudio"):      # V4 with --source-audio off/bed shares the creative mixer + QC
        return _render_creative(src, art, o, p)
    thumbs = generate_thumbnails(src, p, _provider("vision", o.vision), art)
    p["thumbnail"]["timestamp"] = thumbs["timestamp"]
    _dump(os.path.join(art, PLAN), p)
    files["thumbnail"] = os.path.join(art, "thumbnail.jpg")
    for f in o.formats:
        try:
            with __import__("aikyam_video.metrics", fromlist=["timed"]).timed(METRICS.render):
                if o.renderer == "remotion":
                    from .render_remotion import render_format as render_fn
                elif o.renderer == "ffmpeg":
                    render_fn = render_ffmpeg
                else:
                    raise ValueError(f"unknown renderer {o.renderer!r}")
                files[f] = render_fn({**p, "outputFormat": FORMATS[f]["outputFormat"], "aspectRatio": FORMATS[f]["aspect"]},
                                     src, os.path.join(art, FORMATS[f]["file"]), f, art)
        except Exception:
            METRICS.render_failures.inc(); raise
        METRICS.output_seconds.inc(p["durationSeconds"])
    return files


def _render_creative(src: str, art: str, o: Options, p: dict) -> Dict[str, str]:
    """Creative engine: audio mix with stems -> picture -> QC on the encoded file -> targeted re-edit if QC fails -> remaining formats. One pass per reel."""
    import logging
    from .creative import engine as _eng
    render_fn = None
    if o.renderer == "remotion":
        logging.getLogger("aikyam").warning("Remotion does not implement per-edge transitions/layouts yet: rendering the creative edit with FFmpeg")
    elif o.renderer == "diffusion":
        from .render_diffusion import render_format as render_fn
    if p.get("assets"):                                                    # multi-asset: one audio profile per video asset
        audio = {a["id"]: AudioProfile.load(os.path.join(art, "assets", a["id"], AUDIO)) for a in p["assets"] if a["kind"] == "video"}
    else:
        audio = AudioProfile.load(os.path.join(art, AUDIO))
    files = {}
    plans = [p] + [_load(art, f"edit-plan-{k}.json") for k in range(2, max(1, o.reels) + 1) if os.path.exists(os.path.join(art, f"edit-plan-{k}.json"))]
    for k, pl in enumerate(plans):
        out = art if k == 0 else os.path.join(art, f"reel{k + 1}"); os.makedirs(out, exist_ok=True)
        if k == 0:
            files["thumbnail"] = os.path.join(art, "thumbnail.jpg")
            th = generate_thumbnails(src, pl, _provider("vision", o.vision), art); pl["thumbnail"]["timestamp"] = th["timestamp"]
        with __import__("aikyam_video.metrics", fromlist=["timed"]).timed(METRICS.render):
            try:
                fx, qcd, final = _eng.render_reel(pl, src, out, o, audio, o.formats, "reel", render_fn)
            except Exception:
                METRICS.render_failures.inc(); raise
        _dump(os.path.join(out, "qc.json"), qcd); _dump(os.path.join(art, PLAN if k == 0 else f"edit-plan-{k + 1}.json"), final)
        for f, path in fx.items(): files[f if k == 0 else f"{f}_{k + 1}"] = path
        if k == 0: files["qc"] = os.path.join(out, "qc.json")
        for _ in o.formats: METRICS.output_seconds.inc(final["durationSeconds"])
    return files
