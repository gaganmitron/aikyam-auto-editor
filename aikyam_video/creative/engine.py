"""Creative Editing Engine: footage analysis -> story -> transitions -> music -> composition -> EditPlan (v2) -> render -> QC -> deterministic re-edit.

  build_plans(ctx)   : one EditPlan per reel (k reels share the footage fairly); every decision is in plan.creative.decisions
  render_reel(...)   : audio mix (stems kept), picture render, QC on the encoded file, targeted re-edit if QC fails, then the other formats
"""
from __future__ import annotations
import copy, dataclasses, json, os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from .. import ffmpeg as ff, music as M
from ..audio import AudioProfile
from ..captions import retime_cues
from ..models import EntityRef, Moment, SceneVision, Transcript
from ..options import Options
from ..plan import assemble_plan, validate_plan
from ..reframe import camera_path, smooth_path, track_subject_ex
from . import bed, grade, layout, mixer, music_sync, pacing, qc as QC, replan, story, transitions
from .model import Clip, Shot, Timeline
from .stills import motion_for
from .musicdna import analyze_track
from .shots import Population, build_shots

EXPERIMENTAL_COVERAGE = 0.5      # story.py coverage weight used by --experimental-selection (EXP-014: synthetic evidence only)
PRESENCE_EVENTS = ("chant", "bhajan", "speech", "bell", "conch")


@dataclass
class Source:
    """One input asset: a video with its analysis, or an image with its precomputed shot. `id` is the plan's assetId."""
    id: str
    path: str
    kind: str                                   # "video" | "image"
    info: Optional["ff.MediaInfo"] = None
    moments: Optional[List[Moment]] = None
    vision: Optional[List[SceneVision]] = None
    audio: Optional[AudioProfile] = None
    transcript: Optional[Transcript] = None
    shot: Optional[Shot] = None                 # images


@dataclass
class Ctx:
    src: str
    info: "ff.MediaInfo"
    moments: List[Moment]
    vision: List[SceneVision]
    audio: Optional[AudioProfile]
    transcript: Transcript
    entities: Dict[str, List[EntityRef]]
    o: Options
    video_id: str
    sources: Optional[List[Source]] = None      # several assets (multi-input); None = the single video in src/moments/...


def _make_edger(by_id, use_audio: bool = True):
    """edger(shot, a, b, min_len, max_len, first, last) -> (a', b', info): put a window's edges on real boundaries of ITS source (cuts, speech phrases, pauses, motion onsets/lulls; see cutpoints)."""
    from . import cutpoints as cp, edges, mlx
    phr = {k: cp.phrases(getattr(S, "transcript", None)) if use_audio else [] for k, S in by_id.items() if getattr(S, "kind", "video") == "video"}
    for k, S in by_id.items():                                                              # (use_audio=False: picture-only edges -- cuts and motion onsets/lulls, never speech)
        if not use_audio: break                                                              # no transcript (e.g. a language Whisper missed): Silero speech segments stand in for phrases
        if k in phr and not phr[k]: phr[k] = [tuple(x) for x in (mlx.speech(S.path) or [])]
    def edger(shot, a, b, lo, hi, first=False, last=False):
        S = by_id[shot.asset_id]; ph = phr.get(shot.asset_id, [])
        ins, outs = cp.candidates(shot, shot.cuts, [tuple(p) for p in shot.pauses] if use_audio else [], ph)
        return cp.choose_edges(a, b, shot.start, shot.end, ins, outs, ph, lo, hi, first=first, last=last, refine=lambda t: edges.refine_boundary(S.path, t, S.info.fps))
    return edger


def presence_fn(audio):
    if isinstance(audio, dict):
        d = {k: f for k, f in ((k, presence_fn(v)) for k, v in audio.items()) if f}
        return d or None
    if audio is None or not getattr(audio, "events", None):
        return None
    return lambda t: max([audio.event_score(k, t, t + 0.5) for k in PRESENCE_EVENTS if k in audio.events] or [0.0])


def _quiet_snap(audio: Optional[AudioProfile], window: float = 0.75):
    if audio is None or not len(audio.rms_db):
        return None
    def snap(t: float) -> float:
        lo = int(max(0.0, t - window) / audio.hop); seg = audio.rms_db[lo: int((t + window) / audio.hop) + 1]
        if not len(seg) or seg.min() > np.median(seg) - 6.0:                # no genuinely quieter spot nearby (>= 6 dB): leave the cut alone
            return t
        return (lo + int(np.argmin(seg))) * audio.hop + audio.hop / 2
    return snap


# ------------------------------------------------------------------ music choice (context + audio conflicts + tempo/pacing fit)
def _kind(track: M.Track) -> str:
    return track.kind or analyze_track(track.path).kind


def choose_music(plan_like: dict, audio, profile: pacing.PacingProfile, o: Options):
    """(Track|None, reason). Forced id > auto. Auto respects the live sound: strongly devotional live sound -> none; moderate -> drones only."""
    if o.music_track:
        t = M.get_track(o.music_track); return t, "explicitly requested"
    if o.source_audio == "bed":
        return None, "the source's own audio is the soundtrack (bed)"
    if o.music != "auto":
        return None, "music off"
    tracks, _ = M.load_library()
    if not tracks:
        return None, "no licensed music library"
    silent = o.silent_source
    lvl = 0.0 if silent else M.original_devotional_level(plan_like, audio)
    if lvl >= o.music_skip_threshold:
        return None, f"original audio is already devotional music/chanting ({lvl:.2f} >= {o.music_skip_threshold})"
    allowed = {"drone"} if lvl >= 0.35 else {"drone", "melodic", "rhythmic"}
    ranked = []
    for t in tracks:
        k = _kind(t)
        if k not in allowed:
            continue
        s = M.score(t, plan_like)
        s += {"festive": 0.25 if k == "rhythmic" else -0.05, "contemplative": -0.4 if k == "rhythmic" else 0.15, "devotional": 0.0}[profile.name]
        ranked.append((round(s, 3), t.id, t))
    if not ranked:
        return None, f"no track fits the live sound (devotional level {lvl:.2f} allows only drones)"
    ranked.sort(key=lambda x: (-x[0], x[1])); s, _, best = ranked[0]
    if s < 0.3 and not silent:                                  # a silent-source reel needs music more than it needs a perfect match: take the best one
        return None, f"no track matches this reel (best {best.id} scored {s})"
    return best, f"matched '{best.id}' ({_kind(best)}, score {s}; " + ("recorded sound off" if silent else f"live devotional level {lvl:.2f}") + f", pacing {profile.name})"


# ------------------------------------------------------------------ plan building
def _reason(clip: Clip) -> str:
    top = max(clip.shot.labels.items(), key=lambda kv: kv[1], default=(None, 0))
    return top[0] if top[0] and top[1] >= 0.4 else clip.role.lower()


def _ids(entities: Dict[str, List[EntityRef]], typ: str) -> Optional[str]:
    l = entities.get(typ); return max(l, key=lambda r: r.confidence).entityId if l else None


def build_plans(ctx: Ctx) -> List[dict]:
    o = ctx.o; D = ctx.info.duration
    multi = ctx.sources is not None
    srcs = ctx.sources or [Source("a1", ctx.src, "video", ctx.info, ctx.moments, ctx.vision, ctx.audio, ctx.transcript)]
    by_id = {x.id: x for x in srcs}; shots = []
    silent = o.silent_source          # the recorded sound is not part of the reel: no audio evidence in selection, cuts, transitions or the mix
    for x in srcs:
        if x.kind == "image":
            shots.append(x.shot)
        else:
            sh = build_shots(x.path, x.moments, x.vision, None if silent else x.audio, top_k=16 if len(srcs) == 1 else 8, asset_id=x.id, edge_info=o.edge_snap, duration=x.info.duration, use_speech=not silent, diverse=o.beats)
            if multi:
                for k in sh: k.position = (k.start + k.end) / 2 / max(x.info.duration, 1e-9); k.id = f"{x.id}_{k.id}"     # moment ids repeat across videos: shot ids must not
            shots += sh
    if not o.beats:
        for k in shots: k.beats = {}                                          # --no-beats: story planning ignores story beats
    videos = [x for x in srcs if x.kind == "video"]
    avail_s = sum(x.info.duration for x in videos) + 3.0 * sum(1 for x in srcs if x.kind == "image")      # what the footage can supply: the reel-length target may not ask for more
    if o.order:
        bad = [a for a in o.order if a not in by_id]
        if bad: raise ValueError(f"--order: unknown asset {bad}; valid ids: {sorted(by_id)}")
        if o.reels > 1: raise ValueError("--order fixes ONE reel: use --reels 1")
    audio_of = {x.id: x.audio for x in videos}; ctx_audio = ctx.audio if not multi else next((x.audio for x in videos if x.audio), None)
    if not shots:
        raise RuntimeError("no analysable shots")
    pop = Population(shots); prof = pacing.choose_profile(shots, o.pacing)
    if o.opening: prof = dataclasses.replace(prof, opening=o.opening)
    snap = None if silent else _quiet_snap(ctx_audio)
    director = None
    if o.planner == "llm":
        from ..planner_llm import make_director
        director = make_director(getattr(o, "llm", None))
    notes = {k.id: " ".join(t.text for t in x.transcript.segments if t.end > k.start and t.start < k.end) for x in videos if x.transcript for k in shots if k.asset_id == x.id}
    edger = _make_edger(by_id, use_audio=not silent) if o.edge_snap else None
    stories = story.plan_stories(shots, pop, prof, D, max(1, o.reels), o.target_seconds, snap, director=director, notes=notes, order=o.order, edger=edger, motion_dedupe=o.motion_dedupe, hook_first=o.hook_first, coverage=EXPERIMENTAL_COVERAGE if o.experimental_selection else None)
    if not stories:
        raise RuntimeError("no story could be built from the candidate shots")
    plans = []
    for tl in stories:
        # preliminary plan-like dict for the music decision
        prelim = {"source": {"ritualId": _ids(ctx.entities, "RITUAL"), "deityId": _ids(ctx.entities, "DEITY"), "festivalId": _ids(ctx.entities, "FESTIVAL")},
                  "segments": [{"start": c.start, "end": c.end, "reason": _reason(c), "score": c.shot.score} for c in tl.clips]}
        track, why = choose_music(prelim, ctx_audio, prof, o)
        if o.music_web and not o.music_track and o.music == "auto" and (o.silent_source or M.original_devotional_level(prelim, ctx_audio) < o.music_skip_threshold):       # the live sound is not already devotional music (or is not used at all)
            from .. import music_web
            lab = {}
            for c in tl.clips:
                for k, v in c.shot.labels.items(): lab[k] = max(lab.get(k, 0.0), v)
            wt, wwhy = music_web.find(lab, prof.name, sum(c.length for c in tl.clips), n_download=12 if silent else 8, beat_bonus=0.20 if silent else 0.0, extra=["bhajan", "kirtan", "tabla"] if silent else None)      # picture + music only: prefer a track the cuts can land on (must outweigh the contemplative profile's 0.09 penalty on rhythmic music; a 0.25 CLAP-similarity gap still wins)
            if wt is not None: track, why = wt, wwhy
            else: why = f"{why}; {wwhy}"
        analysis = analyze_track(track.path) if track else None
        beat = 60.0 / analysis.bpm if analysis and analysis.bpm else None
        transitions.plan_transitions(tl, pop, prof, beat, {x.id: x.info.duration for x in videos} if multi else D)
        sync = {}; offset = 0.0
        hook_cap = story.HOOK_MAX if (prof.opening == "hook" and tl.clips and tl.clips[0].role == "OPENING") else None
        if analysis:
            ovf = lambda t: [c.transition_in["durationSeconds"] if c.transition_in and c.transition_in["type"] in ("crossfade", "dip_black", "dip_white") else 0.0 for c in t.clips[1:]]
            sync = music_sync.align_cuts(tl, analysis, 0.0, ovf, hook_cap, prof.max_shot) if analysis.bpm else {}            # align against offset 0, then choose an offset ON a beat
            T = sum(c.length for c in tl.clips) - sum(ovf(tl)); climax = next((sum(c.length - (ovf(tl)[i] if i < len(ovf(tl)) else 0) for i, c in enumerate(tl.clips[:j])) + c.length / 2 for j, c in enumerate(tl.clips) if c.role == "CLIMAX"), None)
            offset, mscore = music_sync.pick_offset(analysis, music_sync.edit_energy(tl, pop, T, ovf(tl)), climax, T)
            if analysis.bpm:                                                                        # re-align to the beats as seen from the chosen offset
                sync = music_sync.align_cuts(tl, analysis, offset, ovf, hook_cap, prof.max_shot)
            sync["end"] = music_sync.align_end(tl, analysis, offset, ovf)                          # the reel finishes on a bar line, together with the music
            tl.decisions.append({"type": "music", "track": track.id, "kind": _kind(track), "bpm": analysis.bpm and round(analysis.bpm, 1), "offset": offset, "sync": {k: (round(v, 3) if isinstance(v, (int, float)) else v) for k, v in sync.items()}, "why": why})
        else:
            tl.decisions.append({"type": "music", "track": None, "why": why})
        segs = []; n_img = 0; probes = []
        for i, c in enumerate(tl.clips):
            S = by_id[c.shot.asset_id]
            if c.shot.kind == "image":
                d = {"assetId": S.id, "kind": "image", "start": 0.0, "end": round(c.length, 3), "reason": _reason(c), "score": round(c.shot.score, 4), "role": c.role, "shotId": c.shot.id,
                     "subjectX": round(c.shot.focus[0], 3), "subjectSpread": 0.0, "calm": True, "motion": motion_for(c.shot.size, c.shot.focus, 9 / 16, n_img)}
                n_img += 1
                if i > 0 and c.transition_in: d["transitionIn"] = {k: c.transition_in[k] for k in ("type", "durationSeconds", "reason")}
                segs.append(d); continue
            tr = track_subject_ex(S.path, c.start, c.end)
            spread = float(np.median([e for _, _, e in tr])) if tr else 0.0
            lay = layout.decide({"subjectSpread": spread}, S.info.width, S.info.height, 9 / 16)
            cam = camera_path(tr, lay["window"]); path = cam["path"]
            sl = c.shot.slots; e = pop.energy(sl); s0 = int((c.start - c.shot.start) / 0.5); calm = bool(len(e) and float(np.mean(e[s0:s0 + max(1, int(c.length / 0.5))])) < 0.35)
            d = {"start": round(c.start, 3), "end": round(c.end, 3), "reason": _reason(c), "score": round(c.shot.score, 4), "momentId": c.shot.moment_id, "role": c.role,
                 "shotId": c.shot.id, "subjectX": round(cam["subjectX"], 3), "subjectPath": [[round(t, 2), round(x, 3)] for t, x in path],
                 "subjectSpread": round(spread, 3), "calm": calm, "camera": {"mode": cam["mode"], "reason": cam["reason"]}, "subjectInFrame": round(cam["inFrame"], 3)}
            if i > 0 and c.transition_in: d["transitionIn"] = {k: c.transition_in[k] for k in ("type", "durationSeconds", "reason")}
            if i > 0 and c.audio_lead: d["audioLead"] = c.audio_lead
            if c.shot.beats: d["beat"] = max(c.shot.beats, key=c.shot.beats.get); d["beats"] = {k: round(v, 2) for k, v in sorted(c.shot.beats.items(), key=lambda kv: -kv[1])[:3]}      # what this clip IS in the story
            if multi: d["assetId"] = S.id; d["kind"] = "video"
            segs.append(d); probes.append((S.path, (c.start + c.end) / 2, not getattr(S.info, "hdr", False)))      # HDR footage is measured before tone-mapping: leave it alone
        if o.color_match and o.renderer == "ffmpeg":                  # only the ffmpeg renderer draws the grade
            grade.apply(segs, [(p, t, ok and sg.get("kind") != "image") for (p, t, ok), sg in zip(probes, segs)])
        creative = {"version": 2, "engine": "creative", "edgeSnap": bool(o.edge_snap), "motionDedupe": bool(o.motion_dedupe), "order": tl.order, "profile": prof.name, "arc": tl.arc, "decisions": tl.decisions,
                    "targetRange": [o.target_seconds, o.target_seconds] if o.target_seconds else [round(min(prof.target_min, 0.85 * avail_s), 1), prof.target_max], "outro": {"fadeSeconds": 0.8}, "storyScore": round(tl.score, 3)}
        used = [x for x in srcs if any(sg.get("assetId") == x.id for sg in segs)] if multi else []
        assets = [{"id": x.id, "kind": x.kind, "path": os.path.abspath(x.path), "durationSeconds": round(x.info.duration, 3) if x.kind == "video" else None,
                   "width": x.info.width if x.info else None, "height": x.info.height if x.info else None, "hasAudio": x.kind == "video"} for x in used]
        assets = [{k: v for k, v in a.items() if v is not None} for a in assets]
        plan = assemble_plan(ctx.video_id, os.path.abspath(ctx.src), D, segs, {"type": "cut", "durationSeconds": 0},
                             {x.id: x.transcript for x in videos if x.transcript} if multi else ctx.transcript, ctx.entities,
                             o.caption_lang, o.caption_mode, None, "9:16", "REEL", o.location, creative, assets or None, o.title, o.subtitle)
        plan["captions"]["cues"] = retime_cues(plan["captions"]["cues"], end=plan["durationSeconds"]); plan["captions"]["enabled"] = bool(plan["captions"]["cues"])
        if silent: plan["audio"]["preserveOriginal"] = False; plan["audio"]["sourceAudio"] = o.source_audio
        if o.source_audio == "bed":                                                        # the most music-like stretch over ALL the recordings, from the file it lives in
            pick = bed.pick_across([(x.audio, x.path) for x in videos if x.kind == "video"] if multi else [(ctx_audio, ctx.src)], plan["durationSeconds"] + 1.0)
            if pick: plan["audio"]["bed"] = {"start": pick[0], "score": pick[1], "why": pick[2], "path": os.path.abspath(pick[3])}
        if not o.captions:
            plan["captions"]["cues"] = []; plan["captions"]["enabled"] = False
        d = next((x for x in tl.decisions if x["type"] == "director"), None)
        plan["planner"] = {"mode": "llm-director", "model": d["model"]} if d else ({"mode": "deterministic-fallback", "error": next(x["error"] for x in tl.decisions if x["type"] == "director-fallback")}
                                                                              if any(x["type"] == "director-fallback" for x in tl.decisions) else {"mode": "deterministic"})
        if d and d.get("title") and plan["overlays"].get("template") == "divine_moment": plan["overlays"]["ritual"] = d["title"]
        plan["creative"]["outro"]["fadeSeconds"] = 0.8 if plan["durationSeconds"] >= 15 else 0.4
        if track:
            plan["audio"]["music"] = M.music_block(track, why, o.music_volume); plan["audio"]["music"]["offset"] = offset
            if analysis and analysis.bpm: plan["audio"]["music"]["bpm"] = round(analysis.bpm, 1)
        else:
            plan["audio"]["music"] = {"enabled": False, "reason": why}
        if o.festival_id: plan["source"]["festivalId"] = plan["source"].get("festivalId") or o.festival_id
        validate_plan(plan, D)
        plans.append(plan)
    return plans


# ------------------------------------------------------------------ mix + render + QC + re-edit
def mix_for(plan: dict, src: str, audio: Optional[AudioProfile], ceiling_db: float = mixer.CEILING_DB, voiceover: bool = False, live_on: bool = True, bed_path: Optional[str] = None):
    m = plan["audio"].get("music", {}); path = M.get_track(m["trackId"]).path if m.get("enabled") else None
    if bed_path:                                                                   # the source's own audio as one continuous track (--source-audio bed)
        path, m = bed_path, {"offset": 0.0}
    mix = mixer.render_audio(plan, src, path, float(m.get("offset", 0.0)), presence_fn(audio), float(m.get("volume", 0.5)), ceiling_db=ceiling_db, live_on=live_on)
    if voiceover and plan["captions"].get("cues"):
        from . import narration
        mix = narration.apply_voiceover(mix, plan["captions"]["cues"], plan["durationSeconds"], ceiling_db)
    return mix


def render_reel(plan: dict, src: str, out_dir: str, o: Options, audio: Optional[AudioProfile], formats: List[str], name: str = "reel", render_fn=None) -> Tuple[Dict[str, str], dict, dict]:
    """(files, qc report dict, final plan). QC runs on the primary (first) format; failing checks trigger up to o.qc_attempts targeted re-edits."""
    from ..render import FORMATS, render_format
    render_fn = render_fn or render_format
    primary = formats[0]; ceiling = mixer.CEILING_DB; report = None; history = []
    for attempt in range(max(0, o.qc_attempts) + 1):
        for s in plan["segments"]: s.pop("liveGainDb", None)                       # level matching is recomputed for the (possibly re-edited) clip set
        bp = bed.extract(plan["audio"]["bed"].get("path", src), plan["audio"]["bed"]["start"], plan["durationSeconds"] + 1.0, os.path.join(out_dir, "source_bed.wav")) if plan["audio"].get("sourceAudio") == "bed" and plan["audio"].get("bed") else None
        mix = mix_for(plan, src, audio, ceiling, voiceover=o.voiceover, live_on=not o.silent_source, bed_path=bp)
        path = os.path.join(out_dir, FORMATS[primary]["file"])
        render_fn({**plan, "outputFormat": FORMATS[primary]["outputFormat"], "aspectRatio": FORMATS[primary]["aspect"]}, src, path, primary, out_dir, mix=mix)
        report = QC.run_qc(plan, path, out_dir, mix, primary) if o.qc else None
        history.append({"attempt": attempt, "status": report.status if report else "skipped", "failed": [c.name for c in report.failed()] if report else []})
        if not report or report.status != "fail" or attempt >= o.qc_attempts:
            break
        new, fixes, remix = replan.apply_fixes(plan, report, retimer=lambda cues: retime_cues(cues, end=plan["durationSeconds"]))
        if remix.get("retarget"): ceiling -= 0.7
        if new is None and not remix:
            break
        if new is not None: plan = new
        history[-1]["fixes"] = fixes
    files = {primary: path}
    for f in formats[1:]:
        files[f] = render_fn({**plan, "outputFormat": FORMATS[f]["outputFormat"], "aspectRatio": FORMATS[f]["aspect"]}, src, os.path.join(out_dir, FORMATS[f]["file"]), f, out_dir, mix=mix)
    qcd = report.to_dict() if report else {"status": "skipped", "checks": []}
    qcd["attempts"] = history
    plan.setdefault("creative", {}).setdefault("qc", {}).update({"status": qcd["status"], "attempts": len(history)})
    return files, qcd, plan
