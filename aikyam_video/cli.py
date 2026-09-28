"""aikyam-video CLI (section 24)."""
from __future__ import annotations
import argparse, os, sys
from . import pipeline


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="aikyam-video")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, h in [("analyze", "transcript + scenes + entities"), ("highlights", "analyze + ranked moments"),
                    ("plan", "highlights + edit-plan.json"), ("render", "plan + render one/all formats"),
                    ("process", "everything: all outputs")]:
        s = sub.add_parser(name, help=h)
        s.add_argument("input", nargs="+", help="a video, or several videos/images (mixed inputs: one reel)")
        s.add_argument("-o", "--out", default="output")
        s.add_argument("--format", choices=["reel", "square", "landscape"], help="render only this format")
        s.add_argument("--temple-id"); s.add_argument("--location")
        s.add_argument("--language", help="force source language code (default: detect)")
        s.add_argument("--translate", action="store_true", help="whisper translate to English")
        s.add_argument("--caption-lang"); s.add_argument("--caption-mode", choices=["sentence", "word"], default="sentence")
        s.add_argument("--no-captions", action="store_true", help="render with no caption overlay at all")
        s.add_argument("--source-audio", choices=["keep", "off", "bed"], default="keep", help="off: drop the recorded sound entirely; cuts follow the picture and the music, and music is always added. bed: use the recording's own most music-like stretch as ONE continuous soundtrack (e.g. Tirumala's devotional music), never cut per clip (creative engine)")
        s.add_argument("--no-transcript", action="store_true", help="skip speech-to-text entirely: no Whisper run, no transcript-derived entities or captions")
        s.add_argument("--experimental-selection", action="store_true", help="EXPERIMENTAL: also rank clips by look quality, the learned ranker and coverage of the whole recording (each has weak or synthetic evidence so far)")
        s.add_argument("--sequence-terms", action="store_true", help="EXPERIMENTAL (EXP-026): penalise awkward cuts when ordering clips (same size twice, big energy or brightness jumps, unrelated cross-video action); off by default")
        s.add_argument("--spread-sources", action="store_true", help="EXPERIMENTAL (E4b): avoid two clips in a row from the same video when a reel uses several videos; off by default")
        s.add_argument("--no-beats", action="store_true", help="do not use story beats (what each clip IS: establishing, ritual, procession, darshan...) when planning the story")
        s.add_argument("--no-color-match", action="store_true", help="do not colour-match the clips to each other (creative engine, ffmpeg renderer)")
        s.add_argument("--voiceover", action="store_true", help="read the captions aloud (offline espeak-ng TTS), ducked into the mix; creative engine only")
        s.add_argument("--target-seconds", type=float, default=None, help="reel length (default: chosen by the pacing profile, 25-40 s)")
        s.add_argument("--engine", choices=["creative", "classic"], default="creative", help="creative: story-aware editing engine; classic: greedy score-ordered planner")
        s.add_argument("--pacing", choices=["contemplative", "devotional", "festive"], help="force a pacing profile (default: from the footage)")
        s.add_argument("--reels", type=int, default=1, help="reels to cut from this one source (creative engine)")
        s.add_argument("--no-qc", action="store_true", help="skip post-render quality control")
        s.add_argument("--qc-attempts", type=int, default=2, help="deterministic re-edits when QC fails")
        s.add_argument("--whisper-model", help="tiny|base|small|medium|large-v3 (env WHISPER_MODEL)")
        s.add_argument("--vision", default="auto", help="auto|clip|heuristic")
        s.add_argument("--vision-model", help="open_clip model for --vision clip/auto, e.g. hf-hub:timm/ViT-B-16-SigLIP-256 (EXP-015: a small, unproven gain over the default)")
        s.add_argument("--scoring-config", help="JSON of scorer weights")
        s.add_argument("--festival-id", help="KG festival id, e.g. festival_9 (Dasara) -> festival template")
        s.add_argument("--music", choices=["auto", "off"], default="auto", help="auto: add a licensed, topic-matched track when it suits; off: original audio only")
        s.add_argument("--music-volume", type=float, default=0.5, help="music level 0.1-1.0 (before ducking under the original audio)")
        s.add_argument("--opening", choices=["hook", "establish"], help="how the reel opens: hook = strongest close shot first (default), establish = wide setting first")
        s.add_argument("--order", help="force the order by asset id, comma separated, e.g. v3,v1,i2 (ids are printed at analysis: v1.. videos, i1.. images in input order)")
        s.add_argument("--hook-first", action="store_true", help="always start with the strongest hook shot (opt-in: see docs/OSS_AUDIT.md results)")
        s.add_argument("--title", help="opening title text (first ~3.5 s, inside the platform safe zone)")
        s.add_argument("--subtitle", help="second line under the title")
        s.add_argument("--music-web", action="store_true", help="find music on the internet automatically (Openverse, CC0/public domain/CC BY only), matched to the reel by CLAP; ignored if --music-file/--music-track is given")
        s.add_argument("--variants", type=int, default=1, help="self-editing loop: make N different edits (opening/pacing), keep the best by measured score (see creative/autoedit.py); slower: N x plan+render")
        s.add_argument("--no-edge-snap", action="store_true", help="do not snap clip edges to real cuts/pauses (earlier behaviour)")
        s.add_argument("--no-motion-dedupe", action="store_true", help="appearance-only duplicate rule (earlier behaviour)")
        s.add_argument("--allow-silent", action="store_true", help="accept picture-only videos (implied by --music-file)")
        s.add_argument("--music-file", help="YOUR music file (mp3/wav/m4a/ogg/flac), used instead of the library")
        s.add_argument("--i-own-the-music-rights", action="store_true", help="required with --music-file: you own it or hold a licence for published Reels")
        s.add_argument("--music-track", help="force one track id from the licensed library ($MUSIC_LIBRARY_DIR/library.json)")
        s.add_argument("--transition", choices=["crossfade", "fade", "cut"], default="crossfade", help="how consecutive moments are joined")
        s.add_argument("--transition-seconds", type=float, default=0.5)
        s.add_argument("--renderer", choices=["ffmpeg", "remotion", "diffusion"], default="ffmpeg")
        s.add_argument("--planner", choices=["deterministic", "llm"], default="deterministic")
        s.add_argument("--audio-tagger", choices=["auto", "clap", "none"], default="auto")
    w = sub.add_parser("worker", help="event-driven worker (Kafka): one role, one job at a time")
    w.add_argument("--role", required=True, choices=["orchestrator", "intelligence", "highlight", "planner", "render"])
    w.add_argument("--metrics-port", type=int, default=9100)
    w.add_argument("--workdir", default=os.environ.get("WORK_DIR", "./work"))
    lv = sub.add_parser("live", help="rolling highlight detection on a live stream / growing source")
    lv.add_argument("source"); lv.add_argument("-o", "--out", default="live_out")
    lv.add_argument("--chunk", type=float, default=60.0); lv.add_argument("--overlap", type=float, default=10.0)
    lv.add_argument("--min-score", type=float, default=0.4); lv.add_argument("--no-render", action="store_true")
    lv.add_argument("--max-chunks", type=int)
    sv = sub.add_parser("serve", help="run the Media API")
    sv.add_argument("--host", default="0.0.0.0"); sv.add_argument("--port", type=int, default=8080)
    a = p.parse_args(argv)
    if a.cmd == "worker":
        from prometheus_client import start_http_server
        from . import bus as bus_mod, publish as publish_mod
        from .jobs import JobStore
        from .metrics import METRICS
        from .storage import from_env
        from .workers import Worker
        bus = bus_mod.from_env()
        if bus is None:
            print("error: set KAFKA_BOOTSTRAP", file=sys.stderr); return 1
        start_http_server(a.metrics_port, registry=METRICS.registry)
        pub, note = publish_mod.from_env()
        Worker(a.role, JobStore(), from_env(), bus, a.workdir, pub, note).run()
        return 0
    if a.cmd == "live":
        from .live import LiveConfig, run_live
        ms = run_live(a.source, a.out, pipeline.Options(), LiveConfig(a.chunk, a.overlap, a.min_score, not a.no_render, a.max_chunks),
                      on_moment=lambda g: print(f"[{g['start']:.0f}-{g['end']:.0f}s] {g['reason']} score={g['score']}"))
        print(f"{len(ms)} highlights -> {a.out}/live_highlights.jsonl"); return 0
    if a.cmd == "serve":
        import uvicorn
        uvicorn.run("aikyam_video.api:app", factory=True, host=a.host, port=a.port); return 0
    if a.vision_model: os.environ["VISION_MODEL"] = a.vision_model
    o = pipeline.Options(vision=a.vision, whisper_model=a.whisper_model, language=a.language, translate=a.translate,
                         caption_lang=a.caption_lang, caption_mode=a.caption_mode, captions=not a.no_captions, source_audio=a.source_audio, transcript=not a.no_transcript, color_match=not a.no_color_match, experimental_selection=a.experimental_selection, sequence_terms=a.sequence_terms, spread_sources=a.spread_sources, beats=not a.no_beats, voiceover=a.voiceover, target_seconds=a.target_seconds, engine=a.engine, pacing=a.pacing, reels=a.reels, qc=not a.no_qc, qc_attempts=a.qc_attempts,
                         scoring_config=a.scoring_config, temple_id=a.temple_id, location=a.location, festival_id=a.festival_id,
                         music_track=a.music_track, music=a.music, music_volume=a.music_volume, renderer=a.renderer, transition=a.transition, transition_seconds=a.transition_seconds, planner=a.planner, audio_tagger=a.audio_tagger,
                         formats=[a.format] if a.format else ["reel", "square", "landscape"])
    o.music_web = a.music_web; o.hook_first = a.hook_first; o.title, o.subtitle = a.title, a.subtitle; o.edge_snap = not a.no_edge_snap; o.motion_dedupe = not a.no_motion_dedupe
    o.opening = a.opening; o.order = [x.strip() for x in a.order.split(",") if x.strip()] if a.order else None
    o.allow_silent = a.allow_silent or bool(a.music_file) or a.source_audio in ("off", "bed")          # quiet footage is fine when its per-clip sound is never used
    if a.source_audio in ("off", "bed") and a.engine == "classic":
        print(f"error: --source-audio {a.source_audio} needs the creative engine", file=sys.stderr); return 1
    if a.source_audio == "off" and a.music == "off" and not a.music_file and not a.music_track:
        print("error: --source-audio off with --music off would render a silent reel", file=sys.stderr); return 1
    if a.music_file:
        from . import music as _m
        try:
            o.music_track = _m.register_user_track(a.music_file, a.out, a.i_own_the_music_rights)
        except (ValueError, FileNotFoundError) as e:
            print(f"error: {e}", file=sys.stderr); return 1
    upto = {"analyze": "analyze", "highlights": "highlights", "plan": "plan", "render": "render", "process": "process"}[a.cmd]
    try:
        from . import multi
        def go(oo, out):
            if len(a.input) > 1 or multi.classify(a.input[0]) != "video":
                return pipeline.run_multi(a.input, out, upto if upto in ("plan", "process") else "process", oo)
            return pipeline.run(a.input[0], out, upto, oo)
        if a.variants > 1 and upto == "process":
            from .creative import autoedit
            win, rows = autoedit.best_of(go, a.out, o, a.variants)
            for r in rows: print(f"  {r['variant']:14s} score {r['score']:.3f}  qc {r.get('qc')}  {r.get('error', '')}")
            print(f"winner: {win} (all variants: {a.out}/variants.json)"); files = {"reel": os.path.join(a.out, "reel-9x16.mp4"), "variants": os.path.join(a.out, "variants.json")}
        else:
            files = go(o, a.out)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr); return 1
    for k, v in files.items():
        print(f"{k:12s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
