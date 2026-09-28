# Aikyam Video Intelligence: handoff for the next agent

> **Read `CLAUDE.md` (repo root) first: it is the entry point and has the reading order, the user's rules and the GPU-laptop plan. This file is the build history and file map. Sections 0-11 are older; §12 (2026-09-25) is the current update, and where they conflict §12 and `CLAUDE.md` win.**

Written 2026-09-21. Read this first; then `README.md`, `docs/ARCHITECTURE.md`, `docs/CREATIVE_RESEARCH.md`, `docs/OSS_AUDIT.md`. The repo is **not** under git (no version control here): back up before large refactors.

## 0. Ground rules the user gave (SUPERSEDED in part by CLAUDE.md §3: the user later asked for downloads and for tests on many videos)

1. **Test only on the user's own footage:** the four Pexels videos + the chant mp3 in the repo root (see section 8). Do NOT build other test footage or run bulk/other videos ("focus on one/four, not bunches"). The user reminded us of this twice. Unit tests with tiny generated media are fine for plumbing, but evaluations/A-B renders use the four clips only. `samples/ganga-aarti-haridwar.webm` is the only other kept real clip (the user asked earlier to keep just that one; it is no longer the focus).
2. **Do not download videos/audio unless asked.** When you need footage, tell the user where to get it.
3. **The user judges by real rendered reels, not test counts.** They said the reels were "not even 10% done": earlier status messages equated "code + tests pass" with "done". Measure output quality, report failures plainly, do not oversell.
4. **Stop when told "stop"**; the machine has ~5.7 GB RAM: run ONE heavy job at a time (whisper + SigLIP + CLAP + ffmpeg 4K + chromium). Background jobs: use `nohup ... &` and poll; never chain `sleep`.
5. Licences: reuse only permissive code (Apache-2.0/MIT/Unlicense). GPL/AGPL, non-commercial, or unlicensed repos are analysis-only (see `docs/OSS_AUDIT.md`). Music must be licence-gated (`music.py`); the user's own file needs `--i-own-the-music-rights`.
6. No Anthropic credentials exist on this machine: the LLM director has only been tested with fakes.

## 1. What the product is

A pipeline that turns long temple recordings (plus extra clips, photos, optional music) into a polished 15-60 s vertical Reel:

`Media Intelligence -> Moment Engine -> Story -> Creative Editing -> Audio -> EditPlan -> FFmpeg -> QC`

Single Python package `aikyam_video/` (3.9). Stages read/write artifacts in one directory so the same code runs in-process (CLI/UI) or as Kafka workers. Renderer-independent **EditPlan JSON** (`data/edit_plan.schema.json`, schema v1 and v3) is the contract; `plan.timeline / edge_overlaps / output_duration` are the single source of truth for timing (validation, captions, renderers, mixer, QC).

## 2. Environment (facts that bit us)

- Python venv `.venv` (3.9; pinned deps in `pyproject.toml`). `.venv-beat` = separate Python 3.11 env with **Beat This!** (needs 3.10+ syntax); called through a subprocess (`creative/beats.py`, `beat_this_worker.py`). Override with `AIKYAM_BEAT_THIS_PYTHON`.
- `docker` is a Podman shim (`systemctl --user start podman.socket`; `docker build --load`; MinIO images are on quay.io). The Docker image is several revisions BEHIND the code (no `music/`, no `.venv-beat`, no diffusion setup).
- CPU torch only; pip needs `TMPDIR` under /home (tmpfs `/tmp` is small).
- **Chromium (diffusion renderer) needs its profile on a real disk** (`renderer/diffusion/.profile`); with the default `/tmp` profile `fetch().blob()` of files >~14 MB fails and the picture renders as a placeholder colour. Already handled in `render.mjs`.
- Only HDR clip: `15343321_1920_1080_30fps.mp4` (HLG, BT.2020). FFmpeg path tone-maps it (`render.TONEMAP`, npl=150, tuned to match Chromium).
- 3 of the 4 user videos have NO audio; `--music-file` implies `--allow-silent`. Moments of silent clips are otherwise rejected (`no_audio`).
- `regex` (grapheme clusters) is a declared dependency; `/usr/bin/time` does not exist.

## 3. Run it

```bash
source .venv/bin/activate
# one video
python -m aikyam_video.cli process VIDEO.mp4 -o results/x
# several videos/photos + your music (mixed inputs; ids are printed: v1.. videos, i1.. images)
python -m aikyam_video.cli process a.mp4 b.mp4 c.mp4 d.mp4 --music-file song.mp3 --i-own-the-music-rights \
    --format reel --title "Ganga Aarti" [--order v3,v1,v2] [--opening establish] [--hook-first] \
    [--no-edge-snap] [--no-motion-dedupe] [--renderer ffmpeg|diffusion|remotion] [--planner llm] -o results/x
# the small web UI (upload clips + music, arrange order, make the reel): http://127.0.0.1:8090
python -m aikyam_video.ui            # data under results/ui (AIKYAM_UI_DIR)
# evaluate reels / OLD-vs-NEW on the same analysis
python tools/eval_reel.py results/runA results/runB --tag t
python tools/ab_run.py --name X in1.mp4 in2.mp4 [--music-file ... --i-own-the-music-rights] [--seed-assets DIR] [--variants new,old]
# tests (skip Kafka workers and Remotion unless you have them)
python -m pytest -q --ignore=tests/test_workers.py --ignore=tests/test_remotion.py     # last full run: 237 passed (before the UI + later tweaks)
```
Outputs per run dir: `reel-9x16.mp4` (+ `square-1x1.mp4`, `landscape-16x9.mp4` unless `--format reel`), `edit-plan.json` (every creative decision under `creative.decisions`), `qc.json`, `overlay_reel.ass`, `mix.wav`, `thumbnail.jpg`, `assets/<id>/` (per-video analysis cache, keyed by a stamp in `source.json`).

## 4. File map

### 4.1 `aikyam_video/` (pipeline)
| File | Role |
|---|---|
| `cli.py` | argparse CLI (`process`, `analyze`, `highlights`, `plan`, `render`, `worker`, `live`, `serve`). Several inputs or an image -> `pipeline.run_multi` |
| `pipeline.py` | in-process orchestration (`run`, `run_multi`), input validation, cost/metrics |
| `stages.py` | stage functions on artifacts: `transcription`, `scene_analysis`, `highlights`, `plan`, `render`, `_render_creative` (engine + QC + re-edit) |
| `multi.py` | mixed inputs: classify video/image/audio, per-video analysis in `assets/vN` (cached), image shots, pooled planning (`ingest`, `plan`) |
| `options.py` | `Options` dataclass (all knobs; see CLI flags) |
| `models.py` | pydantic models (Moment, SceneVision, Transcript, EntityRef...) |
| `plan.py` | EditPlan validation (`validate_plan`, v3 assets/images), `timeline`, `edge_overlaps`, `assemble_plan` (captions, overlays, automatic title), `seg_source` |
| `render.py` | FFmpeg renderer (crop/pan/track/fit_blur, Ken Burns `_ken_burns`, xfade/dip per edge, HDR tone-map, ASS overlay, logo, outro fade, muxes the finished mix). `build_ass` (captions, template lines, opening title) |
| `render_diffusion.py` + `renderer/diffusion/` | OPTIONAL third renderer: EditPlan -> diffusionstudio composition (JSX) -> headless Chromium picture -> FFmpeg finishing (ass, logo, our audio). ~1.4x slower than FFmpeg |
| `render_remotion.py` + `renderer/remotion/` | Remotion renderer (works, ~15x slower, no per-edge transitions/layouts; licence unresolved for a company) |
| `captions.py` | cue building (dict of transcripts per asset), retiming (grapheme-aware), line fit per language/safe zone (`fit_chars`) |
| `safezone.py` | platform UI zones (top 12 %, bottom 22 %, sides 8 % on 9:16), `glyphs()` (grapheme clusters) |
| `reframe.py` | subject tracking (`track_subject_ex`), smoothing, `camera_path` (stationary / pan / track), `path_expr` for ffmpeg crop |
| `evaluation.py` | objective metrics on rendered reels (hook, shot lengths, redundancy, blend %, cut-to-beat, LUFS, margin, subject-in-frame, cutoffs, safe zones, QC, edge stats) + `compare()` table |
| `ui.py` | the small web UI (FastAPI, single HTML page, one reel at a time) |
| `vision.py`, `audio.py`, `analysis.py`, `scenes.py`, `transcribe.py`, `entities.py`, `kg.py`, `highlights.py`, `scoring.py`, `thumbnails.py` | Media Intelligence + Moment Engine (SigLIP, CLAP, PySceneDetect, faster-whisper, entity resolver + KG, moment ranking, validation) |
| `music.py` | licence-gated music library, matching, `register_user_track` (user music with rights attestation) |
| `planner_llm.py` | LLM planner: classic proposal + **director mode** for the creative engine (`make_director`, `DIRECTOR_SCHEMA`); `AnthropicLLM` with plain-JSON fallback |
| `ffmpeg.py`, `measure.py` | ffmpeg/ffprobe helpers (`probe`, `has_audio_stream`, HDR flag), loudness (BS.1770), frame series |
| `api.py`, `jobs.py`, `bus.py`, `workers.py`, `events.py`, `storage.py`, `publish.py`, `live.py`, `library.py`, `cost.py`, `metrics.py`, `providers.py` | FastAPI service, Postgres job state machine, Kafka/Memory bus and 5 worker roles, S3/local storage, QC-gated publish, live chunking, personalization/library, cost + Prometheus, provider registry |
| `beat_this_worker.py` | tiny script run inside `.venv-beat` |

### 4.2 `aikyam_video/creative/` (Creative Editing Engine)
| File | Role |
|---|---|
| `engine.py` | `Ctx`/`Source`, `build_plans` (shots -> story -> transitions -> music -> segments -> `assemble_plan`), `render_reel` (mix -> render -> QC -> deterministic re-edit), `choose_music`, `_make_edger` |
| `shots.py` | shot slot features (0.5 s), `Population` percentile ranks, `best_window`, `image_shot`, `partition_moments` (disjoint windows), `carve` |
| `story.py` | beam-search story over roles (OPENING..CLOSING), duplicate rule (`is_duplicate`: SigLIP + motion signature), edge snapping hook, `MIN_REEL_S=15` top-up, LLM director validation, forced order (`--order`), opt-in `hook_first` |
| `roles.py`, `pacing.py` | role affinities (policy in `data/roles.json`), pacing profiles (`data/pacing.json`: contemplative / devotional / festive) |
| `transitions.py` | per-edge cut / crossfade / dip decisions, J/L cuts |
| `mixer.py` | numpy audio engine: level match, ducking, BS.1770 loudness (-16 LUFS), look-ahead true-peak limiter (-1.5 dBTP), returns stems |
| `musicdna.py`, `beats.py`, `music_sync.py` | music analysis (autocorr tempo + Beat This! grid/downbeats), offset choice, **bar-level DP cut alignment** (`align_cuts`, greedy kept as `align_cuts_greedy`) |
| `edges.py` | adapted from hypecut/auto-editor (see `NOTICE`): shot-boundary detection, pause runs, mask smoothing, motion signature, `snap_window` |
| `layout.py`, `stills.py` | crop vs fit_blur decision; subject-aware Ken Burns motion for stills |
| `qc.py`, `replan.py` | post-render QC (duration, black/frozen, crop, subject-in-frame, loudness, music balance, transitions, duplicates+weak, captions/safe zones) and deterministic fixes |
| `model.py` | dataclasses: Shot, Clip, Timeline, SlotFeatures |

### 4.3 Data, docs, tools, infra
- `aikyam_video/data/`: `edit_plan.schema.json`, `roles.json`, `pacing.json`, `templates.json` (overlay layouts incl. `yPortrait`), `languages.json` (font, `em`, chars per line), `kg_seed.json`.
- `docs/`: `ARCHITECTURE.md` (NOT updated for the later work), `CREATIVE_AUDIT.md`, `CREATIVE_RESEARCH.md` (research + diffusionstudio addendum), `OSS_AUDIT.md` (open-source audit + before/after tables), this file. `README.md` is also stale for this session's features.
- `tools/`: `eval_reel.py`, `ab_run.py`, `llm_smoke.py` (real Claude director smoke test; needs `ANTHROPIC_API_KEY`), `eval_edit.py` (older harness), `replay.py`, `make_music.py`, `make_logo.py`, `eval_vision.py`, `tune_clip.py`.
- `infrastructure/`: `docker/`, `helm/`, `kubernetes/`. `docker-compose*.yml`, `Makefile` in root.
- `music/`: 4 generated tracks (`AIKYAM-OWNED`) + `library.json`. `*.analysis.json` / `*.beats.json` are caches next to audio files.
- `research/repos/` (git-ignored): cloned third-party repos for LOCAL analysis only (hypecut, auto-editor, mediapipe AutoFlip subtree, SynthCut, ReelForgeAI, openreelio, ai-video-editor, tiktok_editor, vasy, ClipForge, montage-ai). Never copy from SynthCut / ReelForgeAI / montage-ai.
- `results/`: outputs. Useful: `ab_four_*`, `four_default|four_old|four_hook` (+ `four_only.json`), `ui/` (UI uploads, jobs, outputs), `ab_ganga*`, `ab_multicut*` (older; multicut was a synthetic source built from the user's clips: `samples/derived/multicut.mp4`), `eval_*.json`.
- `tests/`: 29 files. Key: `test_multi_asset.py` (v3 plan + Ken Burns real FFmpeg), `test_multi_engine.py` (mixed engine), `test_edges.py`, `test_bar_alignment.py`, `test_opening_order.py`, `test_safezones.py`, `test_camera.py`, `test_evaluation.py`, `test_diffusion.py`, `test_llm_director.py`, `test_ui.py`, `test_music.py`, `test_creative_*.py`, `test_units.py`, `test_features.py`, `test_pipeline.py`, `test_workers.py` (needs Kafka), `test_remotion.py` (needs npm install), `test_real_footage.py` (Ganga only).

## 5. What has been built (chronological)

1. **Base platform** (first brief): analysis pipeline, moment ranking, EditPlan, FFmpeg + Remotion rendering, captions kn/hi/en, reframing, thumbnails, audio + licence-gated music, S3 / Kafka / Postgres, API, CLI, K8s/Helm, metrics, security, live mode, golden test. Verified end to end (synthetic fixture, real clip, Kafka/Postgres/MinIO stack, Docker image at that time).
2. **Creative Editing Engine v2**: story-aware selection (beam search over roles), pacing profiles, per-edge transitions with J/L cuts, audio engine (ducking, LUFS, limiter), music intelligence, subject-aware layout, QC + deterministic re-edit, multi-reel.
3. **Research** (`CREATIVE_RESEARCH.md`): 10 topics, comparison tables, licence findings (Beat This! MIT adopted; madmom / Vidi non-commercial rejected; Remotion licence flagged).
4. **P0/P1 upgrade list**: EditPlan v3 (assets by `assetId`, images with Ken Burns), mixed inputs, safe zones + grapheme-aware captions/QC, camera modes (stationary/pan/track) + subject-in-frame QC, Beat This! (separate env), bar-level DP cut alignment, user music (`--music-file` with rights attestation), silent-video support, HDR tone-mapping, LLM director (fake-tested), `--order`, `--title`, opt-in `--hook-first`.
5. **Third renderer** `--renderer diffusion` (diffusionstudio/editor, MPL-2.0, pinned commit 0add88d, vendored unmodified under `renderer/diffusion/vendor`, set up by `setup.sh` then `node build.mjs`).
6. **Evaluation harness** (`evaluation.py`, `tools/eval_reel.py`, `tools/ab_run.py`).
7. **OSS audit** (`OSS_AUDIT.md`): adapted hypecut + auto-editor algorithms (edge snapping to cuts/pauses, motion-signature duplicate rule, mask smoothing) -> `creative/edges.py`, `NOTICE`.
8. **Follow-up round**: disjoint windows from overlapping moments (split only >22 s), QC duplicate check made motion-aware, `MIN_REEL_S=15` top-up, QC duration target relative to footage, opening titles.
9. **Web UI** (`ui.py`): upload clips/photos + music, drag/arrow ordering, "use my order", title/subtitle, progress, in-page player + download.

## 6. Measured results that matter (user's four clips + chant, `results/four_only.json`)

| | default (shipped) | old (`--no-edge-snap --no-motion-dedupe`) | `--hook-first` |
|---|---|---|---|
| length / clips | 29.8 s / 4 | 29.8 s / 4 (identical edit) | 20.6 s / 3 |
| hook score | 0.84 | 0.84 | **0.38** |
| cuts on beat | 100 % (7 ms) | 100 % | 100 % |
| QC | warn (subject_in_frame) | warn | warn |
| plan time | 55 s | 42 s | 54 s |

Take-aways: on these clips the newer edge/dup logic changes nothing (single continuous takes, no cuts, no sound); `--hook-first` is worse, so it stays opt-in; the Ganga single-scene case went 10 s -> 18 s and QC pass with motion-aware duplicates (see `OSS_AUDIT.md` sections 4-5).

## 7. Known weaknesses / open items (be honest about these)

1. **Reel quality is still basic:** 3-4 crossfaded clips with music; no captions on silent footage (nothing to caption); the hook score is my own sharpness-based metric, never validated against human judgement; "hook-first" enforcement was tried and measurably worse.
2. Edge snapping / pause trimming were proven only by tests and one 1-frame fix on synthetic multi-cut footage; the user's clips have no cuts or sound to test them.
3. Automatic **title** ("Aarti") comes from the existing ritual detector and can be wrong for some clips.
4. **LLM director never called for real** (no API key). Run `python tools/llm_smoke.py` with `ANTHROPIC_API_KEY`; the request shape (beta headers, structured output) is unverified against the live API (a plain-JSON fallback exists).
5. **Nobody has listened to the audio.** Ducking margins (8-16 dB) were never tuned by ear; the planned "listening kit" (variants at 10/12/14/16 dB + scoring sheet) is not built. `music_margin_db` is n/a for the user's reels because only one clip has (quiet) sound.
6. P1.11 **PySceneDetect AdaptiveDetector vs ContentDetector** on handheld footage: not done. P2: open-vocabulary deity/priest detector (OWLv2 / Grounding DINO licence check), WhisperX on Kannada/Hindi (no data), Remotion parity (blocked on the company-licence question), parallax: not done. There is no deity localiser (deity cutoffs are "n/a" in the harness).
7. **Docker image is stale**; `README.md` and `docs/ARCHITECTURE.md` do not describe: v3 plan, mixed inputs, safe zones, camera modes, Beat This!, diffusion renderer, evaluation harness, OSS-derived edges, UI. Update them.
8. The UI's end-to-end run with real clips was not verified by me (my test collided with the user's live session; the user's own run produced a 29.8 s reel). Run it in an isolated `AIKYAM_UI_DIR` + port, not against the user's live server.
9. Assumed, unverified contracts: knowledge-graph service and feed webhook (`kg.py`, `publish.py`); Kafka tests need Redpanda (`KAFKA_BOOTSTRAP`).
10. Full test suite was last run green at 237 passed before the final tweaks; a re-run was interrupted. Re-run it (about 13 min, one job at a time).

## 8. The user's footage (do not delete; do not add others) -- STALE, see 2026-09-22 update in §11

Repo root: `15318015_2160_3840_60fps.mp4` (portrait 4K, no audio), `15325338_2160_3840_30fps.mp4` (portrait 4K, no audio), `15343321_1920_1080_30fps.mp4` (landscape 1080p, HDR HLG, no audio), `20143614-uhd_3840_2160_24fps.mp4` (landscape 4K, quiet audio), `alex-morgan-ancient-spirit-echoes-om-chanting-548648.mp3` (Pixabay chant, 65 BPM by Beat This!; the built-in tracker's 129 BPM was an octave error). The user has also uploaded newer files through the UI (`results/ui/uploads/`, e.g. `15075169_2160_3840_30fps.mp4` and a bhajan mp3).

**None of this is on disk any more** -- superseded by §10 (deleted at the user's request the same day this was written) and then by §11: as of 2026-09-22 the only real footage in the repo is `inputs/diwali-mumbai.webm` (§11). Do not treat this section as current; it's kept for history only.

## 9. Suggested next steps (highest value first)

1. Let the user try the UI on their clips and collect what they dislike; iterate on the story/hook using THEIR feedback (a human judge is the missing metric).
2. Re-run the full suite; refresh `README.md`, `docs/ARCHITECTURE.md`; rebuild the Docker image (add `music/`, `.venv-beat`, diffusion setup if wanted).
3. Real Claude director test (`tools/llm_smoke.py`), then A/B against the deterministic story on the four clips.
4. Audio listening kit + tune ducking; PySceneDetect detector comparison on the four clips.
5. If the user supplies multi-cut / speech footage, re-measure edge snapping, pause trimming and captions there.

## 10. Update 2026-09-21 (later): start/end integrity, neural boundaries, self-editing loop

State of the repo: the user's clips, `results/`, `samples/*.webm` and `research/repos` were DELETED at their request (repo cleanup); they will supply new footage/music from the internet. `music/` (generated tracks) and `tests/fixtures/sample-temple.mp4` were kept. There is no Aikyam app repo on this machine (the original brief's "inspect the existing repository" was removed from `ARCHITECTURE.md`); the platform is standalone.

Added (all unit-tested on generated media only; NOT yet judged on real reels):
1. **`creative/cutpoints.py`**: every clip in/out point chosen from real candidates (hard cuts, speech-phrase edges, pauses, motion onsets/lulls) with a mid-speech penalty; the FIRST clip must open on an onset/cut/phrase, the LAST must end on a lull/phrase end. Plugged into `story.plan_story` through `engine._make_edger` (which replaced `edges.snap_window` there; `snap_window` is kept for ablation). Phrase ends and cuts lock the out point against the beat aligner.
2. **`music_sync.align_end`**: the reel ends on a bar line (downbeat, else beat) unless the out point is locked.
3. **QC `opening_strong` / `ending_resolved`** (`qc.check_start_end`): warnings only (a re-edit is not triggered by them; the variants loop penalises them).
4. **`creative/mlx.py` + `ml_worker.py`**: TransNetV2 (cuts) and Silero VAD (speech) in `.venv-beat`, cached as `<file>.ml.json`, used by `build_shots` and as phrase source when there is no transcript; fall back to the heuristics (`AIKYAM_NO_ML=1` disables). Verified on synthetic cuts (exact) and espeak speech only.
5. **`creative/autoedit.py` + `--variants N`**: N different edits (default / establish / hook / contemplative / festive; never overriding explicit `--opening/--pacing`), each rendered + QC'd + measured, the best by `reel_score` copied to the run dir; table in `variants.json`. The score is a hand-weighted sum, not a taste model.
6. Earlier this session: `audio_clean.denoise` + `tools/clean_voice.py` (local RNNoise, speech only), `creative/verify.py` + `tools/verify_reel.py` (second-ASR check of the rendered reel), adapted from claude-youtube-editor (MIT). `data/rnnoise/sh.rnnn` licence unverified (see NOTICE).

## 11. Update 2026-09-22: research-loop protocol adopted; `story.py` chronology + role-skip fixes (EXP-001/002/003); one new real clip

The user gave a research-engineering protocol (root `new.md`: baseline-first, measure before building, log every experiment, no cargo-cult additions). Followed for this session's work; full detail with numbers in `docs/research/experiment_matrix.md` -- read that before touching `creative/story.py`'s beam search again.

**New footage:** `inputs/diwali-mumbai.webm` (Wikimedia Commons, CC BY 3.0, credited in `inputs/CREDITS.txt`; a real ~343 s Diwali-mumbai recording, has its own audio). This is now the only real clip in the repo -- §8 above is stale. Renders in `results/diwali*`.

**Found and fixed in `creative/story.py` / `data/roles.json` (all validated: `tests/test_creative_story.py`, `test_target_length.py`, `test_opening_order.py`, `test_camera.py`, `test_multi_engine.py`, `test_features.py`, `test_safezones.py`, `test_transitions.py`, `test_bar_alignment.py`, `test_evaluation.py` -- 89/90 pass, see below for the one exception):

1. **Story order ignored source chronology** (EXP-001): the beam search's existing `chronology_bonus` term was asymmetric (bonus-only) and divided by `n_t`, ~6x too weak to ever compete with affinity/quality -- so on a single continuous take (Diwali), 3 of 5 role transitions ran backward in source time (e.g. the CLOSING shot was chronologically *before* the CLIMAX it followed). Fixed: symmetric penalty (reward forward, penalize backward equally), weight raised 0.12 -> 0.3. Diwali: 3/5 -> 0/5 inversions, held through QC's re-edit pass.
2. **That fix regressed multi-asset reels** (found only because `test_multi_engine.py` was finally run -- it wasn't part of EXP-001's original "25 passed" validation, a real gap in that report). `chronology_bonus=0.3` is large enough to make the search hoard the same video asset over switching to a different one or a still image (neither gets the bonus). Measured: a mixed 2-video+2-image pool dropped from 3 assets+1 Ken-Burns still to 2 assets+0 stills once the bonus crossed ~0.15 -- no single magnitude satisfies both single-asset ordering (needs >=0.20) and multi-asset diversity (needs <=0.12). Fixed by scoping, not tuning: the strong symmetric bonus (0.3) only applies when the pool has exactly one video asset; a multi-asset pool now uses a separately-pinned constant, `chronology_bonus_multi_asset=0.12`, with the byte-identical pre-EXP-001 formula. Both directions verified.
3. **Forcing every role (esp. REVEAL) to fill even with nothing good available** (EXP-002): added an absolute quality floor -- a candidate must beat `-skip_penalty*importance[role]` on its OWN merit (before chronology) or the role goes unfilled. Real mechanism, ships, but **does not change Diwali's actual output** -- checked directly, its REVEAL shot clears the absolute floor comfortably (it's the *relatively* weakest clip in that reel, not an *absolutely* bad one). A relative/story-average floor was tried (EXP-003) specifically to catch that case; it reintroduced the multi-asset regression and cascaded a different role into weakness on Diwali, so it was reverted. **Diwali's weak REVEAL shot is still open** -- three separate attempts today at extending this beam search's per-role gating each fixed one thing and broke another elsewhere; treat that as a real signal, not bad luck. A next attempt should score the whole sequence jointly rather than add a fourth one-off threshold gate, or just accept the QC `warn` this currently produces.

**One test now intentionally left red**, not silently patched: `tests/test_creative_story.py::test_story_follows_the_arc_and_serves_each_role_with_the_right_kind_of_shot` expects `REVEAL == "m3"` on a deliberately thin (exactly 6 shots for 6 roles) synthetic pool. Root cause fully isolated: the EXP-002 gate correctly finds a shorter, higher-quality 4-clip sequence, but a separate pre-existing filter (`n_lo <= len(clips) <= n_hi`, a nominal clip-count band) discards it as too short and falls back to a worse 6-clip one, which reassigns REVEAL to `m5`. Dropping `n_lo` fixes REVEAL but then CLIMAX goes unfilled instead (same class of cascade). Left as-is rather than force a third cascade; see EXP-002b/003 in the experiment log for the full trace before attempting another fix here.

No second real clip was available to validate #2 against (the four Pexels clips + Ganga sample are gone, per §8/§10) -- validation for the multi-asset case leaned on `test_multi_engine.py`'s synthetic-but-real-FFmpeg fixture instead. If the user supplies another real recording, re-run `docs/research/experiment_matrix.md`'s EXP-001/002 checks against it.

Not done / honest limits: no aesthetic scorer (the LAION head needs CLIP ViT-L embeddings, we compute SigLIP), no person detector (YOLO is AGPL; RT-DETR untested/RAM), the DP for in/out points is an exhaustive search over few candidates, and none of this has been compared on real footage: when new clips arrive, run `python -m aikyam_video.cli process ... --variants 3` and judge by eye; check the `decisions` in `edit-plan.json` for `edge` entries (kinds cut/phrase/pause/onset/lull).

## 12. Update 2026-09-25: what changed since §11 (details and numbers in `docs/research/experiment_matrix.md` EXP-012 to EXP-024)
- **Repo is under git now** (private, `github.com/gaganmitron/aikyam-auto-editor`); the "not under git" line at the top of this file is obsolete.
- **New modes/flags:** `--source-audio keep|off|bed` (`bed` = the recording's own most music-like stretch as ONE continuous soundtrack, picked across all inputs, `creative/bed.py`), `--no-transcript`, `--no-captions`, `--voiceover` (offline TTS, rejected by the user as poor, off by default), `--no-color-match`, `--no-beats`, `--experimental-selection`, `--vision-model`, `--opening establish|hook`.
- **Vision/story:** 10 story beats (`beats.py`) steer role fit and variety; graphic/text-overlay guard (`highlights.NO_TEXT_MIN`); dark-opening guard; colour match between clips (`creative/grade.py`); metadata-only titles (no invented deity/ritual names); long-video chunking (>30 min).
- **Footage quality:** dead slots (blocked lens, whip) never inside a clip (`shots.dead_slots`, `story.has_dead`), overlaps between same-take clips re-resolved after the top-up.
- **QC:** `weak_shots` now measures only the real picture rows over 3 frames; `role_label_consistency` needs unambiguous anchor scenes. Both were measuring themselves before (EXP-020).
- **Multi-video reels:** four Tirumala/Tirupati videos were combined into one reel (`results/tirumala_multi2/`, not in git). Lessons: check every download by eye (TV templates, ticket-site recordings, AI images, another city at the end of a "Tirumala" video), and crop/trim flawed sources into new files instead of using them raw.
- **Benchmarks added (`tools/`):** `bench_gate.py` (70 clip windows), `bench_subject.py` (22 frames), `bench_suite.py` (5-video regression + `docs/research/suite_baseline.json`), plus earlier `bench_labels/beats/models/vlm/clips/scoring`, `fit_ranker.py`.
- **Negative results, not shipped:** label calibration, learned ranker, small VLMs on CPU, image-model swap, tighter framing, subject locators (SigLIP crops, OWL-ViT) vs centre crop, clip-quality gate beyond dead footage.
- **Inputs:** the user had all non-Tirumala videos deleted on 2026-09-25 (Golden Temple, Janakpur, Kolkata); `inputs/` and `results/` are git-ignored, so a fresh clone has no footage: see `CLAUDE.md` §7C for how to re-create it.
- **Environment:** the next machine has a 6 GB GPU. The code only uses a GPU in Whisper; vision, CLAP, TransNetV2 and Beat This! must be moved to CUDA first (`CLAUDE.md` §7B).
- **Last full test run (CPU):** 333 passed, 5 skipped, 1 known-red test left red on purpose (`test_story_follows_the_arc_and_serves_each_role_with_the_right_kind_of_shot`); one further failure found in that run was fixed afterwards and the touched files re-tested, but the full suite was not repeated.
