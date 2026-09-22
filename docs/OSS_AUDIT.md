# Open-source audit for Aikyam (2026-09-20)

Method: every repository below was cloned to `research/repos/` (git-ignored, **local analysis only**) and the *source* was read, not only the READMEs. Nothing was copied from a repository whose licence does not permit it. Code that was adapted comes from Apache-2.0 / MIT / Unlicense repositories and is credited in `NOTICE` and in the module docstring.

## 1. Licence matrix

| Repository | Licence | Last push | Verdict for Aikyam |
|---|---|---|---|
| Yu-0312/hypecut | **Apache-2.0** | 2026-09 | Reuse OK (attribution + NOTICE). Best source of ideas: edge snapping, pause trimming, motion-signature de-duplication, evaluation |
| WyattBlue/auto-editor | **Unlicense** (public domain) | 2026-09 | Reuse OK, no obligations. Nim code: port the *algorithms* (mask smoothing, margins, motion analysis) |
| google-ai-edge/mediapipe (AutoFlip) | **Apache-2.0** | current | Reuse OK. C++/Bazel: port the algorithms only. Models used by its graph are separate downloads with their own licences |
| EverythingAI-Pro/ai-video-editor | **MIT** | 2026-06 | Reuse OK. Talking-head oriented (filler removal, transcript hook rubric); little applies to silent temple footage |
| openreelio/openreelio | **MIT** | 2026-09 | Reuse OK. Rust/TS desktop editor with agent CLI; timeline/edit-command model only, no rendering algorithms worth porting |
| opodark/tiktok_editor | **MIT** | 2026-09 | Reuse OK. Photo+video+song beat-synced vertical montage: closest to our task, mostly presets (colour grades, transition pools, Ken Burns) |
| manoj30075/vasy, Touka01/ClipForge | MIT | 2026 | Reuse OK; beat sync via librosa (weaker than our Beat This! tracker), nothing to port |
| Relo-video/SynthCut | **GPL-3.0-or-later** | 2026-07 | **Analysis only.** Copyleft: copying or linking would force Aikyam to GPL. Ideas only (ffmpeg `sendcmd` crop animation, YuNet face detector, keyframe-expression graph) |
| Rudeway-Aungee/ReelForgeAI | **No licence file** (all rights reserved by default) | 2026-06 | **Analysis only.** Cannot be reused or redistributed. Ideas only |
| mfahsold/montage-ai | **PolyForm Noncommercial 1.0.0** | 2026-03 | **Analysis only, non-commercial.** Not usable in a commercial product |

Restrictions to keep in mind for models/assets: AutoFlip's face/object detection graphs pull separate models (each has its own licence); SynthCut downloads a YuNet ONNX face model (OpenCV Zoo, MIT, but SynthCut itself is GPL); ReelForgeAI/tiktok_editor ship fonts and templates whose licences were not audited (not needed by us).

## 2. Components found (repository / file / function / algorithm / deps / licence / reuse / adaptation / performance / tests)

### A. Reused or adapted in this change

| # | Component | Where | Algorithm | Deps | Licence | Adaptation for Aikyam | Performance | Upstream tests |
|---|---|---|---|---|---|---|---|---|
| A1 | **Shot-boundary detection** | hypecut `src/hypecut/snapping.py`: `boundary_strength`, `find_boundaries` | Frame difference divided by its own running median (~4 s window), peak picking, absolute floor `min_diff` 1.5 luma levels so still footage does not "cut" on codec flicker, strongest peak per `min_gap` cluster | numpy | Apache-2.0 | Ported to `aikyam_video/creative/edges.py`; runs on 10 Hz gray frames decoded around each chosen window only | Milliseconds per clip (96x54 gray frames) | `tests/test_snapping.py` (25 tests) |
| A2 | **Frame-exact boundary refinement** | hypecut `snapping.py`: `refine_boundary` | Re-decode ±0.5 s at the source frame rate, take the largest frame difference, land on the frame *after* the change (one frame late is invisible; one early flashes the old shot) | ffmpeg | Apache-2.0 | Same function, using our `ff` helpers | A few dozen tiny frames per edge | same |
| A3 | **Edge snapping with guards** | hypecut `snapping.py`: `snap_segments` | In-point snaps to a cut *before/at* the event, out-point to a cut *after* it; edge may travel at least as far as the roll that placed it; a snap that would break the length budget is **rejected**, not clamped; never crosses a neighbour | numpy | Apache-2.0 | Re-expressed for our `Clip` (window inside a `Shot`), guards = shot bounds + protected core of the window + profile min/max length | negligible | same |
| A4 | **Pause trimming** | hypecut `src/hypecut/trimming.py`: `silence_mask`, `find_pause`, `trim_segments` | Silence threshold **relative to the clip's own speech level** (14 dB drop), pauses ≥ `min_silence`; in-point lands where sound resumes, out-point where it stops, minus/plus a pad; a real cut always wins over a pause | numpy | Apache-2.0 | Uses `AudioProfile.rms_db` (already computed) instead of re-decoding audio | negligible | `tests/test_trimming.py` (22 with similarity) |
| A5 | **Activity-mask smoothing** | auto-editor `src/lib/editutil.nim`: `smoothing(mincut, minclip)`, `mutMargin` | Boolean per-step mask; runs shorter than `minclip` are removed, gaps shorter than `mincut` are filled (iterated to a fixed point, 2-cycle guard); margins pad kept runs | none | Unlicense | Ported to Python for the pause mask (removes flicker of one-step "pauses") | O(n) | none upstream (integration only) |
| A6 | **Motion-signature repeat detection** | hypecut `src/hypecut/refine/similarity.py`: `descriptor`, `cosine`, `Similarity.refine` | Mean absolute frame-to-frame difference pooled to a 6x8 grid, zero-meaned and normalised: describes *where things move*, not what the venue looks like (appearance gave cosine > 0.99 for every clip of a locked camera). Absolute floor 0.1 luma levels so noise is not compared. Replay logic: similar + close in time = one event (kept), similar + far = a genuine repeat (penalised) | numpy | Apache-2.0 | Used to refine our embedding-based duplicate rule (planner AND the QC duplicate check): two shots that look identical (SigLIP ≥ 0.95 / same hash and histogram) are only duplicates if their motion signatures also match. The upstream *replay window* logic (similar + close in time = one event) was NOT adopted | negligible (reuses sampled frames) | `tests/test_similarity.py` |

### B. Studied, not adopted (with the reason)

| Component | Where | What it does | Why not now |
|---|---|---|---|
| Kinematic path solver | AutoFlip `quality/kinematic_path_solver.{h,cc,proto}` | Camera as a physical point: max velocity, dead-zone (`min_motion_to_reframe`), filtering window | Aikyam already has stationary / pan / polynomial-track camera modes with a speed limit (`reframe.camera_path`); measured wobble is already 8-25x lower than the raw path. Revisit if real footage shows pan jerk |
| Polynomial regression path solver | AutoFlip `quality/polynomial_regression_path_solver.{h,cc}` | Least-squares smooth crop path | Same idea already implemented (degree-3 fit) |
| Scene camera motion analyzer | AutoFlip `quality/scene_camera_motion_analyzer.{h,cc}` | Chooses stationary / pan / tracking per scene | Implemented in `camera_path` (same three modes) |
| Blur padding effect | AutoFlip `quality/padding_effect_generator.cc` | Fit + blurred background | Implemented as `fit_blur` layout |
| Mask smoothing / margin | auto-editor `editutil.nim` | see A5 | Adopted (A5) |
| Motion analysis with blur + rect | auto-editor `analyze/motion.nim` | Per-frame count of differing pixels after `gblur`, cached | Our `analyze_window` already measures motion per slot; SIMD pixel counting is a speed optimisation we do not need at 4 Hz |
| Silence/loudness edit methods | auto-editor `analyze/audio.nim`, `render/audio.nim` | dB threshold edit, audio filters | Our numpy mixer + BS.1770 loudness already covers what a reel needs |
| Virality by delivery | ai-video-editor `scripts/ve_virality.py` | Energy peak, dynamic range, laughter, pause-before-punch, transcript hook rubric | Speech-centric; temple footage is mostly chant/ambience. `pause_drama` idea noted for the hook selector |
| Filler-word cutting, taste.yaml | ai-video-editor `ve_cut.py`, `taste.yaml` | Talking-head tightening, one style file | Not applicable to devotional footage; taste.yaml pattern (one editable style file) matches our `data/*.json` policy files |
| Colour grades, transition pools, Ken Burns, chunked segment render | tiktok_editor `autoedit/effects.py`, `render.py` | Preset library, `_plan_pads` chunking of xfade renders | Flashy presets do not suit devotional content; our Ken Burns is subject-aware and tested; chunked rendering only matters for very long reels |
| Beat sync | vasy, ClipForge `clipforge/beat.py` | librosa beat_track with fixed-BPM fallback | Weaker than Beat This! which is already integrated |
| Agent CLI / MCP, timeline commands | openreelio `crates/openreelio-cli`, SynthCut `packages/mcp` | Editing as agent tool calls | Different product; our LLM director covers the same need |
| Crop animation via ffmpeg `sendcmd`, YuNet face detector | SynthCut `reframe/reframe.ts`, `detector.ts` | Keyframed crop, ONNX face model | **GPL-3.0: analysis only.** Our path-expression crop already works |
| Action/highlight selection from video motion + audio energy + speech | ReelForgeAI `action_selector.py` | Motion+audio+speech timeline, window scoring, "boring threshold" | **No licence: analysis only.** We already score windows this way |
| Smart crop with faces + people + motion boxes | ReelForgeAI `smart_crop.py` | Weighted box centre + smoothing | **No licence: analysis only.** Concept covered by `reframe.track_subject_ex` |
| Beat-synced montage engine, cluster mode | montage-ai | Large system | **Non-commercial: analysis only** |

## 3. Decision

Highest value for "multiple/long videos + images + optional music -> polished 15-60 s Reel", smallest change, all permissively licensed: **A1-A6**. They attack two measured defects of the current engine:

1. **Sliced edges.** Windows are chosen on a fixed 0.5 s grid and never look at real cuts or pauses, so a clip can open three frames into a continuous shot or stop in the middle of a sound.
2. **Same-venue footage collapses.** The duplicate rule (SigLIP similarity >= 0.95) treats every shot of a locked camera as the same footage, so a 28 s single-scene recording produced a 10 s, single-clip reel.

Interfaces are preserved: `plan_story`, `Clip`, `Shot`, the EditPlan schema and the renderers are unchanged; both features are switchable (`Options.edge_snap`, `Options.motion_dedupe`) so OLD behaviour stays reproducible for comparisons.

## 4. Results of the change (old vs new, real footage, same analysis, same machine)

Old = `--no-edge-snap --no-motion-dedupe`; new = defaults. `tools/ab_run.py` runs both from one analysis and evaluates with `aikyam_video/evaluation.py`. Results in `results/ab_*.json`.

| | Ganga aarti (28 s, one locked scene) old | new | Four Pexels clips + chant (single takes, mostly silent) old | new |
|---|---|---|---|---|
| Reel length | 10.0 s | **18.1 s** | 29.8 s | 29.8 s |
| Clips | 1 | **2** | 4 | 4 |
| Hook score (0-1) | 0.60 | **0.72** | 0.82 | 0.82 |
| Redundancy (max colour similarity) | n/a | 0.92 | 0.43 | 0.43 |
| Clip edges on a real cut / sliced / no cut nearby | 0 / 0 / 2 | 0 / 0 / 4 | 0 / 0 / 8 | 0 / 0 / 8 |
| Blend share, cut-to-beat, LUFS / true peak | same | same | same | same |
| QC | warn (duration) | warn (duration) | warn (subject in frame) | same |
| Automatic re-edits | 0 | 0 | 0 | 0 |
| Plan time | 5.5 s | 11.5 s | 44.5 s | 58.4 s |
| Render time | 20.7 s | 47.5 s (two clips) | 77.8 s | 98.1 s (identical output; load noise) |

What changed: **only the single-scene recording**, and only through the motion-signature duplicate rule (planner and QC). **Edge snapping changed nothing on either set because none of this footage has cuts or sound to land on** (every clip edge is "no cut nearby"); it is proven by the real-FFmpeg test (an edge lands within one frame of a real cut) but has not yet been demonstrated on real multi-cut footage. Reframing, crop smoothness, audio and captions are untouched by this change and identical.
Cost: +6-14 s planning per run (10 Hz cut/pause detection around each candidate window).


## 5. Follow-up round on the remaining weaknesses (final shipped defaults, same three real sets)

| | Ganga aarti (28 s, one locked scene) before -> after | Four Pexels clips + chant before -> after | Multi-cut recording (real pixels, hard cuts at 6.5 / 10.5 / 15.5 s) before -> after |
|---|---|---|---|
| Reel length / clips | 10.0 s / 1 -> **18.1 s / 2** | 29.8 s / 4 -> 29.8 s / 4 | 22.6 s / 3 -> 22.6 s / 3 |
| Hook score | 0.60 -> **0.75** | 0.82 -> 0.84 | 0.34 -> 0.34 |
| QC | warn -> **pass** (duration target now relative to the footage) | warn -> warn (subject in frame) | pass -> pass |
| Opening title | none -> "Aarti" | none -> "Aarti" | none -> "Aarti" |
| Edges on a real cut / sliced | 0/0 -> 0/0 | 0/0 -> 0/0 | 4/0 -> 4/0 |
| Redundancy (max) | n/a -> 0.92 (two windows of one repetitive scene) | 0.43 -> 0.43 | 0.40 -> 0.40 |
| Plan / render time | 5.5 / 20.7 s -> 10.9 / 57.7 s (two clips, cut/pause detection) | 58.4 / 98.1 -> 54.9 / 96.1 s | 11.8 / 39.7 -> 11.2 / 55.0 s |

What shipped by default: disjoint windows from overlapping moments (split only above 22 s), motion-aware duplicate rule in planner and QC, top-up to 15 s from unused footage, QC length target relative to the footage, opening titles (`--title`, automatic from known ritual/festival/deity/temple, nothing invented), edge snapping.

Tried and **not** shipped as default: *forced hook-first* (`--hook-first`, opt-in). Enforcing the strongest opener as a separate first clip shortened all three reels (Ganga 18.1 -> 8.0 s, four videos 29.8 -> 20.6 s) and on one set picked a visibly worse-scoring opener (hook score 0.82 -> 0.38), because it consumes footage and my added "energy" weight was unvalidated (reverted). The carve-out that returns the rest of the hook's shot to the pool is kept for the opt-in path.

Edge snapping on real multi-cut footage: it acted once (moved a start from 15.48 s to 15.51 s, removing a one-frame flash of the previous shot); otherwise the moment windows already coincided with the scenes. Small, real, not dramatic.
