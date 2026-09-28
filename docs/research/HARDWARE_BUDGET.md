# Hardware budget: what runs where, and how to add a VLM without breaking modest machines (2026-09-26)

Brief §20–21: a VLM is a targeted, expensive tool; every model proposal reports VRAM, RAM, CPU, throughput, context, frame handling, quantisation, licence and offline ability. Cells I have **not** measured or verified are marked `?`.

## 1. What is measured today (CPU laptop, 5.7 GB RAM, no GPU)
Full pipeline (analysis + plan + render of reel/square/landscape + QC), fixed options `--source-audio bed --no-transcript --no-captions`, from `results/suite/timing.json`:
| Source | Length | Wall time | Per video minute |
|---|---|---|---|
| Golden Temple (deleted since) | ~11 min | 805 s | ~73 s |
| Tirumala darshan 2021 | 6.1 min | 450 s | ~74 s |
| Tirumala drone 2025 | 6.3 min | 709 s | ~112 s (the run overlapped heavy memory pressure; treat as an upper bound) |
Highest process memory seen across the suite: **about 3.0 GB** (`ru_maxrss` of children, a running maximum). The machine's low-memory reaper has killed background jobs; the rule is one heavy job at a time.

## 2. Tiers (brief §21) mapped to the code
| Tier | What | Where | Mandatory? |
|---|---|---|---|
| 0 | FFmpeg probe, black-frame detection, metadata | `ffmpeg.py`, `highlights.detect_black` | yes |
| 1 | Scenes, per-0.5 s slot measurements (sharpness, jitter, motion, luma, contrast, blown-out), audio-independent picture metrics | `scenes.py`, `creative/shots.py` | yes |
| 2 | SigLIP labels/embeddings/aesthetic/no_text/beats; (CLAP and Whisper exist but are **not used for editing decisions under the own-audio rule**) | `vision.py`, `beats.py` | yes (SigLIP) |
| 3 | Targeted VLM / quality model / text-prompted segmentation | **not built** | **never mandatory**; only on a shortlist |
Rule: Tier 3 runs on ≤ ~30 candidates per video and ≤ 5 critic calls per reel (`SEQUENCE_SEARCH.md`), is cached in the observation files, and its absence must leave a working (lower-quality) pipeline.

## 3. Prerequisite for any GPU use
Only Whisper has a device switch (`transcribe.py`, `device="auto"`, `compute_type="int8"`). SigLIP (`vision.py` ~l.153–175), CLAP (`audio.py`, `music_web.py`), TransNetV2 (`ml_worker.py`) and Beat This! (`beat_this_worker.py`) are CPU-only today. Add one `AIKYAM_DEVICE` helper, move models and tensors, keep CPU working, and prove equal outputs (label top-1 / beats on the benchmark frames) before any new model.

## 4. Candidate Tier-3 components
| Component | Job | VRAM / RAM | Throughput | Frames / context | Quantisation | Licence | Offline |
|---|---|---|---|---|---|---|---|
| **DOVER** (VQAssessment/DOVER) | aesthetic + technical clip quality (E1) | ? (measure) | ? | short clips, sampled frames | ? | ? verify | yes (weights on Hugging Face, per its README) |
| **SigLIP text prompts** (already loaded) | shot size / angle (E2) | none extra | ~free (reuse embeddings) | keyframes | — | already in use | yes |
| **SAM 3** (848M params, per its paper) | text-prompted subject mask + tracking for framing (E3) | fp16 weights ≈ 1.7 GB by arithmetic; tracker memory `?` | ? | video, memory bank | ? | ? verify; weights may need access approval | yes once downloaded |
| **Qwen3-VL-4B** (2B / 8B exist) | shot captions, VLM critic (E5) | third-party guide: fp16 ~9 GB (too big for 6 GB), **8-bit ~5 GB, 4-bit ~2.5–3 GB** (unverified) | ? | video via fps / `max_pixels`; 256K context | 8-bit, 4-bit GGUF | ? verify (likely permissive; check) | yes |
| **MiniCPM-o quantised** | alternative VLM | 5.7 GB (reported in the L-Storyboard paper) | ? | ? | q4_k_m | ? verify | yes |
Everything marked `?` must be filled by a measurement on the target laptop before a decision (`torch.cuda.max_memory_allocated`, wall time per clip, on the benchmark sets).

## 5. Procedure on the 6 GB laptop
1. Device helper + equivalence check (§3). 2. Re-run `tools/bench_suite.py` to get the GPU baseline (seconds per video minute per stage, peak VRAM/RAM). 3. One new model at a time, loaded, used on the shortlist, **freed** (`del model; torch.cuda.empty_cache()`), never two Tier-3 models resident. 4. Report the table above with real numbers; keep the CPU path working and tested. 5. Decide keep/reject through the experiment protocol (E1–E5).

## 6. What not to do
Run a VLM over every frame or every scene of a long video; make Tier 3 required; assume a model fits because a blog says so; copy code before checking its licence (`LICENSE_REVIEW` is still to be written — the brief notes VideoHighlighter is AGPL-3.0, from its own README as cited in the brief, not checked by me).
