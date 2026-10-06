# TRACK.md — live status. Read THIS first; open other docs only for the task at hand.
Update this file at the end of every session (the "Now" and "Log" parts). Last updated: 2026-10-06.
CLAUDE.md = rules + how to talk to Gagan. This file = where we are. `docs/research/experiment_matrix.md` = evidence (grep it, never read it whole: 1300+ lines).

## What we are building (one line)
`aikyam-video process INPUT... -o results/NAME` turns long temple recordings into a 9:16 Reel with no human editing: analyse → pick moments → story plan → own-music mix → FFmpeg render → QC re-edit. Goal: reels that look like advanced human editing. Private project. Gagan judges only by **watching the reel**.

## Where we are (verified 2026-10-06)
- This machine: Linux, ~5 GB RAM, **no GPU** (`nvidia-smi` absent). The GPU/WSL2 move in CLAUDE.md §7 has NOT happened yet. Nothing in the code uses CUDA.
- Last commit `f17384c` (docs only). **Uncommitted work not described in CLAUDE.md** (11 modified files + new ones):
  - `--engine v4` + `creative/selector_v4.py`: simple deterministic selector (quality → no overlap → 5 clips → source order), own weights `V4_WEIGHTS` in `scoring.py`. Meant as a baseline to A/B against the story engine. Tests: `tests/test_selector_v4.py`. **No EXP entry, no result yet.**
  - `--framing full`: full-bleed 9:16 with slow pan instead of blurred bands (`reframe.py`, `layout`). Tests: `tests/test_full_framing.py`. **No EXP entry, never judged by Gagan.**
  - `--director-file`: edit decided by a human/Claude from `*.candidates.json` + contact sheet (`creative/engine.py`).
  - `analysis.text_track` / `textframes.json`: dense no-text check every 2 s (caches older than this lack it).
  - `tools/kaggle_vlm_score.py` + `colab/aikyam_vlm_test.ipynb`: SmolVLM2-2.2B frame scoring on a free Kaggle/Colab T4 (workaround for no local GPU). `tools/bench_clips_ratings_user.json` = Gagan's own ratings (700 B, tiny).
  - Unknown: whether the Kaggle/VLM run was done or what it showed. Ask Gagan, don't guess.
- 28 rendered 9:16 reels exist in `results/`. Best known: `results/tirumala_multi2/reel-9x16.mp4`.
- Tests: ~17 min full run on CPU; one known-red test left on purpose (see CLAUDE.md §6). Re-run only targeted tests after small edits.

## Settled — do not reopen without NEW evidence
Rejected by measurement: label calibration (012), learned ranker (016), small VLMs on CPU (013), swapping image-text model (015), subject locators incl. tracker/SigLIP/OWL-ViT/MediaPipe (023, 031), clip-quality gate (021), tighter window (022), DOVER (028→034), transition-cost sequencing and spread-sources (026, 029 — blind-judged by Gagan, both failed). Real-ESRGAN: works but 106 s/frame on CPU (032) → GPU only.
Gagan's directives: own audio only (no source-audio in reels); no more exploratory sequencing A/Bs; commit/push only when asked; downloads only when asked and only CC/PD.

## Next, in order (what actually moves the reel)
1. **Finish what is half-done**: render the same source with `--engine creative`, `--engine v4`, and `--framing full` vs classic; look at contact sheets yourself; show Gagan blind pairs; log as EXP-035/036 (positive or negative). Then ask whether to commit.
2. Ask Gagan: did the Kaggle VLM scoring run? Result → decides whether a VLM "darshan/what is this clip" signal is worth integrating (open item: REVEAL almost never finds a deity shot).
3. GPU move (CLAUDE.md §7 A–E): `AIKYAM_DEVICE` helper, then E1 SAM 3 framing, E2 Real-ESRGAN for soft sources. Only if the GPU laptop is actually in front of us.
4. Short single-scene sources (2–4 clips, near-duplicates) and `--opening establish` returning close-ups: both reproducible bugs with known test videos.
5. Listen to the audio mix once (never done).

## Research round 2026-10-06 (4 agents; READMEs/papers only, nothing was run — all numbers unverified)
Your repo = `github.com/rakesh0x/OpenCardboard` ("Cutboard", chat-to-edit editor). Little for us: it is an editing shell, no vision intelligence.
| # | Idea | Source | Test (log as EXP-035+) | Needs |
|---|------|--------|------------------------|-------|
| 1 | **Director + critic loop**: Claude picks clips from candidates + a caption per clip (OpenStoryline rules: use the set, same-scene grouping, no A→B→A, JSON `{clip,start,end,beat,reason}`, self-check runtime), then after render a fresh critic looks at a filmstrip at every cut (±1.5 s) + first/last 2 s, ranks problems with timecodes, max 3 passes, deterministic planner applies fixes | browser-use/video-use, FireRed-OpenStoryline, OpenCardboard (capture-frame verify) | render creative vs director vs director+critic; 3 blind pairs for Gagan | `--director-file` exists; no GPU |
| 2 | **Few-shot SigLIP probe** for darshan/aarti/abhishekam/procession/queue/other: Gagan labels ~20-30 frames/class, logistic regression on stored embeddings, leave-one-video-out vs zero-shot | VLM agent | per-class recall, esp. darshan | CPU, seconds; needs Gagan's labels |
| 3 | **SmolVLM2-2.2B closed-set caption/label** on bench frames (Kaggle T4; fits ~5 GB fp16); fallback Qwen2.5-VL-3B 4-bit | `tools/kaggle_vlm_score.py` | vs SigLIP on `bench_labels_truth.json` | Kaggle now, GPU later |
| 4 | **Saliency framing**: UNISAL (15 MB, CPU) centroid+spread as new locator in `bench_subject.py`; must beat 0.80/0.90 (centre 0.75/0.88) or saliency is dead. If it passes: per-shot STATIONARY/PAN/TRACK rule in `--framing full`, and choose full vs blurred bands per shot by subject width | pyautoflip, unisal, OpenShorts | bench first, then 3-way blind render | CPU |
| 5 | Cut hygiene: 30 ms audio fades + 30-200 ms cut padding (check `render.py`/`mixer.py` first); ducking defaults 25% / 150 ms ramps for comparison; then listen | video-use, OpenCardboard | before/after reel + listen | CPU |
| 6 | Beat-density cutting: high-energy section = cut every beat, medium every 2, low every 4, using our Beat This! beats | OpenCardboard `tools/beat.ts` | opt-in A/B | CPU |
| 7 | Soft clips: `realesr-general-x4v3` at 2x (small video model) instead of x4plus; gate on low sharpness | VLM agent | 3 Padmavathi clips: s/frame, VRAM, blind rating | GPU |
Skip: op-log/undo rewrite, SeedVR2, TranSalNet, auto-vertical-reframe/openshorts tracking (person-centred = tracker failure again). Not found: any good beat-sync or J/L-cut/speed-ramp repo; any deity/ritual dataset (own labelled frames are the only data).
Order: 1 (no new model, targets reveal + short sources) → 2 (needs Gagan ~30 min labelling) → 4 bench → 5 → 3 → 7.

## Where time and tokens are being wasted — and the fix
1. **Re-reading everything each session.** CLAUDE.md (~6k words) + HANDOFF + 6 research docs + 1300-line experiment log. → Read only this file; grep the matrix for the EXP you touch.
2. **Experiments with no decision attached.** 34 EXPs, ~15 rejected; the same ideas (locators ×4, quality models ×3, sequencing ×2) were tried in sequence on tiny single-rater benchmarks. → Before any new experiment write one line: "if it wins, I will ship X". If no such X, don't run it.
3. **Heavy CPU work that cannot finish.** Real-ESRGAN (hours/clip), VLMs on CPU, 17-min test suite, one-job-at-a-time on 5 GB RAM. → Use targeted tests; push model work to the GPU/Kaggle, don't benchmark it on CPU.
4. **Work not recorded.** V4, full framing, director file exist only in the working tree; the next session would have to rediscover them (this one did). → Log each change here the same day, and commit when Gagan says so.
5. **Heuristic tuning vs. the real signal.** Benchmarks are rated by one person (the previous Claude) while Gagan's own judgement is what counts. → Prefer a few blind pairs judged by Gagan over another automatic metric; grow `bench_clips_ratings_user.json`.
6. **Stale caches and metrics that measure themselves** (EXP-020, EXP-034 bug: `bench_gate.fit` reuses old `signals.json`). → Delete/checksum caches before trusting a benchmark number.

## Focus (the one-paragraph answer)
Stop searching for new models/metrics. The pipeline already works end to end. The gap between "works" and "looks hand-edited" is: (a) a true darshan/deity shot for the reveal, (b) soft sources, (c) wide subjects cut by the 9:16 window / blurred bands, (d) short sources, (e) the audio mix nobody has heard. Pick ONE per session, make a reel, look at it, let Gagan judge it.

## Log (newest first; one line each)
- 2026-10-06 (later): Gagan wants the **YouTube-Shorts look** (full-bleed, one consistent style, slow motion, not too fast) and the ultimate goal = long video -> remove its audio -> reel -> our own music, no captions/transcript. Done: CLI defaults now `--source-audio off`, no transcript, no captions (`--with-captions/--with-transcript` opt-in); director loop run by hand (Claude picks clips from `*.candidates.json` + `*.sheet.jpg`; MAX 7 clips per director plan, more = silent fallback to automatic); slow motion added (`Options.slowmo`, `--no-slowmo`, `SLOWMO` role map in `creative/engine.py`, segment `speed`, render `setpts` + `minterpolate=blend`; slowed clips play LONGER on the timeline). Reels: `results/dc1` automatic baseline, `dc3` director, `dc5` director + `--opening establish --framing full` + slow motion (38.8 s). Not yet judged by Gagan. Gotchas: edge snapping shrinks clips to cut-free windows (~3 s) so asked seconds are not honoured; `pkill -f` kills own shell. TODO: automate the critic pass (filmstrip at cuts); make `--framing full` default if Gagan likes dc5; QC `subject_in_frame` warns on full framing.
- 2026-10-06: research round (4 agents) saved above; no code changed.
- 2026-10-06: created TRACK.md; audited working tree (V4 selector, full framing, director file, Kaggle VLM script found uncommitted and unlogged). No code changed.
