# RESEARCH_MASTER — the "thinking editor" research set (2026-09-26)

Start here for the next feature: an editor that analyses footage, keeps the best clips, decides the **order** and the framing, and explains its choices. The user's brief is `thinking_editor_research.md` (an implementation brief, kept exactly as written except for the amendment block at its top). The set below was requested as a **lean** subset of the brief's 14 documents; the rest are written only when their topic is actually started.

## Documents
| File | Answers | Status |
|---|---|---|
| `thinking_editor_research.md` | The user's brief (direction, rules, priorities, anti-rewrite policy) | user's file; amendment added |
| `ARCHITECTURE_AUDIT.md` | Brief §28 steps 1–3: existing modules mapped to the vNext stages; real gaps | written |
| `sequencing_evidence.md` | Literature: ordering shots, best-moment finding, framing; experiment plan E1–E5 | written (restored) |
| `SEQUENCE_SEARCH.md` | Design: editorial state, transition costs, top-K, critic, trace, taste learning | written, not implemented |
| `HARDWARE_BUDGET.md` | Tiers, measured CPU numbers, candidate GPU models with unknowns, procedure | written; GPU numbers unmeasured |
| Not written yet (brief §31) | TEMPORAL_EVENT_LOCALIZATION, VIDEO_SUMMARIZATION, CINEMATIC_EDITING, MULTIMODAL_SCORING, VLM_RESEARCH, LONG_VIDEO_MEMORY, LICENSE_REVIEW, REPO_* | write when started; each must answer the brief's 10 questions |

## Decisions taken (user, 2026-09-26)
1. **Own audio only.** Recorded audio of the input videos is never used in a reel. The brief's speech / Whisper / CLAP-on-source / "audio-driven editing" parts therefore do not apply to source audio; audio work means **our own music** (beat alignment, loudness, ducking-free mix). Open: make `--source-audio off` the default and skip Whisper/CLAP by default (question still pending), and where the licensed music files are.
2. **Lean document set** first, no coding before the documents.
3. My earlier research is restored as `sequencing_evidence.md`.

## What the audit concluded (one paragraph)
Most of the brief's architecture already exists: a per-asset observation cache, `Shot` as the candidate record, a beam-search story planner, an edit-plan boundary between intelligence and renderer, QC as a deterministic critic, variants scored after render, a decisions trace. The real gaps are: the beam has **no memory of the reel so far** (shot size, energy, motion, brightness), **transitions are decided after the order**, **47 of 48 beam sequences are thrown away**, there is **no pre-render critic** and **no WHY_REJECTED trace**, no **shot-language features**, no **trained quality signal** and no **subject extent** for framing.

## Priority (brief §32, adjusted by the audit)
- **P0 (design done):** architecture audit ✔; observation schema (small); candidate representation (`Shot` + shot size/motion); event construction (wire `events.py` behind a flag); **editorial state + transition costs** ; top-K + deterministic critic; decision trace.
- **E2 done (EXP-025):** shot size from the existing image-text model is a usable *wide-vs-tighter* signal (tightness Spearman +0.74; classifier accuracy 0.68 missed the 0.75 bar; medium vs close is at chance), stored passively as `shot_size`/`tightness`; nothing reads it yet.
- **E1 partly done (EXP-028):** DOVER fused is the best quality signal measured (+0.58 within-video rho vs +0.24 for the current measure) but its edge over the current measure is not established with 28 windows from 2 videos; licence is NON-COMMERCIAL (S-Lab 1.0), so any product use needs a decision.
- **User directive 2026-09-27: stop sequencing A/B experiments; integrate the existing pipeline instead.** Done (EXP-030): `events.py` boundary detection (already used in candidate validation since EXP-004b) now also tags `Shot.event_index`, carried into `edit-plan.json`. Verified end to end on real footage (`tools/verify_pipeline.py`, all 9 checks pass). No new scoring term added -- see EXP-030's rationale.
- **E4 + E4b done (EXP-026, EXP-029): BOTH REJECTED.** E4 (4 terms): 4 of 10 decided pairs. E4b (same-source-repeat alone, isolated test of E4's exploratory lead): 3 of 12, opposite direction -- the mechanism worked (all 12 ON reels really had fewer same-source cuts) but was not preferred. Combined: 14 of 32. Do not retry transition costs without a stronger reason or a way to separate repetition from clip choice.
- **Searched for open-source solutions per user request (2026-09-27), before building anything new:**
  - **Framing (EXP-031):** Google MediaPipe Face Detection (Apache-2.0, mature, pip-installable) tested on the same 22-frame benchmark as EXP-023. Face found on only 6/12 compact-subject frames (temple subjects are usually objects, not faces); where found, no better than centre crop. REJECTED, same conclusion as EXP-023 (tracker, SigLIP crops, OWL-ViT all failed too). SAM 3 (text-prompted) remains the one untested candidate; needs the GPU laptop.
  - **Weak/soft sources (EXP-032):** Real-ESRGAN (BSD-3-Clause, real pretrained weights) measured at **106 s to 4x-upscale ONE 640x422 frame on this CPU** — 3.7-8.8 hours per 5-10s clip, infeasible for video here. Plausibly usable for a single still/thumbnail on this machine; re-test for video on the GPU laptop.
  - **Clip quality (EXP-028 -> EXP-033 -> EXP-034, CLOSED, REJECT):** extended to 100 windows across all 4 current videos. DOVER's apparent edge (EXP-028, 28 windows/2 videos) did NOT hold: on the 2 new videos it split (won tourist, lost badly on padmavathi, our softest source, where the existing cheap slot-quality signal is at its BEST). Mean rho now +0.22 (DOVER fused) vs +0.33 (current measure); improvement-over-current 95% CI centred below zero. No change to `scoring.py`. The licence question (EXP-033: pyiqa non-commercial, idealo archived/TF, nima.pytorch has no weights) is moot -- nothing is being integrated.
  - **Sequencing:** confirmed (again) that neither FilmGPT nor the cinematology-embedding paper released code. Nothing to copy; our own two attempts (EXP-026, EXP-029) are already rejected. No third attempt without a new, real reason.
- **P0 experiments next:** more E1 windows (~100) to finish EXP-028, A/B preference collection — all can start on the CPU machine. E3 (SAM 3 framing) and further Real-ESRGAN video testing need the GPU laptop.
- **P1:** temporal event localisation, multimodal fusion, diversity optimisation, targeted VLM verification. **P2:** learned reward model, graph planning, MCTS — only if the beam + transition terms plateau.

## Working rules for whoever implements (also in `CLAUDE.md`)
Measure first, log every experiment (EXP-025+) including negatives; new terms default to **weight 0** so today's output is unchanged until an A/B says otherwise; run `tools/bench_suite.py compare` before and after; never make a Tier-3 model mandatory; check licences before copying code; keep `story.py`'s known-red test as it is; commit/push only when the user asks.
