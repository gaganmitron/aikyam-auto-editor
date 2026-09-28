# Sequence search: which clip first, which next (design, 2026-09-26) — nothing here is implemented

Evidence: `sequencing_evidence.md`. Existing code: `ARCHITECTURE_AUDIT.md`. Rules: brief (`thinking_editor_research.md`) + amendment: **recorded audio is never an input; own music only.**

## 1. Problem and method
Choose the *sequence* of clips, not the top-scoring clips. Source order is meaningless for us (long recordings, several videos), and recovering "the original order" is ill-posed (humans ≈ 40% on 3 shots). So the objective is **editorial grammar**: how the clips read one after another. Method: keep the deterministic beam search, give it a memory of the reel so far (editorial state) and a cost for each cut (transition cost) built from measurable shot features, return the top-K sequences, and let a critic pick — deterministic first, a small VLM later. Weights of the new terms are fitted once from the user's A/B choices, then everything is automatic.

## 2. What Aikyam already has (details in the audit)
Beam (width 48) over the six roles; role affinity incl. beats; redundancy, chronology and same-beat penalties; skip penalty; top-up; overlap resolution; QC after render; `autoedit` variants scored after render; a `decisions` trace with WHY_SELECTED-like fields.

## 3. What is missing
1. State: nothing about the previous clip's **shot size, energy, camera motion, brightness** is used in the search.
2. Transitions are chosen **after** the order is fixed (`transitions.plan_transitions`), so they cannot influence the order.
3. The beam returns one sequence; 47 alternatives are discarded.
4. No pre-render critic over alternative sequences of the *same* clips.
5. No WHY_REJECTED / per-term score breakdown.
6. No shot-language features (the only cue that mattered in the published ablation).

## 4. Design (smallest steps, in this order)
### 4.1 Shot card (extend `Shot`, `creative/model.py`)
Add `shot_size` (wide / medium / close / detail, as a small distribution), `camera_motion` (still / pan / handheld-shake / zoom, from existing slot `motion`, `jitter` and the global shift in `shots._shift`), `energy` (already `Population.energy`), keep `luma` (slots), keep `embedding`, `beats`. **Status: `shot_size` is implemented as a passive per-scene/per-shot distribution and `shotsize.tightness()` (EXP-025); reliable for wide-vs-tighter only, medium-vs-close is at chance, so `size_step` / `size_repeat` must use `tightness` thresholds, not class names.** Original plan: `shot_size` first from cheap signals (`concentration`, face area, SigLIP text prompts — experiment E2); a VLM only if E2 fails. Optional `quality_model` (DOVER) if E1 earns it.

### 4.2 Editorial state (a small dataclass inside `plan_story`, not a new subsystem)
`{duration_so_far, prev_size, prev2_size, prev_energy, prev_motion, prev_luma, prev_beat, prev_asset, count_by_asset}`. It is derived from `chosen` at each expansion, so no change to the beam's data structure.

### 4.3 Transition cost `T(prev, cand)` — added to `inc` in the beam
Terms (each computed from the shot cards; each has its own weight, **default 0 = current behaviour**):
| Term | Meaning | Why |
|---|---|---|
| `size_repeat` | two close/detail shots in a row | shot category was the strongest ordering cue |
| `size_step` | ordinal size change (wide→medium→close vs wide→detail) | do not hard-code "wide→medium→close": let A/B decide the sign |
| `energy_jump` | \|Δ energy\| beyond a threshold, except where the arc calls for it (CLIMAX) | pacing continuity |
| `motion_unrelated` | both clips move strongly, different assets, low embedding similarity | FilmGPT's failure: "cuts on action between unrelated videos" (multi-video reels) |
| `luma_jump` | large brightness change at the cut | flash / colour mismatch (colour match only partly covers it) |
| `semantic_gap` | U-shaped in embedding similarity: penalise near-duplicates (already `red`) and completely unrelated neighbours mildly | continuity without repetition |
| `same_beat` | existing `beat_repeat_penalty` | keep |
Never assume film rules are universal (brief §17): every term is a feature with a weight fitted by §4.6.

### 4.4 Search and alternatives
Keep the beam. Return the top-K **distinct** members (different shot sets or order; K = 3–5) in `Timeline.alternatives` with their per-term breakdown. Do not change what `plan_story` returns as its main sequence unless the weights are non-zero.

### 4.5 Critic (before render)
Deterministic critic scores each alternative: role coverage, redundancy, summed transition cost, shot-length variance (pacing), target-range fit, dead-slot share = 0, weak-clip count. It picks one; `autoedit` variants (opening / pacing / length) still run on top. **VLM critic (later, GPU):** given the shot cards plus one keyframe per shot, rank the top-K; ≤ 5 model calls per reel; compared against the rule critic in E5. Never an LLM as the only sequencer (L-Storyboard: LLM-only ordering was worse than trained visual models).

### 4.6 Learning the user's taste once
Collect pairwise judgements: two reels from the *same* clip pool, differing only in sequence; the user picks the better. ≥ 30 pairs; fit the term weights with Bradley–Terry / logistic regression; hold out a third of the pairs; adopt only if held-out agreement ≥ 60% and the A/B win rate over the current planner is convincing (E4: ≥ 14 of 20 pairs; a 20-pair test cannot resolve small gains). Store weights in `data/roles.json` `story`.

### 4.7 Decision trace
Extend `creative.decisions`: per selected clip the term breakdown (affinity, quality, redundancy, chronology, beat, transition terms); per role the top-5 rejected candidates with the reason (`clash`, `duplicate`, `below affinity floor`, `inc ≤ skip`, `lost to X by Δ`); the alternatives table. Small (tens of JSON lines per reel), enables comparing algorithms without rendering.

## 5. Metrics (for `tools/bench_suite.py` and E4)
Selection: redundancy (max/mean embedding similarity), role coverage, distinct beats, distinct shot sizes. Sequence: summed transition cost, size-repeat count, energy variance. Editing: mean shot length, pacing variance. Technical: QC status, dead-slot share (must stay 0). Preference: A/B win rate. Efficiency: analysis seconds per video minute (currently ~73 s/min CPU), plan time.

## 6. How to test
- Unit tests per term (synthetic shots with known size/energy/motion); each term must change the score in the stated direction.
- **Regression guard:** with all new weights at 0 the chosen sequence must be **identical** to today's on the suite videos (`bench_suite.py compare`: no regressions).
- E2 (shot size accuracy) before E4; E4 before E5; every result logged as EXP-025+ with negative results included.

## 7. What NOT to implement
- An LLM/VLM as the sole sequencer; MCTS, DPPs or learned reward models before the beam + transition terms are measured (brief P2).
- A FilmGPT clone (6,200 h of training data; code unreleased).
- Any feature computed from the recorded audio.
- A database (the file cache is enough; brief §29).
- A rewrite of `story.py`: extend `inc`, keep the known-red test's behaviour (EXP-002b/003).

## 8. Cost
Search terms: negligible CPU (a few array operations per expansion; beam 48 × pool 16 × 6 roles). Shot size via SigLIP text prompts: reuse the existing embeddings, ~0 extra cost. DOVER and a VLM critic: GPU, see `HARDWARE_BUDGET.md`.

## 9. Status update (2026-09-27): E4 result
The four transition terms of §4.3 with prior weights 0.10/0.10/0.20/0.10 were **rejected** by the user's blind A/B (ON preferred 4 of 10 decided pairs, p = 0.75; EXP-026). They stay as the opt-in `--sequence-terms` experiment. Do not tune them from 10 pairs. The one lead worth a separate test is a same-source repetition penalty (fewer back-to-back clips from the same video), see EXP-026.

## 10. Status update (2026-09-27): E4b result
`--spread-sources` (same_source_repeat alone, EXP-029) was also **rejected**: 3 of 12 decided pairs, opposite direction from the E4 exploratory lead it was built to test. Combined with E4, 14 of 32 decided pairs across two independently designed experiments. Transition costs as designed here do not have user support; do not retry without a new hypothesis or a design that separates repetition-avoidance from the clip substitution it causes.
