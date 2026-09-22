# Creative layer audit (before the redesign)

Scope: everything between "validated moments" and "encoded file" (plan.py, render.py, music.py, captions.py, reframe.py, the music/transition code added in
earlier passes). Ingestion, analysis, candidate validation, EditPlan contract, API, Kafka, storage and infrastructure were kept.

## Evidence: the old engine measured on 8 real temple clips (`tools/eval_edit.py`, results/eval_baseline.json)
_The 8 source videos were removed afterwards to keep the repo small; only the Ganga aarti clip remains in `samples/`. The measurements below are kept as the record._
| clip | reel | segments | mean shot | transition | LUFS | true peak | other |
|---|---|---|---|---|---|---|---|
| durga-aarti (74 s) | 43.5 s | 4 | 11.2 s | crossfade | -16.4 | -3.7 | **1.1 s frozen picture** |
| madan-mohan-aarti | 21.5 s | 2 | 11.0 s | crossfade | -16.1 | -6.1 | |
| sudalai-madan-aarti | 15.0 s | **1** | 15.0 s | - | -15.9 | -5.5 | |
| ganga-aarti-haridwar | 22.6 s | 2 | 11.6 s | crossfade | -16.1 | -3.8 | |
| rath-yatra-procession | 15.0 s | **1** | 15.0 s | - | -17.4 | -3.3 | |
| theyyam-ritual | 6.1 s | **1** | 6.1 s | - | -15.9 | -2.5 | |
| mahashivratri | 11.1 s | 2 | 5.8 s | crossfade | -15.6 | -4.5 | |
| dussehra-jakhu-temple | 15.0 s | **1** | 15.0 s | - | -17.2 | **+0.2 dBTP** | 1 clipped sample |

Half of the reels are one 15 s clip: there is no edit at all. Shots average 11 s (a social short usually cuts every 2-6 s). Nothing caught the frozen
picture or the +0.2 dBTP overshoot; they shipped.

## Weak abstractions found (with where)
1. **Selection is a greedy score sort, not a story.** `plan.plan_edit` sorts moments by score, takes the best that fit, then re-sorts chronologically. No roles, no arc,
   no notion of redundancy: two shots of the same idol are as welcome as an idol and an aarti. Story order is forced to source order (`validate_plan` rejected anything else).
2. **In/out points are arbitrary.** A clip starts at the moment's start and is cut at a fixed `max_seg_s = 15`; only a "quiet snap" nudges the end. Nothing looks at sharpness,
   shake, exposure or motion, so a clip can start in a blur or a lens flare.
3. **One transition for the whole reel.** `transitions` is a single `{type, seconds}` (0.5 s crossfade): the same dissolve between two shots of the same scene (ghosting),
   across a night->day exposure jump (flash) and inside continuous chanting. `render.py` had no way to express a cut next to a blend.
4. **Fixed constants everywhere**: 15 s max shot, 0.5 s blend, 0.75 s snap window, 0.35/0.5 music volume, ducking `threshold=0.05 ratio=5`, 45 s target. None depends on the footage.
5. **Music is not part of the edit.** `music.choose` picks by label/energy from the *finished* plan; it knows no tempo, no structure; cuts are never aligned to a beat; the start
   offset is always 0; ducking is a fixed sidechain that cannot express "16 dB under chanting, 8 dB under ambience". Loudness is `loudnorm` on the whole graph, unmeasured.
6. **Audio is built inside the FFmpeg filtergraph**, so nothing can be inspected or re-mixed: no stems, no per-clip level matching (clips differ by 20 dB in temples), no J/L-cuts,
   no way to test "music never overpowers live sound", true peak overshoots after AAC.
7. **Composition = one static x-path per clip.** Wide subjects (processions, crowds) are amputated by the 32% width of a 9:16 crop; there is no fit/blur layout and no push-in.
8. **No QC.** Nothing measures the encoded file, so nothing can refuse to publish it, and nothing can fix it.
9. **One plan per source.** `plan_edit` returns a single plan; multiple reels are structurally impossible.
10. **Captions**: char-count wrapping, fixed bottom margin inside platform UI zones, no reading-speed limit, no overflow measurement.

## What was replaced, and how (see docs/CREATIVE_ENGINE.md)
| weakness | replacement |
|---|---|
| 1, 9 | `creative/story.py`: role affinities + deterministic beam search over the arc; `plan_stories(k)` shares the footage between k reels |
| 2 | `creative/shots.py`: 0.5 s slot features (sharpness, jitter, motion, exposure, concentration, face, live level), percentile-ranked against THIS footage; `best_window` picks in/out |
| 3 | plan schema v2: per-edge `transitionIn`; `creative/transitions.py` decides cut / crossfade / dip per edge from similarity, exposure jump, energy, sound continuity, pacing |
| 4 | `data/pacing.json`, `data/roles.json`: named policy, adaptive shot length from energy; quality signals are ranks, not pixel thresholds |
| 5 | `creative/musicdna.py` (tempo, beat grid, energy, kind), `music_sync.py` (offset on a beat, cuts aligned to beats), `engine.choose_music` (context + live-sound conflicts + pacing) |
| 6 | `creative/mixer.py`: numpy mix with stems, level matching, J/L-cuts, automated ducking, BS.1770 loudness, look-ahead true-peak limiter |
| 7 | `creative/layout.py`: continuous window (core of the action + margin, capped) with blurred fill for the leftover height; slow push-in on calm shots |
| 8 | `creative/qc.py` + `replan.py`: measure the encoded reel; deterministic targeted re-edit; publish gate |
| 10 | safe-area margins, `retime_cues`, libass-measured overflow/overlap checks in QC |
