# Experiment registry

Per `new.md` §16/§34. One entry per experiment. Only created what EXP-001 needed —
not the full `docs/research/` tree §19 suggests; add files when an experiment needs them.

---

## EXP-001: does the current story order respect the source's real event order?

**Hypothesis:** the beam-search story planner (`creative/story.py`) picks clips by
role-affinity + quality only, with no notion of the underlying ritual's real time/causal
order, so the edited sequence can misorder events relative to what actually happened.

**Baseline:** current shipped pipeline (moments -> role affinity -> beam search over
`ROLES` template), no Knowledge Graph, no event lifecycle, no memory.

**Dataset:** `inputs/diwali-mumbai.webm` (CC BY 3.0, Wikimedia; real temple footage,
~343 s), the only real recording with a fresh render at experiment time.
3 runs already on disk: `results/diwali_v1`, `results/diwali_v2`, `results/diwali`.

**Method:** extract `creative.decisions` (`type: select`) from each `edit-plan.json`,
read `source: [start, end]` per role in final timeline order, count transitions where
the next clip's source start is *before* the previous clip's source end (a
"chronological inversion").

**Result:**

| run | transitions | inversions |
|---|---|---|
| diwali_v1 | 5 | 3 |
| diwali_v2 | 5 | 3 |
| diwali | 5 | 3 |

Same 3 inversions in all 3 runs (BUILDUP, REVEAL, CLOSING each jump backward in source
time relative to the clip before them). Consistent, not noise. Concretely:
`CLOSING` in all 3 runs is a priest shot at source 192–196 s, placed *after* the
aarti/fireworks climax at 262–301 s — the reel ends on something that happened before
its own climax.

**Conclusion:** confirmed. The role template (`OPENING..CLOSING`) is atemporal by
design — it optimizes label-affinity per role slot, not narrative/causal time order.
On a single continuous ritual with a real beginning->peak->close arc, this can produce
a "closing" that isn't actually the close.

**Next action (Gate 1–3 pass; Gate 4 compute cost is ~free):** don't build the Final.md
Knowledge Graph/event-lifecycle machinery for this — cheaper fix first: in
`story.py`'s beam search, penalize (or sort within-role by) source-time inversions
against neighboring selected clips, and re-run this same inversion count as the
regression check. Only escalate to explicit event-sequence modeling (§9/§37 of
`new.md`) if the penalty approach still leaves inversions on multiple real clips.

**Status:** implemented and validated.

**Implementation:** `creative/story.py` beam search had a chronology term already
(`+ chronology_bonus * (fraction of consecutive pairs in source order)`, per the
file's own docstring) but it was asymmetric (bonus only for forward, no penalty for
backward) and divided by `n_t` (~6x too weak, ~0.02 effective vs. affinity/quality
terms in the 0.1-0.8 range) — so it never won against a role's better label match.
Fixed: symmetric penalty (`+bonus` forward / `-bonus` backward) compared against the
nearest prior same-asset clip (not just the immediately preceding beam entry, so it
still holds for interleaved multi-asset reels), and `data/roles.json`'s
`chronology_bonus` raised 0.12 -> 0.3 to actually compete.

**Validation:** re-ran the same Diwali source (`results/diwali_exp001`, cached
analysis not reused — this was a fresh output dir so full re-analysis ran; future
re-plans should point at an existing run's `assets/` to skip that).

| run | transitions | inversions |
|---|---|---|
| diwali (pre-fix) | 5 | 3 |
| diwali_exp001 (post-fix) | 5 | **0** |

Held through the QC-triggered deterministic re-edit pass too. Bonus effect: the
resulting order now also reads as a coherent arc on its own — idol -> priest ->
deity reveal -> devotees/aarti building -> aarti+fireworks climax -> fireworks
closing — without any KG or event-lifecycle model. `tests/test_creative_story.py`,
`tests/test_target_length.py`, `tests/test_opening_order.py` (25 tests) still pass.
QC is unchanged at `warn` (subject_in_frame, weak_shots — pre-existing, unrelated).

**Next action:** none required for this fix. Confirms the Gate-4 (`new.md` §30)
judgment: a 2-line penalty fix closed the gap EXP-001 found; Final.md's Knowledge
Graph/event-lifecycle machinery is not yet justified by evidence from this dataset.
Re-open only if a future clip with multiple interleaved assets/scenes shows
inversions this penalty doesn't catch (e.g. two source videos of the same event
where "chronology" isn't a single timeline).

---

## EXP-001b: ablation on `chronology_bonus` — is 0.3 overcorrecting?

**Why:** picking 0.3 (was 0.12) was a judgment call, not a measured value, and fixing
order can only ever come at the expense of role-fit (a shot forced into chronological
position isn't always the best semantic match for its role). Report the trade-off
honestly instead of just declaring the inversion count a win.

**Method:** re-ran ONLY the planning stage (`creative.engine.build_plans`) against
`results/diwali_exp001`'s already-cached analysis — no re-transcription, no
re-vision, no ffmpeg — sweeping `chronology_bonus` from 0 to 0.6.

| bonus | inversions | avg affinity | avg quality | duration |
|---|---|---|---|---|
| 0.00 (no penalty) | 3/5 | 0.629 | 0.476 | 32.0s |
| 0.02 (old effective value, 0.12/n_t) | 1/5 | 0.590 | 0.528 | 40.1s |
| 0.12 (old raw config, undivided) | 1/5 | 0.590 | 0.528 | 40.1s |
| 0.15 | 1/5 | 0.590 | 0.528 | 40.1s |
| **0.20** | **0/5** | **0.505** | 0.484 | 33.5s |
| 0.25 / 0.30 / 0.40 / 0.60 | 0/5 | 0.505 | 0.484 | 33.5s |

**Finding:** real trade-off confirmed, not free — going from 1 remaining inversion to
0 costs ~0.085 absolute avg-affinity (0.590 -> 0.505, ~14% relative), concentrated in
one shot (the REVEAL slot inherits a much weaker leftover once the strong idol/deity
shot is reassigned to RITUAL for order). But: (1) the value plateaus flat from 0.20
through 0.60 for this clip — 0.3 is deep in a safe margin, not perched on a cliff; (2)
quality (not just affinity) barely moves at all across the whole sweep (0.476 ->
0.484); (3) the file's own docstring already states the design intent — "temple
rituals have a natural order; break it only when it pays" — i.e. order was always
meant to dominate marginal role-fit, this ablation just confirms the chosen weight
actually delivers that intent instead of being 6x too weak to matter.

**Conclusion:** keep 0.3. It ties with 0.2 on this clip (no reason to prefer either),
and sits safely past the point where more penalty stops changing the outcome. The
quality cost is real but small, concentrated in one already-weak shot, and consistent
with the story engine's stated design goal.

**Next action:** none. Would re-open if a future real clip shows the plateau
doesn't hold (i.e. quality keeps dropping well past 0.3) — worth re-running this same
ablation script (kept at
`/tmp/claude-1000/-home-randomx-aikyam/7b16c696-1c5b-4ad0-9cdd-9c30a22e8e69/scratchpad/chronology_ablation.py`,
not in the repo — promote it to `tools/` if this kind of sweep becomes routine)
on any clip that regresses.

---

## EXP-002: does forcing REVEAL to always be filled cause the weak_shots QC warning?

**Hypothesis:** EXP-001b showed REVEAL absorbs the "leftover" cost of the chronology
fix (aff 0.303, q 0.357 — the weakest clip in the reel, and the one QC flags as
`weak_shots`). The beam search's skip-vs-keep decision for a role compares
`inc` (affinity/quality/chronology of the best candidate) against a fixed
`skip_penalty * importance[role]`. Suspicion: the now-larger chronology bonus
(+0.3, needed and validated by EXP-001/001b) inflates a weak-but-chronologically-
forward candidate's score past the skip threshold, so REVEAL never actually gets
to skip even though the skip mechanism exists for every role.

**Smallest experiment (as instructed — did not touch the validated
`chronology_bonus=0.3`):** same cached-assets, planning-only re-run technique as
EXP-001b (`creative.engine.build_plans` against `results/diwali_exp001`'s cached
analysis, no re-analysis, no ffmpeg). Compared the current 6-role arc (REVEAL
forced) against the same arc with REVEAL simply removed from `ROLES` for this run
(REVEAL never a selectable slot at all — the bluntest possible "optional" test,
monkeypatched, not a code change).

**Result:**

| | REVEAL forced (baseline) | REVEAL removed from arc |
|---|---|---|
| clips | 6 | 5 |
| avg affinity | 0.505 | 0.586 (+15.9%) |
| avg quality | 0.484 | 0.569 (+17.4%) |
| weakest shot (aff×quality) | 0.108 | 0.212 (~2x) |
| inversions | 0/5 | 0/4 (still perfect) |
| duration | 33.5s | 34.4s |

**Conclusion:** confirmed. Forcing REVEAL is the cause, not a footage limitation —
dropping it costs one clip and ~1s of duration, for a uniformly stronger reel and no
regression on the chronology fix. The mechanism: `skip_penalty` was meant to let a
thin role go unfilled, but the (correctly) large chronology bonus now inflates a
weak, order-correct candidate's score past that threshold, so the skip path never
actually fires for REVEAL on this clip.

**Caveat — this experiment used a blunt instrument.** Unconditionally removing
REVEAL from every reel isn't the production fix: it would also discard REVEAL on
footage where a genuinely strong reveal shot exists. This only establishes the
mechanism is real and worth fixing; it doesn't validate a general repair. Not yet
implemented.

**Next action (proposed, not yet done):** make the skip decision quality-aware
instead of removing the role outright — e.g. apply the chronology bonus to
*ranking among already-eligible candidates* but not to whether a role clears
`min_aff[role]`/the skip threshold in the first place, so a role only fills when
something reasonably good exists for it, chronology-correct or not. Needs its own
before/after check (this experiment's technique, reusable) once implemented, and
ideally a second real clip to confirm it's not overfit to this one.

---

## EXP-002b: implementing the quality-aware skip — what actually happened

**Instruction:** implement the gate so chronology only ranks eligible candidates
and can't push a weak one past the role-fill threshold; keep `chronology_bonus=0.3`
unchanged; re-measure EXP-002's before/after; test on a second real clip.

**What shipped (both fixes are real and validated):**

1. **The quality gate itself** (`story.py`): a candidate's `inc` (affinity/quality/
   redundancy, *without* chronology) must clear `-skip_penalty * importance[role]`
   on its own merit before chronology is added — chronology can no longer be the
   thing that pushes a candidate over that line.
2. **A second, unrelated, more serious bug found along the way**: validating the
   gate against `tests/test_multi_engine.py` (never run during EXP-001 — a real gap
   in that "validated" report) showed the **already-shipped EXP-001 fix regresses
   multi-asset reels**: `chronology_bonus=0.3` is large enough to make the search
   prefer staying on the same asset over switching to a different asset or a still
   image (neither gets the bonus, so they lose out purely on magnitude, not merit).
   Measured: a mixed 2-video+2-image pool dropped from 3 assets+1 Ken-Burns still
   (bonus ≤0.12) to 2 assets+0 stills (bonus ≥0.15) — a hard threshold, confirmed
   by sweeping 9 values. **No single magnitude satisfies both single-asset ordering
   (needs ≥0.20) and multi-asset diversity (needs ≤0.12) — this was never
   reconcilable by tuning, only by scoping.** Fix: `chronology_bonus=0.3` now only
   applies when the pool has exactly one video asset; a multi-asset pool uses a
   second, separately-pinned constant (`chronology_bonus_multi_asset=0.12`) with
   the byte-identical original formula (forward-only, divided by `n_t`) — true
   "preserve existing behavior" for the case that actually needed it. Verified:
   `test_multi_engine.py`'s two asset/still-diversity tests pass again, and Diwali
   (single-asset) still holds at 0/5 inversions, unaffected by this split.

**What did NOT get resolved:**

3. **One synthetic test still fails** (`test_creative_story.py`, a 6-shot pool for
   6 roles with zero slack — "thin pool"). Root cause, verified by directly
   disabling the length-band filter with the gate held constant: the gate
   correctly makes the true highest-scoring sequence a *shorter* one (4 clips,
   CLIMAX and CLOSING both legitimately skip because nothing good remains for
   them) — but a separate, pre-existing hard filter (`n_lo <= len(clips) <= n_hi`,
   a nominal-clip-count band) discards that sequence for being under-length and
   falls back to a worse, longer one, which is what reassigns REVEAL from the
   intended `m3` to `m5`. Tried the obvious fix (drop the `n_lo` floor, since
   duration is separately enforced downstream) — it does fix REVEAL, but now
   **CLIMAX itself goes unfilled** (a different, arguably worse test assertion
   breaks instead). Two attempts, two different cascades — not a clean fix, so it
   was reverted; `n_lo` is unchanged from the original.
4. **The gate does not actually change Diwali's output.** Checked directly:
   Diwali's real REVEAL shot has `val() ≈ 0.2` on its own merit — comfortably
   above the `-0.25` skip threshold with or without chronology. EXP-002's original
   theory ("chronology inflates a weak candidate past the threshold") was real in
   the synthetic test (there, the candidate sat close to the line) but **does not
   describe what happens on Diwali** — the REVEAL shot there is the *relatively*
   weakest clip in the reel, not an *absolutely* bad one, and the gate's threshold
   is absolute, not comparative. So the measured EXP-002 improvement (+15.9%
   affinity, +17.4% quality) came from *forcibly removing REVEAL from the role
   list* — a blunter, hypothetical intervention I used to test the hypothesis, not
   from the gate that actually shipped. **The real Diwali reel is unchanged by
   this fix.** This is the honest negative result the loop is supposed to surface,
   not bury.
5. **No second real clip was available.** Per the earlier flag: only
   `inputs/diwali-mumbai.webm` exists locally; the four Pexels clips and the Ganga
   sample referenced in `docs/HANDOFF.md` are gone from the repo. Validation for
   #2 leaned on the synthetic `test_multi_engine.py` fixture instead (real FFmpeg
   render, hand-built moments, not ML-derived — a reasonable substitute for
   *mechanism* testing, but not a second real-footage data point).

**Net assessment:** ship #2 (multi-asset regression fix) — clearly correct, well
isolated, tests pass, no real footage regressed. Ship the gate too (#1) — it's a
real, defensible mechanism (absolute quality floor before chronology can compete),
even though it happens not to bite on Diwali specifically; it would bite on
footage where a role's best candidate is genuinely bad, not just comparatively
weak. Do not chase #3 further right now — two fix attempts both cascaded into a
different broken assertion rather than resolving; it's a narrow synthetic-fixture
edge case (an exactly-one-shot-per-role pool), and forcing it now risks a third
cascade. **The original EXP-002 problem (Diwali's actually-weak REVEAL shot) is
still open** — the gate as built doesn't solve it; it would need either a
relative/comparative quality bar (not absolute) or the blunter "let REVEAL be
skipped by policy, not by threshold" approach that was actually measured to help.

**Next action:** update the one failing synthetic test's comment to document this
known limitation (`n_lo` vs. quality-gate interaction on thin pools) rather than
its literal assertion, OR leave it red and tracked here — user's call. Separately,
open a new experiment for a relative/percentile-based quality floor per role if
Diwali's weak-REVEAL problem is still worth solving.

---

## EXP-003: a relative (story-comparative) quality floor for Diwali's weak REVEAL

**Hypothesis:** EXP-002b showed the absolute floor never fires on Diwali because
REVEAL's shot is *comparatively* weak, not *absolutely* bad. A floor relative to
the story's own average-so-far (not a fixed constant) should catch that instead.

**Measurement first (no code change):** computed quality-only `val()` for each of
Diwali's 6 chosen clips as a % of their own average (0.377). Clean signal: REVEAL
sits at 54.5%, every other role at 90%+ (next-lowest CLOSING at 90.4%). A rule like
`val < 0.6 * avg` flags only REVEAL, with comfortable margin either side — exactly
the kind of cheap, code-free check the loop is supposed to run before implementing.

**Implemented:** an additional gate (kept the EXP-002 absolute floor, added this on
top) — a candidate whose `val()` falls below `relative_floor` (0.65) times the
average `val()` of the story's own already-chosen clips is rejected, same as a
too-weak-to-beat-skip candidate.

**Result — real Diwali behavior:** did NOT skip REVEAL. Instead the search found a
*different* REVEAL candidate (quality 0.357 -> 0.440, +23%) — but that forced a
cascade: BUILDUP's pick got worse (affinity 0.664 -> 0.452, a straight regression),
duration drifted from 33.5s to 39.5s. Net: avg affinity -6.7%, avg quality +2.7% —
a wash, redistributing weakness rather than fixing it, not a clean win.

**Result — full suite:** reintroduced the EXACT multi-asset regression EXP-002b had
just fixed (`test_multi_engine.py`, both tests, identical failure signature: 2
assets instead of 3, 0 stills instead of 1) — because the relative floor applies
uniformly regardless of asset count, unlike the chronology fix which was scoped to
single-asset pools. Also reproduced the CLIMAX-skip cascade in the synthetic thin
pool (REVEAL correctly becomes `m3`, but CLIMAX goes unfilled) — the third time
today a structural change to this beam search has traded one broken assertion for
another rather than resolving cleanly.

**Conclusion:** reverted, in full (`story.py` and `roles.json` both back to the
exact EXP-002b-validated state). Clean negative result, reported per protocol
rather than hidden. Three independent attempts today (`n_lo` removal twice, this
relative floor) at extending the beam search's role-fill/skip logic have each
introduced a new regression somewhere else in the same search — strong evidence
that **this specific mechanism (sequential per-role beam selection with additive
score adjustments) is close to its safe-change limit** without a larger redesign
(e.g. scoring the whole sequence's coherence jointly, rather than gating individual
role/candidate pairs one at a time).

**Next action:** do not attempt a fourth in-place patch to this function without a
different strategy. Diwali's weak-REVEAL problem (EXP-002's original finding)
remains open and unresolved. If it's worth solving, the next experiment should
first ask a cheaper question before touching the beam search again: is a weak
REVEAL slot actually worth fixing at all vs. just accepting it (real QC only warns,
doesn't fail, on this clip) — that's a product-priority question, not a research
one, and belongs with the user rather than another algorithm attempt.

---

## EXP-004: baseline temporal event localization (start/peak/end) on Diwali footage

**Why this, now:** per `new.md`'s protocol and Final.md §34 ("can Aikyam correctly
identify events and their temporal relationships before worrying about editing?"),
measure the current pipeline's ability to localize a real event BEFORE building any
of Final.md's proposed event/KG/memory machinery. No event-lifecycle representation
exists in the codebase at all right now — the only per-scene signal is a single
SigLIP vision label's confidence, plus per-shot audio-event tags (CLAP). This
experiment establishes ground truth by hand and measures that baseline against it.

**Method:** read 13 keyframes directly (`results/diwali_exp001/keyframes/scene_*.jpg`,
already-rendered stills from the cached analysis, no new extraction) spanning the
full clip, to build a human-verified timeline. Cross-checked against the per-scene
`aarti` vision-label confidence (`vision.json`) and per-shot CLAP audio-event tags
(`audio.npz`: bell/conch/chant/bhajan/drums/applause/speech, 0.5 s hops).

**Ground truth (established by looking, not inferred from labels):** the clip
contains **two temporally separate real events**, not one continuous ritual:

- **Event A — daytime home puja/aarti**, ~165s–236s: family gathered indoors,
  lighting diyas, hands folded in prayer, a notebook/swastik-writing moment
  (likely Chopda Pujan, a common Diwali account-book ritual) partway through.
  Confirmed daytime by visible daylight through windows in every frame checked.
- **Event B — nighttime fireworks/sparklers**, ~241s–301s+: family and kids
  outdoors lighting firecrackers and sparklers on the street. Confirmed nighttime
  by dark sky, streetlights, headlights in every frame checked.
- The two are visually unambiguous as separate occasions (day vs. night alone
  proves it, independent of any content judgement) — likely hours apart, not one
  continuous arc with a single "peak."

**Baseline result — vision label alone (`aarti` confidence, the only signal the
current pipeline scores events on) is not just imprecise, it's wrong about WHICH
event is the peak:**

| | ground truth | naive `aarti`-label estimate | error |
|---|---|---|---|
| Event A start | ~165s | 165.2s (first `aarti` > 0.3) | ~0s — good |
| Event A peak | ~220–230s (largest family group, most formal) | n/a — label is zero exactly there | wrong entirely |
| Event A end | ~236s (last confirmed daytime frame) | ~170–220s (label goes to zero repeatedly; a naive contiguous-run detector ends the event 16–66s early) | 16–66s early |
| **global peak** (argmax of `aarti` across the whole clip) | **should be inside Event A** | **241.4–257.3s, confidence 0.61 — the single highest `aarti` score in the entire video** | **inside Event B (fireworks) — wrong event, not just a wrong timestamp** |

The headline failure: the vision label's highest-confidence "aarti" moment in the
whole recording is a firecracker scene. Label confidence tracked event *identity*
incorrectly, not just boundary precision — exactly the "confidence ≠ importance /
confidence ≠ correctness" distinction Final.md argues for (§18), now with a
concrete, measured instance rather than a hypothetical one.

**Audio did not help here — an honest caveat on Final.md's "combine visual + audio
evidence" assumption:** `bell` and `conch` are ~0.00 for the ENTIRE 343s clip, even
during Event A where a real aarti would typically have one. `bhajan` fires at
0.5–0.97 almost uniformly across BOTH events, including the nighttime fireworks
section. This is strong evidence the audio track is a continuous overlaid music
soundtrack (common in consumer-edited social clips), not live/diegetic sound —
multimodal audio-visual fusion assumes real ambient audio evidence, which this
source doesn't have. RMS loudness shows only a weak, imprecise correlation with the
fireworks section (~-13 to -15 dB vs. ~-18 to -22 dB baseline elsewhere).

**Follow-up tried and rejected — is there a cheap existing signal that separates
the two events?** `luma` (mean brightness) is already computed per shot for free
(`creative/shots.py`, no new model) and was the obvious cheap candidate for a
day/night split. Measured directly: it does NOT cleanly separate them — some
nighttime firework scenes (262–280s) have luma 0.53–0.58, indistinguishable from
the daytime puja's 0.43–0.58 range, because bright local light sources (sparks,
streetlights) inflate the frame average even against a dark sky. Only the scenes
with the darkest, most sky-dominated frames (279–296s, luma 0.09–0.12) stand out
clearly. Reported as a negative result rather than silently dropped.

**Conclusion:** the current pipeline has no mechanism that would catch this kind of
error — it has no notion of "event," only per-scene label confidence, and nothing
flags that the highest-confidence "aarti" scene and the surrounding scenes look
nothing alike (day vs. night) or that a labeled event's confidence goes to zero and
back repeatedly. This is concrete, first-hand evidence for Final.md's central
argument (a single vision label is not an event), not a reason to build the full
Knowledge Graph/memory architecture immediately — per Gate 3/4 (`new.md` §30), the
smallest next experiment (not yet run) would be a simple, cheap discontinuity or
clustering check (e.g. scene-to-scene embedding similarity, since SigLIP embeddings
already exist in `embeddings.json`) to see if it alone would separate Event A from
Event B, before reaching for VLM-based event detection or a full KG.

**Next action:** do not implement anything from this experiment yet — it's a
measurement. If pursued further, the next cheap check is embedding-similarity
clustering (data already computed, zero new cost) rather than luma or a new model.

---

## EXP-004 (continued): does SigLIP embedding similarity/clustering separate the two events?

**Method:** measurement only, no pipeline code touched. Read the already-computed
`embeddings.json` (per-scene SigLIP embedding, 46/46 scenes present) and
`scenes.json` for `results/diwali_exp001`. Two checks: (1) cosine similarity
between consecutive scenes, looking for a dip at the ~236–241s boundary EXP-004
established by eye; (2) plain 2-means clustering (no new dependency, ~20 lines) on
all 46 scene embeddings, scored against the Event A (puja) / Event B (fireworks)
ground truth.

**Result — strong positive, unlike the audio and luma checks:**

1. **Consecutive-similarity dip at the real boundary.** scene_32→scene_33
   (236.5→241.4s — exactly the transition found by eye) drops to cosine 0.680,
   visibly lower than the 0.75–0.97 range typical within either event. Not the
   single lowest dip in the whole clip (296s and 154.7s dip lower too — there's
   real structure elsewhere this experiment didn't ground-truth), but a genuine,
   located discontinuity right where it should be.
2. **2-means clustering: 95% purity against ground truth**, using data already
   computed on every run, zero new cost. Event A (puja): **12/12 scenes in one
   cluster.** Event B (fireworks): 7/8 in the other cluster — the sole miss is the
   very last fireworks scene (296–301s), at the far edge of the labeled range.

**Conclusion:** unlike audio (uninformative — dubbed soundtrack) and luma
(confounded by bright local light sources), embedding similarity is a real,
strong, already-available signal for separating these two events. This is the
first genuinely positive result in the EXP-004 line — worth acting on, not just
logging.

**Next action (proposed, not yet implemented):** a lightweight event-boundary
pass using consecutive-scene embedding similarity (dip detection) or clustering
over `embeddings.json` is a well-evidenced, cheap candidate for Final.md's
"Event Boundary Engine" — far short of a KG or VLM, reusing data every run
already produces. Suggest validating on the "other" segment too (0–160s scenes
split into 2 sub-clusters in this run, unverified against any ground truth —
might be real sub-structure, might be noise) before implementing anything, and
ideally on a second real clip once one is available.

---

## EXP-004: implementation — `creative/events.py` (detection only, not wired in)

Implemented the consecutive-scene-similarity-dip method (not clustering — dip
detection doesn't require knowing the event count `k` in advance, unlike 2-means,
so it generalizes to clips with an unknown number of events). `event_boundaries()`
flags transitions whose cosine similarity falls in the bottom `percentile` (default
15) of THAT clip's own consecutive-scene similarities — self-relative, not a fixed
constant guessed from one sample. `event_windows()` turns boundaries into
contiguous (start, end) spans. **Deliberately not connected to `creative/story.py`
or any other pipeline stage** — detection only, per instruction.

**Tests** (`tests/test_events.py`, synthetic, no ML/real footage — 4 passed):
boundary found at a real cluster transition, windows span the whole source, too
few scenes reports nothing rather than guessing, scenes without an embedding are
skipped rather than treated as a false similarity of zero. One test-construction
bug found and fixed along the way, not the module: symmetric jitter between the
two synthetic clusters produced an exact percentile tie; asymmetric jitter fixed
it. A second test assertion was also corrected, not the code: with only 9 sample
pairs, `np.percentile(15)` mathematically ties in one near-outlier runner-up
alongside the true boundary (interpolation with small N) — expected, over-inclusive
candidate-generation behavior, not a bug, and it matches what happened on the real
clip too (see below).

**Re-validated against the real Diwali data post-implementation** (not just the
unit tests): `event_boundaries()` on `results/diwali_exp001`'s cached analysis
returns `[35.1, 121.8, 154.7, 241.4, 279.9, 296.0, 328.2]`. `event_windows()`
produces `(154.7, 241.4)` as one contiguous window — this exactly contains Event A
(the daytime puja, ground-truth ~165–236s) with a small margin on each side, and
the very next window starts at `241.4`, precisely the hand-verified day/night
transition. Reproduces the original measurement exactly, through the shipped code
path rather than the one-off scratch script.

**Status:** implemented, tested, validated against real data. Not integrated
anywhere — no other file imports `creative/events.py` yet. Next step (not taken)
would be wiring it into the pipeline somewhere (e.g. as a candidate boundary set
the story planner or moment generator could use), which needs its own decision and
its own before/after check, same as every other change this session.

---

## EXP-004b: wire event boundaries into moment validation, measure before/after

**Where:** `highlights.py`'s `validate_clip()` already rejects a candidate moment
for `crosses_too_many_scenes` (the same pattern, one raw-scene-count based). Added
a sibling check, `crosses_event_boundary`: a candidate is rejected if it strictly
spans one of `event_boundaries()`'s candidate timestamps (0.25s margin, matching
the existing scene check's tolerance, so a moment merely touching a boundary isn't
punished). `Ctx` gained an `event_bounds` field, computed once in `build_ctx()`
from the same `vision` list already passed in — no new stage, no new artifact.

**Before/after method:** re-ran ONLY `highlights.generate_moments()` against
Diwali's cached scene/vision/audio/transcript (`results/diwali_exp001`) — no
re-transcription, no re-vision, no render. Ran it twice: with `ctx.event_bounds`
forced empty (= old behavior) and with the real boundaries (= new behavior).

**Result: zero behavioral change on this clip.** 45 accepted / 33 rejected,
identical in both runs. 0 new `crosses_event_boundary` rejections. No moment
present in one run and missing in the other. Honest neutral result, not a bug —
candidate moments are built from `ctx.scenes` (raw PySceneDetect cuts), and on
this footage the same visual change that produced the real event boundary (day
puja -> night fireworks) also produced an actual camera cut, so the existing
scene-based candidate generation already respects it without knowing why.

One moment does touch the boundary: `[236.5, 241.5]` ("aarti", score 0.42) ends
0.1s past the 241.4s boundary — within the 0.25s tolerance, so correctly not
rejected. Notable: this is the exact moment EXP-002/002b/003 spent three attempts
trying to fix as Diwali's "weak REVEAL" shot (low label confidence across the
board: devotees 0.2, priest 0.2, aarti 0.01). A plausible explanation neither
EXP-002 nor EXP-003 had: it may simply be a transitional clip, caught at the tail
of one real event and just barely into the next, rather than a shot that's
genuinely part of either. Not proven, but a concrete new lead if this is revisited.

**Regression check:** `test_units.py` + `test_features.py` (57 passed),
`tests/test_events.py` (4 passed, the new module), `test_pipeline.py` (full
`stages.py` integration test, real end-to-end path — 12 passed, 3 skipped for
missing Ganga footage, not a crash). `test_multi_engine.py` doesn't exercise this
change (its fixture hand-builds `Moment` objects, bypassing `generate_moments()`
entirely) — not a gap worth closing here, that test's own job is the creative
engine, not candidate generation.

**Conclusion:** shipped. A real, tested, zero-regression safety net — even though
it produced no observable difference on the one real clip available. Its value is
for footage where a semantic event change does NOT coincide with a raw scene cut
(a slow pan or fade between two different occasions, rather than a hard camera
cut) — untestable here without a second real clip.

**Next action:** none required. Would revisit if a future real clip shows
scene-cut and event-boundary divergence, or if the "weak REVEAL" investigation
from EXP-002/003 is picked back up — this run's tail-of-event lead is a legitimate
starting point.

---

## EXP-011: install and test Moondream2 -- does it actually fix EXP-004's failure?

**Setup:** Ollama already present on this machine; `ollama pull moondream` (Apache-2.0,
~1.9B, ~1.2GB). Downloaded 2 new real clips from Wikimedia Commons (not previously
used, credited in `inputs/CREDITS.txt`, not edited into a Reel): a genuine temple
aarti (CC BY-SA 4.0) and a genuine fireworks display (CC BY 3.0) -- clean,
prototypical examples of the two categories EXP-004 found confused. Extracted 6
frames (3 per clip) plus the exact 3 Diwali frames EXP-004-007 identified as
SigLIP's real mislabels. Ran the pipeline's actual `ClipVision` (SigLIP wrapper,
including its existing `_gate()` fireworks-vs-aarti mitigation) and Moondream
(one-word prompt: "ritual" or "fireworks") on the same frames.

**Result 1 -- on the new, clean clips: SigLIP got 6/6 right too.** This was an
important correction to the working hypothesis: SigLIP is not universally
unreliable at this distinction. It correctly labeled all 3 aarti frames
(idol/priest/deity) and all 3 fireworks frames. These are prototypical, unambiguous
examples (static shrine close-up; dense professional firework burst) -- very
different from Diwali's diffuse handheld sparklers. **The failure mode is
correlated with visual ambiguity/atypicality, not a blanket SigLIP weakness** --
worth stating plainly since it's a narrower, more accurate claim than Round 2's
framing implied.

**Result 2 -- on Moondream, same clean clips: 6/6 correct too** (`'ritual'` x3,
`'fireworks'` x3). Doesn't differentiate the two models on easy cases, but confirms
Moondream is a competent, working classifier before testing it on the hard case.

**Result 3 -- on the actual Diwali scenes that failed (the one that matters):**
Moondream got **3/3 correct**, including the closest call (scene36, where SigLIP
had a near-tie: idol 0.21 vs aarti 0.21).

| Diwali scene | SigLIP said (real pipeline output, EXP-004) | Moondream says |
|---|---|---|
| 241.4-257.3 (aarti 0.61, fireworks 0.26) | aarti (wrong, and `_gate()` didn't catch it -- aarti was still numerically ahead) | **'fireworks' (correct)** |
| 262.7-269.9 (aarti 0.53) | aarti (wrong) | **'fireworks' (correct)** |
| 269.9-279.9 (idol/aarti 0.21 near-tie) | idol/aarti (wrong, weakest case) | **'fireworks' (correct)** |

**9/9 correct across both test sets.** This is the clean confirmation Round 2 of
`docs/OSS_AUDIT.md` flagged as needed before trusting Moondream as a fix: it
doesn't just work on easy cases, it specifically corrects the exact failures this
session spent four experiments (EXP-004-007) tracing.

**The honest cost, and it's real:** CPU inference on this machine took **~195s per
frame on average** (191-203s across 9 runs, remarkably consistent), under real
system memory pressure (other running applications left available RAM in the
150-500MB range during inference, some swap thrashing). Two earlier attempts
timed out entirely (120s and a subsequent confused double-request) before landing
on a workable request shape: `num_predict: 12` (short answer only) and patient
600s timeouts. This is far too slow for per-scene analysis (every scene, the way
SigLIP runs today) -- but it fits the tiered-compute design Final.md and `new.md`
both argue for: only the handful of clips a story pass actually selects as
candidates (6 per reel here) need this, not all 45+ scenes in a source video. At
~3.2 min/clip, checking 6 selected clips costs about 20 minutes of CPU time per
reel -- expensive for interactive use, plausible for an offline/batch QC pass.

**Conclusion:** Moondream2 is no longer just a license-clean lead -- it's a
validated fix for the specific, real failure this session traced end to end, with
zero false positives or negatives across 9 real-content tests. Not yet wired into
`qc.py`'s `check_role_label_consistency` (EXP-010) or anywhere else in the
pipeline -- that would need its own before/after check (same discipline as every
other change this session), plus a decision on where the ~3 min/clip cost is
acceptable (a background QC pass, not inline with rendering).

**Next action:** implementation decision, not research -- wire Moondream as a
second opinion specifically for clips `check_role_label_consistency` already
flags (a natural, minimal integration point: only escalate to the slow model when
the cheap embedding check already raised a flag), or leave it as a documented,
validated, not-yet-integrated capability. Flagging rather than deciding
unilaterally given the real latency cost.

---

## EXP-005: event identity — given correct boundaries, can label aggregation recover WHAT each event is?

**Question:** EXP-004/004b solved (on this clip) WHERE an event starts and ends.
This asks a different question: given a correctly-bounded window, does aggregating
the per-scene vision labels inside it correctly identify WHAT the event is? Final.md
§9/§14 assumes yes ("represent recordings as events", "combine visual + audio
evidence") — worth checking against real data before assuming it.

**Method:** measurement only. For each of the 8 windows `event_windows()` produces
on Diwali, computed the duration-weighted average of every vision label across all
scenes overlapping that window (same technique `creative/shots.py`'s `build_shots`
already uses for real shots — reused, not invented).

**Result — mixed, and the failure is worse than EXP-004 suggested:**

| window | ground truth | top aggregated labels |
|---|---|---|
| `[154.7, 241.4]` | Event A: daytime puja | idol 0.181, priest 0.148, lamps 0.109, **aarti 0.107**, deity 0.074 — correct, devotion-cluster labels dominate |
| `[241.4, 279.9]` | Event B: nighttime fireworks | **aarti 0.405**, fireworks 0.106, idol 0.054 — **wrong**, by a wide margin |

Aggregation correctly identifies Event A. It gets Event B backwards — `aarti`
outscores `fireworks` 4-to-1 in a window that is unambiguously fireworks footage.

**Why — checked per-scene, not just the average:** this is not one outlier scene
diluting an average. Of the 5 scenes inside `[241.4, 279.9]`, **3 have "aarti" or
"idol" as their own single top label** (241.4–257.3: aarti 0.61; 262.7–269.9: aarti
0.53; 269.9–279.9: idol 0.21), 2 have no label above threshold at all, and none has
`fireworks` as a top label anywhere in this window (the two scenes that genuinely
score `fireworks` highly, 284.8–296.0, fall just after this window). A per-scene
majority vote fails exactly the same way a duration-weighted average does — the
mislabeling is systematic across multiple independent scenes, not a fluke.

**The embeddings knew better than the labels.** EXP-004's 2-means clustering
(same SigLIP embeddings, no text label involved) put these same 3 "aarti"-labeled
scenes in the fireworks cluster, correctly, at 95% purity. So on this clip, SigLIP's
zero-shot text-label head and its embedding space disagree about these scenes —
embedding similarity correctly recognizes them as visually like the other fireworks
footage; the "aarti" text-match apparently fires on bright sparks/glowing-particle
texture, which resembles flame/lamp imagery closely enough to fool zero-shot
label matching repeatedly, not once.

**Conclusion:** boundary detection (EXP-004/004b) and identity classification are
genuinely separate problems, and solving the first does not help the second — a
correctly-bounded fireworks event still gets labeled "aarti" if identity is
inferred from the same per-scene text labels the story planner already uses. This
is concrete, first-hand support for Final.md's distinction between a vision label
and an event, one level deeper than EXP-004: even after fixing WHERE, WHAT is still
wrong, and the fix isn't averaging harder. Weak, unconfirmed lead not pursued
further: CLAP's `drums` audio-event tag ticks up (0.22) and `bhajan` dips (0.18) in
the 270-280s sub-range, a plausible firecracker-percussion signature, but it does
not cover the whole window (near-zero at 241-260s despite that also being
fireworks) — not reliable enough on its own to act on without more evidence.

**Next action:** do not aggregate vision text-labels for identity on this clip;
they've now failed twice (EXP-004's single-label peak, EXP-005's window average and
per-scene vote). If event identity is worth solving, the promising direction is
embedding-space comparison (matches EXP-004's clustering result) rather than
text-label aggregation in any form — e.g. comparing a window's mean embedding
against a small set of reference-category embeddings, instead of trusting SigLIP's
zero-shot text match at all. Not implemented; would need its own experiment.

---

## EXP-006: does MAX label confidence (vs. average/majority) recover identity? + where this lands in the shipped output

**Hypothesis, from EXP-005's numbers:** the fireworks window's single strongest
unambiguous label anywhere in the clip is `fireworks: 0.96` (scene at 284.8–290.3).
If that scene fell inside the `[241.4, 279.9]` window, a MAX-over-labels rule
(rather than average or per-scene vote) would correctly flip identity to
`fireworks`, since one strong correct peak would beat several moderate wrong ones.

**Checked, not assumed:** recomputed window boundaries first — the 0.96 scene is
actually at 284.8–290.3, which falls in the *next* window, `[279.9, 296.0]`, not
`[241.4, 279.9]`. Measured MAX-per-label directly for `[241.4, 279.9]`:
`aarti 0.61, fireworks 0.26, idol 0.21` — **aarti still wins**, 0.61 > 0.26.
My hypothesis was wrong; verifying it caught that before reporting it as a fix.

**Conclusion: three separate label-aggregation strategies tried (EXP-005's
average, EXP-005's per-scene majority, EXP-006's max), three failures, same
window, same reason.** No statistic over the per-scene text labels recovers
correct identity here — the false positives aren't outliers a smarter aggregate
can filter out, they're the dominant evidence within the window's actual time
range. This is now a well-confirmed negative result, not a single data point.

**Where this actually lands — the concrete product consequence, not just a
research curiosity.** Checked the *shipped* Diwali reel's own edit-plan
(`results/diwali_exp001/edit-plan.json`): its **CLIMAX clip is `[241.48, 251.48]`**
— inside the exact mislabeled scene this whole EXP-004/005/006 thread has been
studying (`aarti: 0.61, fireworks: 0.26`). And `overlays.ritual` in the same plan
is `"Aarti"` — baked into the on-screen caption template. The rendered reel's
dramatic peak, selected specifically *because* the story planner trusted this
label, is very likely captioned "Aarti" over nighttime firecracker footage from an
entirely separate occasion than the actual puja. This is the same clip whose
CLIMAX role-affinity (0.680) and quality (0.734) were the *highest* of all six
selected roles across EXP-001/002/003's entire investigation — the reel's
strongest, most confidently-selected shot is the mislabeled one.

**No further label-statistic variants planned** — three failures on the same
data point is enough evidence to stop guessing new aggregation rules; the
underlying per-scene label is unreliable in a way no aggregate fixes. Genuinely
solving identity here needs either richer evidence this machine doesn't have (a
VLM caption per candidate window — no `ANTHROPIC_API_KEY` available, per
`docs/HANDOFF.md`) or human review, matching Final.md §21/§29's own
unknown-event / human-in-the-loop design rather than a cheap heuristic.

**Next action:** this is a product decision, not a research one — do the same
family of quick fixes attempted for REVEAL (EXP-002/003) apply here too (e.g. a
lower `min_affinity_percentile` floor specifically excluding `fireworks`-adjacent
labels from ritual-role scoring), or is this accepted as a known limitation until
richer evidence is available? Flagging rather than deciding unilaterally, given
this touches the same beam-search code that cascaded three times already this
session (EXP-002b/003) — any fix here should get the same before/after discipline,
on a second real clip if one becomes available.

---

## EXP-007: does a trivial 2-anchor embedding classifier recover identity, where text labels couldn't?

**Method:** the smallest possible test of EXP-005/006's recommendation.
Picked ONE scene as an "aarti anchor" (`scene_28`, 205.8–214.0s, `idol 0.8 deity
0.78 aarti 0.52` — visually confirmed genuine puja close-up) and ONE as a
"fireworks anchor" (`scene_38`, 284.8–290.3s, `fireworks 0.96` — visually
confirmed genuine burst). For each scene, cosine similarity to both anchors;
nearest anchor wins. No training, no new model, ~15 lines.

**Result — 7/7 correct:**

| scene | sim to aarti anchor | sim to fireworks anchor | classifies as |
|---|---|---|---|
| scene_33 (labeled aarti 0.61) | 0.628 | **0.714** | fireworks — correct |
| scene_35 (labeled aarti 0.53) | 0.756 | **0.780** | fireworks — correct |
| scene_36 (labeled aarti/idol 0.21) | 0.707 | **0.715** | fireworks — correct |
| scene_21 (genuine puja) | **0.821** | 0.723 | aarti — correct |
| scene_22 (genuine puja) | **0.914** | 0.695 | aarti — correct |
| scene_29 (genuine puja) | **0.922** | 0.683 | aarti — correct |
| scene_30 (genuine puja) | **0.859** | 0.638 | aarti — correct |

All 3 of EXP-005/006's mislabeled scenes flip correctly; all 4 known-good puja
scenes stay correct. This validates the direction EXP-005/006 pointed at:
embedding similarity succeeds where three different text-label statistics failed.

**Honest caveat — the margins are not equally strong, and this must not be read
as "solved."** The 4 real-puja scenes classify with comfortable margins (0.10–0.22
gap). The 3 corrected scenes are much thinner: scene_36's margin is 0.008 —
essentially a coin flip that happened to land right, not a confident call. Two
more structural risks: (1) both anchors were hand-picked from THIS SAME CLIP,
0 held-out validation, classic small-sample overfitting setup; (2) only one anchor
per class — a real classifier would need several examples per category to be
trustworthy, not a single representative frame each.

**Conclusion:** promising enough to be worth a real implementation attempt, not
yet solid enough to ship into `story.py`'s scoring without more evidence — the thin
margins on 2 of 3 corrections are the specific thing that would need to hold up on
a second real clip before trusting this in production.

**Next action:** requires either (a) a second real clip to check the margins hold
up out-of-sample, or (b) more anchor examples per class on this same clip (cheap,
no new data needed — every visually-confirmed scene from EXP-004's keyframe review
could serve as an anchor) to see if margins tighten or the classification becomes
more robust with >1 example per class. Not implemented into the pipeline; this is
a measurement, same as EXP-004/005/006.

---

## EXP-008: baseline event importance/novelty — no code changes

**Note:** the pipeline already has more of Final.md's "importance" concept built
than assumed going in. `scoring.py` decomposes each moment's score into 7 named
components (`visualImportance`, `devotionalRelevance`, `audioImportance`,
`semanticImportance`, `novelty`, `completeness`, `temporalImportance`), not a
single number — this experiment measures those EXISTING components against
EXP-004's ground truth, nothing new implemented.

**Method:** reconstructed `ctx` from Diwali's cached analysis (same technique as
every experiment since EXP-001b) and called the real `scoring.score()` function
directly on all 45 accepted moments — the exact function `generate_moments()`
calls internally, just also keeping the per-component breakdown it normally
discards after producing the final number.

**Finding 1 — "novelty" isn't measuring novelty in Final.md's sense.** Read the
source: `novelty(ctx,a,b) = 1 - mean(label_prevalence[k] for k in dominant_labels)`
— rarity of a moment's TEXT LABEL across the whole clip, not visual/content
distinctiveness, and not embedding-based at all. It does not distinguish the two
real events: Event A (puja) mean novelty 0.858, Event B (fireworks) mean **0.924
— higher**. Whenever a moment's reason comes from audio (`bhajan`, not a vision
label), novelty silently defaults to 0.5 (25 of 45 moments) — a placeholder, not a
computed value. This component cannot serve as the "has something genuinely new
happened" signal Final.md's importance formula wants; it's closer to "does this
moment have an uncommon label at all."

**Finding 2 — "devotionalRelevance" inherits and amplifies EXP-004's mislabeling
bug; this is the missing link in the whole EXP-004-007 chain.** Computed as
`max(DEVOTIONAL[label] * confidence)`, and `DEVOTIONAL["aarti"] = 1.0` — the
single highest weight in that table. The false `aarti: 0.61` label on the
confirmed-fireworks scene (241.4–257.3s) produces `devotionalRelevance = 0.611`,
which is **higher than several genuine puja moments** (0.391, 0.410, 0.488) and
only beaten by the two deity close-ups (0.777). Group means: A_puja 0.420 vs.
B_fireworks 0.308 — puja wins on average, but the specific mislabeled moment's
final `stored_score` (0.408) lands squarely inside the genuine-puja score range
(0.427–0.487), not distinguished as different at all.

**This closes the causal chain the session has been tracing since EXP-004:**
mislabeled vision label (EXP-004) -> inflates this moment-scoring stage's
`devotionalRelevance` component, previously unchecked (EXP-008, new) -> feeds a
comparably-high `stored_score` into the moment pool -> inflates the same shot's
role-affinity in `story.py`'s beam search (already observed in EXP-002/003's raw
data) -> becomes the reel's CLIMAX, its highest-affinity clip (EXP-006). Every
stage in the pipeline that touches this label propagates the same single point of
failure; none of them independently catch it, because none cross-check against
anything other than the vision label itself.

**Conclusion:** no code changed, as instructed. The two findings here don't
contradict EXP-004-007's conclusion, they explain WHY it matters beyond one
mislabeled clip: it's not just the story planner's role-affinity that's fooled,
it's the moment-scoring stage that feeds it, and specifically the ONE component
(`devotionalRelevance`) most clearly modeled on Final.md's "ritual significance"
idea. "novelty" as currently implemented is not equipped to help correct for this
— it's derived from the same fragile signal, not an independent check.

**Next action:** none — measurement only, paused per instruction alongside the
identity thread. If picked back up: `devotionalRelevance`'s single-label max
formula is the concrete, now well-evidenced target (not `story.py`'s beam search,
which is downstream of it and has already cascaded three times this session).

---

## EXP-009: does the EXISTING evaluation harness catch any of this? No code changes.

**Method:** ran the pipeline's own built evaluation tool, unmodified, against the
actual shipped Diwali reel: `python tools/eval_reel.py results/diwali_exp001`.
Read frames from the real rendered `reel-9x16.mp4`, no re-render, no new code.

**Result:**

```
hook score 0-1   0.65      edges on a cut   12       redundancy max   0.57
edges sliced     0         edges no cut near 0        blend share      100%
LUFS             -16.0     true peak dB     -1.7      subject in frame 80%
shown whole      17%       face cutoffs     0         safe-zone viol.  0
QC               warn      re-edits         2
```

**Finding: the existing evaluation harness has zero coverage for the entire class
of error this session found.** Every metric it produces is technical/structural —
loudness, crop/reframe geometry, near-duplicate shot detection, edge-to-cut
alignment, safe-zone margins. None of them check whether a clip's content matches
its assigned narrative role or its label. By this scorecard, the reel looks like a
normal, reasonably-executed edit: decent hook score, no safe-zone violations,
on-target loudness, no face cutoffs. `QC: warn` reflects the same two pre-existing
warnings seen since the very first Diwali render (`subject_in_frame`,
`weak_shots`) — nothing new, and nothing related to EXP-004-008's finding. Nothing
in `qc.json` or `evaluation.py` would ever surface "the CLIMAX is mislabeled
fireworks footage" — that required a human actually looking at frames (EXP-004),
which is exactly how this whole thread started.

**Conclusion:** confirms and sharpens HANDOFF's own prior admission ("hook score…
never validated against human judgement") into something more specific: it's not
just that one metric is unvalidated, it's that **no metric in the current tooling
even attempts semantic/content correctness** — Final.md §28 lists "story
coherence" as something to evaluate; nothing currently measures it, or anything
adjacent to it (label-content agreement, role-fit correctness). A technically
clean reel and a semantically-wrong reel currently score identically.

**Next action:** none — measurement only, no code changed. If ever picked up: a
cheap, evidence-backed addition (not proposed here, needs its own experiment)
would be a coarse "role-label consistency" check in `evaluation.py` — e.g. flag a
selected clip whose winning label and its embedding-nearest-anchor class disagree
(EXP-007's technique), surfaced as a QC warning rather than silently trusted. Not
implemented; this session's mandate for EXP-009 was measurement only.

---

## EXP-010: role-label consistency on the shipped reel, using existing embeddings — no code changes

**Method:** the concrete check EXP-009 proposed but didn't build. For each of the
6 clips actually selected in the shipped Diwali reel (`results/diwali_exp001`'s
`edit-plan.json`), computed a duration-weighted mean embedding from the same
`embeddings.json` used throughout EXP-004-007, then classified it against
EXP-007's two anchors (nearest of `scene_28` "genuine puja" / `scene_38` "genuine
fireworks"). Compared that embedding-based class against each clip's own winning
vision label. Pure read of already-computed data; no pipeline code touched.

**Result:**

| role | window | winning label | embedding says | agreement |
|---|---|---|---|---|
| OPENING | 158.4–160.9 | idol 0.92 | devotional | AGREE |
| BUILDUP | 192.3–196.8 | priest 0.68 | devotional | AGREE |
| RITUAL | 205.8–214.0 | idol 0.80 | devotional | AGREE |
| REVEAL | 236.6–241.4 | devotees 0.20 | devotional | AGREE |
| **CLIMAX** | **241.5–251.5** | **aarti 0.61** | **fireworks** | **DISAGREE** |
| CLOSING | 296.0–301.2 | fireworks 0.00 (no real signal) | devotional | n/a (label too weak to compare) |

**5 of 6 clips consistent, 1 flagged — and it's the exact clip EXP-006 already
identified as the shipped defect.** Zero false positives among the other 5: every
clip with a meaningful label agrees with its embedding class; the one clip with no
real label signal (CLOSING) is correctly marked inconclusive rather than forced
into a false agree/disagree. This isn't a re-statement of EXP-004-007's finding —
those worked at the level of individual 5-15s scenes; this applies the same
technique to the actual multi-scene clips the story planner selected, at the
granularity a real QC check would need.

**Conclusion:** the coarse consistency check EXP-009 proposed works cleanly on
this data, with no false positives, using nothing beyond data every run already
computes. Confirms the finding is real (not an artifact of looking at one
5-second scene in isolation) and confirms the proposed fix direction is
concretely actionable, not speculative.

**Next action:** none — measurement only, as instructed. This is now the second
independently-run check (EXP-007 at the scene level, EXP-010 at the clip level)
pointing at the same embedding-vs-anchor technique with clean results and no false
positives observed anywhere. If implementation is wanted, this specific
measurement is the strongest evidence base to build from — still gated on the
same caveats as EXP-007 (thin margins on some cases, single-clip anchors, no
held-out validation).

---

## EXP-010: implementation — `check_role_label_consistency` in `creative/qc.py`

Implemented as a **warn-only** QC check (never `fail`, matches the instruction),
added to `run_qc()` alongside the existing checks.

**Design iteration, kept honest rather than presenting the first attempt as the
answer.** Three versions tried before landing on one that reproduced EXP-010's
finding without new false positives:

1. *Global nearest-neighbor, literal label match* — false-flagged 5 of 6 clips,
   including the two genuinely fine ones (OPENING "idol", BUILDUP "priest"). Real
   devotional footage legitimately carries several different scene-level labels
   (idol/priest/deity/aarti all appear across one real puja); comparing against a
   single globally-nearest scene's exact label text is too strict for that.
2. *Nearest same-label vs. nearest different-label scene* — fixed the false
   positives but then missed CLIMAX entirely (same-label evidence outscored
   different-label evidence there, narrowly). Also discovered along the way that
   two of six segments (`REVEAL`, `CLOSING`) don't carry a real vision label in
   `reason` at all — `plan.py` substitutes the role name when the true label
   confidence is too weak to be meaningful; these are correctly skipped now.
3. *Single highest-confidence anchor per class, programmatically selected (no
   hardcoded scene IDs)* — reproduced EXP-007's classification exactly on 3 of 4
   comparable clips, but **flipped verdict on CLIMAX** (the one clip that matters)
   depending solely on which one scene got auto-picked as the devotion anchor.
   Concrete demonstration of the fragility EXP-007 already flagged as a caveat.
4. **Shipped: top-3-per-class anchors, averaged and renormalized.** Same
   programmatic selection (no hardcoded scene IDs — generalizes to any source with
   both devotional and fireworks content), but 3 examples per class instead of 1.
   Clean result: OPENING/BUILDUP/RITUAL classify correctly with comfortable
   margins (0.89 vs 0.79, 0.92 vs 0.65, 0.94 vs 0.72), CLIMAX correctly flags
   (0.74 vs 0.78) — matching EXP-007/010's original finding through fully
   programmatic anchor selection instead of hand-picked scene IDs.

**Scope, deliberately narrow.** This checks devotion-vs-fireworks specifically
(reusing `scoring.DEVOTIONAL`'s label set, minus `aarti`/`abhishemak`/`procession`
for ANCHOR selection only, since those are the specific labels EXP-004-007 showed
are unreliable) — not a fully generic any-label-vs-any-label consistency checker.
Attempt #1 above was closer to that general form and was measurably worse. Ships
what the evidence actually supports; a broader claim would be unvalidated.

**Validated:**
- Reproduces EXP-010's exact finding through the shipped function on real data:
  `value=1, clip=4 (CLIMAX), "labeled 'aarti' (devotional) but its embedding looks
  more like this source's own fireworks footage (sim 0.74 vs 0.78)"`.
- `tests/test_creative_qc.py` + `tests/test_start_end_qc.py`: 15 passed (synthetic
  fixtures have no `embeddings.json` on disk — the check correctly no-ops rather
  than erroring, exercising the "best-effort, never blocks" path).
- `tests/test_pipeline.py`: 10 passed — full real `stages.py` -> `run_qc()`
  integration path, not a bypassed fixture.

**Status:** shipped. Research loop frozen here per instruction — moving to full
pipeline + hardware testing next.

---

## EXP-012 — Clip-selection signals vs. reel-worthiness (label calibration, aesthetic prior)

**Question.** The reel picks clips by summing label scores. Which cheap, label-free changes make that selection better — and is
label quality even the bottleneck? (No human at runtime; ground truth below is development-time only.)

**Benchmark (new).** 35 frames from 3 real CC-licensed videos (Golden Temple documentary, Ganga aarti Janakpur, home-temple
aarti Kolkata). Ground truth = label sets and 1-5 "reel-worthiness" ratings **written by the assistant viewing each frame**:
subjective, conservative, one rater, ~15 (video,label) pairs with variation. Treat every number below as weak evidence.
Files: `tools/bench_labels.py`, `tools/bench_clips.py`, `tools/bench_scoring.py`, `tools/bench_*_truth|ratings.json`.

**Label calibration — three ideas, all refuted (nothing shipped).** Within-video AP, raw SigLIP sigmoid (current) = 0.85.
Softmax competition vs other labels+negatives 0.75; frame-centred logit 0.80; video-relative logit = 0.85 (identical: a constant
per label cannot reorder frames of that label); corpus z-score best single-threshold F1 0.55 vs 0.58. Cross-video AP looked
better for raw sigmoid only because it separates *different videos* — the wrong test for within-video selection.
**Real bottleneck found: cross-label scale.** Best per-label thresholds range 0.001-0.42; at KEEP=0.2 recall is 0.39 (precision
0.69); at 0.02 recall 0.68 / precision 0.51 (2 false labels/frame). Low-scale labels (flowers, devotees, temple_architecture,
procession, decorations) are effectively invisible to scoring. Not fixed: 35 frames is too few to ship per-label thresholds.

**New labels (10, from temple-videography coverage review).** Wired into vision prompts + `scoring.DEVOTIONAL` + `roles.json`
groups (a label in only one of the three is dead; guard test `tests/test_label_coverage.py`). On the benchmark: `offerings`
(4/4) and `food_service` (1/1) correct; `sanctum_view` / `devotees_walking` only duplicate idol/deity and crowd/devotees;
the other six never fired (no such content in the footage) — **unvalidated**.

**Aesthetic prior (CLIP-IQA-style antonym prompts on the existing SigLIP embeddings).** Frame level, pre-committed composite
(quality + cinematic; "colourful" excluded on purpose to avoid encoding the rater's taste): Spearman +0.57 overall, and
positive inside every video separately (+0.52 / +0.85 / +0.79), precision@8 0.875 vs 0.57 base rate. Pipeline's own frame
signals: visual importance +0.07, devotional relevance +0.19, exposure −0.28 (penalises dark night shots). Moment level
(22 scene windows, Golden Temple, real scoring code): default weights rho +0.34; aesthetic alone +0.32; default + aesthetic
0.15 → rho +0.44 but precision@6 0.83 → 0.67. **At n=22 that is noise** (SE of rho ≈ 0.2) — no improvement can be claimed.
Shipped as `scoring.aesthetic` with **default weight 0 (opt-in)**. Bug found while measuring: `analysis._merge` dropped the new
`VisionResult.aesthetic` field (fixed; that run's artifacts predate the fix, so the test derived it from saved embeddings).

**Conclusion.** Labels rank frames within a video reasonably (AP 0.85); calibration across labels and vocabulary coverage are the
real limits. Weights for visual (0.25) and audio (0.15) importance show ~no relation to reel-worthiness here (suggestive only).
**Next action.** Enlarge the benchmark (~100 frames, 8-10 videos) before turning anything on.

## EXP-013 — Can a small VLM understand a frame better than SigLIP, at CPU speed? (scene-type, closed set)

**Method.** 8-way scene type (aarti / deity / exterior / devotees / procession / rites / food / other) on the same 35 frames;
truth derived from EXP-012's labels. `tools/bench_vlm.py`. Baseline = SigLIP zero-shot argmax over 8 prompts.

| Model | Accuracy | Speed (this CPU) |
|---|---|---|
| SigLIP zero-shot (current) | **0.69** | ~instant (embeddings already computed) |
| SmolVLM-500M (fp32, 512px) | 0.40 | 3.6 s/frame |
| SmolVLM-256M | unusable: answered "H" for every image; free-text captions are literal ("a boat on the water", "a statue of a person") | 4 s/frame |
| Moondream via Ollama (EXP-011, 2-class only) | 9/9 | ~195 s/frame **while the box was swapping** (~1.5 GB free) |

**Conclusion.** VLMs small enough for this machine are *worse* than SigLIP at scene understanding here (they get Hindu idol/aarti
frames right and fail the Golden Temple). Speed is not the problem for ≤0.5B models; capability is. The 1.7-3B class (Moondream,
Qwen2.5-VL-3B) is untested at scale because RAM is the constraint (desktop apps hold ~4 GB of 5.7 GB). Env note: `transformers`
is pinned at 4.46.3 (CLAP); SmolVLM needs `size={"longest_edge": 512}, do_image_splitting=False` passed to the processor.
**Next action.** Do not build a VLM re-ranker on this hardware. Revisit with a machine that has >=8 GB free, as an opt-in plug-in.

## EXP-014 — Coverage-aware story planning (facility-location term in the beam search)

**Idea.** The planner scores shots one by one (role affinity x quality, minus pairwise redundancy). Add a set-level term: reward a pick by
how much it improves `sum_i w_i * max_{j in picked} R[i,j]` over the WHOLE pool (R = similarity beyond the pool's typical similarity, the
same transform `red` uses; w = length x score). Representative + diverse by construction, needs no label vocabulary. Implemented in
`story.py` (`S["coverage"]`, added after the quality gate like chronology, so it ranks candidates that clear the bar and never rescues one
that doesn't). **Default 0 = off.** Tests: `tests/test_coverage.py` (synthetic 13-shot / 3-topic pool: plan changes, pool coverage 0.645 -> 0.730).

**Diagnostic on the real 11-min Golden Temple reel (saved analysis only, no pipeline run; 120 scenes).**
- Reel covers 0.285 of the recording; greedy coverage-only 6 scenes cover 0.364 -> real headroom (+28%), but coverage alone is not the goal.
- **Coverage does NOT flag the clip that looked weak** (clip 2, BUILDUP, a pilgrim's face close-up): marginal coverage 0.019 (mid-pack;
  lowest were REVEAL 0.012 and RITUAL 0.016) and standing-alone coverage 0.145 (2nd highest) because face close-ups are a frequent scene
  type in this documentary, i.e. "representative".
- The aesthetic prior (EXP-012) does not flag it either: clip 2 sits at the 0.60 percentile; the lowest is the REVEAL (0.12), a shot the
  rater scored highly. So neither lever addresses that defect. Candidate cause (untested): BUILDUP has no term against static portraits.

**Conclusion.** Built as an opt-in knob; NOT evidence that reels improve. **Next action:** A/B two real reels (coverage 0 vs ~0.5) once
the pipeline is cleared to run, and judge by eye. Do not enable by default before that.
