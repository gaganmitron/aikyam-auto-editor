# Aikyam Creative Editing — Research Report (September 2026)

**Goal:** decide what to build (and what NOT to build) so that the generated Reel looks and sounds good. Recommendations are ranked by expected effect on Reel quality, not novelty.

**Evidence levels used below:** **[P]** primary page read (licence text, repo README, standard); **[A]** paper abstract/arXiv page only (full text not read: results not independently checked); **[B]** practitioner/blog/guide (useful, not authoritative). Nothing here was benchmarked by us unless stated.

---

## 1. Where Aikyam is today vs. the new brief
Already built (earlier work, tested on the Ganga aarti clip + synthetic media): shot/scene analysis, SigLIP vision, CLAP audio tags, moment ranking, story arc (opening→closing), per-edge transitions, J/L-cuts, numpy audio mixer with stems (ducking, level matching, BS.1770 loudness, true-peak limiter), music tempo/energy analysis and cut-to-beat alignment, continuous 9:16 layout (crop / blurred fill / push-in), post-render QC with deterministic re-edit, publish gate.

**Gaps against the new brief (this is the real work):**
| Gap | Today | Needed |
|---|---|---|
| Inputs | one video | many videos + images + optional user music, mixed |
| EditPlan | one `source.path`, segments are time ranges | `assets[]`; segments reference `assetId`; image segments with motion |
| Images | none | timeline elements, Ken Burns, subject-aware, not slideshow-like |
| Opening | "establishing shot" first | evidence says the first 2–3 s must hook (see 2.5) |
| Reframing | one smoothed x-path; deity/priest priority is only *implied* by face/motion | explicit subject priority, camera modes, no cut-off faces/deities |
| Captions | 14–17% bottom margin | platform UI zones are ~320–500 px of 1920 (see 2.8): current margin is too small |
| Music | our own tempo tracker (fine on synthetic tracks) | robust beats/downbeats on real music; bar-aware alignment; loop/trim; user-supplied music |
| QC | 11 checks | + subject cut-off, aspect ratio, render-corruption decode, timestamp validity |
| Remotion | ffmpeg-only for creative plans | parity or explicit unsupported (+ licence decision, see 3) |

---

## 2. Findings by topic → decision

### 2.1 Highlight detection / moment retrieval / editing (papers)
- The DETR family (Moment-DETR, QD-DETR, UMT, CG-DETR, TR-DETR, UVCOM) and newer FlashVTG / VideoLights are **query-conditioned** and trained on QVHighlights-style data (vlogs/news) [A]. Aikyam has no text query and no labelled temple data. **Decision: do not adopt these models.** Their useful idea is a *joint saliency + boundary* objective; we already get boundaries from shot/audio structure and validate against real media.
- **Audio matters in highlights, in two ways** — semantic (what sound) and spectro-temporal dynamics (transients): *Sounding Highlights* (ICASSP 2026) argues existing models under-use audio [A]. **This matches our design** (CLAP = semantic, onset/flux = dynamics) and supports keeping both; it is not a reason to change architecture.
- **Elastic music alignment** (BEAT, 2026): professional editors do *not* map one shot per beat; fast cuts go with high-energy bars and sustained shots span quiet bars; they use an energy-adaptive dynamic programme over bars [A]. **Decision: replace our greedy nudge with a bar-level DP that trades beat error against shot-length deviation, allowing many-to-one.** This directly implements "don't force every cut onto a beat".
- **Hierarchical planning** (DIRECT: Screenwriter → Director → Editor; LAVE: LLM sequencing/trimming) [A]. Confirms our staged design (story → edit directives → render). **Decision: keep the deterministic core; LLM stays optional and validated** (already true).
- **Long-video temporal retrieval** (Vidi, ByteDance): strong on hour-long video, text+vision+audio [A]. **Its `LICENSE.txt` is CC BY-NC 4.0 [P]** → cannot be used in a commercial product. **Decision: do not depend on it.**
- **Generative beat-retiming** (MVAA, 2025): inserts keyframes at beats and diffuses in-between frames; ~10 min GPU fine-tune per video on CogVideoX-5B [A]. **Decision: do not adopt** — it synthesises frames (unacceptable for devotional footage) and is far too costly.

### 2.2 Open-source systems (ideas, not copying)
- **PySceneDetect** (BSD-3) [P/B]: `AdaptiveDetector` uses a rolling-average threshold, which reduces false cuts under fast camera motion [P]. We use `ContentDetector`. **Cheap upgrade for handheld temple footage; test it.**
- **Auto-Editor** (Unlicense) [P/B]: cuts by audio loudness or motion, exports timelines. We already compute both signals; nothing to adopt beyond validating our thresholds are adaptive.
- **OpenCut** (MIT, TS/Rust, editor API + MCP server for agents) [B]: the useful idea is *operations on a timeline exposed as an API*. Our EditPlan already is that; no adoption.
- **Remotion** — **licence [P]:** free for individuals, for-profit orgs **with up to 3 employees**, non-profits and evaluation; **any other organisation needs a paid Company License.** **Decision: keep FFmpeg as the default renderer; treat Remotion as optional and get a business decision on the licence before shipping it.**
- **AutoFlip (MediaPipe, Apache-2.0)** [P]: shot detection by colour histogram; per-shot camera mode **stationary / panning / tracking**; the camera path is a **least-squares low-degree polynomial fit** to the salient boxes; **too-wide content is padded with a blurred copy of the frame**. **We already do the blur fill. Adopt the camera-mode choice and the polynomial path** (ours always moves slightly via EMA; a static subject should get a static camera).

### 2.3 Beat / rhythm / structure
- **Beat This!** (ISMIR 2024; **MIT code and weights** [P]): beats **and downbeats**, no DBN post-processing (which fails on tempo changes / unusual meters), CPU-capable, checkpoints 78 MB / 8 MB. **Adopt for real music** (our hand-rolled tracker is fine only on clean synthetic tracks, and has no downbeats).
- **madmom**: code BSD but **pretrained models CC BY-NC-SA [P]** → do not use commercially.
- **Music structure** (All-In-One, SongFormer 2025) [A]: functional segmentation (intro/verse/chorus). For 25–40 s reels, an **energy-novelty curve + the downbeat grid** should suffice; test before adding a structure model.
- Practitioner consensus [B]: use sync to emphasise key moments; **over-syncing is a known failure**. Consistent with BEAT's elastic idea.

### 2.4 Reframing 16:9 → 9:16
See AutoFlip above. Additional decisions: (a) explicit subject priority **deity → priest/person → speaker → ritual/action → object** needs *localisation* of the deity/priest, which our image-level SigLIP labels do not provide — an **open-vocabulary detector** (OWL-ViT/OWLv2, Grounding DINO — *licences not verified*) is the candidate; (b) camera mode per shot; (c) polynomial/L2 path; (d) QC to verify no face/deity is cut by the frame edge.

### 2.5 Short-form editing principles [B]
- Viewers decide in roughly **2–3 s**; the hook belongs in the first seconds. **Our default opening is an establishing wide shot — likely a weak hook. Decision: default to a hook-first opening (strongest visual/ritual moment), establishing shot second; keep "establish" as a profile; A/B test.**
- Shot length: short-form fast content ~1–2 s/shot, talking-head 4–8 s; our devotional profiles (2.5–8 s) are a deliberate genre choice — keep, but measure against the distribution.
- **Dissolves slow pacing; use them for mood/time shifts. Cuts are the default; match action/eyeline for continuity.** Our per-edge decision already follows this; add a cap on the *share* of blends (target ≤ ~40% of edges) and log it.

### 2.6 Audio post-production
- **EBU R128: −23 LUFS ±0.5 LU, true peak ≤ −1 dBTP [P]**; YouTube prefers **−14 LUFS** [B]; platforms normalise, so "louder" buys nothing. Our **−16 LUFS / −1.5 dBTP ceiling** is defensible; make the target configurable per platform and **verify Reels' normalisation behaviour** (not verified).
- Speech-to-background: **minimum speech–background loudness difference of 4 LU (DPP; older Netflix)**, dialogue **5–8 LU above bed music** for small speakers, music commonly ~10 dB under speech [B]. **Our 8 dB (ambience) / 16 dB (chant/speech) margins and QC floor of 12 dB are stricter than broadcast practice.** For chanting that is defensible, but 16 dB may make the music inaudible. **Decision: keep the never-overpower guarantee; tune the chant margin by listening tests in the 10–14 dB range; keep QC floor ≥ 8 LU.**
- Crossfades: equal-power for uncorrelated sources (already done); J/L-cuts (done).

### 2.7 Images
- **2D Ken Burns** (subject-aware pan/zoom) is cheap and safe. **3D Ken Burns / parallax** (Niklaus et al., 2019) needs depth + inpainting [P/A]; **Depth Anything V2: only the Small (24.8 M) model is Apache-2.0; Base/Large/Giant are CC-BY-NC [P]**. Geometric warping of sacred imagery risks visible artefacts.
- **Decision:** ship 2D subject-aware Ken Burns first (start wide → end on the subject, eased, direction chosen from the subject, never a constant-speed drift); avoid slideshow feel by mixing images with video, varying duration by content, cutting on beat. Parallax later, Small model only, behind a QC check.

### 2.9 Captions (safe zones, timing, scripts)
- **Platform UI zones [B, sources disagree]:** Reels ~**400–500 px** at the bottom (one guide: 420 px bottom / 220 px top); TikTok **320 px bottom, 108 top, 60 left, 120 right**; a universal safe box of **900×1160 px, centred, on 1080×1920** is widely cited. **Our bottom margin is 17% (≈327 px) and QC allows down to 86% of the height (≈269 px from the bottom): too little.** **Decision: bottom ≥ 22% (≈420 px), top ≥ 12%, sides ≥ 8%; verify in QC.**
- **Netflix timed-text norms [B]:** ≤ 42 chars/line, ≤ 2 lines, **20 cps (adult, English)**, min ~5/6 s, max 7 s. Our 20 cps limit matches. **For Indic scripts, count grapheme clusters, not code points** (conjuncts must not be split; Unicode 15.1 changed cluster rules for virama sequences [B]) — our current `len(text)` overstates the rate.
- **Shaping:** libass + HarfBuzz handles Indic scripts (we already render Kannada/Hindi/Tamil/Telugu and QC-measure ink through libass) — keep; a HarfBuzz/Noto-Kannada issue is reported in the wild [B], so keep the render-based overflow check.
- **Word timing:** WhisperX (BSD-2) [P] adds wav2vec2 forced alignment; **its default alignment languages are en/fr/de/es/it, Indic not confirmed [P]**. We use faster-whisper word timestamps. **Evaluate on Kannada/Hindi before switching.**
- **Avoid covering faces/deities:** choose the top or bottom safe zone per clip from the tracked subject box; do not just pin captions to the bottom.

---

## 3. Comparison tables

### 3.1 Open-source projects / models
| Project | Take | Adopt | Do NOT adopt | Cost | Licence [P unless marked] | Difficulty |
|---|---|---|---|---|---|---|
| PySceneDetect | shot cuts | `AdaptiveDetector` (rolling threshold) | — | CPU, decode-bound | BSD-3 [B] | Low |
| Auto-Editor | silence/motion cutting | validate our adaptive thresholds | its timeline export | CPU | Unlicense [B] | — |
| OpenCut | web editor + agent API | "timeline as API" idea (we have EditPlan) | the editor | — | MIT [B] | — |
| Remotion | React video rendering | optional richer captions/layouts | as default renderer | 15× slower than ffmpeg (measured) | free ≤3-person for-profit / non-profit / individual; **else paid** | Medium |
| AutoFlip (MediaPipe) | reframing | camera modes, polynomial path, blur pad | its detectors | CPU/GPU | Apache-2.0 | Medium |
| Beat This! | beats + downbeats | **yes** (real music) | — | 78 MB/8 MB, CPU ok | **MIT** | Low |
| madmom | beats | — | **models are CC BY-NC-SA** | — | code BSD / models NC | — |
| WhisperX | word timing | test on kn/hi | switching blindly | GPU-friendly | BSD-2 | Medium |
| Depth Anything V2 | depth for parallax | **Small only**, later | Base/Large/Giant | 24.8 M params | Small Apache-2.0; others **NC** | Medium |
| Vidi | temporal retrieval LMM | — | **use in product** | 7–9 B LMM (GPU heavy) | **CC BY-NC 4.0** (repo LICENSE) | — |

### 3.2 Papers
| Paper (year) [A] | Idea | Adopt? | Cost / code | Difficulty |
|---|---|---|---|---|
| BEAT: rhythm-elastic alignment (2026) | bar-level DP, many-to-one shots↔bars, energy-adaptive | **Yes: DP alignment** (not the learned encoder) | code not stated | Medium |
| DIRECT (2026) | Screenwriter/Director/Editor agents, mashup coherency | staging idea only | LLM calls; repo exists (licence unverified) | High if agentic |
| Sounding Highlights (ICASSP 2026) | audio semantics + dynamics both matter | validates CLAP + onsets | code not stated | — |
| VideoLights (2024) | joint HD+MR, cross-modal fusion | No (needs supervised data) | code+ckpt released | High |
| TR-DETR / CG-DETR / FlashVTG (2024–25) | query-conditioned DETR MR/HD | No | need QVHighlights-style training | High |
| Vidi (2025) | hour-long temporal retrieval | No (NC licence) | 7–9 B LMM | — |
| MVAA (2025) | generative beat retiming | No | ~10 min GPU/video, CogVideoX-5B | High |
| LAVE (2024) | LLM clip sequencing/trimming | optional LLM director (validated) | API calls | Low–Med |
| 3D Ken Burns (2019, background) | depth parallax + inpainting | later, Small depth model | GPU | High |

---

## 4. Architecture recommendations
1. **EditPlan v3 (additive, v1/v2 stay valid):** `assets[]` (`{id, kind: video|image|audio, path, durationSeconds, width, height, rights}`), every segment gets `assetId`; image segments carry `kind:"image"`, `durationSeconds`, `motion` (start/end crop rect + easing); per-segment `layout`/`reframe` (mode: stationary|pan|track, path); `captions.placement` (top|bottom per cue, from subject box); `audio.timeline` (already implicit in stems) recorded as explicit tracks + gain automation summary; `music.file` for user-supplied audio with `rights` attestation. The renderer stays an executor.
2. **Ingest → pool:** analyse each asset separately; pool shots across assets; redundancy/continuity across assets; images enter the pool as shots with duration chosen by content (a face/deity needs longer than a wide).
3. **Hook-first story option** and a **share-of-blends cap**; log both in `creative.decisions`.
4. **Reframing v2:** subject priority via open-vocab detection (licence to verify) + camera modes + polynomial path + subject-cutoff QC.
5. **Audio:** Beat This! for real tracks (keep our analyser as fallback); **bar-level DP** aligning cuts; loop/trim music at bar boundaries with equal-power crossfade; user-supplied music treated exactly like library music (analysed, ducked), licence-gated by an explicit attestation; no suitable music → original audio only (already true).
6. **Captions:** safe zones from §2.9, grapheme-aware cps, placement chosen per cue.
7. **QC additions:** subject cut-off, aspect ratio/resolution, full-decode corruption check, plan-timestamp vs media-duration validity, caption safe-zone violation.
8. **Renderer parity:** either implement the v3 semantics in Remotion or refuse creative plans there explicitly (already logged); decide the licence first.

## 5. What to test (and how)
| Test | Design | Success signal |
|---|---|---|
| Hook-first vs establishing opening | same footage, both plans, blind A/B (≥ 20 reels, several raters) | preference win-rate; first-2 s quality rank |
| Bar-DP vs greedy alignment | measure cut-to-beat error **and** shot-length deviation; blind A/B | lower error without forcing every cut to a beat |
| Chant margin 10 / 12 / 14 / 16 dB | listening test on live-chant reels | music audible but never masks the chant |
| Camera modes + polynomial path | path jerk (∑|d³x/dt³|), cut-off faces in QC | lower jerk, zero cut-off |
| AdaptiveDetector vs Content | false-cut count on handheld footage | fewer false cuts, same recall |
| Beat This! vs our tracker | beat F1 against hand-annotated real music | ≥ parity, plus downbeats |
| Ken Burns variants | slideshow-feel rating; subject centred/cut-off | no cut-off, rated "not slideshow" |
| WhisperX vs faster-whisper on kn/hi | word-boundary error on hand-aligned clips | switch only if clearly better |

## 6. Evaluation metrics
- **Structure:** shot-length distribution; blend share; role coverage; max pairwise similarity (redundancy); first-2 s quality rank.
- **Audio:** integrated LUFS, true peak, LRA; **speech–music margin (LU) percentiles**; clipped samples; silence ratio.
- **Sync:** cut-to-beat error distribution; fraction of cuts on a beat (should NOT be ≈100%).
- **Framing:** subject-in-frame ratio; cut-off count; path jerk.
- **Captions:** cps, lines, safe-zone violations, overlap with subject box.
- **Overall:** QC pass rate after ≤ 2 re-edits; blind human A/B preference vs the old pipeline.

## 7. Prioritised roadmap (by expected effect on Reel quality)
| # | Item | Why it matters for the Reel | Effort |
|---|---|---|---|
| P0-1 | EditPlan v3 + multi-asset ingest + image segments (2D Ken Burns) | required by the brief; biggest capability gap | High |
| P0-2 | Safe zones + subject-aware caption placement + grapheme-aware cps | captions currently sit in UI zones on real platforms | Low–Med |
| P0-3 | Hook-first opening + blend-share cap (+ A/B harness) | first seconds decide retention | Low |
| P0-4 | Reframing v2 (camera modes, polynomial path, subject-cutoff QC) | fixes jitter/cut-off faces | Medium |
| P1-5 | Beat This! + bar-level DP + loop/trim + user-supplied music | music that supports, not merely sits | Medium |
| P1-6 | Chant margin listening test; per-platform loudness | audio balance is subjective | Low |
| P1-7 | AdaptiveDetector; QC extras | fewer bad cuts / bad renders | Low |
| P2-8 | Open-vocab detector for deity/priest localisation | enables the true priority chain | Medium–High |
| P2-9 | WhisperX for kn/hi (if better); Remotion parity (after licence decision) | polish | Medium |
| P2-10 | Parallax (Depth Anything V2-Small), optional LLM director | nice-to-have | High |

**Explicitly not recommended:** DETR/MR models (no training data, query-based), Vidi (NC), madmom models (NC), generative beat retiming, depth models other than Small, a manual-editor UI.

## 8. Open questions for the business
1. **Remotion:** is Aikyam a for-profit with > 3 employees? If yes a Company License is required to ship the Remotion path.
2. **User-supplied music:** what rights attestation is acceptable?
3. **Evaluation media:** only one real clip (Ganga aarti) is kept in the repo; a handful more licensed/owned clips are needed for the A/B tests above.


## Addendum: diffusionstudio/editor (evaluated 2026-09-20, repo cloned and read, code NOT run)

MPL-2.0 (brand assets excluded). "Video editor built for agents": JSX compositions, a headless runtime (`@diffusionstudio/runtime`, koota ECS), an encoder (`@diffusionstudio/encoder`, Mediabunny) and a `dapi` CLI / MCP server that drives a **running Electron desktop app**.
Findings that decide fit:
- **Export needs a DOM canvas and WebCodecs**: `encoder.ts` asserts `canvas instanceof HTMLCanvasElement`; `dapi export` renders inside the app's renderer process, "one export at a time", CLI waits up to 60 min. Only macOS (DMG/ZIP) makers are configured. A Linux server would need Electron under Xvfb with WebCodecs H.264: unverified.
- **Audio**: per-clip volume in dB and mute only; no ducking, loudness normalisation or limiter (our numpy mixer would still have to produce the audio).
- **Transitions**: dissolve, slide, fade-to-black/white (covers our crossfade/dip_black/dip_white; no per-edge control verified). Captions are canvas text (browser shaping, so Indic likely fine; not verified).
- **Input is JSX, not JSON**: an EditPlan -> JSX adapter would be needed; the reverse (JSX -> EditPlan) for human fixes is a larger job.
Decision: not adopted as a render backend (FFmpeg already does everything the EditPlan needs and is covered by tests; this would repeat the Remotion parity cost with a GUI runtime). Worth revisiting only as a human-in-the-loop editing surface.
