# Evidence: ordering clips, finding the best moments, framing — what the research says (2026-09-25)

This is the literature note behind `SEQUENCE_SEARCH.md` and `HARDWARE_BUDGET.md` (it was first written as `thinking_editor_research.md`, which the user later replaced with the brief `Aikyam — Research & Reuse Blueprint`; restored here under its own name at the user's request). Scope: temple footage, **no recorded audio used (own music only)**, private project, 6 GB GPU next.

**How reliable is this note.** Numbers come from arXiv HTML/abstract pages as extracted by a fetch tool that summarises pages with a small model; I did not check them against the PDFs. Treat them as pointers to the papers. Where a page gave only an abstract, I say so. None of these papers tests temple footage.

## 1. What the research says

### 1.1 Ordering shots ("which first, which next")
| Work | What it does | Result that matters to us |
|---|---|---|
| **Shot Sequence Ordering** — [arXiv 2503.17975](https://arxiv.org/abs/2503.17975) (2025) | Benchmarks (AVE-Order, ActivityNet-Order): given 3 shots of one scene, recover the original order (6-way choice). Model: MViT + "cinematology embedding" (shot size / angle / motion labels + film genre). | Best model **35.3%** top-1 on AVE-Order (random 16.7%); **humans 39.9%**. Ablation: **shot category (size, angle, motion) matters far more than genre.** Limits: only 3 shots; "fell short of human level". |
| **FilmGPT** — [arXiv 2607.14645](https://arxiv.org/abs/2607.14645) (2026) | Decoder-only transformer trained on ~6,200 h of professional film; each frame = 32 tokens, `<CUT>` token between shots; "footage-constrained decoding" picks the next shot from the raw footage. | **53.9%** on AVE-Order (+18.6 over the 2025 model). User study (83 people, 1,242 comparisons): preferred over Transcript2Video 83%, over EditDuet 61%. Limits: ~2.5 min context; **"cuts on action between two unrelated videos"**; needs audio/metadata conditioning; **code listed as "coming soon" — not usable**. |
| **From Shots to Stories (L-Storyboard / StoryFlow)** — [arXiv 2505.12237](https://arxiv.org/abs/2505.12237) (2025) | Turns each shot into a text card (description, shot scale/angle/motion, timestamp; the paper also uses subtitles/ASR, which we will not) and lets an LLM order/select. StoryFlow: generate many candidate orders (diverse temperatures), then a second pass **selects the most coherent one**. | Qwen3-32B few-shot ordering only 20.8% (a trained video model got 32%); StoryFlow 29.8% but better Kendall-tau. A LoRA-tuned 8B VLM classified shot size 81% / angle 88% / motion 62%. Limits: information lost turning pictures into words; only 3-shot sequences; VLMs weak at motion. |
| **EditDuet** — [arXiv 2509.10761](https://arxiv.org/abs/2509.10761) (2025) | Multi-agent editor: an Editor agent (search clips with CLIP, add/trim on a timeline) and a **Critic** agent that gives feedback until satisfied. | Lowest failure rate among LLM editors (8.2%), 89.8% time coverage, low repeated footage. Beaten by FilmGPT in a user study (61%). |
| **LAVE** — [arXiv 2402.10294](https://arxiv.org/abs/2402.10294) (2024); **Unified Agentic Video Editing** — [arXiv 2609.12769](https://arxiv.org/abs/2609.12769) (2026, abstract only read) | LLM agents that plan edits from language descriptions of clips. | Same pattern: describe clips in text, let an LLM plan. |

**What this means.** (1) "Recover the original order" is the wrong target for us: even humans get it right about 40% of the time on 3 shots, and our sources are 6–11 minute recordings with no editorial order at all. The target is *editorial grammar*. (2) The one consistent, measured cue is **shot language: size, angle, motion**; we do not use it yet (our beats describe content, not how it is shot). (3) Free-form LLM reasoning over text descriptions is **worse than trained visual models** at ordering; what helps is **generate several candidate sequences, then select** (StoryFlow; EditDuet's critic). We already generate candidates (beam width 48) but keep only the best. (4) FilmGPT's failure — cutting on action between *unrelated* videos — applies to our multi-video reels.

### 1.2 Finding the best moments
- Query-based benchmarks ([QVHighlights, arXiv 2107.09609](https://arxiv.org/abs/2107.09609); recent leaders such as CoSTL, [arXiv 2606.01149](https://arxiv.org/abs/2606.01149), snippet only) need a text query. We can make queries per story role, but that is what SigLIP zero-shot labels already do, so little is gained.
- Missing from our own measurements (EXP-021: no signal beat the current quality measure; only dead-footage avoidance held) is a **trained video-quality model**. **DOVER** ([github.com/VQAssessment/DOVER](https://github.com/VQAssessment/DOVER), ICCV 2023) scores **aesthetic** and **technical** quality separately, has open code and weights, and reports ~0.89–0.91 rank-correlation with human quality scores on user-generated video (KoNViD-1k, LSVQ, YouTube-UGC). Untested on temple footage.

### 1.3 Framing (9:16)
- **AutoFlip** ([Google](https://research.google/blog/autoflip-an-open-source-framework-for-intelligent-video-reframing/), 2020): per shot, detect salient content, optimise a camera path (stationary / pan / track) or pad with blur. Our `reframe.py` / `layout.py` already do this.
- Our measured gap (EXP-023): the tracker's "subject width" has no relation to the truth (Spearman 0.02) and no locator we could run beat a centre crop. Missing: a **text-prompted subject locator that tracks**. **SAM 3** ([arXiv 2511.16719](https://arxiv.org/abs/2511.16719)) segments and tracks all instances of a concept given as a short noun phrase ("gopuram", "priest", "lamp"); 848M parameters. Weights may need an access request; licence unverified.

### 1.4 A 6 GB GPU
A third-party guide gives Qwen3-VL-4B (video capable; 2B/4B/8B exist) as ~9 GB in fp16, **~5 GB at 8-bit, ~2.5–3 GB at 4-bit** (unverified). L-Storyboard reports a quantised MiniCPM-o at 5.7 GB.

## 2. Experiment plan (log as EXP-025 onward; each judged against the current method)
| # | Question | Measure | Bar to ship |
|---|---|---|---|
| E1 | Does DOVER (aesthetic, technical) predict our clip ratings? | `tools/bench_gate.py` ratings (70 windows; only 2 of 5 source videos still on disk — re-download) | within-video Spearman clearly above the current +0.28 (best 3-signal combination was +0.41) |
| E2 | Can zero-shot SigLIP prompts give shot size / angle? | new small truth set (frames rated by size) | ≥ ~75% size accuracy (LoRA-tuned 8B VLM: 81%) |
| E3 | Does SAM 3 (text prompt) beat the tracker / centre crop for subject position and extent? | `tools/bench_subject.py` (22 frames) | hold more subject than the tracker's 0.77 / 0.87 at window 0.45 / 0.60; oracle is 0.90 / 0.98 |
| E4 | Do transition-cost features make a better sequence? | A/B reels from the same clip pool, user picks | picked in ≥ 14 of 20 pairs (20 pairs cannot resolve small gains) |
| E5 | Does a small-VLM critic choosing among the top-K sequences beat the rule critic? | same A/B set | beats E4's winner by the same margin |

## 3. What I could not establish
- Whether any of these numbers transfer from films / ActivityNet to temple footage.
- FilmGPT-quality ordering is out of reach (6,200 h of data, no code); only its ideas can be borrowed.
- DOVER and SAM 3 were not run by me; fit to our footage is a hypothesis until E1 and E3.
- The ordering benchmarks use 3-shot sequences; our reels use 6–7 clips.
