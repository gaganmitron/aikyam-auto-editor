AMENDMENTS (approved by the user 2026-09-26; they override anything below that conflicts)
1. OWN AUDIO ONLY. Recorded audio of the input videos is never used in a reel. Sections below that treat Whisper / speech / source-audio events / "audio-driven editing" as editorial observations do NOT apply to source audio. Audio work means our own music (beat alignment, loudness).
2. LEAN DOCUMENT SET instead of all 14 files of section 31: see docs/research/RESEARCH_MASTER.md (index), ARCHITECTURE_AUDIT.md, SEQUENCE_SEARCH.md, HARDWARE_BUDGET.md and sequencing_evidence.md (literature + experiments E1-E5). Other documents are written when their topic starts.
3. The text below is otherwise unchanged (including the stray "citeturn..." markers copied from another tool, and repository claims that have not been verified against those repositories).

AIKYAM — Research & Reuse Blueprint
Nurby + VideoHighlighter + Auto-Editor + New Research Directions
Purpose: Give the implementation agent a research-first blueprint for evolving Aikyam from an existing auto-editor into a stronger long-video understanding and editorial-intelligence system.
Status: RESEARCH / DESIGN / IMPLEMENTATION BRIEF
Rule: Reuse the existing Aikyam code wherever it already solves a problem. Do not rebuild working subsystems merely because another repository has a different implementation.

1. Executive direction
Aikyam should not become:
long video → LLM → random clips → FFmpeg
The target architecture is:
LONG VIDEO
   ↓
existing Aikyam ingestion / chunking
   ↓
existing Aikyam perception
   ↓
structured observations
   ↓
event construction / temporal reasoning
   ↓
candidate moments
   ↓
editorial state + story planning
   ↓
sequence search / beam search
   ↓
top-K candidate edits
   ↓
deterministic critic + optional VLM critic
   ↓
EditPlan
   ↓
existing Creative Engine
   ↓
FFmpeg
   ↓
QC + decision trace
The three external repositories should be treated as research references and architectural inspiration, not as code to copy wholesale.

2. Repositories researched
2.1 Nurby
Repository:
https://github.com/Eshpelin/nurby
Nurby is a context-aware camera monitoring platform. Its architecture separates ingestion, perception, events, agent reasoning, search/memory and integrations.
Important architectural pattern:
Ingestion
    ↓
Perception
    ↓
Events / rules
    ↓
Agent / reasoning
    ↓
Search / memory
The project exposes structured observations to a tool-using agent rather than asking an LLM to directly inspect an entire raw stream every time.
Useful ideas for Aikyam:
    • structured perception outputs
    • persistent event/observation memory
    • embeddings/vector search
    • temporal journeys / linked observations
    • grounded tool use
    • evidence-backed answers
    • explicit uncertainty
    • decision/audit traces
    • tiered/queued perception
    • CPU-friendly perception paths
    • open-vocabulary search as a targeted expensive operation
Important Aikyam adaptation:
Nurby is a surveillance/context system. Aikyam is an editorial system. Therefore:
Nurby:
observation → understanding → answer/action

Aikyam:
observation → understanding → story → edit
Do NOT copy Nurby's entire service architecture into Aikyam.
Instead, extract the architectural principle:
Store what the video analysis system knows as structured, queryable evidence.
Source: Nurby's public repository describes separate ingestion, perception, events, agent, search and digest components, including YOLO/tracking/VLM/audio perception and grounded tool-use Q&A. citeturn0search3

3. VideoHighlighter research
Repository:
https://github.com/Aseiel/VideoHighlighter
This is the closest external reference to Aikyam's highlight-selection problem.
It analyzes footage using multiple signals:
    • scene
    • motion
    • audio
    • objects
    • actions
    • transcript
It then scores moments, explains why moments were selected, and can convert selections into a reel.
The repository explicitly treats detection mechanisms separately and uses composition rules to combine detections into higher-level events. citeturn0search1
Most important idea: composition
Instead of:
object_score
action_score
audio_score
Aikyam should construct meaningful events from combinations:
visual condition A
+
visual condition B
+
temporal relationship
+
audio condition
        ↓
semantic event
For Aikyam this can become:
candidate detections
        ↓
temporal grouping
        ↓
event hypothesis
        ↓
event confidence
        ↓
event importance
Example:
person detected
+
ritual object detected
+
specific hand motion
+
audio activity
+
shot remains stable for N seconds
        ↓
EVENT
The exact domain vocabulary should remain configurable rather than hardcoded into generic infrastructure.

4. VideoHighlighter's explainability model
Aikyam should adopt the idea that every editorial decision needs an explanation.
For each selected moment:
WHY_SELECTED

visual_score       = ...
audio_score        = ...
event_score        = ...
quality_score      = ...
story_role_score   = ...
novelty_score      = ...
chronology_bonus   = ...
diversity_penalty  = ...

final_score        = ...
Also store:
WHY_REJECTED
for strong candidates that lost.
This matters because:
    1. it makes debugging possible;
    2. it makes experiments scientifically measurable;
    3. it prevents an LLM from becoming an unexplained black box;
    4. it lets us compare algorithms without rendering every version.
VideoHighlighter's report design explicitly exposes per-signal contribution, fired detections, selected moments, strong moments that were not selected, and limitations. citeturn0search1

5. VideoHighlighter's pipeline architecture
Its automatic pipeline separates:
ingest
→ script
→ music
→ cut list
→ transitions
→ runner
The repository also uses an editable EDL/cut-list concept.
This is highly relevant to Aikyam.
Aikyam should maintain:
Analysis
    ↓
Candidate moments
    ↓
Story plan
    ↓
EditPlan / EDL
    ↓
Renderer
The important boundary is:
Intelligence decides WHAT should happen. Renderer decides HOW to execute it.
Do not mix these.

6. Auto-Editor research
Repository:
https://github.com/WyattBlue/auto-editor
Auto-Editor is primarily useful as a reference for the deterministic editing layer.
It provides structured timeline/edit concepts, automatic analysis, clip operations, transitions and multiple timeline/export representations.
Its current project supports timeline formats, editor exports and layered timeline concepts. citeturn0search4turn0search5
The important architectural lesson:
AI output
    ↓
structured timeline
    ↓
deterministic renderer
not:
AI output
    ↓
direct FFmpeg command generation
Aikyam already has a Creative Engine. Therefore:
reuse the existing Aikyam Creative Engine instead of replacing it with Auto-Editor.
Study Auto-Editor's:
    • timeline representation
    • clip boundaries
    • transition representation
    • frame/time precision
    • audio/video synchronization
    • partial/lossless rendering concepts
    • filter graph architecture
    • export formats
    • validation
    • deterministic rendering
Recent Auto-Editor releases also show attention to frame-accurate editing, fractional frame rates, transitions, partial-lossless rendering and timeline exports. citeturn0search2

7. What Aikyam should reuse
Existing Aikyam code is the primary implementation base.
Before writing ANY new subsystem:
    1. locate existing implementation;
    2. understand its public API;
    3. run its tests;
    4. identify its limitations;
    5. extend it if possible;
    6. only replace it if there is measured evidence that replacement is necessary.
Do NOT rebuild:
    • FFmpeg rendering
    • Creative Engine
    • Ken Burns motion
    • existing music alignment
    • existing crossfade infrastructure
    • existing subtitle/caption pipeline
    • existing long-video chunking
    • existing Whisper transcription
    • existing PySceneDetect integration
    • existing SigLIP analysis
    • existing CLAP analysis
    • existing story planner
    • existing experiment logging
    • existing QC
    • existing audio cleaning
    • existing variant generation
unless an experiment proves the current implementation is inadequate.

8. Known Aikyam assets to preserve
The existing Aikyam system already contains:
faster-whisper
PySceneDetect
SigLIP
CLAP
FFmpeg
Creative Engine
story planner
long-video chunking
audio cleaning
QC / verification
variant generation
experiment logging
The existing project has already established a research loop:
baseline
→ measure
→ experiment
→ log
→ compare
→ only then change architecture
Keep that methodology.

9. Existing Aikyam architecture should become the foundation
The intended evolution is:
CURRENT AIKYAM

long video
   ↓
chunking
   ↓
transcript / scenes / vision / audio
   ↓
story planner
   ↓
creative engine
   ↓
FFmpeg
Evolve it toward:
AIKYAM vNext

long video
   ↓
existing chunking
   ↓
existing perception
   ↓
Observation Store
   ↓
Event Builder
   ↓
Candidate Generator
   ↓
Editorial State
   ↓
Sequence Search
   ↓
Critic
   ↓
existing Creative Engine
   ↓
FFmpeg
   ↓
existing QC

10. New component: Observation Store
Do not immediately introduce a large database.
Start with a local, versioned artifact format.
Example:
{
  "video_id": "...",
  "shot_id": "...",
  "start": 120.4,
  "end": 126.7,
  "observations": {
    "scene": [],
    "objects": [],
    "actions": [],
    "audio": [],
    "speech": [],
    "quality": {}
  },
  "confidence": {},
  "source": {
    "model": "...",
    "version": "...",
    "timestamp": "..."
  }
}
The key principle:
Expensive perception should be computed once and reused by many downstream experiments.
This is one of the highest-value architectural changes.

11. New component: Event Builder
The event builder should transform low-level observations into temporal events.
frame detections
     ↓
shot observations
     ↓
temporal aggregation
     ↓
event hypotheses
     ↓
event confidence
Event schema:
Event
├── id
├── start
├── end
├── type
├── observations[]
├── confidence
├── importance
├── novelty
├── lifecycle/state
├── story_roles[]
└── provenance
The event builder should NOT generate final prose.
It should generate structured facts.

12. Research topic: temporal event localization
Claude must research current methods for:
    • temporal action localization
    • temporal event detection
    • temporal grounding
    • video moment retrieval
    • weakly supervised temporal localization
    • dense video captioning
    • multimodal temporal reasoning
Questions to answer:
    1. How can an event start/end be localized without running an expensive VLM on every frame?
    2. How should overlapping events be represented?
    3. How can event confidence be calibrated?
    4. How can events be linked across chunks?
    5. How can temporal uncertainty be represented?
Do not immediately implement a research paper.
First produce:
method
compute cost
input requirements
output representation
Aikyam compatibility
expected benefit

13. Research topic: long-video memory
Research:
    • video memory architectures
    • hierarchical temporal representations
    • segment-level embeddings
    • event graphs
    • multimodal vector retrieval
    • temporal knowledge graphs
    • memory compression
    • retrieval over long videos
The goal is not to put the entire video into an LLM context window.
The goal is:
hours of footage
    ↓
compressed structured memory
    ↓
retrieve only relevant evidence

14. Research topic: editorial intelligence
This is the most important new research area.
Search academic and engineering literature for:
    • automatic video summarization
    • highlight detection
    • story generation from video
    • narrative-aware video summarization
    • cinematic video editing
    • computational cinematography
    • video montage generation
    • personalized highlight generation
    • sequence optimization for video
    • diversity-aware summarization
    • coverage vs redundancy optimization
    • temporal coherence
The research question:
How do we choose a SEQUENCE of clips rather than independently selecting the highest-scoring clips?
This is the central distinction.
Bad:
top 5 scores
Better:
find sequence S maximizing:

story coherence
+ importance
+ diversity
+ chronology
+ visual quality
+ pacing
+ novelty
+ semantic continuity
- redundancy
- bad transitions
- weak clips

15. Sequence search research
Aikyam already has story-planning work and chronology bonus experiments.
Extend that rather than replacing it.
Research:
    • beam search
    • dynamic programming
    • MCTS for sequence planning
    • constrained optimization
    • submodular summarization
    • determinantal point processes
    • diversity-aware ranking
    • graph search
    • learned reward models
Initial implementation should probably remain deterministic.
Candidate:
beam = [empty_sequence]

for position in story_positions:
    expansions = []

    for sequence in beam:
        for candidate in candidate_pool:
            new_sequence = append(sequence, candidate)

            score = (
                candidate_importance
                + story_role_fit
                + chronology
                + novelty
                + visual_quality
                + transition_quality
                + pacing
                + semantic_continuity
                - redundancy
            )

            expansions.append((score, new_sequence))

    beam = top_k(expansions)
Do not immediately replace this with an LLM.

16. Editorial state
Aikyam should maintain a compact state describing the current reel.
Example:
EditorialState

current_duration
current_energy
recent_subjects
recent_shot_sizes
recent_camera_motion
recent_colors
recent_event_types
story_position
last_transition
redundancy
semantic_focus
Then candidate selection becomes state-aware.
Example:
previous clip = close-up
previous energy = high
previous event = climax

candidate:
close-up + high energy + same event

→ penalize

candidate:
wide shot + medium energy + contextual event

→ potentially reward
This creates actual editorial reasoning.

17. Research topic: cinematography-aware editing
Research:
    • shot scale
    • shot-size sequencing
    • wide → medium → close
    • close → wide reset
    • visual continuity
    • screen direction
    • camera motion continuity
    • color continuity
    • composition continuity
    • match cuts
    • action continuity
    • gaze continuity
    • visual rhythm
Do not assume film-school rules are universally correct.
Turn them into measurable features and run experiments.

18. Research topic: audio-driven editing
Aikyam already has CLAP/audio processing and music alignment.
Research:
    • beat-aware editing
    • onset-aware cuts
    • speech/music separation
    • audio energy curves
    • semantic audio events
    • loudness normalization
    • dialogue/music ducking
    • cross-modal audio-visual alignment
Use existing music-analysis code before adding another audio library.

19. Research topic: multimodal scoring
Study methods for combining:
visual
audio
speech
semantic
motion
quality
temporal
Do not simply sum everything with arbitrary weights.
Research:
    • late fusion
    • early fusion
    • learned fusion
    • calibrated scoring
    • rank fusion
    • Bayesian fusion
    • uncertainty-aware fusion
First implement an interpretable baseline:
weighted_score
Then compare against alternatives experimentally.

20. VLM research
VLMs should be a targeted expensive tool, not the first stage.
Good:
cheap perception
      ↓
300 candidates
      ↓
VLM analyzes 30
      ↓
5 story candidates
Bad:
VLM analyzes every frame of a 2-hour video
Research:
    • Qwen-VL family
    • InternVL
    • LLaVA-style video models
    • Video-LLaMA-style approaches
    • Gemini/Claude video reasoning as external baselines where applicable
    • local VLM inference
    • temporal frame sampling
    • keyframe selection
    • VLM uncertainty
    • VLM judge reliability
Because Aikyam hardware is constrained, every model proposal must report:
VRAM
RAM
CPU
throughput
context length
frame handling
quantization options
license
offline capability

21. Hardware-aware research
Aikyam must remain usable on CPU / modest hardware.
Research a tiered system:
Tier 0
FFmpeg / metadata
      ↓
Tier 1
cheap scene/audio/motion analysis
      ↓
Tier 2
existing SigLIP / CLAP / Whisper
      ↓
Tier 3
targeted VLM
The system should never make Tier 3 mandatory for every video.

22. Candidate scoring
Create an explicit candidate schema.
CandidateMoment
├── start
├── end
├── source_shot
├── event_ids[]
├── importance
├── novelty
├── quality
├── motion
├── audio_energy
├── semantic_relevance
├── story_roles[]
├── visual_features
├── confidence
└── provenance
Then:
CandidateGenerator
should be independently testable.

23. Story roles
Do not assume every candidate is a “highlight”.
Possible abstract roles:
ESTABLISH
CONTEXT
INTRODUCTION
BUILD
ACTION
TRANSITION
DETAIL
EMOTION
CLIMAX
RELEASE
CLOSING
These should be represented as editorial roles.
Domain-specific mapping can happen later.

24. Critic architecture
Build a deterministic critic first.
Critic
├── duration
├── duplicate content
├── weak clips
├── bad transitions
├── excessive repetition
├── pacing
├── story coverage
├── audio clipping
├── black frames
├── visual artifacts
└── caption issues
Then optionally add:
VLM Critic
Only after the deterministic critic has a baseline.

25. Top-K generation
Do not generate only one reel.
Generate:
candidate reel A
candidate reel B
candidate reel C
with different editorial objectives.
For example:
Variant A = story
Variant B = energetic
Variant C = cinematic
Then compare them using the same critic.
This also fits Aikyam's existing variant/reel-score infrastructure.

26. Experiment protocol
Every new research idea must become an experiment.
Template:
EXP-XXX

Hypothesis:
...

Baseline:
...

Change:
...

Dataset:
...

Metrics:
...

Result:
...

Conclusion:
...

Decision:
KEEP / REJECT / INVESTIGATE
Never implement a large architectural change without an experiment when a measurable baseline can be created.

27. Metrics
Research and implement metrics for:
Selection
    • precision of selected moments
    • recall of known important moments
    • redundancy
    • diversity
    • temporal coverage
Story
    • chronological coherence
    • event coverage
    • story-role coverage
    • semantic continuity
Editing
    • transition validity
    • average shot duration
    • pacing variance
    • beat alignment
    • audio continuity
Technical
    • render failures
    • dropped frames
    • black frames
    • audio clipping
    • subtitle errors
    • aspect-ratio errors
Efficiency
    • analysis time / video minute
    • RAM
    • VRAM
    • CPU
    • number of VLM calls
    • cache hit rate

28. Reuse-first implementation policy
Claude MUST follow this order:
Step 1
Inventory the current Aikyam repository.
Step 2
Map existing modules to:
ingestion
perception
memory
events
candidate generation
story planning
creative engine
rendering
QC
Step 3
Identify existing implementations before adding files.
Step 4
Write tests around current behavior.
Step 5
Add the smallest missing abstraction.
Step 6
Run the existing test suite.
Step 7
Run an experiment.
Step 8
Only then consider replacing an existing implementation.

29. Explicit anti-rewrite rule
Do NOT:
    • rewrite the Creative Engine because Auto-Editor has a timeline;
    • rewrite FFmpeg integration;
    • replace Whisper without measurement;
    • replace SigLIP without measurement;
    • replace CLAP without measurement;
    • throw away the existing story planner;
    • create another chunking implementation;
    • introduce a database before proving a file-based observation cache is insufficient;
    • add an LLM agent merely because Nurby has one;
    • run VLM inference over the entire video;
    • copy external repository code without checking licenses.

30. License / dependency research
Before reusing code rather than ideas, inspect licenses.
Known:
    • VideoHighlighter is AGPL-3.0. Its repository states that modified versions, including network-served versions, must provide corresponding source under the license. citeturn0search1
    • Auto-Editor and Nurby must also be checked directly in their LICENSE files before copying code.
    • Architectural ideas and independently reimplemented algorithms are different from copying source code.
Therefore:
PREFER:
study architecture
→ understand algorithm
→ implement independently in Aikyam

ONLY COPY:
when license compatibility has been explicitly verified.
Do not paste external source into Aikyam just because it solves a similar problem.

31. Research deliverables Claude must produce BEFORE major coding
Create:
docs/research/
with:
RESEARCH_MASTER.md
REPO_NURBY.md
REPO_VIDEOHIGHLIGHTER.md
REPO_AUTO_EDITOR.md
TEMPORAL_EVENT_LOCALIZATION.md
VIDEO_SUMMARIZATION.md
EDITORIAL_INTELLIGENCE.md
CINEMATIC_EDITING.md
MULTIMODAL_SCORING.md
VLM_RESEARCH.md
LONG_VIDEO_MEMORY.md
SEQUENCE_SEARCH.md
HARDWARE_BUDGET.md
LICENSE_REVIEW.md
Do not make these documents generic literature dumps.
Every document must answer:
What is the method?
What problem does it solve?
What does Aikyam already have?
What is missing?
What can be reused?
What should be implemented?
What should NOT be implemented?
How expensive is it?
How can we test it?

32. Research priority
P0 — do first
    1. Existing Aikyam architecture audit
    2. Observation/cache model
    3. Candidate moment representation
    4. Event construction
    5. Sequence selection
    6. Editorial state
    7. Deterministic critic
    8. Reuse existing Creative Engine
P1
    9. temporal event localization
    10. long-video memory
    11. multimodal fusion
    12. cinematography-aware scoring
    13. diversity/redundancy optimization
    14. VLM targeted verification
P2
    15. learned editorial reward model
    16. advanced video-language models
    17. graph-based story planning
    18. MCTS / advanced sequence search
    19. learned personalization

33. Claude's research instructions
For each research topic:
    1. Search academic papers.
    2. Search strong open-source implementations.
    3. Inspect current Aikyam code first.
    4. Compare the method with Aikyam's existing approach.
    5. Record computational requirements.
    6. Record licensing.
    7. Identify the smallest useful experiment.
    8. Do not implement until the experiment is defined.
    9. Prefer local/offline approaches.
    10. Prefer deterministic/interpretable methods before black-box agents.
Research should prioritize:
papers
official repositories
official documentation
technical reports
benchmark datasets
Avoid building architecture from random blog posts.

34. What the final Aikyam architecture should look like
                         ┌─────────────────────┐
                         │     LONG VIDEO      │
                         └──────────┬──────────┘
                                    │
                         EXISTING AIKYAM
                         INGEST/CHUNKING
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    PERCEPTION       │
                         │ Whisper / Scene     │
                         │ SigLIP / CLAP       │
                         │ motion / quality    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ OBSERVATION STORE   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    EVENT BUILDER    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ CANDIDATE GENERATOR │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  EDITORIAL STATE    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  SEQUENCE SEARCH    │
                         │ existing planner +   │
                         │ beam search/research │
                         └──────────┬──────────┘
                                    │
                                    ▼
                              TOP-K REELS
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      CRITIC         │
                         │ deterministic first │
                         │ VLM later            │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      EDIT PLAN      │
                         └──────────┬──────────┘
                                    │
                         EXISTING AIKYAM
                         CREATIVE ENGINE
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │       FFMPEG        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         EXISTING AIKYAM QC

35. Final instruction to Claude
You are not being asked to rewrite Aikyam.
You are being asked to make Aikyam smarter while preserving everything that already works.
The external repositories provide research patterns:
Nurby
→ structured perception + memory + grounded reasoning

VideoHighlighter
→ multi-signal composition + explainable highlight selection + EDL pipeline

Auto-Editor
→ deterministic timeline + editing + rendering architecture
Aikyam provides:
existing perception
existing chunking
existing story planner
existing Creative Engine                                                  
existing music/audio system
existing FFmpeg rendering
existing QC
existing experiment methodology
Therefore the primary engineering objective is:
Build the missing editorial-intelligence layer around the existing Aikyam system, not another video editor from scratch.
Before changing code, produce the research documents and architecture mapping.
Before adding a new model, measure the existing model.
Before replacing a component, prove the replacement is better.
Before adding an LLM, identify exactly what deterministic system cannot solve.
Before copying external code, verify its license.
The final system should be:
cheap first
→ structured evidence
→ temporal understanding
→ event memory
→ candidate generation
→ sequence reasoning
→ explainable selection
→ deterministic editing
→ targeted VLM verification
→ measurable QC
That is the research and implementation direction for Aikyam.

