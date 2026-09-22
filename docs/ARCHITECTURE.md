Build: Aikyam Temple Video Intelligence & Auto-Editing Platform
Context
Aikyam is a devotional/cultural social platform for temples, priests and devotees.
We want to build a backend media intelligence pipeline that takes long temple recordings/live-stream recordings and automatically produces high-quality devotional short-form videos.
The system should NOT attempt to build a generic CapCut clone.
The core product is:
Temple Video → Understand what happened → Identify meaningful devotional moments → Generate structured edit plan → Render multiple formats → Publish to Aikyam
Use open-source projects and architectural ideas from:
SNIPRA — AI video director / highlight detection
OpenCut AI — AI editing capabilities
OpenCut/Floom — programmatic video generation
AutoClip — long-video → short-video pipeline
ClipForge — highlight scoring / reframing
Chopify — local video clipping
Remotion — programmatic video rendering
FFmpeg — media processing
Whisper / faster-whisper — transcription
Do NOT blindly copy any repository. Study the architecture and implement a clean Aikyam-native system.

1. Target Architecture
Build this architecture:
                        ┌──────────────────────┐
                         │      Aikyam App      │
                         └──────────┬───────────┘
                                    │
                              Upload / Event
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Media API            │
                         │ Spring Boot / Java   │
                         └──────────┬───────────┘
                                    │
                              Create Job
                                    │
                                    ▼
                              Kafka Topic
                         media-processing-jobs
                                    │
                                    ▼
                     ┌─────────────────────────┐
                     │ Media Orchestrator      │
                     └───────────┬─────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
       Audio/Transcript      Vision Analysis    Metadata
              │                  │                  │
              ▼                  ▼                  ▼
       Whisper/STT          CLIP/VLM           Aikyam KG
              │                  │                  │
              └──────────────────┼──────────────────┘
                                 ▼
                       AI Edit Planner
                                 │
                                 ▼
                         EditPlan JSON
                                 │
                ┌────────────────┴───────────────┐
                ▼                                ▼
        Remotion/OpenCut                  FFmpeg Workers
        Programmatic render                Media operations
                │                                │
                └────────────────┬───────────────┘
                                 ▼
                         Rendered Assets
                                 │
                                 ▼
                       Object Storage / R2
                                 │
                                 ▼
                         Cloudflare CDN
                                 │
                                 ▼
                           Aikyam Feed


2. First Principle
Separate the system into five layers:
Layer 1 — Media Intelligence
Understand the source video.
Extract:
audio
transcript
scenes
shot boundaries
faces
people
deity
temple
ritual
objects
text appearing on screen
timestamps
important visual moments
audio peaks
silence
speech segments
Layer 2 — Aikyam Semantic Intelligence
Map detected content into Aikyam entities:
Temple
Deity
Ritual
Festival
Location
Priest
Event
Language
Tradition
Region

Use IDs, not only strings.
Example:
{
  "templeId": "temple_123",
  "deityId": "deity_45",
  "ritualId": "ritual_12",
  "festivalId": "festival_9"
}

Layer 3 — Moment Ranking
Determine which portions of a long recording are worth turning into content.
Create a scoring system.
Example:
momentScore =
    0.25 * visualImportance
  + 0.20 * devotionalRelevance
  + 0.15 * audioImportance
  + 0.15 * semanticImportance
  + 0.10 * novelty
  + 0.10 * completeness
  + 0.05 * temporalImportance

Do NOT hard-code this permanently.
Make scoring pluggable/configurable.
Layer 4 — Edit Planning
Generate an intermediate representation:
{
  "sourceVideoId": "video_123",
  "outputFormat": "REEL",
  "durationSeconds": 45,
  "segments": [
    {
      "start": 125.2,
      "end": 138.4,
      "reason": "deity_closeup",
      "score": 0.94
    },
    {
      "start": 420.1,
      "end": 451.8,
      "reason": "aarti",
      "score": 0.91
    }
  ],
  "captions": {
    "enabled": true,
    "language": "kn"
  },
  "overlay": {
    "temple": "Chamundeshwari Temple",
    "ritual": "Morning Aarti"
  },
  "music": {
    "enabled": false
  }
}

The EditPlan must be renderer-independent.
Layer 5 — Rendering
Convert EditPlan into:
9:16 Reel
1:1 square
16:9 landscape
thumbnail
preview
Use FFmpeg for deterministic media operations.
Use Remotion/programmatic rendering where dynamic layouts, captions and overlays are needed.

3. Repository Structure
Create a clean monorepo:
aikyam-video-intelligence/

├── README.md
├── docker-compose.yml
├── Makefile
│
├── services/
│   ├── media-api/
│   ├── media-orchestrator/
│   ├── intelligence-worker/
│   ├── highlight-worker/
│   ├── edit-planner/
│   └── render-worker/
│
├── packages/
│   ├── media-model/
│   ├── edit-plan/
│   ├── aikyam-entities/
│   ├── scoring/
│   └── ffmpeg/
│
├── renderer/
│   ├── remotion/
│   ├── templates/
│   └── components/
│
├── models/
│   ├── transcription/
│   ├── vision/
│   └── embeddings/
│
├── infrastructure/
│   ├── docker/
│   ├── kubernetes/
│   └── helm/
│
└── tests/

Choose technologies pragmatically.
Do not introduce unnecessary microservices during MVP.
Prefer:
Python
FastAPI
FFmpeg
faster-whisper
OpenCV
PySceneDetect
CLIP-compatible vision model
Remotion
Redis
Kafka
Postgres
S3-compatible object storage

Spring Boot can remain the existing Aikyam-facing API if appropriate.

4. MVP MUST WORK END-TO-END
Implement an actual working pipeline, not mocks.
Input:
sample.mp4

Run:
aikyam-video process sample.mp4

Output:
output/
├── transcript.json
├── scenes.json
├── moments.json
├── edit-plan.json
├── reel-9x16.mp4
├── square-1x1.mp4
├── landscape-16x9.mp4
└── thumbnail.jpg

The pipeline must run locally with Docker.

5. Transcription
Use faster-whisper initially.
Requirements:
timestamps
word timestamps if available
language detection
segment confidence
optional translation
Example:
{
  "language": "kn",
  "segments": [
    {
      "start": 12.3,
      "end": 18.8,
      "text": "..."
    }
  ]
}

Design an abstraction so Google STT V2 can later replace Whisper.
Interface:
class TranscriptionProvider:
    def transcribe(self, media_path: str) -> Transcript:
        ...


6. Scene Detection
Implement:
shot boundary detection
scene grouping
keyframe extraction
Use PySceneDetect or OpenCV.
Output:
{
  "sceneId": "scene_14",
  "start": 120.4,
  "end": 142.8,
  "keyframe": "..."
}


7. Visual Intelligence
Create an abstraction:
class VisionProvider:
    def analyze(self, frame) -> VisionResult:
        ...

Initially support a local/open model.
Detect:
deity
idol
priest
devotees
temple architecture
flowers
lamps
aarti
abhishekam
procession
crowd
decorations
festival elements
text
Do not require perfect classification.
The system must expose confidence scores.

8. Aikyam Entity Extraction
Create an entity resolver:
Raw observation
      ↓
Entity extraction
      ↓
Candidate Aikyam entities
      ↓
Entity resolver
      ↓
Canonical IDs

Example:
{
  "text": "Chamundeshwari",
  "entityType": "DEITY",
  "entityId": "deity_123",
  "confidence": 0.96
}

Keep entity extraction separate from rendering.

9. Highlight Detection
Build a candidate-generation pipeline.
Candidate signals:
Visual
deity close-up
ritual activity
crowd reaction
fire/lamp
flowers
procession
temple architecture
Audio
bell
conch
chanting
mantra
music
applause
speech
Semantic
ritual names
deity names
festival names
important phrases
Temporal
beginning/end of ritual
event start
aarti
darshan
procession
Generate candidate clips first.
Then rank them.
Never let the LLM directly choose arbitrary timestamps without validation.

10. Clip Quality Validation
Every selected clip must pass:
duration validation
+
source availability
+
audio availability
+
scene continuity
+
no excessive silence
+
no black frames
+
no duplicate clip

Reject bad clips automatically.

11. Edit Planner
Create an LLM-independent deterministic EditPlan first.
Then optionally add an LLM planner.
The LLM should propose:
WHAT to show
WHY to show it
HOW to arrange it

It should NOT directly render video.
Validate every LLM-generated timestamp against source media.
Schema:
EditPlan
 ├── source
 ├── duration
 ├── aspectRatio
 ├── segments
 ├── captions
 ├── overlays
 ├── transitions
 ├── audio
 └── thumbnail

Use JSON Schema validation.

12. Aikyam Video Templates
Create reusable templates.
Template: Ritual Highlight
[Temple Name]

[Deity]

[Short ritual title]

VIDEO

Aikyam logo

Template: Festival
[Festival]

[Temple]

[Location]

VIDEO

Aikyam

Template: Divine Moment
Minimal design.
Avoid excessive overlays.
The actual temple footage should dominate the screen.

13. Caption System
Implement:
word-level captions
sentence captions
Kannada
Hindi
English initially
Design language support so Tamil/Telugu/Marathi/etc. can be added without code changes.
Support:
font
fontSize
position
background
animation
language


14. Automatic Reframing
For 16:9 → 9:16:
Track the important subject.
Priority:
deity
→ priest
→ speaker
→ ritual
→ central action

Do NOT simply crop the center.
Implement subject-aware cropping.

15. Thumbnail Generation
Generate several candidate thumbnails.
Score:
face/deity visibility
+
visual clarity
+
semantic relevance
+
brightness
+
composition

Select the highest scoring candidate.
Store all candidates for future experimentation.

16. Audio
Initially preserve original audio.
Do not automatically add copyrighted music.
Create an audio abstraction:
class AudioProcessor:
    def normalize(...)
    def duck(...)
    def mix(...)

Later support an Aikyam-owned royalty-safe devotional music library.

17. Storage
Support S3-compatible storage.
Interface:
class ObjectStorage:
    upload(...)
    download(...)
    signed_url(...)
    delete(...)

Make it compatible with:
AWS S3
Cloudflare R2
Linode Object Storage
MinIO
Aikyam currently uses object storage + Cloudflare CDN.

18. Processing Jobs
Use Kafka.
Events:
MediaUploaded
TranscriptionRequested
TranscriptionCompleted
SceneAnalysisRequested
SceneAnalysisCompleted
HighlightGenerationRequested
HighlightGenerationCompleted
EditPlanGenerated
RenderRequested
RenderCompleted
MediaPublished

Every event should contain:
{
  "jobId": "...",
  "mediaId": "...",
  "tenantId": "...",
  "timestamp": "...",
  "attempt": 1
}

Implement idempotency.

19. Job State Machine
Implement:
UPLOADED
 ↓
PROCESSING
 ↓
ANALYZED
 ↓
HIGHLIGHTS_READY
 ↓
EDIT_PLAN_READY
 ↓
RENDERING
 ↓
READY
 ↓
PUBLISHED

Failure:
FAILED
 ↓
RETRYING
 ↓
PROCESSING

Store state in Postgres.

20. Kubernetes
The rendering and intelligence workers must be horizontally scalable.
Example:
render-worker
  replicas: N

intelligence-worker
  replicas: N

Use resource requests/limits.
Design workers to process one job at a time.
Later support GPU nodes.

21. Cost Awareness
Every processing job should record:
{
  "cpuSeconds": 120,
  "gpuSeconds": 0,
  "model": "faster-whisper",
  "inputDurationSeconds": 7200,
  "outputDurationSeconds": 45
}

Create:
processing_cost

metrics.
We need to know:
₹ / hour of source video
₹ / generated reel

This is critical for the temple-camera business model.

22. Observability
Expose Prometheus metrics:
media_processing_duration_seconds
transcription_duration_seconds
scene_detection_duration_seconds
highlight_generation_duration_seconds
render_duration_seconds

media_processing_failures_total
render_failures_total

clips_generated_total
clips_rejected_total

source_video_duration_seconds
output_video_duration_seconds

Add structured logging.
Every job must have:
jobId
mediaId
templeId


23. API
Create:
POST /v1/media
POST /v1/media/{id}/process
GET  /v1/media/{id}
GET  /v1/media/{id}/analysis
GET  /v1/media/{id}/highlights
GET  /v1/media/{id}/edit-plan
POST /v1/media/{id}/render
GET  /v1/media/{id}/outputs

Example:
POST /v1/media/video_123/process

returns:
{
  "jobId": "job_123",
  "status": "PROCESSING"
}


24. CLI
Provide:
aikyam-video analyze input.mp4

aikyam-video highlights input.mp4

aikyam-video plan input.mp4

aikyam-video render input.mp4 \
  --format reel

aikyam-video process input.mp4

The complete command should produce all outputs.

25. Testing
Create real automated tests.
Unit tests:
scene detection
timestamp validation
clip scoring
EditPlan validation
crop calculation
caption timing
storage abstraction
Integration tests:
input.mp4
 ↓
pipeline
 ↓
EditPlan
 ↓
render
 ↓
output.mp4

Validate:
file exists
playable
correct duration
correct aspect ratio
audio present
captions present where enabled
Add a small test video to the repository if licensing permits; otherwise document how to supply one.

26. Golden Test
Create one deterministic golden test:
sample-temple.mp4

Expected:
>= 1 valid highlight

valid EditPlan

9:16 render

1:1 render

16:9 render

thumbnail

transcript

Don't require pixel-perfect video output.
Validate structural properties.

27. AI Provider Abstraction
Never hard-code one model.
Create:
TranscriptionProvider
VisionProvider
EmbeddingProvider
EntityExtractionProvider
HighlightProvider

Implement local providers first.
Allow future:
Google Gemini
Google STT
Vertex AI
Qwen
Whisper
CLIP
OpenAI

without rewriting the pipeline.

28. Security
Never expose object-storage credentials to clients.
Use signed URLs.
Validate:
MIME type
file size
video duration
codecs
Protect APIs.

29. Important Product Constraint
Do NOT create a complicated UI initially.
The first milestone is:
Upload video
     ↓
Process
     ↓
Automatically find moments
     ↓
Generate Reel
     ↓
Preview
     ↓
Publish

Build the intelligence and pipeline before building a sophisticated editor.

30. Phase 1 Acceptance Criteria
I should be able to run:
docker compose up

Then:
aikyam-video process ./sample.mp4

and receive:
./output/
  transcript.json
  scenes.json
  moments.json
  edit-plan.json
  reel-9x16.mp4
  square-1x1.mp4
  landscape-16x9.mp4
  thumbnail.jpg

The generated Reel must:
contain selected source footage
be <= 60 seconds
be 9:16
have valid audio
have captions
have Aikyam temple/deity metadata
have no invalid timestamps
be playable using standard players

31. Phase 2
After MVP works, implement:
live stream → rolling highlight detection
automatic publishing
festival-specific templates
Kannada/Hindi/Tamil/Telugu
deity recognition
ritual recognition
Aikyam Knowledge Graph integration
personalized clip generation
automatic notification
creator/temple dashboard
content moderation
duplicate detection
semantic search
video embeddings
recommendation integration

32. Critical Design Principle
Do not build:
"Another AI video editor."
Build:
Aikyam Video Intelligence — a system that understands Indian temple events and automatically turns temple activity into structured devotional content.
The reusable asset is the intelligence layer:
Video
 ↓
Vision
 ↓
Speech
 ↓
Aikyam Knowledge Graph
 ↓
Ritual understanding
 ↓
Moment ranking
 ↓
Story generation
 ↓
Programmatic rendering

That intelligence should eventually work across:
temples
festivals
Dasara
Navaratri
Ganesh Chaturthi
Durga Puja
processions
aarti
abhishekam
pravachana
bhajans
cultural events

Implementation Instructions
Produce an architecture document before making major changes.
Then implement Phase 1 end-to-end.
Prefer working code over placeholders.
Do not create fake AI outputs to satisfy tests.
Every external dependency must be documented.
Pin versions.
Provide Docker Compose for local development.
Provide Kubernetes manifests/Helm structure for production.
Add README with exact commands.
Add automated tests.
At the end, provide:
files changed
architecture decisions
commands to run
known limitations
next recommended implementation steps

