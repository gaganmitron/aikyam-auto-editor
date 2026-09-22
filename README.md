# Aikyam Video Intelligence

Long temple recording → transcript, scenes, vision + audio understanding, ranked devotional moments → validated **EditPlan** →
9:16 / 1:1 / 16:9 renders + thumbnail → preview → publish. Design and decisions: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## 1. Run it locally (no Docker, no Kafka)
Needs Python ≥3.9, `ffmpeg` (libass + libx264), Noto fonts (Kannada/Devanagari/Tamil/Telugu) for Indic captions.
```bash
make venv && . .venv/bin/activate
# real models (SigLIP vision, CLAP audio; ~1.5 GB download on first use). CPU torch: the default wheel is multi-GB CUDA.
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu && pip install -e '.[clip]'

aikyam-video process ./sample.mp4 -o output --temple-id temple_123
```
Outputs in `output/`: `transcript.json scenes.json moments.json edit-plan.json reel-9x16.mp4 square-1x1.mp4 landscape-16x9.mp4 thumbnail.jpg`
plus `entities.json vision.json embeddings.json audio.npz cost.json metrics.prom thumbnails/` (all candidates + scores).
Without the `[clip]` extra it degrades (with a warning) to a colour/face heuristic; no audio-event tagging.

```
aikyam-video analyze|highlights|plan|render|process INPUT [-o DIR] [--format reel|square|landscape]
   [--temple-id T] [--festival-id festival_9] [--location "Mysuru"] [--language kn] [--translate]
   [--caption-lang kn] [--caption-mode sentence|word] [--target-seconds 45] [--whisper-model small]
   [--vision auto|clip|heuristic] [--audio-tagger auto|clap|none] [--planner deterministic|llm]
   [--renderer ffmpeg|remotion] [--transition crossfade|fade|cut] [--transition-seconds 0.5]
   [--music auto|off] [--music-track ID] [--music-volume 0.5] [--scoring-config weights.json]
aikyam-video live SOURCE [--chunk 60 --overlap 10 --min-score 0.4 --no-render]   # rolling highlights from a stream/file
aikyam-video serve                       # Media API            aikyam-video worker --role orchestrator|intelligence|highlight|planner|render
```

## 2. Run the full stack (Docker Compose: Postgres, MinIO, Redpanda/Kafka, API, 5 workers)
```bash
docker compose up -d --build
KEY=dev-key-change-me; H="-H X-API-Key:$KEY -H X-Tenant-Id:t1"
curl $H -F file=@sample.mp4 -F templeId=temple_123 localhost:8080/v1/media                       # -> {"mediaId": ...}
curl $H -X POST -H 'content-type: application/json' -d '{"festival_id":"festival_9","caption_lang":"kn"}' localhost:8080/v1/media/<id>/process
curl $H localhost:8080/v1/media/<id>                     # job status: UPLOADED→PROCESSING→ANALYZED→HIGHLIGHTS_READY→EDIT_PLAN_READY→RENDERING→READY→PUBLISHED
curl $H -X POST localhost:8080/v1/media/<id>/preview-link   # -> {"url": "/preview/<id>?t=..."}: open it, watch the reel, press Publish
docker compose run --rm cli process /data/sample.mp4 -o /data/output                      # one-shot CLI (no Kafka)
```
API: `POST /v1/media` · `POST /v1/media/{id}/process|render|publish|preview-link` · `GET /v1/media/{id}` `/analysis` `/highlights` `/edit-plan` `/outputs` `/similar` `/embedding` ·
`GET /v1/search?q=` · `POST /v1/jobs/{id}/retry` · `POST /v1/dashboard-link` · `GET /metrics` `/healthz`. Auth: `X-API-Key` + `X-Tenant-Id`. Clients only ever get signed URLs.
Without `KAFKA_BOOTSTRAP` the API runs jobs in-process (one at a time); with it, the API only publishes events and the workers do the work.

## 3. Kubernetes
`infrastructure/kubernetes/aikyam-video.yaml` (plain), `keda.yaml` (autoscale on Kafka lag), `infrastructure/helm/aikyam-video` (values per role,
GPU toggle `workers.<role>.gpu`, `keda.enabled`). Each worker pod = one role, one job at a time; scale with replicas.

## 4. What is implemented
| Area | Where |
|---|---|
| Transcription (faster-whisper, word timestamps, language detect, hallucination filter, optional translate) | `transcribe.py` |
| Scenes, keyframes | `scenes.py` |
| Vision (SigLIP zero-shot labels, moderation, deity guess, embeddings) / audio events (CLAP zero-shot) | `vision.py`, `audio.py` |
| Entities → canonical IDs (local catalog, or Aikyam KG over HTTP) | `entities.py`, `kg.py` |
| Candidate generation → validation → pluggable scoring | `highlights.py`, `scoring.py` |
| EditPlan (JSON Schema, timestamp validation) — deterministic planner and LLM planner (Claude, validated, falls back) | `plan.py`, `planner_llm.py` |
| Renderers: FFmpeg (default) and Remotion (React); tracked subject reframing; templates (ritual, festival ×6, divine moment); logo; captions (word/sentence, kn/hi/en/ta/te/mr by font table); music with ducking (Aikyam-owned library only) | `render.py`, `render_remotion.py`, `reframe.py`, `captions.py` |
| Thumbnails (scored candidates, all kept) | `thumbnails.py` |
| Kafka events, orchestrator + 4 worker roles, idempotency, retries, state machine (Postgres) | `bus.py`, `workers.py`, `events.py`, `jobs.py` |
| Cost per job (₹/source-hour, ₹/reel), Prometheus metrics, JSON logs | `cost.py`, `metrics.py` |
| Publish, auto-publish policy, notifications, preview page, dashboard | `publish.py`, `api.py` |
| Semantic search, duplicate detection, "similar" recommendations, video embeddings | `library.py` |
| Live rolling highlights, personalization (profile boosts), moderation gate | `live.py`, `options.py`, `highlights.py` |

## 5. Configuration
| Env | Meaning |
|---|---|
| `AIKYAM_API_KEY` | API key (the API refuses requests if unset) |
| `DATABASE_URL` | `postgresql+psycopg://…` (default local SQLite) |
| `KAFKA_BOOTSTRAP` (+ `KAFKA_SASL_*`, `KAFKA_SECURITY_PROTOCOL`) | enables event-driven mode |
| `S3_BUCKET`, `S3_ENDPOINT_URL`, `S3_PUBLIC_ENDPOINT_URL`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_REGION` | S3-compatible storage (AWS/R2/Linode/MinIO), else `STORAGE_DIR` |
| `AIKYAM_FEED_URL`, `AIKYAM_FEED_TOKEN` / `FEED_FILE`, `AIKYAM_NOTIFY_URL` | publish target (assumed webhook contract, see `publish.py`) / dev feed file / notifications |
| `AIKYAM_KG_URL`, `AIKYAM_KG_TOKEN` | Aikyam Knowledge Graph (assumed contract, see `kg.py`); local catalog otherwise |
| `ANTHROPIC_API_KEY`, `AIKYAM_LLM_MODEL` (default `claude-opus-5`) | LLM planner |
| `WHISPER_MODEL`, `VISION_MODEL`, `MODERATION_BLOCK`, `DUPLICATE_COSINE` | model / thresholds |
| `MUSIC_LIBRARY_DIR` (default `./music`) | licensed music library: `library.json` + audio files (see section 9) |
| `AIKYAM_ALLOWED_MUSIC_LICENCES` | comma list, default `AIKYAM-OWNED,CC0,PUBLIC-DOMAIN,CC-BY-4.0,CC-BY-3.0,PIXABAY-CONTENT-LICENSE` (NC/SA never allowed by default) |
| `COST_INR_PER_CPU_HOUR`, `COST_INR_PER_GPU_HOUR` | **assumptions** (3.0 / 60.0): set to your real node price |
| `AIKYAM_REMOTION_DIR`, `AIKYAM_CHROMIUM` | Remotion renderer (needs `npm install` in `renderer/remotion`) |

## 6. Tests
```bash
make test                                   # all tiers; heavy ones load SigLIP/CLAP/Whisper (several minutes, ~3 GB RAM)
python -m pytest tests/test_units.py tests/test_features.py                      # fast (seconds), no models
KAFKA_BOOTSTRAP=localhost:19092 python -m pytest tests/test_workers.py            # also runs the real-Kafka tests
```
Start a broker: `podman|docker run -d -p 19092:19092 redpandadata/redpanda:v24.2.4 redpanda start --overprovisioned --smp 1 --memory 512M --kafka-addr PLAINTEXT://0.0.0.0:19092 --advertise-kafka-addr PLAINTEXT://127.0.0.1:19092`.
The default fixture is **synthetic** (generated). `samples/` has one real Wikimedia Commons temple clip, a Ganga aarti (licence in `samples/LICENSES.md`), used by
`tests/test_real_footage.py`, and `tools/eval_vision.py` / `tools/run_batch.sh` for eyeballing. Bring your own: `AIKYAM_SAMPLE=/path/x.mp4 make test`.

## 7. External dependencies (pinned in `pyproject.toml` / `package.json`)
faster-whisper 1.1.1 · scenedetect 0.6.7.1 · opencv-python-headless 4.10.0.84 · torch 2.5.1(+cpu) / torchvision 0.20.1 · open_clip_torch 2.29.0 (SigLIP `timm/ViT-B-16-SigLIP`) ·
transformers 4.46.3 (CLAP `laion/clap-htsat-unfused`) · confluent-kafka 2.5.3 · anthropic 0.125.0 · FastAPI 0.128.8 · SQLAlchemy 2.0.36 + psycopg 3.2.3 ·
boto3 1.42.97 · prometheus-client 0.26.0 · Remotion 4.0.242 (Node ≥18, Chromium) · system: ffmpeg ≥6, Noto fonts, espeak-ng (test fixture only).
Model weights come from HuggingFace on first use (baked into the Docker image).

## 8. Known limitations
See the final report in the conversation and ARCHITECTURE.md. Highlights: zero-shot vision/audio (checked on a handful of clips, not a benchmark; deity *identity* is weak);
one reel per source video; audio is loaded whole into RAM (multi-hour videos need a big machine); Remotion is ~15× slower than FFmpeg; KG and feed contracts are assumed.

## 9. Devotional music and transitions
**Transitions.** Consecutive moments are joined with a **crossfade** (picture and sound blend, default 0.5 s); `--transition cut|fade` to change.
The reel is shorter by the overlaps; the plan, captions and both renderers use that same rule. Clips trimmed mid-moment end on the quietest spot nearby.

**Music (`--music auto`, the default).** The original audio is always kept. A licensed track is added under it (ducked while the original has sound, faded in/out) only when:
1. `music/library.json` lists a track that matches the reel (ritual / deity / festival ids, visual labels like aarti or procession, energy calm..driving), **and**
2. the original audio is not already devotional music or chanting (bell/conch/chant/bhajan level < 0.6): music is never laid over a live bhajan.
Otherwise there is no music and `edit-plan.json` records why (`audio.music.reason`). `--music off` disables it; `--music-track ID` forces one library track; `--music-volume` sets the level.

**Licence gate.** A track is used only if it is in `library.json` with `licenceVerified: true` and an allowed licence; CC-BY tracks need `attribution` (it is put in the plan and the feed payload `credits`);
NC/SA and unlisted files are refused. To add your own (commissioned or properly licensed) track:
```json
{"id": "my_bhajan_flute_01", "title": "Flute for Darshan", "file": "my_bhajan_flute_01.mp3", "licence": "AIKYAM-OWNED", "licenceVerified": true,
 "attribution": null, "source": "commissioned, release signed 2026-09-01", "energy": 0.25,
 "tags": {"labels": ["deity","idol","priest"], "deities": ["deity_45"], "rituals": ["ritual_12"], "festivals": []}}
```
`music/` ships 4 **synthesised starter tracks** (tanpura, temple bells, flute, festive dhol rhythm; generated by `tools/make_music.py`, so Aikyam-owned). They are simple placeholders: replace them with real recordings.
