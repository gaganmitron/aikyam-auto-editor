# Setting up Aikyam on a fresh machine

Everything needed to go from a bare machine to a rendered reel. Steps 1-4 are
required; 5-7 are optional (the pipeline runs and degrades gracefully without
them — each says what you lose by skipping it).

## 0. Hardware expectations

- CPU-only is fine — the reference dev machine has no GPU (AMD integrated
  graphics, no CUDA) and everything here was built and tested on CPU torch.
- **RAM matters more than GPU**: budget ~4-6 GB free. Whisper + SigLIP + CLAP +
  an FFmpeg 4K render together can approach that on a single `process` call.
  On a small machine, don't run other heavy jobs (another render, the full test
  suite) at the same time as a `process` call.
- `/tmp` as tmpfs can be small (some cloud/VM images give it ~3 GB or less) --
  `pip install` of torch can fail there. See step 2 if that happens.

## 1. System prerequisites

```bash
python3 --version   # need >= 3.9
ffmpeg -version      # need ffmpeg + ffprobe on PATH
git --version
```

Debian/Ubuntu:
```bash
sudo apt-get update && sudo apt-get install -y python3 python3-venv ffmpeg git
```

Optional, only if you'll use the pieces in step 6/7:
- Node.js >= 18 + npm (Remotion or diffusion renderer)
- Docker or Podman (the full Kafka/Postgres/MinIO stack)

## 2. Clone and create the Python environment

```bash
git clone <this repo's URL> aikyam
cd aikyam
make venv && . .venv/bin/activate
```

Then install the ML extras (SigLIP vision, CLAP audio) with the **CPU** torch
build explicitly -- the default PyPI wheel pulls several GB of CUDA you don't
need:

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[clip]'
```

**If `pip install` fails or hangs on a small `/tmp`:** point `TMPDIR` at
somewhere under your home directory first:

```bash
mkdir -p ~/.pip-tmp && TMPDIR=~/.pip-tmp pip install -e '.[clip]'
```

First real run downloads the SigLIP/CLAP model weights (~1.5 GB) from
Hugging Face -- needs internet the first time only, then they're cached.

## 3. Smoke-test it

```bash
make fixture   # builds a tiny synthetic test video, no real footage needed
make demo      # runs the full pipeline on it
```

`demo` writes to `output/` (`reel-9x16.mp4`, `edit-plan.json`, `qc.json`, ...).
If this completes, the core pipeline is working.

## 4. Run the test suite

```bash
python -m pytest -q --ignore=tests/test_workers.py --ignore=tests/test_remotion.py
```

Skips the two suites that need external services (Kafka broker, `npm install`
in the Remotion renderer -- see steps 6/7). Takes roughly 10-15 minutes on a
modest CPU machine; that's expected, not a hang.

**Known, intentional exception:** `test_creative_story.py::test_story_follows_
the_arc_and_serves_each_role_with_the_right_kind_of_shot` fails on purpose --
documented in the test file itself and in `docs/research/experiment_matrix.md`
(EXP-002b/EXP-003). Everything else should pass.

## 5. Run it on your own footage

```bash
python -m aikyam_video.cli process YOUR_VIDEO.mp4 -o results/x
# several clips/photos + your own music:
python -m aikyam_video.cli process a.mp4 b.mp4 photo.jpg --music-file song.mp3 \
    --i-own-the-music-rights --format reel --title "My Reel" -o results/y
# the small web UI:
python -m aikyam_video.ui   # http://127.0.0.1:8090, uploads/outputs under results/ui
```

Run **one** of these at a time on a RAM-constrained machine (see step 0).

This repo does not ship real temple footage (large video files are excluded --
see `.gitignore`; GitHub also hard-rejects files over 100 MB). Bring your own
clip, or see `inputs/CREDITS.txt` for where the clip used during development
came from (Wikimedia Commons, CC BY 3.0).

## 6. Optional: Beat This! bar-aligned cuts

Needs its own Python (>= 3.10; the main env is pinned to 3.9 for everything
else) because `beat_this` requires newer syntax:

```bash
python3.10 -m venv .venv-beat
. .venv-beat/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install beat_this
deactivate
```

Called as a subprocess (`aikyam_video/creative/beats.py` ->
`beat_this_worker.py`); point at a different interpreter with
`AIKYAM_BEAT_THIS_PYTHON=/path/to/python` if you name the env something else.
**Skip this and the pipeline falls back to a simpler heuristic beat tracker
automatically** -- nothing breaks without it.

## 7. Optional: the full distributed stack (Kafka / Postgres / MinIO / API)

```bash
docker compose up -d --build
docker compose run --rm cli process /data/sample.mp4 -o /data/output
```

If `docker` is actually a Podman shim on your machine (as on the reference dev
box): `systemctl --user start podman.socket` first, and MinIO's image lives on
`quay.io`, not Docker Hub -- `docker-compose.yml` already points there.

For Kubernetes: see `infrastructure/kubernetes/` and README.md section 3.

## 8. Optional: the diffusion / Remotion renderers

Only needed for `--renderer diffusion` or `--renderer remotion` (FFmpeg is the
default and needs neither):

```bash
cd renderer/diffusion && ./setup.sh && node build.mjs && cd ../..   # pins a specific external commit, see the script
cd renderer/remotion && npm install && cd ../..
```

The diffusion renderer needs a real (non-`/tmp`) disk path for its Chromium
profile -- already handled in `render.mjs`, just don't relocate it onto tmpfs.

## 9. Where to read next

- `README.md` -- feature list, configuration knobs, API reference.
- `docs/ARCHITECTURE.md` -- system design (some sections predate later work).
- `docs/HANDOFF.md` -- environment quirks, what's been built and why, honest
  list of open weaknesses.
- `docs/research/experiment_matrix.md` -- the experiment log behind several
  fixes already in `creative/story.py`, `highlights.py`, and `creative/qc.py`;
  read before changing the story planner or moment scoring again.
