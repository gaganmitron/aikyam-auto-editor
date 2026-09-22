# One image, many roles: `aikyam-video <cmd>` (CLI) | `aikyam-video worker --role X` | `aikyam-video serve` (API).
# Build:  docker build -f infrastructure/docker/Dockerfile -t aikyam/video:0.2.0 .
# With Remotion renderer (adds Node + Chromium, ~600MB):  --build-arg WITH_REMOTION=1
FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core fonts-noto-core fonts-noto-ui-core \
      fonts-noto-extra espeak-ng && rm -rf /var/lib/apt/lists/*
ARG WITH_REMOTION=0
RUN if [ "$WITH_REMOTION" = "1" ]; then apt-get update && apt-get install -y --no-install-recommends nodejs npm chromium && rm -rf /var/lib/apt/lists/*; fi
WORKDIR /app
COPY pyproject.toml ./
COPY aikyam_video ./aikyam_video
# CPU-only torch first (the default PyPI wheel pulls multi-GB CUDA libraries). GPU nodes: use a CUDA base image + default wheels.
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir ".[clip]"
# Pre-fetch model weights so the first job doesn't download ~2GB (HF_HOME=/models is a volume-friendly location)
ENV HF_HOME=/models VISION_MODEL=hf-hub:timm/ViT-B-16-SigLIP WHISPER_MODEL=base TOKENIZERS_PARALLELISM=false
RUN python -c "import open_clip; open_clip.create_model_and_transforms('hf-hub:timm/ViT-B-16-SigLIP'); open_clip.get_tokenizer('hf-hub:timm/ViT-B-16-SigLIP')" \
 && python -c "from transformers import ClapModel, ClapProcessor; ClapModel.from_pretrained('laion/clap-htsat-unfused'); ClapProcessor.from_pretrained('laion/clap-htsat-unfused')" \
 && python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')"
COPY renderer ./renderer
COPY music ./music
RUN if [ "$WITH_REMOTION" = "1" ]; then cd renderer/remotion && npm ci --no-audit --no-fund 2>/dev/null || npm install --no-audit --no-fund; fi
ENV MUSIC_LIBRARY_DIR=/app/music AIKYAM_REMOTION_DIR=/app/renderer/remotion AIKYAM_CHROMIUM=/usr/bin/chromium
COPY tests ./tests
RUN useradd -m aikyam && mkdir -p /work /storage && chown -R aikyam /work /storage /models
USER aikyam
EXPOSE 8080 9100
ENTRYPOINT ["aikyam-video"]
