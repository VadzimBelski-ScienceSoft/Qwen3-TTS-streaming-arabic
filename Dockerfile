# Qwen3-TTS streaming server — DGX Spark (ARM64 / Grace Blackwell)
#
# Base image: NVIDIA NGC PyTorch (already includes CUDA 13, cuDNN, torch, torchaudio).
# No torch reinstall needed — just add our package and server deps on top.
#
# Build:
#   docker build --platform linux/arm64 -t qwen3-tts-server .
#   # or use ./build.sh
#
# Run (HuggingFace Hub model):
#   docker run --gpus all -p 8000:8000 -e MODEL=myuser/qwen3-arabic-tts qwen3-tts-server
#
# Run (local checkpoint):
#   docker run --gpus all -p 8000:8000 \
#     -v /path/to/checkpoint:/model:ro -e MODEL=/model qwen3-tts-server

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.10-py3
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# HuggingFace model cache — mount /cache as a named volume to persist downloads
ENV HF_HOME=/cache/huggingface

# Runtime defaults (all overridable via -e or docker-compose environment:)
ENV MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base
ENV VOICE_DIR=/voices
ENV HOST=0.0.0.0
ENV PORT=8000
ENV DEVICE=cuda:0
ENV ATTN_IMPL=sdpa
ENV DTYPE=bfloat16
ENV HF_TOKEN=

# sox is required by the qwen-tts package for audio I/O
RUN apt-get update && apt-get install -y --no-install-recommends \
        sox libsox-fmt-all \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy package source and install with server extras.
# Pin torch/torchaudio to whatever the NGC base already has so pip can't
# replace the CUDA build with a CPU-only one from PyPI.
COPY pyproject.toml MANIFEST.in ./
COPY qwen_tts/ ./qwen_tts/
# Pin the NGC torch version so pip can't replace the CUDA build with a CPU one.
# torchaudio is not in the NGC image; it comes from PyPI but must not pull in a
# different torch. The constraint file guarantees that.
RUN python -c "import torch; open('/tmp/torch_pin.txt','w').write(f'torch=={torch.__version__}\n')" && \
    cat /tmp/torch_pin.txt && \
    pip install --no-cache-dir -e ".[server]" --constraint /tmp/torch_pin.txt && \
    rm /tmp/torch_pin.txt

# transformers==4.57.3 requires a newer torchao API than ships in the NGC base.
# Upgrade torchao in-place; it has no compiled GPU kernels so this is safe.
RUN pip install --no-cache-dir -U torchao

# torchvision is not needed for TTS and can have a C++ ABI mismatch with the
# torch version in the NGC base after our pip install. Remove it cleanly.
RUN pip uninstall -y torchvision 2>/dev/null || true

# Copy server and entrypoint
COPY server.py openai_speech_client.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

# flash-attn: optional — ARM64 build can take a while; falls back to ATTN_IMPL=sdpa
RUN pip install --no-cache-dir flash-attn --no-build-isolation 2>/dev/null || \
    echo "[WARN] flash-attn build skipped — set ATTN_IMPL=sdpa (default)"

EXPOSE 8000

# /cache  — HuggingFace model cache (use a named volume to persist between runs)
# /voices — reference voice .wav files for Base model (optional)
# /model  — mount point for local checkpoints (set MODEL=/model when using)
VOLUME ["/cache", "/voices"]

ENTRYPOINT ["./entrypoint.sh"]
CMD []
