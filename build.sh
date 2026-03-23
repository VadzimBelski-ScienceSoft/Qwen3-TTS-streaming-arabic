#!/usr/bin/env bash
# Build the Qwen3-TTS server Docker image for DGX Spark (ARM64 / Grace Blackwell).
#
# Optional env vars:
#   IMAGE_TAG   — image name:tag         (default: qwen3-tts-server:latest)
#   BASE_IMAGE  — NGC PyTorch base image (default: nvcr.io/nvidia/pytorch:25.10-py3)
#   NO_CACHE    — set to 1 to pass --no-cache

set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-qwen3-tts-server:latest}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:25.10-py3}"
CACHE_FLAG=""
[[ "${NO_CACHE:-0}" == "1" ]] && CACHE_FLAG="--no-cache"

echo "Building ${IMAGE_TAG}"
echo "  base  : ${BASE_IMAGE}"
echo "  arch  : linux/arm64 (DGX Spark / Grace Blackwell)"
echo ""

docker build ${CACHE_FLAG} \
    --platform linux/arm64 \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    -t "${IMAGE_TAG}" \
    .

echo ""
echo "Done: ${IMAGE_TAG}"
echo ""
echo "Quick start:"
echo "  MODEL=youruser/qwen3-arabic-tts ./run.sh"
echo "  MODEL=/data/checkpoint ./run.sh"
