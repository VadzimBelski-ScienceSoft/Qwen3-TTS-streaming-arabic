#!/usr/bin/env bash
# Run the Qwen3-TTS server in Docker.
#
# MODEL can be a HuggingFace repo ID or a local path:
#   MODEL=myuser/qwen3-arabic-tts         ./run.sh   # HF Hub (downloaded to cache volume)
#   MODEL=/data/checkpoint-epoch-9        ./run.sh   # local directory (auto-mounted)
#   HF_TOKEN=hf_xxx MODEL=myuser/private  ./run.sh   # private / gated repo
#
# Other options:
#   VOICE_DIR    — directory with .wav + .txt voice files for Base model
#   PORT         — host port                   (default: 8000)
#   DEVICE       — CUDA device                 (default: cuda:0)
#   ATTN_IMPL    — flash_attention_2 | sdpa    (default: sdpa)
#   DTYPE        — bfloat16 | float16 | float32 (default: bfloat16)
#   IMAGE_TAG    — image name:tag              (default: qwen3-tts-server:latest)
#   CACHE_VOL    — named volume for HF cache   (default: qwen3-tts-hf-cache)
#   EXTRA_ARGS   — extra server.py flags

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-Base}"
VOICE_DIR="${VOICE_DIR:-./voices}"
PORT="${PORT:-8000}"
DEVICE="${DEVICE:-cuda:0}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
DTYPE="${DTYPE:-bfloat16}"
IMAGE_TAG="${IMAGE_TAG:-qwen3-tts-server:latest}"
CACHE_VOL="${CACHE_VOL:-qwen3-tts-hf-cache}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Detect whether MODEL is a local filesystem path or a HuggingFace repo ID.
# Local paths must start with /, ./, or ../ — everything else is a repo ID.
EXTRA_MOUNTS=()
MODEL_ARG="$MODEL"

if [[ "$MODEL" == /* || "$MODEL" == ./* || "$MODEL" == ../* ]]; then
    MODEL_ABS="$(realpath "$MODEL")"
    EXTRA_MOUNTS+=(-v "${MODEL_ABS}:/model:ro")
    MODEL_ARG="/model"
    echo "Loading local model : ${MODEL_ABS}"
else
    echo "Loading HF Hub model: ${MODEL}"
fi

# Mount voice directory if it exists
if [[ -d "$VOICE_DIR" ]]; then
    VOICE_ABS="$(realpath "$VOICE_DIR")"
    EXTRA_MOUNTS+=(-v "${VOICE_ABS}:/voices:ro")
fi

echo "  device  : ${DEVICE}"
echo "  attn    : ${ATTN_IMPL}"
echo "  dtype   : ${DTYPE}"
echo "  port    : ${PORT}"
echo ""

# shellcheck disable=SC2086
docker run --gpus all --rm \
    -p "${PORT}:8000" \
    -v "${CACHE_VOL}:/cache" \
    "${EXTRA_MOUNTS[@]}" \
    -e MODEL="${MODEL_ARG}" \
    -e DEVICE="${DEVICE}" \
    -e ATTN_IMPL="${ATTN_IMPL}" \
    -e DTYPE="${DTYPE}" \
    ${HF_TOKEN:+-e HF_TOKEN="${HF_TOKEN}"} \
    "${IMAGE_TAG}" \
    ${EXTRA_ARGS}
