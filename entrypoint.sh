#!/usr/bin/env bash
# Docker entrypoint for the Qwen3-TTS server.
#
# All configuration is read from environment variables so the same image works
# for both HuggingFace Hub models and locally-mounted checkpoints.
#
# Environment variables:
#   MODEL      — HF repo ID  (e.g. "myuser/qwen3-arabic-tts")
#                OR container path (e.g. "/model") when a volume is mounted
#                Default: Qwen/Qwen3-TTS-12Hz-1.7B-Base
#   VOICE_DIR  — directory with .wav + .txt voice files (Base model)
#                Default: /voices  (silently skipped if directory is absent)
#   HOST       — bind address   (default: 0.0.0.0)
#   PORT       — server port    (default: 8000)
#   DEVICE     — CUDA device    (default: cuda:0)
#   ATTN_IMPL  — flash_attention_2 | sdpa | eager  (default: sdpa)
#   DTYPE      — bfloat16 | float16 | float32      (default: bfloat16)
#   HF_TOKEN   — HuggingFace token for private repos (optional)
#
# Extra server.py flags can be appended as CMD arguments, e.g.:
#   docker run ... qwen3-tts-server --optimize --workers 2

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-Base}"
VOICE_DIR="${VOICE_DIR:-/voices}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DEVICE="${DEVICE:-cuda:0}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
DTYPE="${DTYPE:-bfloat16}"

echo "[entrypoint] model     : ${MODEL}"
echo "[entrypoint] device    : ${DEVICE}  attn=${ATTN_IMPL}  dtype=${DTYPE}"
echo "[entrypoint] endpoint  : ${HOST}:${PORT}"

ARGS=(
    --model     "$MODEL"
    --host      "$HOST"
    --port      "$PORT"
    --device    "$DEVICE"
    --attn-impl "$ATTN_IMPL"
    --dtype     "$DTYPE"
)

# Only pass --voice-dir when the directory actually exists
if [[ -d "$VOICE_DIR" ]]; then
    echo "[entrypoint] voices    : ${VOICE_DIR}"
    ARGS+=(--voice-dir "$VOICE_DIR")
fi

# Login to HuggingFace if a token is provided (needed for private/gated models)
if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "[entrypoint] HF token found — logging in ..."
    python - <<'EOF'
import os, huggingface_hub
huggingface_hub.login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
EOF
fi

exec python server.py "${ARGS[@]}" "$@"
