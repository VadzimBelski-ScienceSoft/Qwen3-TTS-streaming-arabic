# Qwen3-TTS Streaming

Streaming inference implementation for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) with an OpenAI-compatible server, fine-tuning support, and Docker deployment.

The official team mentions "Extreme Low-Latency Streaming Generation" in their paper but the streaming code was never released — they point users to vLLM-Omni, which still doesn't support online serving.

This fork adds:
- Real-time PCM streaming (`stream_generate_pcm`)
- ~6x inference speedup vs upstream qwen-tts
- OpenAI-compatible `/v1/audio/speech` server
- Arabic language fine-tuning support
- Docker deployment

Based on [dffdeeq/Qwen3-TTS-streaming](https://github.com/dffdeeq/Qwen3-TTS-streaming) and [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS).

---

## Benchmark (RTX 5090)

### Non-streaming (full inference)

<img width="602" height="145" alt="image" src="https://github.com/user-attachments/assets/0cbfcc71-e854-46e2-81bc-ec3955ff3ff0" />

### Streaming

<img width="766" height="183" alt="image" src="https://github.com/user-attachments/assets/f5df9a38-e091-47ae-a08f-ef364f8710ea" />

---

## Installation (Python 3.12)

> Note: torch versions differ between Linux/Windows due to available flash_attn prebuilt wheels.

### 1. Install SOX

**Linux:**
```bash
sudo apt install sox libsox-fmt-all
```

**Windows:**
Download from https://sourceforge.net/projects/sox/ and add to PATH.

### 2. Create environment
```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
```

### 3. Install dependencies

**Linux:**
```bash
pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.6.8/flash_attn-2.8.3%2Bcu130torch2.9-cp312-cp312-linux_x86_64.whl
```

**Windows:**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.12/flash_attn-2.8.3%2Bcu130torch2.10-cp312-cp312-win_amd64.whl
pip install -U "triton-windows<3.7"
```

### 4. Install package
```bash
git clone https://github.com/dffdeeq/Qwen3-TTS-streaming.git
cd Qwen3-TTS-streaming
pip install -e .
```

---

## Streaming Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `emit_every_frames` | 4 | Emit audio every N frames (~0.33s at 12Hz) |
| `decode_window_frames` | 80 | Decoder context window |

See `examples/` for usage:
- [test_streaming_optimized.py](examples/test_streaming_optimized.py)
- [test_optimized_no_streaming.py](examples/test_optimized_no_streaming.py)

---

## OpenAI-Compatible Server

The server exposes a `/v1/audio/speech` endpoint compatible with the OpenAI TTS API, streaming PCM audio as Server-Sent Events.

### Start the server

```bash
# Install server extras first
pip install -e ".[server]"

# CustomVoice fine-tuned model
python server.py --model ./output/checkpoint-epoch-9 --host 0.0.0.0 --port 8000

# Base model with voice directory
python server.py --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --voice-dir ./voices \
    --host 0.0.0.0 --port 8000

# With torch.compile optimizations
python server.py --model ./output/checkpoint-epoch-9 --optimize
```

### Server arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Model path or HF repo ID |
| `--device` | `cuda:0` | CUDA device |
| `--dtype` | `bfloat16` | Weight dtype |
| `--attn-impl` | `flash_attention_2` | Attention backend |
| `--voice-dir` | `./voices` | Directory of `.wav` + `.txt` voice files (Base model) |
| `--optimize` | off | Enable `torch.compile` + CUDA graphs |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8000` | Port |

### Example request

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"Hello world","voice":"default","stream":true}'
```

Or use the included client:

```bash
python openai_speech_client.py \
  --text "Hello, how are you?" \
  --voice my_speaker \
  --output out.wav
```

### Supported endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /v1/models` | List models |
| `GET /v1/voices` | List available voices |
| `POST /v1/audio/speech` | Generate speech (streaming SSE or full WAV) |

### Model types

| `task_type` | Description | Key params |
|-------------|-------------|------------|
| `CustomVoice` | Fine-tuned single-speaker | `voice=<speaker_name>`, `instructions=<emotion>` |
| `VoiceDesign` | Instruction-controlled | `instructions=<voice description>` |
| `Base` | Voice cloning | `voice=<registered_name>` or `ref_audio=<base64/path>` |

---

## Docker

### Build

```bash
./build.sh
# or
docker build -t qwen3-tts-server .
```

Custom CUDA / PyTorch versions:
```bash
CUDA_VERSION=12.6.3-cudnn9-devel-ubuntu22.04 TORCH_INDEX=cu126 ./build.sh
```

### Run

```bash
MODEL_PATH=/path/to/model ./run.sh

# With options
MODEL_PATH=/path/to/model \
  VOICE_DIR=/path/to/voices \
  PORT=9000 \
  ATTN_IMPL=flash_attention_2 \
  ./run.sh
```

### Docker Compose

```bash
# Basic
MODEL_PATH=/path/to/model docker compose up

# With .env file
cat > .env <<EOF
MODEL_PATH=/data/models/qwen3-tts-checkpoint
VOICE_DIR=/data/voices
PORT=8000
ATTN_IMPL=sdpa
DTYPE=bfloat16
EOF
docker compose up -d
```

The server will be available at `http://localhost:8000`.

---

## Fine-tuning

See [finetuning/README.md](finetuning/README.md) for the complete fine-tuning guide, including Arabic language support.

---

## Why This Exists

From the official Qwen3-TTS README:
> Now only offline inference is supported. Online serving will be supported later.

This fork provides streaming now, without waiting for vLLM-Omni updates.
