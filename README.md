# Qwen3-TTS Streaming — Arabic Edition

Streaming inference for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) with Arabic language support, an OpenAI-compatible server, and Docker deployment for NVIDIA DGX Spark.

The official team mentions "Extreme Low-Latency Streaming Generation" in their paper but the streaming code was never released. This fork adds real streaming, Arabic fine-tuning, and a production-ready server on top of [dffdeeq/Qwen3-TTS-streaming](https://github.com/dffdeeq/Qwen3-TTS-streaming).

**What's added over upstream:**
- Real-time PCM streaming via `stream_generate_pcm`
- ~6x inference speedup vs upstream qwen-tts
- OpenAI-compatible `/v1/audio/speech` SSE server
- **Arabic language fine-tuning and inference**
- Docker support for NVIDIA DGX Spark (ARM64 / Grace Blackwell)

Based on [dffdeeq/Qwen3-TTS-streaming](https://github.com/dffdeeq/Qwen3-TTS-streaming) and [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS).

---

## Arabic Support

The base Qwen3-TTS-12Hz model supports a fixed set of languages defined in the model config. Arabic is not included by default. This fork adds Arabic as a first-class language through targeted changes to both the fine-tuning pipeline and the inference server.

### How it works

#### 1. Language embedding registration

The model's `talker` component uses a `codec_embedding` lookup table to condition generation on the input language. Each language maps to a token ID in `config.talker_config.codec_language_id`.

During fine-tuning, Arabic is registered at token ID `2072` with a warm start — its embedding is initialised as the mean of all existing language embeddings rather than random noise:

```python
ARABIC_LANG_ID = 2072
config.talker_config.codec_language_id['arabic'] = ARABIC_LANG_ID

codec_emb = model.talker.model.codec_embedding
existing_ids = [v for k, v in config.talker_config.codec_language_id.items() if k != 'arabic']
avg = codec_emb.weight[existing_ids].float().mean(0)
codec_emb.weight[ARABIC_LANG_ID] = avg
```

This gives Arabic a sensible starting point and significantly speeds up convergence.

#### 2. Language-conditioned codec prefix (4-token think block)

The upstream fine-tuning code uses a 3-token codec prefix (no-think, bos, eos). To carry language information into every generation step, this fork replaces it with a 4-token language-conditioned think block:

```
pos 3: codec_think_id
pos 4: codec_think_bos_id
pos 5: lang_id          ← Arabic token 2072 for Arabic samples
pos 6: codec_think_eos_id
pos 7: speaker embedding (shifted from pos 6 in upstream)
```

This means every forward pass explicitly encodes which language is being generated, allowing the model to switch cleanly between Arabic and other languages within the same checkpoint.

The `collate_fn` in `dataset.py` builds this sequence automatically per sample. The `max_length` buffer is `+9` instead of the upstream `+8` to accommodate the extra token.

#### 3. Automatic language detection

If a training sample or inference request does not specify a language, it is auto-detected by scanning for Arabic Unicode characters (U+0600–U+06FF):

```python
def _detect_language(self, text: str) -> str:
    for c in text:
        if '\u0600' <= c <= '\u06FF':
            return 'arabic'
    return 'english'
```

This means mixed datasets work without adding `"language"` fields to every JSONL entry.

#### 4. Checkpoint self-containment

When a checkpoint is saved, the Arabic language ID is written back into `config.json` under `codec_language_id` so the saved model is fully self-contained and loads correctly without any external configuration:

```json
"codec_language_id": {
  "chinese": ..., "english": ..., ...,
  "arabic": 2072
}
```

#### 5. Inference server passthrough

The OpenAI-compatible server accepts a `language` field in the request body and passes it directly to the model's generation pipeline. For fine-tuned CustomVoice models, Arabic text is routed through the correct language embedding automatically:

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "مرحبًا، كيف حالك؟",
    "voice": "my_arabic_speaker",
    "language": "arabic",
    "stream": true
  }'
```

---

## Benchmark (RTX 5090)

### Non-streaming (full inference)

<img width="602" height="145" alt="image" src="https://github.com/user-attachments/assets/0cbfcc71-e854-46e2-81bc-ec3955ff3ff0" />

### Streaming

<img width="766" height="183" alt="image" src="https://github.com/user-attachments/assets/f5df9a38-e091-47ae-a08f-ef364f8710ea" />

---

## Installation (Python 3.12)

### 1. Install SOX

**Linux:**
```bash
sudo apt install sox libsox-fmt-all
```

**Windows:** Download from https://sourceforge.net/projects/sox/ and add to PATH.

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
git clone https://github.com/VadzimBelski-ScienceSoft/Qwen3-TTS-streaming-arabic.git
cd Qwen3-TTS-streaming-arabic
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
pip install -e ".[server]"

# Fine-tuned Arabic model
python server.py --model ./output/checkpoint-epoch-9 --host 0.0.0.0 --port 8000

# Or load from HuggingFace Hub
python server.py --model youruser/qwen3-arabic-tts --host 0.0.0.0 --port 8000
```

### Arabic request

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "مرحبًا، كيف حالك؟",
    "voice": "my_arabic_speaker",
    "language": "arabic",
    "stream": true
  }'
```

### Request fields

| Field | Default | Description |
|-------|---------|-------------|
| `input` | required | Text to synthesize |
| `voice` | `default` | Speaker name (CustomVoice) or voice name (Base) |
| `language` | `Auto` | `arabic`, `english`, `chinese`, etc. Auto-detected if omitted |
| `instructions` | — | Emotion/style (CustomVoice) or voice description (VoiceDesign) |
| `stream` | `true` | SSE streaming or full WAV response |
| `task_type` | auto | `CustomVoice`, `VoiceDesign`, or `Base` |
| `emit_every_frames` | `4` | PCM chunks per SSE event |

### Server arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Model path or HF repo ID |
| `--device` | `cuda:0` | CUDA device |
| `--dtype` | `bfloat16` | Weight dtype |
| `--attn-impl` | `flash_attention_2` | Attention backend (`sdpa` if no flash-attn) |
| `--voice-dir` | `./voices` | `.wav` + `.txt` voice files (Base model) |
| `--optimize` | off | Enable `torch.compile` + CUDA graphs |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8000` | Port |

### Supported endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /v1/models` | List models |
| `GET /v1/voices` | List available voices / speakers |
| `POST /v1/audio/speech` | Generate speech |

### Model types

| `task_type` | Description | Key params |
|-------------|-------------|------------|
| `CustomVoice` | Fine-tuned single-speaker | `voice=<speaker_name>`, `instructions=<emotion>` |
| `VoiceDesign` | Instruction-controlled | `instructions=<voice description>` |
| `Base` | Voice cloning | `voice=<registered>` or `ref_audio=<base64>` |

---

## Docker (DGX Spark / ARM64)

The Dockerfile uses the NVIDIA NGC PyTorch base image (`nvcr.io/nvidia/pytorch:25.10-py3`), which ships with CUDA 13, cuDNN, and PyTorch pre-installed for Grace Blackwell ARM64.

### Build

```bash
./build.sh
```

Custom base image:
```bash
BASE_IMAGE=nvcr.io/nvidia/pytorch:24.08-py3 ./build.sh
```

### Run — HuggingFace Hub model

```bash
MODEL=youruser/qwen3-arabic-tts ./run.sh

# Private model
HF_TOKEN=hf_xxx MODEL=youruser/private-model ./run.sh
```

The model is downloaded on first start and cached in a named Docker volume (`qwen3-tts-hf-cache`). Restarts reuse the cache.

### Run — local checkpoint

```bash
MODEL=/path/to/checkpoint-epoch-9 ./run.sh
```

`run.sh` detects the local path, auto-mounts it to `/model`, and sets `MODEL=/model`.

### Docker Compose

```bash
cat > .env <<EOF
MODEL=youruser/qwen3-arabic-tts
HF_TOKEN=hf_xxx
PORT=8000
ATTN_IMPL=sdpa
EOF

docker compose up -d
```

For a local checkpoint, uncomment the volume line in `docker-compose.yml` and set `MODEL=/model`.

---

## Fine-tuning

See [finetuning/README.md](finetuning/README.md) for the complete guide, including:
- JSONL format with optional `language` field
- Arabic warm-start embedding initialisation
- `--resume_checkpoint` for continuing training
- `--use_wandb` for experiment tracking

---

## Why This Exists

From the official Qwen3-TTS README:
> Now only offline inference is supported. Online serving will be supported later.

This fork adds streaming, Arabic, and a production server now.
