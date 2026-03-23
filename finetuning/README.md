## Fine-Tuning Qwen3-TTS-12Hz-1.7B/0.6B-Base

Single-speaker fine-tuning with Arabic language support.

Run `pip install -e .` from the repo root first, then follow the steps below.

---

### 1) Input JSONL format

Prepare your training file as a JSONL (one JSON object per line). Each line must contain:

- `audio`: path to the target training audio (wav)
- `text`: transcript corresponding to `audio`
- `ref_audio`: path to the reference speaker audio (wav)
- `language` _(optional)_: explicit language tag — `english`, `arabic`, etc.
  If omitted, language is auto-detected from the script (Unicode Arabic block detection).

Example:
```jsonl
{"audio":"./data/utt0001.wav","text":"其实我真的有发现，我是一个特别善于观察别人情绪的人。","ref_audio":"./data/ref.wav"}
{"audio":"./data/utt0002.wav","text":"She said she would be here by noon.","ref_audio":"./data/ref.wav"}
{"audio":"./data/utt0003.wav","text":"مرحبًا، كيف حالك؟","ref_audio":"./data/ref.wav","language":"arabic"}
```

`ref_audio` recommendation:
- Use the same `ref_audio` for all samples for best speaker consistency.

---

### 2) Prepare data (extract `audio_codes`)

Convert your raw JSONL into a training JSONL that includes `audio_codes`:

```bash
python prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl
```

---

### 3) Fine-tune

```bash
python sft_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path output \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 2 \
  --lr 2e-5 \
  --num_epochs 3 \
  --speaker_name my_speaker
```

#### All flags

| Flag | Default | Description |
|------|---------|-------------|
| `--init_model_path` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Base model path or HF repo ID |
| `--output_model_path` | `output` | Directory to save checkpoints |
| `--train_jsonl` | _(required)_ | Path to `train_with_codes.jsonl` |
| `--batch_size` | `2` | Samples per GPU |
| `--lr` | `2e-5` | Learning rate |
| `--num_epochs` | `3` | Training epochs |
| `--speaker_name` | `speaker_test` | Name embedded in the saved config |
| `--resume_checkpoint` | `None` | Resume talker weights from a previous checkpoint |
| `--use_wandb` | off | Enable Weights & Biases logging (requires `wandb login`) |

Checkpoints are written to `output/checkpoint-epoch-{N}`.

#### Arabic support

The training script automatically registers Arabic (token ID `2072`) in the codec language embedding table, initialised from the mean of all other language embeddings as a warm start. No extra flags are needed — include Arabic samples in your JSONL and they will be routed correctly.

The saved `config.json` at each checkpoint includes `"arabic": 2072` under `codec_language_id` so the model is self-contained at inference time.

---

### 4) Quick inference test

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

tts = Qwen3TTSModel.from_pretrained(
    "output/checkpoint-epoch-2",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

wavs, sr = tts.generate_custom_voice(
    text="She said she would be here by noon.",
    speaker="my_speaker",
)
sf.write("output.wav", wavs[0], sr)
```

Arabic inference:
```python
wavs, sr = tts.generate_custom_voice(
    text="مرحبًا، كيف حالك؟",
    speaker="my_speaker",
    language="arabic",
)
sf.write("output_ar.wav", wavs[0], sr)
```

---

### One-click shell script

```bash
#!/usr/bin/env bash
set -e

DEVICE="cuda:0"
TOKENIZER_MODEL_PATH="Qwen/Qwen3-TTS-Tokenizer-12Hz"
INIT_MODEL_PATH="Qwen/Qwen3-TTS-12Hz-1.7B-Base"

RAW_JSONL="train_raw.jsonl"
TRAIN_JSONL="train_with_codes.jsonl"
OUTPUT_DIR="output"

BATCH_SIZE=2
LR=2e-5
EPOCHS=3
SPEAKER_NAME="my_speaker"

python prepare_data.py \
  --device ${DEVICE} \
  --tokenizer_model_path ${TOKENIZER_MODEL_PATH} \
  --input_jsonl ${RAW_JSONL} \
  --output_jsonl ${TRAIN_JSONL}

python sft_12hz.py \
  --init_model_path ${INIT_MODEL_PATH} \
  --output_model_path ${OUTPUT_DIR} \
  --train_jsonl ${TRAIN_JSONL} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --num_epochs ${EPOCHS} \
  --speaker_name ${SPEAKER_NAME}
```
