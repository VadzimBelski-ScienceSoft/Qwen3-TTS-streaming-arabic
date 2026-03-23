# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import os
import shutil

import torch
from accelerate import Accelerator
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

target_speaker_embedding = None
def train():
    global target_speaker_embedding

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Path to a checkpoint to resume talker weights from "
                             "(base model must still be provided via --init_model_path)")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable Weights & Biases logging (requires wandb login)")
    args = parser.parse_args()

    loggers = ["tensorboard", "wandb"] if args.use_wandb else ["tensorboard"]
    accelerator = Accelerator(
        gradient_accumulation_steps=4,
        mixed_precision="bf16",
        log_with=loggers,
        project_dir=args.output_model_path,
    )
    accelerator.init_trackers(
        project_name="qwen3-tts-finetune",
        config=vars(args),
    )

    MODEL_PATH = args.init_model_path
    MODEL_LOCAL_PATH = MODEL_PATH
    ARABIC_LANG_ID = 2072

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    if args.resume_checkpoint:
        from safetensors.torch import load_file as load_safetensors
        ckpt_path = os.path.join(args.resume_checkpoint, "model.safetensors")
        ckpt_state = load_safetensors(ckpt_path, device="cpu")
        missing, unexpected = qwen3tts.model.load_state_dict(ckpt_state, strict=False)
        accelerator.print(
            f"Loaded talker weights from {ckpt_path} — "
            f"missing: {len(missing)}, unexpected: {len(unexpected)}"
        )

    # Register Arabic language ID and initialise its codec embedding from the
    # mean of all existing language embeddings (warm start instead of random).
    config.talker_config.codec_language_id['arabic'] = ARABIC_LANG_ID
    with torch.no_grad():
        codec_emb = qwen3tts.model.talker.model.codec_embedding
        existing_ids = [v for k, v in config.talker_config.codec_language_id.items()
                        if k != 'arabic']
        avg = codec_emb.weight[existing_ids].float().mean(0).to(codec_emb.weight.dtype)
        codec_emb.weight[ARABIC_LANG_ID] = avg
    accelerator.print(f"Arabic language embedding initialised at codec token {ARABIC_LANG_ID}")

    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn
    )

    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    num_epochs = args.num_epochs
    model.train()
    global_step = 0

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                ref_mels = batch['ref_mels']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']

                speaker_embedding = model.speaker_encoder(
                    ref_mels.to(model.device).to(model.dtype)
                ).detach()
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                input_codec_embedding[:, 7, :] = speaker_embedding  # speaker slot shifted to pos 7

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, 1:]]
                talker_codec_ids = codec_ids[codec_mask]

                sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(
                    talker_codec_ids, talker_hidden_states
                )

                loss = outputs.loss + 0.3 * sub_talker_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            global_step += 1
            accelerator.log(
                {
                    "train/loss": loss.item(),
                    "train/talker_loss": outputs.loss.item(),
                    "train/sub_talker_loss": sub_talker_loss.item(),
                    "train/epoch": epoch,
                },
                step=global_step,
            )
            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        if accelerator.is_main_process:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            shutil.copytree(MODEL_LOCAL_PATH, output_dir, dirs_exist_ok=True)

            input_config_file = os.path.join(MODEL_LOCAL_PATH, "config.json")
            output_config_file = os.path.join(output_dir, "config.json")
            with open(input_config_file, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            config_dict["tts_model_type"] = "custom_voice"
            talker_config = config_dict.get("talker_config", {})
            talker_config["spk_id"] = {
                args.speaker_name: 3000
            }
            talker_config["spk_is_dialect"] = {
                args.speaker_name: False
            }
            # Persist Arabic language ID so the saved model is self-contained
            lang_ids = dict(talker_config.get("codec_language_id", {}))
            lang_ids["arabic"] = ARABIC_LANG_ID
            talker_config["codec_language_id"] = lang_ids
            config_dict["talker_config"] = talker_config

            with open(output_config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}

            drop_prefix = "speaker_encoder"
            keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
            for k in keys_to_drop:
                del state_dict[k]

            weight = state_dict['talker.model.codec_embedding.weight']
            state_dict['talker.model.codec_embedding.weight'][3000] = (
                target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
            )
            save_path = os.path.join(output_dir, "model.safetensors")
            save_file(state_dict, save_path)

if __name__ == "__main__":
    train()
