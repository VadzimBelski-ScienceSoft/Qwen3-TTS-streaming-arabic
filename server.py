#!/usr/bin/env python3
"""
OpenAI-compatible TTS server for Qwen3-TTS (CustomVoice / VoiceDesign / Base).

/v1/audio/speech streams audio as Server-Sent Events (SSE) with hex-encoded
int16 PCM — same format expected by openai_client_tts.py.

Model types supported
---------------------
  custom_voice  – speaker-id model (voice=<speaker>, instructions=<emotion>)
  voice_design  – instruction-controlled model (instructions=<voice description>)
  base          – voice-clone model (voice registry or per-request ref_audio)

Usage
-----
    python server.py --model ./output/checkpoint-epoch-9 --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import struct
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("tts_server")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_model = None
_model_type: str = ""           # "custom_voice" | "voice_design" | "base"
_voice_registry: Dict[str, "VoiceEntry"] = {}
_default_voice_name: str = "default"
_sample_rate: int = 24000
_executor = ThreadPoolExecutor(max_workers=1)
_args = None


# ---------------------------------------------------------------------------
# Voice registry (Base model only)
# ---------------------------------------------------------------------------

@dataclass
class VoiceEntry:
    name: str
    audio_path: str
    transcript: Optional[str]
    prompt_item: Optional[object] = field(default=None, repr=False)


def _load_voice_dir(voice_dir: str) -> Dict[str, VoiceEntry]:
    entries: Dict[str, VoiceEntry] = {}
    p = Path(voice_dir)
    if not p.is_dir():
        logger.warning("--voice-dir '%s' not found", voice_dir)
        return entries
    for wav in sorted(p.glob("*.wav")):
        name = wav.stem.lower()
        txt = wav.with_suffix(".txt")
        transcript = (txt.read_text(encoding="utf-8").strip() or None) if txt.exists() else None
        entries[name] = VoiceEntry(name=name, audio_path=str(wav), transcript=transcript)
        logger.info("Registered voice '%s' (transcript=%s)", name, transcript is not None)
    return entries


def _precache_voice_prompts() -> None:
    for name, entry in _voice_registry.items():
        try:
            items = _model.create_voice_clone_prompt(
                ref_audio=entry.audio_path,
                ref_text=entry.transcript,
                x_vector_only_mode=(entry.transcript is None),
            )
            entry.prompt_item = items[0]
            logger.info("Voice '%s' cached.", name)
        except Exception as e:
            logger.error("Failed to cache voice '%s': %s", name, e)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _f32_to_i16(audio: np.ndarray) -> np.ndarray:
    return np.clip(audio * 32767, -32768, 32767).astype(np.int16)


def _chunks_to_wav_bytes(chunks: List[np.ndarray], sr: int) -> bytes:
    audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(_f32_to_i16(audio).tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse_audio_event(pcm_i16: np.ndarray) -> bytes:
    payload = json.dumps({
        "choices": [{"delta": {"audio": pcm_i16.tobytes().hex()}, "finish_reason": None}]
    })
    return f"data: {payload}\n\n".encode()


_SSE_DONE = b"data: [DONE]\n\n"


def _sse_error(msg: str) -> bytes:
    return f"data: {json.dumps({'error': msg})}\n\n".encode()


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

class SpeechRequest(BaseModel):
    model: str = Field(default="tts-1")
    input: str = Field(..., description="Text to synthesize.")
    voice: str = Field(default="default", description="Speaker name (CustomVoice) or voice name (Base).")
    response_format: str = Field(default="wav")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)

    # Qwen3-TTS extensions
    task_type: Optional[str] = Field(default=None, description="'CustomVoice', 'VoiceDesign', or 'Base'.")
    language: Optional[str] = Field(default=None)
    instructions: Optional[str] = Field(default=None, description="Emotion/style (CustomVoice) or voice description (VoiceDesign).")
    stream: bool = Field(default=True)

    # Base per-request reference
    ref_audio: Optional[str] = Field(default=None, description="Base64 or URL reference audio (Base model).")
    ref_text: Optional[str] = Field(default=None)
    x_vector_only_mode: bool = Field(default=False)

    # Streaming
    emit_every_frames: int = Field(default=4, ge=1, le=200)
    decode_window_frames: int = Field(default=80, ge=8, le=512)

    # Generation
    max_new_tokens: Optional[int] = Field(default=None)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Qwen3-TTS OpenAI-compatible server", version="1.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None, "model_type": _model_type}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "tts-1",    "object": "model", "created": 1700000000, "owned_by": "qwen3-tts"},
            {"id": "tts-1-hd", "object": "model", "created": 1700000000, "owned_by": "qwen3-tts"},
        ],
    }


@app.get("/v1/voices")
async def list_voices():
    if _model_type == "custom_voice":
        return {"model_type": _model_type, "voices": _model.get_supported_speakers() or []}
    elif _model_type == "base":
        return {
            "model_type": _model_type,
            "default": _default_voice_name,
            "voices": [
                {"name": n, "has_transcript": e.transcript is not None}
                for n, e in _voice_registry.items()
            ],
        }
    else:
        return {"model_type": _model_type, "voices": [], "note": "Use 'instructions' to describe voice."}


# ---------------------------------------------------------------------------
# Streaming helpers — all call stream_generate_pcm() for real-time PCM
# ---------------------------------------------------------------------------

def _stream_custom_voice(req: SpeechRequest) -> Iterator[Tuple[np.ndarray, int]]:
    """
    Real-time streaming for CustomVoice model via stream_generate_pcm().
    Builds tokenised inputs then delegates directly to the low-level PCM
    streaming method so audio is yielded as it is generated.
    """
    language = req.language or "Auto"
    _model._validate_languages([language])

    input_ids = _model._tokenize_texts([_model._build_assistant_text(req.input)])

    instruct_ids: List[Optional[Any]] = [None]
    if req.instructions and req.instructions.strip():
        instruct_ids = [_model._tokenize_texts([_model._build_instruct_text(req.instructions)])[0]]

    gen_kw = _model._merge_generate_kwargs(
        **(({"max_new_tokens": req.max_new_tokens}) if req.max_new_tokens else {})
    )

    return _model.model.stream_generate_pcm(
        input_ids=input_ids,
        instruct_ids=instruct_ids,
        languages=[language],
        speakers=[req.voice],
        emit_every_frames=req.emit_every_frames,
        decode_window_frames=req.decode_window_frames,
        overlap_samples=0,
        do_sample=gen_kw.get("do_sample", True),
        top_k=gen_kw.get("top_k", 50),
        top_p=gen_kw.get("top_p", 1.0),
        temperature=gen_kw.get("temperature", 0.9),
        subtalker_dosample=gen_kw.get("subtalker_dosample", True),
        subtalker_top_k=gen_kw.get("subtalker_top_k", 50),
        subtalker_top_p=gen_kw.get("subtalker_top_p", 1.0),
        subtalker_temperature=gen_kw.get("subtalker_temperature", 0.9),
    )


def _stream_voice_design(req: SpeechRequest) -> Iterator[Tuple[np.ndarray, int]]:
    """
    Real-time streaming for VoiceDesign model via stream_generate_pcm().
    """
    language = req.language or "Auto"
    _model._validate_languages([language])

    input_ids = _model._tokenize_texts([_model._build_assistant_text(req.input)])

    instruct_ids: List[Optional[Any]] = [None]
    if req.instructions and req.instructions.strip():
        instruct_ids = [_model._tokenize_texts([_model._build_instruct_text(req.instructions)])[0]]

    gen_kw = _model._merge_generate_kwargs(
        **(({"max_new_tokens": req.max_new_tokens}) if req.max_new_tokens else {})
    )

    return _model.model.stream_generate_pcm(
        input_ids=input_ids,
        instruct_ids=instruct_ids,
        languages=[language],
        emit_every_frames=req.emit_every_frames,
        decode_window_frames=req.decode_window_frames,
        overlap_samples=0,
        do_sample=gen_kw.get("do_sample", True),
        top_k=gen_kw.get("top_k", 50),
        top_p=gen_kw.get("top_p", 1.0),
        temperature=gen_kw.get("temperature", 0.9),
        subtalker_dosample=gen_kw.get("subtalker_dosample", True),
        subtalker_top_k=gen_kw.get("subtalker_top_k", 50),
        subtalker_top_p=gen_kw.get("subtalker_top_p", 1.0),
        subtalker_temperature=gen_kw.get("subtalker_temperature", 0.9),
    )


def _stream_base(req: SpeechRequest) -> Iterator[Tuple[np.ndarray, int]]:
    voice_name = req.voice.lower()
    if voice_name in ("default", "alloy", "echo", "fable", "onyx", "nova", "shimmer"):
        voice_name = _default_voice_name

    if req.ref_audio:
        voice_clone_prompt = None
        ref_audio = req.ref_audio
        ref_text = req.ref_text
        x_vec = req.x_vector_only_mode
    else:
        entry = _voice_registry.get(voice_name)
        if entry is None:
            raise ValueError(
                f"Voice '{voice_name}' not found. Available: {list(_voice_registry.keys())}"
            )
        if entry.prompt_item is None:
            raise RuntimeError(f"Voice '{voice_name}' prompt not cached.")
        voice_clone_prompt = entry.prompt_item
        ref_audio = None
        ref_text = None
        x_vec = False

    kwargs = {}
    if req.max_new_tokens:
        kwargs["max_new_tokens"] = req.max_new_tokens

    return _model.stream_generate_voice_clone(
        text=req.input,
        language=req.language or "Auto",
        ref_audio=ref_audio,
        ref_text=ref_text,
        x_vector_only_mode=x_vec,
        voice_clone_prompt=voice_clone_prompt,
        emit_every_frames=req.emit_every_frames,
        decode_window_frames=req.decode_window_frames,
        overlap_samples=0,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# /v1/audio/speech
# ---------------------------------------------------------------------------

@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest) -> Response:
    if not _model:
        raise HTTPException(503, "Model not loaded")
    if not req.input or not req.input.strip():
        raise HTTPException(400, "'input' must be non-empty")

    # Resolve effective task type
    raw_task = (req.task_type or _model_type).lower().replace("_", "").replace("-", "")
    if raw_task in ("customvoice", "custom"):
        task = "custom_voice"
    elif raw_task in ("voicedesign", "design"):
        task = "voice_design"
    else:
        task = "base"

    fmt = req.response_format.lower()
    use_sse = req.stream or fmt == "sse"
    t0 = time.time()

    logger.info("Speech: task=%s voice=%s fmt=%s stream=%s len=%d",
                task, req.voice, fmt, use_sse, len(req.input))

    # Pick the streaming generator for the requested task type.
    # All three paths call stream_generate_pcm() under the hood.
    _stream_dispatch = {
        "custom_voice": _stream_custom_voice,
        "voice_design": _stream_voice_design,
        "base":         _stream_base,
    }
    _stream_fn = _stream_dispatch[task]

    # ---------------------------------------------------------------
    # SSE streaming — yields PCM chunks as they are produced
    # ---------------------------------------------------------------
    async def _sse_gen() -> AsyncGenerator[bytes, None]:
        loop = asyncio.get_event_loop()

        try:
            gen = await loop.run_in_executor(_executor, lambda: _stream_fn(req))
        except Exception as e:
            logger.exception("Stream init error")
            yield _sse_error(str(e))
            yield _SSE_DONE
            return

        def _next(it):
            try:
                return next(it)
            except StopIteration:
                return None

        first_ts = None
        num = 0
        while True:
            result = await loop.run_in_executor(_executor, _next, gen)
            if result is None:
                break
            chunk_f32, sr = result
            if first_ts is None:
                first_ts = time.time() - t0
                logger.info("First chunk latency: %.3fs", first_ts)
            yield _sse_audio_event(_f32_to_i16(chunk_f32))
            num += 1

        yield _SSE_DONE
        logger.info("Stream done: %d chunks in %.2fs", num, time.time() - t0)

    if use_sse:
        return StreamingResponse(
            _sse_gen(),
            media_type="text/event-stream",
            headers={"X-Sample-Rate": str(_sample_rate), "Cache-Control": "no-cache"},
        )

    # ---------------------------------------------------------------
    # Non-streaming — collect all chunks, return complete WAV
    # ---------------------------------------------------------------
    async def _collect() -> bytes:
        loop = asyncio.get_event_loop()

        def _run():
            chunks: List[np.ndarray] = []
            sr = _sample_rate
            for c, s in _stream_fn(req):
                chunks.append(c)
                sr = s
            return _chunks_to_wav_bytes(chunks, sr) if chunks else b""

        return await loop.run_in_executor(_executor, _run)

    return Response(content=await _collect(), media_type="audio/wav")


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OpenAI-compatible Qwen3-TTS server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn-impl", default="flash_attention_2")
    p.add_argument("--voice-dir", default="./voices",
                   help="Directory with .wav (+ .txt) files for Base model.")
    p.add_argument("--default-voice", default=None)
    p.add_argument("--optimize", action="store_true",
                   help="Enable torch.compile optimizations (Base model only).")
    p.add_argument("--optimize-decode-window", type=int, default=80)
    p.add_argument("--no-cuda-graphs", action="store_true")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--log-level", default="info",
                   choices=["debug", "info", "warning", "error"])
    return p.parse_args()


def _startup(args) -> None:
    global _model, _model_type, _voice_registry, _default_voice_name
    global _sample_rate, _executor

    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    logger.info("Loading '%s' on %s dtype=%s ...", args.model, args.device, args.dtype)
    t0 = time.time()
    from qwen_tts import Qwen3TTSModel

    _model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=dtype_map[args.dtype],
        attn_implementation=args.attn_impl,
    )
    _model_type = getattr(_model.model, "tts_model_type", "base")
    logger.info("Loaded in %.1fs — model_type: %s", time.time() - t0, _model_type)

    sr_attr = getattr(getattr(_model, "model", None), "sample_rate", None)
    if sr_attr:
        _sample_rate = int(sr_attr)
    logger.info("Sample rate: %d Hz", _sample_rate)

    if _model_type == "base":
        _voice_registry = _load_voice_dir(args.voice_dir)
        if _voice_registry:
            first = next(iter(_voice_registry))
            _default_voice_name = (
                args.default_voice.lower()
                if args.default_voice and args.default_voice.lower() in _voice_registry
                else first
            )
            logger.info("Default voice: '%s'", _default_voice_name)
            _precache_voice_prompts()
        else:
            logger.warning("No voices loaded — requests must include ref_audio inline.")

    elif _model_type == "custom_voice":
        speakers = _model.get_supported_speakers()
        if speakers:
            logger.info("Speakers (%d): %s%s", len(speakers),
                        speakers[:10], " ..." if len(speakers) > 10 else "")

    elif _model_type == "voice_design":
        logger.info("VoiceDesign model — set 'instructions' to describe the desired voice.")

    if args.optimize:
        # Enable TF32 for matrix multiplications (free speedup on Ampere+ GPUs)
        torch.set_float32_matmul_precision("high")
        # Skip CUDA graph captures for dynamic shapes — only fixed-size windows get graphs,
        # preventing the "9 distinct sizes" explosion and its associated overhead.
        torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True

        logger.info("Enabling streaming optimizations (model_type=%s) ...", _model_type)
        t1 = time.time()
        _model.enable_streaming_optimizations(
            decode_window_frames=args.optimize_decode_window,
            use_compile=True,
            use_cuda_graphs=not args.no_cuda_graphs,
            compile_mode="reduce-overhead",
            use_fast_codebook=False,
            compile_codebook_predictor=True,
            compile_talker=True,
        )
        logger.info("Optimizations ready in %.1fs", time.time() - t1)

    _executor = ThreadPoolExecutor(max_workers=args.workers)
    logger.info("Server ready on %s:%d", args.host, args.port)


if __name__ == "__main__":
    _args = _parse_args()
    _startup(_args)
    uvicorn.run(app, host=_args.host, port=_args.port, log_level=_args.log_level, lifespan="off")
