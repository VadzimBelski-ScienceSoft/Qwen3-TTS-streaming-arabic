"""OpenAI-compatible client for Qwen3-TTS via /v1/audio/speech endpoint.

Examples:
    python openai_speech_client.py --text "Hello, how are you?" --voice emirati_speaker
    python openai_speech_client.py --text "I'm so happy!" --voice emirati_speaker \\
        --instructions "Speak with excitement"
    python openai_speech_client.py --text "Hello world" --task-type VoiceDesign \\
        --instructions "A warm, friendly female voice"
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import threading
import time
import wave

import httpx
import numpy as np

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

DEFAULT_API_BASE = "http://localhost:8000"
DEFAULT_API_KEY  = "EMPTY"


# ---------------------------------------------------------------------------
# Lock-protected ring buffer
# ---------------------------------------------------------------------------

class _RingBuffer:
    def __init__(self, capacity: int):
        self._buf  = np.zeros(capacity, dtype=np.float32)
        self._cap  = capacity
        self._head = 0
        self._tail = 0
        self._size = 0
        self._lock = threading.Lock()

    def write(self, data: np.ndarray) -> None:
        data = np.asarray(data, dtype=np.float32).ravel()
        n = len(data)
        with self._lock:
            space = self._cap - self._size
            if n > space:
                skip = n - space
                self._tail = (self._tail + skip) % self._cap
                self._size -= skip
            end = (self._head + n) % self._cap
            if end > self._head:
                self._buf[self._head:end] = data
            else:
                first = self._cap - self._head
                self._buf[self._head:] = data[:first]
                self._buf[:end]        = data[first:]
            self._head  = end
            self._size += n

    def read(self, n: int) -> tuple:
        """Returns (array of exactly n float32 samples, count actually from buffer)."""
        out = np.zeros(n, dtype=np.float32)
        with self._lock:
            avail = min(n, self._size)
            if avail == 0:
                return out, 0
            end = (self._tail + avail) % self._cap
            if end > self._tail:
                out[:avail] = self._buf[self._tail:end]
            else:
                first = self._cap - self._tail
                out[:first]      = self._buf[self._tail:]
                out[first:avail] = self._buf[:end]
            self._tail  = end
            self._size -= avail
        return out, avail

    @property
    def available(self) -> int:
        with self._lock:
            return self._size


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_audio_to_base64(path: str) -> str:
    ext  = os.path.splitext(path)[1].lower().lstrip(".")
    mime = {"wav": "audio/wav", "mp3": "audio/mpeg",
            "flac": "audio/flac", "ogg": "audio/ogg"}.get(ext, "audio/wav")
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_tts_generation(args) -> None:
    SAMPLE_RATE   = 24000
    BLOCKSIZE     = 2400          # 100 ms per callback block
    PREBUFFER_SEC = 2.0           # buffer 2 s before opening audio device
    RING_CAPACITY = SAMPLE_RATE * 60

    payload: dict = {
        "model":           args.model,
        "input":           args.text,
        "voice":           args.voice,
        "response_format": "pcm",
        "stream":          True,
    }
    if args.instructions:   payload["instructions"]   = args.instructions
    if args.task_type:      payload["task_type"]       = args.task_type
    if args.language:       payload["language"]        = args.language
    if args.max_new_tokens: payload["max_new_tokens"]  = args.max_new_tokens
    if args.ref_audio:
        payload["ref_audio"] = (args.ref_audio if args.ref_audio.startswith("http")
                                else encode_audio_to_base64(args.ref_audio))
    if args.ref_text:       payload["ref_text"]        = args.ref_text
    if args.x_vector_only:  payload["x_vector_only_mode"] = True

    print(f"Task:  {args.task_type or 'CustomVoice'}")
    print(f"Voice: {args.voice}")
    print(f"Text:  {args.text[:100]}{'...' if len(args.text) > 100 else ''}")
    print("Connecting ...")

    ring        = _RingBuffer(RING_CAPACITY)
    all_pcm     = bytearray()
    t0          = time.time()
    first_chunk = None
    chunk_count = 0
    stream_done = threading.Event()
    started     = threading.Event()   # playback device is open
    stream_obj  = [None]              # sd.OutputStream
    underruns   = [0]
    _last_val   = np.zeros(1, dtype=np.float32)

    # ------------------------------------------------------------------
    # sounddevice callback (audio thread — must never block)
    # ------------------------------------------------------------------
    def _audio_cb(outdata, frames, _ti, _st):
        samples, got = ring.read(frames)
        if got == frames:
            # full buffer: clean read
            outdata[:, 0] = samples
            _last_val[0] = samples[-1]
        elif got > 0:
            # partial: fill gap with a short fade to zero to avoid hard click
            outdata[:got, 0] = samples[:got]
            fade = np.linspace(_last_val[0], 0.0, frames - got, dtype=np.float32)
            outdata[got:, 0] = fade
            _last_val[0] = 0.0
            underruns[0] += 1
        else:
            # completely empty
            outdata.fill(0)
            if stream_done.is_set():
                raise sd.CallbackStop
            underruns[0] += 1

    def _start_stream():
        s = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=_audio_cb,
            blocksize=BLOCKSIZE,
            latency="low",
        )
        s.start()
        stream_obj[0] = s
        started.set()

    # ------------------------------------------------------------------
    # HTTP streaming
    # ------------------------------------------------------------------
    api_url = f"{args.api_base}/v1/audio/speech"
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {args.api_key}"}

    with httpx.Client(timeout=300.0) as client:
        with client.stream("POST", api_url, json=payload, headers=headers) as resp:
            if resp.status_code != 200:
                print(f"HTTP {resp.status_code}: {resp.text}")
                return

            for line in resp.iter_lines():
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                if line.startswith("data: [DONE]"):
                    break
                if not line.startswith("data: "):
                    continue
                try:
                    data = json.loads(line[6:])
                except Exception:
                    continue

                audio_hex = (data.get("choices") or [{}])[0].get("delta", {}).get("audio")
                if not audio_hex:
                    continue

                chunk_bytes = bytes.fromhex(audio_hex)
                all_pcm.extend(chunk_bytes)
                chunk_count += 1

                chunk_f32 = (np.frombuffer(chunk_bytes, dtype=np.int16)
                             .astype(np.float32) / 32767.0)
                ring.write(chunk_f32)

                now = time.time()
                if first_chunk is None:
                    first_chunk = now
                    print(f"First chunk in {1000*(now - t0):.0f} ms  "
                          f"({len(chunk_f32)/SAMPLE_RATE*1000:.0f} ms audio)")

                buf_ms = ring.available / SAMPLE_RATE * 1000

                # Start playback once we have PREBUFFER_SEC of audio
                if (HAS_SOUNDDEVICE and stream_obj[0] is None and
                        ring.available >= int(SAMPLE_RATE * PREBUFFER_SEC)):
                    _start_stream()
                    print(f"  Playback started  buf={buf_ms:.0f} ms")

                print(f"  chunk #{chunk_count:3d}  buf={buf_ms:.0f} ms"
                      + (f"  UNDERRUNS={underruns[0]}" if underruns[0] else ""))

    stream_done.set()

    # Start playback even if prebuffer was never reached (short utterance)
    if HAS_SOUNDDEVICE and stream_obj[0] is None and ring.available > 0:
        _start_stream()
        print("  Playback started (end of stream, short utterance)")

    # Wait for drain
    if stream_obj[0] is not None:
        while stream_obj[0].active and ring.available > 0:
            time.sleep(0.05)
        time.sleep(0.15)
        stream_obj[0].stop()
        stream_obj[0].close()

    if underruns[0]:
        print(f"\nWARNING: {underruns[0]} audio underrun(s) — "
              "increase PREBUFFER_SEC in the script if clicks persist")

    # Save WAV
    out_path = args.output or "tts_output.wav"
    if not out_path.endswith(".wav"):
        out_path = os.path.splitext(out_path)[0] + ".wav"
    if all_pcm:
        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(bytes(all_pcm))
        dur = len(all_pcm) // 2 / SAMPLE_RATE
        print(f"Saved {dur:.2f}s -> {out_path}  "
              f"(total {time.time()-t0:.2f}s, {chunk_count} chunks)")
    else:
        print("No audio received.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenAI-compatible Qwen3-TTS client")
    p.add_argument("--api-base",        default=DEFAULT_API_BASE)
    p.add_argument("--api-key",         default=DEFAULT_API_KEY)
    p.add_argument("--model", "-m",     default="tts-1")
    p.add_argument("--task-type", "-t", default=None,
                   choices=["CustomVoice", "VoiceDesign", "Base"])
    p.add_argument("--text",            required=True)
    p.add_argument("--voice",           default="emirati_speaker")
    p.add_argument("--language",        default=None)
    p.add_argument("--instructions",    default=None)
    p.add_argument("--ref-audio",       default=None)
    p.add_argument("--ref-text",        default=None)
    p.add_argument("--x-vector-only",   action="store_true")
    p.add_argument("--max-new-tokens",  type=int, default=None)
    p.add_argument("--output", "-o",    default=None)
    return p.parse_args()


if __name__ == "__main__":
    run_tts_generation(_parse_args())
