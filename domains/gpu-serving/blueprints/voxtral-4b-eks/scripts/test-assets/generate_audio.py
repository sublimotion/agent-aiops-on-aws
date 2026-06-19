#!/usr/bin/env python3
"""Generate 3 deterministic test audio files (3s, 10s, 30s) at 16 kHz mono PCM.

The signal is a chirp + tone mixture so it's non-trivial for the encoder.
This is NOT real speech — Voxtral will likely produce empty / nonsense
transcription text, but that's fine for *perf* benchmarking (we measure
audio-seconds processed per wall-second; transcription quality is a separate
LibriSpeech-gated workload that's deferred this session).
"""
from __future__ import annotations
import math
import struct
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
SR = 16000  # Voxtral internal Mel rate

DURATIONS = {"short-3s": 3, "medium-10s": 10, "long-30s": 30}


def make_chirp(duration_s: int, sr: int = SR) -> bytes:
    n = duration_s * sr
    f0, f1 = 220.0, 1760.0  # A3 -> A6
    samples = bytearray()
    for i in range(n):
        t = i / sr
        # linear chirp
        f = f0 + (f1 - f0) * (t / duration_s)
        # phase = integral of f
        phase = 2 * math.pi * (f0 * t + 0.5 * (f1 - f0) * t * t / duration_s)
        # AM envelope (~3 Hz wobble) so it's not a pure tone
        env = 0.6 + 0.3 * math.sin(2 * math.pi * 3.0 * t)
        s = int(32767 * 0.5 * env * math.sin(phase))
        samples += struct.pack("<h", s)
    return bytes(samples)


def write_wav(path: Path, duration_s: int) -> None:
    pcm = make_chirp(duration_s)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(SR)
        w.writeframes(pcm)


def main() -> None:
    for name, dur in DURATIONS.items():
        out = HERE / f"{name}.wav"
        write_wav(out, dur)
        size_kb = out.stat().st_size / 1024
        print(f"wrote {out.name}: {dur}s, {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
