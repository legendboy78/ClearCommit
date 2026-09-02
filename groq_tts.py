"""Groq Orpheus text-to-speech backend.

Groq's TTS is the ``canopylabs/orpheus`` family: an OpenAI-compatible
``/audio/speech`` endpoint returning **WAV** audio with a small set of
**preset** voices. Two hard limits shape this implementation:

* Only an **English** model and an **Arabic (Saudi)** model exist, so
  non-English/Arabic targets are warned about (the model will mispronounce).
* Each request is capped at **200 characters**, so longer text is split into
  sentence/word chunks and the resulting WAVs are concatenated.
"""

from __future__ import annotations

import io
import os
import re
import wave
from pathlib import Path
from typing import Optional

from config import (
    GROQ_TTS_MAX_CHARS,
    GROQ_TTS_MODEL_AR,
    GROQ_TTS_MODEL_EN,
    GROQ_TTS_VOICES_AR,
    GROQ_TTS_VOICES_EN,
)
from logger import get_logger
from synthesizer import BaseSynthesizer

logger = get_logger("groq_tts")

try:  # pragma: no cover - import guard
    from groq import Groq, GroqError
except Exception:  # noqa: BLE001
    Groq = None  # type: ignore
    GroqError = Exception  # type: ignore


class GroqSynthesizer(BaseSynthesizer):
    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        voice: str = "troy",
        model: Optional[str] = None,
        max_chars: int = GROQ_TTS_MAX_CHARS,
    ):
        if Groq is None:
            raise RuntimeError("The 'groq' package is required (`pip install groq`).")
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set.")
        self.client = Groq(api_key=key)
        self.voice = voice
        self.model = model
        self.max_chars = max_chars
        self.output_format = "wav"

    # ------------------------------------------------------------------ #
    def _select_model(self, target_language: Optional[str]) -> str:
        if self.model:
            return self.model
        return GROQ_TTS_MODEL_AR if target_language == "ar" else GROQ_TTS_MODEL_EN

    def _select_voice(self, target_language: Optional[str]) -> str:
        if target_language == "ar":
            return self.voice if self.voice in GROQ_TTS_VOICES_AR else "fahad"
        return self.voice if self.voice in GROQ_TTS_VOICES_EN else "troy"

    def synthesize(
        self,
        text: str,
        *,
        reference_audio: Optional[str] = None,
        target_language: Optional[str] = None,
    ) -> bytes:
        if not text.strip():
            raise ValueError("Cannot synthesize empty text.")

        model = self._select_model(target_language)
        voice = self._select_voice(target_language)
        self.voice_id = f"{voice}@{model}"

        if target_language not in (None, "en", "ar"):
            logger.warning(
                "Groq TTS only supports English/Arabic. Target '%s' will be "
                "read using the English model and may be mispronounced.",
                target_language,
            )

        chunks = self._chunk_text(text, self.max_chars)
        logger.info(
            "Groq TTS: model=%s voice=%s chunks=%d", model, voice, len(chunks)
        )

        wavs: list[bytes] = []
        for i, chunk in enumerate(chunks, 1):
            try:
                resp = self.client.audio.speech.create(
                    model=model,
                    voice=voice,
                    input=chunk,
                    response_format="wav",
                )
                wavs.append(resp.read() if hasattr(resp, "read") else bytes(resp))
            except GroqError as exc:
                logger.error("Groq TTS chunk %d failed: %s", i, exc)
                raise RuntimeError(f"Groq TTS failed: {exc}") from exc

        return self._concat_wavs(wavs)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _chunk_text(text: str, max_chars: int) -> list[str]:
        """Split ``text`` into <=max_chars pieces at sentence/word boundaries."""
        chunks: list[str] = []
        current = ""

        def flush() -> None:
            nonlocal current
            if current.strip():
                chunks.append(current.strip())
            current = ""

        # Split into sentences, keeping the boundary whitespace.
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
            if len(sentence) <= max_chars:
                if len(current) + len(sentence) + 1 <= max_chars:
                    current = (current + " " + sentence).strip()
                else:
                    flush()
                    current = sentence
                continue

            # Sentence itself too long: break by words.
            flush()
            piece = ""
            for word in sentence.split():
                if len(piece) + len(word) + 1 <= max_chars:
                    piece = (piece + " " + word).strip()
                else:
                    if piece:
                        chunks.append(piece)
                    piece = word
            if piece:
                chunks.append(piece)

        flush()
        return [c for c in chunks if c.strip()]

    @staticmethod
    def _concat_wavs(wavs: list[bytes]) -> bytes:
        if not wavs:
            raise RuntimeError("No audio produced.")
        if len(wavs) == 1:
            return wavs[0]

        buf = io.BytesIO()
        with wave.open(buf, "wb") as out:
            first = True
            for w in wavs:
                with wave.open(io.BytesIO(w), "rb") as wf:
                    if first:
                        out.setnchannels(wf.getnchannels())
                        out.setsampwidth(wf.getsampwidth())
                        out.setframerate(wf.getframerate())
                        first = False
                    out.writeframes(wf.readframes(wf.getnframes()))
        return buf.getvalue()
