"""Speech-to-text via Groq's Whisper-large-v3.

Whisper-large-v3 is used because it offers very low-latency, high-accuracy
transcription and can auto-detect the spoken language for us.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from logger import get_logger
from utils import KeyRotator

logger = get_logger("transcriber")

# Lazily imported so the module can be imported without the SDK installed.
try:  # pragma: no cover - import guard
    from groq import Groq, GroqError
except Exception:  # noqa: BLE001
    Groq = None  # type: ignore
    GroqError = Exception  # type: ignore


@dataclass
class TranscriptionResult:
    text: str
    language: str
    duration: Optional[float] = None


class Transcriber:
    """Wrap Groq audio transcription with round-robin key rotation."""

    def __init__(self, api_key: Optional[str] = None,
                 api_keys: Optional[list[str]] = None,
                 model: str = "whisper-large-v3"):
        if Groq is None:
            raise RuntimeError(
                "The 'groq' package is required. Install it with "
                "`pip install groq`."
            )
        keys = list(api_keys or [])
        if not keys and api_key:
            keys.append(api_key)
        if not keys:
            from config import groq_api_keys

            keys = groq_api_keys()
        if not keys:
            raise RuntimeError("GROQ_API_KEY is not set.")
        self._rotator = KeyRotator(keys)
        self.client = None  # created per request (key rotates)
        self.model = model

    def transcribe(self, audio_path: str | Path,
                   language: Optional[str] = None) -> TranscriptionResult:
        """Transcribe ``audio_path`` to text.

        Parameters
        ----------
        audio_path:
            Path to the audio file (<= 25 MB, see Groq limits).
        language:
            Optional ISO-639-1 code to force the source language. When omitted
            Whisper auto-detects it for us.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        size = audio_path.stat().st_size
        if size > 25 * 1024 * 1024:
            raise ValueError(
                f"Audio file is {size/1e6:.1f} MB; Groq limit is 25 MB. "
                "Compress/trim it first."
            )

        logger.info("Transcribing %s with %s ...", audio_path.name, self.model)
        try:
            client = Groq(api_key=self._rotator.next())
            with open(audio_path, "rb") as fh:
                response = client.audio.transcriptions.create(
                    model=self.model,
                    file=(audio_path.name, fh, "audio/wav"),
                    response_format="verbose_json",
                    language=language,
                )
        except GroqError as exc:  # surface a friendly message
            logger.error("Groq transcription failed: %s", exc)
            raise RuntimeError(f"Transcription failed: {exc}") from exc

        text = (response.text or "").strip()
        detected = getattr(response, "language", None) or language or "unknown"
        duration = getattr(response, "duration", None)

        if not text:
            raise RuntimeError(
                "Transcription returned empty text. The audio may be silent, "
                "too noisy, or in an unsupported language."
            )

        logger.info(
            "Transcribed (%s, %.1fs): %s",
            detected, duration or 0.0, text[:80] + ("..." if len(text) > 80 else ""),
        )
        return TranscriptionResult(text=text, language=detected, duration=duration)
