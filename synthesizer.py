"""Synthesizer abstraction.

Two concrete backends implement this interface so the pipeline stays agnostic
to *how* speech is produced:

* ``GroqSynthesizer``  - Groq Orpheus TTS (preset voices, English/Arabic only).
* ``ElevenLabsSynthesizer`` - ElevenLabs ``eleven_multilingual_v2`` voice
  cloning (speaks in the *user's* voice, in many languages).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class BaseSynthesizer(ABC):
    """Minimal contract for turning text into audio bytes."""

    #: File extension the backend emits (e.g. "wav", "mp3").
    output_format: str = "wav"
    #: Human/debug identifier for the chosen voice.
    voice_id: str = "unknown"

    @abstractmethod
    def synthesize(
        self,
        text: str,
        *,
        reference_audio: Optional[str] = None,
        target_language: Optional[str] = None,
    ) -> bytes:
        """Return raw audio bytes for ``text``.

        ``reference_audio`` is the source clip (required for cloning backends).
        ``target_language`` is the ISO-639-1 code the text is written in.
        """
        raise NotImplementedError
