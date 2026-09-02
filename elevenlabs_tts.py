"""ElevenLabs voice-cloning synthesizer backend.

This is the backend that delivers the *original* product promise: the output
is spoken in the **user's own cloned voice**, in the target language, via
ElevenLabs ``eleven_multilingual_v2``. It requires the source audio so a
temporary voice can be cloned and then cleaned up.
"""

from __future__ import annotations

import os
from typing import Optional

from config import ELEVENLABS_MODEL
from logger import get_logger
from synthesizer import BaseSynthesizer
from voice_cloner import VoiceCloner

logger = get_logger("elevenlabs_tts")


class ElevenLabsSynthesizer(BaseSynthesizer):
    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        model: str = ELEVENLABS_MODEL,
        similarity_boost: float = 0.75,
        stability: float = 0.35,
    ):
        self.cloner = VoiceCloner(api_key, model)
        self.similarity_boost = similarity_boost
        self.stability = stability
        self.output_format = "mp3"
        self.voice_id = "elevenlabs-clone-pending"

    def synthesize(
        self,
        text: str,
        *,
        reference_audio: Optional[str] = None,
        target_language: Optional[str] = None,
    ) -> bytes:
        if not reference_audio:
            raise RuntimeError(
                "ElevenLabs voice cloning requires the source audio clip."
            )
        audio, voice_id = self.cloner.clone_and_speak(
            reference_audio,
            text,
            similarity_boost=self.similarity_boost,
            stability=self.stability,
        )
        self.voice_id = voice_id
        return audio
