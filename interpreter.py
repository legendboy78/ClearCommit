"""Two-party live interpreter built on the translation pipeline.

Workflow (as specified):
  Person 1 speaks (in their language, e.g. Spanish)
      -> STT  -> translate to English -> show the USER the English text.
  The USER (English) speaks
      -> STT  -> translate to Person 1's language -> TTS -> Person 1 hears it.

The English-speaking operator always reads English; Person 1 always hears their
own language. This is just two configured runs of ``VoiceTranslationPipeline``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from config import Settings, select_tts_backend
from pipeline import VoiceTranslationPipeline


class Interpreter:
    def __init__(
        self,
        pipeline: VoiceTranslationPipeline,
        person1_language: str,
        user_language: str = "en",
        backend: Optional[str] = None,
    ):
        self.pipeline = pipeline
        self.person1_language = Settings.validate_language(person1_language)
        self.user_language = Settings.validate_language(user_language)
        # TTS only happens in the "You spoke" turn, whose target is Person 1's
        # language, so choose the backend for that language (auto = ElevenLabs
        # for anything other than English/Arabic).
        self.backend = select_tts_backend(self.person1_language, preferred=backend)
        # Rebuild the pipeline's TTS backend to match.
        self.pipeline.use_backend(self.backend)

    # Person 1 -> User: speech in P1's language, output shown in English.
    def person1_to_user(self, audio_path: str | Path, **kwargs):
        return self.pipeline.run(
            audio_path,
            target_language=self.user_language,
            source_language=self.person1_language,
            speak=False,  # user reads English; no TTS needed
            **kwargs,
        )

    # User -> Person 1: English speech, output spoken in P1's language.
    def user_to_person1(self, audio_path: str | Path, **kwargs):
        return self.pipeline.run(
            audio_path,
            target_language=self.person1_language,
            source_language=self.user_language,
            speak=True,  # TTS speaks to Person 1 in their language
            **kwargs,
        )

    # Convenience for a single CLI invocation.
    def turn(self, audio_path: str | Path, direction: str, **kwargs):
        if direction == "p1":
            return self.person1_to_user(audio_path, **kwargs)
        if direction == "user":
            return self.user_to_person1(audio_path, **kwargs)
        raise ValueError("direction must be 'p1' or 'user'.")
