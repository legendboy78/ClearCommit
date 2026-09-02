"""End-to-end orchestration: transcribe -> translate -> synthesize speech."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from config import Settings, SUPPORTED_LANGUAGES, TTS_BACKEND, TRANSLATE_BACKEND
from elevenlabs_tts import ElevenLabsSynthesizer
from groq_tts import GroqSynthesizer
from logger import get_logger
from nvidia_translator import NvidiaTranslator
from synthesizer import BaseSynthesizer
from transcriber import Transcriber, TranscriptionResult
from translator import Translator, TranslationResult
from utils import audio_duration_seconds

logger = get_logger("pipeline")


@dataclass
class TranslationRun:
    source_audio: str
    transcribed_text: str
    detected_language: str
    target_language: str
    target_language_name: str
    translated_text: str
    output_audio: str
    backend: str
    voice_id: str
    elapsed_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def build_synthesizer(
    settings: Settings,
    backend: str,
    *,
    voice: str = "troy",
    similarity_boost: float = 0.75,
    stability: float = 0.35,
) -> BaseSynthesizer:
    """Construct the TTS backend selected by ``backend``."""
    backend = backend.lower()
    if backend == "groq":
        return GroqSynthesizer(settings.groq_api_key, voice=voice)
    if backend in ("elevenlabs", "clone"):
        return ElevenLabsSynthesizer(
            settings.elevenlabs_api_key,
            similarity_boost=similarity_boost,
            stability=stability,
        )
    raise ValueError(
        f"Unknown TTS backend '{backend}'. Use 'groq' or 'elevenlabs'."
    )


def build_translator(settings: Settings, backend: str):
    """Construct the translation backend selected by ``backend``."""
    backend = backend.lower()
    if backend == "nvidia":
        return NvidiaTranslator()  # reads NVIDIA_KEY_1 from env
    if backend == "groq":
        return Translator(settings.groq_api_key)
    raise ValueError(
        f"Unknown translate backend '{backend}'. Use 'nvidia' or 'groq'."
    )


class VoiceTranslationPipeline:
    """Coordinates transcription, translation, and speech output."""

    def __init__(
        self,
        settings: Settings,
        *,
        transcriber: Optional[Transcriber] = None,
        translator: Optional[Translator] = None,
        synthesizer: Optional[BaseSynthesizer] = None,
        backend: Optional[str] = None,
        translate_backend: Optional[str] = None,
        voice: str = "troy",
        similarity_boost: float = 0.75,
        stability: float = 0.35,
    ):
        self.settings = settings
        self.transcriber = transcriber or Transcriber(settings.groq_api_key)
        self.translator = translator or build_translator(
            settings, translate_backend or TRANSLATE_BACKEND
        )
        self.backend = (backend or TTS_BACKEND).lower()
        self.synthesizer = synthesizer or build_synthesizer(
            settings,
            self.backend,
            voice=voice,
            similarity_boost=similarity_boost,
            stability=stability,
        )

    def use_backend(self, backend: str) -> None:
        """Swap the TTS backend at runtime (e.g. per target language)."""
        self.backend = (backend or TTS_BACKEND).lower()
        voice = getattr(self.synthesizer, "voice", "troy")
        self.synthesizer = build_synthesizer(
            self.settings, self.backend, voice=voice,
            similarity_boost=0.75, stability=0.35,
        )

    # ------------------------------------------------------------------ #
    def run(
        self,
        audio_path: str | Path,
        target_language: str,
        *,
        source_language: Optional[str] = None,
        output_path: Optional[str | Path] = None,
        display: bool = True,
        save_audio: bool = True,
        speak: bool = True,
        voice: Optional[str] = None,
    ) -> TranslationRun:
        target_language = Settings.validate_language(target_language)
        if source_language:
            source_language = Settings.validate_language(source_language)

        audio_path = Path(audio_path)
        target_name = SUPPORTED_LANGUAGES[target_language]

        # Allow per-call voice override for the Groq backend.
        if voice and isinstance(self.synthesizer, GroqSynthesizer):
            self.synthesizer.voice = voice

        dur = audio_duration_seconds(str(audio_path))
        if dur is not None and dur < 10:
            logger.warning(
                "Reference audio is only %.1fs; voice cloning quality may be "
                "reduced (ElevenLabs clones best with >= 10s of clean speech).",
                dur,
            )

        start = time.perf_counter()

        # 1) Speech -> text
        transcription: TranscriptionResult = self.transcriber.transcribe(
            audio_path, language=source_language
        )

        # 2) Text -> translated text
        translation: TranslationResult = self.translator.translate(
            transcription.text,
            target_language=target_language,
            source_language=(
                source_language if source_language else transcription.language
            ),
            target_language_name=target_name,
        )

        # 3) Optionally synthesize the translated text into speech
        if speak:
            audio_bytes = self.synthesizer.synthesize(
                translation.translated_text,
                reference_audio=str(audio_path),
                target_language=target_language,
            )
        else:
            audio_bytes = b""
            logger.info("Speech synthesis skipped (text-only turn).")

        ext = self.synthesizer.output_format
        out_path = (
            Path(output_path)
            if output_path
            else audio_path.with_name(
                f"{audio_path.stem}_{target_language}_spoken.{ext}"
            )
        )
        if speak and save_audio:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(audio_bytes)
            logger.info("Saved speech -> %s", out_path)

        elapsed = time.perf_counter() - start

        run = TranslationRun(
            source_audio=str(audio_path),
            transcribed_text=transcription.text,
            detected_language=transcription.language,
            target_language=target_language,
            target_language_name=target_name,
            translated_text=translation.translated_text,
            output_audio=str(out_path) if speak else "",
            backend=self.backend,
            voice_id=self.synthesizer.voice_id if speak else "n/a",
            elapsed_seconds=round(elapsed, 2),
        )

        if display:
            self._display(run)
        return run

    # ------------------------------------------------------------------ #
    @staticmethod
    def _display(run: TranslationRun) -> None:
        bar = "=" * 64
        print(bar)
        print("VOICE-TO-VOICE TRANSLATION RESULT")
        print(bar)
        print(f"Source audio      : {run.source_audio}")
        print(f"Detected language : {run.detected_language}")
        print(f"Target language   : {run.target_language_name} ({run.target_language})")
        print(f"TTS backend       : {run.backend} (voice: {run.voice_id})")
        print(f"Transcribed text  : {run.transcribed_text}")
        print(f"Translated text   : {run.translated_text}")
        print(f"Output audio      : {run.output_audio}")
        print(f"Processing time   : {run.elapsed_seconds}s")
        print(bar)
