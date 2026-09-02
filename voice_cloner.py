"""Cross-lingual voice cloning + speech synthesis via ElevenLabs.

We use ElevenLabs' ``eleven_multilingual_v2`` model, which can both *clone* a
voice from a reference clip and *speak* in many languages while preserving the
speaker's timbre. The workflow is:

1. Upload the source audio to create a temporary cloned voice.
2. Synthesise the translated text using that voice.
3. Delete the temporary voice so we don't leave clutter in the account.

We hit the REST API directly (``requests``) to avoid tight coupling to a
specific SDK version.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import requests

from config import ELEVENLABS_MODEL
from logger import get_logger
from utils import with_retry

logger = get_logger("voice_cloner")

BASE_URL = "https://api.elevenlabs.io/v1"
DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"


class VoiceClonerError(RuntimeError):
    """Raised when ElevenLabs operations fail."""


class VoiceCloner:
    """Clone a voice from a clip and speak text with that voice."""

    def __init__(self, api_key: Optional[str] = None, model: str = ELEVENLABS_MODEL):
        key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set.")
        self.api_key = key
        self.model = model
        self._session = requests.Session()
        self._session.headers.update({"xi-api-key": self.api_key})

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def clone_and_speak(
        self,
        audio_path: str | Path,
        text: str,
        *,
        voice_name: str = "voice_translator_temp",
        stability: float = 0.35,
        similarity_boost: float = 0.75,
        output_format: str = DEFAULT_OUTPUT_FORMAT,
        cleanup: bool = True,
    ) -> tuple[bytes, str]:
        """Clone the voice in ``audio_path`` then speak ``text``.

        Returns ``(audio_bytes, voice_id)``. When ``cleanup`` is True the
        temporary voice is removed after synthesis.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Voice reference not found: {audio_path}")
        if not text.strip():
            raise ValueError("Cannot synthesise empty text.")

        voice_id = self._add_voice(audio_path, voice_name)
        logger.info("Cloned temporary voice id=%s", voice_id)

        try:
            audio = self._text_to_speech(
                voice_id, text,
                stability=stability,
                similarity_boost=similarity_boost,
                output_format=output_format,
            )
        except Exception:
            if cleanup:
                self._safe_delete_voice(voice_id)
            raise
        finally:
            if cleanup:
                self._safe_delete_voice(voice_id)

        logger.info("Synthesised %d bytes of cloned-speech audio", len(audio))
        return audio, voice_id

    def speak_with_voice(
        self,
        voice_id: str,
        text: str,
        *,
        stability: float = 0.35,
        similarity_boost: float = 0.75,
        output_format: str = DEFAULT_OUTPUT_FORMAT,
    ) -> bytes:
        """Speak ``text`` with an *existing* ElevenLabs voice id."""
        if not text.strip():
            raise ValueError("Cannot synthesise empty text.")
        return self._text_to_speech(
            voice_id, text,
            stability=stability,
            similarity_boost=similarity_boost,
            output_format=output_format,
        )

    # ------------------------------------------------------------------ #
    # REST helpers
    # ------------------------------------------------------------------ #
    def _add_voice(self, audio_path: Path, name: str) -> str:
        url = f"{BASE_URL}/voices/add"

        def _call() -> requests.Response:
            with open(audio_path, "rb") as fh:
                files = {"files": (audio_path.name, fh, "audio/wav")}
                data = {"name": name, "voice_description": "Auto-cloned for voice translation."}
                resp = self._session.post(url, data=data, files=files, timeout=60)
            return resp

        resp = with_retry(_call, label="elevenlabs.add_voice")
        if resp.status_code != 200:
            raise VoiceClonerError(
                f"add_voice failed ({resp.status_code}): {resp.text}"
            )

        voice_id = resp.json().get("voice_id")
        if not voice_id:
            raise VoiceClonerError(f"add_voice returned no voice_id: {resp.text}")
        return voice_id

    def _text_to_speech(
        self,
        voice_id: str,
        text: str,
        *,
        stability: float,
        similarity_boost: float,
        output_format: str,
    ) -> bytes:
        url = f"{BASE_URL}/text-to-speech/{voice_id}"
        payload = {
            "text": text,
            "model_id": self.model,
            "output_format": output_format,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost,
                "use_speaker_boost": True,
            },
        }
        headers = {"accept": f"audio/{output_format.split('_')[0]}"}

        def _call() -> requests.Response:
            # stream=False keeps the full payload in memory; clips are small.
            resp = self._session.post(
                url, json=payload, headers=headers, timeout=120
            )
            return resp

        resp = with_retry(_call, label="elevenlabs.text_to_speech")
        if resp.status_code != 200:
            raise VoiceClonerError(
                f"text_to_speech failed ({resp.status_code}): {resp.text}"
            )
        return resp.content

    def _safe_delete_voice(self, voice_id: str) -> None:
        try:
            resp = self._session.delete(
                f"{BASE_URL}/voices/{voice_id}", timeout=30
            )
            if resp.status_code == 200:
                logger.info("Deleted temporary voice id=%s", voice_id)
            else:
                logger.warning(
                    "Could not delete temporary voice %s (%s): %s",
                    voice_id, resp.status_code, resp.text,
                )
        except requests.exceptions.RequestException as exc:
            logger.warning("Failed to delete temporary voice %s: %s", voice_id, exc)
