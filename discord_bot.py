"""Discord bot voice interpreter — the ideal, routing-free design.

Person 1 = the other participant in the Discord voice channel. Their audio is
received *directly* from Discord's voice gateway, so there is no Stereo Mix and
no default-playback routing to misconfigure.

You = your local microphone (captured on this machine).

Your translated speech is played into the channel by the bot itself, so Person 1
hears it with no virtual microphone and no acoustic feedback loop.

Requires: `pip install discord.py`  (the voice extra is bundled in the base pkg)
Bot token: set DISCORD_BOT_TOKEN (or pass --token). Invite the bot with the
"Connect", "Speak", "Use Voice Activity", and "Send Messages" permissions.

Commands (in any text channel):
    !join     connect to your voice channel and start interpreting
    !leave    disconnect and stop
    !lang el  change Person 1's language at runtime
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import threading
import time
import wave

import numpy as np

from audio_capture import MicCapture
from config import (
    DEEPGRAM_API_KEY,
    REALTIME_TTS_BACKEND,
    Settings,
    select_realtime_tts_backend,
    select_translate_backend,
)
from logger import get_logger
from nvidia_stream import NvidiaStreamTranslator
from stream_pipeline import (
    HALLUCINATION_PHRASES,
    MIN_SPEECH_RMS,
    POST_TTS_COOLDOWN,
    VAD_THRESHOLD,
)
from translator import Translator
from transcriber import Transcriber

logger = get_logger("discord_bot")

# Discord's voice receive/playback format.
DISCORD_RATE = 48000
TTS_SILENCE_TIMEOUT = 0.7  # seconds of silence that ends a Person 1 utterance


# --------------------------------------------------------------------------- #
# Small PCM helpers (kept local so this module has no sounddevice dependency
# at import time).
def _resample_int16(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or samples.size < 2:
        return samples
    n = max(1, int(round(samples.size * dst_rate / src_rate)))
    x_old = np.linspace(0.0, samples.size - 1, samples.size)
    x_new = np.linspace(0.0, samples.size - 1, n)
    return np.interp(x_new, x_old, samples.astype(np.float64)).astype(np.int16)


def _mono_to_stereo(mono_int16: np.ndarray) -> np.ndarray:
    stereo = np.empty(mono_int16.size * 2, dtype=np.int16)
    stereo[0::2] = mono_int16
    stereo[1::2] = mono_int16
    return stereo


def _wav_bytes(pcm_int16: np.ndarray, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm_int16.tobytes())
    return buf.getvalue()


def _avg_rms(wav_bytes: bytes) -> float:
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            data = w.readframes(w.getnframes())
    except Exception:  # noqa: BLE001
        return 0.0
    arr = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr ** 2)) / 32768.0)


# --------------------------------------------------------------------------- #
# Discord receive sink: buffers Person 1's voice frames, runs a tiny VAD, and
# emits a 16 kHz mono WAV per utterance.
try:  # pragma: no cover - import guard (discord installed at runtime)
    from discord.sinks import Sink as _DiscordSink
except Exception:  # noqa: BLE001
    _DiscordSink = object  # type: ignore


class Person1Sink(_DiscordSink):
    def __init__(self, on_utterance, vad_threshold: float = VAD_THRESHOLD,
                 silence_timeout: float = TTS_SILENCE_TIMEOUT):
        self.on_utterance = on_utterance
        self.vad_threshold = vad_threshold
        self.silence_timeout = silence_timeout
        self._buf = bytearray()
        self._speaking = False
        self._silence_samples = 0
        self._rate = DISCORD_RATE
        self.bot_user = None

    def wants_user(self, user) -> bool:
        if self.bot_user is not None and getattr(user, "id", None) == self.bot_user.id:
            return False
        return True

    def write(self, data, user) -> None:
        pcm = getattr(data, "data", data)
        if not isinstance(pcm, (bytes, bytearray)):
            return
        stereo = np.frombuffer(bytes(pcm), dtype=np.int16)
        if stereo.size % 2 == 0:
            mono = stereo.reshape(-1, 2).mean(axis=1).astype(np.int16)
        else:
            mono = stereo
        if mono.size == 0:
            return
        rms = float(np.sqrt(np.mean(mono.astype(np.float32) ** 2)) / 32768.0)
        if rms > self.vad_threshold:
            if not self._speaking:
                self._speaking = True
                self._silence_samples = 0
            self._buf.extend(mono.tobytes())
        elif self._speaking:
            self._silence_samples += mono.size
            self._buf.extend(mono.tobytes())
            if self._silence_samples >= self.silence_timeout * self._rate:
                self._flush()
        if len(self._buf) > self._rate * 2 * 15:  # hard cap ~15s
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            self._speaking = False
            return
        mono = np.frombuffer(bytes(self._buf), dtype=np.int16)
        self._buf = bytearray()
        self._speaking = False
        self._silence_samples = 0
        if mono.size == 0:
            return
        mono16 = _resample_int16(mono, DISCORD_RATE, 16000)
        self.on_utterance(_wav_bytes(mono16, 16000))


# --------------------------------------------------------------------------- #
class DiscordInterpreter:
    """Owns STT + translate + TTS and processes utterances from both sources."""

    def __init__(self, person1_language: str, *, groq_api_key,
                 deepgram_api_key=None, eleven_api_key=None,
                 realtime_backend=None, mic_device=None, translate_backend=None):
        self.person1_language = Settings.validate_language(person1_language)
        self.user_language = "en"
        self.mic_device = mic_device

        self.transcriber = Transcriber(groq_api_key)
        backend = select_translate_backend(translate_backend)
        self.translator = (
            NvidiaStreamTranslator(os.environ.get("NVIDIA_KEY_1"))
            if backend == "nvidia"
            else Translator(groq_api_key)
        )

        self._deepgram_key = deepgram_api_key or DEEPGRAM_API_KEY
        self._eleven_key = eleven_api_key
        self._realtime_backend = realtime_backend
        self._tts = None
        self._tts_lang = None

        # Echo guard (same idea as the local pipeline).
        self._tts_active = False
        self._last_tts_end = 0.0

        self._stop = threading.Event()
        self._mic = None
        self.voice_client = None
        self._caption_channel = None
        self._caption_cb = None  # callable(str)

    # -- TTS ---------------------------------------------------------------- #
    def _ensure_tts(self, lang: str):
        if self._tts is not None and self._tts_lang == lang:
            return
        rt = select_realtime_tts_backend(lang, preferred=self._realtime_backend)
        if rt == "deepgram":
            from deepgram_tts import DeepgramStreamTTS

            self._tts = DeepgramStreamTTS(api_key=self._deepgram_key, target_language=lang)
            logger.info("Realtime TTS -> Deepgram Aura model=%s", self._tts.model)
        else:
            from elevenlabs_ws import ElevenLabsStreamTTS

            if (self._realtime_backend or REALTIME_TTS_BACKEND).lower() == "deepgram":
                logger.warning("Deepgram Aura has no voice for '%s'; using ElevenLabs.", lang)
            self._tts = ElevenLabsStreamTTS(
                api_key=self._eleven_key, target_language=lang,
                voice_id=os.environ.get("ELEVENLABS_VOICE_ID"),
            )
            logger.info("Realtime TTS -> ElevenLabs")
        self._tts_lang = lang

    def _play_tts(self, pcm_mono_bytes: bytes, src_rate: int):
        vc = self.voice_client
        if vc is None or not vc.is_connected():
            logger.warning("No voice client to play TTS into.")
            return
        mono = np.frombuffer(pcm_mono_bytes, dtype=np.int16)
        mono48 = _resample_int16(mono, src_rate, DISCORD_RATE)
        stereo = _mono_to_stereo(mono48)
        try:
            if vc.is_playing():
                vc.stop()
            import discord

            vc.play(discord.PCMAudio(io.BytesIO(stereo.tobytes())))
        except Exception as exc:  # noqa: BLE001
            logger.error("TTS playback failed: %s", exc)

    # -- STT ---------------------------------------------------------------- #
    def _transcribe(self, wav_bytes: bytes, language=None) -> str:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
            fh.write(wav_bytes)
            path = fh.name
        try:
            return self.transcriber.transcribe(path, language=language).text
        finally:
            try:
                os.unlink(path)
            except Exception:  # noqa: BLE001
                pass

    def _caption(self, line: str):
        print(line)
        if self._caption_cb is not None:
            try:
                self._caption_cb(line)
            except Exception:  # noqa: BLE001
                pass

    # -- Per-utterance processing ------------------------------------------ #
    def process_person1(self, wav_bytes: bytes):
        try:
            text = self._transcribe(wav_bytes, language=self.person1_language)
        except Exception as exc:  # noqa: BLE001
            logger.error("Person 1 transcribe failed: %s", exc)
            return
        if not text.strip():
            return
        if text.strip().lower() in HALLUCINATION_PHRASES:
            logger.info("Dropped P1 hallucination: %r", text)
            return
        self._caption(f"[p1] {text}")
        try:
            translated = "".join(
                self.translator.translate_stream(
                    text,
                    target_language=self.user_language,
                    source_language=self.person1_language,
                )
            )
            self._caption(f"[-> {self.user_language}] {translated}")
        except Exception as exc:  # noqa: BLE001
            logger.error("Person 1 translate failed: %s", exc)

    def process_user(self, wav_bytes: bytes):
        try:
            text = self._transcribe(wav_bytes, language=self.user_language)
        except Exception as exc:  # noqa: BLE001
            logger.error("You transcribe failed: %s", exc)
            return
        if not text.strip():
            return
        if text.strip().lower() in HALLUCINATION_PHRASES:
            logger.info("Dropped user hallucination: %r", text)
            return
        self._caption(f"[you] {text}")
        target = self.person1_language
        self._ensure_tts(target)
        try:
            translated = "".join(
                self.translator.translate_stream(
                    text,
                    target_language=target,
                    source_language=self.user_language,
                )
            )
            self._caption(f"[-> {target}] {translated}")
            self._tts_active = True
            pcm = self._tts.synthesize(translated)
            self._play_tts(pcm, self._tts.sample_rate)
        except Exception as exc:  # noqa: BLE001
            logger.error("You TTS failed: %s", exc)
            self._tts = None  # force rebuild next time
        finally:
            self._tts_active = False
            self._last_tts_end = time.time()

    # -- Local mic (You) --------------------------------------------------- #
    def start_mic(self):
        cap = MicCapture(device=self.mic_device, vad_threshold=VAD_THRESHOLD)
        self._mic = cap
        t = threading.Thread(target=self._mic_loop, args=(cap,), daemon=True)
        t.start()

    def _mic_loop(self, cap: MicCapture):
        cap.start()
        try:
            for wav in cap.utterances():
                if self._stop.is_set():
                    break
                if self._tts_active or (time.time() - self._last_tts_end) < POST_TTS_COOLDOWN:
                    continue
                if _avg_rms(wav) < MIN_SPEECH_RMS:
                    continue
                try:
                    self.process_user(wav)
                except Exception as exc:  # noqa: BLE001
                    logger.error("User utterance failed: %s", exc)
        finally:
            cap.stop()

    def stop(self):
        self._stop.set()
        if self._mic is not None:
            try:
                self._mic.stop()
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
# Bot wrapper.
def build_bot(interpreter: DiscordInterpreter):
    import asyncio

    import discord
    from discord.ext import commands

    intents = discord.Intents.default()
    intents.message_content = True
    intents.voice_states = True

    bot = commands.Bot(command_prefix="!", intents=intents)

    @bot.event
    async def on_ready():
        logger.info("Bot logged in as %s", bot.user)
        interpreter._bot_user = bot.user

    @bot.command()
    async def join(ctx):
        if not ctx.author.voice:
            await ctx.send("You must be in a voice channel first.")
            return
        channel = ctx.author.voice.channel
        vc = await channel.connect()
        interpreter.voice_client = vc
        interpreter._caption_channel = ctx.channel
        interpreter._caption_cb = lambda line: asyncio.run_coroutine_threadsafe(
            ctx.channel.send(line), bot.loop
        )
        sink = Person1Sink(interpreter.process_person1)
        sink.bot_user = bot.user
        vc.listen(sink)
        interpreter.start_mic()
        await ctx.send(
            f"Interpreting. Person 1 = Discord audio (->{interpreter.user_language}), "
            f"You = local mic (->{interpreter.person1_language}). Use !leave to stop."
        )

    @bot.command()
    async def leave(ctx):
        vc = ctx.voice_client
        if vc is not None:
            vc.stop_listening()
            await vc.disconnect()
        interpreter.stop()
        await ctx.send("Stopped.")

    @bot.command()
    async def lang(ctx, code: str = "es"):
        try:
            interpreter.person1_language = Settings.validate_language(code)
            await ctx.send(f"Person 1 language set to {interpreter.person1_language}.")
        except Exception as exc:  # noqa: BLE001
            await ctx.send(f"Invalid language: {exc}")

    return bot


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Discord voice interpreter bot.")
    p.add_argument("--person1-lang", default="es", help="Language Person 1 speaks.")
    p.add_argument("--mic-device", type=int, default=None,
                   help="Your microphone device index (see stream.py --list-mics).")
    p.add_argument("--realtime-backend", default=None, choices=["deepgram", "elevenlabs"])
    p.add_argument("--translate-backend", default=None, choices=["groq", "nvidia"])
    p.add_argument("--token", default=os.environ.get("DISCORD_BOT_TOKEN"),
                   help="Discord bot token (or set DISCORD_BOT_TOKEN).")
    args = p.parse_args(argv)

    try:
        settings = Settings.from_env()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=__import__("sys").stderr)
        return 2

    if not args.token:
        print("ERROR: set DISCORD_BOT_TOKEN or pass --token", file=__import__("sys").stderr)
        return 2

    interpreter = DiscordInterpreter(
        args.person1_lang,
        groq_api_key=settings.groq_api_key,
        deepgram_api_key=os.environ.get("DEEPGRAM_API_KEY"),
        eleven_api_key=os.environ.get("ELEVENLABS_API_KEY"),
        realtime_backend=args.realtime_backend,
        mic_device=args.mic_device,
        translate_backend=args.translate_backend,
    )
    bot = build_bot(interpreter)
    try:
        bot.run(args.token)
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
