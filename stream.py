#!/usr/bin/env python3
"""Live streaming interpreter CLI.

Example
-------
    python stream.py --person1-lang el --direction p1     # Person 1 spoke -> English text
    python stream.py --person1-lang el --direction user   # You spoke -> Greek voice
    python stream.py --person1-lang es --direction user --voice-id <id>
"""

from __future__ import annotations

import argparse
import os
import sys

from audio_capture import list_microphones, list_audio_devices
from config import Settings
from logger import configure_root, get_logger
from stream_pipeline import StreamingInterpreter

logger = get_logger("stream_cli")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stream",
        description="Live streaming voice-to-voice interpreter "
                    "(Groq STT + NVIDIA translate + Deepgram/Aura or ElevenLabs WS TTS).",
    )
    p.add_argument("--person1-lang", default="es", help="Language Person 1 speaks.")
    p.add_argument("--direction", choices=["p1", "user"], default=None,
                   help="Legacy single-mic mode: 'p1' = Person1 at mic (English "
                        "captions) | 'user' = You at mic (speak P1's language). "
                        "Omit this and set --person1-device for two-way mode.")
    p.add_argument("--voice-id", default=None,
                   help="ElevenLabs pre-made voice id (auto-picks first if omitted).")
    p.add_argument("--realtime-backend", default=None, choices=["deepgram", "elevenlabs"],
                   help="Realtime TTS engine (default: deepgram; falls back to "
                        "ElevenLabs for languages Aura has no voice for).")
    p.add_argument("--translate-backend", default=None, choices=["groq", "nvidia"],
                   help="Translation engine (default: groq/Llama-3.3-70B; "
                        "nvidia = NVIDIA riva-translate-4b).")
    p.add_argument("--tts-mode", default="utterance", choices=["utterance", "streaming"],
                   help="TTS generation style: 'utterance' (default) generates the "
                        "whole sentence at once for natural, continuous, emotional "
                        "speech; 'streaming' is lower-latency but more segmented.")
    p.add_argument("--device", type=str, default=None,
                   help="Microphone device index or name substring (see --list-mics).")
    p.add_argument("--mic-device", type=str, default=None,
                   help="Your microphone device (You) — index or name. Defaults to --device.")
    p.add_argument("--person1-device", type=str, default=None,
                   help="Recording device (e.g. 'Stereo Mix') capturing system audio "
                        "for Person 1. Only used when --no-loopback is set.")
    p.add_argument("--loopback", dest="loopback", action="store_true", default=True,
                   help="Capture Person 1 from system audio via WASAPI loopback "
                        "(what you hear, incl. through a USB headset). Default.")
    p.add_argument("--no-loopback", dest="loopback", action="store_false",
                   help="Disable loopback; use --person1-device (Stereo Mix) instead.")
    p.add_argument("--system-device", type=str, default=None,
                   help="Output device to mirror for loopback (default: system "
                        "default output, i.e. your headset). Person 1's audio.")
    p.add_argument("--tts-output-device", type=str, default=None,
                   help="Output device for your translated TTS — set this to a "
                        "virtual microphone (e.g. 'CABLE Input') so Discord hears it.")
    p.add_argument("--noise-cancel", dest="noise_cancel", action="store_true", default=False,
                   help="EXPERIMENTAL Krisp-style noise suppression on your "
                        "microphone. Off by default — it can eat quiet/distant "
                        "speech and hurt transcription quality.")
    p.add_argument("--silence-timeout", type=float, default=0.7,
                   help="Seconds of silence that end an utterance. Lower = "
                        "snappier responses; too low splits mid-thought "
                        "pauses and fragments words. Default 0.7.")
    p.add_argument("--max-utterances", type=int, default=None,
                   help="Stop after N utterances (useful for demos).")
    p.add_argument("--elevenlabs-stability", type=float, default=0.55,
                   help="ElevenLabs stability (0-1). Higher = more consistent, calmer voice. Default 0.55.")
    p.add_argument("--elevenlabs-similarity", type=float, default=0.4,
                   help="ElevenLabs similarity boost (0-1). Lower = calmer, less expressive. Default 0.4.")
    p.add_argument("--monitor-tts", dest="monitor_tts", action="store_true", default=False,
                   help="Duplicate the translated TTS to your own speakers so you "
                        "hear what the voice says (toggleable in the GUI).")
    p.add_argument("--list-mics", action="store_true", help="List input (microphone) devices and exit.")
    p.add_argument("--list-devices", action="store_true",
                   help="List ALL audio devices with their input/output channel "
                        "counts. Use this to pick --tts-output-device (must have "
                        "output channels) and --person1-device/--mic-device.")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv=None) -> int:
    # Emoji console prints crash with UnicodeEncodeError when stdout is
    # redirected (non-console launcher); force UTF-8 so run() never dies.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    args = build_arg_parser().parse_args(argv)
    configure_root(level="DEBUG" if args.verbose else "INFO")

    if args.list_devices:
        print("All audio devices (in=input/mic, out=output/speaker-or-virtual-mic):")
        for idx, name, max_in, max_out in list_audio_devices():
            tags = []
            if max_in > 0:
                tags.append(f"in={max_in}")
            if max_out > 0:
                tags.append(f"out={max_out}")
            print(f"  [{idx}] {name}  ({', '.join(tags) or 'no channels'})")
        print("\nTip: --tts-output-device must be a device with out>0 "
              "(e.g. a virtual microphone / speakers).")
        return 0

    if args.list_mics:
        for idx, name in list_microphones():
            print(f"  [{idx}] {name}")
        return 0

    try:
        settings = Settings.from_env()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        interp = StreamingInterpreter(
            args.person1_lang,
            user_language="en",
            device=args.device,
            mic_device=args.mic_device,
            person1_device=args.person1_device,
            tts_output_device=args.tts_output_device,
            person1_loopback=args.loopback,
            system_device=args.system_device,
            voice_id=args.voice_id,
            groq_api_key=settings.groq_api_key,
            deepgram_api_key=os.environ.get("DEEPGRAM_API_KEY"),
            realtime_backend=args.realtime_backend,
            translate_backend=args.translate_backend,
            tts_mode=args.tts_mode,
            noise_cancel=args.noise_cancel,
            silence_timeout=args.silence_timeout,
            elevenlabs_stability=args.elevenlabs_stability,
            elevenlabs_similarity=args.elevenlabs_similarity,
            monitor_tts=args.monitor_tts,
        )
        interp.run(direction=args.direction, max_utterances=args.max_utterances)
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.error("Live stream failed: %s", exc)
        if args.verbose:
            import traceback

            traceback.print_exc()
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
