#!/usr/bin/env python3
"""Command-line interface for the voice-to-voice translation system.

Example
-------
    export GROQ_API_KEY=... ELEVENLABS_API_KEY=...
    python main.py input.wav --target es
    python main.py input.wav --target fr --source en --output out.mp3 --play
    python main.py input.wav --target ja --json > result.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from config import SUPPORTED_LANGUAGES, Settings
from logger import configure_root, get_logger
from pipeline import VoiceTranslationPipeline

logger = get_logger("cli")


# --------------------------------------------------------------------------- #
# Optional audio playback (best-effort, never fatal)
# --------------------------------------------------------------------------- #
def play_audio(path: str) -> None:
    """Play ``path`` using the OS default handler. Failures are ignored."""
    import subprocess

    try:
        if sys.platform == "darwin":
            subprocess.run(["afplay", path], check=False)
        elif sys.platform == "win32":
            # Uses the Windows shell to open with the default player.
            subprocess.run(["cmd", "/c", "start", "", path], check=False, shell=True)
        else:
            subprocess.run(["xdg-open", path], check=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unable to auto-play audio: %s", exc)


def build_arg_parser() -> argparse.ArgumentParser:
    langs = ", ".join(sorted(SUPPORTED_LANGUAGES))
    parser = argparse.ArgumentParser(
        prog="voice-translator",
        description="Voice-to-voice translation: transcribe -> translate -> "
                    "clone voice -> speak (Groq + ElevenLabs).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("audio", type=Path, help="Path to the source audio file.")
    parser.add_argument(
        "--target", "-t", required=True,
        help=f"Target language code. Supported: {langs}",
    )
    parser.add_argument(
        "--source", "-s", default=None,
        help="Source language code (optional; auto-detected if omitted).",
    )
    parser.add_argument(
        "--backend", "-b", default=None,
        choices=["groq", "elevenlabs"],
        help="TTS backend: 'groq' (Orpheus, preset voices, EN/AR only) or "
             "'elevenlabs' (clones your voice, multilingual). Default: $TTS_BACKEND or groq.",
    )
    parser.add_argument(
        "--voice", default="troy",
        help="Groq Orpheus voice id (troy/hannah/austin/...). Ignored by ElevenLabs.",
    )
    # --- Interpreter mode (two-party live translation) -------------------- #
    parser.add_argument(
        "--interpreter", action="store_true",
        help="Run as a 2-party interpreter (Person 1 <-> English user).",
    )
    parser.add_argument(
        "--person1-lang", default="es",
        help="Language Person 1 speaks (interpreter mode). Default: es.",
    )
    parser.add_argument(
        "--direction", choices=["p1", "user"], default="p1",
        help="Interpreter turn: 'p1' = Person1 spoke (show English) | "
             "'user' = You spoke (speak Person1's language).",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output path for the cloned-speech audio (default: <input>_<target>_cloned.mp3).",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Suppress the on-screen text summary.",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Do not write the audio file (e.g. when only --play is wanted).",
    )
    parser.add_argument(
        "--play", action="store_true",
        help="Play the resulting audio with the OS default player.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit the result as JSON (machine-readable) instead of plain text.",
    )
    parser.add_argument(
        "--similarity-boost", type=float, default=0.75,
        help="ElevenLabs similarity_boost (0-1, default 0.75).",
    )
    parser.add_argument(
        "--stability", type=float, default=0.35,
        help="ElevenLabs stability (0-1, default 0.35).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    configure_root(level=("DEBUG" if args.verbose else "INFO"))

    # --- validate inputs -------------------------------------------------- #
    if not args.audio.exists():
        parser.error(f"Audio file does not exist: {args.audio}")
    if not (0.0 <= args.similarity_boost <= 1.0):
        parser.error("--similarity-boost must be between 0 and 1.")
    if not (0.0 <= args.stability <= 1.0):
        parser.error("--stability must be between 0 and 1.")

    try:
        settings = Settings.from_env(backend=args.backend)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # --- run pipeline ----------------------------------------------------- #
    try:
        pipeline = VoiceTranslationPipeline(
            settings,
            backend=args.backend,
            voice=args.voice,
            similarity_boost=args.similarity_boost,
            stability=args.stability,
        )

        if args.interpreter:
            from interpreter import Interpreter

            interp = Interpreter(
                pipeline, args.person1_lang, user_language="en",
                backend=args.backend,
            )
            run = interp.turn(
                args.audio,
                direction=args.direction,
                output_path=args.output,
                display=(not args.no_display) and (not args.json),
                save_audio=not args.no_save,
                voice=args.voice,
            )
        else:
            if not args.target:
                parser.error("--target is required unless using --interpreter")
            run = pipeline.run(
                args.audio,
                target_language=args.target,
                source_language=args.source,
                output_path=args.output,
                display=(not args.no_display) and (not args.json),
                save_audio=not args.no_save,
                voice=args.voice,
            )
    except Exception as exc:  # noqa: BLE001 - top-level guard
        logger.error("Pipeline failed: %s", exc)
        if args.verbose:
            import traceback

            traceback.print_exc()
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    # --- output ----------------------------------------------------------- #
    if args.json:
        print(run.to_json())

    if args.play and not args.no_save:
        play_audio(run.output_audio)
    elif args.play and args.no_save:
        # When not saving, we cannot play from disk; inform the user.
        logger.warning("--play has no effect with --no-save.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
