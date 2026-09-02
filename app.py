"""Polished Gradio web UI for the voice-to-voice translation system.

Run with:
    python app.py
Then open the printed local URL. You can either export your API keys in the
shell / .env, or paste them into the (password-masked) fields in the UI.

The UI reuses the production pipeline (config / pipeline / services) so the
web and CLI paths share identical behaviour and error handling.
"""

from __future__ import annotations

import tempfile
import traceback
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr

from config import SUPPORTED_LANGUAGES, Settings, GROQ_TTS_VOICES_EN
from logger import configure_root, get_logger
from pipeline import VoiceTranslationPipeline

logger = get_logger("ui")
configure_root(level="INFO")

# Friendly dropdown choices: "Spanish (es)" -> "es"
LANG_CHOICES = [
    (f"{name} ({code})", code) for code, name in sorted(SUPPORTED_LANGUAGES.items())
]

# Groq Orpheus voice personas (English set; allow custom for Arabic etc.).
VOICE_CHOICES = [(v, v) for v in GROQ_TTS_VOICES_EN]


# --------------------------------------------------------------------------- #
def run_translation(
    audio: Optional[str],
    target_lang: str,
    source_lang: str,
    backend: str,
    voice: str,
    groq_key: str,
    eleven_key: str,
    similarity: float,
    stability: float,
    progress: gr.Progress = None,
) -> Tuple[str, str, str, str, Optional[str]]:
    """Gradio handler: returns (status, transcribed, detected, translated, audio)."""
    empty: Tuple[str, str, str, str, Optional[str]] = ("", "", "", "", None)

    if not audio:
        return ("❌ Please record or upload an audio clip first.", *empty[1:])

    groq_key = (groq_key or "").strip()
    eleven_key = (eleven_key or "").strip()

    try:
        if progress:
            progress(0.1, "Resolving credentials ...")
        # Prefer UI-supplied keys; fall back to environment / .env.
        settings = Settings.from_env(
            groq_api_key=groq_key or None,
            elevenlabs_api_key=eleven_key or None,
            backend=backend,
        )

        if progress:
            progress(0.2, "Loading services ...")
        pipeline = VoiceTranslationPipeline(
            settings,
            backend=backend,
            voice=voice or "troy",
            similarity_boost=similarity,
            stability=stability,
        )

        if progress:
            progress(0.3, "Transcribing speech (Groq Whisper) ...")
        src = source_lang if source_lang and source_lang != "auto" else None

        run = pipeline.run(
            audio,
            target_language=target_lang,
            source_language=src,
            output_path=None,          # auto-named next to input
            display=False,             # UI handles display
            save_audio=True,
            voice=voice or "troy",
        )

        if progress:
            progress(1.0, "Done ✅")
        status = (
            f"✅ Done in {run.elapsed_seconds}s · backend={run.backend} · "
            f"voice={run.voice_id}"
        )
        return (
            status,
            run.transcribed_text,
            f"{run.detected_language}",
            run.translated_text,
            run.output_audio,
        )

    except RuntimeError as exc:  # missing keys / dependency issues
        logger.error("Config error: %s", exc)
        return (f"❌ Configuration error: {exc}", *empty[1:])
    except Exception as exc:  # noqa: BLE001 - top-level guard
        logger.error("Pipeline failed: %s\n%s", exc, traceback.format_exc())
        return (f"❌ {type(exc).__name__}: {exc}", *empty[1:])


# --------------------------------------------------------------------------- #
def run_interpreter(
    audio: Optional[str],
    person1_lang: str,
    direction: str,
    backend: str,
    voice: str,
    groq_key: str,
    eleven_key: str,
    similarity: float,
    stability: float,
    progress: gr.Progress = None,
) -> Tuple[str, str, str, str, Optional[str]]:
    """Interpreter-mode handler: Person 1 <-> English user."""
    empty: Tuple[str, str, str, str, Optional[str]] = ("", "", "", "", None)
    if not audio:
        return ("❌ Please record or upload an audio clip first.", *empty[1:])

    groq_key = (groq_key or "").strip()
    eleven_key = (eleven_key or "").strip()

    try:
        if progress:
            progress(0.1, "Resolving credentials ...")
        settings = Settings.from_env(
            groq_api_key=groq_key or None,
            elevenlabs_api_key=eleven_key or None,
            backend=backend,
        )
        if progress:
            progress(0.2, "Loading services ...")
        pipeline = VoiceTranslationPipeline(
            settings, backend=backend, voice=voice or "troy",
            similarity_boost=similarity, stability=stability,
        )
        from interpreter import Interpreter

        interp = Interpreter(pipeline, person1_lang, user_language="en", backend=backend)

        if progress:
            progress(0.3, "Transcribing + translating ...")
        run = interp.turn(
            audio, direction, display=False, save_audio=True, voice=voice or "troy",
        )

        if progress:
            progress(1.0, "Done ✅")
        label = "Person 1 → You (English)" if direction == "p1" else "You → Person 1 (their language)"
        status = f"✅ {label} · {run.elapsed_seconds}s · {run.translated_text[:60]}..."
        return (
            status,
            run.transcribed_text,
            f"{run.detected_language}",
            run.translated_text,
            run.output_audio or None,
        )
    except RuntimeError as exc:
        logger.error("Config error: %s", exc)
        return (f"❌ Configuration error: {exc}", *empty[1:])
    except Exception as exc:  # noqa: BLE001
        logger.error("Interpreter failed: %s\n%s", exc, traceback.format_exc())
        return (f"❌ {type(exc).__name__}: {exc}", *empty[1:])


# --------------------------------------------------------------------------- #
def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Voice-to-Voice Translator",
    ) as demo:
        gr.Markdown(
            "# 🎙️ Voice-to-Voice Translator\n"
            "<div class='subtitle'>Groq STT + NVIDIA translation + Groq Orpheus TTS. "
            "Use the <b>Interpreter</b> tab for a live two-party conversation.</div>"
        )

        with gr.Tabs():
            # ---------------- Single clip ---------------- #
            with gr.Tab("🎧 Single clip"):
                with gr.Row():
                    with gr.Column(scale=2):
                        audio_in = gr.Audio(
                            label="🎤 Source speech (record or upload)",
                            sources=["microphone", "upload"],
                            type="filepath",
                        )
                        with gr.Row():
                            target_lang = gr.Dropdown(
                                label="🌐 Target language",
                                choices=LANG_CHOICES, value="es", interactive=True,
                            )
                            source_lang = gr.Dropdown(
                                label="🗣️ Source language (auto-detect)",
                                choices=[("Auto-detect", "auto")] + LANG_CHOICES,
                                value="auto", interactive=True,
                            )
                        with gr.Row():
                            backend = gr.Radio(
                                label="🔊 TTS engine",
                                choices=[
                                    ("Groq Orpheus (preset voice, EN/AR)", "groq"),
                                    ("ElevenLabs (clone YOUR voice, multilingual)", "elevenlabs"),
                                ],
                                value="groq", interactive=True,
                            )
                            voice = gr.Dropdown(
                                label="🎭 Groq voice persona",
                                choices=VOICE_CHOICES, value="troy",
                                interactive=True, allow_custom_value=True,
                            )
                        with gr.Accordion("⚙️ Advanced & API keys", open=False):
                            with gr.Row():
                                groq_key = gr.Textbox(
                                    label="Groq API key (optional)",
                                    placeholder="gsk_...  (uses env if blank)",
                                    type="password",
                                )
                                eleven_key = gr.Textbox(
                                    label="ElevenLabs API key (optional)",
                                    placeholder="...  (uses env if blank)",
                                    type="password",
                                )
                            with gr.Row():
                                similarity = gr.Slider(0.0, 1.0, value=0.75, step=0.05, label="Voice similarity boost")
                                stability = gr.Slider(0.0, 1.0, value=0.35, step=0.05, label="Voice stability")
                        run_btn = gr.Button("🚀 Translate & Speak", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        status = gr.Textbox(label="Status", value="Idle.", interactive=False, elem_classes=["status-box"])
                        transcribed = gr.Textbox(label="📝 Transcribed (original language)", lines=3, interactive=False)
                        detected = gr.Textbox(label="🔎 Detected language", lines=1, interactive=False)
                        translated = gr.Textbox(label="🌍 Translated text", lines=3, interactive=False)
                        audio_out = gr.Audio(label="🔊 Output speech", type="filepath", interactive=False)

                run_btn.click(
                    fn=run_translation,
                    inputs=[audio_in, target_lang, source_lang, backend, voice,
                            groq_key, eleven_key, similarity, stability],
                    outputs=[status, transcribed, detected, translated, audio_out],
                )

            # ---------------- Interpreter ---------------- #
            with gr.Tab("🗣️ Interpreter (Person 1 ↔ You)"):
                gr.Markdown(
                    "Live two-party loop. **Person 1** speaks a foreign language; "
                    "**You** are the English operator.\n\n"
                    "- **Person 1 spoke** → you see the English translation (no audio).\n"
                    "- **You spoke** → Person 1 hears it in *their* language via TTS."
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        p1_lang = gr.Dropdown(
                            label="🌐 Person 1's language",
                            choices=LANG_CHOICES, value="es", interactive=True,
                        )
                        direction = gr.Radio(
                            label="Whose turn is this?",
                            choices=[
                                ("🧑 Person 1 spoke (show me English)", "p1"),
                                ("🎤 I (You) spoke (speak to Person 1)", "user"),
                            ],
                            value="p1", interactive=True,
                        )
                        audio_int = gr.Audio(
                            label="🎤 Record / upload this turn's speech",
                            sources=["microphone", "upload"], type="filepath",
                        )
                        with gr.Row():
                            backend2 = gr.Radio(
                                label="🔊 TTS engine (for 'You spoke')",
                                choices=[
                                    ("Auto (ElevenLabs for non-EN/AR)", "auto"),
                                    ("Groq Orpheus (preset voice, EN/AR)", "groq"),
                                    ("ElevenLabs (clone YOUR voice, multilingual)", "elevenlabs"),
                                ],
                                value="auto", interactive=True,
                            )
                            voice2 = gr.Dropdown(
                                label="🎭 Groq voice persona",
                                choices=VOICE_CHOICES, value="troy",
                                interactive=True, allow_custom_value=True,
                            )
                        with gr.Accordion("⚙️ API keys", open=False):
                            groq_key2 = gr.Textbox(label="Groq API key (optional)", placeholder="gsk_...", type="password")
                            eleven_key2 = gr.Textbox(label="ElevenLabs API key (optional)", placeholder="...", type="password")
                        similarity2 = gr.Slider(0.0, 1.0, value=0.75, step=0.05, visible=False)
                        stability2 = gr.Slider(0.0, 1.0, value=0.35, step=0.05, visible=False)
                        int_btn = gr.Button("🚀 Run turn", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        i_status = gr.Textbox(label="Status", value="Idle.", interactive=False, elem_classes=["status-box"])
                        i_transcribed = gr.Textbox(label="📝 What was said (original)", lines=3, interactive=False)
                        i_detected = gr.Textbox(label="🔎 Detected language", lines=1, interactive=False)
                        i_translated = gr.Textbox(label="🌍 Translation", lines=3, interactive=False)
                        i_audio_out = gr.Audio(label="🔊 Speech for Person 1", type="filepath", interactive=False)

                int_btn.click(
                    fn=run_interpreter,
                    inputs=[audio_int, p1_lang, direction, backend2, voice2,
                            groq_key2, eleven_key2, similarity2, stability2],
                    outputs=[i_status, i_transcribed, i_detected, i_translated, i_audio_out],
                )

        gr.Markdown(
            "### How it works\n"
            "1. **Transcribe** — Groq `whisper-large-v3` turns speech into text "
            "(auto-detecting the language).\n"
            "2. **Translate** — NVIDIA `riva-translate-4b-instruct-v2` translates the text.\n"
            "3. **Speak** (optional) — Groq Orpheus TTS (preset voices, English/Arabic) "
            "or ElevenLabs voice cloning (multilingual)."
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="violet"),
        css="""
        .title { text-align: center; margin-bottom: 0.2rem; }
        .subtitle { text-align: center; color: #6b7280; margin-bottom: 1rem; }
        .status-box { font-weight: 600; }
        footer { visibility: hidden; }
        """,
    )
