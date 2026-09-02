"""First-run setup wizard for packaged builds.

Runs when %APPDATA%/LiveInterpreter/settings.json is missing (true first
launch): welcome + consent, API keys (validated live), device auto-detect,
and a guided Discord click (the one step Discord's API cannot automate).
Developers running from source with a repo .env skip straight past it.
"""

from __future__ import annotations

import os
from pathlib import Path

import customtkinter as ctk

ACCENT = "#4c8bf5"
GREEN = "#34d399"
RED = "#ef4444"
BG = "#14161a"
CARD = "#191c22"


def app_env_path() -> Path:
    from config import app_data_dir

    return app_data_dir() / ".env"


def keys_present() -> bool:
    """True when at least Groq + ElevenLabs/Deepgram credentials are available."""
    import config

    if not config.groq_api_keys():
        # Cover odd import orders: explicitly load the app-data .env if present.
        p = app_env_path()
        if p.exists():
            try:
                from dotenv import load_dotenv

                load_dotenv(p, override=True)
            except Exception:  # noqa: BLE001
                pass
    ok_groq = bool(config.groq_api_keys())
    ok_voice = bool(os.environ.get("ELEVENLABS_API_KEY") or os.environ.get("DEEPGRAM_API_KEY"))
    return ok_groq and ok_voice


def run_wizard() -> None:
    """Block until the wizard finishes. Writes .env into app-data on success."""
    wizard = Wizard()
    wizard.root.mainloop()


class Wizard:
    STEPS = ("Welcome", "API keys", "Devices", "Discord", "Done")

    def __init__(self):
        ctk.set_appearance_mode("dark")
        self.root = ctk.CTk()
        self.root.title("Live Interpreter — Setup")
        self.root.geometry("640x480")
        self.root.resizable(False, False)
        self.root.configure(fg_color=BG)
        self.root.protocol("WM_DELETE_WINDOW", self._cancel)

        self.step = 0
        self.header = ctk.CTkLabel(
            self.root, text="", font=ctk.CTkFont("Segoe UI Semibold", 20, "bold"),
        )
        self.header.pack(pady=(28, 4))
        self.sub = ctk.CTkLabel(
            self.root, text="", justify="center", text_color="#9ca3af",
            font=ctk.CTkFont("Segoe UI", 12), wraplength=520,
        )
        self.sub.pack(pady=(0, 18))

        self.body = ctk.CTkFrame(self.root, fg_color="transparent")
        self.body.pack(fill="both", expand=True, padx=48)

        self.nav = ctk.CTkFrame(self.root, fg_color="transparent")
        self.nav.pack(fill="x", padx=48, pady=18)
        self.back_btn = ctk.CTkButton(self.nav, text="Back", width=90,
                                      fg_color="#232830", hover_color="#2d333d",
                                      command=self._back)
        self.back_btn.pack(side="left")
        self.next_btn = ctk.CTkButton(self.nav, text="Next →", width=120,
                                      fg_color=ACCENT, hover_color="#3a76d8",
                                      font=ctk.CTkFont("Segoe UI Semibold", 13, "bold"),
                                      command=self._next)
        self.next_btn.pack(side="right")

        self._show(0)

    # -- step rendering ------------------------------------------------- #
    def _clear_body(self) -> None:
        for w in self.body.winfo_children():
            w.destroy()

    def _show(self, step: int) -> None:
        self.step = max(0, min(step, len(self.STEPS) - 1))
        self._clear_body()
        getattr(self, f"_step{self.step}")()
        self.back_btn.configure(state="normal" if self.step > 0 else "disabled")

    def _step0(self) -> None:  # Welcome + consent
        self.header.configure(text="Welcome to Live Interpreter")
        self.sub.configure(text="Talk to anyone in any call — each side hears "
                                "their own language.\nSetup takes about a minute.")
        ctk.CTkLabel(
            self.body, justify="left", anchor="w", text_color="#d1d5db",
            font=ctk.CTkFont("Segoe UI", 12),
            text=("How it works\n"
                  "  • Your microphone → translated into their language (spoken)\n"
                  "  • Their voice → shown as captions in your language\n\n"
                  "Privacy, plainly:\n"
                  "  • Speech is sent to AI services (Groq, ElevenLabs/Deepgram) "
                  "for transcription, translation and voice.\n"
                  "  • Nothing is stored on any server by this app; transcripts "
                  "stay on your PC.\n"
                  "  • Please let people on the call know it's translated — the "
                  "app can play a short notice chime."),
        ).pack(anchor="w", pady=10)

    def _step1(self) -> None:  # API keys
        self.header.configure(text="Connect your AI keys")
        self.sub.configure(text="Free keys take ~2 minutes. They're stored only "
                                "on this PC (%APPDATA%\\LiveInterpreter).")
        self.status = ctk.CTkLabel(self.body, text="", text_color="#9ca3af")
        rows = []
        for label, url, env in (
            ("Groq (speech + translation)", "https://console.groq.com/keys", "GROQ_API_KEY"),
            ("ElevenLabs (voice)", "https://elevenlabs.io/app/settings/api-keys", "ELEVENLABS_API_KEY"),
        ):
            row = ctk.CTkFrame(self.body, fg_color="transparent")
            row.pack(fill="x", pady=6)
            ctk.CTkLabel(row, text=label, width=220, anchor="w").pack(side="left")
            entry = ctk.CTkEntry(row, width=250, show="•",
                                 placeholder_text="paste key…")
            entry.pack(side="left", padx=6)
            link = ctk.CTkButton(row, text="get key", width=70,
                                 fg_color="#232830", hover_color="#2d333d",
                                 command=lambda u=url: os.startfile(u))  # noqa: S605
            link.pack(side="left")
            rows.append((env, entry))
        self.key_rows = rows
        self.status.pack(pady=10)

    def _step2(self) -> None:  # Devices (auto)
        self.header.configure(text="Audio setup")
        self.sub.configure(text="We auto-pick your microphone and the call audio. "
                                "The virtual cable carries your translated voice "
                                "into Discord.")
        self.dev_status = ctk.CTkLabel(self.body, text="Detecting devices…",
                                       text_color="#9ca3af")
        self.dev_status.pack(pady=12)
        self.root.after(150, self._detect_devices)

    def _detect_devices(self) -> None:
        lines = []
        try:
            from audio_capture import list_audio_devices, resolve_device

            mic = None
            for idx, name, max_in, max_out in list_audio_devices():
                if max_in > 0 and "CABLE" not in name.upper():
                    mic = (idx, name)
                    break
            if mic:
                lines.append(f"✓ Microphone: {mic[1]}")
            else:
                lines.append("✗ No microphone found — plug one in and retry.")

            cable = any("cable input" in n.lower() for _i, n, _mi, _mo in list_audio_devices())
            if cable:
                lines.append("✓ VB-Cable installed (translated voice → Discord).")
            else:
                lines.append("✗ VB-Cable missing — install it, then re-run setup:")
                lines.append("    https://vb-audio.com/Cable/")
            lines.append("✓ Call audio: captured from your default output (loopback).")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"Device scan failed: {exc}")
        self.dev_status.configure(text="\n".join(lines), justify="left")

    def _step3(self) -> None:  # Discord guided click
        self.header.configure(text="Point Discord at the translator")
        self.sub.configure(text="One manual step (Discord doesn't allow apps to "
                                "change it). Takes 20 seconds:")
        ctk.CTkLabel(
            self.body, justify="left", anchor="w", text_color="#d1d5db",
            font=ctk.CTkFont("Segoe UI", 13),
            text=("1.  Open Discord → Settings (gear) → Voice & Video\n"
                  "2.  Input Device → choose  \"CABLE Output (VB-Audio Virtual Cable)\"\n"
                  "3.  Output Device → leave on your headphones\n"
                  "4.  Set  Input Volume  to ~75%\n\n"
                  "That's the whole trick: Discord now hears your translated voice."),
        ).pack(anchor="w", pady=8)
        ctk.CTkButton(self.body, text="Open Discord voice settings",
                      fg_color="#5865F2", hover_color="#4752c4",
                      command=lambda: os.startfile("discord://-/settings/voice-and-video")).pack(pady=8)  # noqa: S605

    def _step4(self) -> None:  # Done — save keys
        self.header.configure(text="You're set 🎉")
        self.sub.configure(text="Settings saved. Launch the interpreter and talk.")
        saved = self._save_keys()
        ctk.CTkLabel(
            self.body, justify="center",
            text=("Keys stored ✓  (you can change them later in Settings)\n"
                  f"Saved to: {app_env_path()}"),
            text_color=GREEN if saved else RED,
            font=ctk.CTkFont("Segoe UI Semibold", 13, "bold"),
        ).pack(pady=16)
        self.next_btn.configure(text="Finish")

    # -- navigation ------------------------------------------------------ #
    def _validate_keys(self) -> bool:
        self.status.configure(text="Validating keys…", text_color="#9ca3af")
        self.root.update()
        values = {env: e.get().strip() for env, e in self.key_rows}
        try:
            from groq import Groq

            if values["GROQ_API_KEY"]:
                Groq(api_key=values["GROQ_API_KEY"]).models.list()
        except Exception as exc:  # noqa: BLE001
            self.status.configure(text=f"Groq key invalid: {exc}", text_color=RED)
            return False
        try:
            import requests

            if values["ELEVENLABS_API_KEY"]:
                r = requests.get(
                    "https://api.elevenlabs.io/v1/voices",
                    headers={"xi-api-key": values["ELEVENLABS_API_KEY"]},
                    timeout=15,
                )
                if r.status_code != 200:
                    self.status.configure(
                        text=f"ElevenLabs key invalid ({r.status_code})", text_color=RED)
                    return False
        except Exception as exc:  # noqa: BLE001
            self.status.configure(text=f"ElevenLabs check failed: {exc}", text_color=RED)
            return False
        self.status.configure(text="Keys valid ✓", text_color=GREEN)
        self._pending_keys = values
        return True

    def _save_keys(self) -> bool:
        path = app_env_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            values = getattr(self, "_pending_keys", {})
            for env, entry in self.key_rows:
                val = values.get(env) or entry.get().strip()
                if val:
                    lines.append(f"{env}={val}")
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            for line in lines:
                k, _, v = line.partition("=")
                os.environ[k] = v
            return True
        except Exception as exc:  # noqa: BLE001
            self.sub.configure(text=f"Could not save keys: {exc}", text_color=RED)
            return False

    def _next(self) -> None:
        if self.step == 1 and not self._validate_keys():
            return
        if self.step == len(self.STEPS) - 1:
            self._finish()
            return
        self._show(self.step + 1)

    def _back(self) -> None:
        self._show(self.step - 1)

    def _finish(self) -> None:
        from stream_gui import mark_setup_complete

        mark_setup_complete()
        self.root.destroy()

    def _cancel(self) -> None:
        # Closing mid-setup simply exits the app; nothing was harmed.
        self.root.destroy()
        raise SystemExit(0)