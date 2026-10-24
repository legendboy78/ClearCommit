#!/usr/bin/env python3
"""Live streaming interpreter with a modern CustomTkinter control panel.

Dark Windows-11 style UI: settings sidebar (language, models, runtime knobs),
two caption panes with live VU input meters, latency chip, error strip, and
copy/save/clear/pause controls. The interpreter runs in a background thread;
all UI updates go through ``after()`` polling (thread-safe).

Example
-------
    python stream_gui.py --person1-lang es --direction user
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

from config import (
    GROQ_TRANSLATE_MODELS,
    SUPPORTED_LANGUAGES,
    Settings,
    app_data_dir,
    select_realtime_tts_backend,
)
from logger import attach_session_file, configure_root, get_logger
from setup_wizard import keys_present
from stream import build_arg_parser
from stream_pipeline import StreamingInterpreter, VAD_THRESHOLD

from clearcommit_integration import (
    ClearCommitIntegration, CoachingUIResult, MeetingUIResult,
)
from ai_providers import ProviderConfig
from ai_service import AIRequestStatus
from ai_config import (
    AIModelSettings, ALL_FEATURES, DEFAULT_SYSTEM_PROMPTS,
    FEATURE_ASSISTANT, FEATURE_COACH_COMM, FEATURE_COACH_IDIOM,
    FEATURE_COACH_PITCH, FEATURE_COACH_TONE, FEATURE_MEETING,
    normalize_override, sanitize_overrides,
)
from mode_config import CommunicationMode, ModeConfig

logger = get_logger("stream_gui")

APP_VERSION = "1.0.0"
_RELEASES_API = "https://api.github.com/repos/legendboy78/Voice-end-to-end/releases/latest"
# One-slot handoff from the update-check thread to the UI tick.
_UPDATE_INFO = {"tag": None, "url": None}

APP_NAME = "LiveInterpreter"
_MUTEX_NAME = f"Global\\{APP_NAME}_SingleInstance"
_ERR_ALREADY_EXISTS = 183


def _acquire_single_instance() -> bool:
    """True if this is the only running instance (Windows named mutex).

    Two instances would fight over the microphone/CABLE devices and both end
    up with silence — the second launch exits with a friendly message.
    """
    import ctypes

    ctypes.windll.kernel32.CreateMutexW(None, False, _MUTEX_NAME)
    return ctypes.windll.kernel32.GetLastError() != _ERR_ALREADY_EXISTS


def _disable_console_quickedit() -> None:
    """Stop the console's click-to-select mode from FREEZING the whole app.

    On Windows, selecting text in the console blocks every write to
    stdout/stderr — including log calls from the UI thread — so a stray click
    in the terminal hangs the entire interpreter until Enter is pressed.
    Disabling QuickEdit makes the console safe to click.
    """
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE
        mode = ctypes.c_uint()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value & ~0x0040)  # QUICK_EDIT
    except Exception:  # noqa: BLE001 - no console (windowed exe) is fine
        pass


def _check_update_async() -> None:
    """Fire-and-forget update check -> UI toast when a newer release exists.

    Point _RELEASES_API at a repo with public releases; a private/offline
    repo simply 404s and this silently does nothing.
    """
    import json
    import urllib.request

    def work():
        try:
            req = urllib.request.Request(
                _RELEASES_API, headers={"User-Agent": "LiveInterpreter"}
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.load(resp)
            tag = str(data.get("tag_name", "")).lstrip("v")
            if tag and tag.split(".") > APP_VERSION.split("."):
                _UPDATE_INFO["tag"] = tag
                _UPDATE_INFO["url"] = data.get("html_url", "")
        except Exception:  # noqa: BLE001 - offline is perfectly fine
            pass

    threading.Thread(target=work, daemon=True).start()


def _sweep_stale_temp() -> None:
    """Delete leftover tmp*.wav utterance files from crashed old sessions."""
    try:
        import tempfile
        cutoff = time.time() - 24 * 3600
        for f in Path(tempfile.gettempdir()).glob("tmp*.wav"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink(missing_ok=True)
            except OSError:  # noqa: PERF203
                pass
    except Exception:  # noqa: BLE001
        pass

# --------------------------------------------------------------------------- #
# Visual palette — one restrained blue-violet accent, quiet dark neutrals.
# --------------------------------------------------------------------------- #
ACCENT = "#6c8cff"          # primary accent (buttons, selections)
ACCENT_HOVER = "#5a76e8"
GREEN = "#3fd68f"           # positive / live
RED = "#ef4444"             # errors / destructive
AMBER = "#e8b13f"           # paused / attention
PANE_BG = "#12161d"         # transcript surface (lighter than app bg = depth)
METER_BG = "#171b22"
COLORS = {
    "bg": "#0d1014",
    "panel": "#141922",
    "panel_alt": "#1a2130",
    "field": "#1d2433",
    "field_hover": "#263043",
    "border": "#242c3c",
    "text": "#e6eaf2",
    "text_dim": "#a4aec2",
    "text_faint": "#6e7a92",
    "accent": ACCENT,
    "accent_hover": ACCENT_HOVER,
    "green": GREEN,
    "red": RED,
    "amber": AMBER,
}
# Older captions dim to a uniform muted tone so the recent conversation
# visually dominates without changing readability of anything current.
AGED_FG = "#68738a"
RECENT_BLOCKS_KEPT = 6      # newest caption blocks at full brightness

# AI workspace: compact secondary area below the transcripts. On short
# windows the transcripts win — the workspace auto-collapses. Thresholds are
# generous enough for 125% DPI displays (where physical height > logical).
WORKSPACE_COLLAPSE_BELOW = 700
WORKSPACE_RESTORE_ABOVE = 750
WORKSPACE_BODY_HEIGHT = 180


def _workspace_auto_state(height: float, collapsed: bool,
                          auto_collapsed: bool) -> tuple[bool, bool]:
    """Auto-collapse the AI workspace on short windows (hysteresis).

    Prioritizes live captions: below ``WORKSPACE_COLLAPSE_BELOW`` px the
    workspace collapses; it restores automatically above
    ``WORKSPACE_RESTORE_ABOVE`` px, but only if the collapse was automatic
    (never overrides a manual user choice).
    """
    if height < WORKSPACE_COLLAPSE_BELOW and not collapsed:
        return True, True
    if height >= WORKSPACE_RESTORE_ABOVE and auto_collapsed and collapsed:
        return False, False
    return collapsed, auto_collapsed


def _compose_counts_text(coaching_on: bool, coach_n: int,
                         meeting_counts: "tuple[int, int] | None",
                         agree_counts: "tuple[int, int, int] | None") -> str:
    """Compose the compact AI-workspace status text (pure, testable).

    Only real data produces text: no suggestion/decision/ledger data means an
    empty string (visually quiet). Coaching off states it once, briefly.
    """
    if not coaching_on:
        return "AI coaching off"
    parts: list[str] = []
    if coach_n:
        parts.append(f"{coach_n} suggestion" + ("s" if coach_n != 1 else ""))
    if meeting_counts:
        decided, open_q = meeting_counts
        bits = []
        if decided:
            bits.append(f"{decided} decided")
        if open_q:
            bits.append(f"{open_q} open")
        if bits:
            parts.append(" · ".join(bits))
    if agree_counts:
        open_n, confirmed, rejected = agree_counts
        bits = []
        if open_n:
            bits.append(f"{open_n} open")
        if confirmed:
            bits.append(f"{confirmed} confirmed")
        if rejected:
            bits.append(f"{rejected} rejected")
        if bits:
            parts.append(" · ".join(bits))
    return "  |  ".join(parts)


def _language_options() -> list[str]:
    """Human-readable entries like 'Spanish (es)' sorted by language name."""
    return sorted(f"{name} ({code})" for code, name in SUPPORTED_LANGUAGES.items())


def _code_from_label(label: str) -> str:
    return label.rsplit("(", 1)[-1].rstrip(")")


def _settings_path() -> Path:
    from config import app_data_dir

    return app_data_dir() / "settings.json"


def load_settings() -> dict:
    try:
        import json

        # utf-8-sig tolerates a BOM (PowerShell's Set-Content -Encoding UTF8
        # writes one; a stray BOM must not silently nuke all settings).
        with open(_settings_path(), encoding="utf-8-sig") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 - missing/corrupt file = defaults
        return {}


def mark_setup_complete() -> None:
    data = load_settings()
    data["wizard_done"] = True
    save_settings(data)


def save_settings(data: dict) -> None:
    try:
        import json

        path = _settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
    except Exception:  # noqa: BLE001
        pass


class ControlBox:
    def __init__(
        self, interp: StreamingInterpreter, direction: str,
        settings: "dict | None" = None,
        integration: "ClearCommitIntegration | None" = None,
    ):
        self.interp = interp
        self.direction = direction
        self.settings = settings or {}
        self.mode_config = ModeConfig.from_dict(self.settings.get("mode_config"))
        self.integration = integration
        if self.integration is not None:
            # Fix 1 (Tk thread safety): worker threads invoke this callback,
            # so it must never touch Tk directly. Marshal onto the Tk main
            # thread via the existing UI queue (drained by _tick).
            self.integration.set_ui_callback(
                lambda: self._ui_call(self._on_clearcommit_results)
            )

        ctk.set_appearance_mode("dark")
        self.root = ctk.CTk()
        self.root.title("ClearCommit")
        self.root.geometry("1180x720")
        self.root.minsize(700, 420)
        self.root.configure(fg_color="#14161a")

        # uid -> (widget, mark): translations fill in under their exact original
        # even when a later utterance was already appended (mark moves with text).
        self._pending: "dict[int, tuple[tk.Text, str]]" = {}
        self._paused = False
        self._prev_vad: dict[int, float] = {}
        self._err_clear_after = None
        # Mini caption HUD shown while the main window is minimized.
        self._overlay: "ctk.CTkToplevel | None" = None
        self._ov_drag: "tuple[int, int] | None" = None
        self._last_p1 = {"orig": "", "trans": ""}
        # UX state.
        self._font_size = max(9, min(20, int(self.settings.get("font_size", 12))))
        self._timestamps = bool(self.settings.get("timestamps", False))
        self._stick = {"p1": True, "you": True}   # autoscroll per pane
        # Caption block starts per pane (recent/older dimming boundaries).
        self._block_starts: "dict[str, list[str]]" = {"p1": [], "you": []}
        self._line_count = 0
        self._toast_after = None
        # Device-loss watchdog state: capture id -> (last cb_count, monotonic)
        self._cb_seen: "dict[int, tuple[int, float]]" = {}
        self._lines_since_autosave = 0
        self._settings_dirty = False
        self._persist_after = None
        self._last_restart = 0.0
        # Thread-safe bridge: worker threads queue callables; the UI poller
        # runs them on the Tk thread. NEVER touch Tk from another thread —
        # cross-thread Tk calls corrupt Tcl state and hard-crash the UI.
        self._ui_q: "queue.Queue" = queue.Queue()

        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        # Sidebar: weight=0 (fixed width) but can be collapsed entirely.
        self.root.grid_columnconfigure(0, weight=0)
        self._build_header()
        self._build_sidebar()
        self._build_panes()
        self._build_footer()

        self.root.protocol("WM_DELETE_WINDOW", self._on_stop)
        # ONE unified UI ticker (150ms) instead of three competing timer
        # chains — fewer Tcl callbacks, far less churn, no UI overload.
        self._tick_count = 0
        self._last_chip_text = ""
        self._iconified = False
        self._voices: list[dict] = []
        self._voice_ids: dict[str, str] = {}
        self._schedule_tick()
        self._restore_settings()
        self._bind_shortcuts()

    def _schedule_tick(self) -> None:
        self.root.after(150, self._tick)

    def _tick(self) -> None:
        try:
            self._drain_ui_calls()
            self._tick_count += 1

            if _UPDATE_INFO["tag"] and not getattr(self, "_update_shown", False):
                self._update_shown = True
                self.show_error(
                    f"Update v{_UPDATE_INFO['tag']} available — "
                    f"download: {_UPDATE_INFO['url']}"
                )

            # HUD show/hide (Tk calls the minimized state "iconic").
            try:
                state = self.root.state()
            except Exception:  # noqa: BLE001
                state = "normal"
            self._iconified = state in ("iconic", "iconified")
            if self._iconified:
                self._show_overlay(True)
            elif self._overlay is not None and self._overlay.state() == "normal":
                self._show_overlay(False)

            self._drain_captions()
            self._poll_clearcommit_results()

            if self._tick_count % 4 == 0:      # responsive workspace ~2.6x/s
                self._check_workspace_height()
            if self._tick_count % 2 == 0:      # status pulse ~3.3x/s
                self._pulse_live_dot()

            if not self._iconified:
                if self._tick_count % 2 == 0:      # meters ~4x/s
                    self._update_meters()
                if self._tick_count % 7 == 0:      # status chip ~1x/s
                    self._update_status_chip()
                if self._tick_count % 13 == 0:     # watchdog ~2s
                    self._watchdog_check()
        finally:
            self._schedule_tick()

    # ------------------------------------------------------------------ #
    def _restore_settings(self) -> None:
        s = self.settings
        if s.get("win_geom"):
            try:
                self.root.geometry(s["win_geom"])
            except Exception:  # noqa: BLE001
                pass
        if s.get("always_on_top"):
            self.top_switch.select()
            self.root.attributes("-topmost", True)
        if s.get("noise_cancel"):
            self.noise_switch.select()   # applied to captures on start below
        for c in getattr(self.interp, "_captures", []):
            c.noise_cancel = bool(self.noise_switch.get())
        if s.get("auto_detect"):
            self.interp._force_language = False
            self.detect_switch.select()
        if s.get("monitor_tts"):
            self.interp.set_monitor_tts(True)
            self.monitor_switch.select()
        if s.get("consent_chime"):
            self.interp._consent_chime = True
            self.consent_switch.select()
        if s.get("voice_id"):
            self.interp.set_voice(s["voice_id"])
        # Restore STT backend (Groq Whisper or Deepgram Nova-3).
        try:
            if s.get("stt_backend") == "deepgram":
                self.interp._stt_backend = "deepgram"
                from transcriber import DeepgramTranscriber
                self.interp.transcriber_p1 = DeepgramTranscriber()
                self.interp.transcriber_user = DeepgramTranscriber()
                self.interp.transcriber = self.interp.transcriber_p1
                self.stt_provider_seg.set("Deepgram Nova-3")
                self.stt_quality_seg.configure(state="disabled")
        except Exception:
            pass
        # Audio knobs (safe defaults when missing/corrupt).
        try:
            # Mic echo guard: force OFF — the hallucination filter handles
            # TTS artifacts. A saved "true" from an older version would
            # block speech during TTS playback.
            self.interp._mic_echo_guard = False
            self.echo_switch.deselect()
        except Exception:  # noqa: BLE001
            pass
        try:
            if "vad_threshold" in s:
                vad = max(0.002, min(0.05, float(s["vad_threshold"])))
                for c in getattr(self.interp, "_captures", []):
                    c.vad_threshold = vad
                self.vad_slider.set(vad)
                self.vad_val.configure(text=f"{vad:.3f}")
        except Exception:  # noqa: BLE001
            pass
        try:
            if s.get("tts_mode") in ("utterance", "streaming"):
                self.interp._tts_mode = s["tts_mode"]
                self.tts_style_seg.set(
                    "Fast" if s["tts_mode"] == "streaming" else "Natural")
        except Exception:  # noqa: BLE001
            pass
        # AI workspace: restore last tab + collapsed state (safe defaults).
        try:
            tab = s.get("ai_tab")
            if tab in self.AI_TABS:
                self.ai_tabs.set(tab)
                self._show_ai_view(tab)
        except Exception:  # noqa: BLE001
            pass
        try:
            if "workspace_collapsed" in s:
                # A previously saved state is a deliberate user choice.
                self._ws_collapsed = bool(s.get("workspace_collapsed"))
                self._ws_user_toggled = True
                self._apply_workspace_state()
        except Exception:  # noqa: BLE001
            pass
        # Settings have settled: align every state-aware empty state with the
        # restored configuration (coaching ON must never read as "enable it").
        self._refresh_coach_empty_state()
        self._update_ai_counts()
        # Restore the sidebar visible state.
        try:
            if s.get("sidebar_visible") is False:
                self._sidebar_visible = False
                self._sidebar_frame.grid_remove()
        except Exception:  # noqa: BLE001
            pass
        # Restore the advanced-settings disclosure state.
        try:
            if s.get("advanced_open"):
                self.adv_open = True
                self.adv_frame.grid()
                self.adv_toggle.configure(text="Hide advanced settings")
        except Exception:  # noqa: BLE001
            pass
        self._change_font(0)  # apply saved caption size to tags

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-Shift-C>", lambda e: self._clear_captions())
        self.root.bind("<Control-Shift-S>", lambda e: self._save_transcript())
        self.root.bind("<Control-Shift-P>", lambda e: self._toggle_pause())
        self.root.bind("<Control-Shift-X>", lambda e: self._on_stop())

    # ------------------------------------------------------------------ #
    # Layout builders
    # ------------------------------------------------------------------ #
    def _build_header(self) -> None:
        bar = ctk.CTkFrame(self.root, fg_color=COLORS["panel"], corner_radius=0,
                           height=46)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        bar.grid_propagate(False)
        bar.grid_columnconfigure(3, weight=1)

        # Product identity.
        ctk.CTkLabel(
            bar, text="ClearCommit",
            font=ctk.CTkFont("Segoe UI Semibold", 15, "bold"),
            text_color=COLORS["text"],
        ).grid(row=0, column=0, padx=(16, 2), pady=8)
        ctk.CTkLabel(
            bar, text="LIVE INTERPRETER",
            font=ctk.CTkFont("Segoe UI", 10),
            text_color=COLORS["text_faint"],
        ).grid(row=0, column=1, pady=8, sticky="w")

        # Real interpreter status pill (ready / live / paused / error — all
        # derived from actual runtime state; error is set by show_error()).
        self.live_dot = ctk.CTkLabel(
            bar, text="●  Ready", font=ctk.CTkFont("Segoe UI Semibold", 12),
            text_color=COLORS["text_dim"],
        )
        self.live_dot.grid(row=0, column=2, padx=(18, 8), pady=8)

        # One right-aligned info line (language context + real latency/TTS).
        # A single quiet label instead of competing chips.
        self.info_chip = ctk.CTkLabel(
            bar, text="", font=ctk.CTkFont("Segoe UI", 12),
            text_color=COLORS["text_dim"], anchor="e",
        )
        self.info_chip.grid(row=0, column=5, padx=(8, 16), pady=8)

        # Hairline separating the header chrome from the content below.
        bar.grid_rowconfigure(1, weight=0)
        ctk.CTkFrame(bar, fg_color=COLORS["border"], corner_radius=0,
                     height=1).grid(row=1, column=0, columnspan=6, sticky="ew")

        # Sidebar collapse/expand toggle (a real space-saver on narrow windows).
        self.sidebar_toggle = ctk.CTkButton(
            bar, text="☰", width=32, height=28,
            fg_color="transparent", hover_color=COLORS["field"],
            text_color=COLORS["text_dim"],
            font=ctk.CTkFont("Segoe UI", 14),
            command=self._on_toggle_sidebar,
        )
        self.sidebar_toggle.grid(row=0, column=0, padx=(4, 0), pady=8)

    def _build_sidebar(self) -> None:
        sb = ctk.CTkFrame(self.root, fg_color=COLORS["panel"], corner_radius=0,
                          width=248)
        self._sidebar_frame = sb
        sb.grid(row=1, column=0, sticky="nsw")
        sb.grid_propagate(False)
        sb.grid_columnconfigure(0, weight=1)
        sb.grid_rowconfigure(0, weight=1)

        # Scrollable content: every setting stays reachable on short windows.
        sbi = ctk.CTkScrollableFrame(
            sb, fg_color="transparent", scrollbar_button_color=COLORS["field"],
            scrollbar_button_hover_color=COLORS["field_hover"],
        )
        sbi.grid(row=0, column=0, sticky="nsew")
        sbi.grid_columnconfigure(0, weight=1)

        def section(text: str, row: int, parent=None) -> None:
            ctk.CTkLabel(
                parent or sbi, text=text, anchor="w",
                text_color=COLORS["text_dim"],
                font=ctk.CTkFont("Segoe UI", 11),
            ).grid(row=row, column=0, padx=16, pady=(16, 4), sticky="ew")

        def field_btn(**kw):
            kw.setdefault("fg_color", COLORS["field"])
            kw.setdefault("button_color", ACCENT)
            kw.setdefault("button_hover_color", ACCENT_HOVER)
            kw.setdefault("font", ctk.CTkFont("Segoe UI", 12))
            kw.setdefault("width", 196)
            kw.setdefault("height", 28)
            return ctk.CTkOptionMenu(sbi, **kw)

        def segmented(values: list, command) -> ctk.CTkSegmentedButton:
            return ctk.CTkSegmentedButton(
                sbi, values=values, width=196, height=28,
                selected_color=ACCENT, selected_hover_color=ACCENT_HOVER,
                unselected_color=COLORS["field"], fg_color=COLORS["field"],
                font=ctk.CTkFont("Segoe UI", 12), command=command,
            )

        def switch(text: str, command) -> ctk.CTkSwitch:
            return ctk.CTkSwitch(
                sbi, text=text, width=196, progress_color=ACCENT,
                font=ctk.CTkFont("Segoe UI", 12), command=command,
            )

        # -- Core: always visible (what users touch most) ------------------ #
        section("Communication", 0)
        self.mode_menu = segmented(["Social", "Professional"], self._on_mode)
        self.mode_menu.set(
            "Professional" if self.mode_config.is_professional() else "Social"
        )
        self.mode_menu.grid(row=1, column=0, padx=16, sticky="ew")
        self.coaching_switch = switch("AI coaching suggestions", self._on_coaching)
        if self.mode_config.coaching_active():
            self.coaching_switch.select()
        self.coaching_switch.grid(row=2, column=0, padx=16, pady=(6, 4), sticky="w")

        section("Languages", 3)
        self.lang_menu = field_btn(values=_language_options(),
                                   command=self._on_language)
        self.lang_menu.set(self._current_label())
        self.lang_menu.grid(row=4, column=0, padx=16, sticky="ew")
        self.voice_menu = field_btn(values=["loading voices…"],
                                    command=self._on_voice)
        self.voice_menu.set("loading voices…")
        self.voice_menu.grid(row=5, column=0, padx=16, pady=(6, 0), sticky="ew")
        self.root.after(400, self._load_voices)

        section("Translation quality", 6)
        self.model_seg = segmented(["Accurate", "Fast"], self._on_model)
        self.model_seg.set(self._current_model_label())
        self.model_seg.grid(row=7, column=0, padx=16, sticky="ew")

        section("Your speech recognition", 8)
        self.stt_provider_seg = segmented(["Groq Whisper", "Deepgram Nova-3"],
                                          self._on_stt_provider)
        current_stt = getattr(self.interp, "_stt_backend", "groq")
        self.stt_provider_seg.set(
            "Deepgram Nova-3" if current_stt == "deepgram" else "Groq Whisper")
        self.stt_provider_seg.grid(row=9, column=0, padx=16, sticky="ew")

        self.stt_quality_seg = segmented(["Accurate", "Turbo"], self._on_stt)
        self.stt_quality_seg.set("Accurate")
        self.stt_quality_seg.grid(row=10, column=0, padx=16, pady=(6, 0), sticky="ew")
        self.stt_quality_seg.configure(
            state="disabled" if current_stt == "deepgram" else "normal")

        # -- Advanced: collapsible disclosure ------------------------------ #
        self.adv_open = bool(self.settings.get("advanced_open", False))
        self.adv_toggle = ctk.CTkButton(
            sbi, text=("Hide advanced settings" if self.adv_open
                       else "Show advanced settings"),
            width=196, height=26,
            fg_color="transparent", hover_color=COLORS["field"],
            text_color=COLORS["text_dim"],
            font=ctk.CTkFont("Segoe UI", 11),
            command=self._on_toggle_advanced,
        )
        self.adv_toggle.grid(row=10, column=0, padx=16, pady=(10, 2), sticky="ew")

        adv = ctk.CTkFrame(sbi, fg_color="transparent")
        self.adv_frame = adv  # kept for the disclosure toggle
        adv.grid(row=11, column=0, sticky="ew")
        adv.grid_columnconfigure(0, weight=1)

        def adv_section(text: str, row: int) -> None:
            section(text, row, parent=adv)

        def adv_switch(text: str, command) -> ctk.CTkSwitch:
            return ctk.CTkSwitch(
                adv, text=text, width=196, progress_color=ACCENT,
                font=ctk.CTkFont("Segoe UI", 12), command=command,
            )

        adv_section("Audio", 0)
        self.noise_switch = adv_switch("Noise cancel (experimental)",
                                       self._on_noise)
        self.noise_switch.deselect()
        self.noise_switch.grid(row=1, column=0, padx=16, pady=(4, 4), sticky="w")

        sil_frame = ctk.CTkFrame(adv, fg_color="transparent")
        sil_frame.grid(row=2, column=0, padx=16, sticky="ew")
        self.sil_val = ctk.CTkLabel(
            sil_frame, text=f"{self.interp._silence_timeout:.2f} s",
            font=ctk.CTkFont("Segoe UI", 12), text_color=COLORS["text_dim"],
            width=48,
        )
        self.sil_val.pack(side="right")
        ctk.CTkLabel(
            sil_frame, text="Silence timeout", anchor="w",
            font=ctk.CTkFont("Segoe UI", 12),
        ).pack(side="left")
        self.sil_slider = ctk.CTkSlider(
            adv, from_=0.3, to=1.5, number_of_steps=24, width=196,
            progress_color=ACCENT, button_color=ACCENT,
            command=self._on_silence,
        )
        self.sil_slider.set(self.interp._silence_timeout)
        self.sil_slider.grid(row=3, column=0, padx=16, sticky="ew")

        # Mic sensitivity: lower = hears distant/quiet speech better.
        vad_frame = ctk.CTkFrame(adv, fg_color="transparent")
        vad_frame.grid(row=4, column=0, padx=16, sticky="ew")
        self.vad_val = ctk.CTkLabel(
            vad_frame, text=f"{VAD_THRESHOLD:.3f}",
            font=ctk.CTkFont("Segoe UI", 12), text_color=COLORS["text_dim"],
            width=48,
        )
        self.vad_val.pack(side="right")
        ctk.CTkLabel(
            vad_frame, text="Mic sensitivity", anchor="w",
            font=ctk.CTkFont("Segoe UI", 12),
        ).pack(side="left")
        self.vad_slider = ctk.CTkSlider(
            adv, from_=0.002, to=0.05, number_of_steps=24, width=196,
            progress_color=ACCENT, button_color=ACCENT,
            command=self._on_vad,
        )
        self.vad_slider.set(VAD_THRESHOLD)
        self.vad_slider.grid(row=5, column=0, padx=16, sticky="ew")

        # Hear the translated TTS on your own speakers/headphones (like
        # Windows "Listen to this device", but only while this is on).
        self.monitor_switch = adv_switch("Hear TTS locally", self._on_monitor)
        if getattr(self.interp, "_monitor_tts", False):
            self.monitor_switch.select()
        self.monitor_switch.grid(row=6, column=0, padx=16, pady=(6, 4), sticky="w")

        # Translated voice rendering: Natural generates the whole utterance
        # for continuous prosody; Fast streams lower-latency, choppier audio.
        adv_section("Translated voice style", 5)
        self.tts_style_seg = ctk.CTkSegmentedButton(
            adv, values=["Natural", "Fast"], width=196, height=28,
            selected_color=ACCENT, selected_hover_color=ACCENT_HOVER,
            unselected_color=COLORS["field"], fg_color=COLORS["field"],
            font=ctk.CTkFont("Segoe UI", 12), command=self._on_tts_style,
        )
        self.tts_style_seg.set(
            "Fast" if getattr(self.interp, "_tts_mode", "utterance") == "streaming"
            else "Natural"
        )
        self.tts_style_seg.grid(row=6, column=0, padx=16, sticky="ew")

        # Mic echo guard: drops mic captures while our TTS plays (speaker
        # leakage made Whisper transcribe the PC's voice as yours). OFF =
        # allow barge-in over the translated voice at your own risk.
        self.echo_switch = adv_switch("Mic echo guard", self._on_echo_guard)
        if getattr(self.interp, "_mic_echo_guard", False):
            self.echo_switch.select()
        self.echo_switch.grid(row=7, column=0, padx=16, pady=(8, 4), sticky="w")

        self.consent_switch = adv_switch(
            "Say 'call is translated' notice", self._on_consent,
        )
        if getattr(self.interp, "_consent_chime", False):
            self.consent_switch.select()
        self.consent_switch.grid(row=8, column=0, padx=16, pady=(4, 4), sticky="w")

        # -- View ---------------------------------------------------------- #
        adv_section("View", 9)
        self.top_switch = adv_switch("Always on top", self._on_topmost)
        self.top_switch.grid(row=10, column=0, padx=16, pady=(4, 4), sticky="w")

        self.ts_switch = adv_switch("Timestamps", self._on_timestamps)
        self.ts_switch.grid(row=11, column=0, padx=16, pady=(4, 4), sticky="w")
        if self._timestamps:
            self.ts_switch.select()

        # OFF (default): Whisper is locked to the selected language. ON:
        # auto-detect what's actually spoken (Telugu, ...) and translate from
        # that instead — for when Person 1 speaks a different language than
        # the dropdown says.
        self.detect_switch = adv_switch("Auto-detect spoken language",
                                        self._on_detect)
        if not getattr(self.interp, "_force_language", True):
            self.detect_switch.select()
        self.detect_switch.grid(row=12, column=0, padx=16, pady=(4, 4), sticky="w")

        fs_frame = ctk.CTkFrame(adv, fg_color="transparent")
        fs_frame.grid(row=13, column=0, padx=16, pady=(6, 8), sticky="ew")
        ctk.CTkLabel(
            fs_frame, text="Caption size", anchor="w",
            font=ctk.CTkFont("Segoe UI", 12),
        ).pack(side="left")
        minus = ctk.CTkButton(
            fs_frame, text="−", width=30, height=26,
            fg_color=COLORS["field"], hover_color=COLORS["field_hover"],
            command=lambda: self._change_font(-1),
        )
        minus.pack(side="right", padx=(4, 0))
        plus = ctk.CTkButton(
            fs_frame, text="+", width=30, height=26,
            fg_color=COLORS["field"], hover_color=COLORS["field_hover"],
            command=lambda: self._change_font(1),
        )
        plus.pack(side="right")

        if not self.adv_open:
            adv.grid_remove()

    def _build_panes(self) -> None:
        wrap = ctk.CTkFrame(self.root, fg_color="transparent")
        wrap.grid(row=1, column=1, sticky="nsew", padx=(0, 0))
        wrap.grid_columnconfigure(0, weight=1, uniform="pane")
        wrap.grid_columnconfigure(1, weight=1, uniform="pane")
        wrap.grid_rowconfigure(2, weight=1)   # transcripts take all spare space

        self.pane_headers: dict[str, ctk.CTkLabel] = {}
        self.pane_status: dict[str, ctk.CTkLabel] = {}
        for col, (side, title) in enumerate((
            ("p1", "Person 1 · system / Discord"),
            ("you", "You · microphone"),
        )):
            head = ctk.CTkFrame(wrap, fg_color="transparent")
            head.grid(row=0, column=col, padx=(12, 6) if col == 0 else (6, 12),
                      pady=(10, 2), sticky="ew")
            head.grid_columnconfigure(0, weight=1)
            label = ctk.CTkLabel(
                head, text=title, anchor="w",
                font=ctk.CTkFont("Segoe UI Semibold", 12, "bold"),
                text_color=self.SIDE_TEXT[side],
            )
            label.grid(row=0, column=0, sticky="w")
            # Real-time state from the capture's live RMS: a speaker-colored
            # dot + "Listening" only while the mic/loopback is actually above
            # the VAD threshold (real state — never fabricated).
            status = ctk.CTkLabel(
                head, text="", anchor="e",
                font=ctk.CTkFont("Segoe UI Semibold", 10),
                text_color=self.SIDE_TEXT[side],
            )
            status.grid(row=0, column=1, sticky="e")
            self.pane_headers[side] = label
            self.pane_status[side] = status

        self.meters: dict[str, list] = {}
        for side, col in (("p1", 0), ("you", 1)):
            meter = tk.Canvas(
                wrap, height=4, highlightthickness=0, bg=METER_BG,
            )
            meter.grid(row=1, column=col, padx=(12, 6) if col == 0 else (6, 12),
                       pady=(0, 4), sticky="ew")
            self.meters[side] = [meter, 0]

        # Requested height is the MINIMUM Tk grid guarantees; the actual pane
        # size is governed by grid weight (fills available space). A tiny floor
        # keeps the workspace bar and footer reachable at every window size —
        # a 15-line request overflowed the window when the workspace expanded
        # and pushed the footer below the visible edge.
        text_kw = dict(height=1, wrap="word", bg=PANE_BG, fg=COLORS["text"],
                       insertbackground=COLORS["text"], relief="flat",
                       padx=12, pady=10, spacing1=4, spacing3=5,
                       highlightthickness=1, highlightbackground=COLORS["border"],
                       highlightcolor=COLORS["border"],
                       selectbackground="#2b3b5e", selectforeground=COLORS["text"],
                       font=("Segoe UI", self._font_size))
        self.p1_text = tk.Text(wrap, **text_kw)
        self.p1_text.grid(row=2, column=0, padx=(12, 6), pady=(0, 6), sticky="nsew")
        self.you_text = tk.Text(wrap, **text_kw)
        self.you_text.grid(row=2, column=1, padx=(6, 12), pady=(0, 6), sticky="nsew")

        # "Jump to new lines" pills (visible when the user scrolled up).
        self.pills: dict[str, ctk.CTkButton] = {}
        for side, widget in (("p1", self.p1_text), ("you", self.you_text)):
            pill = ctk.CTkButton(
                wrap, text="New lines ↓", width=92, height=24,
                fg_color=ACCENT, hover_color=ACCENT_HOVER,
                font=ctk.CTkFont("Segoe UI", 10),
                command=lambda s=side: self._jump_to_bottom(s),
            )
            pill.place(in_=widget, relx=1.0, rely=1.0, anchor="se", x=-8, y=-8)
            pill.place_forget()
            self.pills[side] = pill

        self._show_hints()

        hint = ctk.CTkLabel(
            wrap, anchor="w", text_color=COLORS["text_faint"],
            font=ctk.CTkFont("Segoe UI", 10),
            text=("Heard from the system → English captions"
                  "          ·          your speech → spoken in their language"),
        )
        hint.grid(row=3, column=0, columnspan=2, padx=12, pady=(0, 4), sticky="w")

        # Red strip for pipeline errors (thin; hidden while empty).
        self.err_label = ctk.CTkLabel(
            wrap, text="", anchor="w", justify="left",
            font=ctk.CTkFont("Segoe UI", 11), text_color=RED,
            fg_color="#2a1518",
        )
        self.err_label.grid(row=4, column=0, columnspan=2, padx=12, pady=(0, 4),
                            sticky="ew")
        # Hidden until a real error arrives (show_error re-grids it).
        self.err_label.grid_remove()

        self._build_ai_workspace(wrap, row=5)

    # ------------------------------------------------------------------ #
    # Compact AI workspace: Coach / Meeting / Agreements in one tabbed,
    # collapsible strip. Only the selected view occupies space; the live
    # transcripts always stay visible above it.
    # ------------------------------------------------------------------ #
    AI_TABS = ("Coach", "Assistant", "Meeting", "Agreements", "Models")

    # Prompt-editor feature order (feature id, display label).
    _PROMPT_FEATURES = (
        (FEATURE_COACH_IDIOM, "Coach — idioms"),
        (FEATURE_COACH_TONE, "Coach — tone"),
        (FEATURE_COACH_COMM, "Communication coach"),
        (FEATURE_COACH_PITCH, "Pitch coach"),
        (FEATURE_ASSISTANT, "Assistant"),
        (FEATURE_MEETING, "Meeting intelligence"),
    )

    def _build_ai_workspace(self, wrap, row: int) -> None:
        self._ws_collapsed = True       # default: transcripts win
        self._ws_auto_collapsed = False
        self._ws_user_toggled = False
        self._ws_auto_expanded_once = False
        self._ws_last_height = 0
        # Real data counters for the quiet status text (no fabricated data).
        self._counts_coach = 0
        self._counts_meeting = None      # (decided, open) | None
        self._counts_agree = None        # (open, confirmed, rejected) | None

        # One surface (not card-in-card): header row + hairline divider + body.
        panel = ctk.CTkFrame(wrap, fg_color=COLORS["panel"], corner_radius=8)
        panel.grid(row=row, column=0, columnspan=2, padx=12, pady=(4, 8), sticky="ew")
        panel.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(panel, fg_color="transparent", height=36)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)
        header.grid_columnconfigure(1, weight=1)

        self.ai_tabs = ctk.CTkSegmentedButton(
            header, values=list(self.AI_TABS), height=26,
            font=ctk.CTkFont("Segoe UI", 11),
            selected_color=ACCENT, selected_hover_color=ACCENT_HOVER,
            unselected_color=COLORS["field"], fg_color=COLORS["panel"],
            command=self._on_ai_tab,
        )
        self.ai_tabs.set("Coach")
        self.ai_tabs.grid(row=0, column=0, padx=(10, 8), pady=5)

        # Quiet status: real counts only (empty string when nothing to show).
        self.ai_counts = ctk.CTkLabel(
            header, text="", anchor="w", font=ctk.CTkFont("Segoe UI", 11),
            text_color=COLORS["text_dim"],
        )
        self.ai_counts.grid(row=0, column=1, padx=4, pady=5, sticky="w")

        self.ai_toggle = ctk.CTkButton(
            header, text="Show", width=64, height=24,
            fg_color=COLORS["field"], hover_color=COLORS["field_hover"],
            font=ctk.CTkFont("Segoe UI", 10),
            command=self._toggle_workspace,
        )
        self.ai_toggle.grid(row=0, column=2, padx=(6, 10), pady=6)

        # Hairline divider between the workspace header and its content.
        divider = ctk.CTkFrame(panel, fg_color=COLORS["border"], corner_radius=0,
                               height=1)
        divider.grid(row=1, column=0, sticky="ew")

        # Body: one view visible at a time; lives inside the same surface.
        self.ai_body = ctk.CTkFrame(
            panel, fg_color="transparent", height=WORKSPACE_BODY_HEIGHT,
        )
        self.ai_body.grid(row=2, column=0, sticky="ew")
        self.ai_body.grid_propagate(False)
        self.ai_body.grid_columnconfigure(0, weight=1)
        self.ai_body.grid_rowconfigure(0, weight=1)

        # -- Coach view --------------------------------------------------- #
        self.coach_view = ctk.CTkFrame(self.ai_body, fg_color="transparent")
        self.coach_view.grid(row=0, column=0, sticky="nsew")
        self.coach_view.grid_columnconfigure(0, weight=1)
        self.coach_view.grid_rowconfigure(0, weight=1)
        self.suggestion_label = ctk.CTkLabel(
            self.coach_view,
            text="",
            anchor="center", justify="center", wraplength=680,
            text_color=COLORS["text_dim"], font=ctk.CTkFont("Segoe UI", 12),
        )
        self.suggestion_label.grid(row=0, column=0, padx=16, pady=12, sticky="nsew")
        self.suggestion_dismiss = ctk.CTkButton(
            self.coach_view, text="Dismiss", width=70, height=24,
            fg_color=COLORS["field"], hover_color=COLORS["field_hover"],
            font=ctk.CTkFont("Segoe UI", 10), command=self._dismiss_suggestion,
        )
        self.suggestion_dismiss.grid(row=0, column=1, padx=14, pady=12, sticky="ne")
        self.suggestion_dismiss.grid_remove()
        self._current_suggestion: Optional[CoachingUIResult] = None
        self._refresh_coach_empty_state()

        # -- Meeting view -------------------------------------------------- #
        self.meeting_view = ctk.CTkFrame(self.ai_body, fg_color="transparent")
        self.meeting_view.grid(row=0, column=0, sticky="nsew")
        self.meeting_view.grid_columnconfigure(0, weight=1)
        self.meeting_view.grid_rowconfigure(0, weight=1)
        self.meeting_text = tk.Text(
            self.meeting_view, wrap="word", bg=COLORS["panel"],
            fg=COLORS["text"], insertbackground=COLORS["text"],
            relief="flat", padx=16, pady=10, spacing1=2, spacing3=3,
            font=("Segoe UI", 10), height=8,
        )
        self.meeting_text.grid(row=0, column=0, sticky="nsew")
        self._set_meeting_text(
            "Nothing yet\n\nDecisions, action items and open questions "
            "detected in the conversation appear here.")

        # -- Agreements view ------------------------------------------------ #
        self.agreement_view = ctk.CTkFrame(self.ai_body, fg_color="transparent")
        self.agreement_view.grid(row=0, column=0, sticky="nsew")
        self.agreement_view.grid_columnconfigure(0, weight=1)
        self.agreement_view.grid_rowconfigure(0, weight=1)
        self.agreement_list = ctk.CTkScrollableFrame(
            self.agreement_view, fg_color="transparent",
            scrollbar_button_color=COLORS["field"],
            scrollbar_button_hover_color=COLORS["field_hover"],
        )
        self.agreement_list.grid(row=0, column=0, sticky="nsew")
        self.agreement_list.grid_columnconfigure(0, weight=1)
        self._agreement_version_seen = -1
        hint = ctk.CTkLabel(
            self.agreement_list,
            text="No commitments yet\n\nProposed commitments appear here for "
                 "explicit confirmation by both participants.",
            anchor="center", justify="center", text_color=COLORS["text_dim"],
            font=ctk.CTkFont("Segoe UI", 12),
        )
        hint.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        # -- Assistant view (Stage A): grounded conversation Q&A ------------ #
        self.assistant_view = ctk.CTkFrame(self.ai_body, fg_color="transparent")
        self.assistant_view.grid(row=0, column=0, sticky="nsew")
        self.assistant_view.grid_rowconfigure(0, weight=1)
        self.assistant_view.grid_columnconfigure(0, weight=1)

        self.assistant_history = tk.Text(
            self.assistant_view, wrap="word", bg=COLORS["panel"],
            fg=COLORS["text"], insertbackground=COLORS["text"],
            relief="flat", padx=14, pady=10, spacing1=2, spacing3=4,
            font=("Segoe UI", 10), height=6,
        )
        self.assistant_history.grid(row=0, column=0, sticky="nsew")
        self.assistant_history.tag_configure(
            "q", foreground=ACCENT, font=("Segoe UI", 10, "bold"))
        self.assistant_history.tag_configure(
            "a", foreground=COLORS["text"])
        self.assistant_history.tag_configure(
            "ev", foreground=COLORS["text_faint"],
            font=("Segoe UI", 9))
        self.assistant_history.tag_configure(
            "err", foreground=RED)
        self.assistant_history.tag_configure(
            "hint", foreground=COLORS["text_faint"],
            font=("Segoe UI", 10, "italic"))
        self._set_assistant_history(
            "Ask about the current conversation. Try:\n"
            "  - What are the most important points so far?\n"
            "  - What dates or deadlines were mentioned?\n"
            "  - What did each participant agree to?\n"
            "  - What is still unclear or disputed?\n"
            "  - Summarize the discussion so far.",
            tag="hint",
        )

        input_bar = ctk.CTkFrame(self.assistant_view, fg_color="transparent",
                                 height=34)
        input_bar.grid(row=1, column=0, padx=10, pady=(4, 8), sticky="ew")
        input_bar.grid_columnconfigure(0, weight=1)
        self.assistant_input = ctk.CTkEntry(
            input_bar, placeholder_text="Ask about this conversation…",
            height=28, fg_color=COLORS["field"],
            border_color=COLORS["border"], bg_color="transparent",
            font=ctk.CTkFont("Segoe UI", 11),
        )
        self.assistant_input.grid(row=0, column=0, sticky="ew")
        self.assistant_input.bind("<Return>", lambda e: self._on_assistant_send())
        self.assistant_send = ctk.CTkButton(
            input_bar, text="Send", width=70, height=28,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            font=ctk.CTkFont("Segoe UI Semibold", 11),
            command=self._on_assistant_send,
        )
        self.assistant_send.grid(row=0, column=1, padx=(6, 0))
        self.assistant_clear = ctk.CTkButton(
            input_bar, text="New chat", width=70, height=28,
            fg_color=COLORS["field"], hover_color=COLORS["field_hover"],
            font=ctk.CTkFont("Segoe UI", 11),
            command=self._on_assistant_clear,
        )
        self.assistant_clear.grid(row=0, column=2, padx=(6, 0))

        # -- Models view (Stage A4): model config + coach options + prompts -- #
        # Wrapped in a regular CTkFrame: CTkScrollableFrame doesn't reliably
        # unmap with grid_forget/grid_remove, but a plain frame does.
        self.models_view = ctk.CTkFrame(self.ai_body, fg_color="transparent")
        self.models_view.grid(row=0, column=0, sticky="nsew")
        self.models_view.grid_rowconfigure(0, weight=1)
        self.models_view.grid_columnconfigure(0, weight=1)
        models_scroll = ctk.CTkScrollableFrame(
            self.models_view, fg_color="transparent",
            scrollbar_button_color=COLORS["field"],
            scrollbar_button_hover_color=COLORS["field_hover"],
        )
        models_scroll.grid(row=0, column=0, sticky="nsew")
        models_scroll.grid_columnconfigure(0, weight=1)
        self._ai_config = AIModelSettings.from_dict(
            self.settings.get("ai_config", {}))
        self._prompt_overrides = sanitize_overrides(
            self.settings.get("prompt_overrides", {}))
        if self.integration is not None:
            self.integration.update_ai_config(
                self._ai_config, self._prompt_overrides)

        _mrow = [0]

        def mlabel(text: str, parent=None) -> None:
            ctk.CTkLabel(
                parent or models_scroll, text=text, anchor="w",
                text_color=COLORS["text_dim"],
                font=ctk.CTkFont("Segoe UI", 11),
            ).grid(row=_mrow[0], column=0, padx=12, pady=(10, 2), sticky="ew")
            _mrow[0] += 1

        def mwidget(widget, parent=None) -> None:
            widget.grid(row=_mrow[0], column=0, padx=12, sticky="ew")
            _mrow[0] += 1

        # Coach category options.
        mlabel("Coach categories")
        chk_row = ctk.CTkFrame(models_scroll, fg_color="transparent")
        mwidget(chk_row)
        self._coach_checks: dict[str, ctk.CTkCheckBox] = {}
        for col, (fid, label) in enumerate((
            (FEATURE_COACH_IDIOM, "Idioms"),
            (FEATURE_COACH_TONE, "Tone"),
            (FEATURE_COACH_COMM, "Diplomacy"),
            (FEATURE_COACH_PITCH, "Pitch"),
        )):
            cb = ctk.CTkCheckBox(
                chk_row, text=label, width=100,
                checkbox_width=16, checkbox_height=16,
                fg_color=ACCENT, hover_color=ACCENT_HOVER,
                font=ctk.CTkFont("Segoe UI", 11),
            )
            cb.grid(row=0, column=col, padx=(0, 8), sticky="w")
            if getattr(self._ai_config, fid, True):
                cb.select()
            self._coach_checks[fid] = cb

        # Provider selection.
        mlabel("AI provider")
        from ai_providers import PROVIDER_PRESETS
        self._provider_config = ProviderConfig.from_dict(
            self.settings.get("provider_config", {}))
        preset_labels = [p.label for p in PROVIDER_PRESETS.values()]
        preset_keys = list(PROVIDER_PRESETS.keys())
        self.provider_menu = ctk.CTkOptionMenu(
            self.models_view, height=28, width=196,
            fg_color=COLORS["field"], button_color=ACCENT,
            button_hover_color=ACCENT_HOVER,
            font=ctk.CTkFont("Segoe UI", 11),
            values=preset_labels,
            command=self._on_provider_change,
        )
        current_preset = PROVIDER_PRESETS.get(
            self._provider_config.preset, PROVIDER_PRESETS["groq"])
        self.provider_menu.set(current_preset.label)
        mwidget(self.provider_menu)

        # Base URL (for custom preset).
        self.provider_url_entry = ctk.CTkEntry(
            self.models_view, height=28, fg_color=COLORS["field"],
            border_color=COLORS["border"], bg_color="transparent",
            placeholder_text="Base URL (custom only)",
            font=ctk.CTkFont("Segoe UI", 11),
        )
        self.provider_url_entry.insert(
            0, self._provider_config.base_url or current_preset.base_url)
        if self._provider_config.preset != "custom":
            self.provider_url_entry.configure(state="disabled")
        mwidget(self.provider_url_entry)

        # API key entry: per-preset, masked. Stored in the app settings file;
        # an empty field falls back to the preset's env var (.env).
        self.provider_key_entry = ctk.CTkEntry(
            self.models_view, height=28, fg_color=COLORS["field"],
            border_color=COLORS["border"], bg_color="transparent",
            show="•", font=ctk.CTkFont("Segoe UI", 11),
            placeholder_text=f"API key (or set {current_preset.key_env_var} in .env)",
        )
        _stored_key = self._provider_config.api_keys.get(
            self._provider_config.preset, "")
        if _stored_key:
            self.provider_key_entry.insert(0, _stored_key)
        self.provider_key_entry.bind(
            "<KeyRelease>", lambda _e: self._refresh_provider_key_hint())
        mwidget(self.provider_key_entry)

        # API key status hint (never shows the key itself).
        self._provider_key_hint = ctk.CTkLabel(
            self.models_view, text="", anchor="w",
            text_color=COLORS["text_dim"],
            font=ctk.CTkFont("Segoe UI", 10),
        )
        self._provider_key_hint.grid(
            row=_mrow[0], column=0, padx=12, pady=(2, 0), sticky="ew")
        _mrow[0] += 1
        self._refresh_provider_key_hint()

        # Model selection.
        mlabel("Model")
        self.model_entry = ctk.CTkEntry(
            models_scroll, height=28, fg_color=COLORS["field"],
            border_color=COLORS["border"], bg_color="transparent",
            font=ctk.CTkFont("Segoe UI", 11),
        )
        self.model_entry.insert(0, self._ai_config.model)
        mwidget(self.model_entry)

        temp_row = ctk.CTkFrame(models_scroll, fg_color="transparent")
        mwidget(temp_row)
        temp_row.grid_columnconfigure(0, weight=1)
        self.temp_val = ctk.CTkLabel(
            temp_row, text=f"Temperature  {self._ai_config.temperature:.2f}",
            text_color=COLORS["text_dim"], font=ctk.CTkFont("Segoe UI", 11),
        )
        self.temp_val.grid(row=0, column=0, sticky="w")
        self.temp_slider = ctk.CTkSlider(
            temp_row, from_=0.0, to=1.5, number_of_steps=15, width=196,
            progress_color=ACCENT, button_color=ACCENT,
            command=self._on_temp_slide,
        )
        self.temp_slider.set(self._ai_config.temperature)
        self.temp_slider.grid(row=1, column=0, pady=(4, 0), sticky="ew")

        # System prompt editor (per feature).
        mlabel("System prompt")
        self.prompt_feature = ctk.CTkOptionMenu(
            models_scroll, height=28, width=196,
            fg_color=COLORS["field"], button_color=ACCENT,
            button_hover_color=ACCENT_HOVER,
            font=ctk.CTkFont("Segoe UI", 11),
            values=[lbl for _, lbl in self._PROMPT_FEATURES],
            command=self._on_prompt_feature,
        )
        mwidget(self.prompt_feature)
        self.prompt_box = ctk.CTkTextbox(
            models_scroll, height=90, fg_color=COLORS["field"],
            border_color=COLORS["border"], border_width=1,
            font=ctk.CTkFont("Segoe UI", 11),
        )
        mwidget(self.prompt_box)

        btn_row = ctk.CTkFrame(models_scroll, fg_color="transparent")
        mwidget(btn_row)
        ctk.CTkButton(
            btn_row, text="Reset default", width=100, height=26,
            fg_color=COLORS["field"], hover_color=COLORS["field_hover"],
            font=ctk.CTkFont("Segoe UI", 10),
            command=self._on_prompt_reset,
        ).pack(side="left", padx=(0, 6))
        ctk.CTkButton(
            btn_row, text="Apply prompt", width=100, height=26,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            font=ctk.CTkFont("Segoe UI", 10),
            command=self._on_prompt_apply,
        ).pack(side="left")

        self._models_status = ctk.CTkLabel(
            models_scroll, text="", anchor="w",
            text_color=GREEN, font=ctk.CTkFont("Segoe UI", 10),
        )
        ctk.CTkLabel(models_scroll, text="").grid(
            row=_mrow[0], column=0, pady=(2, 0))
        _mrow[0] += 1
        self._models_status.grid(row=_mrow[0], column=0, padx=12, sticky="ew")

        ctk.CTkButton(
            models_scroll, text="Save configuration", width=150, height=28,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            font=ctk.CTkFont("Segoe UI Semibold", 11),
            command=self._on_models_save,
        ).grid(row=_mrow[0] + 1, column=0, padx=12, pady=(4, 8), sticky="w")

        # Load the currently selected feature's prompt into the editor.
        self._on_prompt_feature(self.prompt_feature.get())

        # Show only the selected view; start collapsed (header only).
        self._show_ai_view("Coach")
        self._apply_workspace_state(initial=True)

    def _set_meeting_text(self, content: str) -> None:
        """Meeting view is a read-only Text widget; safe against Tcl errors."""
        try:
            self.meeting_text.configure(state="normal")
            self.meeting_text.delete("1.0", "end")
            self.meeting_text.insert("1.0", content)
            self.meeting_text.configure(state="disabled")
        except Exception:  # noqa: BLE001 - widget may be destroyed (TclError)
            logger.debug("Meeting text update skipped: %s", sys.exc_info()[1])

    # ------------------------------------------------------------------ #
    # Assistant view (Stage A): grounded conversation Q&A
    # ------------------------------------------------------------------ #
    def _set_assistant_history(self, content: str, tag: str = "a") -> None:
        """Replace the assistant history text (safe against Tcl errors)."""
        try:
            self.assistant_history.configure(state="normal")
            self.assistant_history.delete("1.0", "end")
            self.assistant_history.insert("1.0", content, tag)
            self.assistant_history.configure(state="disabled")
            self.assistant_history.see("end")
        except Exception:  # noqa: BLE001 - widget may be destroyed (TclError)
            logger.debug("Assistant history update skipped: %s",
                         sys.exc_info()[1])

    def _assistant_append(self, text: str, tag: str) -> None:
        """Append one line to the assistant conversation view."""
        try:
            self.assistant_history.configure(state="normal")
            self.assistant_history.insert("end", text + "\n", tag)
            self.assistant_history.configure(state="disabled")
            self.assistant_history.see("end")
        except Exception:  # noqa: BLE001 - widget may be destroyed (TclError)
            logger.debug("Assistant append skipped: %s", sys.exc_info()[1])

    def _on_assistant_send(self) -> None:
        """Send the typed question to the grounded assistant (Tk thread)."""
        if self.integration is None or getattr(self, "_stopping", False):
            return
        try:
            question = self.assistant_input.get().strip()
        except Exception:  # noqa: BLE001
            return
        if not question:
            return
        if self.integration.is_assistant_busy():
            self._assistant_append(
                "[busy] The assistant is still answering the previous "
                "question.", "err")
            return
        try:
            self.assistant_input.delete(0, "end")
        except Exception:  # noqa: BLE001
            pass
        self._assistant_append(f"You: {question}", "q")
        handle = self.integration.ask_assistant(question)
        if handle is None or handle.status != AIRequestStatus.PENDING:
            reason = (handle.error if handle is not None else "unavailable") or ""
            friendly = {
                "ai_service_disabled": "AI coaching is off — enable it in "
                                       "the sidebar to use the assistant.",
                "service_closed": "The assistant service is shutting down.",
                "request_dropped_queue_full": "The assistant is busy — try "
                                              "again in a moment.",
                "empty_question": "Type a question first.",
            }.get(reason, f"Assistant unavailable ({reason}).")
            self._assistant_append(friendly, "err")

    def _on_assistant_clear(self) -> None:
        """Start a new assistant conversation (history cleared both sides)."""
        if self.integration is not None:
            try:
                self.integration.clear_assistant_history()
            except Exception:  # noqa: BLE001
                pass
        self._set_assistant_history(
            "New conversation. Ask about the current discussion.", tag="hint")

    # ------------------------------------------------------------------ #
    # Models tab (model config + coach options + system prompts)
    # ------------------------------------------------------------------ #
    def _on_temp_slide(self, value: float) -> None:
        try:
            self.temp_val.configure(text=f"Temperature  {value:.2f}")
        except Exception:  # noqa: BLE001
            pass

    def _selected_feature_id(self) -> str:
        label = self.prompt_feature.get()
        for fid, lbl in self._PROMPT_FEATURES:
            if lbl == label:
                return fid
        return FEATURE_ASSISTANT

    def _feature_prompt_text(self, fid: str) -> str:
        """The effective prompt for a feature: override or default."""
        override = self._prompt_overrides.get(fid)
        if override:
            return override
        return DEFAULT_SYSTEM_PROMPTS.get(fid, "")

    def _on_prompt_feature(self, _choice: str = "") -> None:
        """Load the selected feature's effective prompt into the editor."""
        try:
            fid = self._selected_feature_id()
            self.prompt_box.delete("1.0", "end")
            self.prompt_box.insert("1.0", self._feature_prompt_text(fid))
        except Exception:  # noqa: BLE001 - widget may be destroyed (TclError)
            logger.debug("Prompt feature switch skipped: %s", sys.exc_info()[1])

    def _on_prompt_reset(self) -> None:
        """Restore the default prompt for the selected feature."""
        fid = self._selected_feature_id()
        self._prompt_overrides.pop(fid, None)
        try:
            self.prompt_box.delete("1.0", "end")
            self.prompt_box.insert("1.0", DEFAULT_SYSTEM_PROMPTS.get(fid, ""))
        except Exception:  # noqa: BLE001
            return
        if self.integration is not None:
            self.integration.update_ai_config(
                self._ai_config, self._prompt_overrides)
        self._persist()
        try:
            self._models_status.configure(text="Default prompt restored.")
        except Exception:  # noqa: BLE001
            pass

    def _on_prompt_apply(self) -> None:
        """Save the edited prompt for the selected feature (validated)."""
        fid = self._selected_feature_id()
        try:
            text = self.prompt_box.get("1.0", "end").strip()
        except Exception:  # noqa: BLE001
            return
        override = normalize_override(text)
        if override is None:
            self._prompt_overrides.pop(fid, None)
        else:
            self._prompt_overrides[fid] = override
        if self.integration is not None:
            self.integration.update_ai_config(
                self._ai_config, self._prompt_overrides)
        self._persist()
        try:
            self._models_status.configure(
                text=f"Prompt saved for {fid}.")
        except Exception:  # noqa: BLE001
            pass

    def _on_provider_change(self, label: str) -> None:
        """Switch provider preset; update URL field + key hint."""
        from ai_providers import PROVIDER_PRESETS
        for key, preset in PROVIDER_PRESETS.items():
            if preset.label == label:
                self._provider_config.preset = key
                break
        is_custom = self._provider_config.preset == "custom"
        if is_custom:
            self.provider_url_entry.configure(state="normal")
            self.provider_url_entry.delete(0, "end")
        else:
            preset = PROVIDER_PRESETS[self._provider_config.preset]
            self.provider_url_entry.delete(0, "end")
            self.provider_url_entry.insert(0, preset.base_url)
            self.provider_url_entry.configure(state="disabled")
        # Swap the key field to the new preset's stored key + env hint.
        new_preset = PROVIDER_PRESETS[self._provider_config.preset]
        self.provider_key_entry.configure(
            placeholder_text=f"API key (or set {new_preset.key_env_var} in .env)")
        self.provider_key_entry.delete(0, "end")
        _stored = self._provider_config.api_keys.get(
            self._provider_config.preset, "")
        if _stored:
            self.provider_key_entry.insert(0, _stored)
        self._provider_config.model = ""
        self.model_entry.delete(0, "end")
        self.model_entry.insert(0, self._provider_config.resolve_model())
        self._refresh_provider_key_hint()
        self._persist()

    def _refresh_provider_key_hint(self) -> None:
        """Show whether the API key is present (never the key value)."""
        from ai_providers import PROVIDER_PRESETS
        import os
        preset = PROVIDER_PRESETS.get(self._provider_config.preset)
        env_var = preset.key_env_var if preset else "GROQ_API_KEY"
        typed = ""
        entry = getattr(self, "provider_key_entry", None)
        if entry is not None:
            typed = entry.get().strip()
        if typed or self._provider_config.api_key_source() == "custom":
            self._provider_key_hint.configure(
                text=f"API key: set (entered here — overrides {env_var})",
                text_color=GREEN)
        elif os.environ.get(env_var, ""):
            self._provider_key_hint.configure(
                text=f"API key: set ({env_var} from .env)", text_color=GREEN)
        else:
            self._provider_key_hint.configure(
                text=f"API key: not set — paste it above or add {env_var} to .env",
                text_color=AMBER)

    def _on_vad(self, value: float) -> None:
        """Adjust mic sensitivity (lower = more sensitive to distant speech)."""
        for c in getattr(self.interp, "_captures", []):
            c.vad_threshold = float(value)
        self.vad_val.configure(text=f"{value:.3f}")
        self._persist()
        logger.info("GUI mic sensitivity (VAD) -> %.3f", value)

    def _on_models_save(self) -> None:
        """Read the model configuration widgets and apply + persist."""
        try:
            model = self.model_entry.get().strip() or "openai/gpt-oss-120b"
            temperature = float(self.temp_slider.get())
            max_tokens = self._ai_config.max_tokens
            cfg = AIModelSettings(
                model=model,
                temperature=max(0.0, min(1.5, temperature)),
                max_tokens=max_tokens,
                coach_idioms=bool(self._coach_checks[FEATURE_COACH_IDIOM].get()),
                coach_tone=bool(self._coach_checks[FEATURE_COACH_TONE].get()),
                coach_communication=bool(
                    self._coach_checks[FEATURE_COACH_COMM].get()),
                coach_pitch=bool(self._coach_checks[FEATURE_COACH_PITCH].get()),
            )
        except Exception:  # noqa: BLE001 - malformed input keeps last config
            try:
                self._models_status.configure(
                    text="Invalid model settings — keeping previous.",
                    text_color=RED)
            except Exception:  # noqa: BLE001
                pass
            return
        self._ai_config = cfg
        # Sync provider config from UI fields.
        from ai_providers import PROVIDER_PRESETS
        for key, preset in PROVIDER_PRESETS.items():
            if preset.label == self.provider_menu.get():
                self._provider_config.preset = key
                break
        if self._provider_config.preset == "custom":
            self._provider_config.base_url = self.provider_url_entry.get().strip()
        typed_key = self.provider_key_entry.get().strip()
        if typed_key:
            self._provider_config.api_keys[self._provider_config.preset] = typed_key
        else:
            self._provider_config.api_keys.pop(self._provider_config.preset, None)
        self._provider_config.model = cfg.model
        if self.integration is not None:
            self.integration.update_ai_config(
                cfg, self._prompt_overrides,
                provider_config=self._provider_config)
        self._persist()
        try:
            self._models_status.configure(
                text=f"Saved — model {cfg.model}, "
                     f"temperature {cfg.temperature:.2f}.",
                text_color=GREEN)
        except Exception:  # noqa: BLE001
            pass

    def _poll_assistant_reply(self) -> None:
        """Drain the latest assistant reply (Tk thread; called from _tick)."""
        if self.integration is None or getattr(self, "_stopping", False):
            return
        try:
            reply = self.integration.take_assistant_reply()
        except Exception:  # noqa: BLE001
            return
        if reply is None:
            return
        question = reply.get("question", "")
        if reply.get("ok"):
            self._assistant_append(f"Assistant: {reply.get('reply', '')}", "a")
            cited = reply.get("cited_uids") or []
            context = reply.get("context_uids") or []
            if cited:
                self._assistant_append(
                    "   evidence: transcript uid "
                    + ", ".join(str(u) for u in cited), "ev")
            elif context:
                self._assistant_append(
                    f"   grounded in {len(context)} transcript excerpts "
                    "(no specific citation)", "ev")
        else:
            err = reply.get("error") or "unknown_error"
            text = reply.get("reply") or ""
            if err == "assistant_unavailable":
                line = ("Assistant unavailable — no provider connected. "
                        "Enable AI coaching and connect a provider to use "
                        "conversation Q&A.")
            elif str(err).startswith("fabricated_evidence"):
                line = ("The assistant referenced transcript evidence that "
                        "does not exist — answer discarded.")
            elif str(err).startswith("provider_error"):
                detail = str(err).split(":", 1)[1] if ":" in str(err) else ""
                if detail == "rate_limited":
                    line = ("The AI provider is rate-limiting you — free "
                            "models are the strictest. Wait a few seconds and "
                            "try again.")
                elif detail == "invalid_api_key":
                    line = ("The provider rejected the API key — check it in "
                            "the Models tab and Save.")
                elif detail.startswith("api_key_not_set"):
                    line = ("No API key for this provider yet — paste one in "
                            "the Models tab and Save.")
                else:
                    line = ("Assistant provider error"
                            + (f" ({detail})." if detail else " — try again."))
            elif err in ("provider_timeout", "request_cancelled"):
                line = "The assistant took too long — try again."
            else:
                line = f"Assistant error ({err})." + (f" {text}" if text else "")
            self._assistant_append(line, "err")

    def _show_ai_view(self, name: str) -> None:
        """Grid exactly one AI view (Tk thread only).

        Uses grid_forget() + explicit re-grid rather than grid_remove():
        CTkScrollableFrame (Agreements/Models) doesn't reliably unmap with
        grid_remove(). All views share the same cell, so no options are lost.
        """
        views = {
            "Coach": "coach_view",
            "Assistant": "assistant_view",
            "Meeting": "meeting_view",
            "Agreements": "agreement_view",
            "Models": "models_view",
        }
        for tab, attr in views.items():
            widget = getattr(self, attr, None)
            if widget is not None:
                widget.grid_forget()
        target = views.get(name)
        if target is not None:
            widget = getattr(self, target, None)
            if widget is not None:
                widget.grid(row=0, column=0, sticky="nsew")

    def _on_ai_tab(self, choice: str) -> None:
        if choice not in self.AI_TABS:
            self.ai_tabs.set("Coach")
            return
        self._show_ai_view(choice)
        self._persist()

    def _toggle_workspace(self) -> None:
        self._ws_collapsed = not self._ws_collapsed
        self._ws_auto_collapsed = False   # manual choice wins over auto
        self._ws_user_toggled = True      # no one-touch auto behaviors after
        self._apply_workspace_state()
        self._persist()

    def _apply_workspace_state(self, initial: bool = False) -> None:
        """Show/hide the AI workspace body; the header always stays visible.

        Height is animated (≈130 ms) for a deliberate feel; ``initial=True``
        applies instantly (no animation during startup/restore).
        """
        try:
            if getattr(self, "_ws_anim_after", None):
                try:
                    self.root.after_cancel(self._ws_anim_after)
                except Exception:  # noqa: BLE001
                    pass
                self._ws_anim_after = None
            body = self.ai_body
            target = 0 if self._ws_collapsed else WORKSPACE_BODY_HEIGHT
            self.ai_toggle.configure(text="Hide" if not self._ws_collapsed else "Show")

            if initial:
                if self._ws_collapsed:
                    body.grid_remove()
                else:
                    body.grid()
                    body.configure(height=WORKSPACE_BODY_HEIGHT)
                return

            body.grid()  # gridded during both directions of the animation
            start = body.winfo_height()
            if start <= 1 or start > WORKSPACE_BODY_HEIGHT:
                start = 0 if not self._ws_collapsed else WORKSPACE_BODY_HEIGHT
            steps = 8

            def step(i: int) -> None:
                try:
                    if i > steps:
                        if self._ws_collapsed:
                            body.grid_remove()
                        else:
                            body.configure(height=WORKSPACE_BODY_HEIGHT)
                        self._ws_anim_after = None
                        return
                    h = int(start + (target - start) * i / steps)
                    body.configure(height=max(1, h))
                    self._ws_anim_after = self.root.after(16, lambda: step(i + 1))
                except Exception:  # noqa: BLE001 - widget may be destroyed
                    self._ws_anim_after = None

            step(1)
        except Exception:  # noqa: BLE001 - widget may be destroyed (TclError)
            logger.debug("Workspace toggle skipped: %s", sys.exc_info()[1])

    def _check_workspace_height(self) -> None:
        """Auto-collapse the AI workspace on short windows so transcripts
        keep priority. Also scales the sidebar width for narrow windows."""
        try:
            height = self.root.winfo_height()
            width = self.root.winfo_width()
        except Exception:  # noqa: BLE001
            return
        if height <= 1 or height == self._ws_last_height:
            return
        self._ws_last_height = height
        collapsed, auto = _workspace_auto_state(
            height, self._ws_collapsed, self._ws_auto_collapsed,
        )
        if collapsed != self._ws_collapsed:
            self._ws_collapsed = collapsed
            self._ws_auto_collapsed = auto
            self._apply_workspace_state()

        # Responsive sidebar: shrink proportionally on narrow windows.
        try:
            if width < 1100 and hasattr(self, '_sidebar_frame'):
                new_w = max(200, int(248 * width / 1180))
                if abs(self._sidebar_frame.winfo_width() - new_w) > 8:
                    self._sidebar_frame.configure(width=new_w)
        except Exception:  # noqa: BLE001
            pass

    def _maybe_auto_expand_for_results(self) -> None:
        """Expand the workspace once, on the first real result, so the
        features are discoverable — unless the user chose collapsed."""
        if self._ws_collapsed and not self._ws_user_toggled \
                and not self._ws_auto_expanded_once:
            self._ws_auto_expanded_once = True
            self._ws_collapsed = False
            self._apply_workspace_state()

    def _update_ai_counts(self) -> None:
        """Refresh the quiet status text from real data only."""
        try:
            text = _compose_counts_text(
                self.mode_config.coaching_active(),
                self._counts_coach,
                self._counts_meeting,
                self._counts_agree,
            )
            self.ai_counts.configure(text=text)
        except Exception:  # noqa: BLE001 - widget may be destroyed (TclError)
            logger.debug("AI counts update skipped: %s", sys.exc_info()[1])

    def _build_footer(self) -> None:
        bar = ctk.CTkFrame(self.root, fg_color=COLORS["panel"], corner_radius=0,
                           height=52)
        bar.grid(row=2, column=0, columnspan=2, sticky="ews")
        bar.grid_propagate(False)
        # Hairline separator above the action bar.
        ctk.CTkFrame(bar, fg_color=COLORS["border"], corner_radius=0,
                     height=1).pack(side="top", fill="x")

        def btn(text: str, cmd: str) -> None:
            b = ctk.CTkButton(
                bar, text=text, width=84, height=30,
                fg_color=COLORS["field"], hover_color=COLORS["field_hover"],
                text_color=COLORS["text"],
                font=ctk.CTkFont("Segoe UI", 12), command=getattr(self, cmd),
            )
            b.pack(side="left", padx=(12, 0), pady=10)
            return b

        btn("Copy P1", "_copy_p1")
        btn("Copy You", "_copy_you")
        btn("Save…", "_save_transcript")
        btn("Clear", "_clear_captions")

        self.pause_btn = ctk.CTkButton(
            bar, text="Pause", width=84, height=30,
            fg_color=COLORS["field"], hover_color=COLORS["field_hover"],
            text_color=COLORS["text"],
            font=ctk.CTkFont("Segoe UI", 12), command=self._toggle_pause,
        )
        self.pause_btn.pack(side="left", padx=(12, 0), pady=10)

        self.stop_btn = ctk.CTkButton(
            bar, text="Stop", width=92, height=30,
            fg_color="#7f1d1d", hover_color=RED,
            font=ctk.CTkFont("Segoe UI Semibold", 12),
            command=self._on_stop,
        )
        self.stop_btn.pack(side="right", padx=(0, 16), pady=10)

    SIDE_COLORS = {
        "p1": ("#9ecbff", "#7aa7d9"),   # Person 1 (system) — blues
        "you": ("#a3e6b8", "#79c795"),  # You (mic) — greens
    }
    SIDE_TEXT = {"p1": "#9ecbff", "you": "#a3e6b8"}
    METER_COLOR = {"p1": "#7aa7d9", "you": "#79c795"}

    # ------------------------------------------------------------------ #
    # Sidebar callbacks
    # ------------------------------------------------------------------ #
    def _current_label(self) -> str:
        code = self.interp.person1_language
        name = SUPPORTED_LANGUAGES.get(code, code)
        return f"{name} ({code})"

    def _current_model_label(self) -> str:
        model = getattr(self.interp.translator, "model", "")
        return "Fast" if model == GROQ_TRANSLATE_MODELS["fast"] else "Accurate"

    def _on_language(self, choice: str) -> None:
        code = _code_from_label(choice)
        try:
            self.interp.set_person1_language(code)
        except ValueError as exc:
            self.show_error(str(exc))
            self.lang_menu.set(self._current_label())
            return
        logger.info("GUI language -> %s", code)
        self._persist()

    def _on_model(self, choice: str) -> None:
        key = "fast" if choice == "Fast" else "accurate"
        self.interp.set_translate_model(key)
        logger.info("GUI translate model -> %s (%s)", choice, GROQ_TRANSLATE_MODELS[key])
        self._persist()

    def _on_stt(self, choice: str) -> None:
        fast = choice == "Turbo"
        self.interp.set_stt_model(fast, side="user")
        logger.info("GUI user STT -> %s", choice)
        self._persist()

    def _on_stt_provider(self, choice: str) -> None:
        """Switch STT backend (Groq Whisper or Deepgram Nova-3)."""
        backend = "deepgram" if "Deepgram" in choice else "groq"
        from transcriber import Transcriber, DeepgramTranscriber
        if backend == "deepgram":
            try:
                new = DeepgramTranscriber()
                self.interp.transcriber_p1 = new
                self.interp.transcriber_user = DeepgramTranscriber()
                self.interp.transcriber = new
                self.stt_quality_seg.configure(state="disabled")
            except RuntimeError as exc:
                self.show_error(str(exc))
                self.stt_provider_seg.set("Groq Whisper")
                return
        else:
            self.interp.transcriber_p1 = Transcriber(
                api_keys=self.interp._groq_keys, model="whisper-large-v3")
            self.interp.transcriber_user = Transcriber(
                api_keys=self.interp._groq_keys, model="whisper-large-v3")
            self.interp.transcriber = self.interp.transcriber_p1
            self.stt_quality_seg.configure(state="normal")
        self.interp._stt_backend = backend
        self._persist()
        logger.info("GUI STT provider -> %s", backend)

    def _captures_by_role(self) -> dict[str, object]:
        out: dict[str, object] = {}
        for c in getattr(self.interp, "_captures", []):
            role = getattr(c, "role", None)
            if role:
                out[role] = c
        return out

    def _on_noise(self) -> None:
        enabled = bool(self.noise_switch.get())
        caps = self._captures_by_role()
        user_cap = caps.get("user") or caps.get("p1")
        if user_cap is not None:
            user_cap.noise_cancel = enabled
        logger.info("GUI noise cancel -> %s", enabled)
        self._persist()

    def _on_silence(self, value: float) -> None:
        self.sil_val.configure(text=f"{value:.2f} s")
        self.interp._silence_timeout = float(value)
        for c in getattr(self.interp, "_captures", []):
            c.silence_timeout = float(value)
        self._persist()

    def _on_topmost(self) -> None:
        self.root.attributes("-topmost", bool(self.top_switch.get()))
        self._persist()

    # ------------------------------------------------------------------ #
    # Footer actions
    # ------------------------------------------------------------------ #
    @staticmethod
    def _pane_text(widget: tk.Text) -> str:
        return widget.get("1.0", "end").strip()

    def _copy_p1(self) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(self._pane_text(self.p1_text))
        self._toast("Person 1 captions copied")

    def _copy_you(self) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(self._pane_text(self.you_text))
        self._toast("Your captions copied")

    def _save_transcript(self) -> None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile=f"interpreter_transcript_{ts}.txt",
            filetypes=[("Text files", "*.txt")],
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("=== PERSON 1 (system / Discord) ===\n")
            fh.write(self._pane_text(self.p1_text) + "\n\n")
            fh.write("=== YOU (microphone) ===\n")
            fh.write(self._pane_text(self.you_text) + "\n")
        logger.info("Transcript saved -> %s", path)
        self._toast(f"Saved: {os.path.basename(path)}")

    def _clear_captions(self) -> None:
        for widget in (self.p1_text, self.you_text):
            widget.configure(state="normal")
            widget.delete("1.0", "end")
            widget.configure(state="disabled")
        for _uid, (widget, mark) in list(self._pending.items()):
            try:
                widget.mark_unset(mark)
            except Exception:  # noqa: BLE001
                pass
        self._pending.clear()
        self._block_starts = {"p1": [], "you": []}  # aged boundaries reset
        self._line_count = 0
        for side in ("p1", "you"):
            self._stick[side] = True
            self.pills[side].place_forget()
        self._show_hints()
        self._toast("Captions cleared")

    def _toggle_pause(self) -> None:
        caps = getattr(self.interp, "_captures", [])
        if not self._paused:
            self._prev_vad = {id(c): c.vad_threshold for c in caps}
            for c in caps:  # VAD never triggers -> utterances stop flushing
                c.vad_threshold = 999.0
            self._paused = True
            self.pause_btn.configure(text="Resume", fg_color="#3a5a34",
                                     hover_color="#4a7040")
            logger.info("GUI paused (VAD muted)")
        else:
            for c in caps:
                c.vad_threshold = self._prev_vad.get(id(c), c.vad_threshold)
            self._prev_vad.clear()
            self._paused = False
            self.pause_btn.configure(text="Pause", fg_color=COLORS["field"],
                                     hover_color=COLORS["field_hover"])
            logger.info("GUI resumed")

    # ------------------------------------------------------------------ #
    # UX helpers (hints, autoscroll, font, toasts, persistence)
    # ------------------------------------------------------------------ #
    def _show_hints(self) -> None:
        for widget in (self.p1_text, self.you_text):
            if not widget.get("1.0", "end").strip():
                widget.configure(state="normal")
                widget.insert("end", "waiting for speech…\n", "hint")
                widget.configure(state="disabled")

    def _drop_hint(self, widget: tk.Text) -> None:
        if widget.get("1.0", "3.0").strip().startswith("waiting for speech"):
            widget.delete("1.0", "2.0")

    def _jump_to_bottom(self, side: str) -> None:
        self._stick[side] = True
        widget = self.p1_text if side == "p1" else self.you_text
        widget.see("end")
        self.pills[side].place_forget()

    def _after_insert(self, side: str, widget: tk.Text) -> None:
        """Smart auto-scroll: follow the tail unless the user scrolled up."""
        at_bottom = widget.yview()[1] >= 0.999
        if at_bottom:
            self._stick[side] = True
        if self._stick[side]:
            widget.see("end")
            self.pills[side].place_forget()
        else:
            self.pills[side].place(in_=widget, relx=1.0, rely=1.0,
                                   anchor="se", x=-8, y=-8)

    def _change_font(self, delta: int) -> None:
        self._font_size = max(9, min(20, self._font_size + delta))
        for widget in (self.p1_text, self.you_text):
            widget.configure(font=("Segoe UI", self._font_size))
            widget.tag_configure(
                "trans", font=("Segoe UI", max(9, self._font_size - 1), "bold"))
            widget.tag_configure(
                "pending", font=("Segoe UI", max(8, self._font_size - 2), "italic"))
            widget.tag_configure("aged", font=("Segoe UI", self._font_size))
        self._persist()

    def _on_timestamps(self) -> None:
        self._timestamps = bool(self.ts_switch.get())
        self._persist()

    def _on_detect(self) -> None:
        auto = bool(self.detect_switch.get())
        self.interp._force_language = not auto
        logger.info("GUI STT language mode -> %s",
                    "auto-detect" if auto else "locked to dropdown")
        self._persist()

    def _load_voices(self) -> None:
        """Fill the voice dropdown from the ElevenLabs account (background)."""
        def work():
            voices = self.interp.list_voices()
            self._ui_call(lambda: self._populate_voices(voices))

        threading.Thread(target=work, daemon=True).start()

    def _populate_voices(self, voices: list[dict]) -> None:
        self._voices = voices
        if not voices:
            self.voice_menu.configure(values=["(no voices found)"])
            self.voice_menu.set("(no voices found)")
            return
        current = getattr(self.interp, "_voice_id", "") or ""
        labels = []
        for v in voices:
            short = v["id"][:6]
            labels.append(f"{v['name']} ({short})")
            self._voice_ids[f"{v['name']} ({short})"] = v["id"]
        self.voice_menu.configure(values=labels)
        current_label = next((lb for lb, vid in self._voice_ids.items()
                              if vid == current), labels[0])
        self.voice_menu.set(current_label)

    def _on_voice(self, label: str) -> None:
        vid = self._voice_ids.get(label)
        if not vid:
            return
        self.interp.set_voice(vid)
        self._toast(f"Voice -> {label.split(' (')[0]}")
        self._persist()

    def _on_monitor(self) -> None:
        enabled = bool(self.monitor_switch.get())
        self.interp.set_monitor_tts(enabled)
        self._toast("TTS monitor " + ("on — you'll hear the voice" if enabled else "off"))
        self._persist()

    def _on_consent(self) -> None:
        self.interp._consent_chime = bool(self.consent_switch.get())
        logger.info("GUI consent notice -> %s", self.interp._consent_chime)
        self._persist()

    def _on_toggle_sidebar(self) -> None:
        """Collapse/expand the settings sidebar entirely (Tk thread)."""
        self._sidebar_visible = not getattr(self, "_sidebar_visible", True)
        try:
            if self._sidebar_visible:
                self.root.grid_columnconfigure(0, weight=0)
                # Re-grid the sidebar at its stored position.
                self._sidebar_frame.grid(row=1, column=0, sticky="nsw")
                self.sidebar_toggle.configure(text="☰")
            else:
                # Collapse: remove from grid so the transcripts reclaim the space.
                self._sidebar_frame.grid_remove()
                self.sidebar_toggle.configure(text="☰")
        except Exception:  # noqa: BLE001
            pass
        self._persist()

    def _on_toggle_advanced(self) -> None:
        """Show/hide the advanced settings group (persistence preserved)."""
        self.adv_open = not self.adv_open
        try:
            if self.adv_open:
                self.adv_frame.grid()
            else:
                self.adv_frame.grid_remove()
        except Exception:  # noqa: BLE001 - widget may be destroyed (TclError)
            return
        self.adv_toggle.configure(
            text=("Hide advanced settings" if self.adv_open
                  else "Show advanced settings"))
        self._persist()

    def _on_tts_style(self, choice: str) -> None:
        # Runtime-safe: the TTS worker reads _tts_mode per job.
        self.interp._tts_mode = ("streaming" if choice == "Fast"
                                 else "utterance")
        self._persist()
        logger.info("GUI TTS style -> %s (%s)", choice, self.interp._tts_mode)

    def _on_echo_guard(self) -> None:
        # Runtime-safe: the capture loop reads _mic_echo_guard per capture.
        self.interp._mic_echo_guard = bool(self.echo_switch.get())
        self._persist()
        logger.info("GUI mic echo guard -> %s", self.interp._mic_echo_guard)

    def _on_mode(self, choice: str) -> None:
        mode = CommunicationMode.normalize(choice)
        self.mode_config.mode = mode
        if self.integration is not None:
            self.integration.update_mode_config(self.mode_config)
        self._update_ai_counts()   # keep the workspace status in sync
        self._persist()
        logger.info("GUI communication mode -> %s", mode)

    def _on_coaching(self) -> None:
        self.mode_config.coaching.enabled = bool(self.coaching_switch.get())
        if self.integration is not None:
            self.integration.update_mode_config(self.mode_config)
        # Immediate visual feedback: the workspace status must reflect the
        # toggle the user just flipped (it only otherwise re-renders when
        # new AI data arrives).
        self._update_ai_counts()
        self._refresh_coach_empty_state()
        if self.mode_config.coaching.enabled and self._ws_collapsed:
            # The user just asked for coaching — reveal the workspace so the
            # state change is visible where the suggestions will appear.
            self._ws_collapsed = False
            self._ws_user_toggled = True
            self._apply_workspace_state()
        self._persist()
        logger.info("GUI coaching -> %s", self.mode_config.coaching.enabled)

    # ------------------------------------------------------------------ #
    # ClearCommit result polling and display
    # ------------------------------------------------------------------ #
    def _on_clearcommit_results(self) -> None:
        """Runs on the Tk main thread (via _ui_q). Drains results safely."""
        if getattr(self, "_stopping", False):
            return  # window closing: never touch widgets after destroy
        self._poll_clearcommit_results()

    def _poll_clearcommit_results(self) -> None:
        """Drain coaching suggestions and meeting records from the integration."""
        if self.integration is None:
            return
        if getattr(self, "_stopping", False):
            return  # Fix 5: no result processing or widget access after stop
        # Coaching suggestions.
        for result in self.integration.take_coaching_results():
            self._show_coaching_suggestion(result)
        # Meeting intelligence.
        meeting = self.integration.take_meeting_record()
        if meeting is not None:
            self._show_meeting_intelligence(meeting)
        # Assistant reply (Stage A).
        self._poll_assistant_reply()
        # Agreement ledger (Stage 7).
        self._refresh_agreement_panel()

    def _show_coaching_suggestion(self, result: CoachingUIResult) -> None:
        """Display a private coaching suggestion in the Coach view."""
        try:
            self._current_suggestion = result
            kind_label = {
                "idiom_explanation": "Idiom",
                "tone_suggestion": "Tone",
                "communication_coach": "Communication",
                "pitch_coach": "Pitch",
            }.get(result.kind, result.kind)
            text = (
                f"{kind_label}: “{result.original}” → “{result.suggestion}”\n"
                f"  {result.reason}"
            )
            self.suggestion_label.configure(
                text=text, text_color=GREEN,
                anchor="nw", justify="left")
            self.suggestion_dismiss.grid()
            self._counts_coach = self._counts_coach + 1 if \
                self._current_suggestion else 1
            self._update_ai_counts()
            # Discoverability: reveal the workspace once on first real data.
            self._maybe_auto_expand_for_results()
        except Exception:  # noqa: BLE001 - widget may be destroyed (TclError)
            logger.debug("Coaching suggestion display skipped: %s",
                         sys.exc_info()[1])

    def _refresh_coach_empty_state(self) -> None:
        """Make the Coach empty state reflect the REAL coaching state.

        The build-time text assumes coaching is off; when settings restore
        coaching ON (or the user flips the toggle), the empty state must say
        so — never tell the user to enable something that is already on.
        """
        if getattr(self, "_current_suggestion", None) is not None:
            return  # a suggestion is displayed; empty state not in use
        try:
            if self.mode_config.coaching_active():
                self.suggestion_label.configure(
                    text="Coaching is on\n\nSuggestions appear here when the "
                         "conversation contains something worth coaching on.",
                    text_color=COLORS["text_dim"],
                )
            else:
                self.suggestion_label.configure(
                    text="Coaching is off\n\nEnable AI coaching in the "
                         "sidebar (Communication section) to receive private "
                         "suggestions here.",
                    text_color=COLORS["text_dim"],
                )
        except Exception:  # noqa: BLE001 - widget may be destroyed (TclError)
            logger.debug("Coach empty-state refresh skipped: %s",
                         sys.exc_info()[1])

    def _dismiss_suggestion(self) -> None:
        try:
            self._current_suggestion = None
            self.suggestion_label.configure(justify="center", anchor="center")
            self.suggestion_dismiss.grid_remove()
            self._counts_coach = 0
            self._update_ai_counts()
            self._refresh_coach_empty_state()
        except Exception:  # noqa: BLE001 - widget may be destroyed (TclError)
            logger.debug("Suggestion dismiss skipped: %s", sys.exc_info()[1])

    def _show_meeting_intelligence(self, meeting: MeetingUIResult) -> None:
        """Display the meeting intelligence record."""
        lines = []
        if meeting.topics:
            lines.append(f"Topics: {', '.join(meeting.topics)}")
        for d in meeting.decisions:
            status = d.get("status", "?")
            lines.append(
                f"Decision [{status}]: {d.get('topic','')} = "
                f"{d.get('proposed_value','')}"
                + (f" -> {d.get('decided_value')}" if d.get('decided_value') else "")
            )
        for a in meeting.action_items:
            owner = a.get("owner") or "unassigned"
            deadline = a.get("deadline") or "no deadline"
            lines.append(
                f"Action [{a.get('status','?')}] {a.get('description','')}"
                f" (owner={owner}, deadline={deadline})"
            )
        for q in meeting.open_questions:
            state = "answered" if q.get("answered") else "unresolved"
            lines.append(f"Question [{state}]: {q.get('question','')}")
        for c in meeting.candidate_commitments:
            lines.append(
                f"Candidate commitment [{c.get('status','?')}] "
                f"{c.get('topic','')} = {c.get('proposed_value','')}"
            )
        for e in getattr(meeting, "events", []) or []:
            nd = e.get("normalized_date") or "unresolved"
            review = e.get("review_status", "?")
            spoken = e.get("spoken_date") or "none"
            lines.append(
                f"Event [{e.get('event_type','?')}|{review}] "
                f"{e.get('title','')} (spoken: {spoken}; resolved: {nd})"
                + (f" — needs: {e.get('missing_details')}"
                   if e.get("missing_details") else "")
            )
        if not lines:
            lines.append("No meeting items detected yet.")
        try:
            self._set_meeting_text("\n".join(lines))
            # Quiet counts in the workspace bar (real data only).
            n_open = sum(1 for q in meeting.open_questions if not q.get("answered"))
            n_decided = sum(1 for d in meeting.decisions if d.get("status") == "decided")
            self._counts_meeting = (n_decided, n_open)
            self._update_ai_counts()
            # Discoverability: reveal the workspace once on first real data.
            self._maybe_auto_expand_for_results()
        except Exception:  # noqa: BLE001 - widget may be destroyed (TclError)
            logger.debug("Meeting intelligence display skipped: %s",
                         sys.exc_info()[1])

    # ------------------------------------------------------------------ #
    # Agreement ledger panel (Stage 7)
    # ------------------------------------------------------------------ #
    _AGREEMENT_STATUS_COLORS = {
        "proposed": "#f59e0b",
        "needs_clarification": "#f59e0b",
        "disputed": "#ef4444",
        "confirmed": "#34d399",
        "rejected": "#ef4444",
    }
    _MAX_AGREEMENT_ROWS = 8

    def _refresh_agreement_panel(self) -> None:
        """Rebuild agreement rows when the ledger version changed (Tk thread)."""
        if self.integration is None:
            return
        if getattr(self, "_stopping", False):
            return
        try:
            version, items = self.integration.get_agreement_snapshot()
        except Exception:  # noqa: BLE001
            return
        if version == getattr(self, "_agreement_version_seen", -1):
            return
        self._agreement_version_seen = version
        try:
            for w in self.agreement_list.winfo_children():
                w.destroy()
            active = [i for i in items
                      if i["status"] not in ("confirmed", "rejected")]
            n_confirmed = sum(1 for i in items if i["status"] == "confirmed")
            n_rejected = sum(1 for i in items if i["status"] == "rejected")
            # Quiet counts in the workspace bar (real data only).
            self._counts_agree = (len(active), n_confirmed, n_rejected) \
                if items else None
            self._update_ai_counts()
            if not items:
                hint = ctk.CTkLabel(
                    self.agreement_list,
                    text="No commitments yet\n\nProposed commitments appear "
                         "here for explicit confirmation by both "
                         "participants.",
                    anchor="center", justify="center",
                    text_color=COLORS["text_dim"],
                    font=ctk.CTkFont("Segoe UI", 12),
                )
                hint.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
                return
            for row_idx, item in enumerate(active[: self._MAX_AGREEMENT_ROWS]):
                self._build_agreement_row(item, row_idx)
            if len(active) > self._MAX_AGREEMENT_ROWS:
                more = ctk.CTkLabel(
                    self.agreement_list,
                    text=f"…and {len(active) - self._MAX_AGREEMENT_ROWS} more open items",
                    text_color=COLORS["text_dim"], font=ctk.CTkFont("Segoe UI", 10),
                )
                more.grid(row=self._MAX_AGREEMENT_ROWS, column=0, sticky="w")
            # Discoverability: reveal the workspace once on first real data.
            self._maybe_auto_expand_for_results()
        except Exception:  # noqa: BLE001 - widget may be destroyed (TclError)
            logger.debug("Agreement panel refresh skipped: %s",
                         sys.exc_info()[1])

    def _build_agreement_row(self, item: dict, row_idx: int) -> None:
        """One ledger item: typed hierarchy — proposal, quiet metadata,
        explicit confirmation controls."""
        row = ctk.CTkFrame(
            self.agreement_list, fg_color=COLORS["panel_alt"], corner_radius=6,
        )
        row.grid(row=row_idx, column=0, sticky="ew", pady=3)
        row.grid_columnconfigure(0, weight=1)

        status = item["status"]
        confirmed_by = ", ".join(item["confirmed_by"]) or "none yet"
        uids = ", ".join(str(u) for u in item["evidence_uids"])
        evidence_note = f"transcript evidence uid {uids}" if uids \
            else "no transcript evidence"
        info = ctk.CTkLabel(
            row,
            text=f"{item['topic']}: {item['value']}",
            anchor="w", justify="left",
            text_color=self._AGREEMENT_STATUS_COLORS.get(status, COLORS["text"]),
            font=ctk.CTkFont("Segoe UI Semibold", 11),
        )
        info.grid(row=0, column=0, padx=12, pady=(7, 0), sticky="w")
        meta = ctk.CTkLabel(
            row,
            text=(f"{status} · confirmed by: {confirmed_by} · {evidence_note}"),
            anchor="w", justify="left",
            text_color=COLORS["text_dim"],
            font=ctk.CTkFont("Segoe UI", 10),
        )
        meta.grid(row=1, column=0, padx=12, pady=(0, 7), sticky="w")

        iid = item["id"]
        btn_kwargs = dict(
            width=60, height=24, fg_color=COLORS["field"],
            hover_color=COLORS["field_hover"],
            font=ctk.CTkFont("Segoe UI", 10),
        )
        you_btn = ctk.CTkButton(
            row, text="You ✓",
            command=lambda i=iid: self._on_agree_confirm(i, "you"),
            **btn_kwargs,
        )
        you_btn.grid(row=0, column=1, padx=(6, 2), pady=6)
        p1_btn = ctk.CTkButton(
            row, text="P1 ✓",
            command=lambda i=iid: self._on_agree_confirm(i, "p1"),
            **btn_kwargs,
        )
        p1_btn.grid(row=0, column=2, padx=2, pady=6)
        reject_btn = ctk.CTkButton(
            row, text="Reject", width=60, height=24,
            fg_color="#7f1d1d", hover_color=RED,
            font=ctk.CTkFont("Segoe UI", 10),
            command=lambda i=iid: self._on_agree_reject(i, "you"),
        )
        reject_btn.grid(row=0, column=3, padx=(2, 12), pady=6, rowspan=2)

    def _on_agree_confirm(self, item_id: str, speaker: str) -> None:
        """Explicit human confirmation (Tk thread)."""
        if self.integration is None:
            return
        try:
            ok = self.integration.confirm_commitment(item_id, speaker)
        except Exception:  # noqa: BLE001
            ok = False
        if ok:
            self._toast(f"Confirmed by {speaker}")
        else:
            self._toast("Cannot confirm this item")
        self._refresh_agreement_panel()

    def _on_agree_reject(self, item_id: str, speaker: str) -> None:
        """Explicit human rejection (Tk thread)."""
        if self.integration is None:
            return
        try:
            ok = self.integration.reject_commitment(item_id, speaker)
        except Exception:  # noqa: BLE001
            ok = False
        if ok:
            self._toast(f"Rejected by {speaker}")
        else:
            self._toast("Cannot reject this item")
        self._refresh_agreement_panel()

    def _toast(self, message: str) -> None:
        if not hasattr(self, "_toast_label"):
            self._toast_label = ctk.CTkLabel(
                self.root, text="", fg_color="#1f2937", text_color="#e5e7eb",
                corner_radius=8, font=ctk.CTkFont("Segoe UI", 12),
            )
        self._toast_label.configure(text=message)
        self._toast_label.place(relx=1.0, rely=1.0, anchor="se", x=-20, y=-72)
        if self._toast_after is not None:
            self.root.after_cancel(self._toast_after)
        self._toast_after = self.root.after(1800, self._toast_label.place_forget)

    @staticmethod
    def _human_error(message: str) -> str:
        low = message.lower()
        if "quota_exceeded" in low or "character_limit" in low:
            return ("ElevenLabs is out of credits for this language — top up "
                    "at elevenlabs.io or pick a Deepgram-supported language "
                    "(en/es).")
        if "output_format_not_allowed" in low:
            return "TTS engine rejected the audio format (internal bug — please report)."
        if "model_not_found" in low or "deaccessioned" in low:
            return "The AI model was removed by the provider — update the app."
        if "api key" in low or "unauthorized" in low or "401" in low:
            return "An API key is missing or invalid — check your .env file."
        if "rate limit" in low or "429" in low:
            return "Rate limited by the provider — retrying shortly usually fixes it."
        return message

    def _collect_settings(self) -> dict:
        data = dict(self.settings)  # preserve unknown keys
        data.update({
            "person1_language": self.interp.person1_language,
            "translate_model": self.model_seg.get(),
            "stt_model": self.stt_quality_seg.get(),
            "stt_backend": getattr(self.interp, "_stt_backend", "groq"),
            "silence_timeout": float(getattr(self.interp, "_silence_timeout", 0.7)),
            "noise_cancel": bool(self.noise_switch.get()),
            "always_on_top": bool(self.top_switch.get()),
            "timestamps": self._timestamps,
            "auto_detect": not getattr(self.interp, "_force_language", True),
            "monitor_tts": bool(getattr(self.interp, "_monitor_tts", False)),
            "consent_chime": bool(getattr(self.interp, "_consent_chime", False)),
            "voice_id": getattr(self.interp, "_voice_id", None),
            "font_size": self._font_size,
            "mode_config": self.mode_config.to_dict(),
            "mic_echo_guard": bool(getattr(self.interp, "_mic_echo_guard", False)),
            "vad_threshold": float(getattr(self.interp, "_captures", [{}])[0].vad_threshold if self.interp._captures else VAD_THRESHOLD),
            "tts_mode": getattr(self.interp, "_tts_mode", "utterance"),
            "advanced_open": bool(getattr(self, "adv_open", False)),
            "ai_config": self._ai_config.to_dict(),
            "prompt_overrides": dict(self._prompt_overrides),
            "provider_config": self._provider_config.to_dict(),
            "ai_tab": self.ai_tabs.get(),
            "workspace_collapsed": bool(self._ws_collapsed),
            "sidebar_visible": getattr(self, "_sidebar_visible", True),
        })
        try:
            data["win_geom"] = self.root.geometry()
        except Exception:  # noqa: BLE001
            pass
        if self._overlay is not None:
            try:
                data["overlay_pos"] = f"+{self._overlay.winfo_x()}+{self._overlay.winfo_y()}"
            except Exception:  # noqa: BLE001
                pass
        return data

    def _persist(self) -> None:
        """Mark settings dirty; actual disk write is debounced (sliders fire
        this dozens of times per drag — writing JSON each tick janks the UI)."""
        self._settings_dirty = True
        if getattr(self, "_persist_after", None) is None:
            self._persist_after = self.root.after(700, self._flush_settings)

    def _flush_settings(self) -> None:
        self._persist_after = None
        if getattr(self, "_settings_dirty", False):
            self._settings_dirty = False
            save_settings(self._collect_settings())

    # ------------------------------------------------------------------ #
    # Polling loops (thread-safe UI updates via after)
    # ------------------------------------------------------------------ #
    def show_error(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        try:
            self.err_label.grid()
            self.err_label.configure(text=f"Error  {stamp}   {message}")
        except Exception:  # noqa: BLE001 - widget may be destroyed (TclError)
            return
        # Surface the failure in the header status pill (real error state,
        # cleared after the strip hides — never fabricated).
        self._error_until = time.time() + 8.0
        if self._err_clear_after is not None:
            self.root.after_cancel(self._err_clear_after)
        def _clear():
            try:
                self.err_label.configure(text="")
                self.err_label.grid_remove()   # no phantom red band
            except Exception:  # noqa: BLE001
                pass
            self._err_clear_after = None
        self._err_clear_after = self.root.after(8000, _clear)

    # ------------------------------------------------------------------ #
    # Mini caption HUD (visible while the main window is minimized)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _workarea_bottom() -> int:
        """Bottom y of the usable desktop (above the Windows taskbar)."""
        try:
            import ctypes

            class RECT(ctypes.Structure):
                _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                            ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

            rect = RECT()
            ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0)
            return int(rect.bottom)
        except Exception:  # noqa: BLE001
            return 0

    def _ensure_overlay(self) -> None:
        if self._overlay is not None:
            return
        ov = ctk.CTkToplevel(self.root)
        ov.overrideredirect(True)
        ov.attributes("-topmost", True)
        ov.attributes("-alpha", 0.93)

        card = ctk.CTkFrame(ov, fg_color="#0c0f14", corner_radius=10,
                            border_width=1, border_color=COLORS["border"])
        card.pack(fill="both", expand=True, padx=2, pady=2)

        # Header: product tag + drag handle.
        hdr = ctk.CTkFrame(card, fg_color="transparent", height=22)
        hdr.pack(fill="x", padx=10, pady=(6, 0))
        ctk.CTkLabel(
            hdr, text="ClearCommit", anchor="w",
            font=ctk.CTkFont("Segoe UI Semibold", 10),
            text_color=COLORS["text_faint"],
        ).pack(side="left")

        # Original speech (speaker-coloured, compact).
        self.ov_orig = ctk.CTkLabel(
            card, text="", anchor="w", justify="left",
            text_color="#9ecbff", font=ctk.CTkFont("Segoe UI", 11),
            wraplength=460,
        )
        self.ov_orig.pack(fill="x", padx=12, pady=(2, 0))

        # Translated speech (bold, slightly larger, speaker-coloured).
        self.ov_trans = ctk.CTkLabel(
            card, text="listening…", anchor="w", justify="left",
            text_color="#a3e6b8",
            font=ctk.CTkFont("Segoe UI Semibold", 14),
            wraplength=460,
        )
        self.ov_trans.pack(fill="x", padx=12, pady=(2, 4))

        # Status line: real state only.
        self.ov_status = ctk.CTkLabel(
            card, text="", anchor="w",
            font=ctk.CTkFont("Segoe UI", 9),
            text_color=COLORS["text_faint"],
        )
        self.ov_status.pack(fill="x", padx=12, pady=(0, 6))

        # Default position: centred horizontally, above the taskbar.
        ov.update_idletasks()
        saved_pos = self.settings.get("overlay_pos")
        if saved_pos and str(saved_pos).startswith("+"):
            ov.geometry(str(saved_pos))
        else:
            sw = ov.winfo_screenwidth()
            bottom = self._workarea_bottom() or (ov.winfo_screenheight() - 48)
            w_px = max(ov.winfo_width(), 480)
            h_px = max(ov.winfo_height(), 110)
            x, y = (sw - w_px) // 2, max(0, bottom - h_px - 24)
            ov.geometry(f"{w_px}x{h_px}+{x}+{y}")

        # Drag to adjust.
        for widget in (ov, card, hdr, self.ov_orig, self.ov_trans, self.ov_status):
            widget.bind("<Button-1>", self._ov_drag_start)
            widget.bind("<B1-Motion>", self._ov_drag_move)
            widget.bind("<ButtonRelease-1>", self._ov_drag_end)

        self._overlay = ov
        self._update_overlay()
        ov.withdraw()

    def _show_overlay(self, show: bool) -> None:
        if show:
            if self._overlay is None:
                self._ensure_overlay()
            if self._overlay.state() == "withdrawn":
                self._overlay.deiconify()
                self._overlay.lift()
        elif self._overlay is not None and self._overlay.state() == "normal":
            self._overlay.withdraw()

    def _ov_drag_start(self, event) -> None:
        self._ov_drag = (event.x_root - self._overlay.winfo_x(),
                         event.y_root - self._overlay.winfo_y())

    def _ov_drag_move(self, event) -> None:
        if self._ov_drag is None:
            return
        x = event.x_root - self._ov_drag[0]
        y = event.y_root - self._ov_drag[1]
        self._overlay.geometry(f"+{max(0, x)}+{max(0, y)}")

    def _ov_drag_end(self, _event) -> None:
        if self._overlay is not None:
            try:
                self.settings["overlay_pos"] = (
                    f"+{self._overlay.winfo_x()}+{self._overlay.winfo_y()}"
                )
                save_settings(self._collect_settings())
            except Exception:  # noqa: BLE001
                pass

    def _update_overlay(self) -> None:
        if self._overlay is None:
            return
        orig = self._last_p1.get("orig", "")
        trans = self._last_p1.get("trans", "")
        self.ov_orig.configure(text=orig if orig else "")
        self.ov_trans.configure(
            text=trans if trans else "Listening…",
            text_color="#a3e6b8" if trans else COLORS["text_faint"],
        )
        # Status line: real interpreter state only.
        lang = self._current_label()
        live = bool(self.interp._tts_active) and not self._paused
        if self._paused:
            status = "Paused"
        elif live:
            status = "Speaking"
        elif orig:
            status = "Processing"
        else:
            status = "Listening"
        self.ov_status.configure(text=f"{lang}  ·  {status}")

    # ------------------------------------------------------------------ #
    def _ui_call(self, fn) -> None:
        """Schedule a callable on the Tk thread (safe from any worker thread)."""
        self._ui_q.put(fn)

    def _drain_ui_calls(self) -> None:
        try:
            while True:
                fn = self._ui_q.get_nowait()
                try:
                    fn()
                except Exception:  # noqa: BLE001
                    logger.exception("UI-bridge callback failed")
        except queue.Empty:
            pass

    def _update_meters(self) -> None:
        caps = self._captures_by_role()
        for side in ("p1", "you"):
            cap = caps.get(side)
            rms = float(getattr(cap, "last_rms", 0.0)) if cap is not None else 0.0
            canvas, _old = self.meters[side]
            width = max(canvas.winfo_width(), 1)
            level = min(1.0, rms / 0.06)  # speech peaks around ~0.05–0.15 RMS
            canvas.delete("bar")
            canvas.create_rectangle(
                0, 0, int(width * level), 5,
                fill=self.METER_COLOR[side], tags="bar", width=0,
            )
            # Real listening state from the capture's own VAD threshold —
            # shown only while speech is actually detected (never fabricated).
            vad = float(getattr(cap, "vad_threshold", 0.02)) if cap is not None else 0.02
            listening = rms > vad
            status = self.pane_status[side]
            new_text = "● Listening" if listening else ""
            try:
                if status.cget("text") != new_text:
                    status.configure(text=new_text,
                                     text_color=self.SIDE_TEXT[side])
            except Exception:  # noqa: BLE001 - widget may be destroyed
                pass

    def _pulse_live_dot(self) -> None:
        """Gentle pulse on the status pill while TTS is actually speaking.

        A real-state animation only: alternates the Live green brightness
        while the interpreter is talking; untouched in every other state
        (which _update_status_chip owns at ~1 Hz).
        """
        if getattr(self, "_stopping", False):
            return
        live = bool(self.interp._tts_active) and not self._paused \
            and time.time() >= getattr(self, "_error_until", 0.0)
        if not live:
            self._pulse_phase = False
            return
        self._pulse_phase = not getattr(self, "_pulse_phase", False)
        color = GREEN if self._pulse_phase else "#2fb878"
        try:
            self.live_dot.configure(text="●  Live", text_color=color)
        except Exception:  # noqa: BLE001 - widget may be destroyed
            pass

    def _update_status_chip(self) -> None:
        live = bool(self.interp._tts_active) and not self._paused
        error_active = time.time() < getattr(self, "_error_until", 0.0)
        if error_active:
            # A real error was reported by show_error() within the last 8s.
            self.live_dot.configure(text="●  Error", text_color=RED)
        elif self._paused:
            self.live_dot.configure(text="●  Paused", text_color=AMBER)
        elif live:
            self.live_dot.configure(text="●  Live", text_color=GREEN)
        else:
            self.live_dot.configure(text="●  Ready", text_color=COLORS["text_dim"])

        # One quiet info line: language context + real latency/TTS backend.
        lang = f"{self._current_label()}  →  English"
        backend = select_realtime_tts_backend(
            self.interp.person1_language, preferred=self.interp._realtime_backend
        )
        lat = self.interp.last_latency
        if lat:
            lats = list(self.interp._latencies) or [lat]
            avg = sum(lats) / len(lats)
            info = f"{lang}   ·   voice {lat:.1f}s   ·   TTS: {backend}"
        else:
            info = f"{lang}   ·   TTS: {backend}"
        if info != self._last_chip_text:      # touch Tk only on real changes
            self._last_chip_text = info
            self.info_chip.configure(text=info)

    def _refresh_captions(self) -> None:
        self._drain_captions()

    def _drain_captions(self) -> None:
        events = self.interp.pop_captions()
        for ev in events:
            kind, side, lang, text, uid = ev
            if kind == "error":
                self.show_error(self._human_error(text))
                continue
            if side == "p1" and kind in ("orig", "trans"):
                self._last_p1[kind] = text
                self._update_overlay()
            widget = self.p1_text if side == "p1" else self.you_text
            if kind == "orig":
                self._drop_hint(widget)
                self._line_count += 1
                self._lines_since_autosave += 1
                prefix = f"[{time.strftime('%H:%M')}] " if self._timestamps else ""
                mark = f"b{uid}"
                widget.configure(state="normal")
                widget.mark_set(mark, "end-1c")
                # Left gravity: the mark stays pinned at the start of the block
                # even when text is appended right after it.
                widget.mark_gravity(mark, "left")
                # Recent/older separation: remember each block's start line;
                # once more than RECENT_BLOCKS_KEPT exist, everything before
                # the oldest retained block is dimmed via the "aged" tag.
                block_start = widget.index("end-1c linestart")
                widget.insert("end", f"{prefix}[{lang.upper()}] {text}\n", "orig")
                widget.insert("end", "…\n", "pending")
                starts = self._block_starts.setdefault(side, [])
                starts.append(block_start)
                if len(starts) > RECENT_BLOCKS_KEPT:
                    # Everything before the oldest RETAINED block dims.
                    boundary = starts[len(starts) - RECENT_BLOCKS_KEPT]
                    widget.tag_remove("aged", "1.0", "end")
                    widget.tag_add("aged", "1.0", boundary)
                self._after_insert(side, widget)
                widget.configure(state="disabled")
                self._pending[uid] = (widget, mark)
            else:
                pending = self._pending.pop(uid, None)
                if pending is None:
                    widget.configure(state="normal")
                    widget.insert("end", f"[{lang.upper()}] {text}\n", "trans")
                    self._after_insert(side, widget)
                    widget.configure(state="disabled")
                    continue
                widget, mark = pending
                label = text or "(no translation)"
                was_stuck = self._stick[side] and widget.yview()[1] >= 0.999
                widget.configure(state="normal")
                # Replace the placeholder line (mark +1) exactly where it sits,
                # even if later utterances were appended meanwhile.
                widget.delete(f"{mark} +1 lines linestart", f"{mark} +1 lines lineend")
                widget.insert(f"{mark} +1 lines linestart", f"[{lang.upper()}] {label}", "trans")
                if was_stuck:
                    widget.see("end")
                    self.pills[side].place_forget()
                else:
                    self.pills[side].place(in_=widget, relx=1.0, rely=1.0,
                                           anchor="se", x=-8, y=-8)
                widget.configure(state="disabled")
                widget.mark_unset(mark)
        if self._lines_since_autosave >= 10:
            self._autosave_transcript()

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # Reliability: device watchdog + transcript autosave
    # ------------------------------------------------------------------ #
    def _watchdog_check(self) -> None:
        """Reopen the MICROPHONE if its callback stopped firing (dead device).

        IMPORTANT SAFETY RULES (a naive version crashed PortAudio):
          - Only watch the MIC (role="user"): input callbacks deliver silence
            frames continuously, so a frozen counter really means death.
            WASAPI LOOPBACK is event-driven (silent when nobody talks/plays),
            so a frozen counter there is NORMAL — never auto-restart it.
          - Require the freeze to persist across checks + a 30 s cooldown.
          - Restart from a worker thread; never block/crash the UI thread.
        """
        try:
            if not getattr(self, "_stopping", False):
                now = time.monotonic()
                caps = {getattr(c, "role", "user"): c
                        for c in list(getattr(self.interp, "_captures", []))}
                cap = caps.get("user")
                if cap is not None:
                    cid = id(cap)
                    count = int(getattr(cap, "cb_count", 0))
                    last = self._cb_seen.get(cid)
                    if last is None or count != last[0]:
                        self._cb_seen[cid] = (count, now)
                    elif (now - last[1] > 5.0
                          and now - getattr(self, "_last_restart", 0.0) > 30.0):
                        self._last_restart = now
                        self._cb_seen[cid] = (count, now)
                        logger.warning(
                            "Watchdog: mic callback frozen >5s — restarting."
                        )
                        self.show_error(
                            "Microphone stopped responding — reconnecting…"
                        )

                        def restart():
                            try:
                                cap.stop()
                                cap.start()
                                logger.info("Watchdog: mic restarted OK.")
                                self._ui_call(
                                    lambda: self.show_error("Microphone reconnected ✓")
                                )
                            except Exception as exc:  # noqa: BLE001
                                logger.error("Watchdog restart failed: %s", exc)
                                self._ui_call(
                                    lambda: self.show_error(f"Mic reconnect failed: {exc}")
                                )

                        threading.Thread(target=restart, daemon=True).start()
        except Exception:  # noqa: BLE001
            logger.exception("watchdog check failed")

    def _autosave_transcript(self, force: bool = False) -> None:
        """Rolling crash-safe transcript copy (every 10 lines and on exit)."""
        if self._line_count == 0 and not force:
            return
        if not force and self._lines_since_autosave < 10:
            return
        try:
            path = app_data_dir() / "autosave_transcript.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(f"=== autosaved {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                fh.write("=== PERSON 1 ===\n")
                fh.write(self._pane_text(self.p1_text) + "\n\n")
                fh.write("=== YOU ===\n")
                fh.write(self._pane_text(self.you_text) + "\n")
            self._lines_since_autosave = 0
        except Exception:  # noqa: BLE001
            pass

    def _on_stop(self) -> None:
        if getattr(self, "_stopping", False):
            return  # Stop button + window X can both fire; only clean up once
        self._stopping = True
        self._settings_dirty = True
        self._flush_settings()   # synchronous final save
        self._autosave_transcript(force=True)
        try:
            self.interp.stop()
        except Exception:  # noqa: BLE001
            pass
        if self._overlay is not None:
            try:
                self._overlay.destroy()
            except Exception:  # noqa: BLE001
                pass
        try:
            self.root.destroy()
        except Exception:  # noqa: BLE001
            pass

    def mainloop(self) -> None:
        # Surface UI-thread exceptions into the session log instead of dying
        # silently (Tk's default prints to stderr and can take the app down).
        def _log_ui_exc(exc_type, exc_value, tb):
            logger.error("UI thread exception", exc_info=(exc_type, exc_value, tb))

        self.root.report_callback_exception = _log_ui_exc
        # Text-tag styling, applied before the event loop takes over.
        # Translation lines are indented under their original (lmargin), which
        # replaces the old arrow prefixes with a clean two-level hierarchy.
        for side, widget in (("p1", self.p1_text), ("you", self.you_text)):
            orig, trans = self.SIDE_COLORS[side]
            widget.tag_configure("orig", foreground=orig,
                                 spacing1=2)
            widget.tag_configure("trans", foreground=trans,
                                 font=("Segoe UI", max(9, self._font_size - 1), "bold"),
                                 lmargin1=16, lmargin2=16, spacing1=1, spacing3=6)
            widget.tag_configure("pending", foreground=COLORS["text_faint"],
                                 font=("Segoe UI", max(8, self._font_size - 2), "italic"),
                                 lmargin1=16, lmargin2=16)
            widget.tag_configure("aged", foreground=AGED_FG,
                                 font=("Segoe UI", self._font_size))
            widget.tag_configure("hint", foreground=COLORS["text_faint"],
                                 font=("Segoe UI", self._font_size, "italic"))
        self.root.mainloop()


def main(argv=None) -> int:
    # Emoji console prints crash with UnicodeEncodeError when stdout is
    # redirected (non-console launcher); force UTF-8 so run() never dies.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    if not _acquire_single_instance():
        # A copy is already running: bring ITS window to the front instead of
        # nagging (covers the common double-double-click / hotkey spam case),
        # falling back to a message box when the window can't be found.
        import ctypes

        user32 = ctypes.windll.user32

        def _find_and_focus(target_title: str) -> bool:
            found = []

            @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
            def cb(hwnd, _lparam):
                buf = ctypes.create_unicode_buffer(256)
                user32.GetWindowTextW(hwnd, buf, 256)
                if buf.value == target_title:
                    found.append(hwnd)
                return True

            user32.EnumWindows(cb, None)
            if not found:
                return False
            hwnd = found[0]
            user32.ShowWindow(hwnd, 9)   # SW_RESTORE (undo minimize)
            user32.SetForegroundWindow(hwnd)
            return True

        if not _find_and_focus("ClearCommit"):
            ctypes.windll.user32.MessageBoxW(
                None,
                "Live Interpreter is already running.\n"
                "Use the existing window (check the taskbar) or close it first.",
                "Already running",
                0x30,  # MB_ICONWARNING
            )
        return 0

    _disable_console_quickedit()

    args = build_arg_parser().parse_args(argv)
    configure_root(level="DEBUG" if args.verbose else "INFO")
    if getattr(sys, "frozen", False):
        log_path = attach_session_file(app_data_dir() / "logs")
    else:
        log_path = attach_session_file(Path(__file__).parent / "logs")
    _sweep_stale_temp()
    if log_path:
        logger.info("Session log -> %s", log_path)
    logger.info("Live Interpreter v%s", APP_VERSION)
    _check_update_async()

    if args.list_mics:
        from audio_capture import list_microphones

        for idx, name in list_microphones():
            print(f"  [{idx}] {name}")
        return 0

    # Saved settings apply unless the user explicitly passed a CLI flag.
    saved = load_settings()
    # First-run wizard (packaged installs): keys/setup missing → walk through
    # it before the main window. Devs with a repo .env / existing settings
    # skip silently.
    if not saved.get("wizard_done") and not keys_present():
        logger.info("First run detected — starting setup wizard.")
        from setup_wizard import run_wizard

        run_wizard()
        saved = load_settings()
    argv_set = set(argv or [])
    if "--person1-lang" not in argv_set and saved.get("person1_language"):
        args.person1_lang = saved["person1_language"]
    if "--mic-device" not in argv_set and "--device" not in argv_set \
            and saved.get("mic_device"):
        args.mic_device = saved["mic_device"]
    if "--tts-output-device" not in argv_set and saved.get("tts_output_device"):
        args.tts_output_device = saved["tts_output_device"]
    elif "--tts-output-device" not in argv_set and not saved.get("tts_output_device"):
        # No saved device (never set up): default to the virtual cable so the
        # translated voice reaches Discord instead of the system speakers.
        try:
            from audio_capture import resolve_device

            resolve_device("CABLE Input", "output")
            args.tts_output_device = "CABLE Input"
            logger.info("No TTS device saved — defaulting to VB-Cable 'CABLE Input'.")
        except Exception:  # noqa: BLE001 - no cable installed; leave default
            pass
    if "--silence-timeout" not in argv_set and saved.get("silence_timeout"):
        args.silence_timeout = float(saved["silence_timeout"])

    try:
        settings_env = Settings.from_env()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # ClearCommit mode config and integration (additive; disabled by default).
    mode_config = ModeConfig.from_dict(saved.get("mode_config", {}))
    integration = ClearCommitIntegration(mode_config=mode_config)
    # Reconnect a configured AI provider on launch. Previously the provider
    # only connected when the Models-tab Save button was clicked, so every
    # restart silently left the AI features inert despite saved settings.
    integration.update_ai_config(
        AIModelSettings.from_dict(saved.get("ai_config", {})),
        sanitize_overrides(saved.get("prompt_overrides", {})),
        provider_config=ProviderConfig.from_dict(
            saved.get("provider_config", {})),
    )

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
        groq_api_key=settings_env.groq_api_key,
        deepgram_api_key=os.environ.get("DEEPGRAM_API_KEY"),
        realtime_backend=args.realtime_backend,
        translate_backend=args.translate_backend,
        tts_mode=args.tts_mode,
        noise_cancel=args.noise_cancel,
        silence_timeout=args.silence_timeout,
        force_language=not bool(saved.get("auto_detect", False)),
        elevenlabs_stability=args.elevenlabs_stability,
        elevenlabs_similarity=args.elevenlabs_similarity,
        monitor_tts=bool(saved.get("monitor_tts", False)),
        consent_chime=bool(saved.get("consent_chime", False)),
        on_final_utterance=integration.on_final_utterance,
    )
    # Wire per-service health + language mismatch callbacks (additive).
    interp._service_health_cb = integration.report_service_health
    interp._lang_mismatch_cb = integration.detect_language_mismatch
    # Remember working device names so plain relaunches are identical. Even
    # when no flag was passed, save what actually got used (e.g. the VB-Cable
    # default) so the next launch resolves to the exact same device.
    if args.mic_device:
        saved["mic_device"] = str(args.mic_device)
    if args.tts_output_device:
        saved["tts_output_device"] = str(args.tts_output_device)
    elif interp.tts_output_device is not None:
        # The interpreter auto-defaulted to the virtual cable — remember it.
        saved["tts_output_device"] = "CABLE Input"
    saved["person1_language"] = interp.person1_language
    save_settings(saved)

    box = ControlBox(interp, args.direction, settings=saved, integration=integration)

    worker = threading.Thread(
        target=interp.run,
        kwargs=dict(direction=args.direction),
        daemon=True,
    )
    worker.start()

    try:
        box.mainloop()
    finally:
        interp.stop()
        worker.join(timeout=2)
        integration.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())