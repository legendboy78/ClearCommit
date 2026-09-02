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
from stream_pipeline import StreamingInterpreter

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

ACCENT = "#4c8bf5"
GREEN = "#34d399"
RED = "#ef4444"
PANE_BG = "#101418"
METER_BG = "#20242b"


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

        with open(_settings_path(), encoding="utf-8") as fh:
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
    def __init__(self, interp: StreamingInterpreter, direction: str,
                 settings: "dict | None" = None):
        self.interp = interp
        self.direction = direction
        self.settings = settings or {}

        ctk.set_appearance_mode("dark")
        self.root = ctk.CTk()
        self.root.title("Live Interpreter")
        self.root.geometry("1160x680")
        self.root.minsize(1000, 580)
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
        bar = ctk.CTkFrame(self.root, fg_color="#191c22", corner_radius=0, height=52)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        bar.grid_propagate(False)
        bar.grid_columnconfigure(2, weight=1)

        self.live_dot = ctk.CTkLabel(
            bar, text="○ IDLE", font=ctk.CTkFont("Segoe UI Semibold", 14, "bold"),
            text_color="#6b7280",
        )
        self.live_dot.grid(row=0, column=0, padx=(16, 8), pady=12)

        ctk.CTkLabel(
            bar, text="LIVE INTERPRETER",
            font=ctk.CTkFont("Segoe UI Semibold", 15, "bold"),
        ).grid(row=0, column=1, pady=12)

        self.latency_chip = ctk.CTkLabel(
            bar, text="", font=ctk.CTkFont("Segoe UI", 12), text_color="#9ca3af"
        )
        self.latency_chip.grid(row=0, column=3, padx=16, pady=12)

    def _build_sidebar(self) -> None:
        sb = ctk.CTkFrame(self.root, fg_color="#191c22", corner_radius=0, width=232)
        sb.grid(row=1, column=0, sticky="nsw")
        sb.grid_propagate(False)
        for r in range(11):
            sb.grid_rowconfigure(r, weight=0)

        def section(text: str, row: int) -> None:
            ctk.CTkLabel(
                sb, text=text, anchor="w", text_color="#9ca3af",
                font=ctk.CTkFont("Segoe UI", 11),
            ).grid(row=row, column=0, padx=16, pady=(14, 2), sticky="ew")

        section("PERSON 1 LANGUAGE", 0)
        self.lang_menu = ctk.CTkOptionMenu(
            sb, values=_language_options(), width=200,
            fg_color="#232830", button_color=ACCENT, button_hover_color="#3a76d8",
            font=ctk.CTkFont("Segoe UI", 12),
            command=self._on_language,
        )
        self.lang_menu.set(self._current_label())
        self.lang_menu.grid(row=1, column=0, padx=16, sticky="ew")

        section("TRANSLATE MODEL", 2)
        self.model_seg = ctk.CTkSegmentedButton(
            sb, values=["Accurate", "Fast"], width=200,
            selected_color=ACCENT, selected_hover_color="#3a76d8",
            unselected_color="#232830", fg_color="#232830",
            command=self._on_model,
        )
        self.model_seg.set(self._current_model_label())
        self.model_seg.grid(row=3, column=0, padx=16, sticky="ew")

        section("YOUR SPEECH RECOGNITION", 4)
        self.stt_seg = ctk.CTkSegmentedButton(
            sb, values=["Accurate", "Turbo"], width=200,
            selected_color=ACCENT, selected_hover_color="#3a76d8",
            unselected_color="#232830", fg_color="#232830",
            command=self._on_stt,
        )
        self.stt_seg.set("Accurate")
        self.stt_seg.grid(row=5, column=0, padx=16, sticky="ew")

        section("RUNTIME KNOBS", 6)
        self.noise_switch = ctk.CTkSwitch(
            sb, text="Noise cancel (experimental)", width=200,
            progress_color=ACCENT, font=ctk.CTkFont("Segoe UI", 12),
            command=self._on_noise,
        )
        self.noise_switch.deselect()
        self.noise_switch.grid(row=7, column=0, padx=16, pady=(0, 2), sticky="w")

        sil_frame = ctk.CTkFrame(sb, fg_color="transparent")
        sil_frame.grid(row=8, column=0, padx=16, sticky="ew")
        self.sil_val = ctk.CTkLabel(
            sil_frame, text=f"{self.interp._silence_timeout:.2f} s",
            font=ctk.CTkFont("Segoe UI", 12), text_color="#9ca3af", width=48,
        )
        self.sil_val.pack(side="right")
        ctk.CTkLabel(
            sil_frame, text="Silence timeout", anchor="w",
            font=ctk.CTkFont("Segoe UI", 12),
        ).pack(side="left")
        self.sil_slider = ctk.CTkSlider(
            sb, from_=0.3, to=1.5, number_of_steps=24, width=200,
            progress_color=ACCENT, button_color=ACCENT,
            command=self._on_silence,
        )
        self.sil_slider.set(self.interp._silence_timeout)
        self.sil_slider.grid(row=9, column=0, padx=16, sticky="ew")

        self.top_switch = ctk.CTkSwitch(
            sb, text="Always on top", width=200,
            progress_color=ACCENT, font=ctk.CTkFont("Segoe UI", 12),
            command=self._on_topmost,
        )
        self.top_switch.grid(row=10, column=0, padx=16, pady=(6, 2), sticky="w")

        self.ts_switch = ctk.CTkSwitch(
            sb, text="Timestamps", width=200,
            progress_color=ACCENT, font=ctk.CTkFont("Segoe UI", 12),
            command=self._on_timestamps,
        )
        self.ts_switch.grid(row=11, column=0, padx=16, pady=(2, 2), sticky="w")
        if self._timestamps:
            self.ts_switch.select()

        # OFF (default): Whisper is locked to the selected language. ON:
        # auto-detect what's actually spoken (Telugu, ...) and translate from
        # that instead — for when Person 1 speaks a different language than
        # the dropdown says.
        self.detect_switch = ctk.CTkSwitch(
            sb, text="Auto-detect spoken language", width=200,
            progress_color=ACCENT, font=ctk.CTkFont("Segoe UI", 12),
            command=self._on_detect,
        )
        if not getattr(self.interp, "_force_language", True):
            self.detect_switch.select()
        self.detect_switch.grid(row=12, column=0, padx=16, pady=(2, 2), sticky="w")

        # Hear the translated TTS on your own speakers/headphones (like
        # Windows "Listen to this device", but only while this is on).
        self.monitor_switch = ctk.CTkSwitch(
            sb, text="Hear TTS locally", width=200,
            progress_color=ACCENT, font=ctk.CTkFont("Segoe UI", 12),
            command=self._on_monitor,
        )
        if getattr(self.interp, "_monitor_tts", False):
            self.monitor_switch.select()
        self.monitor_switch.grid(row=13, column=0, padx=16, pady=(2, 2), sticky="w")

        self.consent_switch = ctk.CTkSwitch(
            sb, text="Say 'call is translated' notice", width=200,
            progress_color=ACCENT, font=ctk.CTkFont("Segoe UI", 12),
            command=self._on_consent,
        )
        if getattr(self.interp, "_consent_chime", False):
            self.consent_switch.select()
        self.consent_switch.grid(row=14, column=0, padx=16, pady=(2, 8), sticky="w")

        fs_frame = ctk.CTkFrame(sb, fg_color="transparent")
        fs_frame.grid(row=15, column=0, padx=16, pady=(0, 16), sticky="ew")
        ctk.CTkLabel(
            fs_frame, text="Caption size", anchor="w",
            font=ctk.CTkFont("Segoe UI", 12),
        ).pack(side="left")
        minus = ctk.CTkButton(
            fs_frame, text="−", width=30, height=26,
            fg_color="#232830", hover_color="#2d333d",
            command=lambda: self._change_font(-1),
        )
        minus.pack(side="right", padx=(4, 0))
        plus = ctk.CTkButton(
            fs_frame, text="+", width=30, height=26,
            fg_color="#232830", hover_color="#2d333d",
            command=lambda: self._change_font(1),
        )
        plus.pack(side="right")

    def _build_panes(self) -> None:
        wrap = ctk.CTkFrame(self.root, fg_color="transparent")
        wrap.grid(row=1, column=1, sticky="nsew", padx=(0, 0))
        wrap.grid_columnconfigure(0, weight=1, uniform="pane")
        wrap.grid_columnconfigure(1, weight=1, uniform="pane")
        wrap.grid_rowconfigure(2, weight=1)

        self.pane_headers: dict[str, ctk.CTkLabel] = {}
        for col, (side, title) in enumerate((
            ("p1", "⬅  PERSON 1 · system / Discord"),
            ("you", "YOU · microphone  ➡"),
        )):
            head = ctk.CTkLabel(
                wrap, text=title, anchor="w",
                font=ctk.CTkFont("Segoe UI Semibold", 12, "bold"),
                text_color=self.SIDE_TEXT[side],
            )
            head.grid(row=0, column=col, padx=(12, 6) if col == 0 else (6, 12),
                      pady=(10, 2), sticky="ew")
            self.pane_headers[side] = head

        self.meters: dict[str, list] = {}
        for side, col in (("p1", 0), ("you", 1)):
            meter = tk.Canvas(
                wrap, height=5, highlightthickness=0, bg=METER_BG,
            )
            meter.grid(row=1, column=col, padx=(12, 6) if col == 0 else (6, 12),
                       pady=(0, 4), sticky="ew")
            self.meters[side] = [meter, 0]

        text_kw = dict(height=15, wrap="word", bg=PANE_BG, fg="#dddddd",
                       insertbackground="#dddddd", relief="flat",
                       padx=10, pady=8,
                       font=("Segoe UI", self._font_size))
        self.p1_text = tk.Text(wrap, **text_kw)
        self.p1_text.grid(row=2, column=0, padx=(12, 6), pady=(0, 4), sticky="nsew")
        self.you_text = tk.Text(wrap, **text_kw)
        self.you_text.grid(row=2, column=1, padx=(6, 12), pady=(0, 4), sticky="nsew")

        # "Jump to new lines" pills (visible when the user scrolled up).
        self.pills: dict[str, ctk.CTkButton] = {}
        for side, widget in (("p1", self.p1_text), ("you", self.you_text)):
            pill = ctk.CTkButton(
                wrap, text="↓ new lines", width=92, height=24,
                fg_color=ACCENT, hover_color="#3a76d8",
                font=ctk.CTkFont("Segoe UI", 10),
                command=lambda s=side: self._jump_to_bottom(s),
            )
            pill.place(in_=widget, relx=1.0, rely=1.0, anchor="se", x=-8, y=-8)
            pill.place_forget()
            self.pills[side] = pill

        self._show_hints()

        hint = ctk.CTkLabel(
            wrap, anchor="w", text_color="#6b7280",
            font=ctk.CTkFont("Segoe UI", 10),
            text=("⬆ heard from system → English captions"
                  "                    ⬇ your speech → spoken in their language"),
        )
        hint.grid(row=3, column=0, columnspan=2, padx=12, pady=(0, 6), sticky="w")

        # Red strip for pipeline errors (hidden while empty).
        self.err_label = ctk.CTkLabel(
            wrap, text="", anchor="w", justify="left",
            font=ctk.CTkFont("Segoe UI", 11), text_color=RED, fg_color="#2a1518",
        )
        self.err_label.grid(row=4, column=0, columnspan=2, padx=12, pady=(0, 8),
                            sticky="ew")

    def _build_footer(self) -> None:
        bar = ctk.CTkFrame(self.root, fg_color="#191c22", corner_radius=0, height=56)
        bar.grid(row=2, column=0, columnspan=2, sticky="ews")
        bar.grid_propagate(False)

        def btn(text: str, cmd: str, side: str) -> None:
            b = ctk.CTkButton(
                bar, text=text, width=86, height=30,
                fg_color="#232830", hover_color="#2d333d",
                font=ctk.CTkFont("Segoe UI", 12), command=getattr(self, cmd),
            )
            b.pack(side=side, padx=(16, 0) if side == "left" else (0, 16), pady=13)
            return b

        btn("Copy P1", "_copy_p1", "left")
        btn("Copy You", "_copy_you", "left")
        btn("Save…", "_save_transcript", "left")
        btn("Clear", "_clear_captions", "left")
        self.pause_btn = ctk.CTkButton(
            bar, text="⏸ Pause", width=86, height=30,
            fg_color="#232830", hover_color="#2d333d",
            font=ctk.CTkFont("Segoe UI", 12), command=self._toggle_pause,
        )
        self.pause_btn.pack(side="left", padx=(16, 0), pady=13)

        self.stop_btn = ctk.CTkButton(
            bar, text="■  STOP", width=110, height=32,
            fg_color="#7f1d1d", hover_color=RED,
            font=ctk.CTkFont("Segoe UI Semibold", 12, "bold"),
            command=self._on_stop,
        )
        self.stop_btn.pack(side="right", padx=(0, 16), pady=12)

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
            self.pause_btn.configure(text="▶ Resume", fg_color="#3a5a34",
                                     hover_color="#4a7040")
            logger.info("GUI paused (VAD muted)")
        else:
            for c in caps:
                c.vad_threshold = self._prev_vad.get(id(c), c.vad_threshold)
            self._prev_vad.clear()
            self._paused = False
            self.pause_btn.configure(text="⏸ Pause", fg_color="#232830",
                                     hover_color="#2d333d")
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

    def _on_monitor(self) -> None:
        enabled = bool(self.monitor_switch.get())
        self.interp.set_monitor_tts(enabled)
        self._toast("TTS monitor " + ("on — you'll hear the voice" if enabled else "off"))
        self._persist()

    def _on_consent(self) -> None:
        self.interp._consent_chime = bool(self.consent_switch.get())
        logger.info("GUI consent notice -> %s", self.interp._consent_chime)
        self._persist()

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
            "stt_model": self.stt_seg.get(),
            "silence_timeout": float(getattr(self.interp, "_silence_timeout", 0.7)),
            "noise_cancel": bool(self.noise_switch.get()),
            "always_on_top": bool(self.top_switch.get()),
            "timestamps": self._timestamps,
            "auto_detect": not getattr(self.interp, "_force_language", True),
            "monitor_tts": bool(getattr(self.interp, "_monitor_tts", False)),
            "consent_chime": bool(getattr(self.interp, "_consent_chime", False)),
            "font_size": self._font_size,
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
        self.err_label.configure(text=f"⚠ {stamp}  {message}")
        if self._err_clear_after is not None:
            self.root.after_cancel(self._err_clear_after)
        self._err_clear_after = self.root.after(8000, lambda: self.err_label.configure(text=""))

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
        ov.overrideredirect(True)          # borderless floating card
        ov.attributes("-topmost", True)
        ov.attributes("-alpha", 0.92)

        card = ctk.CTkFrame(ov, fg_color="#0b0d10", corner_radius=12)
        card.pack(fill="both", expand=True, padx=3, pady=3)
        self.ov_orig = ctk.CTkLabel(
            card, text="", anchor="w", justify="left",
            text_color="#7aa7d9", font=ctk.CTkFont("Segoe UI", 11),
            wraplength=460,
        )
        self.ov_orig.pack(fill="x", padx=14, pady=(8, 0))
        self.ov_trans = ctk.CTkLabel(
            card, text="listening…", anchor="w", justify="left",
            text_color="#9ecbff",
            font=ctk.CTkFont("Segoe UI Semibold", 15, "bold"),
            wraplength=460,
        )
        self.ov_trans.pack(fill="x", padx=14, pady=(2, 10))
        hint = ctk.CTkLabel(
            card, text="drag to move · restores with main window",
            text_color="#4b5563", font=ctk.CTkFont("Segoe UI", 9),
        )
        hint.pack(fill="x", padx=14, pady=(0, 6))

        # Default position: centered horizontally, a bit above the taskbar.
        # A previously saved (dragged) position always wins.
        ov.update_idletasks()
        saved_pos = self.settings.get("overlay_pos")
        if saved_pos and str(saved_pos).startswith("+"):
            ov.geometry(str(saved_pos))
        else:
            sw = ov.winfo_screenwidth()
            bottom = self._workarea_bottom() or (ov.winfo_screenheight() - 48)
            w_px = max(ov.winfo_width(), 300)
            h_px = max(ov.winfo_height(), 100)
            x, y = (sw - w_px) // 2, max(0, bottom - h_px - 24)
            ov.geometry(f"+{x}+{y}")

        # Drag to adjust.
        for widget in (ov, card, self.ov_orig, self.ov_trans):
            widget.bind("<Button-1>", self._ov_drag_start)
            widget.bind("<B1-Motion>", self._ov_drag_move)
            widget.bind("<ButtonRelease-1>", self._ov_drag_end)

        self._overlay = ov
        self._update_overlay()
        ov.withdraw()  # hidden until the main window minimizes

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
        self.ov_orig.configure(text=self._last_p1["orig"] or " ")
        self.ov_trans.configure(text=self._last_p1["trans"] or "listening…")

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

    def _update_status_chip(self) -> None:
        live = bool(self.interp._tts_active) and not self._paused
        if self._paused:
            self.live_dot.configure(text="⏸ PAUSED", text_color="#f59e0b")
        elif live:
            self.live_dot.configure(text="● LIVE", text_color=GREEN)
        else:
            self.live_dot.configure(text="● READY", text_color="#6b7280")

        backend = select_realtime_tts_backend(
            self.interp.person1_language, preferred=self.interp._realtime_backend
        )
        lat = self.interp.last_latency
        if lat:
            lats = list(self.interp._latencies) or [lat]
            avg = sum(lats) / len(lats)
            chip = (f"voice delay  last {lat:.1f}s · avg {avg:.1f}s"
                    f"  ·  {self._current_label()}  |  TTS: {backend}")
        else:
            chip = f"{self._current_label()}  |  TTS: {backend}"
        if chip != self._last_chip_text:      # touch Tk only on real changes
            self._last_chip_text = chip
            self.latency_chip.configure(text=chip)

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
                widget.insert("end", f"{prefix}[{lang.upper()}] {text}\n", "orig")
                widget.insert("end", "   └→ ⏳ …\n", "pending")
                self._after_insert(side, widget)
                widget.configure(state="disabled")
                self._pending[uid] = (widget, mark)
            else:
                pending = self._pending.pop(uid, None)
                if pending is None:
                    widget.configure(state="normal")
                    widget.insert("end", f"   └→ [{lang.upper()}] {text}\n", "trans")
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
                widget.insert(f"{mark} +1 lines linestart", f"   └→ [{lang.upper()}] {label}", "trans")
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
        for side, widget in (("p1", self.p1_text), ("you", self.you_text)):
            orig, trans = self.SIDE_COLORS[side]
            widget.tag_configure("orig", foreground=orig)
            widget.tag_configure("trans", foreground=trans,
                                 font=("Segoe UI", max(9, self._font_size - 1), "bold"))
            widget.tag_configure("pending", foreground="#5a6a75",
                                 font=("Segoe UI", max(8, self._font_size - 2), "italic"))
            widget.tag_configure("hint", foreground="#4b5563",
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
        import ctypes

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
    if "--silence-timeout" not in argv_set and saved.get("silence_timeout"):
        args.silence_timeout = float(saved["silence_timeout"])

    try:
        settings_env = Settings.from_env()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

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
    )
    # Remember working device names so plain `stream_gui.py` relaunches identically.
    if args.mic_device:
        saved["mic_device"] = str(args.mic_device)
    if args.tts_output_device:
        saved["tts_output_device"] = str(args.tts_output_device)
    saved["person1_language"] = interp.person1_language
    save_settings(saved)

    box = ControlBox(interp, args.direction, settings=saved)

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())