# Live Interpreter — Product Guide

Real-time two-way voice interpreter for calls (built around Discord, works with
any app that takes a microphone):

- **Person 1** = the remote caller's voice coming out of your system audio
  → transcribed → shown as **English captions**.
- **You** = your microphone
  → transcribed → translated → **spoken in their language** into a virtual
  microphone, so the other side hears you in *their* language.

```
 Person 1's voice ──(Discord)──> Headset ──loopback──> STT ──> English captions (GUI / HUD)
 You speak ──> Mic ──> STT ──> LLM translate ──> TTS ──> VB-Cable ──> Discord ──> Person 1
```

## Setup

1. **Python 3.12+** on Windows.
2. `pip install -r requirements.txt`
3. Create `.env` next to the code:
   ```
   GROQ_API_KEY=gsk_...
   ELEVENLABS_API_KEY=...      # needed for non en/es TTS languages
   DEEPGRAM_API_KEY=...        # used for en/es TTS
   ELEVENLABS_VOICE_ID=...     # optional; auto-picks if omitted
   ```
4. Install **VB-Cable** (https://vb-audio.com/Cable/) once; set Discord's
   **input device = CABLE Output**.
5. Double-click **`Start Interpreter.bat`** (or run
   `python stream_gui.py --person1-lang el --mic-device "Microphone (Realtek(R) Audio)" --tts-output-device "CABLE Input"`).

Your language/model/device choices are saved to
`%APPDATA%\LiveInterpreter\settings.json` and restored automatically.

## Using it

| Control | What it does |
|---|---|
| Sidebar → Person 1 language | Switches output language from the next utterance on |
| Translate model | `Accurate` (gpt-oss-120b) vs `Fast` (gpt-oss-20b) |
| Your speech recognition | `Accurate` (large-v3) vs `Turbo` (faster, less accurate) |
| Silence timeout slider | How long a pause ends an utterance (0.3–1.5 s) |
| Always on top | Keeps window above other apps |
| Copy P1 / Copy You | Clipboard per pane · **Ctrl+Shift+C** clears |
| Save… | Export both panes to .txt · **Ctrl+Shift+S** |
| ⏸ Pause / ▶ Resume | Mute capture without ending the session · **Ctrl+Shift+P** |
| Minimize | Small caption HUD appears above the taskbar — drag to move |

Session logs land in `logs\session_*.log` (last 10 kept).

## Troubleshooting

| Symptom | Cause & fix |
|---|---|
| No captions at all, mic meter flat | Another instance is running (single-instance guard will warn), or wrong mic name in settings — delete `%APPDATA%\LiveInterpreter\settings.json` to reset. |
| "ElevenLabs is out of credits…" toast | Free tier exhausted — top up at elevenlabs.io or switch Person 1 language to `en`/`es` (Deepgram Aura). |
| Captions show gibberish like `αελ αελ αελ` | Whisper hallucinated on unclear audio; repeat-gibberish filter drops these automatically. Speak a bit closer / raise mic level. |
| Other side hears double/robotic audio | Ensure Discord input = **CABLE Output**, and only one interpreter instance runs. |
| Translations sound literal/stiff | Keep translate model on `Accurate`; the polyglot system prompt already enforces idiomatic style. |
| Everything silent after force-closing mid-call | Stale device locks — close and reopen Discord, relaunch the app. |

## Files of interest

- `stream_gui.py` — desktop app (this is what you run)
- `stream_pipeline.py` — capture/STT/translate/TTS orchestration
- `audio_capture.py`, `loopback_capture.py` — mic + WASAPI loopback capture
- `translator.py`, `transcriber.py`, `elevenlabs_ws.py`, `deepgram_tts.py` — service wrappers
- Backups: `backup_*/` folders — full snapshots taken before each major change
