# ClearCommit

Real-time two-way interpreter with AI-powered communication coaching and meeting intelligence.

## Quick Start (2 minutes)

1. **Double-click `Setup and Run.bat`**
   - Installs dependencies automatically
   - Creates `.env` if missing and opens it for you to paste keys
   - Launches the app

2. **Add your API keys** to `.env`:
   ```
   GROQ_API_KEY=gsk_...        ← required (translation + AI features)
   DEEPGRAM_API_KEY=...        ← required (text-to-speech)
   ELEVENLABS_API_KEY=...      ← optional (fallback TTS voice)
   ```

3. **Talk.** Person 1 speaks (Discord/system audio) → English captions. You speak (mic) → their language via TTS.

## Optional: AI Features

The Models tab (in the AI workspace) lets you connect a provider for Coach, Assistant, and Meeting Intelligence:

| Provider | API Key | Get it at |
|----------|---------|-----------|
| **Groq** | `GROQ_API_KEY` | https://console.groq.com/keys (free) |
| **OpenRouter** | `OPENROUTER_API_KEY` | https://openrouter.ai/keys |
| **Custom** | `CUSTOM_AI_API_KEY` | Any OpenAI-compatible endpoint |

Pick a provider in the Models tab, confirm the key hint is green, Save. All AI features go live.

## Audio Setup (for Discord calls)

1. Install [VB-Cable](https://vb-audio.com/Cable/) (free virtual audio device)
2. Discord → Settings → Voice & Video → Input Device → **CABLE Output**
3. ClearCommit → TTS output → **CABLE Input** (auto-detected)

## Controls

| Key | Action |
|-----|--------|
| Ctrl+Shift+C | Clear captions |
| Ctrl+Shift+S | Save transcript |
| Ctrl+Shift+P | Pause/Resume |
| Ctrl+Shift+X | Stop |

☰ button in the header toggles the settings sidebar. "Show advanced settings" reveals audio/enhancement controls.

## Running Tests

```
python -m pytest
```

(Any suite can also be run standalone, e.g. `python test_ai_service.py`.)

## Building the EXE

```
python -m PyInstaller --noconfirm LiveInterpreter.spec
```

Output: `dist\LiveInterpreter\LiveInterpreter.exe`
