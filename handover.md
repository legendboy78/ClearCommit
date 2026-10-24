# ClearCommit — Engineer Handover Document

## What This Is

A Windows desktop app built on Python + CustomTkinter that does real-time two-way speech translation with AI-powered communication coaching, meeting intelligence, and an evidence-grounded agreement ledger.

The core interpreter (capture → STT → translate → TTS) is from an existing working app and has been preserved. All new features are additive layers on top of it.

---

## Architecture Overview

```
Browser/Discord audio ──→ LoopbackCapture (P1) ──→ STT Worker ──→ Translation Worker ──→ Captions
Your mic ────────────────→ MicCapture (You) ────→ STT Worker ──→ Translation Worker ──→ TTS Worker
                                                    │                                    │
                                                    └──→ ClearCommit Integration ←──────┘
                                                              │
                                              ┌───────────────┼───────────────┐
                                              ▼               ▼               ▼
                                          Social Coach   Professional Coach   Meeting Intelligence
                                              │               │               │
                                              └───────→ Agreement Ledger ←────┘
                                                              │
                                                        Grounded Assistant
```

### Pipeline Stages (all async, bounded, non-blocking)

| Stage | Thread(s) | What it does | Bounded by |
|-------|-----------|--------------|------------|
| Capture | 2 (P1 loopback + mic) | VAD, silence detection, utterance flush | 15s max utterance |
| STT | 1 per side (2 total) | Whisper API call or Deepgram Nova-3 | 2 retries, 0.3s/0.6s delays |
| Translation | 1 per side (2 total) | Groq LLM translation with streaming | 3 retries, 0.6s/1.2s/2.4s delays |
| TTS | 1 (shared) | Deepgram Aura or ElevenLabs WS | FIFO (audio is sequential) |
| AI Coaching | 1 (AIService) | Idiom/tone/diplomacy/pitch detection | queue=8, timeout=3s, circuit breaker |
| Meeting Intel | 1 (AIService) | Extract decisions/actions/questions/events | queue=4, timeout=5s, circuit breaker |
| Assistant | 1 (AIService) | Grounded Q&A over transcript | queue=2, timeout=12s, circuit breaker |

### File Map

| File | Purpose | Lines |
|------|---------|-------|
| `stream_pipeline.py` | Core interpreter: capture, STT, translate, TTS, echo guard, per-side workers | ~1258 |
| `stream_gui.py` | Full UI: header, sidebar, 5-tab AI workspace, footer, overlay HUD, all callbacks | ~3070 |
| `clearcommit_integration.py` | Glues AI features into the pipeline: evidence store, meeting intelligence, coaching dispatcher, agreement ledger, assistant, provider config | ~1050 |
| `ai_service.py` | Bounded async AI execution: queue, timeout, cancellation, circuit breaker, idempotent shutdown | ~674 |
| `ai_config.py` | Model settings (model/temperature/max_tokens), coach category toggles, default + custom system prompts | ~180 |
| `ai_providers.py` | Provider presets (OpenRouter/Groq/Custom), OpenAI-compatible chat-completions client | ~220 |
| `assistant.py` | Grounded Q&A: bounded context builder, citation validation, conversation history | ~240 |
| `social_coaching.py` | Idiom + tone detection with pre-filter, dedup, JSON validation | ~395 |
| `professional_coaching.py` | Communication + pitch coaching with pre-filter, dedup, JSON validation | ~400 |
| `meeting_intelligence.py` | Extracts decisions, action items, open questions, commitments, events with evidence validation | ~740 |
| `agreement_ledger.py` | Deterministic state machine: proposed → needs_clarification → confirmed/rejected. Two-party human confirmation required | ~300 |
| `mode_config.py` | Communication mode (Social/Professional) + coaching toggle | ~85 |
| `transcriber.py` | Groq Whisper + Deepgram Nova-3 STT backends (drop-in interchangeable) | ~190 |
| `mic_enhancement_lab.py` | Offline A/B/C/D experiment harness for speech enhancement (standalone, not imported by app) | ~280 |
| `mode_config.py` | Communication mode + coaching config | ~85 |

---

## How to Run

### For Development
```
python stream_gui.py
```
Requires `.env` with API keys (see `.env.example`).

### For Testing/Submission
Double-click `Setup and Run.bat` — installs deps, creates `.env`, launches.

### For Distribution
```
python -m PyInstaller --noconfirm LiveInterpreter.spec
```
Output: `dist\LiveInterpreter\LiveInterpreter.exe`

---

## API Keys Required

| Key | Used For | Required? |
|-----|----------|-----------|
| `GROQ_API_KEY` | Translation (always) + STT (when Groq Whisper selected) + AI features (when Groq provider selected) | Yes |
| `DEEPGRAM_API_KEY` | TTS (Aura) + STT (when Deepgram Nova-3 selected) | Yes |
| `ELEVENLABS_API_KEY` | TTS fallback for non-en/es languages | Optional |
| `OPENROUTER_API_KEY` | AI features via OpenRouter (alternative to Groq) | Optional |
| `CUSTOM_AI_API_KEY` | AI features via custom OpenAI-compatible endpoint | Optional |

---

## The 5 AI Workspace Tabs

### 1. Coach
- **Social mode**: detects idioms/slang + tone issues, suggests explanations/rephrasing
- **Professional mode**: detects unclear/aggressive/complex wording (communication coach) + pitch opportunities
- Pre-filter gates model calls (only consults LLM when hints are present)
- System prompts editable per category (Idioms / Tone / Diplomacy / Pitch)

### 2. Assistant
- Type questions about the live conversation ("What did we agree on?", "Summarize so far")
- Answers cite transcript evidence as [E<uid>]
- Fabricated evidence references are rejected (never displayed)
- Conversation history is bounded (10 exchanges)
- Grounding contract is unconditional (always present, custom prompts can't weaken it)

### 3. Meeting
- Extracts from the conversation: decisions (proposed/decided/rejected), action items (with/without owner+deadline), open questions (answered/unresolved), candidate commitments, events (meetings, deadlines, risks, blockers)
- Events have review_status: explicit / inferred / needs_review
- Normalized dates only when unambiguous (relative expressions stay unresolved)
- Conflicting statements preserved as separate items, never merged
- Runs asynchronously on its own service; stale results never overwrite newer ones

### 4. Agreements
- Candidate commitments ingested from meeting intelligence as "proposed"
- Two-party explicit human confirmation required (You ✓ + P1 ✓) to reach "confirmed"
- One "Reject" from either participant marks it "rejected"
- Non-participant speakers (e.g., "model") can never confirm
- Status color-coded: amber=proposed, green=confirmed, red=rejected

### 5. Models
- **AI provider**: OpenRouter / Groq / Custom (OpenAI-compatible endpoint)
- **Model**: text entry (e.g., `openai/gpt-oss-120b`)
- **Temperature**: slider (0.0–1.5)
- **Coach categories**: checkboxes (Idioms / Tone / Diplomacy / Pitch)
- **System prompts**: per-feature editor with Apply/Reset
- **Save configuration**: applies + persists everything

---

## Settings Persistence

All settings saved to `%APPDATA%\LiveInterpreter\settings.json`:

| Key | Type | Default | What it stores |
|-----|------|---------|----------------|
| `person1_language` | str | `"es"` | Person 1's language |
| `translate_model` | str | `"Accurate"` | Translation quality |
| `stt_model` | str | `"Accurate"` | Speech recognition quality |
| `stt_backend` | str | `"groq"` | STT provider (`groq` or `deepgram`) |
| `mic_echo_guard` | bool | `False` | Drop mic captures during TTS playback |
| `vad_threshold` | float | `0.008` | Mic sensitivity (lower = more sensitive) |
| `silence_timeout` | float | `0.7` | Seconds of silence that end an utterance |
| `tts_mode` | str | `"utterance"` | TTS style (`utterance` or `streaming`) |
| `noise_cancel` | bool | `False` | Spectral gate on mic |
| `monitor_tts` | bool | `False` | Hear TTS locally |
| `consent_chime` | bool | `False` | Play "call is translated" notice |
| `mode_config` | dict | `{"mode": "social", "coaching": {"enabled": false}}` | Communication mode |
| `ai_config` | dict | model/temperature/max_tokens/coach toggles | AI model settings |
| `prompt_overrides` | dict | `{}` | Per-feature system prompt overrides |
| `provider_config` | dict | `{"preset": "groq", ...}` | AI provider selection |
| `ai_tab` | str | `"Coach"` | Last selected AI workspace tab |
| `workspace_collapsed` | bool | `True` | AI workspace collapsed state |
| `advanced_open` | bool | `False` | Sidebar advanced settings open |
| `sidebar_visible` | bool | `True` | Sidebar visible state |
| `font_size` | int | `12` | Caption font size |
| `always_on_top` | bool | `False` | Window always on top |
| `timestamps` | bool | `False` | Show timestamps in captions |
| `auto_detect` | bool | `False` | Auto-detect spoken language |
| `voice_id` | str | auto | ElevenLabs voice ID |

---

## STT Backends

| Backend | Model | Language Detection | Quality | Config |
|---------|-------|--------------------|---------|--------|
| Groq Whisper | `whisper-large-v3` or `whisper-large-v3-turbo` | Single language (locked) or auto-detect | Accurate / Fast | Sidebar segmented: Accurate/Turbo |
| Deepgram Nova-3 | `nova-3` | Native multi-language (code-switching) | High | Sidebar segmented: Groq/Deepgram |

Both are drop-in interchangeable (same `transcribe()` → `TranscriptionResult` interface). Select in the sidebar under "Your speech recognition".

---

## Provider System

The Models tab lets you connect any OpenAI-compatible chat-completions endpoint:

| Preset | Base URL | Key env var | Default model |
|--------|----------|-------------|---------------|
| OpenRouter | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | `openai/gpt-oss-120b` |
| Groq | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` | `openai/gpt-oss-120b` |
| Custom | user-supplied | `CUSTOM_AI_API_KEY` | user-supplied |

When a provider is configured (preset + API key present — pasted in the Models tab and stored per-preset in settings.json, or via env vars; a pasted key wins):
- All 4 AI features (Social Coach, Professional Coach, Assistant, Meeting Intelligence) get the real provider
- The provider is restored automatically on app launch (main() re-applies saved ai_config/prompt_overrides/provider_config)
- The provider is used for all model calls
- Model/temperature/max_tokens from the Models tab are passed via the context mechanism

When no provider is configured:
- All AI features are inert (coaching returns None, assistant says "unavailable")
- The translator still works (it uses the existing Groq translation service, independent of the AI provider)

---

## Key Design Decisions

1. **Parallel speech processing**: STT and translation are on separate pipeline stages. STT for sentence 2 starts immediately after STT for sentence 1, while sentence 1's translation happens in parallel. TTS remains FIFO (audio is sequential).

2. **Mic echo guard OFF by default**: The PC's own TTS voice can leak into the mic. The hallucination filter catches these artifacts. The guard is available as an optional toggle but blocks all speech during TTS playback.

3. **Evidence-grounded AI**: All model output is validated deterministically. Fabricated evidence IDs are rejected. Model provenance (evidence_uid, side, original) is bound to trusted request data, never trusted from model output.

4. **Human-only confirmation**: The agreement ledger requires explicit confirmation from both participants. Model output can never confirm an item. Non-participant speakers are refused.

5. **Bounded everything**: Evidence window (40 utterances), assistant history (10 exchanges), translation queue, coaching dedup (50), meeting record dedup. Memory cannot grow unboundedly.

6. **Circuit breakers**: Each AIService tracks timeouts. After 3 consecutive timeouts, the circuit opens and rejects new work until the service is recreated. This bounds stuck-thread accumulation.

---

## Test Coverage

186 tests across 11 suites, all passing:

| Suite | Tests | What it covers |
|-------|-------|----------------|
| `test_mode_config.py` | 7 | Mode validation, coaching toggle, serialization |
| `test_agreement_ledger.py` | 21 | State machine, confirmation rules, evidence immutability, fabricated refs |
| `test_social_coaching.py` | 19 | Idiom/tone detection, pre-filter, dedup, validation, thread safety |
| `test_professional_coaching.py` | 20 | Communication/pitch detection, flags, thread safety |
| `test_meeting_intelligence.py` | 34 | All item types, evidence validation, date rules, conflicts, bounds |
| `test_assistant.py` | 12 | Context bounding, citations, fabricated refs, history bounds |
| `test_ai_config.py` | 9 | Model settings validation, clamping, prompt override flow |
| `test_clearcommit_integration.py` | 25 | End-to-end wiring, async meeting, confirmation workflow, evidence bounds |
| `test_ui_layout_logic.py` | 9 | Workspace auto-collapse, counts text |
| `test_pipeline_worker.py` | 6 | FIFO order, exception survival, slow STT, max utterances, echo guard |
| `test_ai_service.py` | 22 | Queue bounds, timeout, cancellation, circuit breaker, shutdown |

---

## Known Limitations

1. **STT is file-based** (Whisper and Deepgram pre-recorded API) — there is no streaming STT. Latency floor is ~0.5–2s per sentence regardless of pipeline optimization. True streaming would require a WebSocket connection (different API).

2. **AI features are inert without a real provider** — the Models tab stores config, but coaching/assistant/meeting produce nothing until an API key is set and Save is clicked. This is by design (no silent failures).

3. **TTS ordering with parallel STT** — with 2 STT workers per side, translation results may arrive slightly out of order. The translate worker (per side, FIFO) preserves relative order, but the STT completion order may differ from capture order. TTS follows translation order.

4. **Translation always uses Groq** — the Models tab provider selection only affects coaching/assistant/meeting intelligence, not the translation itself. Translation is a separate service with its own model and prompt.

5. **DeepFilterNet not integrated** — the lab harness exists (`mic_enhancement_lab.py`) but was never run against real samples. The spectral gate (noise cancel toggle) is the only built-in enhancement.

6. **Language handling is simple** — one configured language per side. Mixed-language speech, mid-call language changes, and dialect detection are not modeled. The auto-detect switch and language mismatch hint are the current mitigations.

7. **DPI scaling** — the app handles 125% DPI (tested) but the workspace auto-collapse thresholds are physical-pixel based. At different DPI settings, the thresholds may need adjustment.

---

## Summary of All Changes from the Original App

### Modified files
- `stream_gui.py` — complete UI redesign (header, sidebar, tabs, workspace, footer) + all ClearCommit integration
- `stream_pipeline.py` — additive: per-side STT workers, translation workers, echo guard, language mismatch callback, service health callback
- `LiveInterpreter.spec` — expanded exclusion list (pre-existing change)

### New files
- All files listed in the File Map above (except `stream_pipeline.py`)

### Unchanged files
- `config.py`, `logger.py`, `utils.py`, `voice_cloner.py`, `elevenlabs_ws.py`, `groq_tts.py`, `discord_bot.py`, `app.py`, `main.py`, `setup_wizard.py`, `audio_capture.py`, `loopback_capture.py`
- `.env` (untouched)
- `LiveInterpreter.spec` (pre-existing exclusion list change preserved)

---

## What to Do Next

1. **Test with real calls** — the best validation is a real Discord conversation
2. **Connect a provider** — add `OPENROUTER_API_KEY` or use Groq (already configured) → Models tab → Save
3. **Run the mic lab** — `python mic_enhancement_lab.py --record` then `--execute` to measure if enhancement helps
4. **Consider language improvements** — auto-detect + mismatch hint is the current mitigation; true per-speaker language modeling would be a larger change
5. **Consider NiceGUI for a future web UI** — the AI modules are UI-agnostic by design; a NiceGUI shell could import the same `ClearCommitIntegration`
