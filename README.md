# Voice-to-Voice Interpreter

A low-latency, streaming two-party voice interpreter:

**Person 1** (audio **from the system / a Discord call**, captured via a loopback
device) speaks their language → transcribed → translated to English (text only).
**You** (microphone) speak English → transcribed → translated to Person 1's
language → spoken into a **virtual microphone** so Discord (and Person 1) hear it.

Audio flows: loopback/system capture + mic capture → Groq STT → NVIDIA NIM
stream-translate → Deepgram / ElevenLabs TTS → virtual-mic output.

Two modes:

- **Two-way (recommended):** `--person1-device <loopback>` captures Person 1's
  system/Discord audio while your mic captures you. No manual direction switch —
  the source decides who is who. Your TTS is routed to `--tts-output-device`
  (a virtual mic like VB-Cable / BlackHole) so it reaches Discord without
  playing through speakers (which would feed back into your mic).
- **Single-mic (legacy):** one mic, toggle `--direction p1|user` by hand.

## Architecture

- `audio_capture.py` — `MicCapture`: 20ms PCM chunks from the mic, RMS voice-activity detection (VAD) that flushes a complete utterance when speech pauses.
- `transcriber.py` — Groq Whisper (`whisper-large-v3`) speech-to-text. Groq has no streaming STT, so utterances are sent once VAD detects end-of-speech.
- `nvidia_stream.py` — `NvidiaStreamTranslator`: NVIDIA NIM `nvidia/riva-translate-4b-instruct-v2`, Server-Sent-Events token streaming. **Requires full language names** (e.g. "Greek", "Italian"), not ISO codes.
- `elevenlabs_ws.py` — `ElevenLabsStreamTTS`: ElevenLabs WebSocket `stream-input` (`eleven_multilingual_v2`), true token-streaming TTS. `resolve_voice_id()` picks `ELEVENLABS_VOICE_ID` or the first account voice.
- `groq_tts.py` — Groq Orpheus (`canopylabs/orpheus-v1-english` / `-arabic-saudi`), EN/AR only, 200-char cap. **Not realtime** (request/response only). Requires one-time Orpheus terms acceptance at the Groq playground.
- `deepgram_tts.py` — **Deepgram Aura** (`Speak` WebSocket), the realtime TTS engine. True low-latency streaming; raw `linear16` PCM frames. Ships curated voices per language (English + Spanish configured; add more in `DEEPGRAM_TTS_MODELS`).
- `elevenlabs_ws.py` — ElevenLabs WebSocket `stream-input` (`eleven_multilingual_v2`), multilingual realtime fallback used when Aura has no voice for the target language.
- `config.py` — `select_tts_backend()` for batch; `select_realtime_tts_backend()` picks the streaming engine (Deepgram by default, ElevenLabs fallback for unsupported languages).
- `stream_pipeline.py` — `StreamingInterpreter`: wires capture → STT → translate → TTS with the per-speaker direction logic.
- `stream.py` — live CLI.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in your keys
```

Required keys in `.env`:

- `GROQ_API_KEY`
- `NVIDIA_KEY_1`
- `ELEVENLABS_API_KEY`
- `DEEPGRAM_API_KEY`
- `ELEVENLABS_VOICE_ID` (optional; auto-resolved to first voice if omitted)
- `TTS_BACKEND` (default `groq`), `TRANSLATE_BACKEND` (default `groq` = Llama-3.3-70B)
- `REALTIME_TTS_BACKEND` (default `deepgram`)

> Groq Orpheus TTS needs a one-time acceptance of the Orpheus terms at
> `https://console.groq.com/playground?model=canopylabs/orpheus-v1-english`
> before EN/AR audio will stream. Groq TTS is **not** realtime (request/response
> only, 200-char cap), so the live interpreter uses Deepgram Aura for realtime
> and falls back to ElevenLabs for languages Aura lacks a voice for.

## Run the live streaming interpreter

### Two-way (Person 1 on Discord / system audio)

Set up a loopback so Person 1's audio (the Discord call) is exposed as a
recording device, and a virtual microphone for your TTS output:

```bash
# List devices to find the loopback + virtual-mic indices:
python stream.py --list-mics

# Person 1 on Discord (loopback #3), you on mic #1, TTS -> virtual mic #2:
python stream_gui.py --person1-lang es \
    --person1-device 3 --mic-device 1 --tts-output-device 2
```

No `--direction` is needed — the loopback is always Person 1 and your mic is
always You. Your translated voice is sent **only** to the virtual mic, so
Discord transmits it to Person 1 while your speakers stay silent (no feedback
loop). Use headphones if you also monitor the call locally.

### Single-mic (legacy)

```bash
# Person 1 speaks Greek, you speak English:
python stream.py --person1-lang el --direction p1

# Choose a specific device / voice:
python stream.py --person1-lang es --direction p1 --device 1 --voice-id <id>
```

`--direction` selects whose audio is captured from the one mic:
- `p1` — capture Person 1; translate to English (no TTS).
- `user` — capture You; translate to Person 1's language and play TTS.

The spoken TTS uses Deepgram Aura (realtime) when a voice exists for Person 1's
language, otherwise ElevenLabs. Override with `--realtime-backend deepgram|elevenlabs`.

### With a live language control box (Tkinter)

`stream_gui.py` runs the same interpreter in a background thread and shows a
tiny window with a **language dropdown**, a live **captions** box (source text
+ translation), and a **Stop** button. Switch Person 1's language mid-call —
it applies on the **next spoken utterance** (no glitch to audio currently
playing).

```bash
python stream_gui.py --person1-lang es --direction user
```

The interpreter suppresses the mic while TTS is playing (echo guard) so the
speaker's own output isn't re-transcribed into garbage. It also drops
near-silent captures (average RMS below `MIN_SPEECH_RMS`) before they reach
Whisper, which otherwise hallucinates coherent text on quiet room tone. Tune
`MIN_SPEECH_RMS` / `VAD_THRESHOLD` in `stream_pipeline.py` if needed.

Requires a graphical display (standard on desktop Python; no extra install).

> **Direction:** `--direction p1` = Person 1 at the mic → English captions
> (no TTS). `--direction user` = You at the mic (English) → translated TTS in
> Person 1's language. A single mic can't tell the two apart automatically; for
> a true two-party call run **two-way mode** (`--person1-device <loopback>`
> for Person 1's system/Discord audio and `--mic-device` for you) — the sources
> are separate, so no manual switching is needed and your TTS is routed to a
> virtual mic via `--tts-output-device`.

## Legacy file-based pipeline (still available)

- `main.py` — CLI for file-based translate + TTS.
- `app.py` — Gradio UI.
- `pipeline.py`, `interpreter.py` — file-based translation pipeline and two-party interpreter.

## Notes / gotchas

- NVIDIA translate needs **full language names**, not ISO codes (e.g. `el` must be sent as "Greek").
- ElevenLabs WS audio frames are `{"audio":"<base64>"}` with **no `type` field**; the final frame is `{"audio":null,"isFinal":true}` and the server then closes the socket (expected).
- Groq Orpheus caps input at 200 characters per TTS call.
