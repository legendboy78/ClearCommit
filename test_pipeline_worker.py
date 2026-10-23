"""Hermetic tests for the per-side utterance worker architecture.

No audio devices, no network, no real API calls: STT and translation are
monkeypatched on the interpreter instance; only the worker threading model is
exercised (FIFO order per side, exception survival, max_utterances stop,
decoupling of capture from processing).

Run:  python test_pipeline_worker.py
"""

import os
import queue
import sys
import threading
import time

# Mirror the real entry points: emoji console prints crash on cp1252 stdout.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

# Fake credentials BEFORE importing pipeline modules (config reads env).
os.environ.setdefault("GROQ_API_KEY", "test-fake-groq")
os.environ.setdefault("DEEPGRAM_API_KEY", "test-fake-deepgram")

from stream_pipeline import StreamingInterpreter  # noqa: E402


class _FakeTranscription:
    def __init__(self, text: str, language: str = "Spanish"):
        self.text = text
        self.language = language
        self.duration = None


def _make_interp() -> StreamingInterpreter:
    interp = StreamingInterpreter("es", user_language="en")
    interp._translate_thread = interp._start_translate_loops()
    return interp


def _patch_fast(interp, texts):
    """Patch STT/translate to instant deterministic fakes."""
    calls = {"n": 0}
    lock = threading.Lock()

    def fake_transcribe(wav, *, speak=False, language=None):
        with lock:
            i = calls["n"]
            calls["n"] += 1
        return _FakeTranscription(texts[i % len(texts)])

    interp._transcribe = fake_transcribe
    interp._translate_with_retry = lambda text, *, target, source: f"[{target}] {text}"
    return calls


def _drain_captions(interp):
    return interp.pop_captions()

def _wait_translate_drain(interp, timeout_s=5):
    """Wait for the translate queue to drain (sentinel + brief wait)."""
    for tq in interp._translate_qs.values(): tq.put(None)  # sentinel
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if all(tq.qsize() == 0 for tq in interp._translate_qs.values()):
            break
        time.sleep(0.01)
    time.sleep(0.1)  # allow the last translation to complete


def test_worker_processes_in_fifo_order():
    interp = _make_interp()
    _patch_fast(interp, ["one", "two", "three"])
    t = interp._start_work_loop("user", None)
    for text in ("one", "two", "three"):
        interp._work_qs["user"].put((b"wav", "en", "es", True))
    # Give the worker time to drain, then send the stop sentinel.
    deadline = time.time() + 5
    while time.time() < deadline and interp._work_qs["user"].qsize() > 0:
        time.sleep(0.01)
    interp._work_qs["user"].put(None)
    interp._work_qs["user"].put(None)  # 2nd sentinel for 2nd STT worker
    for th in t: th.join(timeout=5)
    assert all(not th.is_alive() for th in t)
    _wait_translate_drain(interp)
    events = _drain_captions(interp)
    origs = [e[4] for e in events if e[0] == "orig"]
    assert len(origs) == 3
    # Single STT worker: FIFO order preserved.
    tts_texts = []
    while True:
        try:
            job = interp._tts_job_q.get_nowait()
        except queue.Empty:
            break
        if job is not None:
            tts_texts.append(job[2])
    assert tts_texts == ["[es] one", "[es] two", "[es] three"]
    print("PASS test_worker_processes_in_fifo_order")


def test_worker_survives_provider_exception():
    interp = _make_interp()
    state = {"n": 0, "fail_first": True}
    lock = threading.Lock()

    def flaky_transcribe(wav, *, speak=False, language=None):
        with lock:
            state["n"] += 1
            n = state["n"]
        if n == 1:
            raise RuntimeError("STT exploded")
        return _FakeTranscription(f"utt{n}")

    interp._transcribe = flaky_transcribe
    interp._translate_with_retry = lambda text, *, target, source: text
    t = interp._start_work_loop("user", None)
    for _ in range(3):
        interp._work_qs["user"].put((b"wav", "en", "es", True))
    all_events = []
    deadline = time.time() + 5
    while time.time() < deadline:
        all_events.extend(_drain_captions(interp))
        if any(e[0] == "error" for e in all_events):
            break
        time.sleep(0.01)
    assert any(e[0] == "error" for e in all_events), all_events
    # Worker must still process the remaining utterances; then stop it.
    deadline = time.time() + 5
    while time.time() < deadline:
        all_events.extend(_drain_captions(interp))
        origs = [e for e in all_events if e[0] == "orig"]
        if len(origs) >= 2:
            break
        time.sleep(0.01)
    origs = [e for e in all_events if e[0] == "orig"]
    assert len(origs) == 2, all_events  # utterances 2 and 3 succeeded
    interp._work_qs["user"].put(None)
    for th in (t if isinstance(t, list) else [t]): th.join(timeout=3)
    assert all(not th.is_alive() for th in (t if isinstance(t, list) else [t]))
    print("PASS test_worker_survives_provider_exception")


def test_slow_stt_does_not_block_queue():
    interp = _make_interp()
    gate = threading.Event()
    started = threading.Event()
    lock = threading.Lock()
    state = {"n": 0}

    def slow_transcribe(wav, *, speak=False, language=None):
        with lock:
            state["n"] += 1
            n = state["n"]
        if n == 1:
            started.set()
            gate.wait(timeout=5)
        return _FakeTranscription(f"utt{n}")

    interp._transcribe = slow_transcribe
    interp._translate_with_retry = lambda text, *, target, source: text
    t = interp._start_work_loop("user", None)

    interp._work_qs["user"].put((b"wav1", "en", "es", False))
    deadline = time.time() + 5
    while time.time() < deadline and not started.is_set():
        time.sleep(0.005)
    assert started.is_set()

    # While worker #1 is blocked in STT, a second utterance must still be
    # accepted by the queue (capture loop never stalls on STT).
    interp._work_qs["user"].put((b"wav2", "en", "es", False))
    assert interp._work_qs["user"].qsize() == 1

    gate.set()
    deadline = time.time() + 5
    while time.time() < deadline and any(th.is_alive() for th in t):
        time.sleep(0.01)
    interp._work_qs["user"].put(None)
    for th in (t if isinstance(t, list) else [t]): th.join(timeout=3)
    assert all(not th.is_alive() for th in (t if isinstance(t, list) else [t]))
    _wait_translate_drain(interp)
    events = _drain_captions(interp)
    origs = [e[4] for e in events if e[0] == "orig"]
    assert origs == sorted(origs) and len(origs) == 2
    print("PASS test_slow_stt_does_not_block_queue")


def test_max_utterances_stops_worker():
    interp = _make_interp()
    _patch_fast(interp, ["a", "b", "c", "d"])
    t = interp._start_work_loop("user", 2)
    for _ in range(4):
        interp._work_qs["user"].put((b"wav", "en", "es", False))
    for th in (t if isinstance(t, list) else [t]): th.join(timeout=5)
    assert all(not th.is_alive() for th in (t if isinstance(t, list) else [t]))
    assert interp._side_stop["user"].is_set()
    _wait_translate_drain(interp)
    events = _drain_captions(interp)
    origs = [e for e in events if e[0] == "orig"]
    assert len(origs) == 2  # exactly the budget processed
    print("PASS test_max_utterances_stops_worker")


def test_stop_event_exits_worker():
    interp = _make_interp()
    _patch_fast(interp, ["x"])
    t = interp._start_work_loop("p1", None)
    interp._work_qs["p1"].put((b"wav", "es", "en", False))
    time.sleep(0.05)
    interp.stop()  # sets _stop_ev
    for th in (t if isinstance(t, list) else [t]): th.join(timeout=3)
    assert all(not th.is_alive() for th in (t if isinstance(t, list) else [t]))
    print("PASS test_stop_event_exits_worker")


def test_echo_guard_decisions():
    """Pure echo-guard logic: P1 always guarded; mic guarded only when the
    flag is on; both pass once the cooldown expires."""
    from stream_pipeline import MIC_ECHO_COOLDOWN, POST_TTS_COOLDOWN
    now = 1000.0
    # P1: during playback and within cooldown -> drop.
    assert StreamingInterpreter._should_drop_for_echo("p1", True, now - 0.5, now) is True
    assert StreamingInterpreter._should_drop_for_echo(
        "p1", False, now - POST_TTS_COOLDOWN + 0.5, now) is True
    # P1: after cooldown -> pass.
    assert StreamingInterpreter._should_drop_for_echo(
        "p1", False, now - POST_TTS_COOLDOWN - 0.1, now) is False
    # Mic: guarded during playback and short cooldown (explicit ON).
    assert StreamingInterpreter._should_drop_for_echo("user", True, now - 0.5, now, mic_echo_guard=True) is True
    assert StreamingInterpreter._should_drop_for_echo("user", False, now - MIC_ECHO_COOLDOWN + 0.2, now, mic_echo_guard=True) is True
    assert StreamingInterpreter._should_drop_for_echo("user", False, now - MIC_ECHO_COOLDOWN - 0.1, now, mic_echo_guard=True) is False
    # Mic: default (no arg) = OFF -> never drops (barge-in mode).
    assert StreamingInterpreter._should_drop_for_echo("user", True, now - 0.5, now) is False
    # Mic: guard disabled -> never drops (barge-in mode).
    assert StreamingInterpreter._should_drop_for_echo(
        "user", True, now - 0.5, now, mic_echo_guard=False) is False
    print("PASS test_echo_guard_decisions")


if __name__ == "__main__":
    test_worker_processes_in_fifo_order()
    test_worker_survives_provider_exception()
    test_slow_stt_does_not_block_queue()
    test_max_utterances_stops_worker()
    test_stop_event_exits_worker()
    test_echo_guard_decisions()
    print("\nAll pipeline worker tests passed.")
