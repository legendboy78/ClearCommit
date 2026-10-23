"""Offline microphone-enhancement lab for ClearCommit (Track D).

STANDALONE EXPERIMENT HARNESS — never imported by the application.
Measures whether speech enhancement improves Whisper recognition of
distant/quiet/muffled microphone speech on THIS machine, before any
production change is considered.

Conditions (identical audio per line):
    A  raw          — the sample exactly as captured
    B  gain         — measured peak normalization only
    C  spectralgate — the app's existing spectral-gate approach (standalone
                      reimplementation; production code untouched)
    D  deepfilternet— only if the `deepfilternet` package imports cleanly

Usage:
    python mic_enhancement_lab.py --record               # capture samples
    python mic_enhancement_lab.py --dry-run              # plan only, no API calls
    python mic_enhancement_lab.py --execute              # real Whisper calls

Samples live in ./lab_samples/*.wav with a references.txt file:
    <wav filename>|<exact reference text>

Metrics per condition: WER (normalized word error rate vs reference),
empty-transcription count (dropped speech), processing seconds, and
realtime factor (processing time / audio duration). Measured values only —
nothing is invented. Whisper calls require --execute and a configured key;
each call is counted and reported.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
import wave
from pathlib import Path

LAB_DIR = Path(__file__).parent / "lab_samples"
TARGET_PEAK_DBFS = -6.0


# --------------------------------------------------------------------------- #
# WAV helpers
# --------------------------------------------------------------------------- #


def read_wav(path: Path) -> tuple[int, bytes]:
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        frames = w.readframes(w.getnframes())
    return rate, frames


def write_wav(path: Path, rate: int, frames: bytes) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)


def duration_seconds(rate: int, frames: bytes) -> float:
    return len(frames) / 2 / float(rate) if rate else 0.0


# --------------------------------------------------------------------------- #
# Conditions (pure functions on int16 mono frames)
# --------------------------------------------------------------------------- #


def cond_raw(rate: int, frames: bytes) -> bytes:
    return frames


def cond_gain(rate: int, frames: bytes) -> bytes:
    """Peak-normalize to TARGET_PEAK_DBFS (measured, bounded gain 1x-32x)."""
    import array

    samples = array.array("h")
    samples.frombytes(frames)
    if not samples:
        return frames
    peak = max(abs(s) for s in samples)
    if peak == 0:
        return frames
    target = (10 ** (TARGET_PEAK_DBFS / 20.0)) * 32767.0
    gain = max(1.0, min(32.0, target / peak))
    out = array.array("h", (max(-32768, min(32767, int(s * gain)))
                            for s in samples))
    return out.tobytes()


def cond_spectralgate(rate: int, frames: bytes) -> bytes:
    """Standalone spectral gate mirroring MicCapture._spectral_gate.

    Reimplemented here (not imported) so the experiment can never mutate
    production behaviour. Noise profile = first 2 seconds (or the whole clip
    if shorter), treated as the quiet lead-in.
    """
    import numpy as np

    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if samples.size < 1024:
        return frames
    n_fft = 512
    hop = n_fft // 2
    window = np.hanning(n_fft).astype(np.float32)
    window[0] = window[-1] = 1e-3

    noise_len = min(samples.size, 2 * rate)
    lead = samples[:noise_len]
    specs = []
    for i in range(0, max(1, noise_len - n_fft), hop):
        frame = lead[i:i + n_fft]
        if frame.size < n_fft:
            break
        specs.append(np.abs(np.fft.rfft(frame * window)))
    if not specs:
        return frames
    noise_mag = np.mean(specs, axis=0).astype(np.float32)
    eps = 1e-12
    low_bins = int(60.0 * n_fft / rate) + 1

    out = np.zeros(samples.size + n_fft, dtype=np.float32)
    norm = np.zeros(samples.size + n_fft, dtype=np.float32)
    for i in range(0, max(1, samples.size - n_fft), hop):
        frame = samples[i:i + n_fft]
        if frame.size < n_fft:
            break
        spec = np.fft.rfft(frame * window)
        mag = np.abs(spec)
        mask = np.clip(1.0 - (noise_mag * 0.7) / (mag + eps), 0.0, 1.0)
        mask = np.maximum(mask, 0.3)
        mask[:low_bins] = 0.2
        out[i:i + n_fft] += np.fft.irfft(spec * mask, n=n_fft) * window
        norm[i:i + n_fft] += window * window
    norm = np.maximum(norm, eps)
    y = np.divide(out, norm)[:samples.size]
    y = np.clip(y, -1.0, 1.0)
    return (y * 32767.0).astype(np.int16).tobytes()


def cond_deepfilternet(rate: int, frames: bytes) -> bytes:
    """DeepFilterNet enhancement — only if the package is importable."""
    try:
        from df.enhance import enhance, init_df, load_audio  # type: ignore
    except Exception:
        raise RuntimeError("deepfilternet is not installed")
    import numpy as np
    import torch  # type: ignore

    model, df_state, _ = init_df()
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    tensor = torch.from_numpy(audio)[None, :]
    enhanced = enhance(model, df_state, tensor)
    out = (enhanced.squeeze(0).numpy() * 32767.0).clip(-32768, 32767)
    return out.astype(np.int16).tobytes()


CONDITIONS = {
    "A_raw": cond_raw,
    "B_gain": cond_gain,
    "C_spectralgate": cond_spectralgate,
    "D_deepfilternet": cond_deepfilternet,
}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def _normalize_words(text: str) -> list[str]:
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return [w for w in text.split() if w]


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = _normalize_words(reference), _normalize_words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / len(ref)


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #


def record_samples(lines: int = 20) -> None:
    """Capture utterances live with the app's own MicCapture settings.

    Speak each line at varying distance/level; type the exact reference text
    after each captured utterance. Saves lab_samples/NN.wav + references.txt.
    """
    LAB_DIR.mkdir(parents=True, exist_ok=True)
    from audio_capture import MicCapture

    cap = MicCapture(vad_threshold=0.02, silence_timeout=0.9)
    refs: list[str] = []
    cap.start()
    print("Speak after the prompt; Ctrl+C to stop early.")
    try:
        idx = 1
        while idx <= lines:
            print(f"\n[{idx}/{lines}] Speak now…")
            for wav in cap.utterances():
                name = f"{idx:02d}.wav"
                rate = 16000
                write_wav(LAB_DIR / name, rate, wav)
                text = input("  reference text> ").strip()
                if text:
                    refs.append(f"{name}|{text}")
                    print(f"  saved {name}")
                    idx += 1
                else:
                    print("  (discarded — no reference given)")
                break
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        cap.stop()
    (LAB_DIR / "references.txt").write_text(
        "\n".join(refs) + "\n", encoding="utf-8")
    print(f"\n{len(refs)} samples in {LAB_DIR}")


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def load_samples() -> list[tuple[Path, str]]:
    refs_path = LAB_DIR / "references.txt"
    if not refs_path.exists():
        raise SystemExit(f"No {refs_path}. Run with --record first.")
    out = []
    for line in refs_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        name, text = line.split("|", 1)
        path = LAB_DIR / name.strip()
        if path.exists():
            out.append((path, text.strip()))
    return out


def evaluate(execute: bool) -> None:
    samples = load_samples()
    if not samples:
        raise SystemExit("No samples with references found.")
    print(f"{len(samples)} sample lines loaded.\n")

    available = {}
    for name, fn in CONDITIONS.items():
        if name == "D_deepfilternet":
            try:
                import df  # noqa: F401
                available[name] = fn
            except Exception:
                print(f"[skip] {name}: deepfilternet not installed")
                continue
        else:
            available[name] = fn

    if not execute:
        print("DRY RUN — no transcription calls will be made.")
        print(f"Would transcribe {len(samples)} lines x {len(available)} "
              f"conditions = {len(samples) * len(available)} Whisper calls.")
        for name in available:
            print(f"  condition {name}: ready")
        return

    from transcriber import Transcriber

    transcriber = Transcriber()
    api_calls = 0
    results: dict[str, dict] = {name: {"wer_sum": 0.0, "wer_n": 0,
                                       "empty": 0, "proc_s": 0.0,
                                       "audio_s": 0.0}
                                for name in available}
    tmp = LAB_DIR / "_variant.wav"

    for path, reference in samples:
        rate, frames = read_wav(path)
        audio_s = duration_seconds(rate, frames)
        for name, fn in available.items():
            t0 = time.perf_counter()
            try:
                variant = fn(rate, frames)
            except Exception as exc:
                print(f"  {name} failed on {path.name}: {exc}")
                continue
            proc_s = time.perf_counter() - t0
            write_wav(tmp, rate, variant)
            hyp = transcriber.transcribe(tmp).text
            api_calls += 1
            score = wer(reference, hyp)
            r = results[name]
            r["wer_sum"] += score
            r["wer_n"] += 1
            r["empty"] += 1 if not hyp.strip() else 0
            r["proc_s"] += proc_s
            r["audio_s"] += audio_s
            print(f"  {path.name} {name}: WER={score:.2f} "
                  f"proc={proc_s:.2f}s audio={audio_s:.2f}s")

    print(f"\n=== RESULTS ({api_calls} Whisper calls) ===")
    for name, r in results.items():
        if r["wer_n"] == 0:
            continue
        avg_wer = r["wer_sum"] / r["wer_n"]
        rt_factor = r["proc_s"] / r["audio_s"] if r["audio_s"] else 0.0
        print(f"{name:18s} WER={avg_wer:.3f}  dropped={r['empty']}  "
              f"realtime_factor={rt_factor:.2f} "
              f"({'OK' if rt_factor < 1.0 else 'TOO SLOW'})")
    tmp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record", action="store_true",
                        help="capture samples with the app's MicCapture")
    parser.add_argument("--execute", action="store_true",
                        help="run real Whisper transcription (costs quota)")
    parser.add_argument("--dry-run", action="store_true",
                        help="show the plan without API calls (default)")
    args = parser.parse_args()
    if args.record:
        record_samples()
        return
    evaluate(execute=args.execute)


if __name__ == "__main__":
    main()
