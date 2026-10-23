"""Professional Mode coaching for ClearCommit (Stage 4).

Isolated, testable professional coaching analyzer. No UI, no live-translation
pipeline connection, no raw audio, no real model calls in tests.

Detects two categories of coaching opportunities in finalized transcript text:
    * communication_coach — unclear, ambiguous, unnecessarily aggressive, or
      overly complicated wording; suggests clearer, diplomatic, persuasive
      alternatives while preserving the speaker's intended meaning and
      factual claims.
    * pitch_coach — suggestions for explaining an idea, describing value, and
      responding to questions; concise and usable during a live call; never
      invents benefits, statistics, evidence, or promises.

Returns a :class:`CoachingSuggestion` or ``None`` when no coaching is needed,
coaching is disabled, or the provider is unavailable.

All model output is validated deterministically before any suggestion is
produced. Coaching is disabled by default. Suggestions are private to the user
and never automatically spoken or sent to other participants.

The provider is injectable so a deterministic fake can be used in tests and a
real backend (Groq / Nemotron / other) can be plugged in later without
touching the UI or translation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol
import json
import re
import threading


class CoachingProvider(Protocol):
    """Pluggable provider for professional coaching.

    ``analyze(text, context)`` receives finalized transcript text (bounded in
    size) plus an optional context dict and returns a JSON string describing a
    coaching suggestion, or ``None`` when no suggestion is warranted.

    Providers must be cooperative: return promptly and never block forever.
    """

    def analyze(self, text: str, context: Optional[dict] = None) -> Optional[str]:
        raise NotImplementedError


@dataclass(frozen=True)
class CoachingSuggestion:
    """A private coaching suggestion for the user.

    ``kind`` is one of ``"communication_coach"`` or ``"pitch_coach"``.
    ``original`` is the verbatim phrase being coached. ``suggestion`` is the
    suggested rephrasing. ``reason`` briefly explains why the suggestion may
    help. ``evidence_uid`` references the immutable transcript event this
    suggestion is based on. ``side`` is ``"p1"`` or ``"you"``.
    """

    kind: str
    original: str
    suggestion: str
    reason: str
    evidence_uid: int
    side: str


# Deterministic, non-AI heuristics used as a cheap pre-filter so the model is
# only consulted for text that is likely to contain unclear/aggressive/complex
# wording or pitch-relevant content. These never produce suggestions on their
# own; they only gate whether the (possibly absent) provider is invoked.
_AGGRESSION_HINTS = frozenset({
    "you always", "you never", "this is ridiculous", "unacceptable",
    "absurd", "i demand", "you need to", "you have to", "wrong",
    "stupid", "idiotic", "nonsense", "ridiculous",
})
_COMPLEXITY_HINTS = frozenset({
    "utilize", "facilitate", "leverage", "in the event that",
    "notwithstanding", "heretofore", "aforementioned", "pursuant to",
    "commence", "terminate", "subsequently", "nevertheless",
    "furthermore", "consequently", "moreover",
})
_PITCH_HINTS = frozenset({
    "our product", "our service", "we offer", "value proposition",
    "return on investment", "roi", "cost savings", "efficiency",
    "scalable", "market", "customer", "client", "partner",
    "proposal", "investment", "pricing", "timeline", "deliverable",
    "how does this work", "what is the value", "why should we",
})


@dataclass
class ProfessionalCoach:
    """Professional Mode coaching analyzer.

    Thread-safe. Accepts finalized transcript text + metadata only — never
    raw audio. Coaching is disabled by default.
    """

    def __init__(
        self,
        *,
        provider: Optional[CoachingProvider] = None,
        enabled: bool = False,
        detect_communication: bool = True,
        detect_pitch: bool = True,
        max_text_chars: int = 800,
        dedupe_window: int = 50,
    ) -> None:
        self._provider = provider
        self._enabled = bool(enabled)
        self._detect_communication = bool(detect_communication)
        self._detect_pitch = bool(detect_pitch)
        # Optional persona overrides (system portion only; the JSON response
        # format is always appended by the prompt builders).
        self._system_communication: Optional[str] = None
        self._system_pitch: Optional[str] = None
        self._max_text_chars = int(max_text_chars)
        self._dedupe_window = max(1, int(dedupe_window))

        self._lock = threading.RLock()
        self._recent: list[str] = []

    # ------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------ #
    def is_enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = bool(enabled)

    def set_provider(self, provider: Optional[CoachingProvider]) -> None:
        with self._lock:
            self._provider = provider

    def set_options(self, *, communication: Optional[bool] = None,
                    pitch: Optional[bool] = None) -> None:
        """Toggle detection categories at runtime (None leaves unchanged)."""
        with self._lock:
            if communication is not None:
                self._detect_communication = bool(communication)
            if pitch is not None:
                self._detect_pitch = bool(pitch)

    def set_prompts(self, *, communication: Optional[str] = None,
                    pitch: Optional[str] = None) -> None:
        """Set persona overrides (None/'' restores the default prompt)."""
        with self._lock:
            self._system_communication = (
                communication.strip() if communication and communication.strip()
                else None)
            self._system_pitch = (
                pitch.strip() if pitch and pitch.strip() else None)

    # ------------------------------------------------------------------ #
    # Analysis
    # ------------------------------------------------------------------ #
    def analyze(
        self,
        text: str,
        *,
        side: str,
        uid: int,
        context: Optional[dict] = None,
    ) -> Optional[CoachingSuggestion]:
        """Analyze finalized transcript text for professional coaching.

        Returns a :class:`CoachingSuggestion` or ``None`` when no coaching is
        needed, coaching is disabled, or the provider is unavailable.
        """
        if not self._enabled:
            return None
        if not text or not text.strip():
            return None

        text = text.strip()
        if len(text) > self._max_text_chars:
            text = text[: self._max_text_chars]

        provider = self._provider
        if provider is None:
            return None

        lower = text.lower()

        likely_communication = self._detect_communication and (
            self._likely_communication(lower)
        )
        likely_pitch = self._detect_pitch and self._likely_pitch(lower)
        if not likely_communication and not likely_pitch:
            return None

        if likely_communication:
            system = self._system_communication
            prompt = (
                f"{system}\n\nThe user said: \"{text}\"\n"
                + JSON_COMM_SUFFIX
                if system else self._build_communication_prompt(text)
            )
            result = self._try_category(
                provider, "communication_coach", text, side, uid, context,
                prompt=prompt,
            )
            if result is not None:
                return result

        if likely_pitch:
            system = self._system_pitch
            prompt = (
                f"{system}\n\nThe user said: \"{text}\"\n"
                + JSON_PITCH_SUFFIX
                if system else self._build_pitch_prompt(text)
            )
            result = self._try_category(
                provider, "pitch_coach", text, side, uid, context,
                prompt=prompt,
            )
            if result is not None:
                return result

        return None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _likely_communication(text_lower: str) -> bool:
        return (
            any(hint in text_lower for hint in _AGGRESSION_HINTS)
            or any(hint in text_lower for hint in _COMPLEXITY_HINTS)
        )

    @staticmethod
    def _likely_pitch(text_lower: str) -> bool:
        return any(hint in text_lower for hint in _PITCH_HINTS)

    def _try_category(
        self,
        provider: CoachingProvider,
        kind: str,
        text: str,
        side: str,
        uid: int,
        context: Optional[dict],
        prompt: str,
    ) -> Optional[CoachingSuggestion]:
        key = f"{kind}:{self._normalize(text)}"
        with self._lock:
            if key in self._recent:
                return None

        raw: Optional[str] = None
        try:
            raw = provider.analyze(prompt, context)
        except Exception:  # noqa: BLE001 - provider failures are non-fatal
            return None

        parsed = self._validate_output(kind, raw)
        if parsed is None:
            return None

        with self._lock:
            if key in self._recent:
                return None
            self._recent.append(key)
            if len(self._recent) > self._dedupe_window:
                self._recent = self._recent[-self._dedupe_window:]

        return CoachingSuggestion(
            kind=kind,
            original=parsed["original"],
            suggestion=parsed["suggestion"],
            reason=parsed.get("reason", ""),
            evidence_uid=uid,
            side=side,
        )

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    @staticmethod
    def _validate_output(kind: str, raw: Optional[str]) -> Optional[dict]:
        """Deterministically validate provider output.

        Returns a dict with ``original``, ``suggestion``, ``reason`` or
        ``None`` if the output is malformed, empty, or indicates no coaching
        is needed.
        """
        if not raw or not raw.strip():
            return None

        text = raw.strip()

        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
            if text.endswith("```"):
                text = text[:-3].strip()

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None

        if not isinstance(data, dict):
            return None

        if str(data.get("kind", "")).lower() == "none":
            return None
        if str(data.get("kind", "")).lower() != kind:
            return None

        original = data.get("original")
        suggestion = data.get("suggestion")
        reason = data.get("reason", "")

        if not isinstance(original, str) or not original.strip():
            return None
        if not isinstance(suggestion, str) or not suggestion.strip():
            return None
        if not isinstance(reason, str):
            reason = ""

        return {
            "original": original.strip(),
            "suggestion": suggestion.strip(),
            "reason": reason.strip(),
        }

    @staticmethod
    def _build_communication_prompt(text: str) -> str:
        return (
            DEFAULT_COMM_SYSTEM
            + f"\n\nThe user said: \"{text}\"\n"
            + JSON_COMM_SUFFIX
        )

    @staticmethod
    def _build_pitch_prompt(text: str) -> str:
        return (
            DEFAULT_PITCH_SYSTEM
            + f"\n\nThe user said: \"{text}\"\n"
            + JSON_PITCH_SUFFIX
        )


# Default persona portions + the machine-readable response formats that are
# ALWAYS appended (overrides replace the persona, never the JSON contract).
DEFAULT_COMM_SYSTEM = (
    "You are a professional communication coach for multilingual business "
    "calls. Identify wording that is unclear, ambiguous, unnecessarily "
    "aggressive, or overly complicated, and suggest a clearer, more "
    "diplomatic, or more persuasive alternative while preserving the "
    "speaker's intended meaning and factual claims. Do not invent benefits, "
    "statistics, evidence, or promises."
)
DEFAULT_PITCH_SYSTEM = (
    "You are a pitch coach for multilingual business calls. Suggest a "
    "concise way to explain the idea, describe value, or respond to a "
    "question during a live call. Do not invent benefits, statistics, "
    "evidence, or promises."
)
JSON_INSTRUCTION = "Respond ONLY with valid JSON, no other text:\n"
_JSON_COMM_SCHEMA = (
    '{\n'
    '  "kind": "communication_coach" or "none",\n'
    '  "original": "the exact phrase being coached",\n'
    '  "suggestion": "clearer diplomatic rephrasing",\n'
    '  "reason": "one sentence why this may help"\n'
    '}\n'
)
_JSON_PITCH_SCHEMA = (
    '{\n'
    '  "kind": "pitch_coach" or "none",\n'
    '  "original": "the exact phrase being coached",\n'
    '  "suggestion": "concise usable rephrasing",\n'
    '  "reason": "one sentence why this may help"\n'
    '}\n'
)
JSON_COMM_SUFFIX = JSON_INSTRUCTION + _JSON_COMM_SCHEMA
JSON_PITCH_SUFFIX = JSON_INSTRUCTION + _JSON_PITCH_SCHEMA


# --------------------------------------------------------------------------- #
# Deterministic fake provider (tests only — no network, no API keys)
# --------------------------------------------------------------------------- #


@dataclass
class FakeCoachingProvider:
    """Deterministic stand-in provider for tests.

    Configure ``output`` (a JSON string or None), ``fail`` (raise), and
    ``call_count`` to observe how many times the provider was consulted.
    """

    output: Optional[str] = None
    fail: bool = False
    error_message: str = "provider failure"
    call_count: int = 0

    def analyze(self, text: str, context: Optional[dict] = None) -> Optional[str]:
        self.call_count += 1
        if self.fail:
            raise RuntimeError(self.error_message)
        return self.output
