"""Social Mode coaching for ClearCommit (Stage 3B).

Isolated, testable social coaching analyzer. No UI, no live-translation
pipeline connection, no raw audio, no real model calls in tests.

Detects two categories in finalized transcript text:
    * idiom_explanation  — slang, idioms, or cultural references that may be
      unclear to a non-native speaker.
    * tone_suggestion    — wording that may sound unintentionally rude,
      awkward, or excessively formal in the target language.

Returns a :class:`CoachingSuggestion` or ``None`` when no coaching is needed,
coaching is disabled, or the provider is unavailable.

All model output is validated deterministically before any suggestion is
produced. Coaching is disabled by default (consistent with
:class:`mode_config.CoachingConfig`).

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
    """Pluggable provider for social coaching.

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

    ``kind`` is one of ``"idiom_explanation"`` or ``"tone_suggestion"``.
    ``original`` is the verbatim phrase being coached. ``suggestion`` is the
    suggested rephrasing or plain-language explanation. ``reason`` briefly
    explains why the suggestion may help. ``evidence_uid`` references the
    immutable transcript event this suggestion is based on. ``side`` is
    ``"p1"`` or ``"you"``.
    """

    kind: str
    original: str
    suggestion: str
    reason: str
    evidence_uid: int
    side: str


# Deterministic, non-AI heuristics used as a cheap pre-filter so the model is
# only consulted for text that is likely to contain idioms or tone issues.
# These are intentionally conservative: they never produce suggestions on
# their own, only gate whether the (possibly absent) provider is invoked.
_IDIOM_HINTS = frozenset({
    "break a leg", "piece of cake", "kick the bucket", "bite the bullet",
    "spill the beans", "costs an arm and a leg", "once in a blue moon",
    "raining cats and dogs", "under the weather", "hit the sack",
    "cutting corners", "stabbed in the back", "jump the gun",
    "let the cat out of the bag", "miss the boat", "on the fence",
    "pull someone's leg", "the ball is in your court", "through the roof",
    "under the table", "wrap your head around", "your guess is as good as mine",
})
_TONE_HINTS = frozenset({
    "as i said", "obviously", "clearly", "you should", "you must",
    "i demand", "this is ridiculous", "unacceptable", "absurd",
    "kind of", "sort of", "maybe", "perhaps", "i guess", "i think",
})


@dataclass
class SocialCoach:
    """Social Mode coaching analyzer.

    Thread-safe. Accepts finalized transcript text + metadata only — never
    raw audio. Coaching is disabled by default.
    """

    def __init__(
        self,
        *,
        provider: Optional[CoachingProvider] = None,
        enabled: bool = False,
        detect_idioms: bool = True,
        detect_tone: bool = True,
        max_text_chars: int = 500,
        dedupe_window: int = 50,
    ) -> None:
        self._provider = provider
        self._enabled = bool(enabled)
        self._detect_idioms = bool(detect_idioms)
        self._detect_tone = bool(detect_tone)
        # Optional persona overrides (system portion only; the JSON response
        # format is always appended by the prompt builders).
        self._system_idiom: Optional[str] = None
        self._system_tone: Optional[str] = None
        self._max_text_chars = int(max_text_chars)
        self._dedupe_window = max(1, int(dedupe_window))

        self._lock = threading.RLock()
        # Recent suggestion keys (kind + normalized original) to avoid
        # repeating the same coaching for a recurring phrase.
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

    def set_options(self, *, idioms: Optional[bool] = None,
                    tone: Optional[bool] = None) -> None:
        """Toggle detection categories at runtime (None leaves unchanged)."""
        with self._lock:
            if idioms is not None:
                self._detect_idioms = bool(idioms)
            if tone is not None:
                self._detect_tone = bool(tone)

    def set_prompts(self, *, idiom: Optional[str] = None,
                    tone: Optional[str] = None) -> None:
        """Set persona overrides (None/'' restores the default prompt)."""
        with self._lock:
            self._system_idiom = idiom.strip() if idiom and idiom.strip() else None
            self._system_tone = tone.strip() if tone and tone.strip() else None

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
        """Analyze finalized transcript text for social coaching.

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

        # Cheap pre-filter: only consult the model when the text is likely to
        # contain an idiom or tone issue. This avoids unnecessary model calls.
        likely_idiom = self._detect_idioms and self._likely_idiom(lower)
        likely_tone = self._detect_tone and self._likely_tone(lower)
        if not likely_idiom and not likely_tone:
            return None

        if likely_idiom:
            system = self._system_idiom
            prompt = (
                f"{system}\n\nThe user said: \"{text}\"\n"
                + JSON_IDIOM_SUFFIX
                if system else self._build_idiom_prompt(text)
            )
            result = self._try_category(
                provider, "idiom_explanation", text, side, uid, context,
                prompt=prompt,
            )
            if result is not None:
                return result

        if likely_tone:
            system = self._system_tone
            prompt = (
                f"{system}\n\nThe user said: \"{text}\"\n"
                + JSON_TONE_SUFFIX
                if system else self._build_tone_prompt(text)
            )
            result = self._try_category(
                provider, "tone_suggestion", text, side, uid, context,
                prompt=prompt,
            )
            if result is not None:
                return result

        return None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _likely_idiom(text_lower: str) -> bool:
        return any(hint in text_lower for hint in _IDIOM_HINTS)

    @staticmethod
    def _likely_tone(text_lower: str) -> bool:
        return any(hint in text_lower for hint in _TONE_HINTS)

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

        # Strip surrounding code fences if the model wrapped JSON in them.
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

        # "none" means the model determined no coaching is needed.
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
    def _build_idiom_prompt(text: str) -> str:
        return (
            DEFAULT_IDIOM_SYSTEM
            + f"\n\nThe user said: \"{text}\"\n"
            + JSON_IDIOM_SUFFIX
        )

    @staticmethod
    def _build_tone_prompt(text: str) -> str:
        return (
            DEFAULT_TONE_SYSTEM
            + f"\n\nThe user said: \"{text}\"\n"
            + JSON_TONE_SUFFIX
        )


# Default persona portions + the machine-readable response formats that are
# ALWAYS appended (overrides replace the persona, never the JSON contract).
DEFAULT_IDIOM_SYSTEM = (
    "You are a conversation coach for multilingual social calls. "
    "Determine whether the user's line contains an idiom, slang, or "
    "cultural reference that a non-native speaker might misunderstand."
)
DEFAULT_TONE_SYSTEM = (
    "You are a conversation coach for multilingual social calls. "
    "Determine whether the user's wording may sound unintentionally rude, "
    "awkward, or excessively formal in the target language."
)
JSON_INSTRUCTION = "Respond ONLY with valid JSON, no other text:\n"
_JSON_IDIOM_SCHEMA = (
    '{\n'
    '  "kind": "idiom_explanation" or "none",\n'
    '  "original": "the exact phrase being coached",\n'
    '  "suggestion": "plain-language explanation of the phrase",\n'
    '  "reason": "one sentence why this may help"\n'
    '}\n'
)
_JSON_TONE_SCHEMA = (
    '{\n'
    '  "kind": "tone_suggestion" or "none",\n'
    '  "original": "the exact phrase being coached",\n'
    '  "suggestion": "a clearer, more diplomatic rephrasing",\n'
    '  "reason": "one sentence why this may help"\n'
    '}\n'
)
JSON_IDIOM_SUFFIX = JSON_INSTRUCTION + _JSON_IDIOM_SCHEMA
JSON_TONE_SUFFIX = JSON_INSTRUCTION + _JSON_TONE_SCHEMA


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
