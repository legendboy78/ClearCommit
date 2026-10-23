"""AI model + prompt configuration for ClearCommit.

Pure, validated configuration: which model each AI feature uses, its
temperature/token budget, per-coach feature toggles, and per-feature system
prompt overrides. No network, no UI — flows to providers through the
existing context mechanism; a real provider consumes it when connected.

Prompt overrides REPLACE only the persona/rules (system) portion of each
feature's prompt. The response-format instructions (valid JSON, exact
schema) are always appended by the coaches so user edits can never break
machine-readable output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# Feature ids used for prompt overrides and the UI dropdown.
FEATURE_COACH_IDIOM = "coach_idiom"
FEATURE_COACH_TONE = "coach_tone"
FEATURE_COACH_COMM = "coach_communication"
FEATURE_COACH_PITCH = "coach_pitch"
FEATURE_ASSISTANT = "assistant"
FEATURE_MEETING = "meeting"
ALL_FEATURES = (
    FEATURE_COACH_IDIOM, FEATURE_COACH_TONE, FEATURE_COACH_COMM,
    FEATURE_COACH_PITCH, FEATURE_ASSISTANT, FEATURE_MEETING,
)

DEFAULT_SYSTEM_PROMPTS: dict[str, str] = {
    FEATURE_COACH_IDIOM: (
        "You are a conversation coach for multilingual social calls. You "
        "spot idioms, slang, and cultural references a non-native speaker "
        "might misunderstand, and explain them in plain language."
    ),
    FEATURE_COACH_TONE: (
        "You are a conversation coach for multilingual social calls. You "
        "spot wording that may sound unintentionally rude, awkward, or "
        "excessively formal, and suggest a friendlier way to say it."
    ),
    FEATURE_COACH_COMM: (
        "You are a professional communication coach for multilingual "
        "business calls. You identify unclear, ambiguous, unnecessarily "
        "aggressive, or overly complicated wording and suggest a clearer, "
        "more diplomatic alternative while preserving the speaker's "
        "intended meaning and factual claims. Never invent benefits, "
        "statistics, evidence, or promises."
    ),
    FEATURE_COACH_PITCH: (
        "You are a pitch coach for multilingual business calls. You help "
        "express ideas, value, and answers concisely and precisely, without "
        "inventing benefits, statistics, evidence, or promises."
    ),
    FEATURE_ASSISTANT: (
        "You are ClearCommit's meeting assistant. Answer ONLY from the "
        "provided transcript and meeting record. Cite evidence as [E<uid>] "
        "for factual claims. Say plainly when the transcript does not "
        "contain the answer. Never invent dates, names, agreements, or "
        "consent; never treat a proposal as a decision or silence as "
        "agreement."
    ),
    FEATURE_MEETING: (
        "You are a precise meeting-intelligence extractor. From the "
        "transcript, output only the requested JSON with decisions, action "
        "items, open questions, candidate commitments, and events. Reference "
        "only real evidence ids. Dates stay as spoken unless unambiguous. "
        "Never mark anything confirmed and never invent details."
    ),
}


@dataclass
class AIModelSettings:
    """Model + coach-option settings shared by the AI features.

    ``model`` is a provider model id (consumed when a real provider is
    connected; inert until then). Temperature is clamped to [0, 1.5],
    max_tokens to [64, 4096]. Coach toggles gate which categories the
    coaches will even consider (deterministic pre-filter).
    """

    model: str = "openai/gpt-oss-120b"
    temperature: float = 0.3
    max_tokens: int = 1024
    coach_idioms: bool = True
    coach_tone: bool = True
    coach_communication: bool = True
    coach_pitch: bool = True

    @classmethod
    def from_dict(cls, data: object) -> "AIModelSettings":
        if not isinstance(data, dict):
            return cls()
        model = data.get("model")
        model = model.strip() if isinstance(model, str) and model.strip() \
            else "openai/gpt-oss-120b"
        try:
            temperature = float(data.get("temperature", 0.3))
        except (TypeError, ValueError):
            temperature = 0.3
        temperature = max(0.0, min(1.5, temperature))
        try:
            max_tokens = int(data.get("max_tokens", 1024))
        except (TypeError, ValueError):
            max_tokens = 1024
        max_tokens = max(64, min(4096, max_tokens))

        def _flag(key: str, default: bool = True) -> bool:
            v = data.get(key, default)
            return bool(v) if isinstance(v, bool) else default

        return cls(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            coach_idioms=_flag("coach_idioms"),
            coach_tone=_flag("coach_tone"),
            coach_communication=_flag("coach_communication"),
            coach_pitch=_flag("coach_pitch"),
        )

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "coach_idioms": self.coach_idioms,
            "coach_tone": self.coach_tone,
            "coach_communication": self.coach_communication,
            "coach_pitch": self.coach_pitch,
        }

    def context(self) -> dict:
        """Provider context dict consumed by real backends when connected."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


def normalize_override(value: object) -> Optional[str]:
    """ ''/None -> None (use default); else stripped text."""
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def sanitize_overrides(raw: object) -> dict[str, str]:
    """Keep only known feature keys with non-empty string overrides."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, val in raw.items():
        if key in ALL_FEATURES and isinstance(val, str) and val.strip():
            out[key] = val.strip()
    return out
