"""Communication mode configuration for the ClearCommit upgrade.

Pure, independently testable configuration model. No network, no audio,
no UI. Defaults preserve the existing translator behavior: social mode
with coaching disabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


class CommunicationMode:
    """Valid communication modes.

    ``social``  -> natural conversational translation (current behavior).
    ``professional`` -> business-oriented communication assistance.
    """

    SOCIAL = "social"
    PROFESSIONAL = "professional"

    _VALID = frozenset({SOCIAL, PROFESSIONAL})

    @classmethod
    def is_valid(cls, value: Optional[str]) -> bool:
        return value in cls._VALID

    @classmethod
    def normalize(cls, value: Optional[str], default: str = SOCIAL) -> str:
        """Return a valid mode, falling back to ``default`` for anything else."""
        if value is None:
            return default
        value = str(value).strip().lower()
        return value if value in cls._VALID else default


@dataclass
class CoachingConfig:
    """Whether private AI coaching suggestions are active."""

    enabled: bool = False

    @classmethod
    def from_dict(cls, data: object) -> "CoachingConfig":
        if not isinstance(data, dict):
            return cls()
        return cls(enabled=bool(data.get("enabled", False)))

    def to_dict(self) -> dict:
        return {"enabled": self.enabled}


@dataclass
class ModeConfig:
    """Communication mode and coaching configuration.

    Defaults preserve the current translator behavior: social mode with
    coaching disabled.
    """

    mode: str = CommunicationMode.SOCIAL
    coaching: CoachingConfig = field(default_factory=CoachingConfig)

    @classmethod
    def from_dict(cls, data: object) -> "ModeConfig":
        """Build from a settings dict, safely handling missing/invalid data."""
        if not isinstance(data, dict):
            return cls()
        mode = CommunicationMode.normalize(data.get("mode"))
        coaching = CoachingConfig.from_dict(data.get("coaching"))
        return cls(mode=mode, coaching=coaching)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "coaching": self.coaching.to_dict(),
        }

    def is_professional(self) -> bool:
        return self.mode == CommunicationMode.PROFESSIONAL

    def coaching_active(self) -> bool:
        return self.coaching.enabled
