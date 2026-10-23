"""Deterministic agreement ledger for ClearCommit.

Pure Python state machine for tracking commitments, proposed terms,
clarification state, and explicit confirmations. No network, no audio,
no UI. Independently unit-testable.

Statuses:
    proposed            - an item has been identified but not yet confirmed
    needs_clarification - a consequential detail is unresolved
    disputed            - participants disagree on interpretation
    confirmed           - explicitly confirmed by relevant participants
    rejected            - explicitly rejected (terminal)

Safety properties enforced in application code:
    * A translation is never treated as proof of mutual understanding.
    * Missing prices/deadlines/quantities are never invented.
    * Silence is never treated as consent.
    * An item is confirmed only after explicit participant confirmation
      history exists; model-originated claims never count.
    * Evidence references are immutable transcript pointers (uid + text).
    * No raw audio or personal data is stored.

Status transitions (deterministic):

    proposed  -> needs_clarification | confirmed | rejected
    needs_clarification -> confirmed | rejected | disputed | proposed
    disputed  -> needs_clarification | confirmed | rejected   (intentional addition)
    confirmed -> (terminal)
    rejected  -> (terminal)

``disputed`` is an intentional addition to the original status set. It is
reachable from ``needs_clarification`` when participants disagree on an
interpretation, and it can resolve back to ``needs_clarification`` (for a
new clarification round), forward to ``confirmed`` (once explicitly
confirmed), or to ``rejected``. It is not terminal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import uuid

# Confirmation-record origin. Only ``participant`` records can count toward
# an explicit confirmation; ``model`` records (LLM-generated claims) are
# recorded for transparency but never satisfy the confirmation requirement.
PARTICIPANT_SOURCE = "participant"
MODEL_SOURCE = "model"
ALLOWED_SOURCES = frozenset({PARTICIPANT_SOURCE, MODEL_SOURCE})


@dataclass(frozen=True)
class TranscriptEvidence:
    """Immutable reference to a finalized transcript utterance."""

    uid: int
    side: str                      # "p1" or "you"
    original: str                  # original-language text
    translated: str                # translated text
    accept_ts: float               # epoch seconds when utterance was accepted
    source_lang: str
    target_lang: str


@dataclass
class ConfirmationRecord:
    """One explicit confirmation/rejection action.

    ``source`` identifies where the record originated. Only records whose
    source is ``participant`` can count toward an explicit confirmation;
    records sourced from ``model`` (an LLM claim) are kept for audit
    transparency but never satisfy the confirmation requirement.
    """

    speaker: str                   # "p1" or "you" (the participant who acted)
    action: str                    # "confirm" or "reject"
    ts: float                      # epoch seconds
    source: str                    # "participant" or "model"
    note: str = ""                 # optional free-text reason


@dataclass
class Commitment:
    """A single tracked commitment / proposed term."""

    id: str
    topic: str                     # commitment type / topic
    proposed_value: str            # what was proposed (verbatim from evidence)
    speaker: str                   # who proposed it
    evidence: list[TranscriptEvidence] = field(default_factory=list)
    status: str = "proposed"
    confirmation_history: list[ConfirmationRecord] = field(default_factory=list)
    created_ts: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    updated_ts: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    clarification_question: Optional[str] = None
    resolution_note: Optional[str] = None


class AgreementLedger:
    """In-memory ledger of commitments. Thread-unsafe; callers serialize.

    All state transitions are deterministic. No LLM is consulted here;
    the ledger only records what the application tells it.
    """

    VALID_STATUSES = frozenset({
        "proposed", "needs_clarification", "disputed", "confirmed", "rejected",
    })

    # Deterministic transitions. confirmed/rejected are terminal.
    _VALID_TRANSITIONS = {
        "proposed": {"needs_clarification", "confirmed", "rejected"},
        "needs_clarification": {"confirmed", "rejected", "disputed", "proposed"},
        "disputed": {"needs_clarification", "confirmed", "rejected"},
        "confirmed": set(),
        "rejected": set(),
    }

    def __init__(self) -> None:
        self._items: dict[str, Commitment] = {}

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def get(self, commitment_id: str) -> Optional[Commitment]:
        return self._items.get(commitment_id)

    def items(self) -> list[Commitment]:
        return list(self._items.values())

    def items_by_status(self, status: str) -> list[Commitment]:
        if status not in self.VALID_STATUSES:
            return []
        return [c for c in self._items.values() if c.status == status]

    def is_confirmed(self, commitment_id: str) -> bool:
        item = self._items.get(commitment_id)
        return item is not None and item.status == "confirmed"

    # ------------------------------------------------------------------ #
    # Mutations
    # ------------------------------------------------------------------ #
    def add_item(
        self,
        *,
        topic: str,
        proposed_value: str,
        speaker: str,
        evidence: list[TranscriptEvidence],
        status: str = "proposed",
    ) -> str:
        """Create a new commitment. Returns its id."""
        if status not in self.VALID_STATUSES:
            status = "proposed"
        cid = uuid.uuid4().hex
        now = datetime.now(timezone.utc).timestamp()
        item = Commitment(
            id=cid,
            topic=topic,
            proposed_value=proposed_value,
            speaker=speaker,
            evidence=list(evidence),
            status=status,
            created_ts=now,
            updated_ts=now,
        )
        self._items[cid] = item
        return cid

    def add_evidence(self, commitment_id: str, evidence: TranscriptEvidence) -> bool:
        """Append immutable evidence to an existing item."""
        item = self._items.get(commitment_id)
        if item is None:
            return False
        new_evidence = list(item.evidence) + [evidence]
        now = datetime.now(timezone.utc).timestamp()
        self._items[commitment_id] = Commitment(
            **{**item.__dict__, "evidence": new_evidence, "updated_ts": now},
        )
        return True

    def set_clarification(
        self,
        commitment_id: str,
        question: Optional[str],
    ) -> bool:
        """Set or clear the clarification question for an item."""
        item = self._items.get(commitment_id)
        if item is None:
            return False
        now = datetime.now(timezone.utc).timestamp()
        self._items[commitment_id] = Commitment(
            **{**item.__dict__, "clarification_question": question, "updated_ts": now},
        )
        return True

    def transition(
        self,
        commitment_id: str,
        new_status: str,
        *,
        resolution_note: Optional[str] = None,
    ) -> bool:
        """Deterministically transition an item to a new status."""
        if new_status not in self.VALID_STATUSES:
            return False
        item = self._items.get(commitment_id)
        if item is None:
            return False
        if new_status != item.status and new_status not in self._VALID_TRANSITIONS[item.status]:
            return False
        now = datetime.now(timezone.utc).timestamp()
        self._items[commitment_id] = Commitment(
            **{**item.__dict__, "status": new_status, "resolution_note": resolution_note, "updated_ts": now},
        )
        return True

    def record_confirmation(
        self,
        commitment_id: str,
        speaker: str,
        action: str,   # "confirm" or "reject"
        *,
        source: str = PARTICIPANT_SOURCE,
        note: str = "",
    ) -> bool:
        """Record an explicit confirmation/rejection action.

        ``source`` identifies where the record originated
        (``participant`` or ``model``). Only ``participant`` records can
        later count toward an explicit confirmation; ``model`` records
        (LLM-generated claims) are kept for audit transparency but never
        satisfy the confirmation requirement.

        Returns False (and leaves history unchanged) on a malformed record:
        unknown item, empty speaker, invalid action, or invalid source.
        Does not by itself change item status; the application decides when
        relevant participants have confirmed.
        """
        if action not in ("confirm", "reject"):
            return False
        if source not in ALLOWED_SOURCES:
            return False
        if not speaker or not str(speaker).strip():
            return False
        item = self._items.get(commitment_id)
        if item is None:
            return False
        now = datetime.now(timezone.utc).timestamp()
        record = ConfirmationRecord(
            speaker=str(speaker).strip(),
            action=action,
            ts=now,
            source=source,
            note=note,
        )
        history = list(item.confirmation_history) + [record]
        self._items[commitment_id] = Commitment(
            **{**item.__dict__, "confirmation_history": history, "updated_ts": now},
        )
        return True

    @staticmethod
    def _has_valid_participant_confirmation(item: "Commitment") -> bool:
        """True only if history holds at least one explicit participant
        confirmation (source == participant, action == confirm, speaker set).

        Model-originated claims, reject actions, and speaker-less records
        never count. This is the minimum evidence the ledger requires before
        an item may be marked confirmed; which participants must confirm is
        decided at the application layer.
        """
        return any(
            r.source == PARTICIPANT_SOURCE
            and r.action == "confirm"
            and r.speaker
            for r in item.confirmation_history
        )

    def mark_confirmed_explicitly(
        self,
        commitment_id: str,
        *,
        resolution_note: Optional[str] = None,
    ) -> bool:
        """Mark an item confirmed only if explicit participant confirmation
        history exists.

        Refuses (returns False) when there is no qualifying participant
        confirmation record. Silence, an empty history, or an LLM/model
        claim alone can never confirm an item. The application layer decides
        which participants must confirm; this method enforces only the
        minimum evidence (at least one valid participant confirmation).
        """
        item = self._items.get(commitment_id)
        if item is None:
            return False
        if not self._has_valid_participant_confirmation(item):
            return False
        return self.transition(commitment_id, "confirmed", resolution_note=resolution_note)
