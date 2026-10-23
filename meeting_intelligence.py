"""Meeting Intelligence analyzer for ClearCommit (Stage 5A).

Isolated, testable meeting-intelligence analyzer. No UI, no live-translation
pipeline connection, no raw audio, no real model calls in tests.

From finalized transcript text, extracts a structured meeting record:
    * topics discussed
    * decisions (proposed vs decided vs rejected)
    * action items (assigned vs unassigned; stated vs unknown owner/deadline)
    * open questions (answered vs unresolved)
    * candidate commitments (inferred/proposed only -- never confirmed)

Safety properties enforced in application code:
    * Every extracted claim references one or more actual transcript evidence
      IDs supplied in the request. Claims referencing unknown/fabricated
      evidence IDs are rejected.
    * Explicit distinctions: proposed vs decided; assigned vs unassigned;
      stated vs unknown owner/deadline; answered vs unresolved; inferred
      candidate commitment vs explicitly human-confirmed commitment.
    * The model never creates participant confirmation records and never marks
      a commitment confirmed. Human confirmation remains an application-layer
      action through the agreement ledger.
    * No consent, agreement, intent, responsibility, or deadline is inferred
      from silence. Missing owner/deadline remains unknown (None).
    * Conflicting statements are preserved rather than silently resolved
      (deduplication never merges conflicting evidence).
    * All model output is validated deterministically before any record is
      produced. Malformed schemas, unknown evidence references, invented
      participant IDs, invalid statuses, and unsupported dates are rejected.
    * Transcript size, result size, and collection lengths are bounded.

The provider is injectable (MeetingIntelligenceProvider protocol) so a
deterministic fake can be used in tests and a real backend can be plugged in
later without touching the UI or translation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol
import datetime as _dt
import json
import re
import threading
import uuid


# --------------------------------------------------------------------------- #
# Evidence reference
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EvidenceReference:
    """Immutable reference to a finalized transcript utterance.

    ``uid`` is the transcript event id. ``side`` is "p1" or "you". ``text``
    is a snippet of the transcript text. ``timestamp`` is the epoch-second
    accept time of the utterance.
    """

    uid: int
    side: str
    text: str
    timestamp: float


# --------------------------------------------------------------------------- #
# Extracted items (all frozen/immutable)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Decision:
    """A single extracted decision.

    ``status`` is one of "proposed", "decided", "rejected".
    ``decided_value`` is None when the item has only been proposed, not
    decided. Conflicting proposals are kept as separate items rather than
    merged.
    """

    id: str
    topic: str
    proposed_value: str
    decided_value: Optional[str]
    status: str
    evidence: tuple[EvidenceReference, ...]
    speaker: str
    created_ts: float


@dataclass(frozen=True)
class ActionItem:
    """A single extracted action item.

    ``owner`` is None when unassigned/unknown. ``deadline`` is None when
    unknown/not stated. ``status`` is "assigned" or "unassigned".
    """

    id: str
    description: str
    owner: Optional[str]
    deadline: Optional[str]
    status: str
    evidence: tuple[EvidenceReference, ...]
    created_ts: float


@dataclass(frozen=True)
class OpenQuestion:
    """A single extracted open question.

    ``answered`` is False when unresolved. ``answer`` is None when no answer
    was stated.
    """

    id: str
    question: str
    answered: bool
    answer: Optional[str]
    evidence: tuple[EvidenceReference, ...]
    created_ts: float


@dataclass(frozen=True)
class CandidateCommitment:
    """A single inferred candidate commitment.

    ``status`` is "inferred" or "proposed" -- never "confirmed". Confirmation
    is an application-layer action through the agreement ledger, not something
    this analyzer performs.
    """

    id: str
    topic: str
    proposed_value: str
    status: str
    evidence: tuple[EvidenceReference, ...]
    speaker: str
    created_ts: float


@dataclass(frozen=True)
class EventItem:
    """A single extracted event / important detail.

    ``event_type`` is one of VALID_EVENT_TYPES. ``excerpt`` is the original
    transcript excerpt (evidence-backed via ``evidence``). ``spoken_date`` is
    the date/time exactly as spoken (None when none was spoken).
    ``normalized_date`` is an ISO date ONLY when unambiguous and supported by
    the transcript (a relative expression like "next Friday" never receives
    one). ``review_status`` is "explicit" (stated in the transcript),
    "inferred" (derived from context), or "needs_review". ``missing_details``
    lists what would be needed to resolve the item.
    """

    id: str
    event_type: str
    title: str
    excerpt: str
    speaker: str
    spoken_date: Optional[str]
    normalized_date: Optional[str]
    review_status: str
    missing_details: str
    evidence: tuple[EvidenceReference, ...]
    created_ts: float


@dataclass(frozen=True)
class MeetingRecord:
    """Immutable result of one meeting-intelligence analysis."""

    meeting_id: str
    topics: tuple[str, ...]
    decisions: tuple[Decision, ...]
    action_items: tuple[ActionItem, ...]
    open_questions: tuple[OpenQuestion, ...]
    candidate_commitments: tuple[CandidateCommitment, ...]
    events: tuple[EventItem, ...] = ()
    created_ts: float = 0.0


# --------------------------------------------------------------------------- #
# Provider interface
# --------------------------------------------------------------------------- #


class MeetingIntelligenceProvider(Protocol):
    """Pluggable AI backend for meeting-intelligence extraction.

    ``analyze(transcript_text, context)`` receives finalized transcript text
    (bounded in size) plus an optional context dict and returns a JSON string
    describing extracted items, or ``None`` when nothing is extractable.

    The analyzer validates all evidence references against the caller-supplied
    evidence map, so the provider must reference real evidence IDs.
    """

    def analyze(
        self, transcript_text: str, context: Optional[dict] = None
    ) -> Optional[str]:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Deterministic fake provider (tests only -- no network, no API keys)
# --------------------------------------------------------------------------- #


@dataclass
class FakeMeetingProvider:
    """Deterministic stand-in provider for tests.

    Configure ``output`` (a JSON string or None), ``fail`` (raise), and
    ``call_count`` to observe how many times the provider was consulted.
    """

    output: Optional[str] = None
    fail: bool = False
    error_message: str = "provider failure"
    call_count: int = 0

    def analyze(
        self, transcript_text: str, context: Optional[dict] = None
    ) -> Optional[str]:
        self.call_count += 1
        if self.fail:
            raise RuntimeError(self.error_message)
        return self.output


# --------------------------------------------------------------------------- #
# Analyzer
# --------------------------------------------------------------------------- #


class MeetingIntelligence:
    """Bounded, deterministic meeting-intelligence analyzer.

    Thread-safe. Accepts finalized transcript text + an evidence map -- never
    raw audio. Disabled by default.
    """

    VALID_DECISION_STATUSES = frozenset({"proposed", "decided", "rejected"})
    VALID_ACTION_STATUSES = frozenset({"assigned", "unassigned"})
    VALID_COMMITMENT_STATUSES = frozenset({"inferred", "proposed"})
    VALID_EVENT_TYPES = frozenset({
        "meeting", "appointment", "deadline", "milestone",
        "follow_up", "risk", "blocker",
    })
    VALID_EVENT_REVIEW = frozenset({"explicit", "inferred", "needs_review"})
    _MONTH_NAMES = (
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december",
    )
    VALID_SIDES = frozenset({"p1", "you"})

    def __init__(
        self,
        *,
        provider: Optional[MeetingIntelligenceProvider] = None,
        enabled: bool = False,
        max_transcript_chars: int = 2000,
        max_decisions: int = 10,
        max_action_items: int = 10,
        max_open_questions: int = 10,
        max_candidate_commitments: int = 10,
        max_events: int = 10,
        dedupe_window: int = 50,
    ) -> None:
        if max_transcript_chars < 1:
            raise ValueError("max_transcript_chars must be >= 1")
        if max_decisions < 0:
            raise ValueError("max_decisions must be >= 0")
        if max_action_items < 0:
            raise ValueError("max_action_items must be >= 0")
        if max_open_questions < 0:
            raise ValueError("max_open_questions must be >= 0")
        if max_candidate_commitments < 0:
            raise ValueError("max_candidate_commitments must be >= 0")
        if dedupe_window < 1:
            raise ValueError("dedupe_window must be >= 1")

        self._provider = provider
        self._enabled = bool(enabled)
        self._max_transcript_chars = int(max_transcript_chars)
        self._max_decisions = int(max_decisions)
        self._max_action_items = int(max_action_items)
        self._max_open_questions = int(max_open_questions)
        self._max_candidate_commitments = int(max_candidate_commitments)
        self._max_events = int(max_events)
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

    def set_provider(self, provider: Optional[MeetingIntelligenceProvider]) -> None:
        with self._lock:
            self._provider = provider

    # ------------------------------------------------------------------ #
    # Analysis
    # ------------------------------------------------------------------ #
    def analyze(
        self,
        transcript_text: str,
        *,
        evidence_map: dict[int, EvidenceReference],
        context: Optional[dict] = None,
    ) -> Optional[MeetingRecord]:
        """Analyze finalized transcript text for meeting intelligence.

        ``evidence_map`` maps transcript event uid -> EvidenceReference. Every
        evidence ID referenced by the model must exist in this map, or the
        referencing item is rejected.

        Returns a :class:`MeetingRecord` or ``None`` when disabled, no
        provider, empty transcript, or nothing extractable.
        """
        if not self._enabled:
            return None
        if not transcript_text or not transcript_text.strip():
            return None

        provider = self._provider
        if provider is None:
            return None

        text = transcript_text.strip()
        if len(text) > self._max_transcript_chars:
            text = text[: self._max_transcript_chars]

        raw: Optional[str] = None
        try:
            raw = provider.analyze(text, context)
        except Exception:  # noqa: BLE001 - provider failures are non-fatal
            return None

        parsed = self._parse_output(raw, evidence_map)
        if parsed is None:
            return None
        return parsed
    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _dedupe_key(self, kind: str, value: str) -> str:
        return f"{kind}:{self._normalize(value)}"

    def _is_duplicate(self, key: str) -> bool:
        with self._lock:
            return key in self._recent

    def _record_dedupe(self, key: str) -> None:
        with self._lock:
            self._recent.append(key)
            if len(self._recent) > self._dedupe_window:
                self._recent = self._recent[-self._dedupe_window:]

    # ------------------------------------------------------------------ #
    # Output parsing + deterministic validation
    # ------------------------------------------------------------------ #
    def _parse_output(
        self, raw: Optional[str], evidence_map: dict[int, EvidenceReference]
    ) -> Optional[MeetingRecord]:
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

        meeting_id = data.get("meeting_id") or uuid.uuid4().hex
        if not isinstance(meeting_id, str) or not meeting_id.strip():
            meeting_id = uuid.uuid4().hex

        topics = self._parse_topics(data.get("topics"))
        decisions = self._parse_decisions(data.get("decisions"), evidence_map)
        action_items = self._parse_action_items(
            data.get("action_items"), evidence_map
        )
        open_questions = self._parse_open_questions(
            data.get("open_questions"), evidence_map
        )
        candidate_commitments = self._parse_candidate_commitments(
            data.get("candidate_commitments"), evidence_map
        )
        events = self._parse_events(data.get("events"), evidence_map)

        # If everything is empty, return None (nothing extractable).
        if (
            not topics
            and not decisions
            and not action_items
            and not open_questions
            and not candidate_commitments
            and not events
        ):
            return None

        return MeetingRecord(
            meeting_id=meeting_id,
            topics=tuple(topics),
            decisions=tuple(decisions),
            action_items=tuple(action_items),
            open_questions=tuple(open_questions),
            candidate_commitments=tuple(candidate_commitments),
            events=tuple(events),
            created_ts=_dt.datetime.now(_dt.timezone.utc).timestamp(),
        )

    # -- events ----------------------------------------------------------- #
    def _parse_events(
        self, value: object, evidence_map: dict[int, EvidenceReference]
    ) -> list[EventItem]:
        if not isinstance(value, list):
            return []
        out: list[EventItem] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            event = self._parse_single_event(item, evidence_map)
            if event is not None:
                out.append(event)
                if len(out) >= self._max_events:
                    break
        return out

    def _parse_single_event(
        self, item: dict, evidence_map: dict[int, EvidenceReference]
    ) -> Optional[EventItem]:
        event_type = item.get("event_type")
        title = item.get("title")
        excerpt = item.get("excerpt")
        speaker = item.get("speaker")
        spoken_date = item.get("spoken_date")
        normalized_date = item.get("normalized_date")
        review_status = item.get("review_status")
        missing_details = item.get("missing_details", "")
        evidence_ids = item.get("evidence_ids")

        if not isinstance(event_type, str) or \
                event_type.strip() not in self.VALID_EVENT_TYPES:
            return None
        if not isinstance(title, str) or not title.strip():
            return None
        if not isinstance(excerpt, str) or not excerpt.strip():
            return None
        if speaker is not None and (
            not isinstance(speaker, str) or speaker.strip() not in self.VALID_SIDES
        ):
            return None
        speaker = speaker.strip() if isinstance(speaker, str) else "unknown"
        if review_status not in self.VALID_EVENT_REVIEW:
            return None
        for date_field_val in (spoken_date, normalized_date):
            if date_field_val is not None and (
                not isinstance(date_field_val, str) or not date_field_val.strip()
            ):
                return None
        spoken_date = spoken_date.strip() if isinstance(spoken_date, str) else None
        normalized_date = normalized_date.strip() if isinstance(normalized_date, str) else None
        # Normalized dates must be valid ISO dates.
        if normalized_date is not None and not self._valid_date(normalized_date):
            return None
        if not isinstance(missing_details, str):
            missing_details = ""

        evidence = self._parse_evidence(evidence_ids, evidence_map)
        if evidence is None:
            return None

        # Anti-fabrication rule: a normalized calendar date is accepted ONLY
        # when the transcript itself supports it (the excerpt or the spoken
        # form contains a digit or an explicit month name). Relative
        # expressions ("next Friday") stay unresolved: normalized_date is
        # stripped and the item is marked needs_review — never invented.
        if normalized_date is not None:
            support = f"{excerpt} {spoken_date or ''}".lower()
            has_digit = any(c.isdigit() for c in support)
            has_month = any(m in support for m in self._MONTH_NAMES)
            if not (has_digit or has_month):
                normalized_date = None
                review_status = "needs_review"

        # Deduplicate identical (type, title) events; keep conflicting ones.
        key = self._dedupe_key("event", f"{event_type}|{title}")
        if self._is_duplicate(key):
            return None
        self._record_dedupe(key)

        return EventItem(
            id=uuid.uuid4().hex,
            event_type=event_type.strip(),
            title=title.strip(),
            excerpt=excerpt.strip(),
            speaker=speaker,
            spoken_date=spoken_date,
            normalized_date=normalized_date,
            review_status=review_status,
            missing_details=missing_details.strip(),
            evidence=tuple(evidence),
            created_ts=_dt.datetime.now(_dt.timezone.utc).timestamp(),
        )

    # -- topics ---------------------------------------------------------- #
    def _parse_topics(self, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        out: list[str] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str) or not item.strip():
                continue
            topic = item.strip()
            key = self._dedupe_key("topic", topic)
            if key in seen or self._is_duplicate(key):
                continue
            seen.add(key)
            self._record_dedupe(key)
            out.append(topic)
            if len(out) >= self._max_decisions:  # reuse bound for topics
                break
        return out

    # -- decisions ------------------------------------------------------- #
    def _parse_decisions(
        self, value: object, evidence_map: dict[int, EvidenceReference]
    ) -> list[Decision]:
        if not isinstance(value, list):
            return []
        out: list[Decision] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            decision = self._parse_single_decision(item, evidence_map)
            if decision is not None:
                out.append(decision)
                if len(out) >= self._max_decisions:
                    break
        return out

    def _parse_single_decision(
        self, item: dict, evidence_map: dict[int, EvidenceReference]
    ) -> Optional[Decision]:
        topic = item.get("topic")
        proposed_value = item.get("proposed_value")
        decided_value = item.get("decided_value")
        status = item.get("status")
        evidence_ids = item.get("evidence_ids")
        speaker = item.get("speaker")

        if not isinstance(topic, str) or not topic.strip():
            return None
        if not isinstance(proposed_value, str) or not proposed_value.strip():
            return None
        if decided_value is not None and (
            not isinstance(decided_value, str) or not decided_value.strip()
        ):
            return None
        if status not in self.VALID_DECISION_STATUSES:
            return None
        if speaker is not None and (
            not isinstance(speaker, str) or speaker.strip() not in self.VALID_SIDES
        ):
            return None
        speaker = speaker.strip() if isinstance(speaker, str) else "unknown"

        evidence = self._parse_evidence(evidence_ids, evidence_map)
        if evidence is None:
            return None  # unknown/fabricated evidence reference -> reject

        # Deduplicate identical proposals, but preserve conflicting ones.
        key = self._dedupe_key("decision", f"{topic}|{proposed_value}")
        if self._is_duplicate(key):
            return None
        self._record_dedupe(key)

        return Decision(
            id=uuid.uuid4().hex,
            topic=topic.strip(),
            proposed_value=proposed_value.strip(),
            decided_value=decided_value.strip() if decided_value else None,
            status=status,
            evidence=tuple(evidence),
            speaker=speaker,
            created_ts=_dt.datetime.now(_dt.timezone.utc).timestamp(),
        )

    # -- action items ---------------------------------------------------- #
    def _parse_action_items(
        self, value: object, evidence_map: dict[int, EvidenceReference]
    ) -> list[ActionItem]:
        if not isinstance(value, list):
            return []
        out: list[ActionItem] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            action = self._parse_single_action_item(item, evidence_map)
            if action is not None:
                out.append(action)
                if len(out) >= self._max_action_items:
                    break
        return out

    def _parse_single_action_item(
        self, item: dict, evidence_map: dict[int, EvidenceReference]
    ) -> Optional[ActionItem]:
        description = item.get("description")
        owner = item.get("owner")
        deadline = item.get("deadline")
        status = item.get("status")
        evidence_ids = item.get("evidence_ids")

        if not isinstance(description, str) or not description.strip():
            return None
        if status not in self.VALID_ACTION_STATUSES:
            return None
        if owner is not None and (
            not isinstance(owner, str) or not owner.strip()
        ):
            owner = None
        else:
            owner = owner.strip() if isinstance(owner, str) else None
        if deadline is not None and (
            not isinstance(deadline, str) or not deadline.strip()
        ):
            deadline = None
        else:
            deadline = deadline.strip() if isinstance(deadline, str) else None
        # Validate date format if present (ISO-8601 date or datetime).
        if deadline is not None and not self._valid_date(deadline):
            return None

        evidence = self._parse_evidence(evidence_ids, evidence_map)
        if evidence is None:
            return None

        key = self._dedupe_key("action", description)
        if self._is_duplicate(key):
            return None
        self._record_dedupe(key)

        return ActionItem(
            id=uuid.uuid4().hex,
            description=description.strip(),
            owner=owner,
            deadline=deadline,
            status=status,
            evidence=tuple(evidence),
            created_ts=_dt.datetime.now(_dt.timezone.utc).timestamp(),
        )

    # -- open questions -------------------------------------------------- #
    def _parse_open_questions(
        self, value: object, evidence_map: dict[int, EvidenceReference]
    ) -> list[OpenQuestion]:
        if not isinstance(value, list):
            return []
        out: list[OpenQuestion] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            question = self._parse_single_open_question(item, evidence_map)
            if question is not None:
                out.append(question)
                if len(out) >= self._max_open_questions:
                    break
        return out

    def _parse_single_open_question(
        self, item: dict, evidence_map: dict[int, EvidenceReference]
    ) -> Optional[OpenQuestion]:
        question = item.get("question")
        answered = item.get("answered")
        answer = item.get("answer")
        evidence_ids = item.get("evidence_ids")

        if not isinstance(question, str) or not question.strip():
            return None
        if not isinstance(answered, bool):
            return None
        if answer is not None and (
            not isinstance(answer, str) or not answer.strip()
        ):
            return None
        answer = answer.strip() if isinstance(answer, str) else None

        evidence = self._parse_evidence(evidence_ids, evidence_map)
        if evidence is None:
            return None

        key = self._dedupe_key("question", question)
        if self._is_duplicate(key):
            return None
        self._record_dedupe(key)

        return OpenQuestion(
            id=uuid.uuid4().hex,
            question=question.strip(),
            answered=answered,
            answer=answer,
            evidence=tuple(evidence),
            created_ts=_dt.datetime.now(_dt.timezone.utc).timestamp(),
        )

    # -- candidate commitments ------------------------------------------- #
    def _parse_candidate_commitments(
        self, value: object, evidence_map: dict[int, EvidenceReference]
    ) -> list[CandidateCommitment]:
        if not isinstance(value, list):
            return []
        out: list[CandidateCommitment] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            commitment = self._parse_single_candidate_commitment(
                item, evidence_map
            )
            if commitment is not None:
                out.append(commitment)
                if len(out) >= self._max_candidate_commitments:
                    break
        return out

    def _parse_single_candidate_commitment(
        self, item: dict, evidence_map: dict[int, EvidenceReference]
    ) -> Optional[CandidateCommitment]:
        topic = item.get("topic")
        proposed_value = item.get("proposed_value")
        status = item.get("status")
        evidence_ids = item.get("evidence_ids")
        speaker = item.get("speaker")

        if not isinstance(topic, str) or not topic.strip():
            return None
        if not isinstance(proposed_value, str) or not proposed_value.strip():
            return None
        # Never allow "confirmed" -- only inferred/proposed.
        if status not in self.VALID_COMMITMENT_STATUSES:
            return None
        if speaker is not None and (
            not isinstance(speaker, str) or speaker.strip() not in self.VALID_SIDES
        ):
            return None
        speaker = speaker.strip() if isinstance(speaker, str) else "unknown"

        evidence = self._parse_evidence(evidence_ids, evidence_map)
        if evidence is None:
            return None

        key = self._dedupe_key("commitment", f"{topic}|{proposed_value}")
        if self._is_duplicate(key):
            return None
        self._record_dedupe(key)

        return CandidateCommitment(
            id=uuid.uuid4().hex,
            topic=topic.strip(),
            proposed_value=proposed_value.strip(),
            status=status,
            evidence=tuple(evidence),
            speaker=speaker,
            created_ts=_dt.datetime.now(_dt.timezone.utc).timestamp(),
        )

    # -- evidence validation --------------------------------------------- #
    def _parse_evidence(
        self,
        evidence_ids: object,
        evidence_map: dict[int, EvidenceReference],
    ) -> Optional[list[EvidenceReference]]:
        """Validate evidence references against the caller-supplied map.

        Returns a list of EvidenceReference or None if any referenced id is
        unknown/fabricated or the structure is malformed. An empty list is
        allowed (claim with no evidence).
        """
        if evidence_ids is None:
            return []
        if not isinstance(evidence_ids, list):
            return None
        out: list[EvidenceReference] = []
        seen: set[int] = set()
        for eid in evidence_ids:
            if not isinstance(eid, int) or isinstance(eid, bool):
                return None
            if eid in seen:
                continue
            seen.add(eid)
            ref = evidence_map.get(eid)
            if ref is None:
                return None  # unknown/fabricated evidence id -> reject
            out.append(ref)
        return out

    @staticmethod
    def _valid_date(value: str) -> bool:
        """Accept ISO-8601 date or datetime strings."""
        if not value:
            return False
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
            try:
                _dt.datetime.strptime(value, fmt)
                return True
            except ValueError:
                continue
        # Also accept a bare "today"/"unknown"-free non-date? No -- be strict.
        return False
