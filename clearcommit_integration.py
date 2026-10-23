"""ClearCommit integration module (Stage 6 / hardened in Stage 6B).

Wires the AI service, coaching analyzers, meeting intelligence, and agreement
ledger into the live translation pipeline without modifying VAD, STT,
translation, TTS, caption matching, or audio capture.

The integration is purely additive:
    * ``StreamingInterpreter`` accepts an optional ``on_final_utterance``
      callback (default None = no behavior change).
    * The callback enqueues work to bounded AI services; it never blocks the
      capture/translate/TTS pipeline. Coaching AND meeting analysis both run
      on dedicated bounded worker services (never on the capture thread).
    * All model output is validated before display; failures are swallowed.
    * Coaching is controlled by ``ModeConfig`` (disabled by default).

Stage 6B hardening:
    * Fix 1 (Tk safety): the UI callback contract requires the callback to be
      thread-safe. The GUI registers a marshalled callback
      (``root.after``/UI-queue based); worker threads never touch Tk widgets.
    * Fix 2 (evidence binding): model-generated provenance (evidence ids,
      side/speaker, original text) is UNTRUSTED. Coaching results are bound to
      the originating request uid and the authoritative evidence record;
      provenance fields are populated from trusted data. Unknown, evicted, or
      malformed results are rejected and never displayed as evidence-grounded.
    * Fix 3 (async meeting analysis): meeting analysis runs on its own bounded
      service (concurrency 1, queue 4). Submission from the capture callback
      returns promptly; overload drops analysis instead of blocking audio.
      Stale results (older uid) can never overwrite newer records.
    * Fix 4 (bounded memory): ``_evidence_map`` and ``_transcript_parts`` are
      evicted in lockstep (oldest first) so the transcript window always
      exactly mirrors the retained evidence. Conflicting statements are never
      merged.
    * Fix 5 (shutdown guard): a ``_closed`` flag set by ``close()`` makes
      every entry point a no-op afterwards; close is idempotent and bounded.
      Note: a provider thread that is already stuck cannot be forcibly
      stopped (Python limitation); the per-service circuit breaker bounds how
      many stuck threads can accumulate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Optional
import json
import logging
import threading
import time
import uuid

from ai_providers import ProviderConfig, get_provider_for_config
from ai_service import AIRequest, AIRequestStatus, AIResult, AIService
from agreement_ledger import AgreementLedger, PARTICIPANT_SOURCE, TranscriptEvidence
from assistant import ConversationAssistant
from meeting_intelligence import (
    EvidenceReference,
    MeetingIntelligence,
    MeetingRecord,
)
from mode_config import ModeConfig
from professional_coaching import ProfessionalCoach
from social_coaching import SocialCoach

logger = logging.getLogger("clearcommit_integration")

#: Coaching suggestion kinds this integration accepts for display. Anything
#: else in model output is rejected (deterministic validation).
KNOWN_COACHING_KINDS = frozenset({
    "idiom_explanation",
    "tone_suggestion",
    "communication_coach",
    "pitch_coach",
})


@dataclass
class CoachingUIResult:
    """A coaching suggestion formatted for the UI.

    Provenance fields (``evidence_uid``, ``side``, ``original``) are populated
    exclusively from authoritative request/evidence data — never from model
    output.
    """

    kind: str
    original: str
    suggestion: str
    reason: str
    evidence_uid: int
    side: str
    mode: str


@dataclass
class MeetingUIResult:
    """A meeting-intelligence record formatted for the UI."""

    meeting_id: str
    topics: list[str]
    decisions: list[dict]
    action_items: list[dict]
    open_questions: list[dict]
    candidate_commitments: list[dict]
    events: list[dict] = field(default_factory=list)


class _CoachDispatcher:
    """AIProvider-compatible dispatcher for social/professional coaching.

    Routes each request to the appropriate coach based on the communication
    mode. ``config_getter`` returns the CURRENT ModeConfig so runtime mode
    switches are always observed (no stale config capture). Coaches return
    None when no real provider is connected, so this is safe without keys.
    """

    def __init__(
        self,
        social_coach: SocialCoach,
        professional_coach: ProfessionalCoach,
        config_getter: Callable[[], ModeConfig],
        context_getter: Optional[Callable[[], dict]] = None,
    ) -> None:
        self._social = social_coach
        self._professional = professional_coach
        self._config_getter = config_getter
        self._context_getter = context_getter

    def analyze(self, request: AIRequest, context: Optional[dict] = None) -> Optional[str]:
        """Route to the appropriate coach and return a JSON result string."""
        mode_config = self._config_getter()
        if not mode_config.coaching_active():
            return None

        ctx = dict(context or {})
        if self._context_getter is not None:
            try:
                ctx.update(self._context_getter() or {})
            except Exception:  # noqa: BLE001 - context must never break routing
                pass

        if mode_config.is_professional():
            suggestion = self._professional.analyze(
                request.text, side=request.side, uid=request.uid, context=ctx,
            )
        else:
            suggestion = self._social.analyze(
                request.text, side=request.side, uid=request.uid, context=ctx,
            )

        if suggestion is None:
            return None

        # Only coaching CONTENT is taken from the model-side analyzer result.
        # Provenance is re-bound to the authoritative request uid downstream.
        return json.dumps(
            {
                "kind": suggestion.kind,
                "suggestion": suggestion.suggestion,
                "reason": suggestion.reason,
            }
        )


class _MeetingAnalysisProvider:
    """AIProvider-compatible adapter that runs meeting analysis off-thread.

    Runs inside the dedicated meeting ``AIService`` worker (concurrency 1).
    Takes a coherent snapshot of the bounded evidence map + transcript window
    under the meeting lock, then delegates to the (provider-free by default)
    analyzer. Serializes the validated record as JSON for result delivery.
    """

    def __init__(self, integration: "ClearCommitIntegration") -> None:
        self._integration = integration

    def analyze(self, request: AIRequest, context: Optional[dict] = None) -> str:
        integration = self._integration
        with integration._meeting_lock:
            evidence_snapshot = dict(integration._evidence_map)
            transcript_text = " ".join(integration._transcript_parts)
        # Keep the most recent window bounded (the analyzer re-bounds too, but
        # we keep the TAIL so the newest context is analyzed).
        if len(transcript_text) > integration._max_transcript_chars:
            transcript_text = transcript_text[-integration._max_transcript_chars:]

        # Build the meeting schema context so the real provider knows what
        # JSON to return. Without this, OpenAICompatibleProvider.analyze()
        # would send raw transcript without the schema → unparseable response.
        provider_ctx = integration._ai_provider_context()
        provider_ctx["meeting_schema"] = (
            "Respond ONLY with valid JSON, no other text:\n"
            '{\n'
            '  "topics": ["..."],\n'
            '  "decisions": [{"topic": "...", "proposed_value": "...", '
            '"decided_value": null, "status": "proposed|decided|rejected", '
            '"evidence_ids": [int], "speaker": "p1|you"}],\n'
            '  "action_items": [{"description": "...", "owner": "p1|you|null", '
            '"deadline": "YYYY-MM-DD|null", "status": "assigned|unassigned", '
            '"evidence_ids": [int]}],\n'
            '  "open_questions": [{"question": "...", "answered": bool, '
            '"answer": "...|null", "evidence_ids": [int]}],\n'
            '  "candidate_commitments": [{"topic": "...", "proposed_value": '
            '"...", "status": "inferred|proposed", "evidence_ids": [int], '
            '"speaker": "p1|you"}],\n'
            '  "events": [{"event_type": "meeting|appointment|deadline|'
            'milestone|follow_up|risk|blocker", "title": "...", '
            '"excerpt": "...", "speaker": "p1|you", "spoken_date": "...|null", '
            '"normalized_date": "YYYY-MM-DD|null", "review_status": '
            '"explicit|inferred|needs_review", "missing_details": "...", '
            '"evidence_ids": [int]}]\n'
            '}\n'
        )

        record = integration.meeting_intelligence.analyze(
            transcript_text, evidence_map=evidence_snapshot,
            context=provider_ctx,
        )
        if record is None:
            return ""  # nothing extractable; caller keeps the last record
        return json.dumps(asdict(record))


class _AssistantProviderAdapter:
    """AIProvider-compatible adapter that runs grounded Q&A off-thread.

    Runs inside the dedicated assistant ``AIService`` worker. Reads a
    coherent snapshot of the bounded evidence window, meeting record, and
    agreement summary under the integration locks, then delegates to the
    (provider-free by default) ConversationAssistant. Serializes the
    validated reply as JSON for result delivery.
    """

    def __init__(self, integration: "ClearCommitIntegration") -> None:
        self._integration = integration

    def analyze(self, request: AIRequest, context: Optional[dict] = None) -> str:
        integration = self._integration
        with integration._meeting_lock:
            evidence_snapshot = dict(integration._evidence_map)
        meeting_summary = integration._assistant_meeting_summary()
        ledger_summary = integration._assistant_ledger_summary()

        reply = integration._assistant.ask(
            request.text,
            evidence_map=evidence_snapshot,
            meeting_summary=meeting_summary,
            ledger_summary=ledger_summary,
        )
        return json.dumps({
            "ok": reply.ok,
            "reply": reply.reply,
            "error": reply.error,
            "cited_uids": list(reply.cited_uids),
            "context_uids": list(reply.context_uids),
        })


class ClearCommitIntegration:
    """Glues ClearCommit features into the live interpreter.

    Thread-safe. Result delivery is through the caller-supplied UI callback,
    which MUST be thread-safe (the GUI registers a marshalled callable via
    its existing UI queue / ``root.after`` pattern). This class never touches
    Tk itself.
    """

    #: Maximum utterances retained for meeting analysis (oldest evicted).
    MAX_EVIDENCE_ENTRIES = 40

    def __init__(
        self,
        mode_config: ModeConfig,
        *,
        ai_max_concurrency: int = 1,
        ai_max_queue: int = 8,
        ai_default_timeout_s: float = 3.0,
        meeting_max_queue: int = 4,
        meeting_default_timeout_s: float = 5.0,
        assistant_max_queue: int = 2,
        assistant_default_timeout_s: float = 12.0,
        max_transcript_chars: int = 2000,
    ) -> None:
        self.mode_config = mode_config
        self._ledger = AgreementLedger()
        self._ledger_lock = threading.RLock()
        self._agreement_version = 0
        self._closed = False

        # Coaches (providers injectable later; None = no real AI calls).
        self.social_coach = SocialCoach(enabled=False, provider=None)
        self.professional_coach = ProfessionalCoach(enabled=False, provider=None)
        self.meeting_intelligence = MeetingIntelligence(
            enabled=False, provider=None,
        )

        # AI service for async coaching (bounded, non-blocking).
        self._ai_service = AIService(
            max_concurrency=ai_max_concurrency,
            max_queue=ai_max_queue,
            default_timeout_s=ai_default_timeout_s,
        )
        self._ai_service.set_provider(
            _CoachDispatcher(
                self.social_coach, self.professional_coach,
                config_getter=lambda: self.mode_config,
                context_getter=self._ai_provider_context,
            )
        )
        self._ai_service.set_enabled(mode_config.coaching_active())

        # Dedicated bounded service for meeting analysis (Fix 3): never runs
        # on the capture thread; overload drops analysis instead of blocking.
        self._meeting_service = AIService(
            max_concurrency=1,
            max_queue=meeting_max_queue,
            default_timeout_s=meeting_default_timeout_s,
        )
        self._meeting_service.set_provider(_MeetingAnalysisProvider(self))
        self._meeting_service.set_enabled(mode_config.coaching_active())

        # Grounded conversation assistant (Stage A): user-triggered Q&A on its
        # own bounded service; provider-free by default.
        self._assistant = ConversationAssistant(provider=None)
        self._ai_config = None       # AIModelSettings (set via update_ai_config)
        self._prompt_overrides: dict[str, str] = {}
        self._provider_config: Optional[ProviderConfig] = None

        # Per-service health tracking (Fix 2: visible degradation).
        self._service_health: dict[str, str] = {
            "stt": "unknown", "translate": "unknown", "tts": "unknown",
        }
        # Language mismatch detection (Fix 1).
        self._lang_mismatch_count = 0
        self._lang_mismatch_lang = None
        self._assistant_service = AIService(
            max_concurrency=1,
            max_queue=assistant_max_queue,
            default_timeout_s=assistant_default_timeout_s,
        )
        self._assistant_service.set_provider(_AssistantProviderAdapter(self))
        self._assistant_service.set_enabled(mode_config.coaching_active())
        self._assistant_questions: dict[str, str] = {}   # request id -> question
        self._assistant_busy = 0
        self._assistant_lock = threading.RLock()

        # Meeting intelligence accumulator. ``_transcript_parts`` mirrors
        # ``_evidence_map`` 1:1 (same insertion order) so the transcript
        # window ALWAYS exactly matches the retained evidence (Fix 4).
        self._evidence_map: dict[int, EvidenceReference] = {}
        self._transcript_parts: list[str] = []
        self._meeting_lock = threading.RLock()
        self._max_transcript_chars = int(max_transcript_chars)

        # Results for the GUI. ``_last_meeting_uid`` guards against stale
        # meeting results overwriting newer records (Fix 3).
        self._coaching_results: dict[int, CoachingUIResult] = {}
        self._meeting_record: Optional[MeetingUIResult] = None
        self._assistant_reply: Optional[dict] = None
        self._last_meeting_uid = 0
        self._results_lock = threading.RLock()

        # UI callback (set by the GUI thread; must be thread-safe).
        self._ui_callback: Optional[Callable[[], None]] = None

        self._lock = threading.RLock()
        self.update_mode_config(mode_config)

    # ------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------ #
    def set_ui_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """Set a callback invoked when new results are available.

        The callback is invoked from worker/capture threads and MUST be
        thread-safe. The GUI must register a marshalled callable (e.g. one
        that enqueues onto its UI queue for the Tk main thread) — never one
        that touches Tk widgets directly.
        """
        with self._lock:
            self._ui_callback = callback

    def is_enabled(self) -> bool:
        """True when AI analysis is active (coaching enabled + service running)."""
        return self._ai_service.is_enabled()

    def get_service_health(self) -> dict[str, str]:
        """Per-service health: stt/translate/tts each 'ok'|'error'|'unknown'."""
        return dict(self._service_health)

    def report_service_health(self, service: str, ok: bool) -> None:
        """Record a per-service call outcome (called from the pipeline)."""
        if service in self._service_health:
            self._service_health[service] = "ok" if ok else "error"

    def detect_language_mismatch(self, uid: int, detected_lang: str,
                                 configured_lang: str) -> Optional[str]:
        """Track consecutive language mismatches; returns a suggestion when
        the detected language has differed from the configured one for 3+
        consecutive utterances. The caller decides what to do with the hint.

        Reset on a match so a single odd detection doesn't trigger it.
        """
        if detected_lang and configured_lang and \
                detected_lang.lower() != configured_lang.lower():
            self._lang_mismatch_count += 1
            self._lang_mismatch_lang = detected_lang
            if self._lang_mismatch_count >= 3:
                suggestion = (
                    f"Whisper detected {detected_lang} for {self._lang_mismatch_count} "
                    f"consecutive utterances, but the configured language is "
                    f"{configured_lang}. Consider changing the language dropdown."
                )
                return suggestion
        else:
            self._lang_mismatch_count = 0
            self._lang_mismatch_lang = None
        return None

    def update_mode_config(self, mode_config: ModeConfig) -> None:
        """Update mode/coaching config at runtime."""
        with self._lock:
            if self._closed:
                return
            self.mode_config = mode_config
            self._ai_service.set_enabled(mode_config.coaching_active())
            self._meeting_service.set_enabled(mode_config.coaching_active())
            self._assistant_service.set_enabled(mode_config.coaching_active())
            self.meeting_intelligence.set_enabled(mode_config.coaching_active())
            self.social_coach.set_enabled(mode_config.coaching_active())
            self.professional_coach.set_enabled(mode_config.coaching_active())

    def update_ai_config(self, ai_config, prompt_overrides: dict[str, str],
                         provider_config=None) -> None:
        """Apply model settings, coach option toggles, system-prompt
        overrides, and provider config at runtime.

        When a real provider is configured (preset + API key), it replaces
        the None providers on all four AI features simultaneously. When
        provider_config is None or unconfigured, providers stay None
        (features remain inert but the translator is unaffected).
        """
        with self._lock:
            if self._closed:
                return
            self._ai_config = ai_config
            self._prompt_overrides = dict(prompt_overrides)
            if provider_config is not None:
                self._provider_config = provider_config

        self.social_coach.set_options(
            idioms=ai_config.coach_idioms, tone=ai_config.coach_tone)
        self.professional_coach.set_options(
            communication=ai_config.coach_communication,
            pitch=ai_config.coach_pitch)
        self.social_coach.set_prompts(
            idiom=prompt_overrides.get("coach_idiom"),
            tone=prompt_overrides.get("coach_tone"))
        self.professional_coach.set_prompts(
            communication=prompt_overrides.get("coach_communication"),
            pitch=prompt_overrides.get("coach_pitch"))
        self._assistant.set_system_prompt(prompt_overrides.get("assistant"))

        # Real provider: connect or disconnect all features simultaneously.
        prov_config = self._provider_config
        if prov_config is not None and prov_config.is_configured():
            try:
                real = get_provider_for_config(prov_config)
                self.social_coach.set_provider(real)
                self.professional_coach.set_provider(real)
                self._assistant.set_provider(real)
                self.meeting_intelligence.set_provider(real)
                logger.info("AI provider connected: %s (%s)",
                            prov_config.preset, prov_config.resolve_model())
            except Exception:  # noqa: BLE001
                logger.exception("Provider connection failed")
        else:
            self.social_coach.set_provider(None)
            self.professional_coach.set_provider(None)
            self._assistant.set_provider(None)
            self.meeting_intelligence.set_provider(None)

    def _ai_provider_context(self) -> dict:
        """Provider context consumed by real backends when connected."""
        cfg = self._ai_config
        return cfg.context() if cfg is not None else {}

    # ------------------------------------------------------------------ #
    # Pipeline callback (called from capture thread — must be non-blocking)
    # ------------------------------------------------------------------ #
    def on_final_utterance(
        self,
        *,
        side: str,
        source_lang: str,
        target_lang: str,
        original_text: str,
        translated_text: str,
        uid: int,
        accept_ts: Optional[float],
    ) -> None:
        """Handle a finalized utterance from the pipeline.

        Non-blocking: updates the bounded evidence store and enqueues
        coaching + meeting work to dedicated services. Never throws.
        """
        if self._closed:
            return
        try:
            self._handle_utterance(
                side=side,
                source_lang=source_lang,
                target_lang=target_lang,
                original_text=original_text,
                translated_text=translated_text,
                uid=uid,
                accept_ts=accept_ts,
            )
        except Exception:  # noqa: BLE001 - never let integration affect pipeline
            logger.exception("ClearCommit integration callback failed")

    def _handle_utterance(
        self,
        *,
        side: str,
        source_lang: str,
        target_lang: str,
        original_text: str,
        translated_text: str,
        uid: int,
        accept_ts: Optional[float],
    ) -> None:
        ts = accept_ts if accept_ts is not None else 0.0

        # 1. Update the bounded evidence store (Fix 4). ``_transcript_parts``
        #    mirrors ``_evidence_map`` in lockstep so the analysis window is
        #    always coherent with the retained evidence.
        ref = EvidenceReference(
            uid=uid, side=side, text=original_text, timestamp=ts,
        )
        with self._meeting_lock:
            self._evidence_map[uid] = ref
            self._transcript_parts.append(original_text)
            while len(self._evidence_map) > self.MAX_EVIDENCE_ENTRIES:
                oldest_uid = next(iter(self._evidence_map))
                self._evidence_map.pop(oldest_uid, None)
                if self._transcript_parts:
                    self._transcript_parts.pop(0)

        # 2. Enqueue meeting analysis to its dedicated bounded service
        #    (Fix 3). Submission returns promptly; overload drops analysis.
        if self.mode_config.coaching_active():
            meeting_request = AIRequest(
                id=str(uid),
                text=original_text,  # informational; provider reads snapshot
                side=side,
                uid=uid,
                mode=self.mode_config.mode,
                timestamp=ts,
            )
            try:
                handle = self._meeting_service.submit(meeting_request)
                if handle.status == AIRequestStatus.PENDING:
                    self._meeting_service.on_complete(
                        handle.request_id, self.on_meeting_result
                    )
            except Exception:  # noqa: BLE001
                logger.exception("Meeting analysis submit failed")

        # 3. Enqueue coaching to the AI service (async, non-blocking).
        if not self.mode_config.coaching_active():
            return
        request = AIRequest(
            id=str(uid),
            text=original_text,
            side=side,
            uid=uid,
            mode=self.mode_config.mode,
            timestamp=ts,
        )
        try:
            handle = self._ai_service.submit(request, timeout_s=3.0)
            # Register the result callback ONLY for accepted requests:
            # rejected submissions never complete, and registering a callback
            # for them would leak it forever.
            if handle.status == AIRequestStatus.PENDING:
                self._ai_service.on_complete(handle.request_id, self.on_ai_result)
        except Exception:  # noqa: BLE001
            logger.exception("AI service submit failed")

    # ------------------------------------------------------------------ #
    # AI service result delivery (worker threads — no Tk access here)
    # ------------------------------------------------------------------ #
    def on_ai_result(self, result) -> None:
        """Called by AIService when a coaching result is ready.

        Fix 2: provenance is bound authoritatively. The model's evidence id,
        side, and original text (if any) are IGNORED; the result is bound to
        the originating request uid and its authoritative evidence record.
        Unknown/evicted requests and malformed payloads are rejected.
        """
        if self._closed:
            return
        if result.status != AIRequestStatus.COMPLETED or result.output is None:
            return
        try:
            uid = int(result.request_id)
        except (TypeError, ValueError):
            return
        try:
            data = json.loads(result.output)
        except (json.JSONDecodeError, ValueError):
            return
        if not isinstance(data, dict):
            return

        kind = data.get("kind")
        if kind not in KNOWN_COACHING_KINDS:
            return
        suggestion_text = data.get("suggestion")
        reason = data.get("reason", "")
        if not isinstance(suggestion_text, str) or not suggestion_text.strip():
            return
        if not isinstance(reason, str):
            reason = ""

        # Authoritative binding: the evidence record must still be retained.
        with self._meeting_lock:
            ref = self._evidence_map.get(uid)
        if ref is None:
            # Unknown or evicted request -> reject; never display as
            # evidence-grounded.
            logger.info("Coaching result rejected: no evidence for uid=%s", uid)
            return

        suggestion = CoachingUIResult(
            kind=kind,
            original=ref.text,          # trusted: from the transcript event
            suggestion=suggestion_text,
            reason=reason,
            evidence_uid=ref.uid,       # trusted: request/evidence uid
            side=ref.side,              # trusted: capture-side identity
            mode=self.mode_config.mode,
        )
        with self._results_lock:
            self._coaching_results[uid] = suggestion
            self._notify_ui()

    def on_meeting_result(self, result) -> None:
        """Called by the meeting service when an analysis result is ready.

        Stale results (older uid than the last stored record) can never
        overwrite newer records (Fix 3).
        """
        if self._closed:
            return
        if result.status != AIRequestStatus.COMPLETED or not result.output:
            return
        try:
            uid = int(result.request_id)
        except (TypeError, ValueError):
            return
        try:
            data = json.loads(result.output)
        except (json.JSONDecodeError, ValueError):
            return
        meeting = self._meeting_ui_result_from_dict(data)
        if meeting is None:
            return

        with self._results_lock:
            if uid <= self._last_meeting_uid:
                return  # stale; never overwrite newer records
            self._last_meeting_uid = uid
            self._meeting_record = meeting
            self._notify_ui()

        # Stage 7: ingest validated candidate commitments into the agreement
        # ledger as 'proposed' items. Confirmation is human-only (see
        # confirm_commitment); the model can never confirm.
        self._ingest_candidate_commitments(meeting)

    # ------------------------------------------------------------------ #
    # Queries for the GUI
    # ------------------------------------------------------------------ #
    def take_coaching_results(self) -> list[CoachingUIResult]:
        """Drain and return all pending coaching results (GUI thread only)."""
        with self._results_lock:
            results = list(self._coaching_results.values())
            self._coaching_results.clear()
        return results

    def take_meeting_record(self) -> Optional[MeetingUIResult]:
        """Drain and return the latest meeting record (GUI thread only)."""
        with self._results_lock:
            record = self._meeting_record
            self._meeting_record = None
        return record

    # ------------------------------------------------------------------ #
    # Grounded conversation assistant (Stage A)
    # ------------------------------------------------------------------ #
    def _assistant_meeting_summary(self) -> str:
        """Compact validated meeting summary for the assistant context."""
        try:
            record = self.meeting_intelligence
            lines: list[str] = []
            if record is None:
                return ""
            # The latest validated record is rebuilt from the analyzer's own
            # output at analysis time; summarize from the last stored record.
            with self._results_lock:
                ui = self._meeting_record
            if ui is None:
                return ""
            for d in ui.decisions:
                lines.append(
                    f"decision[{d.get('status','?')}]: {d.get('topic','')} = "
                    f"{d.get('proposed_value','')}"
                    + (f" -> {d.get('decided_value')}" if d.get('decided_value') else "")
                )
            for a in ui.action_items:
                owner = a.get("owner") or "unassigned"
                deadline = a.get("deadline") or "unknown"
                lines.append(
                    f"action[{a.get('status','?')}]: {a.get('description','')}"
                    f" (owner={owner}, deadline={deadline})"
                )
            for q in ui.open_questions:
                state = "answered" if q.get("answered") else "unresolved"
                lines.append(f"question[{state}]: {q.get('question','')}")
            for e in getattr(ui, "events", []) or []:
                nd = e.get("normalized_date") or "unresolved"
                lines.append(
                    f"event[{e.get('event_type','?')}|{e.get('review_status','?')}]: "
                    f"{e.get('title','')} (when spoken: {e.get('spoken_date') or 'none'}; "
                    f"resolved: {nd})"
                )
            return "\n".join(lines[:40])
        except Exception:  # noqa: BLE001 - summary must never break requests
            return ""

    def _assistant_ledger_summary(self) -> str:
        """Compact agreement-ledger summary (facts only, never confirmation)."""
        try:
            _version, items = self.get_agreement_snapshot()
            if not items:
                return "no commitments tracked"
            lines = []
            for it in items[:20]:
                lines.append(
                    f"{it['status']}: {it['topic']} = {it['value']} "
                    f"(confirmed by: {', '.join(it['confirmed_by']) or 'none'})"
                )
            return "\n".join(lines)
        except Exception:  # noqa: BLE001
            return ""

    def ask_assistant(self, question: str) -> AIResult:
        """Submit a user question (non-blocking). GUI thread only.

        Returns the immediate handle: PENDING when accepted, FAILED with an
        explanatory error when rejected (disabled/closed/queue full/empty).
        """
        question = (question or "").strip()
        if not question:
            return AIResult(
                request_id="", status=AIRequestStatus.FAILED,
                error="empty_question",
            )
        handle = self._assistant_service.submit(
            AIRequest(
                id=uuid.uuid4().hex,
                text=question,
                side="you",
                uid=0,
                mode=self.mode_config.mode,
                timestamp=time.time(),
            ),
        )
        if handle.status == AIRequestStatus.PENDING:
            with self._assistant_lock:
                if self._closed:
                    return AIResult(
                        request_id=handle.request_id,
                        status=AIRequestStatus.FAILED,
                        error="service_closed",
                    )
                self._assistant_questions[handle.request_id] = question
                self._assistant_busy += 1
            self._assistant_service.on_complete(
                handle.request_id, self.on_assistant_result
            )
        return handle

    def on_assistant_result(self, result) -> None:
        """Called by the assistant service when an answer is ready."""
        question = ""
        with self._assistant_lock:
            question = self._assistant_questions.pop(result.request_id, "")
            self._assistant_busy = max(0, self._assistant_busy - 1)
        if self._closed:
            return
        if result.status != AIRequestStatus.COMPLETED or result.output is None:
            # Timeouts/cancellation produce an honest error reply for the UI.
            if question:
                with self._results_lock:
                    self._assistant_reply = {
                        "ok": False,
                        "question": question,
                        "reply": "",
                        "error": result.error or "assistant_failed",
                        "cited_uids": [],
                        "context_uids": [],
                    }
                    self._notify_ui()
            return
        try:
            data = json.loads(result.output)
            if not isinstance(data, dict):
                raise ValueError("not a dict")
            payload = {
                "ok": bool(data.get("ok")),
                "question": question,
                "reply": str(data.get("reply", "")),
                "error": data.get("error"),
                "cited_uids": [int(u) for u in data.get("cited_uids", [])
                               if isinstance(u, int)],
                "context_uids": [int(u) for u in data.get("context_uids", [])
                                 if isinstance(u, int)],
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            payload = {
                "ok": False, "question": question, "reply": "",
                "error": "malformed_reply", "cited_uids": [],
                "context_uids": [],
            }
        with self._results_lock:
            self._assistant_reply = payload
            self._notify_ui()

    def take_assistant_reply(self) -> Optional[dict]:
        """Drain the latest assistant reply (GUI thread only)."""
        with self._results_lock:
            reply = self._assistant_reply
            self._assistant_reply = None
        return reply

    def is_assistant_busy(self) -> bool:
        with self._assistant_lock:
            return self._assistant_busy > 0

    def clear_assistant_history(self) -> None:
        self._assistant.clear_history()

    @staticmethod
    def _meeting_ui_result_from_dict(data: object) -> Optional[MeetingUIResult]:
        """Validate + convert an internal record payload to a UI result."""
        if not isinstance(data, dict):
            return None
        try:
            meeting_id = str(data.get("meeting_id") or "")
            topics = data.get("topics")
            decisions = data.get("decisions")
            action_items = data.get("action_items")
            open_questions = data.get("open_questions")
            candidate_commitments = data.get("candidate_commitments")
            events = data.get("events", [])
            for collection in (topics, decisions, action_items,
                               open_questions, candidate_commitments, events):
                if not isinstance(collection, list):
                    return None
            return MeetingUIResult(
                meeting_id=meeting_id,
                topics=[str(t) for t in topics],
                decisions=[d for d in decisions if isinstance(d, dict)],
                action_items=[a for a in action_items if isinstance(a, dict)],
                open_questions=[q for q in open_questions if isinstance(q, dict)],
                candidate_commitments=[
                    c for c in candidate_commitments if isinstance(c, dict)
                ],
                events=[e for e in events if isinstance(e, dict)],
            )
        except Exception:  # noqa: BLE001 - defensive; never raise into worker
            return None

    # ------------------------------------------------------------------ #
    # Agreement ledger (Stage 7): ingestion + human-only confirmation
    # ------------------------------------------------------------------ #
    def _ingest_candidate_commitments(self, meeting: MeetingUIResult) -> None:
        """Add validated candidate commitments to the ledger as 'proposed'.

        Safety: items are ingested as ``proposed`` ONLY. Nothing here or in
        the model path can confirm an item — confirmation happens exclusively
        through :meth:`confirm_commitment` (explicit human action). Duplicate
        (topic, value) pairs with a live item are skipped so repeated
        transcript windows cannot spam the panel. Conflicting proposals
        (different values for the same topic) are kept as separate items.
        """
        ingested = False
        for c in meeting.candidate_commitments:
            topic = c.get("topic")
            value = c.get("proposed_value")
            if not isinstance(topic, str) or not topic.strip():
                continue
            if not isinstance(value, str) or not value.strip():
                continue
            speaker = c.get("speaker")
            speaker = speaker if speaker in ("p1", "you") else "unknown"

            # Evidence dicts were validated by the analyzer against the
            # evidence snapshot at analysis time — trusted provenance.
            evidence: list[TranscriptEvidence] = []
            for e in c.get("evidence") or []:
                if not isinstance(e, dict):
                    continue
                try:
                    evidence.append(TranscriptEvidence(
                        uid=int(e["uid"]),
                        side=str(e.get("side", "")),
                        original=str(e.get("text", "")),
                        translated="",
                        accept_ts=float(e.get("timestamp", 0.0)),
                        source_lang="", target_lang="",
                    ))
                except (KeyError, TypeError, ValueError):
                    continue

            key = (topic.strip().lower(), value.strip().lower())
            with self._ledger_lock:
                if self._closed:
                    return
                if self._has_live_item(key):
                    continue
                self._ledger.add_item(
                    topic=topic.strip(),
                    proposed_value=value.strip(),
                    speaker=speaker,
                    evidence=evidence,
                )
                self._agreement_version += 1
                ingested = True
        if ingested:
            self._notify_ui()

    def _has_live_item(self, key: tuple[str, str]) -> bool:
        """True when a non-terminal item with the same (topic, value) exists."""
        for c in self._ledger.items():
            if c.status in ("confirmed", "rejected"):
                continue
            if (c.topic.strip().lower(), c.proposed_value.strip().lower()) == key:
                return True
        return False

    def confirm_commitment(self, item_id: str, speaker: str) -> bool:
        """Record an explicit human confirmation (application-layer only).

        Policy: an item becomes ``confirmed`` only after BOTH participants
        ("p1" and "you") have explicitly confirmed via this API. Model output
        can never reach this path. Invalid speakers (anything other than
        "p1"/"you") are rejected — model-generated claims cannot masquerade
        as participant confirmation.
        """
        if speaker not in ("p1", "you"):
            return False
        with self._ledger_lock:
            if self._closed:
                return False
            item = self._ledger.get(item_id)
            if item is None or item.status in ("confirmed", "rejected"):
                return False
            if not self._ledger.record_confirmation(
                item_id, speaker, "confirm", source=PARTICIPANT_SOURCE,
            ):
                return False
            if self._both_sides_confirmed(item_id):
                self._ledger.mark_confirmed_explicitly(
                    item_id,
                    resolution_note="both participants explicitly confirmed",
                )
            self._agreement_version += 1
        self._notify_ui()
        return True

    def reject_commitment(self, item_id: str, speaker: str) -> bool:
        """Record an explicit human rejection (application-layer only).

        A single explicit rejection by either participant marks the item
        ``rejected`` — disagreement is never silently resolved.
        """
        if speaker not in ("p1", "you"):
            return False
        with self._ledger_lock:
            if self._closed:
                return False
            item = self._ledger.get(item_id)
            if item is None or item.status in ("confirmed", "rejected"):
                return False
            if not self._ledger.record_confirmation(
                item_id, speaker, "reject", source=PARTICIPANT_SOURCE,
            ):
                return False
            self._ledger.transition(
                item_id, "rejected",
                resolution_note=f"explicitly rejected by {speaker}",
            )
            self._agreement_version += 1
        self._notify_ui()
        return True

    def _both_sides_confirmed(self, item_id: str) -> bool:
        item = self._ledger.get(item_id)
        if item is None:
            return False
        speakers = {
            r.speaker for r in item.confirmation_history
            if r.source == PARTICIPANT_SOURCE and r.action == "confirm"
        }
        return {"p1", "you"} <= speakers

    def get_agreement_snapshot(self) -> tuple[int, list[dict]]:
        """Return (version, items) for the UI agreement panel (Tk thread)."""
        with self._ledger_lock:
            items = []
            for c in self._ledger.items():
                items.append({
                    "id": c.id,
                    "topic": c.topic,
                    "value": c.proposed_value,
                    "status": c.status,
                    "speaker": c.speaker,
                    "evidence_uids": [e.uid for e in c.evidence],
                    "confirmed_by": sorted({
                        r.speaker for r in c.confirmation_history
                        if r.source == PARTICIPANT_SOURCE and r.action == "confirm"
                    }),
                })
            return self._agreement_version, items

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #
    def _notify_ui(self) -> None:
        with self._lock:
            cb = self._ui_callback
        if cb is not None:
            try:
                cb()
            except Exception:  # noqa: BLE001
                logger.exception("UI callback failed")

    # ------------------------------------------------------------------ #
    # Lifecycle (idempotent, bounded)
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        """Shut down both AI services with bounded waits.

        Idempotent. After close, all entry points are no-ops. Note: a
        provider thread already stuck inside a provider call cannot be
        forcibly stopped (Python limitation); each service's circuit breaker
        bounds how many such threads can accumulate.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._ai_service.close()
        except Exception:  # noqa: BLE001
            logger.exception("AI service close failed")
        try:
            self._meeting_service.close()
        except Exception:  # noqa: BLE001
            logger.exception("Meeting service close failed")
        try:
            self._assistant_service.close()
        except Exception:  # noqa: BLE001
            logger.exception("Assistant service close failed")
