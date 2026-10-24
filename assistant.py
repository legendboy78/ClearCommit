"""Conversation-aware assistant for ClearCommit (Stage A2).

A grounded question/answer service over the live conversation. Pure logic —
no threads, no UI, no network. Async execution is the integration's job
(dedicated bounded AIService); the UI talks to the integration.

Grounding rules (deterministic, enforced in code):
    * The context brief contains ONLY transcript utterances from the trusted
      evidence store, labeled ``[E<uid>|<side>]``, plus the validated meeting
      record and agreement summary.
    * The reply may cite evidence as ``[E<uid>]``. Any citation referencing a
      uid NOT in the supplied evidence window is fabricated -> the whole
      reply is REJECTED (never displayed as fact).
    * Uncited replies are allowed (e.g. "nothing unclear yet"), but the reply
      object still carries the evidence window it was grounded in so the UI
      can show what the assistant could see.
    * The assistant never mutates the agreement ledger, never marks anything
      confirmed, and never infers consent.
    * History is bounded; the context window is bounded; oversized questions
      are rejected at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol
import logging
import re
from collections import deque

from meeting_intelligence import EvidenceReference

logger = logging.getLogger(__name__)


class AssistantProvider(Protocol):
    """Pluggable chat backend.

    ``answer(question, context, history)`` receives the user question, the
    bounded grounded context brief, and the bounded conversation history
    (list of {"role": "user"|"assistant", "text": str}). Returns the reply
    text. Raise on failure; the assistant records it as an error.
    """

    def answer(self, question: str, context: str,
               history: list[dict]) -> str:
        raise NotImplementedError


@dataclass
class FakeAssistantProvider:
    """Deterministic stand-in for tests (no network, no keys)."""

    reply: str = "Nothing to report yet."
    fail: bool = False
    error_message: str = "provider failure"
    call_count: int = 0
    last_question: str = ""
    last_context: str = ""
    last_history: Optional[list] = None

    def answer(self, question: str, context: str,
               history: list[dict]) -> str:
        self.call_count += 1
        self.last_question = question
        self.last_context = context
        self.last_history = list(history)
        if self.fail:
            raise RuntimeError(self.error_message)
        return self.reply


@dataclass(frozen=True)
class AssistantReply:
    """The outcome of one assistant question."""

    ok: bool
    reply: str = ""
    error: Optional[str] = None
    cited_uids: tuple[int, ...] = ()
    context_uids: tuple[int, ...] = ()


_CITATION_RE = re.compile(r"\[E(\d+)\]")


@dataclass
class ConversationAssistant:
    """Grounded Q&A over the live conversation (pure logic; thread-safe)."""

    def __init__(
        self,
        *,
        provider: Optional[AssistantProvider] = None,
        system_prompt: Optional[str] = None,
        max_history_exchanges: int = 10,
        max_context_utterances: int = 40,
        max_context_chars: int = 3500,
        max_question_chars: int = 1000,
        max_reply_chars: int = 4000,
        history_text_cap: int = 1200,
    ) -> None:
        self._provider = provider
        self.system_prompt = (
            system_prompt.strip() if system_prompt and system_prompt.strip()
            else None)
        self._max_history = max(1, int(max_history_exchanges))
        self._max_context_utterances = max(1, int(max_context_utterances))
        self._max_context_chars = max(200, int(max_context_chars))
        self._max_question_chars = max(1, int(max_question_chars))
        self._max_reply_chars = max(1, int(max_reply_chars))
        self._history_text_cap = max(1, int(history_text_cap))
        # Completed exchanges only (question asked AFTER the provider returned
        # is not part of the model context — the UI shows it regardless).
        # Each exchange stores two entries (user + assistant).
        self._history: "deque[dict]" = deque(maxlen=self._max_history * 2)

    # ------------------------------------------------------------------ #
    def set_provider(self, provider: Optional[AssistantProvider]) -> None:
        self._provider = provider

    def set_system_prompt(self, prompt: Optional[str]) -> None:
        """Set the assistant persona/instructions override (None = default).

        The grounding contract (cite [E<uid>], never invent) is always
        appended after any custom instructions so edits can't weaken it.
        """
        self.system_prompt = (
            prompt.strip() if prompt and prompt.strip() else None)

    def clear_history(self) -> None:
        self._history.clear()

    def history(self) -> list[dict]:
        return list(self._history)

    # ------------------------------------------------------------------ #
    # Context building
    # ------------------------------------------------------------------ #
    def build_context(
        self,
        evidence_map: dict[int, EvidenceReference],
        meeting_summary: str = "",
        ledger_summary: str = "",
    ) -> tuple[str, tuple[int, ...]]:
        """Build the bounded grounded context brief.

        Returns (context_text, context_uids). The transcript is rendered
        oldest-last, bounded by utterance count and character budget (the
        NEWEST evidence is retained; older lines are dropped first and the
        truncation is disclosed inside the brief).
        """
        refs = sorted(evidence_map.values(), key=lambda r: r.uid)
        refs = refs[-self._max_context_utterances:]
        lines = [f"[E{r.uid}|{r.side}] {r.text}" for r in refs]
        uids = tuple(r.uid for r in refs)

        transcript_block = "\n".join(lines)
        dropped = len(evidence_map) - len(refs)
        header = ""
        if dropped > 0:
            header = f"(older {dropped} utterances not shown)\n"

        # Character budget: keep the newest lines, disclose truncation.
        budget = self._max_context_chars
        while len(transcript_block) > budget and len(lines) > 1:
            lines = lines[1:]
            uids = uids[1:]
            transcript_block = "\n".join(lines)
            dropped += 1
        if dropped > 0 and not header:
            header = f"(older {dropped} utterances not shown)\n"

        parts = []
        if self.system_prompt:
            parts += ["INSTRUCTIONS:", self.system_prompt, ""]
        # The grounding contract is UNCONDITIONAL — custom instructions can
        # add to it but can never weaken it.
        parts += ["GROUNDING CONTRACT (always applies): cite evidence as "
                  "[E<uid>]; answer only from the provided context; say "
                  "plainly when the answer is not in the transcript; never "
                  "invent dates, names, agreements, or consent.", ""]
        parts += ["TRANSCRIPT (oldest to newest):", header, transcript_block]
        if meeting_summary:
            parts += ["", "MEETING RECORD:", meeting_summary]
        if ledger_summary:
            parts += ["", "AGREEMENTS:", ledger_summary]
        return "\n".join(parts), uids

    # ------------------------------------------------------------------ #
    # Question / answer
    # ------------------------------------------------------------------ #
    def ask(
        self,
        question: str,
        *,
        evidence_map: dict[int, EvidenceReference],
        meeting_summary: str = "",
        ledger_summary: str = "",
    ) -> AssistantReply:
        """Ask one grounded question. Synchronous; the caller owns async."""
        question = (question or "").strip()
        if not question:
            return AssistantReply(ok=False, error="empty_question")
        if len(question) > self._max_question_chars:
            return AssistantReply(ok=False, error="question_too_long")

        provider = self._provider
        if provider is None:
            return AssistantReply(
                ok=False,
                error="assistant_unavailable",
                reply="The assistant is not connected to a provider yet. "
                      "Connect a provider to enable conversation Q&A.",
            )

        context, context_uids = self.build_context(
            evidence_map, meeting_summary, ledger_summary,
        )
        history = self.history()
        try:
            reply = provider.answer(question, context, history)
        except Exception as exc:  # noqa: BLE001 - provider failures are non-fatal
            # Keep the failure reason: the UI maps it to an actionable message
            # (rate limit, bad key, ...) and the log records it for diagnosis.
            logger.warning("Assistant provider failure: %s", exc)
            return AssistantReply(
                ok=False, error=f"provider_error:{exc}",
                reply="The assistant failed to answer. Try again.",
                context_uids=context_uids,
            )

        reply = (reply or "").strip()
        if not reply:
            return AssistantReply(
                ok=False, error="empty_reply", context_uids=context_uids,
            )
        if len(reply) > self._max_reply_chars:
            reply = reply[: self._max_reply_chars] + " …"

        # Fabricated-citation guard: every [E<uid>] reference must exist in
        # the supplied evidence window. A reply citing unknown evidence is
        # rejected outright — never displayed as grounded fact.
        cited = tuple(int(m) for m in _CITATION_RE.findall(reply))
        known = set(context_uids)
        bad = [u for u in cited if u not in known]
        if bad:
            return AssistantReply(
                ok=False,
                error=f"fabricated_evidence:{','.join(str(u) for u in sorted(set(bad)))[:80]}",
                context_uids=context_uids,
            )

        self._record_exchange(question, reply)
        return AssistantReply(
            ok=True, reply=reply, cited_uids=tuple(dict.fromkeys(cited)),
            context_uids=context_uids,
        )

    # ------------------------------------------------------------------ #
    def _record_exchange(self, question: str, reply: str) -> None:
        q = question[: self._history_text_cap]
        a = reply[: self._history_text_cap]
        self._history.append({"role": "user", "text": q})
        self._history.append({"role": "assistant", "text": a})
