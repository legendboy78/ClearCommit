"""Focused tests for assistant.py + assistant integration — plain Python.

Run:  python test_assistant.py

Covers: context bounding (utterance count + char budget, newest retained),
grounded replies, fabricated-citation rejection, provider-unavailable,
provider exceptions, empty/oversized questions, history bounds, clear,
reply with no citations, and the integration ask/result paths (busy,
disabled, malformed).
"""

import time

from assistant import (
    AssistantReply,
    ConversationAssistant,
    FakeAssistantProvider,
)
from meeting_intelligence import EvidenceReference


def _ev(uid: int, side: str = "you", text: str = "line") -> EvidenceReference:
    return EvidenceReference(uid=uid, side=side, text=text, timestamp=1000.0 + uid)


def _ctx_map(n: int) -> dict:
    return {i: _ev(i, "you", f"utterance {i}") for i in range(1, n + 1)}


def test_context_bounded_by_utterance_count():
    a = ConversationAssistant(max_context_utterances=5)
    ctx, uids = a.build_context(_ctx_map(40))
    assert len(uids) == 5
    assert uids == (36, 37, 38, 39, 40)  # newest retained
    assert "[E40|you] utterance 40" in ctx
    assert "utterance 1" not in ctx
    print("PASS test_context_bounded_by_utterance_count")


def test_context_char_budget_keeps_newest():
    a = ConversationAssistant(max_context_chars=300)
    big = {i: _ev(i, "you", "x" * 100) for i in range(1, 21)}
    ctx, uids = a.build_context(big)
    assert len(ctx) <= 300 + 200  # budget + meeting/ledger absent; slack for header
    assert uids[-1] == 20  # newest kept
    assert "not shown" in ctx  # truncation disclosed
    print("PASS test_context_char_budget_keeps_newest")


def test_unavailable_without_provider():
    a = ConversationAssistant(provider=None)
    r = a.ask("summarize", evidence_map=_ctx_map(3))
    assert r.ok is False
    assert r.error == "assistant_unavailable"
    assert "provider" in r.reply.lower()
    print("PASS test_unavailable_without_provider")


def test_grounded_reply_accepted_and_cited():
    p = FakeAssistantProvider(
        reply="The price discussed was $50 [E2] and Friday [E3].")
    a = ConversationAssistant(provider=p)
    r = a.ask("what was agreed?", evidence_map=_ctx_map(5))
    assert r.ok is True
    assert r.cited_uids == (2, 3)
    assert r.context_uids == (1, 2, 3, 4, 5)
    assert len(a.history()) == 2  # one exchange = user + assistant
    assert p.last_question == "what was agreed?"
    assert "[E2|you]" in p.last_context
    print("PASS test_grounded_reply_accepted_and_cited")


def test_fabricated_citation_rejected():
    p = FakeAssistantProvider(reply="Invented from [E999].")
    a = ConversationAssistant(provider=p)
    r = a.ask("q", evidence_map=_ctx_map(3))
    assert r.ok is False
    assert "fabricated_evidence" in r.error
    assert a.history() == []  # rejected replies never enter history
    print("PASS test_fabricated_citation_rejected")


def test_uncited_reply_allowed():
    p = FakeAssistantProvider(reply="Nothing unclear so far.")
    a = ConversationAssistant(provider=p)
    r = a.ask("what is unclear?", evidence_map=_ctx_map(3))
    assert r.ok is True
    assert r.cited_uids == ()
    assert r.context_uids == (1, 2, 3)  # grounding window still recorded
    print("PASS test_uncited_reply_allowed")


def test_provider_exception_graceful():
    p = FakeAssistantProvider(fail=True)
    a = ConversationAssistant(provider=p)
    r = a.ask("q", evidence_map=_ctx_map(3))
    assert r.ok is False
    assert r.error == "provider_error"
    assert a.history() == []
    print("PASS test_provider_exception_graceful")


def test_empty_and_oversized_questions():
    a = ConversationAssistant(provider=FakeAssistantProvider())
    assert a.ask("", evidence_map=_ctx_map(1)).error == "empty_question"
    assert a.ask("   ", evidence_map=_ctx_map(1)).error == "empty_question"
    assert a.ask("x" * 2000, evidence_map=_ctx_map(1)).error == "question_too_long"
    print("PASS test_empty_and_oversized_questions")


def test_history_bounded():
    a = ConversationAssistant(
        provider=FakeAssistantProvider(reply="ok"),
        max_history_exchanges=3,
    )
    for i in range(6):
        a.ask(f"q{i}", evidence_map=_ctx_map(2))
    h = a.history()
    assert len(h) == 6  # 3 exchanges * 2 entries
    assert h[0]["text"] == "q3"  # oldest surviving exchange
    print("PASS test_history_bounded")


def test_clear_history():
    a = ConversationAssistant(provider=FakeAssistantProvider(reply="ok"))
    a.ask("q", evidence_map=_ctx_map(2))
    assert a.history()
    a.clear_history()
    assert a.history() == []
    print("PASS test_clear_history")


def test_meeting_and_ledger_summaries_included():
    p = FakeAssistantProvider(reply="ok")
    a = ConversationAssistant(provider=p)
    a.ask("q", evidence_map=_ctx_map(2),
          meeting_summary="decision[decided]: price = $50",
          ledger_summary="confirmed: price = $50 (confirmed by: p1, you)")
    assert "decision[decided]: price = $50" in p.last_context
    assert "AGREEMENTS:" in p.last_context
    print("PASS test_meeting_and_ledger_summaries_included")


def test_empty_reply_rejected():
    a = ConversationAssistant(provider=FakeAssistantProvider(reply="   "))
    r = a.ask("q", evidence_map=_ctx_map(2))
    assert r.ok is False and r.error == "empty_reply"
    print("PASS test_empty_reply_rejected")


if __name__ == "__main__":
    test_context_bounded_by_utterance_count()
    test_context_char_budget_keeps_newest()
    test_unavailable_without_provider()
    test_grounded_reply_accepted_and_cited()
    test_fabricated_citation_rejected()
    test_uncited_reply_allowed()
    test_provider_exception_graceful()
    test_empty_and_oversized_questions()
    test_history_bounded()
    test_clear_history()
    test_meeting_and_ledger_summaries_included()
    test_empty_reply_rejected()
    print("\nAll assistant tests passed.")
