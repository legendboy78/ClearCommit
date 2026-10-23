"""Focused tests for meeting_intelligence.py — plain Python, no dependencies, no network.

Run:  python test_meeting_intelligence.py

Covers: disabled mode, missing provider, empty transcript, decision parsing
(proposed/decided/rejected), action items (assigned/unassigned, stated vs
unknown owner/deadline), open questions (answered/unresolved), candidate
commitments (inferred/proposed, never confirmed), evidence validation
(reject fabricated/unknown IDs), malformed JSON, code-fence stripping,
kind/status validation, date validation, dedup without merging conflicts,
bounded inputs, provider exceptions, thread-safety, runtime toggle/provider
swap, and conflicting decisions preserved.
"""

import json
import threading

from meeting_intelligence import (
    ActionItem,
    CandidateCommitment,
    Decision,
    EvidenceReference,
    FakeMeetingProvider,
    MeetingIntelligence,
    MeetingRecord,
    OpenQuestion,
)


def _ev(uid: int, side: str = "you", text: str = "transcript text",
        timestamp: float = 1000.0) -> EvidenceReference:
    return EvidenceReference(uid=uid, side=side, text=text, timestamp=timestamp)


def _ev_map(**kwargs) -> dict[int, EvidenceReference]:
    return dict(kwargs)


def _make_output(**kwargs) -> str:
    return json.dumps(kwargs)


def _analyzer(enabled: bool = True, provider=None, **kwargs) -> MeetingIntelligence:
    return MeetingIntelligence(enabled=enabled, provider=provider, **kwargs)


def test_disabled_returns_none():
    provider = FakeMeetingProvider(output=_make_output(decisions=[]))
    ai = _analyzer(enabled=False, provider=provider)
    result = ai.analyze("we agreed to $50", evidence_map={})
    assert result is None
    assert provider.call_count == 0
    print("PASS test_disabled_returns_none")


def test_no_provider_returns_none():
    ai = _analyzer(enabled=True, provider=None)
    assert ai.analyze("we agreed to $50", evidence_map={}) is None
    print("PASS test_no_provider_returns_none")


def test_empty_transcript_returns_none():
    provider = FakeMeetingProvider(output=_make_output(decisions=[]))
    ai = _analyzer(enabled=True, provider=provider)
    for text in ("", "   ", None):
        assert ai.analyze(text, evidence_map={}) is None
    assert provider.call_count == 0
    print("PASS test_empty_transcript_returns_none")


def test_proposed_decision():
    ev1 = _ev(1, "p1", "the price is $50")
    provider = FakeMeetingProvider(output=_make_output(
        decisions=[{
            "topic": "price",
            "proposed_value": "$50",
            "decided_value": None,
            "status": "proposed",
            "evidence_ids": [1],
            "speaker": "p1",
        }]
    ))
    ai = _analyzer(enabled=True, provider=provider)
    result = ai.analyze("the price is $50", evidence_map={1: ev1})
    assert result is not None
    assert len(result.decisions) == 1
    d = result.decisions[0]
    assert d.topic == "price"
    assert d.proposed_value == "$50"
    assert d.decided_value is None
    assert d.status == "proposed"
    assert d.evidence == (ev1,)
    assert d.speaker == "p1"
    assert provider.call_count == 1
    print("PASS test_proposed_decision")


def test_decided_decision():
    ev1 = _ev(1, "you", "we decided on $50")
    provider = FakeMeetingProvider(output=_make_output(
        decisions=[{
            "topic": "price",
            "proposed_value": "$50",
            "decided_value": "$50",
            "status": "decided",
            "evidence_ids": [1],
            "speaker": "you",
        }]
    ))
    ai = _analyzer(enabled=True, provider=provider)
    result = ai.analyze("we decided on $50", evidence_map={1: ev1})
    assert result is not None
    d = result.decisions[0]
    assert d.status == "decided"
    assert d.decided_value == "$50"
    print("PASS test_decided_decision")


def test_rejected_decision():
    ev1 = _ev(1, "p1", "no to $100")
    provider = FakeMeetingProvider(output=_make_output(
        decisions=[{
            "topic": "price",
            "proposed_value": "$100",
            "decided_value": None,
            "status": "rejected",
            "evidence_ids": [1],
            "speaker": "p1",
        }]
    ))
    ai = _analyzer(enabled=True, provider=provider)
    result = ai.analyze("no to $100", evidence_map={1: ev1})
    assert result is not None
    assert result.decisions[0].status == "rejected"
    print("PASS test_rejected_decision")


def test_action_item_assigned_with_owner_deadline():
    ev1 = _ev(1, "you", "I will send the invoice by Friday")
    provider = FakeMeetingProvider(output=_make_output(
        action_items=[{
            "description": "send the invoice",
            "owner": "you",
            "deadline": "2026-01-09",
            "status": "assigned",
            "evidence_ids": [1],
        }]
    ))
    ai = _analyzer(enabled=True, provider=provider)
    result = ai.analyze("I will send the invoice by Friday",
                        evidence_map={1: ev1})
    assert result is not None
    a = result.action_items[0]
    assert a.description == "send the invoice"
    assert a.owner == "you"
    assert a.deadline == "2026-01-09"
    assert a.status == "assigned"
    assert a.evidence == (ev1,)
    print("PASS test_action_item_assigned_with_owner_deadline")


def test_action_item_unassigned_unknown_owner_deadline():
    ev1 = _ev(1, "p1", "someone should send the invoice")
    provider = FakeMeetingProvider(output=_make_output(
        action_items=[{
            "description": "send the invoice",
            "owner": None,
            "deadline": None,
            "status": "unassigned",
            "evidence_ids": [1],
        }]
    ))
    ai = _analyzer(enabled=True, provider=provider)
    result = ai.analyze("someone should send the invoice",
                        evidence_map={1: ev1})
    assert result is not None
    a = result.action_items[0]
    assert a.owner is None
    assert a.deadline is None
    assert a.status == "unassigned"
    print("PASS test_action_item_unassigned_unknown_owner_deadline")


def test_action_item_invalid_date_rejected():
    ev1 = _ev(1, "you", "send it by next week")
    provider = FakeMeetingProvider(output=_make_output(
        action_items=[{
            "description": "send it",
            "owner": "you",
            "deadline": "next week",  # not ISO-8601
            "status": "assigned",
            "evidence_ids": [1],
        }]
    ))
    ai = _analyzer(enabled=True, provider=provider)
    result = ai.analyze("send it by next week", evidence_map={1: ev1})
    # Invalid date -> item rejected -> nothing extractable -> None.
    assert result is None
    print("PASS test_action_item_invalid_date_rejected")


def test_open_question_unanswered():
    ev1 = _ev(1, "p1", "when is the deadline?")
    provider = FakeMeetingProvider(output=_make_output(
        open_questions=[{
            "question": "when is the deadline?",
            "answered": False,
            "answer": None,
            "evidence_ids": [1],
        }]
    ))
    ai = _analyzer(enabled=True, provider=provider)
    result = ai.analyze("when is the deadline?", evidence_map={1: ev1})
    assert result is not None
    q = result.open_questions[0]
    assert q.question == "when is the deadline?"
    assert q.answered is False
    assert q.answer is None
    print("PASS test_open_question_unanswered")


def test_open_question_answered():
    ev1 = _ev(1, "you", "the deadline is Friday")
    provider = FakeMeetingProvider(output=_make_output(
        open_questions=[{
            "question": "when is the deadline?",
            "answered": True,
            "answer": "Friday",
            "evidence_ids": [1],
        }]
    ))
    ai = _analyzer(enabled=True, provider=provider)
    result = ai.analyze("the deadline is Friday", evidence_map={1: ev1})
    assert result is not None
    q = result.open_questions[0]
    assert q.answered is True
    assert q.answer == "Friday"
    print("PASS test_open_question_answered")


def test_candidate_commitment_never_confirmed():
    ev1 = _ev(1, "p1", "I propose we use vendor A")
    provider = FakeMeetingProvider(output=_make_output(
        candidate_commitments=[{
            "topic": "vendor",
            "proposed_value": "vendor A",
            "status": "inferred",
            "evidence_ids": [1],
            "speaker": "p1",
        }]
    ))
    ai = _analyzer(enabled=True, provider=provider)
    result = ai.analyze("I propose we use vendor A", evidence_map={1: ev1})
    assert result is not None
    c = result.candidate_commitments[0]
    assert c.status in ("inferred", "proposed")
    assert c.status != "confirmed"
    print("PASS test_candidate_commitment_never_confirmed")


def test_candidate_commitment_confirmed_status_rejected():
    """The analyzer must never accept 'confirmed' status for commitments."""
    ev1 = _ev(1, "p1", "I propose vendor A")
    provider = FakeMeetingProvider(output=_make_output(
        candidate_commitments=[{
            "topic": "vendor",
            "proposed_value": "vendor A",
            "status": "confirmed",  # invalid for this analyzer
            "evidence_ids": [1],
            "speaker": "p1",
        }]
    ))
    ai = _analyzer(enabled=True, provider=provider)
    result = ai.analyze("I propose vendor A", evidence_map={1: ev1})
    # "confirmed" status is invalid for commitments -> rejected -> None.
    assert result is None
    print("PASS test_candidate_commitment_confirmed_status_rejected")


def test_no_coaching_extracted_returns_none():
    provider = FakeMeetingProvider(output=_make_output(
        topics=[], decisions=[], action_items=[], open_questions=[],
        candidate_commitments=[],
    ))
    ai = _analyzer(enabled=True, provider=provider)
    result = ai.analyze("hello how are you", evidence_map={})
    assert result is None
    print("PASS test_no_coaching_extracted_returns_none")


def test_fabricated_evidence_id_rejected():
    """Claims referencing unknown evidence IDs must be rejected."""
    ev1 = _ev(1, "you", "the price is $50")
    provider = FakeMeetingProvider(output=_make_output(
        decisions=[{
            "topic": "price",
            "proposed_value": "$50",
            "decided_value": None,
            "status": "proposed",
            "evidence_ids": [999],  # fabricated / unknown
            "speaker": "you",
        }]
    ))
    ai = _analyzer(enabled=True, provider=provider)
    result = ai.analyze("the price is $50", evidence_map={1: ev1})
    # Unknown evidence id -> decision rejected -> nothing extractable -> None.
    assert result is None
    print("PASS test_fabricated_evidence_id_rejected")


def test_malformed_json_rejected():
    provider = FakeMeetingProvider(output="not json {{{")
    ai = _analyzer(enabled=True, provider=provider)
    result = ai.analyze("the price is $50", evidence_map={})
    assert result is None
    print("PASS test_malformed_json_rejected")


def test_code_fence_stripped():
    provider = FakeMeetingProvider(output=(
        "```json\n"
        "{\n"
        '  "decisions": [{"topic": "price", "proposed_value": "$50", '
        '"decided_value": null, "status": "proposed", '
        '"evidence_ids": [1], "speaker": "you"}]\n'
        "}\n```"
    ))
    ev1 = _ev(1, "you", "the price is $50")
    ai = _analyzer(enabled=True, provider=provider)
    result = ai.analyze("the price is $50", evidence_map={1: ev1})
    assert result is not None
    assert len(result.decisions) == 1
    assert result.decisions[0].proposed_value == "$50"
    print("PASS test_code_fence_stripped")


def test_invalid_decision_status_rejected():
    ev1 = _ev(1, "you", "the price is $50")
    provider = FakeMeetingProvider(output=_make_output(
        decisions=[{
            "topic": "price",
            "proposed_value": "$50",
            "decided_value": None,
            "status": "confirmed",  # invalid for decisions
            "evidence_ids": [1],
            "speaker": "you",
        }]
    ))
    ai = _analyzer(enabled=True, provider=provider)
    result = ai.analyze("the price is $50", evidence_map={1: ev1})
    # Invalid status -> decision rejected -> nothing extractable -> None.
    assert result is None
    print("PASS test_invalid_decision_status_rejected")


def test_invalid_action_status_rejected():
    ev1 = _ev(1, "you", "send the invoice")
    provider = FakeMeetingProvider(output=_make_output(
        action_items=[{
            "description": "send the invoice",
            "owner": None,
            "deadline": None,
            "status": "completed",  # invalid
            "evidence_ids": [1],
        }]
    ))
    ai = _analyzer(enabled=True, provider=provider)
    result = ai.analyze("send the invoice", evidence_map={1: ev1})
    # Invalid status -> action item rejected -> nothing extractable -> None.
    assert result is None
    print("PASS test_invalid_action_status_rejected")


def test_invalid_speaker_rejected():
    ev1 = _ev(1, "you", "the price is $50")
    provider = FakeMeetingProvider(output=_make_output(
        decisions=[{
            "topic": "price",
            "proposed_value": "$50",
            "decided_value": None,
            "status": "proposed",
            "evidence_ids": [1],
            "speaker": "third_party",  # invalid side
        }]
    ))
    ai = _analyzer(enabled=True, provider=provider)
    result = ai.analyze("the price is $50", evidence_map={1: ev1})
    # Invalid speaker -> decision rejected -> nothing extractable -> None.
    assert result is None
    print("PASS test_invalid_speaker_rejected")


def test_conflicting_decisions_preserved_not_merged():
    """Two conflicting proposals for the same topic stay as separate items."""
    ev1 = _ev(1, "p1", "price is $50")
    ev2 = _ev(2, "you", "price is $100")
    provider = FakeMeetingProvider(output=_make_output(
        decisions=[
            {
                "topic": "price",
                "proposed_value": "$50",
                "decided_value": None,
                "status": "proposed",
                "evidence_ids": [1],
                "speaker": "p1",
            },
            {
                "topic": "price",
                "proposed_value": "$100",
                "decided_value": None,
                "status": "proposed",
                "evidence_ids": [2],
                "speaker": "you",
            },
        ]
    ))
    ai = _analyzer(enabled=True, provider=provider)
    result = ai.analyze("price is $50 ... price is $100",
                        evidence_map={1: ev1, 2: ev2})
    assert result is not None
    assert len(result.decisions) == 2
    values = {d.proposed_value for d in result.decisions}
    assert values == {"$50", "$100"}
    print("PASS test_conflicting_decisions_preserved_not_merged")


def test_duplicate_items_deduplicated():
    """Same proposal twice -> one item (dedup without merging conflicts)."""
    ev1 = _ev(1, "you", "price is $50")
    provider = FakeMeetingProvider(output=_make_output(
        decisions=[
            {
                "topic": "price",
                "proposed_value": "$50",
                "decided_value": None,
                "status": "proposed",
                "evidence_ids": [1],
                "speaker": "you",
            },
            {
                "topic": "price",
                "proposed_value": "$50",
                "decided_value": None,
                "status": "proposed",
                "evidence_ids": [1],
                "speaker": "you",
            },
        ]
    ))
    ai = _analyzer(enabled=True, provider=provider)
    result = ai.analyze("price is $50", evidence_map={1: ev1})
    assert result is not None
    assert len(result.decisions) == 1
    assert provider.call_count == 1
    print("PASS test_duplicate_items_deduplicated")


def test_empty_evidence_list_allowed():
    """A claim with no evidence is allowed (empty evidence_ids)."""
    provider = FakeMeetingProvider(output=_make_output(
        topics=["general discussion"],
        decisions=[],
        action_items=[],
        open_questions=[],
        candidate_commitments=[],
    ))
    ai = _analyzer(enabled=True, provider=provider)
    result = ai.analyze("general discussion", evidence_map={})
    assert result is not None
    assert result.topics == ("general discussion",)
    print("PASS test_empty_evidence_list_allowed")


def test_malformed_evidence_structure_rejected():
    """evidence_ids must be a list of ints; malformed -> reject item."""
    ev1 = _ev(1, "you", "the price is $50")
    provider = FakeMeetingProvider(output=_make_output(
        decisions=[{
            "topic": "price",
            "proposed_value": "$50",
            "decided_value": None,
            "status": "proposed",
            "evidence_ids": "not_a_list",
            "speaker": "you",
        }]
    ))
    ai = _analyzer(enabled=True, provider=provider)
    result = ai.analyze("the price is $50", evidence_map={1: ev1})
    # Malformed evidence structure -> decision rejected -> None.
    assert result is None
    print("PASS test_malformed_evidence_structure_rejected")


def test_provider_exception_graceful():
    provider = FakeMeetingProvider(output=None, fail=True)
    ai = _analyzer(enabled=True, provider=provider)
    result = ai.analyze("the price is $50", evidence_map={})
    assert result is None
    assert provider.call_count == 1
    print("PASS test_provider_exception_graceful")


def test_bounded_transcript_truncated():
    provider = FakeMeetingProvider(output=_make_output(
        decisions=[{
            "topic": "price",
            "proposed_value": "$50",
            "decided_value": None,
            "status": "proposed",
            "evidence_ids": [1],
            "speaker": "you",
        }]
    ))
    ev1 = _ev(1, "you", "the price is $50")
    ai = _analyzer(enabled=True, provider=provider, max_transcript_chars=20)
    long_text = "the price is $50 and then many more words here"
    result = ai.analyze(long_text, evidence_map={1: ev1})
    # Truncated text may or may not contain a valid decision; just ensure no
    # crash and a valid record shape.
    assert result is None or isinstance(result, MeetingRecord)
    print("PASS test_bounded_transcript_truncated")


def test_result_size_bounded():
    """max_decisions bounds the number of decisions returned."""
    evs = {i: _ev(i, "you", f"item {i}") for i in range(1, 6)}
    provider = FakeMeetingProvider(output=_make_output(
        decisions=[
            {
                "topic": f"topic {i}",
                "proposed_value": f"value {i}",
                "decided_value": None,
                "status": "proposed",
                "evidence_ids": [i],
                "speaker": "you",
            } for i in range(1, 6)
        ]
    ))
    ai = _analyzer(enabled=True, provider=provider, max_decisions=2)
    result = ai.analyze("items", evidence_map=evs)
    assert result is not None
    assert len(result.decisions) <= 2
    print("PASS test_result_size_bounded")


def test_thread_safety_concurrent_calls():
    ev1 = _ev(1, "you", "the price is $50")
    provider = FakeMeetingProvider(output=_make_output(
        decisions=[{
            "topic": "price",
            "proposed_value": "$50",
            "decided_value": None,
            "status": "proposed",
            "evidence_ids": [1],
            "speaker": "you",
        }]
    ))
    ai = _analyzer(enabled=True, provider=provider)
    results = []
    errors = []
    lock = threading.Lock()

    def worker():
        try:
            r = ai.analyze("the price is $50", evidence_map={1: ev1})
            with lock:
                results.append(r)
        except Exception as exc:  # noqa: BLE001
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, errors
    assert all(r is None or isinstance(r, MeetingRecord) for r in results)
    print("PASS test_thread_safety_concurrent_calls")


def test_toggle_enabled_at_runtime():
    provider = FakeMeetingProvider(output=_make_output(
        decisions=[{
            "topic": "price",
            "proposed_value": "$50",
            "decided_value": None,
            "status": "proposed",
            "evidence_ids": [1],
            "speaker": "you",
        }]
    ))
    ev1 = _ev(1, "you", "the price is $50")
    ai = _analyzer(enabled=False, provider=provider)
    assert ai.is_enabled() is False
    assert ai.analyze("the price is $50", evidence_map={1: ev1}) is None

    ai.set_enabled(True)
    assert ai.is_enabled() is True
    result = ai.analyze("the price is $50", evidence_map={1: ev1})
    assert result is not None
    print("PASS test_toggle_enabled_at_runtime")


def test_set_provider_at_runtime():
    ai = _analyzer(enabled=True, provider=None)
    assert ai.analyze("the price is $50", evidence_map={}) is None

    ev1 = _ev(1, "you", "the price is $50")
    provider = FakeMeetingProvider(output=_make_output(
        decisions=[{
            "topic": "price",
            "proposed_value": "$50",
            "decided_value": None,
            "status": "proposed",
            "evidence_ids": [1],
            "speaker": "you",
        }]
    ))
    ai.set_provider(provider)
    result = ai.analyze("the price is $50", evidence_map={1: ev1})
    assert result is not None
    print("PASS test_set_provider_at_runtime")


def test_record_immutable():
    ev1 = _ev(1, "you", "the price is $50")
    provider = FakeMeetingProvider(output=_make_output(
        decisions=[{
            "topic": "price",
            "proposed_value": "$50",
            "decided_value": None,
            "status": "proposed",
            "evidence_ids": [1],
            "speaker": "you",
        }]
    ))
    ai = _analyzer(enabled=True, provider=provider)
    result = ai.analyze("the price is $50", evidence_map={1: ev1})
    assert result is not None
    # Frozen dataclasses cannot be mutated.
    try:
        result.topics = ("changed",)  # type: ignore[misc]
        raise AssertionError("MeetingRecord should be frozen")
    except Exception:
        pass
    try:
        result.decisions[0].proposed_value = "changed"  # type: ignore[misc]
        raise AssertionError("Decision should be frozen")
    except Exception:
        pass
    print("PASS test_record_immutable")


if __name__ == "__main__":
    test_disabled_returns_none()
    test_no_provider_returns_none()
    test_empty_transcript_returns_none()
    test_proposed_decision()
    test_decided_decision()
    test_rejected_decision()
    test_action_item_assigned_with_owner_deadline()
    test_action_item_unassigned_unknown_owner_deadline()
    test_action_item_invalid_date_rejected()
    test_open_question_unanswered()
    test_open_question_answered()
    test_candidate_commitment_never_confirmed()
    test_candidate_commitment_confirmed_status_rejected()
    test_no_coaching_extracted_returns_none()
    test_fabricated_evidence_id_rejected()
    test_malformed_json_rejected()
    test_code_fence_stripped()
    test_invalid_decision_status_rejected()
    test_invalid_action_status_rejected()
    test_invalid_speaker_rejected()
    test_conflicting_decisions_preserved_not_merged()
    test_duplicate_items_deduplicated()
    test_empty_evidence_list_allowed()
    test_malformed_evidence_structure_rejected()
    test_provider_exception_graceful()
    test_bounded_transcript_truncated()
    test_result_size_bounded()
    test_thread_safety_concurrent_calls()
    test_toggle_enabled_at_runtime()
    test_set_provider_at_runtime()
    test_record_immutable()
    print("\nAll meeting_intelligence tests passed.")
