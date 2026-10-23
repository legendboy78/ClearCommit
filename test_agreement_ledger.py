"""Focused tests for agreement_ledger.py — plain Python, no dependencies.

Run:  python test_agreement_ledger.py
"""

from agreement_ledger import (
    AgreementLedger,
    Commitment,
    ConfirmationRecord,
    TranscriptEvidence,
)


def _evidence(uid: int = 1, side: str = "p1", original: str = "hola",
              translated: str = "hello", source_lang: str = "es",
              target_lang: str = "en", accept_ts: float = 1000.0) -> TranscriptEvidence:
    return TranscriptEvidence(
        uid=uid, side=side, original=original, translated=translated,
        accept_ts=accept_ts, source_lang=source_lang, target_lang=target_lang,
    )


def test_default_status_is_proposed():
    ledger = AgreementLedger()
    cid = ledger.add_item(topic="price", proposed_value="$50", speaker="p1",
                          evidence=[_evidence()])
    item = ledger.get(cid)
    assert item is not None
    assert item.status == "proposed"
    assert item.topic == "price"
    assert item.proposed_value == "$50"
    assert item.speaker == "p1"
    assert len(item.evidence) == 1
    assert item.confirmation_history == []
    assert item.id == cid
    print("PASS test_default_status_is_proposed")


def test_valid_transitions():
    ledger = AgreementLedger()
    cid = ledger.add_item(topic="deadline", proposed_value="Friday", speaker="you",
                          evidence=[_evidence(uid=2, side="you")])

    # proposed -> needs_clarification
    assert ledger.transition(cid, "needs_clarification") is True
    assert ledger.get(cid).status == "needs_clarification"

    # needs_clarification -> confirmed
    assert ledger.transition(cid, "confirmed") is True
    assert ledger.get(cid).status == "confirmed"
    print("PASS test_valid_transitions")


def test_invalid_transition_rejected():
    ledger = AgreementLedger()
    cid = ledger.add_item(topic="qty", proposed_value="10", speaker="p1",
                          evidence=[_evidence()])

    # proposed -> disputed is NOT a valid direct transition
    assert ledger.transition(cid, "disputed") is False
    assert ledger.get(cid).status == "proposed"

    # confirmed/rejected are terminal: confirmed -> needs_clarification invalid
    ledger2 = AgreementLedger()
    cid2 = ledger2.add_item(topic="x", proposed_value="y", speaker="p1",
                            evidence=[_evidence()])
    ledger2.transition(cid2, "confirmed")
    assert ledger2.transition(cid2, "needs_clarification") is False
    assert ledger2.get(cid2).status == "confirmed"
    print("PASS test_invalid_transition_rejected")


def test_invalid_status_rejected():
    ledger = AgreementLedger()
    cid = ledger.add_item(topic="x", proposed_value="y", speaker="p1",
                          evidence=[_evidence()])
    assert ledger.transition(cid, "nonsense") is False
    assert ledger.get(cid).status == "proposed"
    print("PASS test_invalid_status_rejected")


def test_unknown_item_rejected():
    ledger = AgreementLedger()
    assert ledger.transition("nonexistent", "confirmed") is False
    assert ledger.record_confirmation("nonexistent", "p1", "confirm") is False
    assert ledger.add_evidence("nonexistent", _evidence()) is False
    assert ledger.set_clarification("nonexistent", "q") is False
    assert ledger.mark_confirmed_explicitly("nonexistent") is False
    print("PASS test_unknown_item_rejected")


def test_evidence_is_immutable_and_append_only():
    ledger = AgreementLedger()
    cid = ledger.add_item(topic="price", proposed_value="$50", speaker="p1",
                          evidence=[_evidence(uid=1)])
    e2 = _evidence(uid=2, side="you", original="ok", translated="ok")
    assert ledger.add_evidence(cid, e2) is True
    item = ledger.get(cid)
    assert len(item.evidence) == 2
    assert item.evidence[0].uid == 1
    assert item.evidence[1].uid == 2
    # Frozen evidence cannot be mutated in place.
    try:
        item.evidence[0].original = "changed"  # type: ignore[misc]
        raise AssertionError("TranscriptEvidence should be frozen")
    except dataclasses.FrozenInstanceError:
        pass
    print("PASS test_evidence_is_immutable_and_append_only")


def test_cannot_confirm_without_history():
    ledger = AgreementLedger()
    cid = ledger.add_item(topic="price", proposed_value="$50", speaker="p1",
                          evidence=[_evidence()])
    # No confirmation history -> cannot mark confirmed
    assert ledger.mark_confirmed_explicitly(cid) is False
    assert ledger.get(cid).status == "proposed"
    assert ledger.is_confirmed(cid) is False
    print("PASS test_cannot_confirm_without_history")


def test_explicit_confirmation_allows_confirmed():
    ledger = AgreementLedger()
    cid = ledger.add_item(topic="price", proposed_value="$50", speaker="p1",
                          evidence=[_evidence()])

    # Both relevant participants confirm explicitly
    assert ledger.record_confirmation(cid, "p1", "confirm", note="agreed") is True
    assert ledger.record_confirmation(cid, "you", "confirm", note="agreed") is True
    item = ledger.get(cid)
    assert len(item.confirmation_history) == 2
    assert item.confirmation_history[0].speaker == "p1"
    assert item.confirmation_history[0].action == "confirm"
    assert item.confirmation_history[1].speaker == "you"
    assert item.confirmation_history[1].action == "confirm"

    # Now the ledger permits confirmed status (application decides relevant set)
    assert ledger.mark_confirmed_explicitly(cid) is True
    assert ledger.get(cid).status == "confirmed"
    assert ledger.is_confirmed(cid) is True
    print("PASS test_explicit_confirmation_allows_confirmed")


def test_reject_records_history_and_terminal():
    ledger = AgreementLedger()
    cid = ledger.add_item(topic="qty", proposed_value="100", speaker="p1",
                          evidence=[_evidence()])
    assert ledger.record_confirmation(cid, "you", "reject", note="disagree") is True
    assert ledger.transition(cid, "rejected") is True
    assert ledger.get(cid).status == "rejected"
    # Terminal: cannot leave rejected
    assert ledger.transition(cid, "confirmed") is False
    assert ledger.get(cid).status == "rejected"
    print("PASS test_reject_records_history_and_terminal")


def test_needs_clarification_requires_explicit_question():
    ledger = AgreementLedger()
    cid = ledger.add_item(topic="deadline", proposed_value="Friday", speaker="you",
                          evidence=[_evidence(uid=3, side="you")])
    assert ledger.transition(cid, "needs_clarification") is True
    assert ledger.set_clarification(cid, "Do you mean this Friday or next Friday?") is True
    item = ledger.get(cid)
    assert item.clarification_question == "Do you mean this Friday or next Friday?"
    # Clearing the question
    assert ledger.set_clarification(cid, None) is True
    assert ledger.get(cid).clarification_question is None
    print("PASS test_needs_clarification_requires_explicit_question")


def test_duplicate_events_create_distinct_items():
    ledger = AgreementLedger()
    # Same proposed value spoken twice -> two distinct commitments (dedup is
    # the application's job; the ledger treats them as separate items).
    cid1 = ledger.add_item(topic="price", proposed_value="$50", speaker="p1",
                           evidence=[_evidence(uid=1)])
    cid2 = ledger.add_item(topic="price", proposed_value="$50", speaker="p1",
                           evidence=[_evidence(uid=4)])
    assert cid1 != cid2
    assert ledger.get(cid1) is not None
    assert ledger.get(cid2) is not None
    assert len(ledger.items()) == 2
    print("PASS test_duplicate_events_create_distinct_items")


def test_missing_information_remains_unknown():
    ledger = AgreementLedger()
    # Item created with no evidence (information not yet observed).
    cid = ledger.add_item(topic="deadline", proposed_value="", speaker="p1",
                          evidence=[])
    item = ledger.get(cid)
    assert item is not None
    assert item.proposed_value == ""
    assert item.evidence == []
    assert item.status == "proposed"
    # Not confirmed: no history
    assert ledger.mark_confirmed_explicitly(cid) is False
    print("PASS test_missing_information_remains_unknown")


def test_disputed_transition_path():
    ledger = AgreementLedger()
    cid = ledger.add_item(topic="price", proposed_value="$50", speaker="p1",
                          evidence=[_evidence()])
    assert ledger.transition(cid, "needs_clarification") is True
    assert ledger.transition(cid, "disputed") is True
    assert ledger.get(cid).status == "disputed"
    # disputed can go to confirmed (with history) or rejected
    assert ledger.record_confirmation(cid, "you", "reject") is True
    assert ledger.transition(cid, "rejected") is True
    assert ledger.get(cid).status == "rejected"
    print("PASS test_disputed_transition_path")


def test_items_by_status():
    ledger = AgreementLedger()
    c1 = ledger.add_item(topic="a", proposed_value="1", speaker="p1",
                         evidence=[_evidence(uid=1)])
    c2 = ledger.add_item(topic="b", proposed_value="2", speaker="you",
                         evidence=[_evidence(uid=2)])
    ledger.transition(c2, "confirmed")
    assert len(ledger.items_by_status("proposed")) == 1
    assert len(ledger.items_by_status("confirmed")) == 1
    assert len(ledger.items_by_status("bogus")) == 0
    print("PASS test_items_by_status")


def test_model_originated_confirmation_never_counts():
    ledger = AgreementLedger()
    cid = ledger.add_item(topic="price", proposed_value="$50", speaker="p1",
                          evidence=[_evidence()])

    # An LLM/model claims confirmation. It is recorded for audit transparency
    # but must not satisfy the confirmation requirement.
    assert ledger.record_confirmation(cid, "p1", "confirm",
                                      source="model", note="LLM claimed yes") is True
    item = ledger.get(cid)
    assert len(item.confirmation_history) == 1
    assert item.confirmation_history[0].source == "model"

    # No valid participant confirmation -> cannot mark confirmed.
    assert ledger.mark_confirmed_explicitly(cid) is False
    assert ledger.get(cid).status == "proposed"
    assert ledger.is_confirmed(cid) is False
    print("PASS test_model_originated_confirmation_never_counts")


def test_unrelated_history_entry_does_not_confirm():
    ledger = AgreementLedger()
    cid = ledger.add_item(topic="price", proposed_value="$50", speaker="p1",
                          evidence=[_evidence()])

    # History contains a record for a different speaker/action that is not a
    # valid participant confirmation of this item.
    assert ledger.record_confirmation(cid, "p1", "reject",
                                      source="participant", note="disagree") is True
    assert ledger.mark_confirmed_explicitly(cid) is False
    assert ledger.get(cid).status == "proposed"

    # Adding a valid participant confirmation unblocks confirmation.
    assert ledger.record_confirmation(cid, "you", "confirm",
                                      source="participant", note="agreed") is True
    assert ledger.mark_confirmed_explicitly(cid) is True
    assert ledger.get(cid).status == "confirmed"
    print("PASS test_unrelated_history_entry_does_not_confirm")


def test_malformed_confirmation_records_rejected():
    ledger = AgreementLedger()
    cid = ledger.add_item(topic="qty", proposed_value="100", speaker="p1",
                          evidence=[_evidence()])

    before = ledger.get(cid).confirmation_history

    # Invalid action.
    assert ledger.record_confirmation(cid, "p1", "maybe",
                                      source="participant") is False
    # Invalid source.
    assert ledger.record_confirmation(cid, "p1", "confirm",
                                      source="llm") is False
    # Empty speaker.
    assert ledger.record_confirmation(cid, "", "confirm",
                                      source="participant") is False
    assert ledger.record_confirmation(cid, "   ", "confirm",
                                      source="participant") is False
    # Unknown item.
    assert ledger.record_confirmation("nope", "p1", "confirm",
                                      source="participant") is False

    # History must be unchanged.
    assert ledger.get(cid).confirmation_history == before
    print("PASS test_malformed_confirmation_records_rejected")


def test_disputed_allowed_transitions_and_terminal_behavior():
    ledger = AgreementLedger()
    cid = ledger.add_item(topic="price", proposed_value="$50", speaker="p1",
                          evidence=[_evidence()])

    # proposed -> needs_clarification -> disputed
    assert ledger.transition(cid, "needs_clarification") is True
    assert ledger.transition(cid, "disputed") is True
    assert ledger.get(cid).status == "disputed"

    # disputed is NOT terminal: can go to needs_clarification / confirmed / rejected
    assert ledger.transition(cid, "needs_clarification") is True
    assert ledger.transition(cid, "disputed") is True

    # With a valid participant confirmation, disputed -> confirmed is allowed
    assert ledger.record_confirmation(cid, "you", "confirm",
                                      source="participant") is True
    assert ledger.transition(cid, "confirmed") is True
    assert ledger.get(cid).status == "confirmed"

    # confirmed is terminal: cannot leave
    assert ledger.transition(cid, "rejected") is False
    assert ledger.transition(cid, "disputed") is False
    assert ledger.get(cid).status == "confirmed"
    print("PASS test_disputed_allowed_transitions_and_terminal_behavior")


def test_confirmation_history_preserves_source_and_speaker():
    ledger = AgreementLedger()
    cid = ledger.add_item(topic="deadline", proposed_value="Friday",
                          speaker="you", evidence=[_evidence(uid=9, side="you")])
    assert ledger.record_confirmation(cid, "p1", "confirm",
                                      source="participant", note="agreed") is True
    assert ledger.record_confirmation(cid, "p1", "confirm",
                                      source="model", note="LLM guess") is True
    recs = ledger.get(cid).confirmation_history
    assert len(recs) == 2
    assert recs[0].source == "participant"
    assert recs[0].speaker == "p1"
    assert recs[0].action == "confirm"
    assert recs[1].source == "model"
    assert recs[1].action == "confirm"
    # Only the participant record counts.
    assert ledger._has_valid_participant_confirmation(ledger.get(cid)) is True
    print("PASS test_confirmation_history_preserves_source_and_speaker")


# dataclasses import for the frozen check above
import dataclasses  # noqa: E402


def test_evidence_frozen_raises():
    e = _evidence()
    try:
        e.original = "mutated"  # type: ignore[misc]
        raise AssertionError("TranscriptEvidence should be frozen")
    except dataclasses.FrozenInstanceError:
        pass
    print("PASS test_evidence_frozen_raises")


if __name__ == "__main__":
    test_default_status_is_proposed()
    test_valid_transitions()
    test_invalid_transition_rejected()
    test_invalid_status_rejected()
    test_unknown_item_rejected()
    test_evidence_is_immutable_and_append_only()
    test_cannot_confirm_without_history()
    test_explicit_confirmation_allows_confirmed()
    test_reject_records_history_and_terminal()
    test_needs_clarification_requires_explicit_question()
    test_duplicate_events_create_distinct_items()
    test_missing_information_remains_unknown()
    test_disputed_transition_path()
    test_items_by_status()
    test_model_originated_confirmation_never_counts()
    test_unrelated_history_entry_does_not_confirm()
    test_malformed_confirmation_records_rejected()
    test_disputed_allowed_transitions_and_terminal_behavior()
    test_confirmation_history_preserves_source_and_speaker()
    test_evidence_frozen_raises()
    print("\nAll agreement_ledger tests passed.")
