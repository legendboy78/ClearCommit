"""Focused tests for clearcommit_integration.py — Stage 6B hardened.

Run:  python test_clearcommit_integration.py

Covers: disabled mode, coaching result flow, authoritative evidence binding
(fabricated ids, wrong side, altered original, unknown requests), meeting
analysis on a dedicated async service (prompt submission, stale-result guard),
bounded evidence memory, closed-shutdown guard, runtime mode updates, and
bounded close.
"""

import json
import threading
import time

from ai_service import AIRequestStatus
from clearcommit_integration import ClearCommitIntegration
from mode_config import CoachingConfig, ModeConfig


def _mc(mode: str = "social", coaching: bool = True) -> ModeConfig:
    return ModeConfig(mode=mode, coaching=CoachingConfig(enabled=coaching))


def _make_integration(enabled: bool = True, mode: str = "social") -> ClearCommitIntegration:
    return ClearCommitIntegration(mode_config=_mc(mode, enabled))


def _utterance_json(kind: str = "idiom_explanation",
                    suggestion: str = "good luck") -> str:
    """Model output that fabricates provenance (must be ignored/bound)."""
    return json.dumps({
        "kind": kind,
        "original": "MODEL-FABRICATED-ORIGINAL",
        "suggestion": suggestion,
        "reason": "why",
        "evidence_uid": 999,       # fabricated -> must be overridden
        "side": "p1",              # wrong side -> must be overridden
    })


def test_disabled_integration_no_callbacks():
    """When coaching is disabled, the integration does nothing."""
    integration = _make_integration(enabled=False)
    calls = []
    integration.set_ui_callback(lambda: calls.append(1))

    integration.on_final_utterance(
        side="you", source_lang="en", target_lang="es",
        original_text="hello world", translated_text="hola mundo",
        uid=1, accept_ts=1000.0,
    )
    assert len(calls) == 0
    assert integration.take_coaching_results() == []
    assert integration.take_meeting_record() is None
    integration.close()
    print("PASS test_disabled_integration_no_callbacks")


def test_enabled_integration_coaching_result():
    """Coaching results flow through when coaching is enabled."""
    class FakeProvider:
        def analyze(self, request, context=None):
            return _utterance_json()

    integration = _make_integration(enabled=True, mode="social")
    integration.social_coach.set_provider(FakeProvider())

    results = []
    integration.set_ui_callback(lambda: results.append(1))

    integration.on_final_utterance(
        side="you", source_lang="en", target_lang="es",
        original_text="break a leg", translated_text="mucha suerte",
        uid=5, accept_ts=1000.0,
    )

    deadline = time.time() + 5
    while time.time() < deadline:
        if results:
            break
        time.sleep(0.01)

    coaching = integration.take_coaching_results()
    assert len(coaching) == 1
    assert coaching[0].kind == "idiom_explanation"
    assert coaching[0].original == "break a leg"
    assert coaching[0].suggestion == "good luck"
    assert coaching[0].evidence_uid == 5
    integration.close()
    print("PASS test_enabled_integration_coaching_result")


def test_provenance_bound_authoritatively():
    """Fix 2: fabricated evidence_uid/side/original are ignored; trusted
    provenance comes from the request/evidence record."""
    class FabricatingProvider:
        def analyze(self, request, context=None):
            return _utterance_json()  # claims uid=999, side=p1, HACKED original

    integration = _make_integration(enabled=True, mode="social")
    integration.social_coach.set_provider(FabricatingProvider())

    integration.on_final_utterance(
        side="you", source_lang="en", target_lang="es",
        original_text="break a leg", translated_text="x",
        uid=7, accept_ts=1000.0,
    )
    deadline = time.time() + 5
    coaching = []
    while time.time() < deadline:
        coaching = integration.take_coaching_results()
        if coaching:
            break
        time.sleep(0.01)

    assert len(coaching) == 1
    c = coaching[0]
    assert c.evidence_uid == 7          # NOT the fabricated 999
    assert c.side == "you"              # NOT the fabricated p1
    assert c.original == "break a leg"  # NOT "MODEL-FABRICATED-ORIGINAL"
    integration.close()
    print("PASS test_provenance_bound_authoritatively")


def test_unknown_request_result_rejected():
    """Fix 2: results for unknown/evicted uids are never displayed."""
    from ai_service import AIRequestStatus, AIResult
    integration = _make_integration(enabled=True)

    # No utterance with uid 4242 was ever seen.
    integration.on_ai_result(AIResult(
        request_id="4242", status=AIRequestStatus.COMPLETED,
        output=_utterance_json(),
    ))
    assert integration.take_coaching_results() == []
    integration.close()
    print("PASS test_unknown_request_result_rejected")


def test_malformed_result_rejected():
    """Fix 2: malformed payloads / unknown kinds are rejected."""
    from ai_service import AIRequestStatus, AIResult
    integration = _make_integration(enabled=True, mode="social")
    integration.on_final_utterance(
        side="you", source_lang="en", target_lang="es",
        original_text="break a leg", translated_text="x",
        uid=3, accept_ts=1000.0,
    )
    bad_payloads = [
        "not json {{{",
        json.dumps({"kind": "totally_unknown_kind", "suggestion": "x"}),
        json.dumps({"kind": "idiom_explanation"}),                    # no suggestion
        json.dumps({"kind": "idiom_explanation", "suggestion": "  "}),  # blank
        json.dumps(["not", "a", "dict"]),
    ]
    for payload in bad_payloads:
        integration.on_ai_result(AIResult(
            request_id="3", status=AIRequestStatus.COMPLETED, output=payload,
        ))
        assert integration.take_coaching_results() == [], payload
    # Non-COMPLETED statuses are ignored too.
    integration.on_ai_result(AIResult(
        request_id="3", status=AIRequestStatus.FAILED, output="x",
    ))
    assert integration.take_coaching_results() == []
    integration.close()
    print("PASS test_malformed_result_rejected")


def test_evidence_memory_bounded():
    """Fix 4: evidence map and transcript parts are evicted in lockstep."""
    integration = _make_integration(enabled=False)  # coaching off; store only
    total = 200
    for i in range(1, total + 1):
        integration.on_final_utterance(
            side="you", source_lang="en", target_lang="es",
            original_text=f"utterance number {i}", translated_text="x",
            uid=i, accept_ts=1000.0 + i,
        )
    assert len(integration._evidence_map) <= ClearCommitIntegration.MAX_EVIDENCE_ENTRIES
    assert len(integration._transcript_parts) <= ClearCommitIntegration.MAX_EVIDENCE_ENTRIES
    # Lockstep: exactly the newest MAX utterances are retained.
    assert len(integration._evidence_map) == len(integration._transcript_parts)
    retained = sorted(integration._evidence_map.keys())
    assert retained == list(range(total + 1 - ClearCommitIntegration.MAX_EVIDENCE_ENTRIES,
                                  total + 1))
    integration.close()
    print("PASS test_evidence_memory_bounded")


def test_meeting_analysis_is_async_and_prompt():
    """Fix 3: submission returns promptly even while the provider is blocked;
    no per-utterance thread is spawned."""
    release = threading.Event()

    class BlockingProvider:
        def analyze(self, text, context=None):
            release.wait()
            return json.dumps({
                "decisions": [{"topic": "price", "proposed_value": "$50",
                               "decided_value": "$50", "status": "decided",
                               "evidence_ids": [1], "speaker": "you"}],
            })

    integration = _make_integration(enabled=True, mode="professional")
    integration.meeting_intelligence.set_provider(BlockingProvider())
    threads_before = threading.active_count()

    t0 = time.perf_counter()
    integration.on_final_utterance(
        side="you", source_lang="en", target_lang="es",
        original_text="the price is $50", translated_text="x",
        uid=1, accept_ts=1000.0,
    )
    submit_ms = (time.perf_counter() - t0) * 1000.0
    assert submit_ms < 1000  # prompt: provider still blocked, no sync call

    release.set()
    record = None
    deadline = time.time() + 5
    while time.time() < deadline:
        record = integration.take_meeting_record()
        if record is not None:
            break
        time.sleep(0.01)
    assert record is not None
    assert len(record.decisions) == 1
    assert record.decisions[0]["topic"] == "price"
    # No thread-per-utterance: worker count stays bounded (service threads).
    assert threading.active_count() <= threads_before + 4
    integration.close()
    print("PASS test_meeting_analysis_is_async_and_prompt")


def _record_dict(meeting_id: str) -> dict:
    return {
        "meeting_id": meeting_id,
        "topics": ["pricing"],
        "decisions": [{"topic": "price", "proposed_value": "$50",
                       "decided_value": "$50", "status": "decided",
                       "evidence": [], "speaker": "you", "id": "d1",
                       "created_ts": 1.0}],
        "action_items": [],
        "open_questions": [],
        "candidate_commitments": [],
    }


def test_stale_meeting_result_never_overwrites_newer():
    """Fix 3: a late/older meeting result cannot replace a newer record."""
    from ai_service import AIRequestStatus, AIResult
    integration = _make_integration(enabled=True)

    integration.on_meeting_result(AIResult(
        request_id="10", status=AIRequestStatus.COMPLETED,
        output=json.dumps(_record_dict("newer")),
    ))
    newer = integration.take_meeting_record()
    assert newer is not None and newer.meeting_id == "newer"

    # Late result for an older uid must be dropped.
    integration.on_meeting_result(AIResult(
        request_id="5", status=AIRequestStatus.COMPLETED,
        output=json.dumps(_record_dict("stale")),
    ))
    assert integration.take_meeting_record() is None
    integration.close()
    print("PASS test_stale_meeting_result_never_overwrites_newer")


def test_closed_integration_is_inert():
    """Fix 5: after close(), every entry point is a safe no-op."""
    integration = _make_integration(enabled=True, mode="social")
    integration.close()
    integration.close()  # idempotent

    integration.on_final_utterance(
        side="you", source_lang="en", target_lang="es",
        original_text="after close", translated_text="x",
        uid=1, accept_ts=1000.0,
    )
    assert integration._evidence_map == {}          # nothing stored
    assert integration.take_coaching_results() == []
    assert integration.take_meeting_record() is None
    print("PASS test_closed_integration_is_inert")


def test_meeting_intelligence_accumulates():
    """Meeting intelligence accumulates utterances and produces a record."""
    class FakeProvider:
        def analyze(self, text, context=None):
            return json.dumps({
                "decisions": [{"topic": "price", "proposed_value": "$50",
                               "decided_value": "$50", "status": "decided",
                               "evidence_ids": [1], "speaker": "you"}],
            })

    integration = _make_integration(enabled=True, mode="professional")
    integration.meeting_intelligence.set_provider(FakeProvider())

    integration.on_final_utterance(
        side="you", source_lang="en", target_lang="es",
        original_text="the price is $50", translated_text="el precio es $50",
        uid=1, accept_ts=1000.0,
    )

    record = None
    deadline = time.time() + 5
    while time.time() < deadline:
        record = integration.take_meeting_record()
        if record is not None:
            break
        time.sleep(0.01)
    assert record is not None
    assert len(record.decisions) == 1
    assert record.decisions[0]["topic"] == "price"
    assert record.decisions[0]["proposed_value"] == "$50"
    integration.close()
    print("PASS test_meeting_intelligence_accumulates")


def test_update_mode_config_at_runtime():
    """Mode config can be updated at runtime (dispatcher observes it too)."""
    integration = _make_integration(enabled=False, mode="social")
    assert integration.mode_config.mode == "social"

    integration.update_mode_config(_mc("professional", True))
    assert integration.mode_config.mode == "professional"
    assert integration.mode_config.coaching_active() is True
    assert integration.is_enabled() is True
    assert integration._meeting_service.is_enabled() is True
    integration.close()
    print("PASS test_update_mode_config_at_runtime")


def test_integration_close_is_bounded():
    """close() returns quickly even with pending work."""
    integration = _make_integration(enabled=True)
    t0 = time.time()
    integration.close()
    elapsed = (time.time() - t0) * 1000.0
    assert elapsed < 5000  # bounded (both services)
    print("PASS test_integration_close_is_bounded")


def test_is_enabled_property():
    integration = _make_integration(enabled=True)
    assert integration.is_enabled() is True
    integration.close()
    print("PASS test_is_enabled_property")


# --------------------------------------------------------------------------- #
# Stage 7 — agreement ledger wiring + human-only confirmation workflow
# --------------------------------------------------------------------------- #


def _commitment_json(topic: str = "price", value: str = "$50",
                     evidence_ids: str = "[1]") -> str:
    return json.dumps({
        "candidate_commitments": [{
            "topic": topic,
            "proposed_value": value,
            "status": "inferred",
            "evidence_ids": json.loads(evidence_ids),
            "speaker": "you",
        }],
    })


def _wait_for_items(integration, n: int = 1, timeout_s: float = 5.0):
    """Poll the agreement snapshot until at least ``n`` items exist."""
    deadline = time.time() + timeout_s
    version, items = integration.get_agreement_snapshot()
    while time.time() < deadline and len(items) < n:
        time.sleep(0.01)
        version, items = integration.get_agreement_snapshot()
    return version, items


def test_candidate_commitments_ingest_into_ledger():
    """Validated candidate commitments become 'proposed' ledger items."""
    class CommitmentProvider:
        def analyze(self, text, context=None):
            return _commitment_json()

    integration = _make_integration(enabled=True, mode="professional")
    integration.meeting_intelligence.set_provider(CommitmentProvider())

    integration.on_final_utterance(
        side="you", source_lang="en", target_lang="es",
        original_text="we agree on $50", translated_text="x",
        uid=1, accept_ts=1000.0,
    )
    _version, items = _wait_for_items(integration, 1)
    assert len(items) == 1
    item = items[0]
    assert item["topic"] == "price"
    assert item["value"] == "$50"
    assert item["status"] == "proposed"
    assert item["evidence_uids"] == [1]
    assert item["confirmed_by"] == []
    integration.close()
    print("PASS test_candidate_commitments_ingest_into_ledger")


def test_confirmation_requires_both_participants():
    """Fix/human-only: item confirmed only after BOTH sides explicitly
    confirm via the human API; model output can never confirm."""
    class CommitmentProvider:
        def analyze(self, text, context=None):
            return _commitment_json()

    integration = _make_integration(enabled=True, mode="professional")
    integration.meeting_intelligence.set_provider(CommitmentProvider())
    integration.on_final_utterance(
        side="you", source_lang="en", target_lang="es",
        original_text="we agree on $50", translated_text="x",
        uid=1, accept_ts=1000.0,
    )
    _version, items = _wait_for_items(integration, 1)
    item_id = items[0]["id"]

    # One side confirms -> still proposed.
    assert integration.confirm_commitment(item_id, "you") is True
    _version, items = integration.get_agreement_snapshot()
    assert items[0]["status"] == "proposed"
    assert items[0]["confirmed_by"] == ["you"]

    # Both sides -> confirmed.
    assert integration.confirm_commitment(item_id, "p1") is True
    _version, items = integration.get_agreement_snapshot()
    assert items[0]["status"] == "confirmed"

    # Terminal: further confirmations are refused.
    assert integration.confirm_commitment(item_id, "you") is False
    integration.close()
    print("PASS test_confirmation_requires_both_participants")


def test_reject_marks_rejected():
    """A single explicit rejection by either participant rejects the item."""
    class CommitmentProvider:
        def analyze(self, text, context=None):
            return _commitment_json(topic="deadline", value="Friday")

    integration = _make_integration(enabled=True, mode="professional")
    integration.meeting_intelligence.set_provider(CommitmentProvider())
    integration.on_final_utterance(
        side="you", source_lang="en", target_lang="es",
        original_text="deadline is Friday", translated_text="x",
        uid=1, accept_ts=1000.0,
    )
    _version, items = _wait_for_items(integration, 1)
    item_id = items[0]["id"]
    assert integration.reject_commitment(item_id, "you") is True
    _version, items = integration.get_agreement_snapshot()
    assert items[0]["status"] == "rejected"
    # Terminal: cannot confirm a rejected item.
    assert integration.confirm_commitment(item_id, "p1") is False
    integration.close()
    print("PASS test_reject_marks_rejected")


def test_duplicate_commitment_not_reingested():
    """The same (topic, value) across successive records ingests once."""
    calls = {"n": 0}

    class CommitmentProvider:
        def analyze(self, text, context=None):
            calls["n"] += 1
            return _commitment_json()

    integration = _make_integration(enabled=True, mode="professional")
    integration.meeting_intelligence.set_provider(CommitmentProvider())
    integration.on_final_utterance(
        side="you", source_lang="en", target_lang="es",
        original_text="we agree on $50", translated_text="x",
        uid=1, accept_ts=1000.0,
    )
    # Second, newer record repeating the same commitment.
    integration.on_final_utterance(
        side="you", source_lang="en", target_lang="es",
        original_text="we agree on $50", translated_text="x",
        uid=2, accept_ts=1001.0,
    )
    deadline = time.time() + 5
    while time.time() < deadline and calls["n"] < 2:
        time.sleep(0.01)
    _version, items = integration.get_agreement_snapshot()
    assert len(items) == 1  # deduped, not re-ingested
    integration.close()
    print("PASS test_duplicate_commitment_not_reingested")


def test_conflicting_commitments_kept_separate():
    """Conflicting proposals ($50 vs $100) are NOT merged."""
    class TwoCommitmentsProvider:
        def analyze(self, text, context=None):
            return json.dumps({
                "candidate_commitments": [
                    {"topic": "price", "proposed_value": "$50",
                     "status": "inferred", "evidence_ids": [1], "speaker": "you"},
                    {"topic": "price", "proposed_value": "$100",
                     "status": "inferred", "evidence_ids": [1], "speaker": "p1"},
                ],
            })

    integration = _make_integration(enabled=True, mode="professional")
    integration.meeting_intelligence.set_provider(TwoCommitmentsProvider())
    integration.on_final_utterance(
        side="you", source_lang="en", target_lang="es",
        original_text="fifty? no, one hundred", translated_text="x",
        uid=1, accept_ts=1000.0,
    )
    deadline = time.time() + 5
    _version, items = integration.get_agreement_snapshot()
    while time.time() < deadline and len(items) < 2:
        time.sleep(0.01)
        _version, items = integration.get_agreement_snapshot()
    values = {i["value"] for i in items}
    assert values == {"$50", "$100"}  # both preserved
    integration.close()
    print("PASS test_conflicting_commitments_kept_separate")


def test_invalid_speaker_confirmation_rejected():
    """Non-participant speakers (e.g. 'model') can never confirm/reject."""
    class CommitmentProvider:
        def analyze(self, text, context=None):
            return _commitment_json()

    integration = _make_integration(enabled=True, mode="professional")
    integration.meeting_intelligence.set_provider(CommitmentProvider())
    integration.on_final_utterance(
        side="you", source_lang="en", target_lang="es",
        original_text="we agree on $50", translated_text="x",
        uid=1, accept_ts=1000.0,
    )
    _version, items = _wait_for_items(integration, 1)
    item_id = items[0]["id"]

    for bad in ("model", "", "llm", "system", "P1"):
        assert integration.confirm_commitment(item_id, bad) is False, bad
        assert integration.reject_commitment(item_id, bad) is False, bad
    _version, items = integration.get_agreement_snapshot()
    assert items[0]["status"] == "proposed"  # unchanged
    assert items[0]["confirmed_by"] == []
    integration.close()
    print("PASS test_invalid_speaker_confirmation_rejected")


def test_confirmation_after_close_is_refused():
    """Fix 5: ledger mutations are refused after close()."""
    class CommitmentProvider:
        def analyze(self, text, context=None):
            return _commitment_json()

    integration = _make_integration(enabled=True, mode="professional")
    integration.meeting_intelligence.set_provider(CommitmentProvider())
    integration.on_final_utterance(
        side="you", source_lang="en", target_lang="es",
        original_text="we agree on $50", translated_text="x",
        uid=1, accept_ts=1000.0,
    )
    _version, items = _wait_for_items(integration, 1)
    item_id = items[0]["id"] if items else None
    integration.close()

    integration.on_final_utterance(
        side="you", source_lang="en", target_lang="es",
        original_text="after close", translated_text="x",
        uid=2, accept_ts=1001.0,
    )
    _version, items = integration.get_agreement_snapshot()
    assert len(items) <= 1  # nothing new ingested after close
    if item_id is not None:
        assert integration.confirm_commitment(item_id, "you") is False
    print("PASS test_confirmation_after_close_is_refused")


# --------------------------------------------------------------------------- #
# Stage A — conversation assistant integration
# --------------------------------------------------------------------------- #


def test_assistant_ask_and_reply_flow():
    """A grounded question flows to the provider and the reply drains."""
    from assistant import FakeAssistantProvider
    integration = _make_integration(enabled=True, mode="professional")
    integration._assistant.set_provider(FakeAssistantProvider(
        reply="The price $50 was agreed [E1]."))
    # Seed transcript evidence so [E1] exists in the grounding window.
    integration.on_final_utterance(
        side="you", source_lang="en", target_lang="es",
        original_text="we agree on $50", translated_text="x",
        uid=1, accept_ts=1000.0,
    )

    handle = integration.ask_assistant("what was agreed?")
    assert handle.status == AIRequestStatus.PENDING

    deadline = time.time() + 5
    reply = None
    while time.time() < deadline:
        reply = integration.take_assistant_reply()
        if reply is not None:
            break
        time.sleep(0.01)
    assert reply is not None
    assert reply["ok"] is True
    assert reply["question"] == "what was agreed?"
    assert "$50" in reply["reply"]
    assert reply["cited_uids"] == [1]
    assert integration.is_assistant_busy() is False
    integration.close()
    print("PASS test_assistant_ask_and_reply_flow")


def test_assistant_disabled_when_coaching_off():
    integration = _make_integration(enabled=False)
    handle = integration.ask_assistant("hello?")
    assert handle.status == AIRequestStatus.FAILED
    assert "ai_service_disabled" in handle.error
    integration.close()
    print("PASS test_assistant_disabled_when_coaching_off")


def test_assistant_unavailable_without_provider():
    integration = _make_integration(enabled=True)
    handle = integration.ask_assistant("hello?")
    assert handle.status == AIRequestStatus.PENDING

    reply = None
    deadline = time.time() + 5
    while time.time() < deadline:
        reply = integration.take_assistant_reply()
        if reply is not None:
            break
        time.sleep(0.01)
    assert reply is not None
    assert reply["ok"] is False
    assert reply["error"] == "assistant_unavailable"
    integration.close()
    print("PASS test_assistant_unavailable_without_provider")


def test_assistant_empty_question_rejected_locally():
    integration = _make_integration(enabled=True)
    handle = integration.ask_assistant("   ")
    assert handle.status == AIRequestStatus.FAILED
    assert handle.error == "empty_question"
    integration.close()
    print("PASS test_assistant_empty_question_rejected_locally")


def test_assistant_clear_history():
    from assistant import FakeAssistantProvider
    integration = _make_integration(enabled=True)
    integration._assistant.set_provider(FakeAssistantProvider(reply="ok"))
    integration.ask_assistant("q1")
    deadline = time.time() + 5
    while time.time() < deadline and not integration._assistant.history():
        time.sleep(0.01)
    assert integration._assistant.history()
    integration.clear_assistant_history()
    assert integration._assistant.history() == []
    integration.close()
    print("PASS test_assistant_clear_history")


if __name__ == "__main__":
    test_disabled_integration_no_callbacks()
    test_enabled_integration_coaching_result()
    test_provenance_bound_authoritatively()
    test_unknown_request_result_rejected()
    test_malformed_result_rejected()
    test_evidence_memory_bounded()
    test_meeting_analysis_is_async_and_prompt()
    test_stale_meeting_result_never_overwrites_newer()
    test_closed_integration_is_inert()
    test_meeting_intelligence_accumulates()
    test_update_mode_config_at_runtime()
    test_integration_close_is_bounded()
    test_is_enabled_property()
    test_candidate_commitments_ingest_into_ledger()
    test_confirmation_requires_both_participants()
    test_reject_marks_rejected()
    test_duplicate_commitment_not_reingested()
    test_conflicting_commitments_kept_separate()
    test_invalid_speaker_confirmation_rejected()
    test_confirmation_after_close_is_refused()
    test_assistant_ask_and_reply_flow()
    test_assistant_disabled_when_coaching_off()
    test_assistant_unavailable_without_provider()
    test_assistant_empty_question_rejected_locally()
    test_assistant_clear_history()
    print("\nAll clearcommit_integration tests passed.")
