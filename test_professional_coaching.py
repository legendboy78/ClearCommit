"""Focused tests for professional_coaching.py — plain Python, no dependencies, no network.

Run:  python test_professional_coaching.py

Covers: disabled mode, missing provider, empty text, communication coach
detection (aggression + complexity), pitch coach detection, no-coaching-needed
pre-filter, provider errors, malformed output, code-fence stripping,
kind="none", kind mismatch, missing fields, deduplication, text truncation,
detection flags, thread-safety, runtime toggle, and provider swap.
"""

import threading

from professional_coaching import (
    CoachingSuggestion,
    FakeCoachingProvider,
    ProfessionalCoach,
)


def _coach(enabled: bool = True, provider=None, **kwargs) -> ProfessionalCoach:
    return ProfessionalCoach(enabled=enabled, provider=provider, **kwargs)


def _make_output(kind: str, original: str, suggestion: str,
                 reason: str = "why it helps") -> str:
    import json
    return json.dumps({
        "kind": kind,
        "original": original,
        "suggestion": suggestion,
        "reason": reason,
    })


def test_disabled_returns_none():
    provider = FakeCoachingProvider(output=_make_output(
        "communication_coach", "you always", "i notice that"))
    coach = _coach(enabled=False, provider=provider)
    result = coach.analyze("you always miss deadlines", side="you", uid=1)
    assert result is None
    assert provider.call_count == 0
    print("PASS test_disabled_returns_none")


def test_no_provider_returns_none():
    coach = _coach(enabled=True, provider=None)
    assert coach.analyze("you always", side="you", uid=1) is None
    print("PASS test_no_provider_returns_none")


def test_empty_text_returns_none():
    provider = FakeCoachingProvider(output=_make_output(
        "communication_coach", "x", "y"))
    coach = _coach(enabled=True, provider=provider)
    for text in ("", "   ", None):
        assert coach.analyze(text, side="you", uid=1) is None
    assert provider.call_count == 0
    print("PASS test_empty_text_returns_none")


def test_communication_coach_aggression_detection():
    provider = FakeCoachingProvider(output=_make_output(
        "communication_coach", "you always miss deadlines",
        "i notice deadlines are often missed"))
    coach = _coach(enabled=True, provider=provider)
    result = coach.analyze("you always miss deadlines", side="you", uid=5)
    assert result is not None
    assert isinstance(result, CoachingSuggestion)
    assert result.kind == "communication_coach"
    assert result.original == "you always miss deadlines"
    assert result.suggestion == "i notice deadlines are often missed"
    assert result.evidence_uid == 5
    assert result.side == "you"
    assert provider.call_count == 1
    print("PASS test_communication_coach_aggression_detection")


def test_communication_coach_complexity_detection():
    provider = FakeCoachingProvider(output=_make_output(
        "communication_coach", "utilize this solution",
        "use this solution"))
    coach = _coach(enabled=True, provider=provider)
    result = coach.analyze("please utilize this solution", side="p1", uid=2)
    assert result is not None
    assert result.kind == "communication_coach"
    assert result.original == "utilize this solution"
    print("PASS test_communication_coach_complexity_detection")


def test_pitch_coach_detection():
    provider = FakeCoachingProvider(output=_make_output(
        "pitch_coach", "our product saves you money",
        "our product reduces costs by 20%", "specific measurable value"))
    coach = _coach(enabled=True, provider=provider)
    result = coach.analyze("our product saves you money", side="you", uid=8)
    assert result is not None
    assert result.kind == "pitch_coach"
    assert result.original == "our product saves you money"
    assert result.suggestion == "our product reduces costs by 20%"
    assert result.evidence_uid == 8
    assert result.side == "you"
    print("PASS test_pitch_coach_detection")


def test_no_coaching_needed_pre_filter():
    provider = FakeCoachingProvider(output=_make_output(
        "communication_coach", "x", "y"))
    coach = _coach(enabled=True, provider=provider)
    result = coach.analyze("hello how is your day", side="you", uid=3)
    assert result is None
    assert provider.call_count == 0
    print("PASS test_no_coaching_needed_pre_filter")


def test_provider_error_graceful():
    provider = FakeCoachingProvider(output=None, fail=True)
    coach = _coach(enabled=True, provider=provider)
    result = coach.analyze("you always", side="you", uid=1)
    assert result is None
    assert provider.call_count == 1
    print("PASS test_provider_error_graceful")


def test_malformed_json_rejected():
    provider = FakeCoachingProvider(output="not json {{{")
    coach = _coach(enabled=True, provider=provider)
    result = coach.analyze("you always", side="you", uid=1)
    assert result is None
    print("PASS test_malformed_json_rejected")


def test_code_fence_stripped():
    provider = FakeCoachingProvider(output=(
        "```json\n{\n"
        '  "kind": "communication_coach",\n'
        '  "original": "this is ridiculous",\n'
        '  "suggestion": "this is concerning",\n'
        '  "reason": "less confrontational"\n'
        "}\n```"
    ))
    coach = _coach(enabled=True, provider=provider)
    result = coach.analyze("this is ridiculous", side="you", uid=4)
    assert result is not None
    assert result.kind == "communication_coach"
    assert result.original == "this is ridiculous"
    assert result.suggestion == "this is concerning"
    print("PASS test_code_fence_stripped")


def test_kind_none_returns_none():
    provider = FakeCoachingProvider(output=_make_output("none", "", ""))
    coach = _coach(enabled=True, provider=provider)
    assert coach.analyze("you always", side="you", uid=1) is None
    print("PASS test_kind_none_returns_none")


def test_kind_mismatch_rejected():
    provider = FakeCoachingProvider(output=_make_output(
        "pitch_coach", "x", "y"))
    coach = _coach(enabled=True, provider=provider)
    result = coach.analyze("you always miss deadlines", side="you", uid=1)
    assert result is None
    print("PASS test_kind_mismatch_rejected")


def test_missing_required_fields_rejected():
    cases = [
        '{"kind": "communication_coach"}',
        '{"kind": "communication_coach", "original": "x"}',
        '{"kind": "communication_coach", "suggestion": "y"}',
        '{"kind": "communication_coach", "original": "", "suggestion": "y"}',
        '{"kind": "communication_coach", "original": "x", "suggestion": ""}',
    ]
    for bad in cases:
        provider = FakeCoachingProvider(output=bad)
        coach = _coach(enabled=True, provider=provider)
        result = coach.analyze("you always", side="you", uid=1)
        assert result is None, bad
    print("PASS test_missing_required_fields_rejected")


def test_deduplication_same_phrase_once():
    provider = FakeCoachingProvider(output=_make_output(
        "communication_coach", "you always", "i notice"))
    coach = _coach(enabled=True, provider=provider)
    r1 = coach.analyze("you always interrupt", side="you", uid=1)
    assert r1 is not None
    r2 = coach.analyze("you always interrupt", side="you", uid=2)
    assert r2 is None
    assert provider.call_count == 1
    print("PASS test_deduplication_same_phrase_once")


def test_deduplication_different_phrases_both_coached():
    provider = FakeCoachingProvider(output=_make_output(
        "communication_coach", "you always", "i notice"))
    coach = _coach(enabled=True, provider=provider, dedupe_window=10)
    r1 = coach.analyze("you always", side="you", uid=1)
    r2 = coach.analyze("this is ridiculous", side="you", uid=2)
    assert r1 is not None
    assert r2 is not None
    assert provider.call_count == 2
    print("PASS test_deduplication_different_phrases_both_coached")


def test_text_truncation_at_max_chars():
    provider = FakeCoachingProvider(output=_make_output(
        "communication_coach", "you always", "i notice"))
    coach = _coach(enabled=True, provider=provider, max_text_chars=30)
    long_text = "you always interrupt me and then some extra words here"
    result = coach.analyze(long_text, side="you", uid=1)
    assert result is not None
    assert result.original == "you always"
    print("PASS test_text_truncation_at_max_chars")


def test_detection_flags_separately():
    provider = FakeCoachingProvider(output=_make_output(
        "communication_coach", "you always", "i notice"))
    coach = _coach(enabled=True, provider=provider,
                   detect_communication=False, detect_pitch=True)
    assert coach.analyze("you always miss deadlines", side="you", uid=1) is None

    provider2 = FakeCoachingProvider(output=_make_output(
        "pitch_coach", "our product", "our product adds value"))
    coach2 = _coach(enabled=True, provider=provider2,
                    detect_communication=True, detect_pitch=False)
    assert coach2.analyze("our product saves money", side="you", uid=1) is None
    print("PASS test_detection_flags_separately")


def test_thread_safety_concurrent_calls():
    provider = FakeCoachingProvider(output=_make_output(
        "communication_coach", "you always", "i notice"))
    coach = _coach(enabled=True, provider=provider)
    results = []
    errors = []
    lock = threading.Lock()

    def worker(uid: int):
        try:
            r = coach.analyze("you always", side="you", uid=uid)
            with lock:
                results.append(r)
        except Exception as exc:  # noqa: BLE001
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, errors
    suggestions = [r for r in results if r is not None]
    assert len(suggestions) == 1, suggestions
    assert suggestions[0].kind == "communication_coach"
    print("PASS test_thread_safety_concurrent_calls")


def test_toggle_enabled_at_runtime():
    provider = FakeCoachingProvider(output=_make_output(
        "communication_coach", "you always", "i notice"))
    coach = _coach(enabled=False, provider=provider)
    assert coach.is_enabled() is False
    assert coach.analyze("you always", side="you", uid=1) is None

    coach.set_enabled(True)
    assert coach.is_enabled() is True
    result = coach.analyze("you always", side="you", uid=1)
    assert result is not None
    print("PASS test_toggle_enabled_at_runtime")


def test_set_provider_at_runtime():
    coach = _coach(enabled=True, provider=None)
    assert coach.analyze("you always", side="you", uid=1) is None

    provider = FakeCoachingProvider(output=_make_output(
        "communication_coach", "you always", "i notice"))
    coach.set_provider(provider)
    result = coach.analyze("you always", side="you", uid=1)
    assert result is not None
    print("PASS test_set_provider_at_runtime")


if __name__ == "__main__":
    test_disabled_returns_none()
    test_no_provider_returns_none()
    test_empty_text_returns_none()
    test_communication_coach_aggression_detection()
    test_communication_coach_complexity_detection()
    test_pitch_coach_detection()
    test_no_coaching_needed_pre_filter()
    test_provider_error_graceful()
    test_malformed_json_rejected()
    test_code_fence_stripped()
    test_kind_none_returns_none()
    test_kind_mismatch_rejected()
    test_missing_required_fields_rejected()
    test_deduplication_same_phrase_once()
    test_deduplication_different_phrases_both_coached()
    test_text_truncation_at_max_chars()
    test_detection_flags_separately()
    test_thread_safety_concurrent_calls()
    test_toggle_enabled_at_runtime()
    test_set_provider_at_runtime()
    print("\nAll professional_coaching tests passed.")
