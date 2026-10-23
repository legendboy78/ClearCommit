"""Focused tests for social_coaching.py — plain Python, no dependencies, no network.

Run:  python test_social_coaching.py

Covers: disabled mode, missing provider, empty text, idiom detection, tone
detection, no-coaching-needed pre-filter, provider errors, malformed output,
code-fence stripping, kind="none", kind mismatch, missing fields, deduplication,
text truncation, and thread-safety.
"""

import threading

from social_coaching import (
    CoachingSuggestion,
    FakeCoachingProvider,
    SocialCoach,
)


def _coach(enabled: bool = True, provider=None, **kwargs) -> SocialCoach:
    return SocialCoach(enabled=enabled, provider=provider, **kwargs)


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
        "idiom_explanation", "break a leg", "good luck"))
    coach = _coach(enabled=False, provider=provider)
    result = coach.analyze("break a leg", side="you", uid=1)
    assert result is None
    # Provider must not be consulted when coaching is disabled.
    assert provider.call_count == 0
    print("PASS test_disabled_returns_none")


def test_no_provider_returns_none():
    coach = _coach(enabled=True, provider=None)
    result = coach.analyze("break a leg", side="you", uid=1)
    assert result is None
    print("PASS test_no_provider_returns_none")


def test_empty_text_returns_none():
    provider = FakeCoachingProvider(output=_make_output(
        "idiom_explanation", "x", "y"))
    coach = _coach(enabled=True, provider=provider)
    for text in ("", "   ", None):
        assert coach.analyze(text, side="you", uid=1) is None
    assert provider.call_count == 0
    print("PASS test_empty_text_returns_none")


def test_idiom_detection():
    provider = FakeCoachingProvider(output=_make_output(
        "idiom_explanation", "break a leg", "good luck"))
    coach = _coach(enabled=True, provider=provider)
    result = coach.analyze("break a leg", side="you", uid=7)
    assert result is not None
    assert isinstance(result, CoachingSuggestion)
    assert result.kind == "idiom_explanation"
    assert result.original == "break a leg"
    assert result.suggestion == "good luck"
    assert result.evidence_uid == 7
    assert result.side == "you"
    assert provider.call_count == 1
    print("PASS test_idiom_detection")


def test_tone_detection():
    provider = FakeCoachingProvider(output=_make_output(
        "tone_suggestion", "as i said", "as mentioned earlier",
        "less repetitive"))
    coach = _coach(enabled=True, provider=provider)
    result = coach.analyze("as i said before", side="p1", uid=3)
    assert result is not None
    assert result.kind == "tone_suggestion"
    assert result.original == "as i said"
    assert result.suggestion == "as mentioned earlier"
    assert result.evidence_uid == 3
    assert result.side == "p1"
    print("PASS test_tone_detection")


def test_no_coaching_needed_pre_filter():
    """Ordinary text without idiom/tone hints should not consult the model."""
    provider = FakeCoachingProvider(output=_make_output(
        "idiom_explanation", "x", "y"))
    coach = _coach(enabled=True, provider=provider)
    result = coach.analyze("hello how are you today", side="you", uid=2)
    assert result is None
    assert provider.call_count == 0
    print("PASS test_no_coaching_needed_pre_filter")


def test_provider_error_graceful():
    provider = FakeCoachingProvider(output=None, fail=True)
    coach = _coach(enabled=True, provider=provider)
    result = coach.analyze("break a leg", side="you", uid=1)
    assert result is None
    assert provider.call_count == 1
    print("PASS test_provider_error_graceful")


def test_malformed_json_rejected():
    provider = FakeCoachingProvider(output="not json at all {{{")
    coach = _coach(enabled=True, provider=provider)
    result = coach.analyze("break a leg", side="you", uid=1)
    assert result is None
    print("PASS test_malformed_json_rejected")


def test_code_fence_stripped():
    provider = FakeCoachingProvider(output=(
        "```json\n"
        "{\n"
        '  "kind": "idiom_explanation",\n'
        '  "original": "piece of cake",\n'
        '  "suggestion": "very easy",\n'
        '  "reason": "plain language"\n'
        "}\n"
        "```"
    ))
    coach = _coach(enabled=True, provider=provider)
    result = coach.analyze("piece of cake", side="you", uid=4)
    assert result is not None
    assert result.kind == "idiom_explanation"
    assert result.original == "piece of cake"
    assert result.suggestion == "very easy"
    print("PASS test_code_fence_stripped")


def test_kind_none_returns_none():
    provider = FakeCoachingProvider(output=_make_output("none", "", ""))
    coach = _coach(enabled=True, provider=provider)
    result = coach.analyze("break a leg", side="you", uid=1)
    assert result is None
    print("PASS test_kind_none_returns_none")


def test_kind_mismatch_rejected():
    """Ask for idiom but model returns tone → rejected."""
    provider = FakeCoachingProvider(output=_make_output(
        "tone_suggestion", "x", "y"))
    coach = _coach(enabled=True, provider=provider)
    # Force idiom pre-filter by using an idiom hint.
    result = coach.analyze("break a leg", side="you", uid=1)
    assert result is None
    print("PASS test_kind_mismatch_rejected")


def test_missing_required_fields_rejected():
    for bad in (
        '{"kind": "idiom_explanation"}',  # no original/suggestion
        '{"kind": "idiom_explanation", "original": "x"}',  # no suggestion
        '{"kind": "idiom_explanation", "suggestion": "y"}',  # no original
        '{"kind": "idiom_explanation", "original": "", "suggestion": "y"}',
        '{"kind": "idiom_explanation", "original": "x", "suggestion": ""}',
    ):
        provider = FakeCoachingProvider(output=bad)
        coach = _coach(enabled=True, provider=provider)
        result = coach.analyze("break a leg", side="you", uid=1)
        assert result is None, bad
    print("PASS test_missing_required_fields_rejected")


def test_deduplication_same_phrase_once():
    provider = FakeCoachingProvider(output=_make_output(
        "idiom_explanation", "break a leg", "good luck"))
    coach = _coach(enabled=True, provider=provider)

    r1 = coach.analyze("break a leg", side="you", uid=1)
    assert r1 is not None
    r2 = coach.analyze("break a leg", side="you", uid=2)
    assert r2 is None  # deduplicated
    assert provider.call_count == 1  # model consulted once
    print("PASS test_deduplication_same_phrase_once")


def test_deduplication_different_phrases_both_coached():
    provider = FakeCoachingProvider(output=_make_output(
        "idiom_explanation", "break a leg", "good luck"))
    coach = _coach(enabled=True, provider=provider, dedupe_window=10)

    r1 = coach.analyze("break a leg", side="you", uid=1)
    r2 = coach.analyze("piece of cake", side="you", uid=2)
    assert r1 is not None
    assert r2 is not None
    assert provider.call_count == 2
    print("PASS test_deduplication_different_phrases_both_coached")


def test_text_truncation_at_max_chars():
    provider = FakeCoachingProvider(output=_make_output(
        "idiom_explanation", "break a leg", "good luck"))
    coach = _coach(enabled=True, provider=provider, max_text_chars=20)
    long_text = "break a leg and then some extra words"
    result = coach.analyze(long_text, side="you", uid=1)
    # Truncated text "break a leg and th" still contains the idiom hint.
    assert result is not None
    assert result.original == "break a leg"
    print("PASS test_text_truncation_at_max_chars")


def test_idiom_and_tone_disabled_separately():
    provider = FakeCoachingProvider(output=_make_output(
        "idiom_explanation", "break a leg", "good luck"))

    # Idioms off → no idiom suggestion even with idiom hint.
    coach = _coach(enabled=True, provider=provider, detect_idioms=False,
                   detect_tone=True)
    assert coach.analyze("break a leg", side="you", uid=1) is None

    # Tone off → no tone suggestion even with tone hint.
    provider2 = FakeCoachingProvider(output=_make_output(
        "tone_suggestion", "as i said", "as mentioned"))
    coach2 = _coach(enabled=True, provider=provider2,
                    detect_idioms=True, detect_tone=False)
    assert coach2.analyze("as i said before", side="you", uid=1) is None
    print("PASS test_idiom_and_tone_disabled_separately")


def test_thread_safety_concurrent_calls():
    provider = FakeCoachingProvider(output=_make_output(
        "idiom_explanation", "break a leg", "good luck"))
    coach = _coach(enabled=True, provider=provider)
    results = []
    errors = []

    def worker(uid: int):
        try:
            r = coach.analyze("break a leg", side="you", uid=uid)
            results.append(r)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, errors
    # Exactly one suggestion should survive deduplication.
    suggestions = [r for r in results if r is not None]
    assert len(suggestions) == 1, suggestions
    assert suggestions[0].kind == "idiom_explanation"
    print("PASS test_thread_safety_concurrent_calls")


def test_toggle_enabled_at_runtime():
    provider = FakeCoachingProvider(output=_make_output(
        "idiom_explanation", "break a leg", "good luck"))
    coach = _coach(enabled=False, provider=provider)
    assert coach.is_enabled() is False
    assert coach.analyze("break a leg", side="you", uid=1) is None

    coach.set_enabled(True)
    assert coach.is_enabled() is True
    result = coach.analyze("break a leg", side="you", uid=1)
    assert result is not None
    print("PASS test_toggle_enabled_at_runtime")


def test_set_provider_at_runtime():
    coach = _coach(enabled=True, provider=None)
    assert coach.analyze("break a leg", side="you", uid=1) is None

    provider = FakeCoachingProvider(output=_make_output(
        "idiom_explanation", "break a leg", "good luck"))
    coach.set_provider(provider)
    result = coach.analyze("break a leg", side="you", uid=1)
    assert result is not None
    print("PASS test_set_provider_at_runtime")


if __name__ == "__main__":
    test_disabled_returns_none()
    test_no_provider_returns_none()
    test_empty_text_returns_none()
    test_idiom_detection()
    test_tone_detection()
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
    test_idiom_and_tone_disabled_separately()
    test_thread_safety_concurrent_calls()
    test_toggle_enabled_at_runtime()
    test_set_provider_at_runtime()
    print("\nAll social_coaching tests passed.")
