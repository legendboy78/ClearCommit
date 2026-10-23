"""Focused tests for ai_config.py + prompt-override flow — plain Python.

Run:  python test_ai_config.py

Covers: AIModelSettings defaults/validation/clamping, provider context,
override sanitization, and the prompt-override flow through the coaches
(persona replaced, JSON contract always preserved, None restores default),
plus the assistant system-prompt override and grounding contract.
"""

import ai_config as aic
from ai_config import (
    AIModelSettings,
    DEFAULT_SYSTEM_PROMPTS,
    FEATURE_ASSISTANT,
    FEATURE_COACH_IDIOM,
    FEATURE_MEETING,
    normalize_override,
    sanitize_overrides,
)
from assistant import ConversationAssistant, FakeAssistantProvider
from meeting_intelligence import EvidenceReference
from social_coaching import SocialCoach, FakeCoachingProvider
from professional_coaching import ProfessionalCoach


def test_defaults():
    cfg = AIModelSettings()
    assert cfg.model == "openai/gpt-oss-120b"
    assert 0.0 <= cfg.temperature <= 1.5
    assert 64 <= cfg.max_tokens <= 4096
    assert cfg.coach_idioms and cfg.coach_tone
    assert cfg.coach_communication and cfg.coach_pitch
    print("PASS test_defaults")


def test_from_dict_clamps_and_falls_back():
    cfg = AIModelSettings.from_dict({
        "model": "  ", "temperature": 9.9, "max_tokens": 1_000_000,
        "coach_idioms": "yes",  # non-bool -> default
    })
    assert cfg.model == "openai/gpt-oss-120b"  # blank -> default
    assert cfg.temperature == 1.5  # clamped
    assert cfg.max_tokens == 4096  # clamped
    assert cfg.coach_idioms is True

    cfg2 = AIModelSettings.from_dict("not a dict")
    assert cfg2.model == "openai/gpt-oss-120b"
    print("PASS test_from_dict_clamps_and_falls_back")


def test_round_trip_and_context():
    cfg = AIModelSettings(model="m1", temperature=0.7, max_tokens=512,
                          coach_idioms=False)
    restored = AIModelSettings.from_dict(cfg.to_dict())
    assert restored == cfg
    ctx = cfg.context()
    assert ctx == {"model": "m1", "temperature": 0.7, "max_tokens": 512}
    print("PASS test_round_trip_and_context")


def test_normalize_and_sanitize_overrides():
    assert normalize_override(None) is None
    assert normalize_override("   ") is None
    assert normalize_override(" custom ") == "custom"
    assert normalize_override(123) is None
    # Unknown keys and non-strings are dropped.
    out = sanitize_overrides({"coach_idiom": " custom ", "bogus": "x",
                              "assistant": 5})
    assert out == {"coach_idiom": "custom"}
    print("PASS test_normalize_and_sanitize_overrides")


def test_social_prompt_override_keeps_json_contract():
    prompts = []
    provider = FakeCoachingProvider(output='{"kind": "none"}')

    class Spy:
        def analyze(self, prompt, context=None):
            prompts.append(prompt)
            return provider.analyze(prompt, context)

    coach = SocialCoach(enabled=True, provider=Spy())
    r1 = coach.analyze("break a leg", side="you", uid=1)
    assert r1 is None  # kind=none
    assert len(prompts) == 1
    assert "Respond ONLY with valid JSON" in prompts[0]  # default includes it

    coach.set_prompts(idiom="You are a Shakespearean idiom scholar.")
    prompts.clear()
    coach.analyze("break a leg", side="you", uid=2)
    assert "Shakespearean" in prompts[0]
    # The JSON contract must ALWAYS survive an override.
    assert "Respond ONLY with valid JSON" in prompts[0]
    assert '"kind": "idiom_explanation" or "none"' in prompts[0]
    # And the utterance must still reach the model.
    assert "break a leg" in prompts[0]

    # None restores the default persona.
    coach.set_prompts(idiom=None)
    prompts.clear()
    coach.analyze("break a leg", side="you", uid=3)
    assert "conversation coach" in prompts[0]
    assert "Shakespearean" not in prompts[0]
    print("PASS test_social_prompt_override_keeps_json_contract")


def test_social_options_toggle():
    provider = FakeCoachingProvider(output='{"kind": "none"}')
    coach = SocialCoach(enabled=True, provider=provider)
    coach.set_options(idioms=False)
    assert coach.analyze("break a leg", side="you", uid=1) is None
    assert provider.call_count == 0  # pre-filter skipped entirely
    coach.set_options(idioms=True)
    coach.analyze("break a leg", side="you", uid=2)
    assert provider.call_count == 1
    print("PASS test_social_options_toggle")


def test_professional_prompt_override_and_options():
    prompts = []
    provider = FakeCoachingProvider(output='{"kind": "none"}')

    class Spy:
        def analyze(self, prompt, context=None):
            prompts.append(prompt)
            return provider.analyze(prompt, context)

    coach = ProfessionalCoach(enabled=True, provider=Spy())
    coach.set_prompts(communication="You are a diplomat.")
    coach.analyze("you always miss deadlines", side="you", uid=1)
    assert "diplomat" in prompts[-1]
    assert "Respond ONLY with valid JSON" in prompts[-1]
    assert '"kind": "communication_coach" or "none"' in prompts[-1]

    coach.set_options(communication=False)
    prompts.clear()
    assert coach.analyze("you always", side="you", uid=2) is None
    assert prompts == []  # detection disabled -> provider untouched
    print("PASS test_professional_prompt_override_and_options")


def test_assistant_system_prompt_override():
    p = FakeAssistantProvider(reply="ok")
    a = ConversationAssistant(provider=p,
                              system_prompt="Be extremely brief.")
    ctx_map = {i: EvidenceReference(uid=i, side="you", text="t",
                                    timestamp=1.0) for i in (1, 2)}
    a.ask("q", evidence_map=ctx_map)
    assert "Be extremely brief." in p.last_context
    # Grounding contract always present, even with an override.
    assert "GROUNDING CONTRACT" in p.last_context
    assert "[E1|you]" in p.last_context

    a.set_system_prompt(None)
    a.ask("q2", evidence_map=ctx_map)
    assert "Be extremely brief." not in p.last_context
    assert "GROUNDING CONTRACT" in p.last_context
    print("PASS test_assistant_system_prompt_override")


def test_default_prompts_cover_all_features():
    for fid in aic.ALL_FEATURES:
        assert fid in DEFAULT_SYSTEM_PROMPTS and DEFAULT_SYSTEM_PROMPTS[fid]
    assert FEATURE_ASSISTANT in DEFAULT_SYSTEM_PROMPTS
    assert FEATURE_MEETING in DEFAULT_SYSTEM_PROMPTS
    print("PASS test_default_prompts_cover_all_features")


if __name__ == "__main__":
    test_defaults()
    test_from_dict_clamps_and_falls_back()
    test_round_trip_and_context()
    test_normalize_and_sanitize_overrides()
    test_social_prompt_override_keeps_json_contract()
    test_social_options_toggle()
    test_professional_prompt_override_and_options()
    test_assistant_system_prompt_override()
    test_default_prompts_cover_all_features()
    print("\nAll ai_config tests passed.")
