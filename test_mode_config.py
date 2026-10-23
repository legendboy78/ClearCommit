"""Focused tests for mode_config.py — plain Python, no dependencies.

Run:  python test_mode_config.py
"""

from mode_config import CommunicationMode, CoachingConfig, ModeConfig


def test_defaults_preserve_baseline():
    cfg = ModeConfig()
    assert cfg.mode == CommunicationMode.SOCIAL
    assert cfg.coaching.enabled is False
    assert cfg.is_professional() is False
    assert cfg.coaching_active() is False
    print("PASS test_defaults_preserve_baseline")


def test_valid_modes_accepted():
    for mode in ("social", "professional", "Social", "PROFESSIONAL", "  social  "):
        cfg = ModeConfig.from_dict({"mode": mode})
        assert CommunicationMode.is_valid(cfg.mode), mode
        assert cfg.mode in ("social", "professional")
    print("PASS test_valid_modes_accepted")


def test_invalid_mode_falls_back_to_social():
    for bad in (None, "", "business", "casual", 123, "socialx"):
        cfg = ModeConfig.from_dict({"mode": bad})
        assert cfg.mode == CommunicationMode.SOCIAL, (bad, cfg.mode)
        assert cfg.is_professional() is False
    print("PASS test_invalid_mode_falls_back_to_social")


def test_coaching_toggle():
    cfg = ModeConfig.from_dict({"coaching": {"enabled": True}})
    assert cfg.coaching.enabled is True
    assert cfg.coaching_active() is True

    cfg = ModeConfig.from_dict({"coaching": {"enabled": False}})
    assert cfg.coaching.enabled is False
    assert cfg.coaching_active() is False

    # Missing coaching key -> disabled
    cfg = ModeConfig.from_dict({})
    assert cfg.coaching.enabled is False
    print("PASS test_coaching_toggle")


def test_from_dict_handles_missing_and_invalid_data():
    # None input -> defaults
    cfg = ModeConfig.from_dict(None)
    assert cfg.mode == CommunicationMode.SOCIAL
    assert cfg.coaching.enabled is False

    # Non-dict input -> defaults
    cfg = ModeConfig.from_dict("social")
    assert cfg.mode == CommunicationMode.SOCIAL
    assert cfg.coaching.enabled is False

    # coaching is not a dict -> disabled
    cfg = ModeConfig.from_dict({"coaching": "yes"})
    assert cfg.coaching.enabled is False
    print("PASS test_from_dict_handles_missing_and_invalid_data")


def test_to_dict_round_trip():
    cfg = ModeConfig(mode="professional", coaching=CoachingConfig(enabled=True))
    data = cfg.to_dict()
    assert data == {"mode": "professional", "coaching": {"enabled": True}}
    restored = ModeConfig.from_dict(data)
    assert restored.mode == "professional"
    assert restored.coaching.enabled is True
    print("PASS test_to_dict_round_trip")


def test_is_professional_and_coaching_active():
    cfg = ModeConfig(mode="professional", coaching=CoachingConfig(enabled=True))
    assert cfg.is_professional() is True
    assert cfg.coaching_active() is True

    cfg = ModeConfig(mode="social", coaching=CoachingConfig(enabled=False))
    assert cfg.is_professional() is False
    assert cfg.coaching_active() is False
    print("PASS test_is_professional_and_coaching_active")


if __name__ == "__main__":
    test_defaults_preserve_baseline()
    test_valid_modes_accepted()
    test_invalid_mode_falls_back_to_social()
    test_coaching_toggle()
    test_from_dict_handles_missing_and_invalid_data()
    test_to_dict_round_trip()
    test_is_professional_and_coaching_active()
    print("\nAll mode_config tests passed.")
