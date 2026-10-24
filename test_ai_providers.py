"""Tests for ProviderConfig API-key handling (Models-tab keys vs env vars)."""

import os

from ai_providers import ProviderConfig, PROVIDER_PRESETS


def _set_env(name, value):
    """Set or remove an env var; returns a restore callable."""
    old = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value

    def restore():
        _set_env(name, old)
    return restore


def _clear_all_provider_env():
    """Remove every provider env var; returns a restore callable."""
    names = [p.key_env_var for p in PROVIDER_PRESETS.values()]
    olds = {n: os.environ.get(n) for n in names}
    for n in names:
        os.environ.pop(n, None)

    def restore():
        for n, v in olds.items():
            if v is None:
                os.environ.pop(n, None)
            else:
                os.environ[n] = v
    return restore


def test_defaults_have_no_ui_keys():
    r = _clear_all_provider_env()
    try:
        cfg = ProviderConfig()
        assert cfg.api_keys == {}
        assert cfg.api_key_source() == ""
        assert not cfg.is_configured()
    finally:
        r()


def test_from_dict_parses_and_strips_ui_keys():
    cfg = ProviderConfig.from_dict({
        "preset": "openrouter",
        "api_keys": {"openrouter": "  sk-or-v1-abc  ", "groq": "gsk_ui"},
    })
    assert cfg.api_keys["openrouter"] == "sk-or-v1-abc"
    assert cfg.api_keys["groq"] == "gsk_ui"


def test_from_dict_ignores_junk_keys():
    cfg = ProviderConfig.from_dict({
        "api_keys": "not-a-dict",
    })
    assert cfg.api_keys == {}
    cfg = ProviderConfig.from_dict({
        "api_keys": {"unknown_preset": "x", "groq": 123, "openrouter": "   "},
    })
    assert cfg.api_keys == {}


def test_round_trip_preserves_ui_keys():
    cfg = ProviderConfig.from_dict({
        "preset": "openrouter",
        "api_keys": {"openrouter": "sk-or-v1-abc"},
    })
    again = ProviderConfig.from_dict(cfg.to_dict())
    assert again.api_keys == {"openrouter": "sk-or-v1-abc"}


def test_ui_key_overrides_env():
    r = _set_env("GROQ_API_KEY", "gsk_from_env")
    try:
        cfg = ProviderConfig(api_keys={"groq": "gsk_from_ui"})
        assert cfg.resolve_api_key() == "gsk_from_ui"
        assert cfg.api_key_source() == "custom"
    finally:
        r()


def test_env_fallback_when_no_ui_key():
    r = _set_env("GROQ_API_KEY", "gsk_from_env")
    try:
        cfg = ProviderConfig()
        assert cfg.resolve_api_key() == "gsk_from_env"
        assert cfg.api_key_source() == "env"
    finally:
        r()


def test_blank_ui_key_falls_back_to_env():
    r = _set_env("GROQ_API_KEY", "gsk_from_env")
    try:
        cfg = ProviderConfig(api_keys={"groq": "   "})
        assert cfg.resolve_api_key() == "gsk_from_env"
        assert cfg.api_key_source() == "env"
    finally:
        r()


def test_keys_are_per_preset():
    r = _set_env("OPENROUTER_API_KEY", "or_from_env")
    try:
        cfg = ProviderConfig(preset="openrouter", api_keys={"groq": "gsk_ui"})
        # No openrouter UI key -> env fallback, groq's key must not leak over.
        assert cfg.resolve_api_key() == "or_from_env"
        assert cfg.api_key_source() == "env"
    finally:
        r()


def test_is_configured_with_ui_key_and_no_env():
    r = _set_env("GROQ_API_KEY", None)
    try:
        cfg = ProviderConfig(preset="groq", api_keys={"groq": "gsk_ui"})
        assert cfg.is_configured()  # default base URL + default model + UI key
    finally:
        r()


def test_missing_key_everywhere_is_unconfigured():
    r = _clear_all_provider_env()
    try:
        cfg = ProviderConfig()
        assert cfg.api_key_source() == ""
        assert not cfg.is_configured()
    finally:
        r()


def test_integration_connects_provider_from_ui_key():
    from ai_config import AIModelSettings
    from clearcommit_integration import ClearCommitIntegration
    from mode_config import ModeConfig
    r = _set_env("GROQ_API_KEY", None)
    try:
        integration = ClearCommitIntegration(mode_config=ModeConfig.from_dict(
            {"mode": "social", "coaching": {"enabled": True}}))
        try:
            integration.update_ai_config(
                AIModelSettings(),
                {},
                provider_config=ProviderConfig(api_keys={"groq": "gsk_ui"}),
            )
            assert integration._assistant._provider is not None
            assert integration.social_coach._provider is not None
        finally:
            integration.close()
    finally:
        r()


def _fake_session(responses):
    """Session stand-in whose post() pops canned (status, json) responses."""
    import ai_providers

    class FakeResponse:
        def __init__(self, status, body):
            self.status_code = status
            self._body = body

        def json(self):
            return self._body

    class FakeSession:
        def __init__(self):
            self.calls = 0

        def post(self, url, json=None, headers=None, timeout=None):
            self.calls += 1
            status, body = responses[min(self.calls - 1, len(responses) - 1)]
            return FakeResponse(status, body)

    return FakeSession()


def _no_sleep(monkeypatch):
    monkeypatch.setattr("ai_providers.time.sleep", lambda _s: None)


def test_retry_on_429_then_success(monkeypatch):
    from ai_providers import OpenAICompatibleProvider, ProviderConfig
    _no_sleep(monkeypatch)
    r = _set_env("OPENROUTER_API_KEY", None)
    try:
        cfg = ProviderConfig(preset="openrouter",
                             api_keys={"openrouter": "or_test"})
        prov = OpenAICompatibleProvider(cfg)
        session = _fake_session([
            (429, {}), (200, {"choices": [{"message": {"content": " hi "}}]}),
        ])
        prov._session = session
        out = prov._chat("sys", "hello")
        assert out == "hi"
        assert session.calls == 2  # one retry after the 429
    finally:
        r()


def test_rate_limited_after_two_429s(monkeypatch):
    from ai_providers import OpenAICompatibleProvider, ProviderConfig
    _no_sleep(monkeypatch)
    r = _set_env("OPENROUTER_API_KEY", None)
    try:
        cfg = ProviderConfig(preset="openrouter",
                             api_keys={"openrouter": "or_test"})
        prov = OpenAICompatibleProvider(cfg)
        session = _fake_session([(429, {})])
        prov._session = session
        try:
            prov._chat("sys", "hello")
            assert False, "expected rate_limited"
        except RuntimeError as exc:
            assert str(exc) == "rate_limited"
        assert session.calls == 2  # exactly one retry, then raise
    finally:
        r()


def test_401_never_retried(monkeypatch):
    from ai_providers import OpenAICompatibleProvider, ProviderConfig
    _no_sleep(monkeypatch)
    r = _set_env("OPENROUTER_API_KEY", None)
    try:
        cfg = ProviderConfig(preset="openrouter",
                             api_keys={"openrouter": "or_test"})
        prov = OpenAICompatibleProvider(cfg)
        session = _fake_session([(401, {})])
        prov._session = session
        try:
            prov._chat("sys", "hello")
            assert False, "expected invalid_api_key"
        except RuntimeError as exc:
            assert str(exc) == "invalid_api_key"
        assert session.calls == 1  # auth failures are not transient
    finally:
        r()


if __name__ == "__main__":
    test_defaults_have_no_ui_keys()
    test_from_dict_parses_and_strips_ui_keys()
    test_from_dict_ignores_junk_keys()
    test_round_trip_preserves_ui_keys()
    test_ui_key_overrides_env()
    test_env_fallback_when_no_ui_key()
    test_blank_ui_key_falls_back_to_env()
    test_keys_are_per_preset()
    test_is_configured_with_ui_key_and_no_env()
    test_missing_key_everywhere_is_unconfigured()
    test_integration_connects_provider_from_ui_key()
    test_retry_on_429_then_success()
    test_rate_limited_after_two_429s()
    test_401_never_retried()
    print("\nAll ai_providers tests passed.")
