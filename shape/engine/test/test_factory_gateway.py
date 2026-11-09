"""Portable routing checks for the optional benchmark-only Gemini gateway."""

import sys
import types

import pytest

from shape.engine.factory import (
    create_engine,
    provider_of,
    resolve_api_key,
    resolve_llm_base_url,
    resolve_llm_model,
)


def test_llm_model_comes_from_environment(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-5")
    assert resolve_llm_model() == "claude-sonnet-5"


def test_llm_model_is_required_without_explicit_value(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    with pytest.raises(ValueError, match="LLM_MODEL"):
        resolve_llm_model()


def test_explicit_llm_model_wins_over_environment(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "wrong-model")
    assert resolve_llm_model("gpt-4o") == "gpt-4o"


def test_openai_route_uses_only_llm_api_key(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "llm-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-used")
    assert resolve_api_key("gpt-4o") == "llm-test-key"


def test_openai_route_does_not_fallback_to_openai_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-used")
    assert resolve_api_key("gpt-4o") is None


def test_openai_route_requires_gateway_url(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        resolve_llm_base_url()


def test_openai_route_passes_gateway_url_to_client(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "llm-test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://gateway.example/v1")
    fake_openai = types.ModuleType("shape.engine.openai")
    fake_openai.ChatOpenAI = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "shape.engine.openai", fake_openai)

    result = create_engine("gpt-4o", is_multimodal=True)

    assert result["api_key"] == "llm-test-key"
    assert result["base_url"] == "https://gateway.example/v1"
    assert result["is_multimodal"] is True


def _clear_gateway(monkeypatch):
    monkeypatch.delenv("SHAPE_GEMINI_GATEWAY_TOKEN", raising=False)
    monkeypatch.delenv("SHAPE_GEMINI_GATEWAY_BASE_URL", raising=False)


def test_gemini_uses_gateway_by_default(monkeypatch):
    _clear_gateway(monkeypatch)
    assert provider_of("gemini-3.5-flash") == "gemini_gateway"
    with pytest.raises(ValueError, match="No local Gemini gateway token"):
        resolve_api_key("gemini-3.5-flash")


def test_token_only_gateway_configuration_uses_default_loopback(monkeypatch):
    _clear_gateway(monkeypatch)
    monkeypatch.setenv("SHAPE_GEMINI_GATEWAY_TOKEN", "test-token")
    assert provider_of("gemini-3.5-flash") == "gemini_gateway"
    assert resolve_api_key("gemini-3.5-flash") == "test-token"


def test_complete_gateway_configuration_uses_loopback_url(monkeypatch):
    _clear_gateway(monkeypatch)
    monkeypatch.setenv("SHAPE_GEMINI_GATEWAY_TOKEN", "test-token")
    monkeypatch.setenv(
        "SHAPE_GEMINI_GATEWAY_BASE_URL", "http://127.0.0.1:4000/v1"
    )
    assert provider_of("gemini-3.5-flash") == "gemini_gateway"
    assert resolve_api_key("gemini-3.5-flash") == "test-token"


def test_non_loopback_gateway_url_is_rejected(monkeypatch):
    _clear_gateway(monkeypatch)
    monkeypatch.setenv("SHAPE_GEMINI_GATEWAY_TOKEN", "test-token")
    monkeypatch.setenv(
        "SHAPE_GEMINI_GATEWAY_BASE_URL", "http://gateway.invalid/v1"
    )
    with pytest.raises(ValueError, match="loopback/local gateway"):
        resolve_api_key("gemini-3.5-flash")


def test_explicit_api_key_still_wins_without_gateway(monkeypatch):
    _clear_gateway(monkeypatch)
    assert resolve_api_key("gemini-3.5-flash", "explicit-key") == "explicit-key"
