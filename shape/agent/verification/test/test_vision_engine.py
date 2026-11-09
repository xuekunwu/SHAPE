from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from shape.agent.verification.vision_engine import build_vision_engine


def test_vision_engine_uses_llm_api_key(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "llm-test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://gateway.example/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    factory = types.ModuleType("shape.engine.factory")

    def create_engine(**kwargs):
        return kwargs

    factory.create_engine = create_engine
    factory.provider_of = lambda model_string: "openai"
    monkeypatch.setitem(sys.modules, "shape.engine.factory", factory)

    result = build_vision_engine("test-vlm")

    assert result["model_string"] == "test-vlm"
    assert result["is_multimodal"] is True
    assert result["api_key"] == "llm-test-key"
    assert result["base_url"] == "https://gateway.example/v1"


def test_vision_engine_requires_llm_base_url(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "llm-test-key")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    assert build_vision_engine("gpt-4o") is None