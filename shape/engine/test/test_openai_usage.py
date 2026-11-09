"""ChatOpenAI's usage: cached and reasoning tokens only when the API reports them (2026-10-06)."""
from types import SimpleNamespace

from pydantic import BaseModel

from shape.engine.openai import ChatOpenAI


def _response(**details):
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120, **details)
    return SimpleNamespace(usage=usage, choices=[SimpleNamespace(message=SimpleNamespace(parsed=Reply(text="ok"), content=None))])


class Reply(BaseModel):
    text: str


def test_the_usage_dict_carries_cached_and_reasoning_tokens_only_when_reported():
    base = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
    reported = _response(prompt_tokens_details=SimpleNamespace(cached_tokens=64),
                         completion_tokens_details=SimpleNamespace(reasoning_tokens=12))
    assert ChatOpenAI._usage_dict(reported) == {**base, "cached_tokens": 64, "reasoning_tokens": 12}
    zero = _response(prompt_tokens_details=SimpleNamespace(cached_tokens=0),
                     completion_tokens_details=SimpleNamespace(reasoning_tokens=None))
    assert ChatOpenAI._usage_dict(zero) == {**base, "cached_tokens": 0}           # a reported 0 is a value
    assert ChatOpenAI._usage_dict(_response()) == base                            # not reported: no field
    assert ChatOpenAI._usage_dict(_response(prompt_tokens_details=None, completion_tokens_details=None)) == base
    assert ChatOpenAI._usage_dict(SimpleNamespace()) == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def test_the_structured_paths_report_the_usage_dict():
    engine = ChatOpenAI(model_string="gpt-4o", api_key="test", is_multimodal=True)
    response = _response(prompt_tokens_details=SimpleNamespace(cached_tokens=64))
    engine.client = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        with_raw_response=SimpleNamespace(parse=lambda **kwargs: SimpleNamespace(parse=lambda: response))))))
    for out in (engine._generate_multimodal(["prompt"], response_format=Reply),
                engine._generate_text("prompt", response_format=Reply)):
        assert out["content"] == Reply(text="ok")
        assert out["usage"] == {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120, "cached_tokens": 64}
