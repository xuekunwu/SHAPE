"""claude-* through the OpenAI-compatible gateway, the structured-answer reader and the fatal LLM error class.

No network: the engine's OpenAI client is replaced by a fake whose structured call behaves as the SDK's
``beta.chat.completions.parse`` (the answer validated by the schema; an answer that is not clean JSON of it raises
pydantic's ValidationError)."""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import httpx
import openai
import pytest
import tenacity
from pydantic import ValidationError

from shape.engine.errors import NoEngineError, is_fatal_llm_error
from shape.engine.openai import ChatOpenAI
from shape.engine.structured import first_json_value, parse_json_answer
from shape.utils.schemas import AdmissionAnswer, ExecutionError, FinalChoice

GATEWAY = "https://gateway.example/v1"
REQUEST = httpx.Request("POST", GATEWAY + "/chat/completions")
LUNA = '{"call": "c02", "reasoning": "fewest problems"}\n\nRests on the calls table: c02 is the delivered call.'


@pytest.fixture()
def gateway(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", GATEWAY)
    monkeypatch.setenv("LLM_API_KEY", "gateway-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-be-used")
    monkeypatch.setattr(ChatOpenAI.generate.retry, "wait", tenacity.wait_none())


class _Raw:
    """The SDK's raw response of a structured call: `.parse()` validates the answer as the SDK does."""

    def __init__(self, schema, content, refusal=None):
        self.schema, self.content, self.refusal = schema, content, refusal
        self.text = json.dumps({"choices": [{"message": {"content": content, "refusal": refusal}}],
                                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}})

    def parse(self):
        parsed = None if self.content is None or self.refusal else self.schema.model_validate_json(self.content)
        message = SimpleNamespace(parsed=parsed, content=self.content, refusal=self.refusal)
        usage = SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class _Client:
    def __init__(self, content=None, error=None, refusal=None):
        self.requests = []

        def parse(**kwargs):
            self.requests.append(kwargs)
            if error is not None:
                raise error
            return _Raw(kwargs["response_format"], content, refusal)

        def create(**kwargs):                # the plain completion the engine tries after a failed structured one
            self.requests.append(kwargs)
            if error is not None:
                raise error
            message = SimpleNamespace(content=content)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                                   usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2))

        self.beta = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            with_raw_response=SimpleNamespace(parse=parse))))
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def _engine(client, model="claude-sonnet-5"):
    from shape.engine.factory import create_engine

    engine = create_engine(model, is_multimodal=True)
    engine.client = client
    return engine


# ---- the routing ---------------------------------------------------------------------------------------------

def test_claude_with_the_gateway_is_built_as_gpt_is(gateway, monkeypatch):
    from shape.agent.verification.vision_engine import build_vision_engine
    from shape.engine.factory import create_engine, provider_of

    assert provider_of("claude-sonnet-5") == provider_of("gpt-5.6-luna") == "openai"
    for engine in (create_engine("claude-sonnet-5", is_multimodal=True), build_vision_engine("claude-sonnet-5")):
        assert isinstance(engine, ChatOpenAI) and engine._structured is True       # the structured-output call
        assert str(engine.client.base_url).rstrip("/") == GATEWAY and engine.client.api_key == "gateway-key"
    monkeypatch.delenv("LLM_BASE_URL")
    assert provider_of("claude-sonnet-5") == "anthropic"                          # the native path: no gateway


def test_a_claude_structured_question_is_the_gpt_call_without_sampling_params(gateway):
    client = _Client('{"call": "c01", "reasoning": "r"}')
    out = _engine(client).generate(["choose a call"], response_format=FinalChoice)
    assert out["content"] == FinalChoice(call="c01", reasoning="r") and out["usage"]["total_tokens"] == 8
    request = client.requests[0]
    assert request["response_format"] is FinalChoice and request["model"] == "claude-sonnet-5"
    assert "temperature" not in request and "top_p" not in request and request["max_tokens"] == 4000


# ---- the structured-answer reader ----------------------------------------------------------------------------

@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize("fallback", [False, True])
def test_run_budget_reaches_text_vision_and_fallback_requests(gateway, monkeypatch, multimodal, fallback):
    monkeypatch.setenv("SHAPE_MAX_COMPLETION_TOKENS", "16000")
    monkeypatch.setenv("SHAPE_REASONING_EFFORT", "high")
    client = _Client("not JSON" if fallback else '{"call": "c01", "reasoning": "r"}')
    content = ["choose a call"] if multimodal else "choose a call"
    out = _engine(client, "gpt-5.6-luna").generate(content, max_tokens=4000, response_format=FinalChoice)
    assert out["usage"]["total_tokens"] > 0
    assert len(client.requests) == (2 if fallback else 1)
    assert all(r["max_completion_tokens"] == 16000 and r["reasoning_effort"] == "high"
               for r in client.requests)


def test_json_followed_by_prose_is_read_as_its_first_json_value_and_the_prose_logged(gateway, caplog):
    client = _Client(LUNA)
    with caplog.at_level(logging.WARNING):
        out = _engine(client, "gpt-5.6-luna").generate(["choose a call"], response_format=FinalChoice)
    assert out["content"] == FinalChoice(call="c02", reasoning="fewest problems")
    assert len(client.requests) == 1                                   # one request, no second call
    assert "Rests on the calls table" in caplog.text and "the SDK's parse failed" in caplog.text


def test_the_reader_validates_exactly_the_json_value_and_does_not_relax_validation():
    assert first_json_value('```json\n{"a": [1, {"b": 2}]}\n```') == ({"a": [1, {"b": 2}]}, 8, 28)
    assert parse_json_answer(FinalChoice, "Here it is: " + LUNA).call == "c02"
    with pytest.raises(ValidationError, match="severity"):          # a value outside the Literal stays an error
        parse_json_answer(AdmissionAnswer, '{"reasoning": "r", "problem": "p", "severity": "moderate"} done')
    with pytest.raises(ValidationError):                            # a number for a string field stays an error
        parse_json_answer(FinalChoice, '{"call": 2, "reasoning": "r"}')
    with pytest.raises(ValueError, match="no JSON value"):
        parse_json_answer(FinalChoice, "c02 is the delivered call")


def test_a_refusal_is_an_error_not_an_answer(gateway):
    with pytest.raises(RuntimeError, match="the model refused the FinalChoice question: not allowed"):
        _engine(_Client(None, refusal="not allowed"))._parse_structured(
            {"model": "claude-sonnet-5", "messages": [], "response_format": FinalChoice}, FinalChoice)


def test_the_reasoner_reads_a_text_answer_as_its_first_json_value(monkeypatch):
    from shape.agent.verification import base

    verifier = base.AgenticVerifier()
    verifier.llm_engine = SimpleNamespace(generate=lambda content, response_format=None: {"content": LUNA})
    assert verifier._ask("choose", FinalChoice, "final").call == "c02"


# ---- the fatal class -----------------------------------------------------------------------------------------

def test_the_fatal_class_is_classified_by_exception_type():
    auth = openai.AuthenticationError("bad key", response=httpx.Response(401, request=REQUEST), body=None)
    for error in (auth, NoEngineError("no engine"), openai.APIConnectionError(request=REQUEST),
                  openai.PermissionDeniedError("no", response=httpx.Response(403, request=REQUEST), body=None),
                  openai.NotFoundError("no model", response=httpx.Response(404, request=REQUEST), body=None)):
        assert is_fatal_llm_error(error), error
    for error in (openai.APITimeoutError(request=REQUEST), ValueError("malformed"), TimeoutError("slow"),
                  RuntimeError("the model refused"), ExecutionError("crop-triage failed"),
                  openai.RateLimitError("slow down", response=httpx.Response(429, request=REQUEST), body=None)):
        assert not is_fatal_llm_error(error), error
    try:                                                             # wrapped (a vision call's ExecutionError)
        try:
            raise auth
        except openai.AuthenticationError as inner:
            raise ExecutionError("crop-triage failed on one crop") from inner
    except ExecutionError as wrapped:
        assert is_fatal_llm_error(wrapped)
    assert not is_fatal_llm_error(RuntimeError("authentication failed: invalid key"))   # never by message text


def test_the_engine_raises_a_fatal_error_and_returns_any_other_as_a_failed_call(gateway):
    auth = openai.AuthenticationError("bad key", response=httpx.Response(401, request=REQUEST), body=None)
    with pytest.raises(openai.AuthenticationError):
        _engine(_Client(error=auth)).generate(["q"], response_format=FinalChoice)
    with pytest.raises(openai.APIConnectionError):
        _engine(_Client(error=openai.APIConnectionError(request=REQUEST))).generate(["q"], response_format=FinalChoice)
    out = _engine(_Client(error=openai.APITimeoutError(request=REQUEST))).generate(["q"], response_format=FinalChoice)
    assert out["error"] == "RuntimeError" and "timed out" in out["message"]          # one timeout: not fatal


# ---- the structured smoke ------------------------------------------------------------------------------------

def test_the_structured_smoke_needs_the_schema_answer():
    from shape.agent.verification.vision_engine import _SmokeAnswer, structured_smoke

    def engine(content):
        return SimpleNamespace(generate=lambda prompt, response_format=None: content)

    structured_smoke(engine({"content": _SmokeAnswer(answer="OK", number=7)}))
    structured_smoke(engine({"content": '{"answer": "OK", "number": 7}\n\nDone.'}))
    with pytest.raises(ValueError, match="no JSON value"):
        structured_smoke(engine({"content": "OK"}))
    with pytest.raises(RuntimeError, match="structured smoke: BadRequestError: response_format"):
        structured_smoke(engine({"error": "BadRequestError", "message": "response_format is not supported"}))
    with pytest.raises(RuntimeError, match="not a _SmokeAnswer of OK / 7"):
        structured_smoke(engine({"content": _SmokeAnswer(answer="OK", number=8)}))


@pytest.mark.parametrize("status", [401, 403, 404])
def test_real_sdk_auth_and_model_errors_reach_the_shared_verifier(gateway, status):
    from shape.agent.verification.base import AgenticVerifier

    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(status, json={"error": {"message": "unavailable", "type": "api_error"}})

    engine = _engine(openai.OpenAI(api_key="dummy", base_url=GATEWAY, max_retries=0,
                                  http_client=httpx.Client(transport=httpx.MockTransport(respond))))
    verifier = AgenticVerifier()
    verifier.llm_engine = engine
    with pytest.raises(openai.APIStatusError) as caught:
        verifier._ask_optional(["choose"], FinalChoice, "final")
    assert is_fatal_llm_error(caught.value) and caught.value.shape_question == "final"
    assert len(requests) == 1 and verifier.fallback_rounds == 0


def test_real_sdk_parses_gateway_json_with_trailing_prose_without_a_second_request(gateway):
    sent = []

    def respond(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "completion-test", "object": "chat.completion", "created": 1, "model": "claude-sonnet-5",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": LUNA}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}})

    engine = _engine(openai.OpenAI(api_key="dummy", base_url=GATEWAY, max_retries=0,
                                  http_client=httpx.Client(transport=httpx.MockTransport(respond))))
    answer = engine.generate(["choose"], response_format=FinalChoice)
    assert answer["content"].call == "c02" and answer["usage"]["total_tokens"] == 8
    assert len(sent) == 1 and not {"temperature", "top_p"} & sent[0].keys()
