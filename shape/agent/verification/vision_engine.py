"""Shared builder for the multimodal engine the vision-LLM verifiers use.

Building the vision engine is a VLRR concern, so it lives here once. Every consumer
-- the live ``Executor`` and the benchmark runner -- calls this instead of
re-deriving the ``create_engine(is_multimodal=True)`` construction. Returns None (with
a clear reason) when no usable engine is available; every reasoner question and vision
call then raises (``shape.engine.errors.NoEngineError``: there is no deterministic
fallback). ``structured_smoke`` checks that an engine answers a structured question.
"""

from __future__ import annotations

import logging
import os
from typing import Literal, Optional

from pydantic import BaseModel

logger = logging.getLogger("shape.agent.verification")


def build_vision_engine(model_string: str, api_key: Optional[str] = None):
    """Build the multimodal engine for the vision verifiers, or return None.

    None is returned (and logged) when there is no API key, the model is not
    multimodal, or the engine import/construction fails -- the caller has no engine
    (its reasoner questions and vision calls raise).
    """
    try:
        from shape.engine.factory import create_engine, provider_of
        engine_kwargs = {}
        provider = provider_of(model_string)
        if provider == "openai":
            # gpt-* (and claude-* when LLM_BASE_URL is set: factory.provider_of) go through the
            # OpenAI-compatible gateway (LLM_BASE_URL + LLM_API_KEY).
            base_url = os.environ.get("LLM_BASE_URL")
            if not base_url:
                logger.warning(
                    "vision engine: LLM_BASE_URL not set; refusing the public OpenAI default"
                )
                return None
            engine_kwargs["base_url"] = base_url
            api_key = api_key or os.environ.get("LLM_API_KEY")
        elif provider == "anthropic":
            # claude-* without a gateway (no LLM_BASE_URL) authenticate DIRECTLY against
            # Anthropic with ANTHROPIC_API_KEY (the native engine). Prefer the Anthropic key
            # over any OpenAI-shaped key handed in by a provider-neutral caller.
            api_key = os.environ.get("ANTHROPIC_API_KEY") or api_key or os.environ.get("LLM_API_KEY")
        else:
            api_key = api_key or os.environ.get("LLM_API_KEY")
        if not api_key:
            logger.warning("vision engine: no API key for %s; no engine is built (its reasoner questions "
                           "and vision calls raise)", model_string)
            return None
        return create_engine(
            model_string=model_string,
            is_multimodal=True,
            api_key=api_key,
            **engine_kwargs,
        )
    except Exception as e:
        logger.warning("vision engine init failed (%s); no engine is built (its reasoner questions and vision "
                       "calls raise)", e)
        return None


class _SmokeAnswer(BaseModel):
    answer: Literal["OK"]
    number: int


SMOKE_QUESTION = 'Return the answer "OK" and the number 7 in the requested JSON format.'


def structured_smoke(engine) -> None:
    """One structured question with a small schema (`_SmokeAnswer`) through `engine.generate(response_format=...)`,
    the call every reasoner question makes; raises RuntimeError naming the answer when the engine does not return
    the schema's instance with answer "OK" and number 7 (e.g. the gateway does not do structured output for the
    model), so a run fails fast."""
    response = engine.generate([SMOKE_QUESTION], response_format=_SmokeAnswer)
    record = response.get("content") if isinstance(response, dict) else response
    if isinstance(response, dict) and record is None and response.get("error"):
        raise RuntimeError(f"structured smoke: {response.get('error')}: {str(response.get('message'))[:300]}")
    if isinstance(record, str):
        from shape.engine.structured import parse_json_answer
        record = parse_json_answer(_SmokeAnswer, record)
    if not isinstance(record, _SmokeAnswer) or record.number != 7:
        raise RuntimeError(f"structured smoke: the answer is not a _SmokeAnswer of OK / 7: {str(record)[:200]!r}")
