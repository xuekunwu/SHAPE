"""Anthropic (Claude) engine -- mirrors ChatOpenAI's generate contract so the
agent is provider-agnostic. Selected by ``create_engine`` for ``claude-*`` models.

Returns the same ``{"content": <str|parsed pydantic>, "usage": {...}}`` shape as
ChatOpenAI:
  - plain text  -> messages.create, join the text blocks.
  - multimodal  -> image bytes become base64 image blocks (mirrors OpenAI's path).
  - structured  -> a ``response_format`` pydantic class is obtained via Claude
    tool-use: one forced tool whose ``input_schema`` is the model's JSON schema;
    the tool input is parsed back into the pydantic instance (so callers get the
    same parsed object OpenAI's ``.parse`` returns).
"""
try:
    import anthropic
except ImportError:
    raise ImportError(
        "If you'd like to use Anthropic models, please install the anthropic "
        "package by running `pip install anthropic`, and add 'ANTHROPIC_API_KEY' "
        "to your environment variables."
    )

import base64
import json
import logging
import re
from typing import List, Union
from pydantic import ValidationError

from tenacity import retry, stop_after_attempt, wait_random_exponential

from .base import EngineLM, CachedEngine
from .errors import is_fatal_llm_error

from dotenv import load_dotenv
load_dotenv()
logging.getLogger("httpx").setLevel(logging.WARNING)

# Models that reject a FORCED tool_choice ("tool"/"any"); their structured calls use "auto" instead.
_NO_FORCED_TOOL_CHOICE = {"claude-sonnet-5-5", "claude-opus-5-5"}


class ChatAnthropic(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string="claude-sonnet-4-5",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool = False,
        enable_cache: bool = False,
        api_key: str = None,
        **kwargs,
    ):
        self.system_prompt = system_prompt
        if api_key is None:
            raise ValueError(
                "Please set the ANTHROPIC_API_KEY environment variable if you'd "
                "like to use Anthropic models."
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_string = model_string
        self.is_multimodal = is_multimodal
        self.enable_cache = enable_cache

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def generate(self, content: Union[str, List[Union[str, bytes]]], system_prompt=None, **kwargs):
        try:
            if isinstance(content, list) and not self.is_multimodal:
                raise NotImplementedError("Multimodal generation requires is_multimodal=True.")
            out = self._generate(content, system_prompt=system_prompt, **kwargs)
        except ValidationError:
            # Callers already implement a text-response fallback for structured
            # schema mismatches.  Let that fallback run instead of converting
            # the validation error into an untyped error dictionary.
            raise
        except Exception as e:
            if is_fatal_llm_error(e):          # no key / permission / model / endpoint: a system error
                raise
            print(f"Error in generate method: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            out = {"error": type(e).__name__, "message": str(e), "details": getattr(e, "args", None)}
        from shape.engine import usage as _token_usage
        _token_usage.note_call(not _token_usage.is_failure(out))       # fail-fast health
        return out

    def _blocks(self, content):
        """Anthropic content blocks from a str or a [str|bytes] multimodal list."""
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        from shape.engine.openai import _maybe_downscale_image
        blocks = []
        for item in content:
            if isinstance(item, bytes):
                # Anthropic bills images by (W*H)/750, so a full-res overlay costs ~6k tokens
                # each; cap the longest side (SHAPE_VISION_MAX_PX, proportional, no crop) -- the
                # SAME downscale the OpenAI path applies. Without this, opus-5 input tokens (and
                # cost) balloon on the vision-judge overlays.
                item = _maybe_downscale_image(item)
                blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(item).decode("utf-8"),
                    },
                })
            else:
                blocks.append({"type": "text", "text": str(item)})
        return blocks

    @staticmethod
    def _usage(response):
        u = getattr(response, "usage", None)
        inp = getattr(u, "input_tokens", 0) if u else 0
        out = getattr(u, "output_tokens", 0) if u else 0
        result = {"prompt_tokens": inp, "completion_tokens": out, "total_tokens": inp + out}
        from shape.engine import usage as _token_usage   # process-level token accounting
        _token_usage.record(result)
        return result

    def _sampling_params(self, *, temperature, top_p=None):
        """Return only sampling parameters accepted by the routed model.

        The Stanford Sonnet-5 route rejects the legacy ``temperature`` and
        ``top_p`` fields.  Omitting them keeps provider transport identical
        across SHAPE-main, v8g and D1 without changing any workflow logic.
        """
        if self.model_string in {"claude-sonnet-5", "claude-sonnet-5-5", "claude-opus-4-8", "claude-opus-5",
                                  "claude-opus-5-5"}:
            return {}
        params = {"temperature": temperature}
        if top_p is not None:
            params["top_p"] = top_p
        return params

    def _thinking_params(self, *, structured: bool) -> dict:
        """Extended-thinking control for thinking models (claude-opus-5). Anthropic uses a
        thinking BUDGET, not low/medium/high. SHAPE_ANTHROPIC_THINKING: 'disabled' (default --
        fastest/cheapest; thinking tokens otherwise bill at the OUTPUT rate) or an integer
        budget_tokens. Forced tool_use (structured) is INCOMPATIBLE with enabled thinking, so
        structured calls always disable it. No-op for non-thinking models (temperature route)."""
        import os
        if self.model_string not in {"claude-opus-5"}:
            return {}
        if structured:
            return {"thinking": {"type": "disabled"}}
        v = os.environ.get("SHAPE_ANTHROPIC_THINKING", "disabled").strip().lower()
        if v in ("disabled", "off", "0", "none", ""):
            return {"thinking": {"type": "disabled"}}
        try:
            return {"thinking": {"type": "enabled", "budget_tokens": max(1024, int(v))}}
        except ValueError:
            return {"thinking": {"type": "disabled"}}

    @staticmethod
    def _parse_structured_tool_input(response_format, tool_input):
        """Validate Anthropic tool input, normalizing its tagged NextStep form.

        Sonnet-5 may wrap the planner-selected tool in ``tool_call`` XML even
        when the forced tool schema asks for the four flat NextStep fields.  We
        preserve the selected tool and justification and only normalize that
        wire representation.
        """
        try:
            return response_format(**tool_input)
        except ValidationError:
            if getattr(response_format, "__name__", "") != "NextStep":
                raise
            if not isinstance(tool_input, dict):
                raise
            tagged = "\n".join(
                value for value in tool_input.values() if isinstance(value, str)
            )
            match = re.search(
                r"<tool_name>\s*([^<]+?)\s*</tool_name>",
                tagged,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if not match:
                raise
            justification = str(tool_input.get("justification") or "").strip()
            return response_format(
                justification=justification or "Normalized Sonnet planner response",
                context=str(tool_input.get("context") or justification).strip(),
                sub_goal=str(tool_input.get("sub_goal") or justification).strip(),
                tool_name=match.group(1).strip(),
            )

    def _generate(self, content, system_prompt=None, temperature=0.5,
                  max_tokens=4000, top_p=0.99, response_format=None):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        messages = [{"role": "user", "content": self._blocks(content)}]

        if response_format is not None:
            # Structured output: one tool whose schema is the pydantic model's JSON schema; parse the tool
            # input back into the model. Most models take a FORCED tool_choice; claude-sonnet-5-5 rejects
            # forced ("tool"/"any") tool_choice, so it gets "auto" (it reliably calls the lone tool) with a
            # text->JSON fallback if it ever answers in prose instead.
            tool_name = getattr(response_format, "__name__", "structured_response")
            forced = self.model_string not in _NO_FORCED_TOOL_CHOICE
            response = self.client.messages.create(
                model=self.model_string,
                system=sys_prompt_arg,
                messages=messages,
                max_tokens=max_tokens,
                **self._sampling_params(temperature=temperature),
                **self._thinking_params(structured=True),
                tools=[{
                    "name": tool_name,
                    "description": f"Return the result strictly as {tool_name}.",
                    "input_schema": response_format.model_json_schema(),
                }],
                tool_choice={"type": "tool", "name": tool_name} if forced else {"type": "auto"},
            )
            parsed = None
            for block in response.content:
                if getattr(block, "type", None) == "tool_use":
                    parsed = self._parse_structured_tool_input(
                        response_format, block.input
                    )
                    break
            if parsed is None:
                # No tool_use block (only possible on the "auto" route): parse the text as JSON.
                text = "".join(getattr(b, "text", "") for b in response.content
                               if getattr(b, "type", None) == "text").strip()
                match = re.search(r"\{.*\}", text, flags=re.DOTALL)
                if match:
                    parsed = self._parse_structured_tool_input(response_format, json.loads(match.group(0)))
            return {"content": parsed, "usage": self._usage(response)}

        response = self.client.messages.create(
            model=self.model_string,
            system=sys_prompt_arg,
            messages=messages,
            max_tokens=max_tokens,
            **self._sampling_params(temperature=temperature, top_p=top_p),
            **self._thinking_params(structured=False),
        )
        text = "".join(
            getattr(b, "text", "") for b in response.content
            if getattr(b, "type", None) == "text"
        )
        return {"content": text, "usage": self._usage(response)}
