"""LiteLLM engine for provider backends and OpenAI-compatible gateways.

The dependency is optional and imported lazily: importing SHAPE, or using a native
OpenAI/Anthropic engine, does not require LiteLLM.  The benchmark's durable
``gemini-3.5-flash`` ID is routed here by :mod:`shape.engine.factory` with a local
gateway URL/token and a separately recorded transport model.
"""

import base64
import logging
from typing import List, Mapping, Optional, Union

from tenacity import retry, stop_after_attempt, wait_random_exponential

from .base import CachedEngine, EngineLM


try:  # Optional dependency; fail only when this engine is constructed.
    import litellm as _litellm
except ImportError:  # pragma: no cover - the explicit failure path is unit-tested
    _litellm = None


logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

_MISSING = object()


def _require_litellm():
    if _litellm is None:
        raise ImportError(
            "LiteLLM routing is optional; install SHAPE with the 'litellm' extra "
            "to use this model route."
        )
    return _litellm


def _as_mapping(value) -> Mapping:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    for method_name in ("model_dump", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                dumped = method(exclude_none=True)
            except TypeError:
                dumped = method()
            if isinstance(dumped, Mapping):
                return dumped
    attrs = getattr(value, "__dict__", None)
    return attrs if isinstance(attrs, Mapping) else {}


def _path(mapping: Mapping, *names):
    current = mapping
    for name in names:
        current = _as_mapping(current)
        if name not in current:
            return _MISSING
        current = current[name]
    return current


def _first_observed(mapping: Mapping, paths):
    for names in paths:
        value = _path(mapping, *names)
        if value is not _MISSING and value is not None:
            try:
                if isinstance(value, bool):
                    continue
                return int(value)
            except (TypeError, ValueError, OverflowError):
                continue
    return _MISSING


def normalize_usage(response) -> dict:
    """Normalize only usage fields actually reported by the gateway.

    No zero or derived total is invented when a field is absent.  Legacy
    ``prompt_tokens``/``completion_tokens`` aliases are emitted from observed
    input/output counts so existing SHAPE benchmark accounting keeps working.
    """
    usage = _as_mapping(getattr(response, "usage", None))
    if not usage and isinstance(response, Mapping):
        usage = _as_mapping(response.get("usage"))

    sources = {
        "input_tokens": (
            ("input_tokens",),
            ("prompt_tokens",),
            ("prompt_token_count",),
        ),
        "output_tokens": (
            ("output_tokens",),
            ("completion_tokens",),
            ("candidates_token_count",),
        ),
        "total_tokens": (
            ("total_tokens",),
            ("total_token_count",),
        ),
        "cached_tokens": (
            ("cached_tokens",),
            ("cache_read_input_tokens",),
            ("prompt_tokens_details", "cached_tokens"),
            ("input_tokens_details", "cached_tokens"),
        ),
        "reasoning_tokens": (
            ("reasoning_tokens",),
            ("completion_tokens_details", "reasoning_tokens"),
            ("output_tokens_details", "reasoning_tokens"),
        ),
        "thinking_tokens": (
            ("thinking_tokens",),
            ("thoughts_token_count",),
            ("completion_tokens_details", "thinking_tokens"),
            ("output_tokens_details", "thinking_tokens"),
        ),
    }

    result = {}
    for canonical_name, paths in sources.items():
        value = _first_observed(usage, paths)
        if value is not _MISSING:
            result[canonical_name] = value

    # Backward-compatible names are aliases of observed fields, never estimates.
    if "input_tokens" in result:
        result["prompt_tokens"] = result["input_tokens"]
    if "output_tokens" in result:
        result["completion_tokens"] = result["output_tokens"]
    return result


class ChatLiteLLM(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string="gemini/gemini-1.5-pro",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool = False,
        enable_cache: bool = False,
        api_key: str = None,
        api_base: Optional[str] = None,
        transport_model: Optional[str] = None,
        omit_sampling_params: bool = False,
        gateway_provenance: Optional[dict] = None,
        **kwargs,
    ):
        self._client = _require_litellm()
        self.system_prompt = system_prompt
        # This is the durable/requested model identity used by benchmark ledgers.
        self.model_string = model_string
        self.transport_model = transport_model or model_string
        self.api_base = api_base
        # LiteLLM's openai/ prefix selects the OpenAI wire adapter; it is not the
        # benchmark model identity and is therefore kept separate from provenance.
        self._completion_model = self.transport_model
        if api_base and not self._completion_model.startswith("openai/"):
            self._completion_model = "openai/" + self._completion_model
        self.is_multimodal = is_multimodal
        self.enable_cache = enable_cache
        self.api_key = api_key
        self.omit_sampling_params = bool(omit_sampling_params)
        self._gateway_provenance = dict(gateway_provenance or {})

    def provenance(self) -> dict:
        """Credential-free route provenance suitable for durable benchmark output."""
        return dict(self._gateway_provenance)

    def _response_provenance(self, response) -> dict:
        provenance = self.provenance()
        # A gateway alias may resolve to a more specific upstream model.  Preserve
        # that observed identity separately; never overwrite the durable request or
        # configured transport identity.
        response_model = getattr(response, "model", None)
        if response_model is None and isinstance(response, Mapping):
            response_model = response.get("model")
        if isinstance(response_model, str) and response_model:
            provenance["response_model"] = response_model
        return provenance

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def generate(self, content: Union[str, List[Union[str, bytes]]], system_prompt=None, **kwargs):
        try:
            if isinstance(content, list) and not self.is_multimodal:
                raise NotImplementedError("Multimodal generation requires is_multimodal=True.")
            out = self._generate(content, system_prompt=system_prompt, **kwargs)
        except Exception as exc:
            # Provider exceptions often contain request headers/URLs.  Do not stringify
            # them, preserve args, or emit a traceback from a benchmark child process.
            error_type = type(exc).__name__
            logger.warning("LLM gateway request failed (%s)", error_type)
            out = {"error": error_type, "message": "LLM gateway request failed."}
            if self._gateway_provenance:
                out["provenance"] = self.provenance()
        from shape.engine import usage as _token_usage
        _token_usage.note_call(not _token_usage.is_failure(out))
        return out

    def _user_content(self, content):
        """OpenAI-style user content: a plain string, or text + image_url blocks."""
        if isinstance(content, str):
            return content
        blocks = []
        for item in content:
            if isinstance(item, bytes):
                b64 = base64.b64encode(item).decode("utf-8")
                blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
            else:
                blocks.append({"type": "text", "text": str(item)})
        return blocks

    @staticmethod
    def _usage(response):
        result = normalize_usage(response)
        from shape.engine import usage as _token_usage
        _token_usage.record(result)
        return result

    def _generate(
        self,
        content,
        system_prompt=None,
        temperature=0.5,
        max_tokens=4000,
        top_p=0.99,
        response_format=None,
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        messages = [
            {"role": "system", "content": sys_prompt_arg},
            {"role": "user", "content": self._user_content(content)},
        ]
        params = {
            "model": self._completion_model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        # Gemini 3.x benchmark policy is gateway-locked (medium thinking); client
        # sampling overrides would make arms incomparable and are omitted entirely.
        if not self.omit_sampling_params:
            params["temperature"] = temperature
            params["top_p"] = top_p
        if self.api_key:
            params["api_key"] = self.api_key
        if self.api_base:
            params["api_base"] = self.api_base
        if response_format is not None:
            params["response_format"] = response_format

        response = self._client.completion(**params)
        text = response.choices[0].message.content
        content_result = text
        if response_format is not None and isinstance(text, str):
            content_result = response_format.model_validate_json(text)
        result = {"content": content_result, "usage": self._usage(response)}
        if self._gateway_provenance:
            result["provenance"] = self._response_provenance(response)
        return result
