"""LLM engine factory -- pick the provider implementation from the model name.

One ``create_engine(model_string, ...)`` entry point
dispatches by model-name prefix so the agent is provider-agnostic
(``solve(llm_engine_name="claude-...")`` just works). When no explicit model is passed, LLM_MODEL is required;
each provider SDK is imported lazily, so using GPT never requires the Anthropic
package (and vice-versa). Adding a provider = one ``shape/engine/<name>.py`` plus
a prefix entry here.
"""
import os
from typing import Optional
from urllib.parse import urlsplit, urlunsplit


GEMINI_35_FLASH_MODEL_ID = "gemini-3.5-flash"
GEMINI_31_PRO_MODEL_ID = "gemini-3.1-pro-preview"
GEMINI_GATEWAY_MODEL_IDS = frozenset(
    {GEMINI_35_FLASH_MODEL_ID, GEMINI_31_PRO_MODEL_ID}
)
QWEN_GATEWAY_MODEL_ID = "qwen3-vl-8b-instruct-local"

# The benchmark process must only know the short-lived/local proxy credential.  The
# upstream Google credential belongs to the gateway process and is deliberately not
# accepted by SHAPE.  Keep this list public so launchers can construct a clean child
# environment with ``gemini_gateway_subprocess_env`` below.
GEMINI_UPSTREAM_CREDENTIAL_ENV_VARS = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
)

_GEMINI_GATEWAY_TOKEN_ENV = "SHAPE_GEMINI_GATEWAY_TOKEN"
_GEMINI_GATEWAY_BASE_URL_ENV = "SHAPE_GEMINI_GATEWAY_BASE_URL"
_GEMINI_GATEWAY_MODEL_ENV = "SHAPE_GEMINI_GATEWAY_TRANSPORT_MODEL"
_DEFAULT_GEMINI_GATEWAY_BASE_URL = "http://127.0.0.1:4000/v1"
_QWEN_GATEWAY_TOKEN_ENV = "SHAPE_QWEN_GATEWAY_TOKEN"
_QWEN_GATEWAY_BASE_URL_ENV = "SHAPE_QWEN_GATEWAY_BASE_URL"
_QWEN_GATEWAY_MODEL_ENV = "SHAPE_QWEN_GATEWAY_TRANSPORT_MODEL"
_DEFAULT_QWEN_GATEWAY_BASE_URL = "http://127.0.0.1:4500/v1"

# model-name prefix -> provider key. First match wins; unknown -> litellm.
_PREFIXES = (
    ("gpt-", "openai"),
    ("chatgpt", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("claude", "anthropic"),
    ("glm", "glm"),
)

_ENV_VAR = {
    # SHAPE's OpenAI-compatible route is provided by the configured LLM gateway.
    # Keep the credential name and endpoint explicit; never silently fall back to
    # the public OpenAI service or a legacy Stanford credential.
    "openai": "LLM_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "glm": "ZHIPUAI_API_KEY",
}
_GENERIC_API_KEY_ENV = "LLM_API_KEY"

_LLM_BASE_URL_ENV = "LLM_BASE_URL"
_LLM_MODEL_ENV = "LLM_MODEL"

# 智谱 GLM's OpenAI-compatible endpoint (override with GLM_BASE_URL if you're on a proxy/region).
_GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"


def _is_gemini_gateway_model(model_string: str) -> bool:
    return (model_string or "").strip().lower() in GEMINI_GATEWAY_MODEL_IDS


def _is_qwen_gateway_model(model_string: str) -> bool:
    return (model_string or "").strip().lower() == QWEN_GATEWAY_MODEL_ID


def _validate_local_gateway_url(base_url: str) -> str:
    """Return a credential-free local OpenAI-compatible gateway URL.

    SHAPE is intentionally not allowed to talk to the upstream Gemini endpoint.  A
    loopback-only URL makes the credential boundary mechanically checkable instead of
    relying on operator convention.  User-info, query strings and fragments are also
    rejected because they are easy ways to accidentally persist a secret in provenance.
    """
    parsed = urlsplit(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("SHAPE_GEMINI_GATEWAY_BASE_URL must be an http(s) URL.")
    if parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("SHAPE Gemini routing requires a loopback/local gateway URL.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "SHAPE_GEMINI_GATEWAY_BASE_URL must not contain credentials, a query, or a fragment."
        )
    # Recompose so provenance never includes user-info even if urlsplit behaviour changes.
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    netloc = host + (f":{parsed.port}" if parsed.port is not None else "")
    return urlunsplit((parsed.scheme, netloc, parsed.path.rstrip("/"), "", ""))


def _assert_no_gemini_upstream_credentials(env=None) -> None:
    env = os.environ if env is None else env
    leaked_names = [name for name in GEMINI_UPSTREAM_CREDENTIAL_ENV_VARS if env.get(name)]
    if leaked_names:
        # Names are safe to report; values must never enter errors/logs.
        raise ValueError(
            "Upstream Gemini credentials are forbidden in the SHAPE process; "
            "remove: " + ", ".join(leaked_names)
        )


def gemini_gateway_subprocess_env(
    source_env=None,
    *,
    proxy_token: Optional[str] = None,
    base_url: Optional[str] = None,
    transport_model: Optional[str] = None,
) -> dict:
    """Build the environment for a SHAPE child using only the local proxy token.

    This helper never promotes an upstream key to a proxy key.  The caller must provide
    ``proxy_token`` or have ``SHAPE_GEMINI_GATEWAY_TOKEN`` in ``source_env``.
    """
    clean = dict(os.environ if source_env is None else source_env)
    for name in GEMINI_UPSTREAM_CREDENTIAL_ENV_VARS:
        clean.pop(name, None)
    if proxy_token is not None:
        clean[_GEMINI_GATEWAY_TOKEN_ENV] = proxy_token
    if not clean.get(_GEMINI_GATEWAY_TOKEN_ENV):
        raise ValueError("A local SHAPE Gemini gateway token is required.")
    clean[_GEMINI_GATEWAY_BASE_URL_ENV] = _validate_local_gateway_url(
        base_url or clean.get(_GEMINI_GATEWAY_BASE_URL_ENV) or _DEFAULT_GEMINI_GATEWAY_BASE_URL
    )
    configured_transport = (
        transport_model
        if transport_model is not None
        else clean.get(_GEMINI_GATEWAY_MODEL_ENV, GEMINI_35_FLASH_MODEL_ID)
    )
    if configured_transport not in GEMINI_GATEWAY_MODEL_IDS:
        raise ValueError(
            "The Gemini benchmark transport model must be one of the pinned exact IDs."
        )
    clean[_GEMINI_GATEWAY_MODEL_ENV] = configured_transport
    return clean


def _gemini_gateway_settings(
    api_key: Optional[str] = None, *, requested_model: str = GEMINI_35_FLASH_MODEL_ID
) -> dict:
    _assert_no_gemini_upstream_credentials()
    proxy_token = api_key or os.getenv(_GEMINI_GATEWAY_TOKEN_ENV)
    if not proxy_token:
        raise ValueError(
            "No local Gemini gateway token; set SHAPE_GEMINI_GATEWAY_TOKEN or pass api_key."
        )
    base_url = _validate_local_gateway_url(
        os.getenv(_GEMINI_GATEWAY_BASE_URL_ENV, _DEFAULT_GEMINI_GATEWAY_BASE_URL)
    )
    transport_model = os.getenv(_GEMINI_GATEWAY_MODEL_ENV, GEMINI_35_FLASH_MODEL_ID)
    if requested_model not in GEMINI_GATEWAY_MODEL_IDS:
        raise ValueError(
            "The requested Gemini model must be one of the pinned exact IDs."
        )
    if transport_model != requested_model:
        # A transport alias or a different pinned route can silently disagree with
        # the ledger. Require exact request/transport identity for every run.
        raise ValueError(
            "The Gemini gateway requested and transport model IDs must match exactly."
        )
    return {
        "api_key": proxy_token,
        "api_base": base_url,
        "transport_model": transport_model,
        # Sampling and thinking policy are fixed by the gateway.  SHAPE sends no
        # temperature/top_p/thinking override for this route.
        "omit_sampling_params": True,
        "gateway_provenance": {
            "requested_model": requested_model,
            "transport_model": transport_model,
            "transport": "litellm-openai-compatible-gateway",
            "gateway_base_url": base_url,
            "thinking_level": "medium",
            "thinking_enforced_by": "gateway",
            "client_sampling_override": False,
        },
    }


def _qwen_gateway_settings(api_key: Optional[str] = None) -> dict:
    """Credential-bound exact route for the audited loopback Qwen gateway."""
    proxy_token = api_key or os.getenv(_QWEN_GATEWAY_TOKEN_ENV)
    if not proxy_token:
        raise ValueError(
            "No local Qwen gateway token; set SHAPE_QWEN_GATEWAY_TOKEN or pass api_key."
        )
    base_url = _validate_local_gateway_url(
        os.getenv(_QWEN_GATEWAY_BASE_URL_ENV, _DEFAULT_QWEN_GATEWAY_BASE_URL)
    )
    transport_model = os.getenv(_QWEN_GATEWAY_MODEL_ENV, QWEN_GATEWAY_MODEL_ID)
    if transport_model != QWEN_GATEWAY_MODEL_ID:
        raise ValueError("The Qwen gateway transport model ID must match exactly.")
    return {
        "api_key": proxy_token,
        "api_base": base_url,
        "transport_model": transport_model,
        "gateway_provenance": {
            "requested_model": QWEN_GATEWAY_MODEL_ID,
            "transport_model": transport_model,
            "transport": "litellm-openai-compatible-gateway",
            "gateway_base_url": base_url,
        },
    }


def provider_of(model_string: str) -> str:
    """Provider key for a model name. Known prefixes -> a native engine; anything
    else -> the 'litellm' catch-all (Gemini / Together / DeepSeek / local / ...).
    claude-* with the OpenAI-compatible gateway configured (LLM_BASE_URL) -> 'openai':
    the gateway serves them, built as gpt-* are (ChatOpenAI, LLM_API_KEY, the same
    structured-output call); the native Anthropic engine only without a gateway."""
    if _is_gemini_gateway_model(model_string):
        return "gemini_gateway"
    if _is_qwen_gateway_model(model_string):
        return "qwen_gateway"
    name = (model_string or "").lower()
    for prefix, provider in _PREFIXES:
        if name.startswith(prefix):
            if provider == "anthropic" and os.getenv(_LLM_BASE_URL_ENV):
                return "openai"
            return provider
    return "litellm"


def resolve_api_key(model_string: str, api_key: Optional[str] = None) -> Optional[str]:
    """Resolve the credential consumed by a model route.

    ``LLM_API_KEY`` is the provider-neutral default for native providers. Their
    conventional environment variables remain supported as fallbacks. The generic
    LiteLLM catch-all lets LiteLLM resolve its backend. The benchmark Gemini route is
    special: it accepts only ``SHAPE_GEMINI_GATEWAY_TOKEN`` (or the explicit local
    proxy token), never an upstream Google credential.
    """
    if _is_gemini_gateway_model(model_string):
        return _gemini_gateway_settings(
            api_key, requested_model=model_string
        )["api_key"]
    if _is_qwen_gateway_model(model_string):
        return _qwen_gateway_settings(api_key)["api_key"]
    if api_key:
        return api_key
    provider = provider_of(model_string)
    if provider in _ENV_VAR:
        return os.getenv(_GENERIC_API_KEY_ENV) or os.getenv(_ENV_VAR[provider])
    return None


def resolve_llm_base_url() -> str:
    """Resolve the required OpenAI-compatible SHAPE gateway endpoint."""
    base_url = os.getenv(_LLM_BASE_URL_ENV)
    if not base_url:
        raise ValueError(
            "LLM_BASE_URL is required for the SHAPE OpenAI-compatible LLM route."
        )
    return base_url


def resolve_llm_model(model_string: Optional[str] = None) -> str:
    """Resolve the required model from an explicit value or LLM_MODEL."""
    model = (model_string or os.getenv(_LLM_MODEL_ENV) or "").strip()
    if not model:
        raise ValueError(
            "LLM_MODEL is required for SHAPE LLM calls; set it to an "
            "authorized gateway model such as gpt-4o."
        )
    return model


def create_engine(model_string: Optional[str] = None, system_prompt: Optional[str] = None,
                  is_multimodal: bool = False, api_key: Optional[str] = None,
                  **kwargs):
    """Construct the LLM engine for ``model_string``.

    Same call shape as ``ChatOpenAI(...)``; ``api_key`` falls back to the
    provider's env var. Provider SDKs are imported lazily inside each branch.
    """
    model_string = resolve_llm_model(model_string)
    provider = provider_of(model_string)
    common = dict(
        model_string=model_string,
        is_multimodal=is_multimodal,
        api_key=resolve_api_key(model_string, api_key),
        **kwargs,
    )
    if system_prompt is not None:
        common["system_prompt"] = system_prompt

    if provider == "anthropic":
        from shape.engine.anthropic import ChatAnthropic
        return ChatAnthropic(**common)
    if provider == "glm":
        # 智谱 GLM speaks the OpenAI wire protocol, so reuse ChatOpenAI pointed at GLM's
        # endpoint -- structured output (glm-5.1 / glm-4 family), multimodal (the vision
        # judge), the per-request timeout, and token accounting all come for free. base_url /
        # structured can still be overridden via kwargs (or GLM_BASE_URL) without touching this
        # branch. The two GLM quirks below are what jusheng-V2's glm_proxy handled externally,
        # ported here so shape talks to GLM directly (no proxy process):
        #   * thinking: GLM defaults thinking ON (~5 min/img); disable it unless GLM_THINK=on.
        #   * sanitize_messages: GLM (error 1213) rejects the null/empty and pure-text list-form
        #     content OpenAI allows (image turns are preserved for the vision judge).
        from shape.engine.openai import ChatOpenAI
        common.setdefault("base_url", os.getenv("GLM_BASE_URL", _GLM_BASE_URL))
        common.setdefault("supports_structured", True)
        if os.getenv("GLM_THINK", "off").lower() != "on":
            common.setdefault("extra_body", {"thinking": {"type": "disabled"}})
        common.setdefault("sanitize_messages", True)
        return ChatOpenAI(**common)
    if provider == "gemini_gateway":
        from shape.engine.litellm import ChatLiteLLM
        # Re-resolve the route here so a direct create_engine() call receives the
        # same credential-boundary and URL validation as construct_solver().
        common.update(
            _gemini_gateway_settings(api_key, requested_model=model_string)
        )
        return ChatLiteLLM(**common)
    if provider == "qwen_gateway":
        from shape.engine.litellm import ChatLiteLLM
        common.update(_qwen_gateway_settings(api_key))
        return ChatLiteLLM(**common)
    if provider == "openai":
        from shape.engine.openai import ChatOpenAI
        common.setdefault("base_url", resolve_llm_base_url())
        if model_string.lower().startswith("claude"):     # claude-* on the gateway: the structured-output call
            common.setdefault("supports_structured", True)
        return ChatOpenAI(**common)
    if provider == "litellm":
        from shape.engine.litellm import ChatLiteLLM
        return ChatLiteLLM(**common)
    from shape.engine.openai import ChatOpenAI
    return ChatOpenAI(**common)
