try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Install the openai package to use the SHAPE LLM gateway.")

import os
import json
import base64
import logging
import platformdirs
from PIL import Image
from io import BytesIO
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)
from typing import List, Union

from .base import EngineLM, CachedEngine
from .errors import is_fatal_llm_error
from .structured import parse_json_answer

import openai

from dotenv import load_dotenv
load_dotenv()
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Define global constant for structured models
# https://platform.openai.com/docs/guides/structured-outputs
# https://cookbook.openai.com/examples/structured_outputs_intro
from pydantic import BaseModel

class DefaultFormat(BaseModel):
    response: str

# Define global constant for structured models
# GPT-5 mini and GPT-5 support structured outputs like GPT-4o
OPENAI_STRUCTURED_MODELS = ['gpt-4o', 'gpt-4o-2024-08-06','gpt-4o-mini',  'gpt-4o-mini-2024-07-18','gpt-5-mini', 'gpt-5','deepseek']

# Models that require max_completion_tokens instead of max_tokens
MODELS_REQUIRING_MAX_COMPLETION_TOKENS = ['o1', 'o1-mini', 'gpt-5-mini', 'gpt-5']

# Models that only support temperature=1 (default) and cannot accept custom temperature/top_p values
# These models should not have temperature or top_p parameters set (similar to o1/o1-mini)
MODELS_REQUIRING_TEMPERATURE_1 = ['gpt-5-mini', 'gpt-5']


def _is_reasoning_or_next_gen_model(model_string: str) -> bool:
    """o-series and gpt-5+/gpt-6+ models: require max_completion_tokens (not max_tokens), reject a
    custom temperature/top_p, and support structured outputs. Prefix-match so new point releases
    (gpt-5.6-luna, gpt-5.6-sol, gpt-6-astra, ...) are handled without editing the exact lists."""
    m = (model_string or "").lower()
    return m.startswith(("o1", "o3", "o4", "gpt-5", "gpt-6"))


def _requires_max_completion_tokens(model_string: str) -> bool:
    return (model_string in MODELS_REQUIRING_MAX_COMPLETION_TOKENS
            or _is_reasoning_or_next_gen_model(model_string))


def _requires_temperature_1(model_string: str) -> bool:
    return (model_string in MODELS_REQUIRING_TEMPERATURE_1
            or _is_reasoning_or_next_gen_model(model_string)
            or (model_string or "").lower().startswith("claude"))


def _supports_structured(model_string: str) -> bool:
    return (model_string in OPENAI_STRUCTURED_MODELS
            or _is_reasoning_or_next_gen_model(model_string))


def _maybe_downscale_image(item: bytes, cap: int = None) -> bytes:
    """Proportionally shrink an oversized image before it is sent with detail='high'.

    detail='high' bills by patch count (~ceil(W/32)*ceil(H/32)*1.2), so a 2560x1920 overlay
    costs ~5.8k tokens PER send -- and the planner attaches several output figures per call,
    so image tokens dominate the bill. Capping the LONGEST side to ~1024px cuts that ~3-6x
    while the cell masks stay legible. The resize is a THUMBNAIL: aspect ratio is preserved,
    NOTHING is cropped. Env SHAPE_VISION_MAX_PX overrides the cap; <=0 disables (send full-res).
    On any decode error the original bytes are returned unchanged (never break a request).
    """
    if cap is None:
        try:
            cap = int(os.environ.get("SHAPE_VISION_MAX_PX", "1024"))
        except ValueError:
            cap = 1024
    if cap <= 0:
        return item
    try:
        img = Image.open(BytesIO(item))
        if max(img.size) <= cap:
            return item
        img = img.copy()
        img.thumbnail((cap, cap), Image.LANCZOS)   # preserves aspect ratio; no crop
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return item


class ChatOpenAI(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string="gpt-4o-mini-2024-07-18",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool=False,
        # enable_cache: bool=True,
        enable_cache: bool=False, # NOTE: disable cache for now
        api_key: str=None,
        base_url: str=None,                 # OpenAI-COMPATIBLE endpoint (e.g. GLM / 智谱)
        supports_structured: bool=None,     # force the structured-output (.parse) path on/off
        extra_body: dict=None,              # per-request body merged into every call (e.g. GLM {"thinking":...})
        sanitize_messages: bool=False,      # rewrite null/empty/pure-text-list content (GLM err 1213)
        **kwargs):
        """
        :param model_string:
        :param system_prompt:
        :param is_multimodal:
        """
        if enable_cache:
            root = platformdirs.user_cache_dir("shape")
            cache_path = os.path.join(root, f"cache_openai_{model_string}.db")
            # For example, cache_path = /root/.cache/shape/cache_openai_gpt-4o-mini.db
            # print(f"Cache path: {cache_path}")
            
            self.image_cache_dir = os.path.join(root, "image_cache")
            os.makedirs(self.image_cache_dir, exist_ok=True)

            super().__init__(cache_path=cache_path)

        self.system_prompt = system_prompt
        if api_key is None:
            raise ValueError(
                "No API key for this model's provider -- set LLM_API_KEY or pass api_key."
            )

        # Per-request timeout so a stalled/slow call fails FAST (the verifier then falls back to
        # its deterministic verdict) instead of blocking the whole run on the ~600s SDK default.
        # base_url lets an OpenAI-COMPATIBLE backend (e.g. GLM / 智谱) reuse this engine wholesale.
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=float(os.getenv("SHAPE_LLM_TIMEOUT", "120")),
            max_retries=2,
        )
        self.model_string = model_string
        self.is_multimodal = is_multimodal
        self.enable_cache = enable_cache
        # Whether to take the structured-output (.parse) path: default = the known OpenAI list, but
        # an OpenAI-compatible backend can force it on. The structured call is try-wrapped, so an
        # unsupported backend degrades to the deterministic verdict instead of crashing.
        self._structured = (supports_structured if supports_structured is not None
                            else _supports_structured(model_string))
        # OpenAI-compatible backend quirks (e.g. GLM / 智谱), applied to every request via
        # _apply_backend_quirks(); empty/False for OpenAI itself, so GPT calls are unchanged.
        self._extra_body = dict(extra_body) if extra_body else {}
        self._sanitize_messages = bool(sanitize_messages)

    def _apply_backend_quirks(self, api_params: dict) -> None:
        """In-place: adapt one request dict for an OpenAI-compatible backend (e.g. GLM / 智谱).

        * sanitize messages -- GLM (error 1213) rejects null/empty content and the pure-text
          list-form content OpenAI allows; collapse such lists to a plain string and replace
          None/"" with a single space. Image turns (list content carrying an ``image_url``
          part) are left intact so the multimodal vision judge still works.
        * extra_body -- merge a fixed per-request body (e.g. ``{"thinking": {"type": "disabled"}}``
          to turn GLM's default-on thinking off; it is ~5 min/img otherwise).

        No-op when neither is configured (the OpenAI/GPT path).
        """
        if self._sanitize_messages:
            for message in api_params.get("messages") or []:
                content = message.get("content")
                if isinstance(content, list):
                    has_image = any(isinstance(part, dict) and part.get("type") == "image_url"
                                    for part in content)
                    if not has_image:
                        text = " ".join(part.get("text", "") for part in content
                                        if isinstance(part, dict) and part.get("type") == "text")
                        message["content"] = text or " "
                elif content is None or content == "":
                    message["content"] = " "
        if self._extra_body:
            api_params.setdefault("extra_body", {}).update(self._extra_body)
        # Reasoning effort: gpt-5.6/gpt-6/o-series otherwise use a HIGH server-side reasoning
        # budget (~5x the output tokens of a plain call), which inflates cost and can make the
        # VLRR judge over-retune. Pin it so cost/behavior is reproducible. Env override;
        # default "medium". Set SHAPE_REASONING_EFFORT=default to omit the param entirely.
        if _is_reasoning_or_next_gen_model(self.model_string):
            effort = os.environ.get("SHAPE_REASONING_EFFORT", "medium")
            if effort and effort.lower() != "default":
                api_params["reasoning_effort"] = effort.lower()

    @staticmethod
    def _spent_usage(error) -> dict:
        """The tokens a failed structured parse still spent: a parse cut off at the token limit
        (LengthFinishReasonError) carries its completion; any other error counts zero."""
        usage = getattr(getattr(error, "completion", None), "usage", None)
        return {key: int(getattr(usage, key, 0) or 0)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")}

    @staticmethod
    def _usage_dict(response):
        """prompt / completion / total tokens (0 when missing) and, only when the API reports them, cached_tokens
        (usage.prompt_tokens_details) and reasoning_tokens (usage.completion_tokens_details); never a made-up 0."""
        usage = getattr(response, "usage", None)
        out = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
            "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
        }
        for key, details in (("cached_tokens", "prompt_tokens_details"),
                             ("reasoning_tokens", "completion_tokens_details")):
            value = getattr(getattr(usage, details, None), key, None)
            if isinstance(value, int) and not isinstance(value, bool):
                out[key] = value
        return out

    def _glm_structured_completion(self, api_params: dict, response_format):
        """Normalize GLM's OpenAI-compatible structured-output wire behavior.

        The GLM gateway accepts ``response_format`` but complex Pydantic schemas
        can still arrive as prose/XML.  Send the same schema as both an API
        constraint and a concise system instruction, then validate locally.
        This remains one model request and changes no controller/tool logic.
        """
        params = dict(api_params)
        messages = [dict(message) for message in params.get("messages") or []]
        schema = response_format.model_json_schema()
        schema_name = getattr(response_format, "__name__", "structured_response")
        instruction = (
            "Return exactly one JSON object and no prose or markdown. "
            "The object must satisfy this JSON Schema: "
            + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        )
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = str(messages[0].get("content") or "") + "\n\n" + instruction
        else:
            messages.insert(0, {"role": "system", "content": instruction})
        params["messages"] = messages
        params["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        }
        response = self.client.chat.completions.create(**params)
        text = response.choices[0].message.content or ""
        return {"content": parse_json_answer(response_format, text), "usage": self._usage_dict(response)}

    def _parse_structured(self, api_params: dict, response_format):
        """Validate the first JSON value of the same response when SDK parsing rejects surrounding prose."""
        raw = self.client.beta.chat.completions.with_raw_response.parse(**api_params)
        try:
            response = raw.parse()
        except ValueError as exc:
            logger.warning("the SDK's parse failed (%s); reading the same response text", type(exc).__name__)
            body = json.loads(raw.text)
            message = ((body.get("choices") or [{}])[0].get("message")) or {}
            usage = body.get("usage") or {}
            usage_info = {k: usage.get(k) or 0 for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
            for key, details in (("cached_tokens", "prompt_tokens_details"),
                                 ("reasoning_tokens", "completion_tokens_details")):
                value = (usage.get(details) or {}).get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    usage_info[key] = value
            parsed, text, refusal = None, message.get("content"), message.get("refusal")
        else:
            message = response.choices[0].message
            parsed, text = message.parsed, message.content
            refusal = getattr(message, "refusal", None)
            usage_info = self._usage_dict(response)
        if refusal:
            raise RuntimeError(f"the model refused the {response_format.__name__} question: {str(refusal)[:300]}")
        if parsed is None:
            parsed = parse_json_answer(response_format, text)
        return {"content": parsed, "usage": usage_info}

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5), reraise=True,
           retry=retry_if_exception(lambda exc: not is_fatal_llm_error(exc)))
    def generate(self, content: Union[str, List[Union[str, bytes]]], system_prompt=None, **kwargs):
        try:
            # Pin the whole run, including vision and fallback requests, even
            # when a caller supplies an older per-call default.
            completion_limit = os.environ.get("SHAPE_MAX_COMPLETION_TOKENS")
            if completion_limit is not None:
                limit = int(completion_limit)
                if limit <= 0:
                    raise ValueError("SHAPE_MAX_COMPLETION_TOKENS must be positive")
                kwargs["max_tokens"] = limit
            # Print retry attempt information
            attempt_number = self.generate.retry.statistics.get('attempt_number', 0) + 1
            if attempt_number > 1:
                print(f"Attempt {attempt_number} of 5")

            if isinstance(content, str):
                out = self._generate_text(content, system_prompt=system_prompt, **kwargs)
            elif isinstance(content, list):
                if (not self.is_multimodal):
                    raise NotImplementedError("Multimodal generation is only supported for GPT-4 models.")
                out = self._generate_multimodal(content, system_prompt=system_prompt, **kwargs)
            else:
                return None
            from shape.engine import usage as _token_usage   # process-level token accounting
            if isinstance(out, dict):
                _token_usage.record(out.get("usage"))
            _token_usage.note_call(not _token_usage.is_failure(out))  # fail-fast health
            return out

        except openai.LengthFinishReasonError as e:
            print(f"Token limit exceeded: {str(e)}")
            print(f"Tokens used - Completion: {e.completion.usage.completion_tokens}, Prompt: {e.completion.usage.prompt_tokens}, Total: {e.completion.usage.total_tokens}")
            return {
                "error": "token_limit_exceeded",
                "message": str(e),
                "details": {
                    "completion_tokens": e.completion.usage.completion_tokens,
                    "prompt_tokens": e.completion.usage.prompt_tokens,
                    "total_tokens": e.completion.usage.total_tokens
                }
            }
        except openai.RateLimitError as e:
            print(f"Rate limit error encountered: {str(e)}")
            return {
                "error": "rate_limit",
                "message": str(e),
                "details": getattr(e, 'args', None)
            }
        except Exception as e:
            if is_fatal_llm_error(e):
                raise
            print(f"Error in generate method: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {e.args}")
            from shape.engine import usage as _token_usage
            _token_usage.note_call(False)                              # fail-fast health
            return {
                "error": type(e).__name__,
                "message": str(e),
                "details": getattr(e, 'args', None)
            }
        
    def _generate_text(
        self, prompt, system_prompt=None, temperature=0.5, max_tokens=4000, top_p=0.99, response_format=None
    ):

        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        # Check if model requires max_completion_tokens instead of max_tokens
        use_max_completion_tokens = _requires_max_completion_tokens(self.model_string)
        # Check if model only supports temperature=1 (default)
        require_temperature_1 = _requires_temperature_1(self.model_string)

        if self.enable_cache:
            cache_key = sys_prompt_arg + prompt
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                # If cached result doesn't have usage info, add default usage
                if isinstance(cache_or_none, dict) and 'content' in cache_or_none:
                    if 'usage' not in cache_or_none:
                        cache_or_none['usage'] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    cache_or_none['usage']['from_cache'] = True
                return cache_or_none

        if self.model_string in ['o1', 'o1-mini']: # only supports base response currently
            # print(f"Using structured model: {self.model_string}")
            response = self.client.beta.chat.completions.parse(
                model=self.model_string,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=max_tokens
            )
            if response.choices[0].finishreason == "length":
                response_content = "Token limit exceeded"
            else:
                response_content = response.choices[0].message.parsed
            
            # Return both content and usage info
            return {
                "content": response_content,
                "usage": self._usage_dict(response)
            }
        elif self._structured and response_format is not None:
            # print(f"Using structured model: {self.model_string}")
            # For structured models with response_format, we need to use parse method
            # but we also need to get usage information
            try:
                # Use max_completion_tokens for GPT-5 models, max_tokens for others
                api_params = {
                    "model": self.model_string,
                    "messages": [
                        {"role": "system", "content": sys_prompt_arg},
                        {"role": "user", "content": prompt},
                    ],
                    "frequency_penalty": 0,
                    "presence_penalty": 0,
                    "stop": None,
                    "response_format": response_format
                }
                # Only set temperature and top_p if model supports custom values
                if not require_temperature_1:
                    api_params["temperature"] = temperature
                    api_params["top_p"] = top_p
                if use_max_completion_tokens:
                    api_params["max_completion_tokens"] = max_tokens
                else:
                    api_params["max_tokens"] = max_tokens
                self._apply_backend_quirks(api_params)
                if self.model_string.startswith("glm"):
                    return self._glm_structured_completion(api_params, response_format)
                return self._parse_structured(api_params, response_format)
            except Exception as e:
                if is_fatal_llm_error(e):
                    raise
                error_details = str(e)
                print(f"Error in structured model parse: {error_details}")
                import traceback
                traceback.print_exc()
                spent = self._spent_usage(e)
                # Fallback to regular chat completion for usage tracking
                try:
                    # Use max_completion_tokens for GPT-5 models, max_tokens for others
                    api_params = {
                        "model": self.model_string,
                        "messages": [
                            {"role": "system", "content": sys_prompt_arg},
                            {"role": "user", "content": prompt},
                        ],
                        "frequency_penalty": 0,
                        "presence_penalty": 0,
                        "stop": None,
                    }
                    # Only set temperature and top_p if model supports custom values
                    if not require_temperature_1:
                        api_params["temperature"] = temperature
                        api_params["top_p"] = top_p
                    if use_max_completion_tokens:
                        api_params["max_completion_tokens"] = max_tokens
                    else:
                        api_params["max_tokens"] = max_tokens
                    self._apply_backend_quirks(api_params)
                    response = self.client.chat.completions.create(**api_params)
                    response_content = response.choices[0].message.content
                    
                    # Return both content and usage info
                    result = {
                        "content": response_content,
                        "usage": {key: value + spent.get(key, 0)
                                  for key, value in self._usage_dict(response).items()}
                    }
                    # Token usage info removed - no longer tracking costs
                    return result
                except Exception as fallback_error:
                    fallback_error_details = str(fallback_error)
                    print(f"Fallback error: {fallback_error_details}")
                    import traceback
                    traceback.print_exc()
                    raise RuntimeError(
                        f"structured generation failed: {error_details}; plain completion failed: {fallback_error_details}"
                    ) from fallback_error
        else:
            # print(f"Using non-structured model: {self.model_string}")
            # Use max_completion_tokens for GPT-5 models, max_tokens for others
            api_params = {
                "model": self.model_string,
                "messages": [
                    {"role": "system", "content": sys_prompt_arg},
                    {"role": "user", "content": prompt},
                ],
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "stop": None,
            }
            # Only set temperature and top_p if model supports custom values
            if not require_temperature_1:
                api_params["temperature"] = temperature
                api_params["top_p"] = top_p
            if use_max_completion_tokens:
                api_params["max_completion_tokens"] = max_tokens
            else:
                api_params["max_tokens"] = max_tokens
            self._apply_backend_quirks(api_params)
            response = self.client.chat.completions.create(**api_params)
            response_content = response.choices[0].message.content
            
            # Return both content and usage info
            result = {
                "content": response_content,
                "usage": self._usage_dict(response)
            }

        if self.enable_cache:
            self._save_cache(cache_key, result)
        return result

    def __call__(self, prompt, **kwargs):
        result = self.generate(prompt, **kwargs)
        # Return the complete result with usage information
        return result

    def _format_content(self, content: List[Union[str, bytes]]) -> List[dict]:
        formatted_content = []
        # OpenAI supported image formats
        SUPPORTED_FORMATS = ['png', 'jpeg', 'gif', 'webp']
        
        for item in content:
            if isinstance(item, bytes):
                # Proportionally cap the resolution first (detail='high' bills by patch count).
                item = _maybe_downscale_image(item)
                # Detect image format using PIL
                try:
                    img = Image.open(BytesIO(item))
                    original_format = img.format.lower() if img.format else None
                    
                    # Map PIL format to MIME type
                    mime_type_map = {
                        'jpeg': 'jpeg',
                        'jpg': 'jpeg',
                        'png': 'png',
                        'gif': 'gif',
                        'webp': 'webp'
                    }
                    
                    # Check if format is supported
                    if original_format and original_format in mime_type_map:
                        mime_type = mime_type_map[original_format]
                        # Use original bytes if format is supported
                        base64_image = base64.b64encode(item).decode('utf-8')
                    else:
                        # Convert unsupported formats (e.g., TIFF, BMP) to PNG
                        print(f"Warning: Image format '{original_format}' is not supported by OpenAI API. Converting to PNG.")
                        output_buffer = BytesIO()
                        # Convert to RGB if necessary (for formats like RGBA)
                        if img.mode in ('RGBA', 'LA', 'P'):
                            # Create white background for transparent images
                            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                            if img.mode == 'P':
                                img = img.convert('RGBA')
                            rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                            img = rgb_img
                        elif img.mode != 'RGB':
                            img = img.convert('RGB')
                        img.save(output_buffer, format='PNG')
                        converted_bytes = output_buffer.getvalue()
                        base64_image = base64.b64encode(converted_bytes).decode('utf-8')
                        mime_type = 'png'
                except Exception as e:
                    print(f"Error processing image: {e}. Attempting to convert to PNG.")
                    # Fallback: try to open and convert to PNG
                    try:
                        img = Image.open(BytesIO(item))
                        output_buffer = BytesIO()
                        # Convert to RGB if necessary
                        if img.mode in ('RGBA', 'LA', 'P'):
                            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                            if img.mode == 'P':
                                img = img.convert('RGBA')
                            rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                            img = rgb_img
                        elif img.mode != 'RGB':
                            img = img.convert('RGB')
                        img.save(output_buffer, format='PNG')
                        converted_bytes = output_buffer.getvalue()
                        base64_image = base64.b64encode(converted_bytes).decode('utf-8')
                        mime_type = 'png'
                    except Exception as convert_error:
                        print(f"Failed to convert image: {convert_error}. Using original bytes as JPEG.")
                        # Last resort: use original bytes as JPEG
                        base64_image = base64.b64encode(item).decode('utf-8')
                        mime_type = 'jpeg'
                
                formatted_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{mime_type};base64,{base64_image}",
                        # full-resolution vision: without this OpenAI defaults to
                        # "auto" and usually downscales to ~512px (~85 tokens), so the
                        # model sees a blurry thumbnail and cannot judge the masks.
                        "detail": "high",
                    }
                })
            elif isinstance(item, str):
                formatted_content.append({
                    "type": "text",
                    "text": item
                })
            else:
                raise ValueError(f"Unsupported input type: {type(item)}")
        return formatted_content

    def _generate_multimodal(
        self, content: List[Union[str, bytes]], system_prompt=None, temperature=0.5, max_tokens=4000, top_p=0.99, response_format=None
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        formatted_content = self._format_content(content)

        if self.enable_cache:
            cache_key = sys_prompt_arg + json.dumps(formatted_content)
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                # If cached result doesn't have usage info, add default usage
                if isinstance(cache_or_none, dict) and 'content' in cache_or_none:
                    if 'usage' not in cache_or_none:
                        cache_or_none['usage'] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    cache_or_none['usage']['from_cache'] = True
                return cache_or_none

        # Check if model requires max_completion_tokens instead of max_tokens
        use_max_completion_tokens = _requires_max_completion_tokens(self.model_string)
        # Check if model only supports temperature=1 (default)
        require_temperature_1 = _requires_temperature_1(self.model_string)

        if self.model_string in ['o1', 'o1-mini']: # only supports base response currently
            # print(f"Using structured model: {self.model_string}")
            print(f'Max tokens: {max_tokens}')
            response = self.client.chat.completions.create(
                model=self.model_string,
                messages=[
                    {"role": "user", "content": formatted_content},
                ],
                max_completion_tokens=max_tokens
            )
            if response.choices[0].finish_reason == "length":
                response_text = "Token limit exceeded"
            else:
                response_text = response.choices[0].message.content
            
            # Return both content and usage info
            return {
                "content": response_text,
                "usage": self._usage_dict(response)
            }
        elif self._structured and response_format is not None:
            # print(f"Using structured model: {self.model_string}")
            # For structured models with response_format, we need to use parse method
            # but we also need to get usage information
            try:
                # Use max_completion_tokens for GPT-5 models, max_tokens for others
                api_params = {
                    "model": self.model_string,
                    "messages": [
                        {"role": "system", "content": sys_prompt_arg},
                        {"role": "user", "content": formatted_content},
                    ],
                    "response_format": response_format
                }
                # Only set temperature and top_p if model supports custom values
                if not require_temperature_1:
                    api_params["temperature"] = temperature
                    api_params["top_p"] = top_p
                if use_max_completion_tokens:
                    api_params["max_completion_tokens"] = max_tokens
                else:
                    api_params["max_tokens"] = max_tokens
                self._apply_backend_quirks(api_params)
                if self.model_string.startswith("glm"):
                    return self._glm_structured_completion(api_params, response_format)
                return self._parse_structured(api_params, response_format)
            except Exception as e:
                if is_fatal_llm_error(e):
                    raise
                error_details = str(e)
                print(f"Error in multimodal structured model parse: {error_details}")
                import traceback
                traceback.print_exc()
                spent = self._spent_usage(e)
                # Fallback to regular chat completion for usage tracking
                try:
                    # Use max_completion_tokens for GPT-5 models, max_tokens for others
                    api_params = {
                        "model": self.model_string,
                        "messages": [
                            {"role": "system", "content": sys_prompt_arg},
                            {"role": "user", "content": formatted_content},
                        ],
                    }
                    # Only set temperature and top_p if model supports custom values
                    if not require_temperature_1:
                        api_params["temperature"] = temperature
                        api_params["top_p"] = top_p
                    if use_max_completion_tokens:
                        api_params["max_completion_tokens"] = max_tokens
                    else:
                        api_params["max_tokens"] = max_tokens
                    self._apply_backend_quirks(api_params)
                    response = self.client.chat.completions.create(**api_params)
                    response_text = response.choices[0].message.content
                    
                    # Return both content and usage info
                    result = {
                        "content": response_text,
                        "usage": {key: value + spent.get(key, 0)
                                  for key, value in self._usage_dict(response).items()}
                    }
                    # Token usage info removed - no longer tracking costs
                    return result
                except Exception as fallback_error:
                    fallback_error_details = str(fallback_error)
                    print(f"Multimodal fallback error: {fallback_error_details}")
                    import traceback
                    traceback.print_exc()
                    raise RuntimeError(
                        f"structured generation failed: {error_details}; plain completion failed: {fallback_error_details}"
                    ) from fallback_error
        else:
            # print(f"Using non-structured model: {self.model_string}")
            # Use max_completion_tokens for GPT-5 models, max_tokens for others
            api_params = {
                "model": self.model_string,
                "messages": [
                    {"role": "system", "content": sys_prompt_arg},
                    {"role": "user", "content": formatted_content},
                ],
            }
            # Only set temperature and top_p if model supports custom values
            if not require_temperature_1:
                api_params["temperature"] = temperature
                api_params["top_p"] = top_p
            if use_max_completion_tokens:
                api_params["max_completion_tokens"] = max_tokens
            else:
                api_params["max_tokens"] = max_tokens
            self._apply_backend_quirks(api_params)
            response = self.client.chat.completions.create(**api_params)
            response_text = response.choices[0].message.content
            
            # Return both content and usage info
            result = {
                "content": response_text,
                "usage": self._usage_dict(response)
            }

        if self.enable_cache:
            self._save_cache(cache_key, result)
        return result
