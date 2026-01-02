try:
    from openai import OpenAI
except ImportError:
    raise ImportError("If you'd like to use OpenAI models, please install the openai package by running `pip install openai`, and add 'OPENAI_API_KEY' to your environment variables.")

import os
import json
import base64
import platformdirs
from PIL import Image
from io import BytesIO
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from typing import List, Union

from .base import EngineLM, CachedEngine

import openai

from dotenv import load_dotenv
load_dotenv()

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
        **kwargs):
        """
        :param model_string:
        :param system_prompt:
        :param is_multimodal:
        """
        if enable_cache:
            root = platformdirs.user_cache_dir("octotools")
            cache_path = os.path.join(root, f"cache_openai_{model_string}.db")
            # For example, cache_path = /root/.cache/octotools/cache_openai_gpt-4o-mini.db
            # print(f"Cache path: {cache_path}")
            
            self.image_cache_dir = os.path.join(root, "image_cache")
            os.makedirs(self.image_cache_dir, exist_ok=True)

            super().__init__(cache_path=cache_path)

        self.system_prompt = system_prompt
        if api_key is None:
            raise ValueError("Please set the OPENAI_API_KEY environment variable if you'd like to use OpenAI models.")
        
        self.client = OpenAI(
            api_key=api_key,
        )
        self.model_string = model_string
        self.is_multimodal = is_multimodal
        self.enable_cache = enable_cache

        if enable_cache:
            print(f"!! Cache enabled for model: {self.model_string}")
        else:
            print(f"!! Cache disabled for model: {self.model_string}")

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def generate(self, content: Union[str, List[Union[str, bytes]]], system_prompt=None, **kwargs):
        try:
            # Print retry attempt information
            attempt_number = self.generate.retry.statistics.get('attempt_number', 0) + 1
            if attempt_number > 1:
                print(f"Attempt {attempt_number} of 5")

            if isinstance(content, str):
                return self._generate_text(content, system_prompt=system_prompt, **kwargs)
            
            elif isinstance(content, list):
                if (not self.is_multimodal):
                    raise NotImplementedError("Multimodal generation is only supported for GPT-4 models.")
                
                return self._generate_multimodal(content, system_prompt=system_prompt, **kwargs)

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
            print(f"Error in generate method: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {e.args}")
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
        use_max_completion_tokens = self.model_string in MODELS_REQUIRING_MAX_COMPLETION_TOKENS
        # Check if model only supports temperature=1 (default)
        require_temperature_1 = self.model_string in MODELS_REQUIRING_TEMPERATURE_1

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
                "usage": {
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                    "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                    "total_tokens": getattr(response.usage, 'total_tokens', 0)
                } if hasattr(response, 'usage') else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
        elif self.model_string in OPENAI_STRUCTURED_MODELS and response_format is not None:
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
                response = self.client.beta.chat.completions.parse(**api_params)
                response_content = response.choices[0].message.parsed
                
                # Try to get usage information from the response
                usage_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                if hasattr(response, 'usage'):
                    usage_info = {
                        "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                        "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                        "total_tokens": getattr(response.usage, 'total_tokens', 0)
                    }
                elif hasattr(response, '_response') and hasattr(response._response, 'usage'):
                    # Try to access usage through _response attribute
                    usage_info = {
                        "prompt_tokens": getattr(response._response.usage, 'prompt_tokens', 0),
                        "completion_tokens": getattr(response._response.usage, 'completion_tokens', 0),
                        "total_tokens": getattr(response._response.usage, 'total_tokens', 0)
                    }
                
                print(f"Structured model usage info: {usage_info}")
                
                # Return both content and usage info
                return {
                    "content": response_content,
                    "usage": usage_info
                }
            except Exception as e:
                error_details = str(e)
                print(f"Error in structured model parse: {error_details}")
                import traceback
                traceback.print_exc()
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
                    response = self.client.chat.completions.create(**api_params)
                    response_content = response.choices[0].message.content
                    
                    # Return both content and usage info
                    result = {
                        "content": response_content,
                        "usage": {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens
                        }
                    }
                    print(f"Fallback usage info: {result['usage']}")
                    return result
                except Exception as fallback_error:
                    fallback_error_details = str(fallback_error)
                    print(f"Fallback error: {fallback_error_details}")
                    import traceback
                    traceback.print_exc()
                    # Return detailed error message instead of generic one
                    error_message = f"Error in structured generation: Structured parse failed ({error_details}); Fallback also failed ({fallback_error_details})"
                    return {
                        "content": error_message,
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    }
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
            response = self.client.chat.completions.create(**api_params)
            response_content = response.choices[0].message.content
            
            # Return both content and usage info
            result = {
                "content": response_content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
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
                        "url": f"data:image/{mime_type};base64,{base64_image}"
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
        use_max_completion_tokens = self.model_string in MODELS_REQUIRING_MAX_COMPLETION_TOKENS
        # Check if model only supports temperature=1 (default)
        require_temperature_1 = self.model_string in MODELS_REQUIRING_TEMPERATURE_1

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
                "usage": {
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                    "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                    "total_tokens": getattr(response.usage, 'total_tokens', 0)
                } if hasattr(response, 'usage') else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
        elif self.model_string in OPENAI_STRUCTURED_MODELS and response_format is not None:
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
                response = self.client.beta.chat.completions.parse(**api_params)
                response_text = response.choices[0].message.parsed
                
                # Try to get usage information from the response
                usage_info = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                if hasattr(response, 'usage'):
                    usage_info = {
                        "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                        "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                        "total_tokens": getattr(response.usage, 'total_tokens', 0)
                    }
                elif hasattr(response, '_response') and hasattr(response._response, 'usage'):
                    # Try to access usage through _response attribute
                    usage_info = {
                        "prompt_tokens": getattr(response._response.usage, 'prompt_tokens', 0),
                        "completion_tokens": getattr(response._response.usage, 'completion_tokens', 0),
                        "total_tokens": getattr(response._response.usage, 'total_tokens', 0)
                    }
                
                print(f"Multimodal structured model usage info: {usage_info}")
                
                # Return both content and usage info
                return {
                    "content": response_text,
                    "usage": usage_info
                }
            except Exception as e:
                error_details = str(e)
                print(f"Error in multimodal structured model parse: {error_details}")
                import traceback
                traceback.print_exc()
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
                    response = self.client.chat.completions.create(**api_params)
                    response_text = response.choices[0].message.content
                    
                    # Return both content and usage info
                    result = {
                        "content": response_text,
                        "usage": {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens
                        }
                    }
                    print(f"Multimodal fallback usage info: {result['usage']}")
                    return result
                except Exception as fallback_error:
                    fallback_error_details = str(fallback_error)
                    print(f"Multimodal fallback error: {fallback_error_details}")
                    import traceback
                    traceback.print_exc()
                    # Return detailed error message instead of generic one
                    error_message = f"Error in multimodal generation: Structured parse failed ({error_details}); Fallback also failed ({fallback_error_details})"
                    return {
                        "content": error_message,
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    }
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
            response = self.client.chat.completions.create(**api_params)
            response_text = response.choices[0].message.content
            
            # Return both content and usage info
            result = {
                "content": response_text,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }

        if self.enable_cache:
            self._save_cache(cache_key, result)
        return result
