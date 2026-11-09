"""A structured answer read from the model's text: the first complete JSON value in it, validated by the schema.

Some models (or gateways) return the JSON answer followed by prose (``{"call": "c02"}\\n\\nRests on ...``), or
wrapped in a code fence. `parse_json_answer` takes the first complete JSON object / array in the text
(``json.JSONDecoder.raw_decode``), validates exactly that JSON text with the schema (``model_validate_json``: the
same validation as a clean answer, never relaxed) and logs the text around it.
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def first_json_value(text: str) -> tuple[Any, int, int]:
    """(the first complete JSON object or array in `text`, its start, its end); ValueError when there is none."""
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char in "{[":
            try:
                value, end = decoder.raw_decode(text, start)
            except ValueError:
                continue
            return value, start, end
    raise ValueError(f"no JSON value in the answer: {text[:200]!r}")


def parse_json_answer(schema, text: str):
    """`text`'s first complete JSON value validated as `schema` (pydantic); the text before or after it is logged.
    Raises ValueError without a JSON value, the pydantic ValidationError when the value does not validate."""
    if not isinstance(text, str):
        raise ValueError(f"no text answer to read a {getattr(schema, '__name__', 'structured')} answer from")
    _, start, end = first_json_value(text)
    parsed = schema.model_validate_json(text[start:end])
    before, after = text[:start].strip(), text[end:].strip()
    if before or after:
        logger.warning("structured answer (%s): text around the JSON value ignored: before %r, after %r",
                       getattr(schema, "__name__", "?"), before[:200], after[:200])
    return parsed
