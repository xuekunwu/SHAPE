"""Benchmark-local observation at SHAPE engine/tool boundaries; no VRR policy changes."""
from __future__ import annotations

import hashlib
import math
import os
import time
import threading
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from unittest.mock import patch

import numpy as np

from common import MODEL, append, cellseg, sha256

_ASSET_HASHES = {}


def safe(value):
    if hasattr(value, "model_dump"):
        return safe(value.model_dump())
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items() if not any(
            word in str(k).lower() for word in ("api_key", "authorization", "password", "credential"))}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, bytes):
        return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, str):
            for key, secret in os.environ.items():
                if len(secret) >= 8 and any(w in key.upper() for w in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
                    value = value.replace(secret, "[REDACTED]")
        return value
    return safe(str(value))


@contextmanager
def llm_audit(out_dir, strict_structured=False):
    """All GPT arms use ChatOpenAI, including internally constructed planner/verifier engines.

    Captures generate-level requests/replies (SDK retry HTTP attempts are not exposed).
    Missing usage/errors are fatal, so none can silently become a deterministic run.
    """
    from shape.engine.openai import ChatOpenAI
    original = ChatOpenAI.generate
    parse_structured = ChatOpenAI._parse_structured
    records = []

    def structured(engine, *args, **kwargs):
        try:
            return parse_structured(engine, *args, **kwargs)
        except Exception as exc:
            from shape.engine.errors import NoEngineError
            raise NoEngineError("Structured preflight failed; plain-completion fallback is disabled") from exc

    @wraps(original)
    def generate(engine, content, *args, **kwargs):
        effort = os.environ.get("ORGANOID_PINNED_REASONING_EFFORT", "high")
        if engine.model_string != MODEL or os.environ.get("SHAPE_REASONING_EFFORT") != effort:
            raise RuntimeError("LLM model or reasoning effort drifted")
        started = time.time()
        request = safe(content)
        if isinstance(content, list):
            for index, item in enumerate(content):
                if isinstance(item, bytes):
                    folder = Path(out_dir) / "llm_images"
                    folder.mkdir(exist_ok=True)
                    name = hashlib.sha256(item).hexdigest() + ".png"
                    (folder / name).write_bytes(item)
                    request[index]["asset"] = "llm_images/" + name
        schema = kwargs.get("response_format")
        record = {"started_unix": started, "model": MODEL, "reasoning_effort": effort,
                  "multimodal": engine.is_multimodal, "request": request,
                  "system_prompt": safe(kwargs.get("system_prompt") or getattr(engine, "system_prompt", None)),
                  "schema": getattr(schema, "__name__", None),
                  "schema_json": schema.model_json_schema() if hasattr(schema, "model_json_schema") else None,
                  "options": safe({k: v for k, v in kwargs.items() if k != "response_format"})}
        try:
            reply = original(engine, content, *args, **kwargs)
            record["reply"] = safe(reply)
            usage = reply.get("usage") if isinstance(reply, dict) else None
            record["tokens"] = (usage or {}).get("total_tokens")
            if not usage or not isinstance(record["tokens"], int) or record["tokens"] <= 0:
                from shape.engine.errors import NoEngineError
                raise NoEngineError("LLM reply lacks positive reported token usage")
            from shape.engine.usage import is_failure
            if is_failure(reply):
                raise RuntimeError("LLM returned an error; inspect LLM_CALLS.jsonl")
            return reply
        except Exception as exc:
            record["error"] = safe(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            record["seconds"] = time.time() - started
            records.append(record)
            append(Path(out_dir) / "LLM_CALLS.jsonl", record)

    with patch.object(ChatOpenAI, "generate", generate):
        if strict_structured:
            with patch.object(ChatOpenAI, "_parse_structured", structured):
                yield records
        else:
            yield records


@contextmanager
def tool_audit(out_dir, tool_class):
    """Count outermost model.eval inference requests and tool executions, failures included.

    Cellpose internally recurses for a one-element image list; this is one native request.
    """
    from cellpose.models import CellposeModel
    original, native = tool_class.execute, CellposeModel.eval
    calls, evaluations = [], []
    depth = threading.local()

    @wraps(original)
    def execute(tool, *args, **kwargs):
        start = time.time()
        native_before = len(evaluations)
        record = {"started_unix": start, "tool": tool_class.__name__, "parameters": safe(kwargs)}
        try:
            result = original(tool, *args, **kwargs)
            record["result"] = safe(result)
            path = result.get("mask_path") if isinstance(result, dict) else None
            if path and Path(path).is_file():
                snapshots = Path(out_dir) / "audit_masks"
                snapshots.mkdir(exist_ok=True)
                snapshot = snapshots / f"{len(calls) + 1:04d}.npy"
                np.save(snapshot, cellseg.load_mask(path))
                record["snapshot"] = str(snapshot)
            return result
        except Exception as exc:
            record["error"] = safe(str(exc))
            raise
        finally:
            record["seconds"] = time.time() - start
            record["native_requests"] = len(evaluations) - native_before
            calls.append(record)
            append(Path(out_dir) / "TOOL_CALLS.jsonl", record)

    @wraps(native)
    def evaluate(model, *args, **kwargs):
        if getattr(depth, "active", False):
            return native(model, *args, **kwargs)
        start = time.time()
        asset = getattr(model, "pretrained_model", None)
        record = {"started_unix": start, "parameters": safe(kwargs), "weights": safe(asset),
                  "device": str(getattr(model, "device", "unknown"))}
        if isinstance(asset, (str, Path)) and Path(asset).is_file():
            info = Path(asset).stat()
            asset_key = (str(Path(asset).resolve()), info.st_size, info.st_mtime_ns)
            if asset_key not in _ASSET_HASHES:
                _ASSET_HASHES[asset_key] = sha256(asset)
            record["weights_sha256"] = _ASSET_HASHES[asset_key]
        compute_start = time.time()
        depth.active = True
        try:
            return native(model, *args, **kwargs)
        except Exception as exc:
            record["error"] = safe(str(exc))
            raise
        finally:
            depth.active = False
            record["seconds"] = time.time() - compute_start
            evaluations.append(record)
            append(Path(out_dir) / "NATIVE_CALLS.jsonl", record)

    with patch.object(tool_class, "execute", execute), patch.object(CellposeModel, "eval", evaluate):
        yield calls, evaluations
