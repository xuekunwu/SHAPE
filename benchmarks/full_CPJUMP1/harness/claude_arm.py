"""Real Claude Code, with structured native-tool requests and a read-only evidence workspace.

Claude selects parameters and checkpoints; the trusted driver executes native tools.
It does not use SHAPE's LLM engine or its parameter-search loop.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from .data import digest, read_json, within, write_json

OUTPUTS = {"segment": ("segmentation.json",), "crop": ("cells.json", "excluded_cells.json"),
           "embed": ("SSL_SELECTION.json",), "batch": ("corrected.npy", "batch_SEARCH.json")}
INPUTS = {"segment": (), "crop": ("segmentation.json",), "embed": ("cells.json",),
          "batch": ("profiles.npy", "profiles.csv")}
SCRUB = ("LLM_API_KEY", "LLM_BASE_URL", "STANFORD_AI_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL",
         "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", "ZHIPUAI_API_KEY",
         "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY")
SCHEMA = {"type": "object", "additionalProperties": False,
          "properties": {"action": {"type": "string", "enum": ["trial", "finish"]},
                         "params_json": {"type": "string"}, "parent": {"type": "string"},
                         "selected": {"type": "string"}, "reason": {"type": "string"}},
          "required": ["action", "params_json", "parent", "selected", "reason"]}


def registry(stage, spec=None):
    """The stage's lever registry; `spec` (a Dataset) selects its own segmenter's CARD for the segment stage."""
    from benchmarks._shared.harness.shape_arm import tools_common
    from .cli import REPO
    if stage == "segment" and spec is not None and spec.segment_tool != "cell_segmenter":
        from .pipeline import segment_registry
        return segment_registry(spec)
    return tools_common().tool_levers("batch_correct" if stage == "batch" else stage, REPO)


def parameters(stage, given, spec=None):
    from benchmarks._shared.harness.shape_arm import tools_common
    if not isinstance(given, dict):
        raise ValueError("params_json must encode an object")
    levers = registry(stage, spec)
    if set(given) - set(levers):
        raise ValueError("unknown tool parameter")
    C = tools_common()
    # Runtime paths, splits, seeds and channel mapping cannot be supplied as parameters.
    return {key: C.check_value(levers[key], value) for key, value in given.items()}


def setup(root: Path, model: str, evidence: list[Path]) -> Path:
    if not model or not model.startswith("claude-"):
        raise ValueError("Claude Code requires an explicit claude-* gateway model ID")
    if os.name != "posix":
        raise RuntimeError("Claude Code benchmark execution requires Linux")
    for parent in [root, *root.parents]:
        for name in ("CLAUDE.md", "CLAUDE.local.md", ".claude"):
            if (parent / name).exists():
                raise ValueError(f"unexpected inherited Claude instructions: {parent / name}")
    if os.environ.get("ANTHROPIC_BASE_URL", "").rstrip("/") != "https://llm-gateway.example.edu":
        raise ValueError("Claude Code must use the Stanford Anthropic-compatible route")
    for key in SCRUB:
        os.environ.pop(key, None)
    for key in ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
                "ANTHROPIC_DEFAULT_HAIKU_MODEL", "ANTHROPIC_SMALL_FAST_MODEL"):
        os.environ[key] = model
    os.environ.update(CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1", DISABLE_TELEMETRY="1")
    root.mkdir(parents=True, exist_ok=True)
    helper = root / "key_helper.sh"
    helper.write_text('#!/usr/bin/env bash\nsource /home/users/your-user/api.sh >/dev/null 2>&1\n'
                      '[[ "${LLM_BASE_URL%/}" == https://llm-gateway.example.edu/v1 && -n "$LLM_API_KEY" ]] || exit 1\n'
                      'printf "%s" "$LLM_API_KEY"\n', encoding="ascii")
    helper.chmod(0o700)
    roots = root / "read_roots.json"
    write_json(roots, [str(p.resolve()) for p in evidence])
    from shape.agent.arms import read_guard
    hook = shlex.join([sys.executable, read_guard.__file__, str(roots)])
    settings = root / "settings.json"
    write_json(settings, {"apiKeyHelper": shlex.quote(str(helper)),
                         "hooks": {"PreToolUse": [{"matcher": ".*", "hooks": [{"type": "command", "command": hook}]}]}})
    return settings


def command(prompt, model, settings, session=None):
    from benchmarks._shared.harness.agents import claude_cmd
    cmd = claude_cmd(prompt, model, session, [], str(settings))
    cmd[cmd.index("--allowedTools") + 1] = "Read"
    return cmd + ["--tools", "Read", "--setting-sources", "", "--strict-mcp-config",
                  "--mcp-config", '{"mcpServers":{}}', "--json-schema", json.dumps(SCHEMA)]


def response(trace: Path, model: str):
    events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line.strip()]
    init = [e for e in events if e.get("type") == "system" and e.get("subtype") == "init"]
    final = [e for e in events if e.get("type") == "result"]
    if not init or init[-1].get("model") != model:
        raise RuntimeError("Claude Code model missing or changed")
    if not final or final[-1].get("is_error") or final[-1].get("subtype") != "success":
        raise RuntimeError("Claude Code did not return a successful result")
    result = final[-1]
    usage = result.get("usage") or {}
    tokens = sum(usage.get(k, 0) for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
    if tokens <= 0:
        raise RuntimeError("Claude Code reported no positive token usage")
    models = result.get("modelUsage") or {}
    if models and set(models) != {model}:
        raise RuntimeError("Claude Code used an unexpected background/fallback model")
    decision = result.get("structured_output")
    if not isinstance(decision, dict) or set(decision) != set(SCHEMA["required"]):
        raise ValueError("Claude Code structured tool request is missing or malformed")
    if any(not isinstance(v, str) for v in decision.values()) or decision["action"] not in ("trial", "finish"):
        raise ValueError("invalid Claude Code decision types")
    if not decision["reason"].strip():
        raise ValueError("decision must cite evidence")
    return decision, {"total_tokens": tokens, "usage": usage, "model": model}


def invoke(root, prompt, model, settings, deadline, index, session=None):
    from benchmarks._shared.harness.agents import stream, kill_run_processes
    trace = root / "traces" / f"round_{index:03d}.jsonl"
    trace.parent.mkdir(exist_ok=True)
    try:
        result = stream(command(prompt, model, settings, session), root, trace, deadline, run=root)
    finally:
        kill_run_processes(root, grace=2)
    if result["returncode"] or result["timed_out"]:
        raise RuntimeError("Claude Code exited unsuccessfully or exceeded its deadline")
    decision, usage = response(trace, model)
    return decision, usage, result.get("session_id")


def preflight(root, model):
    help_text = subprocess.check_output(["claude", "--help"], text=True, timeout=30)
    for flag in ("--json-schema", "--tools", "--settings", "--setting-sources", "--strict-mcp-config"):
        if flag not in help_text:
            raise RuntimeError(f"installed Claude Code lacks required CLI flag: {flag}")
    version = subprocess.check_output(["claude", "--version"], text=True, timeout=30).strip()
    settings = setup(root, model, [])
    decision, usage, _ = invoke(root, 'Connectivity check. Return action="finish", params_json="{}", '
                              'parent="", selected="", reason="gateway preflight OK". Do not use tools.',
                              model, settings, time.time() + 180, 0)
    if decision["action"] != "finish":
        raise ValueError("unexpected preflight response")
    return {"model": model, "claude_version": version, "preflight_tokens": usage["total_tokens"],
            "base_url": "https://llm-gateway.example.edu", "key_variable": "apiKeyHelper:LLM_API_KEY"}


def verify_trial(folder, record):
    for path, sha in record["artifacts"].items():
        if digest(within(folder, path)) != sha:
            raise ValueError("Claude trial artifact changed")


def visual_evidence(stage, folder):
    """Publish inspectable PNGs; scientific TIFF pixels are never modified."""
    import numpy as np
    if stage == "segment":
        reports = sorted(folder.glob("segment/*/defaults.json"))
        chosen = np.linspace(0, len(reports) - 1, min(16, len(reports)), dtype=int)
        return [read_json(reports[i]).get("overlay_path") for i in chosen]
    if stage != "crop":
        return []
    from PIL import Image, ImageDraw
    import tifffile
    from .data import CHANNELS
    cells = read_json(folder / "cells.json")
    by_image = {}
    for cell in cells:
        by_image.setdefault(cell["image_id"], []).append(cell)
    identifiers = sorted(by_image)
    outputs = []
    for index in np.linspace(0, len(identifiers) - 1, min(16, len(identifiers)), dtype=int):
        ident = identifiers[index]
        chosen = sorted(by_image[ident], key=lambda c: c["cell_id"])[:6]
        canvas = Image.new("RGB", (5 * 160, len(chosen) * 184 + 24), "white")
        draw = ImageDraw.Draw(canvas)
        for j, name in enumerate(CHANNELS):
            draw.text((j * 160 + 4, 4), name, fill="black")
        for i, cell in enumerate(chosen):
            raw = tifffile.imread(cell["crop_path"])
            draw.text((4, 24 + i * 184), cell["cell_id"], fill="black")
            for j, channel in enumerate(raw):
                lo, hi = np.percentile(channel, [1, 99])
                pixels = np.clip((channel.astype(float) - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
                tile = Image.fromarray(pixels).convert("RGB").resize((160, 160))
                canvas.paste(tile, (j * 160, 48 + i * 184))
        path = folder / "evidence" / f"{ident}.png"
        path.parent.mkdir(exist_ok=True)
        canvas.save(path)
        outputs.append(str(path))
    return outputs


def run_trial(stage, prep, run, config, trials, name, params, parent=""):
    from .cli import child
    from .pipeline import dataset
    folder = run / "agent" / stage / "trials" / name
    folder.mkdir(parents=True)
    cfg = {**config, "tool_params": parameters(stage, params), "fit_only": stage == "embed"}
    # Legacy (no CLI route): the Codex embed worker no longer writes a gate, so missing feedback refuses tuning.
    if stage == "embed" and trials and ((trials["t000"].get("feedback") or {}).get("admission") or {}).get("decision") != "RUN":
        raise ValueError("shared SSL measurability gate does not admit tuning")
    if stage == "batch":
        cfg["uncorrected"] = name == "t000"
        if trials and trials["t000"]["feedback"]["admission"] != "RUN":
            raise ValueError("shared batch gate does not admit tuning")
    if parent:
        if stage != "embed" or parent not in trials:
            raise ValueError("only SSL may resume an already completed trial")
        previous = folder.parent / parent
        verify_trial(previous, trials[parent])
        checkpoint = read_json(previous / "SSL_SELECTION.json").get("checkpoint")
        if not checkpoint or cfg["tool_params"].get("adaptation_depth", "zero_shot") == "zero_shot":
            raise ValueError("SSL resume needs a trained checkpoint and non-zero-shot adaptation")
        cfg["resume_checkpoint"] = checkpoint
    for filename in INPUTS[stage]:
        shutil.copy2(run / filename, folder / filename)
    write_json(folder / "CONFIG.json", cfg)
    started = time.time()
    child([sys.executable, "-m", dataset(config).cli, "native-trial", stage, "--prep", str(prep), "--run", str(folder)],
          folder / "tool.log", config["deadline"] - time.time(), new_session=False)
    previews = visual_evidence(stage, folder)
    artifacts = {p.relative_to(folder).as_posix(): digest(p) for p in folder.rglob("*") if p.is_file()}
    record = {"status": "complete", "params": params, "parent": parent, "seconds": time.time() - started,
              "artifacts": artifacts, "feedback": read_json(folder / "FEEDBACK.json") if (folder / "FEEDBACK.json").exists() else {},
              "outputs": {name: str(folder / name) for name in OUTPUTS[stage]},
              "previews": [p for p in previews if p]}
    if stage == "segment":
        masks = read_json(folder / "segmentation.json")
        record["feedback"] = {"views": len(masks), "cells": sum(r["n_cells"] for r in masks),
                              "min_cells": min(r["n_cells"] for r in masks)}
    elif stage == "crop":
        record["feedback"] = {"cells": len(read_json(folder / "cells.json"))}
    trials[name] = record
    return folder


def run_stage(stage, prep, run, config):
    from .cli import child, REPO
    from .pipeline import dataset
    spec = dataset(config)
    root = run / "agent" / stage
    docs = root / "documentation"
    docs.mkdir(parents=True)
    tool = {"segment": spec.segment_tool, "crop": "single_cell_cropper",
            "embed": "cell_state_analyzer", "batch": "batch_corrector"}[stage]
    card = docs / "CARD.md"
    shutil.copy2(REPO / "shape" / "tools" / tool / "CARD.md", card)
    settings = setup(root / "auth", config["model"], [root / "trials", prep / "public", docs])
    trials, rounds, session = {}, [], None
    baseline = {"adaptation_depth": "zero_shot"} if stage == "embed" else {}
    run_trial(stage, prep, run, config, trials, "t000", baseline)
    task = (f"You are the Claude Code arm of {spec.name}, stage {stage}. Choose native-tool parameters autonomously. "
            f"Read the tool's CARD at {card} for parameter semantics; only the registry below is exposed. "
            "This is a controlled tool-orchestration comparison, not an unrestricted coding task. Only Read is available. "
            "Inspect the published tool reports and overlays using absolute paths. Never read private labels or score evaluation genes. "
            "Each trial runs this entire stage on the same images; budget includes every trial. Stop tuning if quality suffices, "
            "admission fails, evidence is weak, or time is short. Prefer broad coverage then controlled narrowing when warranted. "
            "For SSL, trials fit only development train wells and report development validation; max_epochs is additional training "
            "when resuming. parent is a prior trained trial ID or empty for pretrained. Final inference is frozen and requires time. "
            "Return action trial with params_json containing ONLY supported parameters and parent (SSL only), selected empty; "
            "or action finish, selected an existing trial ID, params_json '{}', parent empty. Give evidence in reason. "
            f"All unspecified parameters use native defaults, except SSL starts at zero_shot. Registry: "
            f"{json.dumps({k: asdict(v) for k, v in registry(stage).items() if v.kind != 'per_cell'})}. ")
    for index in range(32):
        write_json(root / "TRIALS.json", trials)
        # Publish summaries directly, with full artifact paths for read-only visual/tool inspection.
        visible = {key: {k: v for k, v in value.items() if k != "artifacts"} for key, value in trials.items()}
        prompt = task + f"Seconds remaining for the entire arm: {config['deadline'] - time.time():.0f}. Trials: {json.dumps(visible)}"
        decision, usage, session = invoke(root, prompt, config["model"], settings,
                                         min(config["deadline"], time.time() + 600), index, session)
        rounds.append({"decision": decision, **usage})
        write_json(root / "ROUNDS.json", rounds)
        if decision["action"] == "finish":
            if decision["selected"] not in trials or decision["parent"] or json.loads(decision["params_json"]) != {}:
                raise ValueError("invalid final trial selection")
            chosen = root / "trials" / decision["selected"]
            verify_trial(chosen, trials[decision["selected"]])
            for filename in OUTPUTS[stage]:
                shutil.copy2(chosen / filename, run / filename)
            write_json(root / "SELECTED.json", {"trial": decision["selected"], "reason": decision["reason"],
                       "total_tokens": sum(r["total_tokens"] for r in rounds), "n_trials": len(trials),
                       "policy": "claude-code-native-tools-v1"})
            if stage == "embed":
                # A trusted fresh process performs full-data inference only after selection is final.
                write_json(root / "INFER_CONFIG.json", {**config, "ssl_selection": read_json(run / "SSL_SELECTION.json")})
                child([sys.executable, "-m", spec.cli, "native-trial", "embed", "--prep", str(prep), "--run", str(run),
                       "--config", str(root / "INFER_CONFIG.json")], run / "logs" / "frozen_inference.log",
                      config["deadline"] - time.time(), new_session=False)
            return
        if decision["selected"]:
            raise ValueError("a new trial cannot also select a final output")
        run_trial(stage, prep, run, config, trials, f"t{len(trials):03d}",
                  json.loads(decision["params_json"]), decision["parent"])
    raise RuntimeError("Claude Code exhausted its decision-round limit without selecting a result")
