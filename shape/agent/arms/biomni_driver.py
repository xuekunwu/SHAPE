"""One Biomni decision round: `python -I biomni_driver.py request.json result.json` (Biomni).

Runs under Biomni's own virtual environment and imports nothing from this repository. Biomni is decision-only, as
the Codex arm is: its A1 agent gets no tools, data lake or software list, and code it asks to execute is refused,
never run. The request holds the prompt, the attached image paths, the decision schema, the model, the gateway base
URL and the round's time cap. The result holds the decision (or why the reply is not one), the summed token usage,
the generate steps, the refused executions and the messages as text with the key redacted. Nothing Biomni prints
while it is constructed reaches a file: A1 prints a masked key and its configuration.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import re
import sys

REFUSAL = "Code execution is not available in this benchmark. Give your decision inside solution tags."
# Models that reject a request whose last message is an assistant message ("does not support assistant message
# prefill. The conversation must end with a user message."). Biomni A1 appends each refused-execution
# observation as an AIMessage, so a later generate can end with one -> _end_with_user fixes the request.
NO_PREFILL_MODELS = frozenset({"claude-sonnet-5-5"})
RECURSION_LIMIT = 30
THREAD_ID = "cpjump1"    # A1's graph has a checkpointer: a stream needs a thread id (A1.go passes 42)
SOLUTION = re.compile(r"<solution>(.*?)</solution>", re.DOTALL | re.IGNORECASE)     # as A1 matches it
FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def instruction(schema: dict) -> str:
    """The line appended to the user prompt (Biomni has no structured output)."""
    return ("Return your decision as one JSON object with exactly the keys action, request_json, reason (schema: "
            + json.dumps(schema, separators=(",", ":")) + ") inside <solution></solution>.")


def text_of(content) -> str:
    """A message's content as text; image parts are named, not inlined."""
    if isinstance(content, str):
        return content
    parts = []
    for part in content or []:
        if isinstance(part, dict) and part.get("type") == "text":
            parts.append(str(part.get("text", "")))
        elif isinstance(part, dict) and part.get("type") == "image_url":
            parts.append("[image]")
        else:
            parts.append(str(part))
    return "\n".join(parts)


def redact(text: str, secret) -> str:
    """Replace every secret (a single value or an iterable of them) with [REDACTED]."""
    secrets = [secret] if isinstance(secret, (str, bytes)) or secret is None else list(secret)
    for value in secrets:
        if value:
            text = text.replace(value, "[REDACTED]")
    return text


def parse_solution(texts: list[str], schema: dict) -> dict:
    """{"decision": ...} from the last <solution> of the replies, or {"malformed": why}. A markdown code fence around
    the JSON is removed; the keys must be exactly the schema's and the action one of its enum."""
    found = [match for text in texts for match in SOLUTION.findall(text)]
    if not found:
        return {"malformed": "no <solution></solution> in the replies"}
    body = found[-1].strip()
    fenced = FENCE.match(body)
    try:
        value = json.loads(fenced.group(1) if fenced else body)
    except ValueError:
        return {"malformed": "the solution is not one JSON object"}
    if not isinstance(value, dict) or set(value) != set(schema["required"]):
        return {"malformed": f"the solution's keys must be exactly {sorted(schema['required'])}"}
    if value["action"] not in schema["properties"]["action"]["enum"]:
        return {"malformed": f"the action must be one of {schema['properties']['action']['enum']}"}
    return {"decision": value}


def usage_of(messages) -> dict:
    """Token usage summed over the generated messages (`usage_metadata`)."""
    total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for message in messages:
        usage = getattr(message, "usage_metadata", None) or {}
        for key in total:
            value = usage.get(key, 0)
            if type(value) is not int or value < 0:
                raise ValueError("Biomni reported malformed token usage")
            total[key] += value
    return total


def usage_counter():
    """A LangChain callback that keeps each LLM call's generated messages, whose `usage_metadata` holds the call's
    tokens. A1's generate step rebuilds its reply as a new AIMessage without that usage (found on the VM with Biomni
    0.0.8), so the conversation's messages carry none."""
    from langchain_core.callbacks import BaseCallbackHandler

    class UsageCounter(BaseCallbackHandler):
        def __init__(self):
            super().__init__()
            self.messages = []

        def on_llm_end(self, response, **kwargs):
            for generations in getattr(response, "generations", None) or []:
                for generation in generations:
                    message = getattr(generation, "message", None)
                    if message is not None:
                        self.messages.append(message)

    return UsageCounter()


class Silenced:
    """Python- and descriptor-level stdout/stderr to the null device. The sink stays open: a log handler made while
    silenced keeps writing to it, never to a file."""

    def __enter__(self):
        sys.stdout.flush()
        sys.stderr.flush()
        self.sink = open(os.devnull, "w")
        self.saved = (os.dup(1), os.dup(2), sys.stdout, sys.stderr)
        os.dup2(self.sink.fileno(), 1)
        os.dup2(self.sink.fileno(), 2)
        sys.stdout = sys.stderr = self.sink
        return self

    def __exit__(self, *exc):
        self.sink.flush()
        out, err, sys.stdout, sys.stderr = self.saved
        os.dup2(out, 1)
        os.dup2(err, 2)
        os.close(out)
        os.close(err)
        return False


def build(request: dict, workspace: Path):
    """A1 with its resources cleared and code execution refused; (agent, the a1 module). The reasoner is Biomni's
    "Custom" OpenAI-compatible source against the gateway (gpt-*, LLM_API_KEY) or its native "Anthropic" source
    (claude-*, ANTHROPIC_API_KEY); the gateway serves no claude model, so claude-* must use the Anthropic source."""
    source = request.get("source", "Custom")
    anthropic = source == "Anthropic"
    with Silenced():
        import biomni.agent.a1 as a1
        kwargs = dict(path=str(workspace / "biomni"), llm=request["model"], source=source,
                      api_key=os.environ["ANTHROPIC_API_KEY" if anthropic else "LLM_API_KEY"],
                      use_tool_retriever=False, expected_data_lake_files=[], commercial_mode=False,
                      timeout_seconds=int(request["timeout_seconds"]))
        if request.get("base_url"):      # omitted for the Anthropic source (its SDK default endpoint)
            kwargs["base_url"] = request["base_url"]
        agent = a1.A1(**kwargs)
        agent.module2api = {}
        agent.library_content_dict = {}
        agent.data_lake_dict = {}
        agent.configure()
        uncap(agent.llm, anthropic)
        if request["model"] in NO_PREFILL_MODELS:
            _end_with_user(agent.llm)
        if request.get("reasoning_effort"):
            configure_reasoning(agent.llm, request["reasoning_effort"])
    return agent, a1


def configure_reasoning(llm, effort):
    if effort not in ("low", "medium", "high"):
        raise ValueError("invalid Biomni reasoning effort")
    llm.reasoning_effort = effort
    llm.temperature = None
    # Fail closed if LangChain drops the field before the OpenAI request.
    if llm._default_params.get("reasoning_effort") != effort:
        raise RuntimeError("Biomni client did not retain reasoning_effort")


def _drop_unsupported(llm) -> None:
    """gpt-6-sol is a reasoning model that rejects the legacy Chat Completions `stop` and non-default `temperature`
    (400 Unsupported parameter). Biomni's own guard (_ChatOpenAIResponsesNoStop) only triggers for gpt-5* names, so
    gpt-6* gets a plain ChatOpenAI that still sends them. Strip both from every request payload (gpt-5.6-luna already
    ran this way through Biomni's gpt-5 branch, so this is parity, not a behaviour change)."""
    base = type(llm)._get_request_payload

    def patched(input_, *, stop=None, **kwargs):
        payload = base(llm, input_, stop=stop, **kwargs)
        payload.pop("stop", None)
        payload.pop("temperature", None)
        return payload

    llm._get_request_payload = patched


def _end_with_user(llm) -> None:
    """Keep every request's conversation ending with a user turn. Biomni A1 adds each refused-execution
    observation (`<observation>...`) as an AIMessage, so a generate after one would send Anthropic a
    conversation ending with an assistant message -- which claude-sonnet-5-5 rejects (no assistant-message
    prefill). Before each invoke, convert a trailing assistant message to a human one (content preserved).
    A1's own stored messages are untouched; only the copy sent to the API is fixed. No-op when the
    conversation already ends with a non-assistant message."""
    from langchain_core.messages import AIMessage, HumanMessage
    base = llm.invoke

    def patched(input_, *args, **kwargs):
        if isinstance(input_, list) and input_ and isinstance(input_[-1], AIMessage):
            input_ = [*input_[:-1], HumanMessage(content=input_[-1].content)]
        return base(input_, *args, **kwargs)

    # ChatAnthropic is a pydantic model: plain `llm.invoke = patched` goes through its __setattr__,
    # which rejects any name that is not a declared field ("object has no field \"invoke\""— seed2's
    # biomni_sonnet55 crash). object.__setattr__ bypasses that validation and sets the instance
    # attribute directly; normal attribute lookup then finds it before the class's bound method.
    object.__setattr__(llm, "invoke", patched)


def uncap(llm, anthropic: bool = False) -> None:
    """As Codex's client (user, 2026-10-12): no output-token cap (Biomni sets max_tokens 8192, the reasoning tokens
    included) and no retries (LangChain's default 2; Codex: request/stream retries 0, so an API error fails the arm).
    The OpenAI client is rebuilt with max_retries 0. Anthropic REQUIRES a max_tokens, so its cap is lifted high
    (32000) rather than removed, and it has no root_client to rebuild."""
    llm.max_retries = 0
    if anthropic:
        llm.max_tokens = 32000
        # claude-opus-5 rejects temperature ("deprecated for this model"); None makes langchain omit it.
        try:
            llm.temperature = None
        except Exception:
            _drop_unsupported(llm)
        return
    llm.max_tokens = None
    _drop_unsupported(llm)
    if getattr(llm, "root_client", None) is not None:
        llm.root_client = llm.root_client.with_options(max_retries=0)
        llm.client = llm.root_client.chat.completions
    if getattr(llm, "root_async_client", None) is not None:
        llm.root_async_client = llm.root_async_client.with_options(max_retries=0)
        llm.async_client = llm.root_async_client.chat.completions


def run(request: dict, workspace: Path) -> dict:
    from langchain_core.messages import HumanMessage
    from langgraph.errors import GraphRecursionError
    agent, a1 = build(request, workspace)
    refused = [0]

    def refuse(*args, **kwargs):
        refused[0] += 1
        return REFUSAL

    a1.run_with_timeout = refuse
    counter = usage_counter()
    agent.llm.callbacks = [*(getattr(agent.llm, "callbacks", None) or []), counter]
    content = [{"type": "text", "text": request["prompt"] + "\n" + instruction(request["schema"])}]
    for path in request["images"]:
        data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}})
    inputs = {"messages": [HumanMessage(content=content)], "next_step": None}
    state, limited = {}, False
    try:
        for state in agent.app.stream(inputs, stream_mode="values", config={"recursion_limit": RECURSION_LIMIT, "configurable": {"thread_id": THREAD_ID}}):
            pass
    except GraphRecursionError:
        limited = True
    messages = list((state or {}).get("messages") or [])
    replies = [m for m in messages if getattr(m, "type", None) == "ai"]
    # A1 adds the execution's observation as an AI message too: the generate steps are the other replies (QC F8)
    generated = [m for m in replies if not text_of(m.content).lstrip().startswith("<observation>")]
    outcome = parse_solution([text_of(m.content) for m in replies], request["schema"])
    if limited and "malformed" in outcome:
        outcome["malformed"] += f" (the step limit, recursion_limit {RECURSION_LIMIT}, ended the round)"
    secret = [os.environ.get("LLM_API_KEY"), os.environ.get("ANTHROPIC_API_KEY")]
    return {**outcome, "usage": usage_of(counter.messages), "steps": len(generated), "refused_executions": refused[0],
            **({"reasoning_effort": agent.llm.reasoning_effort} if request.get("reasoning_effort") else {}),
            "messages": [{"role": getattr(m, "type", type(m).__name__), "content": redact(text_of(m.content), secret)}
                         for m in messages]}


def main(argv=None) -> None:
    request_path, result_path = (argv or sys.argv[1:])[:2]
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    result = run(request, Path(request_path).parent)
    Path(result_path).write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
