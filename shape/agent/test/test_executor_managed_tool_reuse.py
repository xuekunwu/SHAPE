"""The VLRR reruns of a managed step reuse one tool instance per executor (one solve):
building the tool -- and loading its model -- once, not on every call."""

from __future__ import annotations

from shape.agent.executor import Executor


class _CountingTool:
    built = 0

    def __init__(self):
        type(self).built += 1
        self.calls = 0

    def execute(self, image=None, query_cache_dir=None):
        self.calls += 1
        return {"image": image, "call": self.calls}


def test_managed_reruns_reuse_one_tool_instance(tmp_path, monkeypatch):
    _CountingTool.built = 0
    ex = Executor("gpt-x", query_cache_dir=str(tmp_path))
    monkeypatch.setattr(ex, "_resolve_tool_class", lambda name: _CountingTool)
    results = [ex.run_managed_tool("Counting_Tool", {"image": "a.png"}) for _ in range(3)]
    assert _CountingTool.built == 1
    assert [r["call"] for r in results] == [1, 2, 3]
    other = Executor("gpt-x", query_cache_dir=str(tmp_path))     # a new solve builds its own
    monkeypatch.setattr(other, "_resolve_tool_class", lambda name: _CountingTool)
    other.run_managed_tool("Counting_Tool", {"image": "a.png"})
    assert _CountingTool.built == 2
