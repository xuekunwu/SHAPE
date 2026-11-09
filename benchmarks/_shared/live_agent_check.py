"""Live-agent smoke check (Phase 4a): run the REAL solve() loop on one image.

Unlike the benchmark harnesses (which drive a fixed deterministic chain), this
exercises the LLM-planned agent path -- so it validates that, with the post-LLM
`required_next` override removed, the LLM completes the bioimage pipeline on its
own, and that the ACTIVE plan gate blocks a structurally unsound plan (bounded
re-planning) instead of executing-then-failing.

Usage:
    export LLM_API_KEY=...      # the planner/executor call an LLM
    export PYTHONPATH=<repo root>       # package is not pip-installed
    # on a GPU node (cellpose-SAM + DINOv3 need a GPU):
    python benchmarks/live_agent_check.py <image_path> ["<question>"]

Needs: LLM_API_KEY, a GPU env (cellpose-SAM / DINOv3 / scanpy), and outbound
internet for the LLM call + first-time model-weight downloads.
"""
from __future__ import annotations

import os
import sys

from shape.solver import solve

_DEFAULT_Q = ("What morphological cell states are present in this image? "
              "Segment the cells, crop them, and cluster them into states.")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    image = sys.argv[1]
    question = sys.argv[2] if len(sys.argv) > 2 else _DEFAULT_Q

    if not os.getenv("LLM_API_KEY"):
        print("ERROR: set LLM_API_KEY (the planner/executor call an LLM).")
        return 2
    if not os.path.isfile(image):
        print(f"ERROR: image not found: {image}")
        return 2

    print(f"QUESTION: {question}\nIMAGE: {image}\n")
    out = solve(question=question, image_path=image, max_steps=10)

    print("=== STEP TRAIL ===")
    executed, rejected = [], []
    for a in out.get("steps", []):
        cmd = a.get("command", "") or ""
        is_reject = "[plan-rejected" in cmd
        tool = a.get("tool_name")
        marker = "   <-- PLAN GATE REJECTED" if is_reject else ""
        print(f"  step {a.get('step_count')}: {tool}{marker}")
        (rejected if is_reject else executed).append(tool)

    print("\nEXECUTED:", executed)
    print("PLAN-GATE REJECTIONS:", rejected)

    seg = any(t and "Segmenter" in t for t in executed)
    crop = any(t and "Cropper" in t for t in executed)
    ana = any(t and "Analyzer" in t for t in executed)
    print(f"\nPIPELINE: segmentation={seg}  cropping={crop}  analysis={ana}")
    print("\nFINAL OUTPUT:\n", (out.get("direct_output") or "")[:800])

    if seg and crop and ana:
        print("\nPASS: the LLM completed the chain without required_next forcing.")
        return 0
    print("\nCHECK: chain incomplete -- did the LLM stop early (the J.4=B tradeoff) "
          "or did the plan gate over-block? Inspect the step trail above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
