---
tool: Generalist_Solution_Generator_Tool
category: utility
stage: none
depends_on: []
tunable: false
---
# Tool card: Generalist_Solution_Generator_Tool

**When to use:** a last-resort generalist LLM step that answers a free-form question (text,
optionally with an image) when NO specialized tool fits -- e.g. synthesizing retrieved
knowledge into a written answer. **When NOT:** for bioimage tasks always prefer the
specialized pipeline tools; this neither segments nor analyzes images. Input: a query (+
optional image). Output: a step-by-step textual answer.
