---
tool: Python_Code_Generator_Tool
category: utility
stage: none
depends_on: []
tunable: false
---
# Tool card: Python_Code_Generator_Tool

**When to use:** generate + run a small, self-contained Python snippet for basic arithmetic /
math computation. **When NOT:** not a sandbox and not for pipeline I/O or image processing --
use the bioimage tools for those. Input: a math/computation prompt. Output: the computed
result. Runs in a subprocess with a wall-clock timeout (not a security boundary).
