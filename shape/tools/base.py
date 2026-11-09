# shape/tools/base.py

import inspect
import os
from functools import lru_cache

from shape.engine.openai import ChatOpenAI


@lru_cache(maxsize=None)
def _read_card_file(tool_name: str) -> str | None:
    """Raw text of a tool's CARD.md by directory-name convention, or None."""
    if not tool_name:
        return None
    path = os.path.join(os.path.dirname(__file__), tool_name, "CARD.md")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def _parse_frontmatter(text: str):
    """Split a CARD.md into (header: dict, body: str).

    Minimal YAML front-matter (`key: value`, `[a, b]` lists, true/false) so the CARD can be
    the SINGLE SOURCE OF TRUTH for a tool -- category/stage/deps in the header, the prose
    (purpose, when-to-use/not, tuning) in the body -- with no external yaml dependency.
    """
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    header: dict = {}
    for line in parts[1].splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key, val = key.strip(), val.split("#", 1)[0].strip()
        if val.startswith("[") and val.endswith("]"):
            val = [v.strip() for v in val[1:-1].split(",") if v.strip()]
        elif val.lower() in ("true", "false"):
            val = val.lower() == "true"
        header[key] = val
    return header, parts[2].lstrip("\n")


def load_card(tool_name: str):
    """Return (header: dict, body: str) for a tool's CARD.md, or ({}, None) if absent.

    A CARD is a tool's complete declarative self-description: a machine-readable header
    (category / stage / depends_on / tunable) the core reads to organize and route tools,
    plus a prose body (purpose, WHEN TO USE vs NOT, inputs/outputs, and -- for tunable tools
    -- the agent-tuning parameter manual the verifier injects).
    """
    text = _read_card_file(tool_name)
    if text is None:
        return {}, None
    return _parse_frontmatter(text)


def load_tuning_card(tool_name: str) -> str | None:
    """The CARD BODY (front-matter stripped) -- what the verifier injects into the vision
    prompt. None when the tool ships no card."""
    return load_card(tool_name)[1]


@lru_cache(maxsize=1)
def card_index() -> dict:
    """{tool_name: (dir, header, body)} for every tool that ships a CARD.md.

    The single source of truth the CORE reads to organize/route tools (category, stage,
    when-to-use) -- so tool selection reasons over the cards instead of a hard-coded table.
    """
    base = os.path.dirname(__file__)
    out: dict = {}
    for d in sorted(os.listdir(base)):
        if not os.path.isfile(os.path.join(base, d, "tool.py")):
            continue
        header, body = load_card(d)
        name = header.get("tool")
        if name:
            out[name] = (d, header, body)
    return out


def card_summary(tool_name: str) -> str:
    """The tool's WHEN-TO-USE paragraph from its CARD body (whitespace-collapsed), or ''.

    This is the reasoning material surfaced to the planner so the LLM picks the right tool.
    """
    entry = card_index().get(tool_name)
    if not entry:
        return ""
    body = entry[2] or ""
    paras = [p.strip() for p in body.split("\n\n")]
    for p in paras:
        if "When to use" in p:
            return " ".join(p.split())
    for p in paras:                       # fallback: first prose paragraph
        if p and not p.startswith("#") and not p.startswith("---"):
            return " ".join(p.split())
    return ""


class BaseTool:
    """
    A base class for building tool classes that perform specific tasks, such as image processing or text detection.
    """

    require_llm_engine = False  # Default is False, tools that need LLM should set this to True

    def __init__(self, tool_name=None, tool_description=None, tool_version=None, input_types=None, output_type=None, demo_commands=None, output_dir=None, user_metadata=None, model_string=None):
        """
        Initialize the base tool with optional metadata.

        Parameters:
            tool_name (str): The name of the tool.
            tool_description (str): A description of the tool.
            tool_version (str): The version of the tool.
            input_types (dict): The expected input types for the tool.
            output_type (str): The expected output type for the tool.
            demo_commands (list): A list of example commands for using the tool.
            output_dir (str): The directory where the tool should save its output (optional).
            user_metadata (dict): Additional metadata specific to user needs (optional).
            model_string (str): The model string for the LLM engine (optional, only used if require_llm_engine is True).
        """
        self.tool_name = tool_name
        self.tool_description = tool_description
        self.tool_version = tool_version
        self.input_types = input_types
        self.output_type = output_type
        self.demo_commands = demo_commands
        self.output_dir = output_dir
        self.user_metadata = user_metadata
        self.model_string = model_string

    def set_metadata(self, tool_name, tool_description, tool_version, input_types, output_type, demo_commands, user_metadata=None):
        """
        Set the metadata for the tool.

        Parameters:
            tool_name (str): The name of the tool.
            tool_description (str): A description of the tool.
            tool_version (str): The version of the tool.
            input_types (dict): The expected input types for the tool.
            output_type (str): The expected output type for the tool.
            demo_commands (list): A list of example commands for using the tool.
            user_metadata (dict): Additional metadata specific to user needs (optional).
        """
        self.tool_name = tool_name
        self.tool_description = tool_description
        self.tool_version = tool_version
        self.input_types = input_types
        self.output_type = output_type
        self.demo_commands = demo_commands
        self.user_metadata = user_metadata

    def get_metadata(self):
        """
        Returns the metadata for the tool.

        Returns:
            dict: A dictionary containing the tool's metadata.
        """
        metadata = {
            "tool_name": self.tool_name,
            "tool_description": self.tool_description,
            "tool_version": self.tool_version,
            "input_types": self.input_types,
            "output_type": self.output_type,
            "demo_commands": self.demo_commands,
            "require_llm_engine": self.require_llm_engine,
        }
        if self.user_metadata:
            metadata["user_metadata"] = self.user_metadata
        return metadata

    def get_tuning_card(self):
        """Return this tool's agent-tuning CARD.md (co-located with its module), or None.

        The card is the single source of truth for how the tool is tuned; the VLRR
        recommend phase reads it to build the per-tool LLM prompt. Resolved from the
        subclass's module directory so any tool exposes its card uniformly.
        """
        try:
            card_dir = os.path.dirname(inspect.getfile(type(self)))
            path = os.path.join(card_dir, "CARD.md")
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    return _parse_frontmatter(f.read())[1]
        except Exception:
            pass
        return None

    def set_custom_output_dir(self, output_dir):
        """
        Set a custom output directory for the tool.

        Parameters:
            output_dir (str): The new output directory path.
        """
        self.output_dir = output_dir

    def set_llm_engine(self, model_string):
        """
        Set the LLM engine for the tool.

        Parameters:
            model_string (str): The model string for the LLM engine.
        """
        self.model_string = model_string

    def execute(self, *args, **kwargs):
        """
        Execute the tool's main functionality. This method should be overridden by subclasses.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement the execute method.")


