"""
SHAPE: Self-supervised morpHology Agent for cellular Phenotype

An agentic framework for morphological reasoning in biological image analysis.
"""

__version__ = "0.1.0"

# Import main solver functions
from shape.solver import solve, construct_solver, get_available_tools

__all__ = [
    "solve",
    "construct_solver",
    "get_available_tools",
]

