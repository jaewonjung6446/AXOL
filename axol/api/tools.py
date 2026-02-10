"""Tool definitions for the Axol API — JSON Schema descriptions for AI agents."""

from __future__ import annotations

TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "parse",
        "description": "Parse Axol DSL source code and return program metadata.",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Axol DSL source code",
                },
            },
            "required": ["source"],
        },
    },
    {
        "name": "run",
        "description": "Parse and execute an Axol DSL program, returning final state and execution info.",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Axol DSL source code",
                },
                "optimize": {
                    "type": "boolean",
                    "description": "Whether to optimize the program before execution",
                    "default": False,
                },
                "max_iterations": {
                    "type": "integer",
                    "description": "Maximum loop iterations (default: 1000)",
                    "default": 1000,
                },
            },
            "required": ["source"],
        },
    },
    {
        "name": "inspect",
        "description": "Inspect program state at a specific execution step.",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Axol DSL source code",
                },
                "step": {
                    "type": "integer",
                    "description": "Step number to inspect (0 = initial state)",
                    "default": 0,
                },
            },
            "required": ["source"],
        },
    },
    {
        "name": "list_ops",
        "description": "List all available Axol primitive operations with descriptions.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "verify",
        "description": "Run a program and verify the final state against expected values.",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Axol DSL source code",
                },
                "expected": {
                    "type": "object",
                    "description": "Expected final state as {key: [values...]}",
                },
                "tolerance": {
                    "type": "number",
                    "description": "Tolerance for floating-point comparison",
                    "default": 1e-5,
                },
            },
            "required": ["source", "expected"],
        },
    },
]


def get_tool_definitions() -> list[dict]:
    """Return tool definitions for AI agent integration."""
    return TOOL_DEFINITIONS
