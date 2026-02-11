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
    {
        "name": "encrypted_run",
        "description": "Run an Axol program with automatic encryption. "
                       "The program runs on encrypted data and returns decrypted results. "
                       "You don't need to handle encryption — it's automatic.",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Axol DSL source code",
                },
                "dim": {
                    "type": "integer",
                    "description": "Vector dimension to encrypt",
                },
                "optimize": {
                    "type": "boolean",
                    "description": "Whether to optimize before execution",
                    "default": True,
                },
            },
            "required": ["source", "dim"],
        },
    },
    {
        "name": "quantum_search",
        "description": "Search for a marked item using Grover's quantum algorithm. "
                       "Finds the target with high probability using sqrt(N) iterations instead of N.",
        "parameters": {
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": "Search space size (must be power of 2)",
                },
                "marked": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Target indices to search for",
                },
                "encrypt": {
                    "type": "boolean",
                    "default": False,
                    "description": "Run on encrypted data",
                },
            },
            "required": ["n", "marked"],
        },
    },
    {
        "name": "padded_run",
        "description": "Run an Axol program with padding + encryption. "
                       "All vectors are padded to a uniform dimension before encryption, "
                       "hiding both data and dimension structure from the server.",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Axol DSL source code",
                },
                "max_dim": {
                    "type": "integer",
                    "description": "Uniform dimension for padding (must be >= largest vector)",
                },
                "seed": {
                    "type": "integer",
                    "description": "Key derivation seed",
                    "default": 42,
                },
                "optimize": {
                    "type": "boolean",
                    "description": "Whether to optimize before execution",
                    "default": True,
                },
            },
            "required": ["source", "max_dim"],
        },
    },
    {
        "name": "run_encrypted_payload",
        "description": "Run a pre-encrypted program payload from AxolClient. "
                       "The server executes the encrypted program without access to plaintext.",
        "parameters": {
            "type": "object",
            "properties": {
                "program": {
                    "type": "object",
                    "description": "Serialized encrypted program (from AxolClient.prepare)",
                },
            },
            "required": ["program"],
        },
    },
]


def get_tool_definitions() -> list[dict]:
    """Return tool definitions for AI agent integration."""
    return TOOL_DEFINITIONS
