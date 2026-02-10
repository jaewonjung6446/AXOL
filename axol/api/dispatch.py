"""Axol API dispatcher — JSON request/response interface for AI agents.

Usage:
    from axol.api.dispatch import dispatch
    result = dispatch({"action": "run", "source": "@prog\\ns v=[1]\\n: t=transform(v;M=[2])"})
"""

from __future__ import annotations

import traceback

import numpy as np

from axol.core.dsl import parse, ParseError
from axol.core.program import run_program, Program
from axol.core.optimizer import optimize
from axol.core.types import FloatVec, StateBundle
from axol.core.verify import verify_states, VerifySpec
from axol.api.tools import get_tool_definitions as _get_tool_defs


_OPS_INFO = [
    {
        "name": "transform",
        "description": "Linear transformation: vec @ matrix",
        "params": ["key", "M (matrix)"],
    },
    {
        "name": "gate",
        "description": "Element-wise conditional masking: vec * gate",
        "params": ["key", "g (gate key)"],
    },
    {
        "name": "merge",
        "description": "Weighted sum of multiple vectors",
        "params": ["keys (space-separated)", "w (weights)"],
    },
    {
        "name": "distance",
        "description": "Similarity measurement between two vectors",
        "params": ["key_a key_b", "metric (euclidean|cosine|dot)"],
    },
    {
        "name": "route",
        "description": "Discrete branching via argmax: argmax(vec @ router)",
        "params": ["key", "R (router matrix)"],
    },
]


def dispatch(request: dict) -> dict:
    """Main entry point for the Axol tool-use API.

    Args:
        request: JSON-like dict with "action" and action-specific params.

    Returns:
        JSON-like dict with results or error information.
    """
    action = request.get("action")
    if not action:
        return {"error": "Missing 'action' field"}

    try:
        if action == "parse":
            return _handle_parse(request)
        elif action == "run":
            return _handle_run(request)
        elif action == "inspect":
            return _handle_inspect(request)
        elif action == "list_ops":
            return _handle_list_ops()
        elif action == "verify":
            return _handle_verify(request)
        else:
            return {"error": f"Unknown action: {action!r}"}
    except ParseError as e:
        return {"error": f"Parse error: {e}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def get_tool_definitions() -> list[dict]:
    """Return AI agent tool schema definitions."""
    return _get_tool_defs()


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------

def _handle_parse(request: dict) -> dict:
    source = request.get("source")
    if not source:
        return {"error": "Missing 'source' field"}

    prog = parse(source)
    return {
        "program_name": prog.name,
        "state_keys": list(prog.initial_state.keys()),
        "transition_count": len(prog.transitions),
        "has_terminal": prog.terminal_key is not None,
    }


def _handle_run(request: dict) -> dict:
    source = request.get("source")
    if not source:
        return {"error": "Missing 'source' field"}

    prog = parse(source)

    if request.get("max_iterations"):
        prog.max_iterations = int(request["max_iterations"])

    if request.get("optimize", False):
        prog = optimize(prog)

    result = run_program(prog)

    final_state = {}
    for key in result.final_state.keys():
        vec = result.final_state[key]
        final_state[key] = vec.to_list()

    return {
        "final_state": final_state,
        "steps_executed": result.steps_executed,
        "terminated_by": result.terminated_by,
    }


def _handle_inspect(request: dict) -> dict:
    source = request.get("source")
    if not source:
        return {"error": "Missing 'source' field"}

    step = request.get("step", 0)
    prog = parse(source)
    result = run_program(prog)

    if step == 0:
        state = {}
        for key in prog.initial_state.keys():
            state[key] = prog.initial_state[key].to_list()
        return {
            "step": 0,
            "transition_name": "(initial)",
            "state": state,
        }

    if step < 1 or step > len(result.trace):
        return {"error": f"Step {step} out of range (1..{len(result.trace)})"}

    entry = result.trace[step - 1]
    state = {}
    for key in entry.state_after.keys():
        state[key] = entry.state_after[key].to_list()

    return {
        "step": entry.step,
        "transition_name": entry.transition_name,
        "state": state,
    }


def _handle_list_ops() -> dict:
    return {"operations": _OPS_INFO}


def _handle_verify(request: dict) -> dict:
    source = request.get("source")
    if not source:
        return {"error": "Missing 'source' field"}

    expected_dict = request.get("expected")
    if not expected_dict:
        return {"error": "Missing 'expected' field"}

    tolerance = float(request.get("tolerance", 1e-5))

    prog = parse(source)
    result = run_program(prog)

    # Build expected StateBundle
    expected = StateBundle()
    for key, values in expected_dict.items():
        expected[key] = FloatVec.from_list([float(v) for v in values])

    spec = VerifySpec.exact(tolerance=tolerance)
    vr = verify_states(expected, result.final_state, default_spec=spec)

    return {
        "passed": vr.passed,
        "summary": vr.summary(),
    }
