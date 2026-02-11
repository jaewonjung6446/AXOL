"""Axol API dispatcher — JSON request/response interface for AI agents.

Usage:
    from axol.api.dispatch import dispatch
    result = dispatch({"action": "run", "source": "@prog\\ns v=[1]\\n: t=transform(v;M=[2])"})
"""

from __future__ import annotations

import traceback

import numpy as np

import math

from axol.core.dsl import parse, ParseError
from axol.core.program import run_program, Program
from axol.core.optimizer import optimize
from axol.core.types import FloatVec, StateBundle
from axol.core.verify import verify_states, VerifySpec
from axol.core.encryption import random_orthogonal_key, encrypt_program, decrypt_state
from axol.core.operations import measure as _measure_op
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
        elif action == "encrypted_run":
            return _handle_encrypted_run(request)
        elif action == "quantum_search":
            return _handle_quantum_search(request)
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


def _handle_encrypted_run(request: dict) -> dict:
    """Run an Axol program with automatic encryption/decryption."""
    source = request.get("source")
    if not source:
        return {"error": "Missing 'source' field"}

    dim = request.get("dim")
    if not dim:
        return {"error": "Missing 'dim' field"}
    dim = int(dim)

    prog = parse(source)

    if request.get("optimize", True):
        prog = optimize(prog)

    K = random_orthogonal_key(dim, seed=request.get("seed"))
    prog_enc = encrypt_program(prog, K, dim)
    result = run_program(prog_enc)
    decrypted = decrypt_state(result.final_state, K)

    final_state = {}
    for key in decrypted.keys():
        vec = decrypted[key]
        final_state[key] = vec.to_list()

    return {
        "final_state": final_state,
        "steps_executed": result.steps_executed,
        "terminated_by": result.terminated_by,
        "encrypted": True,
    }


def _handle_quantum_search(request: dict) -> dict:
    """Run Grover's quantum search algorithm."""
    n = request.get("n")
    if not n:
        return {"error": "Missing 'n' field"}
    n = int(n)

    if n < 2 or (n & (n - 1)) != 0:
        return {"error": f"'n' must be a power of 2 >= 2, got {n}"}

    marked = request.get("marked")
    if marked is None:
        return {"error": "Missing 'marked' field"}
    marked = [int(x) for x in marked]

    encrypt = request.get("encrypt", False)

    # Build Grover DSL source with unrolled iterations
    iterations = max(1, math.floor(math.pi / 4 * math.sqrt(n)))
    amp = 1.0 / math.sqrt(n)
    state_vals = " ".join([str(amp)] * n)

    # Build transition lines: oracle + diffuse repeated
    lines = [f"@grover_{n}"]
    lines.append(f"s state=[{state_vals}]")
    for i in range(iterations):
        marked_str = ",".join(str(m) for m in marked)
        lines.append(f": oracle_{i}=oracle(state;marked=[{marked_str}];n={n})")
        lines.append(f": diffuse_{i}=diffuse(state;n={n})")

    source = "\n".join(lines)

    prog = parse(source)
    prog = optimize(prog)

    if encrypt:
        K = random_orthogonal_key(n, seed=request.get("seed"))
        prog_enc = encrypt_program(prog, K, n)
        result = run_program(prog_enc)
        decrypted = decrypt_state(result.final_state, K)
        state_data = decrypted["state"].data
    else:
        result = run_program(prog)
        state_data = result.final_state["state"].data

    # Apply Born rule measurement
    probs = state_data * state_data
    total = np.sum(probs)
    if total > 0:
        probs = probs / total
    found_index = int(np.argmax(probs))
    prob = float(probs[found_index])

    return {
        "found_index": found_index,
        "probability": prob,
        "iterations": iterations,
        "encrypted": encrypt,
    }
