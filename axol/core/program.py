"""Axol execution engine — Program, Transition, run_program."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

import numpy as np

from axol.core.types import (
    _VecBase,
    FloatVec,
    GateVec,
    TransMatrix,
    StateBundle,
)
from axol.core import operations as ops
from axol.core.verify import VerifyResult, VerifySpec, verify_states


# ---------------------------------------------------------------------------
# Operation descriptors
# ---------------------------------------------------------------------------

class OpKind(Enum):
    TRANSFORM = auto()
    GATE = auto()
    MERGE = auto()
    DISTANCE = auto()
    ROUTE = auto()
    CUSTOM = auto()


@dataclass(frozen=True)
class TransformOp:
    kind: OpKind = field(default=OpKind.TRANSFORM, init=False)
    key: str
    matrix: TransMatrix
    out_key: str | None = None


@dataclass(frozen=True)
class GateOp:
    kind: OpKind = field(default=OpKind.GATE, init=False)
    key: str
    gate_key: str  # key of GateVec in the bundle
    out_key: str | None = None


@dataclass(frozen=True)
class MergeOp:
    kind: OpKind = field(default=OpKind.MERGE, init=False)
    keys: list[str]
    weights: FloatVec
    out_key: str


@dataclass(frozen=True)
class DistanceOp:
    kind: OpKind = field(default=OpKind.DISTANCE, init=False)
    key_a: str
    key_b: str
    metric: str = "euclidean"
    out_key: str = "_distance"


@dataclass(frozen=True)
class RouteOp:
    kind: OpKind = field(default=OpKind.ROUTE, init=False)
    key: str
    router: TransMatrix
    out_key: str = "_route"


@dataclass(frozen=True)
class CustomOp:
    """Escape hatch: user-supplied function operating on the full StateBundle."""
    kind: OpKind = field(default=OpKind.CUSTOM, init=False)
    fn: Callable[[StateBundle], StateBundle]
    label: str = "custom"


Operation = TransformOp | GateOp | MergeOp | DistanceOp | RouteOp | CustomOp


# ---------------------------------------------------------------------------
# Transition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Transition:
    name: str
    operation: Operation
    metadata: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Execution trace entry
# ---------------------------------------------------------------------------

@dataclass
class TraceEntry:
    step: int
    transition_name: str
    state_before: StateBundle
    state_after: StateBundle


# ---------------------------------------------------------------------------
# Program
# ---------------------------------------------------------------------------

@dataclass
class Program:
    name: str
    initial_state: StateBundle
    transitions: list[Transition]
    terminal_key: str | None = None  # key of a GateVec; all-1 = done
    max_iterations: int = 1000
    expected_state: StateBundle | None = None
    verify_specs: dict[str, VerifySpec] | None = None


# ---------------------------------------------------------------------------
# ExecutionResult
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    final_state: StateBundle
    steps_executed: int
    terminated_by: str  # "pipeline_end" | "terminal_condition" | "max_iterations"
    trace: list[TraceEntry]
    verification: VerifyResult | None = None


# ---------------------------------------------------------------------------
# Applying a single operation to the bundle
# ---------------------------------------------------------------------------

def _apply(op: Operation, state: StateBundle) -> StateBundle:
    s = state.copy()

    if isinstance(op, TransformOp):
        s[op.out_key or op.key] = ops.transform(state[op.key], op.matrix)

    elif isinstance(op, GateOp):
        g = state[op.gate_key]
        if not isinstance(g, GateVec):
            raise TypeError(f"Expected GateVec at key '{op.gate_key}', got {type(g).__name__}")
        s[op.out_key or op.key] = ops.gate(state[op.key], g)

    elif isinstance(op, MergeOp):
        vecs = [state[k] for k in op.keys]
        s[op.out_key] = ops.merge(vecs, op.weights)

    elif isinstance(op, DistanceOp):
        d = ops.distance(state[op.key_a], state[op.key_b], op.metric)
        s[op.out_key] = FloatVec.from_list([d])

    elif isinstance(op, RouteOp):
        idx = ops.route(state[op.key], op.router)
        from axol.core.types import IntVec
        s[op.out_key] = IntVec.from_list([idx])

    elif isinstance(op, CustomOp):
        s = op.fn(state)

    else:
        raise TypeError(f"Unknown operation type: {type(op)}")

    return s


# ---------------------------------------------------------------------------
# run_program
# ---------------------------------------------------------------------------

def run_program(program: Program) -> ExecutionResult:
    """Execute an Axol program.

    Pipeline mode (terminal_key is None):
        Run all transitions once in order.

    Loop mode (terminal_key is set):
        Repeat the full transition pipeline until the GateVec at terminal_key
        is all-ones, or max_iterations is reached.
    """
    state = program.initial_state.copy()
    trace: list[TraceEntry] = []
    step = 0

    if program.terminal_key is None:
        # Pipeline mode — one pass
        for t in program.transitions:
            before = state.copy()
            state = _apply(t.operation, state)
            step += 1
            trace.append(TraceEntry(step=step, transition_name=t.name, state_before=before, state_after=state.copy()))

        terminated_by = "pipeline_end"
    else:
        # Loop mode
        terminated_by = "max_iterations"
        for iteration in range(program.max_iterations):
            for t in program.transitions:
                before = state.copy()
                state = _apply(t.operation, state)
                step += 1
                trace.append(TraceEntry(step=step, transition_name=t.name, state_before=before, state_after=state.copy()))

            # Check terminal condition
            gate_vec = state[program.terminal_key]
            if isinstance(gate_vec, GateVec) and gate_vec.all_open:
                terminated_by = "terminal_condition"
                break

    # Verification
    verification: VerifyResult | None = None
    if program.expected_state is not None:
        verification = verify_states(
            program.expected_state,
            state,
            specs=program.verify_specs,
        )

    return ExecutionResult(
        final_state=state,
        steps_executed=step,
        terminated_by=terminated_by,
        trace=trace,
        verification=verification,
    )
