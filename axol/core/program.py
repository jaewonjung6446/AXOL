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
    STEP = auto()
    BRANCH = auto()
    CLAMP = auto()
    MAP = auto()
    CUSTOM = auto()


class SecurityLevel(Enum):
    ENCRYPTED = "E"   # can run on encrypted data
    PLAINTEXT = "P"   # requires plaintext


@dataclass(frozen=True)
class TransformOp:
    kind: OpKind = field(default=OpKind.TRANSFORM, init=False)
    security: SecurityLevel = field(default=SecurityLevel.ENCRYPTED, init=False)
    key: str
    matrix: TransMatrix
    out_key: str | None = None


@dataclass(frozen=True)
class GateOp:
    kind: OpKind = field(default=OpKind.GATE, init=False)
    security: SecurityLevel = field(default=SecurityLevel.ENCRYPTED, init=False)
    key: str
    gate_key: str  # key of GateVec in the bundle
    out_key: str | None = None


@dataclass(frozen=True)
class MergeOp:
    kind: OpKind = field(default=OpKind.MERGE, init=False)
    security: SecurityLevel = field(default=SecurityLevel.ENCRYPTED, init=False)
    keys: list[str]
    weights: FloatVec
    out_key: str


@dataclass(frozen=True)
class DistanceOp:
    kind: OpKind = field(default=OpKind.DISTANCE, init=False)
    security: SecurityLevel = field(default=SecurityLevel.ENCRYPTED, init=False)
    key_a: str
    key_b: str
    metric: str = "euclidean"
    out_key: str = "_distance"


@dataclass(frozen=True)
class RouteOp:
    kind: OpKind = field(default=OpKind.ROUTE, init=False)
    security: SecurityLevel = field(default=SecurityLevel.ENCRYPTED, init=False)
    key: str
    router: TransMatrix
    out_key: str = "_route"


@dataclass(frozen=True)
class CustomOp:
    """Escape hatch: user-supplied function operating on the full StateBundle."""
    kind: OpKind = field(default=OpKind.CUSTOM, init=False)
    security: SecurityLevel = field(default=SecurityLevel.PLAINTEXT, init=False)
    fn: Callable[[StateBundle], StateBundle]
    label: str = "custom"


@dataclass(frozen=True)
class StepOp:
    """Threshold → binary gate vector (Plaintext only)."""
    kind: OpKind = field(default=OpKind.STEP, init=False)
    security: SecurityLevel = field(default=SecurityLevel.PLAINTEXT, init=False)
    key: str
    threshold: float = 0.0
    out_key: str | None = None


@dataclass(frozen=True)
class BranchOp:
    """Conditional select via gate vector (Plaintext only)."""
    kind: OpKind = field(default=OpKind.BRANCH, init=False)
    security: SecurityLevel = field(default=SecurityLevel.PLAINTEXT, init=False)
    gate_key: str
    then_key: str
    else_key: str
    out_key: str


@dataclass(frozen=True)
class ClampOp:
    """Clip values to [min, max] (Plaintext only)."""
    kind: OpKind = field(default=OpKind.CLAMP, init=False)
    security: SecurityLevel = field(default=SecurityLevel.PLAINTEXT, init=False)
    key: str
    min_val: float = float("-inf")
    max_val: float = float("inf")
    out_key: str | None = None


@dataclass(frozen=True)
class MapOp:
    """Apply named element-wise function (Plaintext only)."""
    kind: OpKind = field(default=OpKind.MAP, init=False)
    security: SecurityLevel = field(default=SecurityLevel.PLAINTEXT, init=False)
    key: str
    fn_name: str
    out_key: str | None = None


Operation = (
    TransformOp | GateOp | MergeOp | DistanceOp | RouteOp
    | StepOp | BranchOp | ClampOp | MapOp
    | CustomOp
)


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

    elif isinstance(op, StepOp):
        s[op.out_key or op.key] = ops.step(state[op.key], op.threshold)

    elif isinstance(op, BranchOp):
        g = state[op.gate_key]
        if not isinstance(g, GateVec):
            raise TypeError(f"Expected GateVec at key '{op.gate_key}', got {type(g).__name__}")
        s[op.out_key] = ops.branch(g, state[op.then_key], state[op.else_key])

    elif isinstance(op, ClampOp):
        s[op.out_key or op.key] = ops.clamp(state[op.key], op.min_val, op.max_val)

    elif isinstance(op, MapOp):
        s[op.out_key or op.key] = ops.map_fn(state[op.key], op.fn_name)

    elif isinstance(op, CustomOp):
        s = op.fn(state)

    else:
        # Check for UseOp from the module system
        from axol.core.module import UseOp, _apply_use_op
        if isinstance(op, UseOp):
            s = _apply_use_op(op, state)
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
