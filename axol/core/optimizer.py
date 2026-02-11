"""Axol compiler optimizer — 3-pass optimization pipeline.

Pipeline: parse() -> optimize() -> run_program()

Passes:
  1. Transform fusion: consecutive TransformOp on same key -> single matrix multiply
  2. Dead state elimination: remove initial state keys never read by transitions
  3. Constant folding: pre-compute transforms on immutable keys
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from axol.core.types import FloatVec, TransMatrix, StateBundle
from axol.core.program import (
    Program,
    Transition,
    TransformOp,
    GateOp,
    MergeOp,
    DistanceOp,
    RouteOp,
    CustomOp,
    StepOp,
    BranchOp,
    ClampOp,
    MapOp,
    Operation,
)


def optimize(
    program: Program,
    *,
    fuse: bool = True,
    eliminate_dead: bool = True,
    fold_constants: bool = True,
) -> Program:
    """Optimize a Program without mutating the original.

    Args:
        program: Source program.
        fuse: Enable transform fusion pass.
        eliminate_dead: Enable dead state elimination pass.
        fold_constants: Enable constant folding pass.

    Returns:
        A new Program with optimizations applied.
    """
    p = _deep_copy_program(program)

    if fuse:
        p = _fuse_transforms(p)
    if eliminate_dead:
        p = _eliminate_dead_states(p)
    if fold_constants:
        p = _fold_constants(p)

    return p


# ---------------------------------------------------------------------------
# Deep copy helper
# ---------------------------------------------------------------------------

def _deep_copy_program(program: Program) -> Program:
    return Program(
        name=program.name,
        initial_state=program.initial_state.copy(),
        transitions=list(program.transitions),
        terminal_key=program.terminal_key,
        max_iterations=program.max_iterations,
        expected_state=program.expected_state,
        verify_specs=program.verify_specs,
    )


# ---------------------------------------------------------------------------
# Pass 1: Transform Fusion
# ---------------------------------------------------------------------------

def _fuse_transforms(program: Program) -> Program:
    """Fuse consecutive TransformOps on the same key chain into a single matrix.

    M1 followed by M2 on same key -> M_fused = M1 @ M2 (single transition).
    Does not cross CustomOp boundaries.
    Uses fixed-point iteration to handle 3+ chains.
    """
    changed = True
    transitions = list(program.transitions)

    while changed:
        changed = False
        new_transitions: list[Transition] = []
        i = 0

        while i < len(transitions):
            t = transitions[i]

            if (
                isinstance(t.operation, TransformOp)
                and i + 1 < len(transitions)
            ):
                t_next = transitions[i + 1]
                op_cur = t.operation
                op_next = t_next.operation if isinstance(t_next.operation, TransformOp) else None

                if (
                    op_next is not None
                    and _effective_out(op_cur) == op_next.key
                    and not isinstance(t_next.operation, CustomOp)
                ):
                    # Fuse: M_fused = M1 @ M2
                    fused_matrix = TransMatrix(
                        data=(op_cur.matrix.data @ op_next.matrix.data).astype(np.float32)
                    )
                    fused_op = TransformOp(
                        key=op_cur.key,
                        matrix=fused_matrix,
                        out_key=op_next.out_key,
                    )
                    fused_name = f"{t.name}+{t_next.name}"
                    new_transitions.append(Transition(name=fused_name, operation=fused_op))
                    i += 2
                    changed = True
                    continue

            new_transitions.append(t)
            i += 1

        transitions = new_transitions

    return Program(
        name=program.name,
        initial_state=program.initial_state.copy(),
        transitions=transitions,
        terminal_key=program.terminal_key,
        max_iterations=program.max_iterations,
        expected_state=program.expected_state,
        verify_specs=program.verify_specs,
    )


def _effective_out(op: TransformOp) -> str:
    """Return the effective output key of a TransformOp."""
    return op.out_key if op.out_key is not None else op.key


# ---------------------------------------------------------------------------
# Pass 2: Dead State Elimination
# ---------------------------------------------------------------------------

def _eliminate_dead_states(program: Program) -> Program:
    """Remove initial state vectors that are never read by any transition."""
    has_custom = any(isinstance(t.operation, CustomOp) for t in program.transitions)
    if has_custom:
        # Conservative: CustomOps may read anything
        return program

    read_keys = _collect_read_keys(program)

    # terminal_key is also "read"
    if program.terminal_key is not None:
        read_keys.add(program.terminal_key)

    new_vectors = {
        k: v for k, v in program.initial_state.items()
        if k in read_keys
    }

    return Program(
        name=program.name,
        initial_state=StateBundle(vectors=new_vectors),
        transitions=list(program.transitions),
        terminal_key=program.terminal_key,
        max_iterations=program.max_iterations,
        expected_state=program.expected_state,
        verify_specs=program.verify_specs,
    )


def _collect_read_keys(program: Program) -> set[str]:
    """Collect all keys read by transitions."""
    keys: set[str] = set()

    for t in program.transitions:
        op = t.operation
        if isinstance(op, TransformOp):
            keys.add(op.key)
        elif isinstance(op, GateOp):
            keys.add(op.key)
            keys.add(op.gate_key)
        elif isinstance(op, MergeOp):
            keys.update(op.keys)
        elif isinstance(op, DistanceOp):
            keys.add(op.key_a)
            keys.add(op.key_b)
        elif isinstance(op, RouteOp):
            keys.add(op.key)
        elif isinstance(op, StepOp):
            keys.add(op.key)
        elif isinstance(op, BranchOp):
            keys.add(op.gate_key)
            keys.add(op.then_key)
            keys.add(op.else_key)
        elif isinstance(op, ClampOp):
            keys.add(op.key)
        elif isinstance(op, MapOp):
            keys.add(op.key)

    return keys


# ---------------------------------------------------------------------------
# Pass 3: Constant Folding
# ---------------------------------------------------------------------------

def _fold_constants(program: Program) -> Program:
    """Pre-compute TransformOps on keys that are never written to.

    A key is immutable if no transition writes to it.
    If a TransformOp reads an immutable key, we can compute the result
    at compile time and store it in initial state.
    """
    has_custom = any(isinstance(t.operation, CustomOp) for t in program.transitions)
    if has_custom:
        return program

    write_keys = _collect_write_keys(program)
    new_state = program.initial_state.copy()
    new_transitions: list[Transition] = []

    for t in program.transitions:
        op = t.operation
        if isinstance(op, TransformOp) and op.key not in write_keys:
            # Key is immutable — fold the constant
            vec = new_state[op.key]
            result_data = (vec.data.astype(np.float32) @ op.matrix.data).astype(np.float32)
            result = FloatVec(data=result_data)
            out = op.out_key if op.out_key is not None else op.key
            new_state[out] = result
        else:
            new_transitions.append(t)

    return Program(
        name=program.name,
        initial_state=new_state,
        transitions=new_transitions,
        terminal_key=program.terminal_key,
        max_iterations=program.max_iterations,
        expected_state=program.expected_state,
        verify_specs=program.verify_specs,
    )


def _collect_write_keys(program: Program) -> set[str]:
    """Collect all keys that are written by transitions."""
    keys: set[str] = set()

    for t in program.transitions:
        op = t.operation
        if isinstance(op, TransformOp):
            keys.add(op.out_key if op.out_key is not None else op.key)
        elif isinstance(op, GateOp):
            keys.add(op.out_key if op.out_key is not None else op.key)
        elif isinstance(op, MergeOp):
            keys.add(op.out_key)
        elif isinstance(op, DistanceOp):
            keys.add(op.out_key)
        elif isinstance(op, RouteOp):
            keys.add(op.out_key)
        elif isinstance(op, StepOp):
            keys.add(op.out_key if op.out_key is not None else op.key)
        elif isinstance(op, BranchOp):
            keys.add(op.out_key)
        elif isinstance(op, ClampOp):
            keys.add(op.out_key if op.out_key is not None else op.key)
        elif isinstance(op, MapOp):
            keys.add(op.out_key if op.out_key is not None else op.key)

    return keys
