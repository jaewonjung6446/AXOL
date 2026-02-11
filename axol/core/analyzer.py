"""Axol encryption coverage analyzer.

Analyzes a Program to determine which transitions can run encrypted (E)
vs. which require plaintext (P), and computes encryption coverage metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    MeasureOp,
    SecurityLevel,
    OpKind,
)


@dataclass
class TransitionInfo:
    name: str
    op_kind: OpKind
    security: SecurityLevel
    read_keys: set[str]
    write_keys: set[str]


@dataclass
class AnalysisResult:
    program_name: str
    total_transitions: int
    encrypted_count: int
    plaintext_count: int
    coverage_pct: float  # encrypted / total * 100
    transitions: list[TransitionInfo]
    encryptable_keys: set[str]  # keys accessed only by E operations
    plaintext_keys: set[str]    # keys accessed by at least one P operation

    def summary(self) -> str:
        lines = [
            f"Program: {self.program_name}",
            f"Transitions: {self.total_transitions} total, "
            f"{self.encrypted_count} encrypted (E), "
            f"{self.plaintext_count} plaintext (P)",
            f"Coverage: {self.coverage_pct:.1f}%",
            f"Encryptable keys: {sorted(self.encryptable_keys) if self.encryptable_keys else '(none)'}",
            f"Plaintext keys: {sorted(self.plaintext_keys) if self.plaintext_keys else '(none)'}",
        ]
        return "\n".join(lines)


def _get_read_keys(op) -> set[str]:
    if isinstance(op, TransformOp):
        return {op.key}
    elif isinstance(op, GateOp):
        return {op.key, op.gate_key}
    elif isinstance(op, MergeOp):
        return set(op.keys)
    elif isinstance(op, DistanceOp):
        return {op.key_a, op.key_b}
    elif isinstance(op, RouteOp):
        return {op.key}
    elif isinstance(op, StepOp):
        return {op.key}
    elif isinstance(op, BranchOp):
        return {op.gate_key, op.then_key, op.else_key}
    elif isinstance(op, ClampOp):
        return {op.key}
    elif isinstance(op, MapOp):
        return {op.key}
    elif isinstance(op, MeasureOp):
        return {op.key}
    else:
        return set()


def _get_write_keys(op) -> set[str]:
    if isinstance(op, TransformOp):
        return {op.out_key if op.out_key is not None else op.key}
    elif isinstance(op, GateOp):
        return {op.out_key if op.out_key is not None else op.key}
    elif isinstance(op, MergeOp):
        return {op.out_key}
    elif isinstance(op, DistanceOp):
        return {op.out_key}
    elif isinstance(op, RouteOp):
        return {op.out_key}
    elif isinstance(op, StepOp):
        return {op.out_key if op.out_key is not None else op.key}
    elif isinstance(op, BranchOp):
        return {op.out_key}
    elif isinstance(op, ClampOp):
        return {op.out_key if op.out_key is not None else op.key}
    elif isinstance(op, MapOp):
        return {op.out_key if op.out_key is not None else op.key}
    elif isinstance(op, MeasureOp):
        return {op.out_key if op.out_key is not None else op.key}
    else:
        return set()


def analyze(program: Program) -> AnalysisResult:
    """Analyze a Program for encryption coverage."""
    infos: list[TransitionInfo] = []
    plaintext_keys: set[str] = set()
    all_keys: set[str] = set()

    for t in program.transitions:
        op = t.operation
        security = getattr(op, "security", SecurityLevel.PLAINTEXT)
        read_k = _get_read_keys(op)
        write_k = _get_write_keys(op)

        infos.append(TransitionInfo(
            name=t.name,
            op_kind=op.kind,
            security=security,
            read_keys=read_k,
            write_keys=write_k,
        ))

        all_keys |= read_k | write_k
        if security == SecurityLevel.PLAINTEXT:
            plaintext_keys |= read_k | write_k

    total = len(infos)
    encrypted_count = sum(1 for i in infos if i.security == SecurityLevel.ENCRYPTED)
    plaintext_count = total - encrypted_count
    coverage = (encrypted_count / total * 100.0) if total > 0 else 0.0
    encryptable_keys = all_keys - plaintext_keys

    return AnalysisResult(
        program_name=program.name,
        total_transitions=total,
        encrypted_count=encrypted_count,
        plaintext_count=plaintext_count,
        coverage_pct=coverage,
        transitions=infos,
        encryptable_keys=encryptable_keys,
        plaintext_keys=plaintext_keys,
    )
