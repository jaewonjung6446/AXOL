"""Self-verification — compare expected vs actual StateBundles."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from axol.core.types import StateBundle


class MatchMode(Enum):
    EXACT = "exact"
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"


@dataclass
class VerifySpec:
    """Per-key matching specification."""

    mode: MatchMode = MatchMode.EXACT
    tolerance: float = 1e-5

    @classmethod
    def exact(cls, tolerance: float = 1e-5) -> VerifySpec:
        return cls(mode=MatchMode.EXACT, tolerance=tolerance)

    @classmethod
    def cosine(cls, tolerance: float = 0.01) -> VerifySpec:
        return cls(mode=MatchMode.COSINE, tolerance=tolerance)

    @classmethod
    def euclidean(cls, tolerance: float = 0.1) -> VerifySpec:
        return cls(mode=MatchMode.EUCLIDEAN, tolerance=tolerance)


@dataclass
class VectorResult:
    """Result for a single vector comparison."""

    key: str
    passed: bool
    mode: MatchMode
    value: float  # actual metric value (0.0 for exact match, distance otherwise)
    tolerance: float
    message: str = ""


@dataclass
class VerifyResult:
    """Aggregate verification result."""

    passed: bool = True
    vector_results: list[VectorResult] = field(default_factory=list)
    missing_keys: list[str] = field(default_factory=list)
    extra_keys: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines: list[str] = []
        status = "PASS" if self.passed else "FAIL"
        lines.append(f"Verification: {status}")

        for vr in self.vector_results:
            mark = "OK" if vr.passed else "FAIL"
            lines.append(f"  [{mark}] {vr.key}: {vr.mode.value} = {vr.value:.6f} (tol={vr.tolerance})")
            if vr.message:
                lines.append(f"        {vr.message}")

        if self.missing_keys:
            lines.append(f"  Missing keys: {self.missing_keys}")
        if self.extra_keys:
            lines.append(f"  Extra keys: {self.extra_keys}")

        return "\n".join(lines)


def _compare_vectors(
    key: str,
    expected: np.ndarray,
    actual: np.ndarray,
    spec: VerifySpec,
) -> VectorResult:
    e = expected.astype(np.float32).flatten()
    a = actual.astype(np.float32).flatten()

    if e.shape != a.shape:
        return VectorResult(
            key=key,
            passed=False,
            mode=spec.mode,
            value=float("inf"),
            tolerance=spec.tolerance,
            message=f"Shape mismatch: expected {e.shape}, got {a.shape}",
        )

    if spec.mode == MatchMode.EXACT:
        diff = float(np.max(np.abs(e - a)))
        ok = diff <= spec.tolerance
        return VectorResult(key=key, passed=ok, mode=spec.mode, value=diff, tolerance=spec.tolerance)

    elif spec.mode == MatchMode.EUCLIDEAN:
        dist = float(np.linalg.norm(e - a))
        ok = dist <= spec.tolerance
        return VectorResult(key=key, passed=ok, mode=spec.mode, value=dist, tolerance=spec.tolerance)

    elif spec.mode == MatchMode.COSINE:
        ne = float(np.linalg.norm(e))
        na = float(np.linalg.norm(a))
        if ne == 0.0 or na == 0.0:
            cosine_dist = 0.0 if (ne == 0.0 and na == 0.0) else 1.0
        else:
            cosine_dist = 1.0 - float(np.dot(e, a)) / (ne * na)
        ok = cosine_dist <= spec.tolerance
        return VectorResult(key=key, passed=ok, mode=spec.mode, value=cosine_dist, tolerance=spec.tolerance)

    raise ValueError(f"Unknown match mode: {spec.mode}")


def verify_states(
    expected: StateBundle,
    actual: StateBundle,
    specs: dict[str, VerifySpec] | None = None,
    default_spec: VerifySpec | None = None,
    strict_keys: bool = False,
) -> VerifyResult:
    """Compare two StateBundles vector-by-vector.

    Args:
        expected: The reference state.
        actual: The state produced by execution.
        specs: Per-key VerifySpec overrides.
        default_spec: Fallback spec for keys not in *specs*.
        strict_keys: If True, extra/missing keys cause failure.
    """
    if specs is None:
        specs = {}
    if default_spec is None:
        default_spec = VerifySpec.exact()

    result = VerifyResult()

    expected_keys = set(expected.keys())
    actual_keys = set(actual.keys())

    result.missing_keys = sorted(expected_keys - actual_keys)
    result.extra_keys = sorted(actual_keys - expected_keys)

    if result.missing_keys:
        result.passed = False
    if strict_keys and result.extra_keys:
        result.passed = False

    for key in sorted(expected_keys & actual_keys):
        spec = specs.get(key, default_spec)
        vr = _compare_vectors(key, expected[key].data, actual[key].data, spec)
        result.vector_results.append(vr)
        if not vr.passed:
            result.passed = False

    return result
