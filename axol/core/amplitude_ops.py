"""Amplitude operations — distribution-to-distribution transforms.

All operations preserve the amplitude representation (no collapse).
Collapse happens only at amp_observe().
"""

from __future__ import annotations

import numpy as np

from axol.core.types import FloatVec, TransMatrix
from axol.core.amplitude import Amplitude


def amp_transform(amp: Amplitude, matrix: TransMatrix) -> Amplitude:
    """Linear transformation: amp @ matrix → Amplitude (phase-preserving).

    The core building block: applies a real matrix to complex amplitudes.
    Phase information flows through the transform.
    """
    v = amp.data.astype(np.complex128)
    m = matrix.data.astype(np.complex128)
    if v.shape[0] != m.shape[0]:
        raise ValueError(
            f"Dimension mismatch: amp({v.shape[0]}) vs matrix({m.shape[0]}x{m.shape[1]})"
        )
    result = v @ m
    return Amplitude(data=result)


def amp_superpose(
    amps: list[Amplitude],
    weights: list[complex],
) -> Amplitude:
    """Superposition of multiple amplitudes with complex weights.

    THIS IS THE INTERFERENCE CORE: when paths have different phases,
    their amplitudes can constructively or destructively interfere.

    result = sum(w_i * amp_i), then normalize.
    """
    if len(amps) != len(weights):
        raise ValueError(f"Count mismatch: {len(amps)} amps vs {len(weights)} weights")
    if len(amps) == 0:
        raise ValueError("amp_superpose requires at least one amplitude")

    n = amps[0].size
    result = np.zeros(n, dtype=np.complex128)
    for amp, w in zip(amps, weights):
        if amp.size != n:
            raise ValueError("All amplitudes in superposition must have same dimension")
        result += complex(w) * amp.data.astype(np.complex128)

    return Amplitude(data=result)


def amp_interfere(
    amp1: Amplitude,
    amp2: Amplitude,
    phase: float = 0.0,
) -> Amplitude:
    """Explicit two-path interference.

    result = amp1 + exp(i*phase) * amp2, normalized.

    phase=0:   constructive (amplify matching components)
    phase=pi:  destructive (cancel matching components)
    """
    if amp1.size != amp2.size:
        raise ValueError(
            f"Dimension mismatch: amp1({amp1.size}) vs amp2({amp2.size})"
        )
    result = amp1.data.astype(np.complex128) + np.exp(1j * phase) * amp2.data.astype(np.complex128)
    return Amplitude(data=result)


def amp_condition(
    amp: Amplitude,
    predicate: np.ndarray,
) -> Amplitude:
    """Conditional update without collapse.

    Zeroes out amplitudes where predicate is False, then renormalizes.
    This is a projective measurement without reading the outcome.

    predicate: boolean mask of same length as amp.
    """
    if predicate.shape[0] != amp.size:
        raise ValueError(
            f"Dimension mismatch: amp({amp.size}) vs predicate({predicate.shape[0]})"
        )
    mask = predicate.astype(np.float64)
    result = amp.data.astype(np.complex128) * mask
    return Amplitude(data=result)


def amp_entangle(amp_a: Amplitude, amp_b: Amplitude) -> Amplitude:
    """Tensor product of two amplitudes → joint distribution.

    Result dimension = dim_a * dim_b.
    Represents the combined state of two variables.
    """
    result = np.outer(
        amp_a.data.astype(np.complex128),
        amp_b.data.astype(np.complex128),
    ).flatten()
    return Amplitude(data=result)


def amp_observe(amp: Amplitude) -> tuple[int, FloatVec]:
    """THE ONLY COLLAPSE POINT: amplitude → (chosen_index, probability_distribution).

    Returns both the collapsed index (argmax) and the full Born-rule
    probability distribution for downstream use.
    """
    probs = amp.probabilities
    chosen = int(np.argmax(probs))
    return chosen, FloatVec(data=probs.astype(np.float32))
