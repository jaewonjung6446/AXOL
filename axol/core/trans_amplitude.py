"""TransAmplitude — superposition of transformations (open relations).

A TransAmplitude represents multiple transformation matrices in complex
superposition — the relation itself is an open variable that defers
collapse until observation.  Different transformation paths can interfere
via their complex weights, enabling constructive/destructive interference
at the relation level.

Theoretical basis:
  - TransMatrix (closed): "this IS the transformation"
  - TransAmplitude (open): "this MIGHT be the transformation, or that one"
  - Collapse happens only at explicit observation (trans_amp_collapse)
  - Effective matrix (trans_amp_effective) includes interference
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from axol.core.types import TransMatrix
from axol.core.amplitude import Amplitude


@dataclass(frozen=True)
class TransAmplitude:
    """Superposition of transformation matrices — an open relation.

    matrices: (K, M, N) float64 — K possible transformation matrices
    weights:  (K,) complex128 — complex amplitude per path (auto-normalized)
    """

    matrices: np.ndarray   # (K, M, N) float64
    weights: np.ndarray    # (K,) complex128, ||w|| = 1

    def __post_init__(self) -> None:
        # Ensure correct dtypes
        if self.matrices.dtype != np.float64:
            object.__setattr__(self, "matrices", self.matrices.astype(np.float64))
        if not np.issubdtype(self.weights.dtype, np.complexfloating):
            object.__setattr__(self, "weights", self.weights.astype(np.complex128))
        # Validate shapes
        if self.matrices.ndim != 3:
            raise ValueError(
                f"matrices must be 3-dimensional (K, M, N), got ndim={self.matrices.ndim}"
            )
        if self.weights.ndim != 1:
            raise ValueError(
                f"weights must be 1-dimensional (K,), got ndim={self.weights.ndim}"
            )
        if self.matrices.shape[0] != self.weights.shape[0]:
            raise ValueError(
                f"matrices has {self.matrices.shape[0]} paths but "
                f"weights has {self.weights.shape[0]} entries"
            )
        # Auto-normalize weights to unit norm
        norm = np.linalg.norm(self.weights)
        if norm > 0 and not np.isclose(norm, 1.0, atol=1e-12):
            object.__setattr__(
                self, "weights", (self.weights / norm).astype(np.complex128)
            )

    # --- Properties ---

    @property
    def n_paths(self) -> int:
        """Number of transformation paths (K)."""
        return int(self.matrices.shape[0])

    @property
    def input_dim(self) -> int:
        """Input dimension (M — rows of each matrix)."""
        return int(self.matrices.shape[1])

    @property
    def output_dim(self) -> int:
        """Output dimension (N — columns of each matrix)."""
        return int(self.matrices.shape[2])

    @property
    def negativity(self) -> float:
        """Sum of negative real parts of weights — non-classicality measure."""
        reals = self.weights.real
        return float(-np.sum(reals[reals < 0]))

    @property
    def is_classical(self) -> bool:
        """True if all weights are real and non-negative (no interference)."""
        return bool(
            np.allclose(self.weights.imag, 0, atol=1e-12)
            and np.all(self.weights.real >= -1e-12)
        )

    @property
    def path_probabilities(self) -> np.ndarray:
        """|w_k|^2 — probability of each transformation path."""
        return np.abs(self.weights) ** 2

    # --- Constructors ---

    @classmethod
    def from_matrix(cls, m: TransMatrix) -> TransAmplitude:
        """Wrap a single TransMatrix as a classical (1-path) TransAmplitude."""
        matrices = m.data.astype(np.float64)[np.newaxis, :, :]  # (1, M, N)
        weights = np.array([1.0 + 0j], dtype=np.complex128)
        return cls(matrices=matrices, weights=weights)

    @classmethod
    def from_matrices(
        cls,
        ms: list[TransMatrix],
        weights: list[complex] | np.ndarray,
    ) -> TransAmplitude:
        """Create from multiple TransMatrix instances with complex weights."""
        if len(ms) == 0:
            raise ValueError("from_matrices requires at least one matrix")
        stacked = np.stack([m.data.astype(np.float64) for m in ms], axis=0)
        w = np.array(weights, dtype=np.complex128)
        return cls(matrices=stacked, weights=w)

    @classmethod
    def uniform(cls, ms: list[TransMatrix]) -> TransAmplitude:
        """Uniform superposition — equal weight for all paths."""
        if len(ms) == 0:
            raise ValueError("uniform requires at least one matrix")
        k = len(ms)
        weights = np.ones(k, dtype=np.complex128) / np.sqrt(k)
        return cls.from_matrices(ms, weights)

    @classmethod
    def from_kraus(cls, operators: list[np.ndarray]) -> TransAmplitude:
        """Create from Kraus operators (each operator is a matrix)."""
        if len(operators) == 0:
            raise ValueError("from_kraus requires at least one operator")
        stacked = np.stack(
            [op.astype(np.float64) for op in operators], axis=0
        )
        k = len(operators)
        weights = np.ones(k, dtype=np.complex128) / np.sqrt(k)
        return cls(matrices=stacked, weights=weights)

    def __repr__(self) -> str:
        return (
            f"TransAmplitude[{self.n_paths} paths, "
            f"{self.input_dim}→{self.output_dim}]"
            f"(neg={self.negativity:.4f})"
        )


# ===================================================================
# Operations
# ===================================================================


def trans_amp_apply(amp: Amplitude, ta: TransAmplitude) -> Amplitude:
    """Apply a TransAmplitude to an Amplitude.

    result = normalize( Σ_k  w_k * (amp.data @ M_k) )

    Each matrix M_k transforms the input in a different direction;
    complex weights w_k create interference between transformation paths.
    """
    if amp.size != ta.input_dim:
        raise ValueError(
            f"Dimension mismatch: amp({amp.size}) vs "
            f"TransAmplitude input({ta.input_dim})"
        )
    # Vectorized: einsum('k,m,kmn->n', weights, amp_data, matrices)
    # Expand amp to (1, M) for broadcasting
    amp_data = amp.data.astype(np.complex128)  # (M,)
    # (K, M) @ (K, M, N) per-path → use einsum
    # path_results[k, n] = sum_m amp_data[m] * matrices[k, m, n]
    # result[n] = sum_k weights[k] * path_results[k, n]
    result = np.einsum(
        "k,m,kmn->n",
        ta.weights,
        amp_data,
        ta.matrices.astype(np.complex128),
    )
    return Amplitude(data=result)


def trans_amp_compose(
    ta1: TransAmplitude,
    ta2: TransAmplitude,
) -> TransAmplitude:
    """Compose two TransAmplitudes — tensor product of paths.

    Result has K1 * K2 paths.
    matrices[i*K2+j] = ta1.matrices[i] @ ta2.matrices[j]
    weights[i*K2+j]  = ta1.weights[i] * ta2.weights[j]
    """
    if ta1.output_dim != ta2.input_dim:
        raise ValueError(
            f"Dimension mismatch: ta1 output({ta1.output_dim}) vs "
            f"ta2 input({ta2.input_dim})"
        )
    k1, k2 = ta1.n_paths, ta2.n_paths
    new_k = k1 * k2
    new_matrices = np.empty(
        (new_k, ta1.input_dim, ta2.output_dim), dtype=np.float64
    )
    new_weights = np.empty(new_k, dtype=np.complex128)

    idx = 0
    for i in range(k1):
        for j in range(k2):
            new_matrices[idx] = ta1.matrices[i] @ ta2.matrices[j]
            new_weights[idx] = ta1.weights[i] * ta2.weights[j]
            idx += 1

    return TransAmplitude(matrices=new_matrices, weights=new_weights)


def trans_amp_superpose(
    tas: list[TransAmplitude],
    outer_weights: list[complex] | np.ndarray,
) -> TransAmplitude:
    """Superpose multiple TransAmplitudes with outer complex weights.

    Concatenates all paths, scaling each TransAmplitude's internal weights
    by the corresponding outer weight.
    """
    if len(tas) != len(outer_weights):
        raise ValueError(
            f"Count mismatch: {len(tas)} TransAmplitudes vs "
            f"{len(outer_weights)} weights"
        )
    if len(tas) == 0:
        raise ValueError("trans_amp_superpose requires at least one TransAmplitude")

    # Validate compatible dimensions
    m, n = tas[0].input_dim, tas[0].output_dim
    for ta in tas[1:]:
        if ta.input_dim != m or ta.output_dim != n:
            raise ValueError(
                f"Dimension mismatch: expected ({m},{n}), "
                f"got ({ta.input_dim},{ta.output_dim})"
            )

    ow = np.array(outer_weights, dtype=np.complex128)
    all_matrices = []
    all_weights = []
    for ta, w in zip(tas, ow):
        all_matrices.append(ta.matrices)
        all_weights.append(ta.weights * w)

    combined_matrices = np.concatenate(all_matrices, axis=0)
    combined_weights = np.concatenate(all_weights)
    return TransAmplitude(matrices=combined_matrices, weights=combined_weights)


def trans_amp_collapse(ta: TransAmplitude) -> tuple[TransMatrix, int]:
    """Observe: select the most probable transformation path.

    Returns (chosen_matrix, chosen_index).
    """
    probs = ta.path_probabilities
    chosen = int(np.argmax(probs))
    matrix_data = ta.matrices[chosen].astype(np.float32)
    return TransMatrix(data=matrix_data), chosen


def trans_amp_effective(ta: TransAmplitude) -> TransMatrix:
    """Compute the effective (interference-included) matrix.

    M_eff = Σ_k  w_k * M_k

    This is a real matrix obtained by taking the real part of the
    weighted sum (imaginary parts cancel for physical observables).
    """
    # (K,1,1) * (K,M,N) → sum over K → (M,N) complex
    weighted = ta.weights[:, np.newaxis, np.newaxis] * ta.matrices.astype(np.complex128)
    effective = np.sum(weighted, axis=0)
    # Take real part for a physical TransMatrix
    return TransMatrix(data=effective.real.astype(np.float32))
