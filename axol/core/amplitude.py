"""Amplitude — complex amplitude vector as a first-class variable.

A distribution (complex amplitude vector) that defers collapse until
explicit observation, enabling interference between computation paths.

Theoretical basis:
  - Veitch et al.: negative quasi-probability → quantum computational advantage
  - Born rule: probabilities = |a_i|^2
  - Interference: complex amplitudes can constructively/destructively combine
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np

from axol.core.types import _VecBase, FloatVec, ComplexVec


@dataclass(frozen=True, eq=False)
class Amplitude(_VecBase):
    """Complex amplitude vector — a distribution that IS the variable.

    - data: complex128 array, auto-normalized (||a|| = 1)
    - Negative/complex entries → interference possible (computational resource)
    - All-positive-real entries → classical (no interference advantage)
    """

    def __post_init__(self) -> None:
        # Ensure complex128
        if not np.issubdtype(self.data.dtype, np.complexfloating):
            object.__setattr__(self, "data", self.data.astype(np.complex128))
        # Auto-normalize to unit norm
        norm = np.linalg.norm(self.data)
        if norm > 0 and not np.isclose(norm, 1.0, atol=1e-12):
            object.__setattr__(self, "data", (self.data / norm).astype(np.complex128))

    # --- Properties ---

    @property
    def probabilities(self) -> np.ndarray:
        """Born rule: |a_i|^2, normalized. Returns raw ndarray (not FloatVec)."""
        p = np.abs(self.data) ** 2
        total = np.sum(p)
        if total > 0:
            p = p / total
        return p.astype(np.float64)

    @property
    def phases(self) -> np.ndarray:
        """Phase angles arg(a_i) in radians."""
        return np.angle(self.data)

    @property
    def negativity(self) -> float:
        """Sum of negative real parts — non-classicality measure (Veitch et al.).

        Higher negativity → more quantum resource → stronger interference.
        Classical states (all real positive) have negativity = 0.
        """
        reals = self.data.real
        return float(-np.sum(reals[reals < 0]))

    @property
    def is_classical(self) -> bool:
        """True if all entries are real and non-negative (no interference advantage)."""
        return bool(
            np.allclose(self.data.imag, 0, atol=1e-12)
            and np.all(self.data.real >= -1e-12)
        )

    # --- Collapse / conversion ---

    def collapse(self) -> int:
        """Observe: argmax(|a_i|^2) → int. The only collapse point."""
        return int(np.argmax(self.probabilities))

    def to_floatvec(self) -> FloatVec:
        """Born rule → FloatVec. Irreversible: phase information is lost."""
        return FloatVec(data=self.probabilities.astype(np.float32))

    def to_complexvec(self) -> ComplexVec:
        """Wrap as ComplexVec (lossless)."""
        return ComplexVec(data=self.data.copy())

    # --- Constructors ---

    @classmethod
    def uniform(cls, n: int) -> Self:
        """Uniform superposition: 1/sqrt(n) for all components."""
        data = np.ones(n, dtype=np.complex128) / np.sqrt(n)
        return cls(data=data)

    @classmethod
    def basis(cls, index: int, n: int) -> Self:
        """Basis state (deterministic / already collapsed)."""
        data = np.zeros(n, dtype=np.complex128)
        data[index] = 1.0 + 0j
        return cls(data=data)

    @classmethod
    def from_floatvec(cls, vec: FloatVec) -> Self:
        """Promote FloatVec → Amplitude (real, phase=0). Reversible via Born rule only if already a distribution."""
        data = vec.data.astype(np.float64)
        # Take sqrt of absolute values to go from probabilities to amplitudes
        # preserving sign information as phase
        signs = np.sign(data)
        magnitudes = np.sqrt(np.abs(data))
        complex_data = (magnitudes * signs).astype(np.complex128)
        return cls(data=complex_data)

    @classmethod
    def from_complexvec(cls, vec: ComplexVec) -> Self:
        """Wrap ComplexVec as Amplitude (auto-normalizes)."""
        return cls(data=vec.data.copy())

    @classmethod
    def from_array(cls, arr: np.ndarray) -> Self:
        """Create from raw numpy array."""
        return cls(data=arr.astype(np.complex128))

    def __repr__(self) -> str:
        neg = self.negativity
        top_idx = self.collapse()
        top_prob = self.probabilities[top_idx]
        return f"Amplitude[{self.size}](top={top_idx}@{top_prob:.3f}, neg={neg:.4f})"
