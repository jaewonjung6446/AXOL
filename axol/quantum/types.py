"""Chaos-theory-based types for the quantum module.

Core mapping:
  - Tapestry = Strange Attractor
  - Omega (Cohesion) = 1/(1+max(lambda,0))   (Lyapunov inverse)
  - Phi (Clarity) = 1/(1+D/D_max)            (Fractal dim inverse)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from axol.core.types import FloatVec, TransMatrix, StateBundle, ComplexVec, DensityMatrix
from axol.core.program import Program
from axol.core import operations as ops

# TYPE_CHECKING import to avoid circular dependency
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from axol.core.amplitude import Amplitude


# ---------------------------------------------------------------------------
# SuperposedState — state vector with labels and Born-rule probabilities
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SuperposedState:
    """A state vector in phase space with labelled basis states.

    Supports both real (FloatVec) and complex (ComplexVec) amplitudes.
    When complex_amplitudes is set, Born rule uses |alpha_i|^2 with
    full phase information, enabling interference effects.
    """

    name: str
    amplitudes: FloatVec
    labels: dict[int, str] = field(default_factory=dict)
    complex_amplitudes: ComplexVec | None = None

    @property
    def dim(self) -> int:
        return self.amplitudes.size

    @property
    def is_quantum(self) -> bool:
        """True if complex amplitudes are available."""
        return self.complex_amplitudes is not None

    @property
    def probabilities(self) -> FloatVec:
        """Born rule: |alpha_i|^2 normalised."""
        if self.complex_amplitudes is not None:
            return ops.measure_complex(self.complex_amplitudes)
        return ops.measure(self.amplitudes)

    @property
    def most_probable_index(self) -> int:
        """Index of highest probability."""
        return int(np.argmax(self.probabilities.data))

    @property
    def most_probable_label(self) -> str | None:
        return self.labels.get(self.most_probable_index)


# ---------------------------------------------------------------------------
# Attractor — strange attractor (core data structure of a tapestry)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Attractor:
    """Strange attractor — the heart of a tapestry.

    Encodes the dynamics of a weaved declaration as a chaotic (or convergent)
    attractor in phase space.
    """

    phase_space_dim: int
    embedding_dim: int
    fractal_dim: float
    lyapunov_spectrum: list[float]
    max_lyapunov: float
    basin_bounds: tuple[float, float]
    trajectory_matrix: TransMatrix

    @property
    def is_chaotic(self) -> bool:
        return self.max_lyapunov > 0.0

    @property
    def omega(self) -> float:
        """Cohesion from max Lyapunov exponent."""
        return 1.0 / (1.0 + max(self.max_lyapunov, 0.0))

    @property
    def phi(self) -> float:
        """Clarity from fractal dimension."""
        if self.phase_space_dim == 0:
            return 1.0
        return 1.0 / (1.0 + self.fractal_dim / self.phase_space_dim)


# ---------------------------------------------------------------------------
# TapestryNode — a single node in the tapestry graph
# ---------------------------------------------------------------------------

@dataclass
class TapestryNode:
    """A node in the tapestry graph, holding state and attractor info."""

    name: str
    state: SuperposedState
    attractor: Attractor
    incoming_edges: list[tuple[str, TransMatrix]] = field(default_factory=list)
    allocated_cost: float = 0.0
    depth: int = 0


# ---------------------------------------------------------------------------
# WeaverReport — quality metrics from the weaving process
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WeaverReport:
    """Report produced by the weaver after constructing a tapestry."""

    target_omega: float
    target_phi: float
    estimated_omega: float
    estimated_phi: float
    max_lyapunov: float
    fractal_dim: float
    total_cost: float
    cost_breakdown: dict[str, float]
    warnings: list[str]
    feasible: bool


# ---------------------------------------------------------------------------
# Tapestry — the woven entanglement map (= strange attractor system)
# ---------------------------------------------------------------------------

@dataclass
class Tapestry:
    """A woven tapestry — the compiled entanglement map.

    Internally stores the attractor structure and an axol.core.Program
    for execution.
    """

    name: str
    nodes: dict[str, TapestryNode]
    input_names: list[str]
    output_names: list[str]
    global_attractor: Attractor
    weaver_report: WeaverReport
    _internal_program: Program
    _composed_matrix: TransMatrix | None = None
    _composed_chain_info: dict | None = None  # {"input_key": str, "output_key": str, "num_composed": int}
    _koopman_matrix: TransMatrix | None = None
    _koopman_chain_info: dict | None = None  # {"input_key", "output_key", "num_composed", "original_dim", "lifted_dim", "degree"}
    _unitary_matrix: TransMatrix | None = None
    _unitary_chain_info: dict | None = None  # {"input_key", "output_key", "num_composed", "dim"}
    _hybrid_matrix: TransMatrix | None = None  # composed raw A, for observation
    _hybrid_rotation: TransMatrix | None = None  # U@Vh (unitary, quantum gate)
    _hybrid_scales: np.ndarray | None = None  # singular values (decoherence)
    _hybrid_chain_info: dict | None = None  # {"input_key", "output_key", "num_composed", "dim"}
    _distilled_matrix: TransMatrix | None = None
    _distilled_chain_info: dict | None = None  # {"input_key", "output_key", "dim"}
    # --- Quantum structure (Direction B) ---
    _quantum: bool = False  # True when complex amplitudes are used
    _density_matrix: DensityMatrix | None = None  # global density matrix
    _kraus_operators: list | None = None  # Kraus ops from Hybrid SVD
    _fit_info: dict | None = None  # {"n_samples", "n_classes", "accuracy", "method"}
    # --- TransAmplitude (open relations) ---
    _trans_amplitude: object | None = None  # TransAmplitude | None (avoids circular import)
    _trans_amplitude_info: dict | None = None


# ---------------------------------------------------------------------------
# Observation — result of observing a tapestry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Observation:
    """Result of a single observation (collapse) on a tapestry."""

    value: FloatVec
    value_index: int
    value_label: str | None
    omega: float
    phi: float
    probabilities: FloatVec
    tapestry_name: str
    observation_count: int = 1
    # --- Quantum extensions ---
    density_matrix: DensityMatrix | None = None  # post-observation density matrix
    quantum_phi: float | None = None  # Phi from purity (when quantum=True)
    quantum_omega: float | None = None  # Omega from coherence (when quantum=True)
    # --- Amplitude extensions (Phase 1) ---
    amplitude: Amplitude | None = None  # pre-collapse amplitude distribution
