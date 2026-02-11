"""AXOL Quantum Module — Chaos-theory-based Declare -> Weave -> Observe pipeline.

Public API:
  - Types: SuperposedState, Attractor, TapestryNode, Tapestry, WeaverReport, Observation
  - Declaration: DeclarationBuilder, EntangleDeclaration, QualityTarget, RelationKind
  - Weaving: weave()
  - Observation: observe(), reobserve()
  - DSL: parse_quantum(), QuantumProgram
  - Math: estimate_lyapunov, omega_from_lyapunov, estimate_fractal_dim, phi_from_fractal
  - Composition: compose_serial, compose_parallel, can_reuse_after_observe
  - Cost: estimate_cost, CostEstimate
  - Errors: QuantumError, WeaverError, ObservatoryError, QuantumParseError
"""

from axol.quantum.errors import (
    QuantumError,
    WeaverError,
    ObservatoryError,
    QuantumParseError,
)
from axol.quantum.types import (
    SuperposedState,
    Attractor,
    TapestryNode,
    Tapestry,
    WeaverReport,
    Observation,
)
from axol.quantum.declare import (
    RelationKind,
    QualityTarget,
    DeclaredInput,
    DeclaredRelation,
    EntangleDeclaration,
    DeclarationBuilder,
)
from axol.quantum.lyapunov import (
    estimate_lyapunov,
    lyapunov_spectrum,
    omega_from_lyapunov,
    omega_from_observations,
)
from axol.quantum.fractal import (
    estimate_fractal_dim,
    phi_from_fractal,
    phi_from_entropy,
)
from axol.quantum.cost import (
    estimate_cost,
    CostEstimate,
)
from axol.quantum.compose import (
    compose_serial,
    compose_parallel,
    can_reuse_after_observe,
)
from axol.quantum.koopman import (
    lifted_dim,
    lift,
    unlift,
    estimate_koopman_matrix,
    compose_koopman_chain,
)
from axol.quantum.weaver import weave
from axol.quantum.observatory import observe, reobserve
from axol.quantum.dsl import (
    parse_quantum,
    QuantumProgram,
    ObserveStatement,
    ReobserveStatement,
    ConditionalBlock,
)

__all__ = [
    # Errors
    "QuantumError", "WeaverError", "ObservatoryError", "QuantumParseError",
    # Types
    "SuperposedState", "Attractor", "TapestryNode", "Tapestry",
    "WeaverReport", "Observation",
    # Declaration
    "RelationKind", "QualityTarget", "DeclaredInput", "DeclaredRelation",
    "EntangleDeclaration", "DeclarationBuilder",
    # Lyapunov
    "estimate_lyapunov", "lyapunov_spectrum",
    "omega_from_lyapunov", "omega_from_observations",
    # Fractal
    "estimate_fractal_dim", "phi_from_fractal", "phi_from_entropy",
    # Cost
    "estimate_cost", "CostEstimate",
    # Compose
    "compose_serial", "compose_parallel", "can_reuse_after_observe",
    # Koopman
    "lifted_dim", "lift", "unlift", "estimate_koopman_matrix", "compose_koopman_chain",
    # Weaver
    "weave",
    # Observatory
    "observe", "reobserve",
    # DSL
    "parse_quantum", "QuantumProgram",
    "ObserveStatement", "ReobserveStatement", "ConditionalBlock",
]
