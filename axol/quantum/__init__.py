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
from axol.quantum.unitary import (
    nearest_unitary,
    estimate_unitary_step,
    compose_unitary_chain,
    reorthogonalize,
    estimate_hybrid_step,
    compose_hybrid_chain,
)
from axol.quantum.density import (
    von_neumann_entropy,
    fidelity,
    apply_channel,
    depolarizing_channel,
    amplitude_damping_channel,
    dephasing_channel,
    svd_to_kraus,
    phi_from_purity,
    omega_from_coherence,
)
from axol.quantum.weaver import weave
from axol.quantum.observatory import observe, reobserve
from axol.quantum.online import (
    OnlineLearner,
    LearningReport,
)
from axol.quantum.conversation import (
    CharTokenizer,
    Verbalizer,
    WorkingMemory,
    ConversationalAxol,
    ConversationReport,
    vocab_from_texts,
)
from axol.quantum.language_model import (
    LanguageModel,
    GenerationResult,
)
from axol.quantum.two_stage import (
    TwoStageLanguageModel,
    TwoStageGeneration,
    TwoStageReport,
)
from axol.quantum.streaming import (
    StreamingLanguageModel,
    StreamResult,
)
from axol.quantum.hierarchical import (
    HierarchicalLanguageModel,
    LevelConfig,
    HierarchicalResult,
    HierarchicalReport,
)
from axol.quantum.sentence_decoder import (
    SentenceDictionary,
    SentenceDecoderLanguageModel,
    SnapResult,
    HybridResponder,
    HybridResponse,
)
from axol.quantum.fractal_text import (
    NoiseField,
    FractalTextGenerator,
    FractalResult,
)
from axol.quantum.ngram_filter import NgramFilter
from axol.quantum import jamo
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
    # Unitary
    "nearest_unitary", "estimate_unitary_step", "compose_unitary_chain", "reorthogonalize",
    "estimate_hybrid_step", "compose_hybrid_chain",
    # Density / Quantum channels
    "von_neumann_entropy", "fidelity", "apply_channel",
    "depolarizing_channel", "amplitude_damping_channel", "dephasing_channel",
    "svd_to_kraus", "phi_from_purity", "omega_from_coherence",
    # Weaver
    "weave",
    # Observatory
    "observe", "reobserve",
    # Online learning
    "OnlineLearner", "LearningReport",
    # Conversational (dual-layer)
    "CharTokenizer", "Verbalizer", "WorkingMemory",
    "ConversationalAxol", "ConversationReport",
    "vocab_from_texts",
    # Language model
    "LanguageModel", "GenerationResult",
    # Two-stage (intent + surface)
    "TwoStageLanguageModel", "TwoStageGeneration", "TwoStageReport",
    # Stream-of-consciousness (recursive intent↔surface)
    "StreamingLanguageModel", "StreamResult",
    # Hierarchical (N-level with chunk propagation)
    "HierarchicalLanguageModel", "LevelConfig",
    "HierarchicalResult", "HierarchicalReport",
    # Sentence-snap decoder (Intent + Dictionary)
    "SentenceDictionary", "SentenceDecoderLanguageModel", "SnapResult",
    "HybridResponder", "HybridResponse",
    # Fractal noise composition (Perlin/fBm-style creative blends)
    "NoiseField", "FractalTextGenerator", "FractalResult",
    # Grammar and Korean-specific helpers
    "NgramFilter", "jamo",
    # DSL
    "parse_quantum", "QuantumProgram",
    "ObserveStatement", "ReobserveStatement", "ConditionalBlock",
]
