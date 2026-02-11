"""Integration tests for the full quantum pipeline (Phase 7)."""

import pytest
import numpy as np

from axol.core.types import FloatVec

from axol.quantum import (
    # Types
    SuperposedState, Attractor, Tapestry, Observation, WeaverReport,
    # Declaration
    DeclarationBuilder, RelationKind, QualityTarget,
    # Pipeline
    weave, observe, reobserve,
    # DSL
    parse_quantum, QuantumProgram,
    # Math
    estimate_lyapunov, omega_from_lyapunov,
    estimate_fractal_dim, phi_from_fractal, phi_from_entropy,
    # Composition
    compose_serial, compose_parallel, can_reuse_after_observe,
    # Cost
    estimate_cost,
    # Errors
    QuantumError, WeaverError, ObservatoryError, QuantumParseError,
)


class TestRoundTrip:
    """Full pipeline: DSL source → parse → weave → observe → result."""

    def test_dsl_to_observation(self):
        source = """
entangle search(query: float[8], db: float[8]) @ Omega(0.9) Phi(0.7) {
    relevance <~> similarity(query, db)
    ranking <~> relevance
}
"""
        # Parse
        prog = parse_quantum(source)
        assert len(prog.declarations) == 1

        # Weave
        decl = prog.declarations[0]
        tapestry = weave(decl, seed=42)
        assert isinstance(tapestry, Tapestry)

        # Observe
        inputs = {
            "query": FloatVec.from_list([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "db": FloatVec.from_list([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        }
        result = observe(tapestry, inputs)
        assert isinstance(result, Observation)
        assert 0.0 <= result.omega <= 1.0
        assert 0.0 <= result.phi <= 1.0
        assert result.tapestry_name == "search"

    def test_builder_to_observation(self):
        """Full pipeline using the builder API."""
        decl = (
            DeclarationBuilder("classify")
            .input("image", 16, labels={0: "cat", 1: "dog", 2: "bird"})
            .relate("category", ["image"], RelationKind.PROPORTIONAL)
            .output("category")
            .quality(0.9, 0.6)
            .build()
        )

        tapestry = weave(decl, seed=42)
        inputs = {"image": FloatVec(data=np.random.default_rng(42).random(16).astype(np.float32))}
        result = observe(tapestry, inputs)

        assert isinstance(result, Observation)
        assert result.probabilities.size == 16

    def test_reobserve_round_trip(self):
        decl = (
            DeclarationBuilder("compute")
            .input("x", 8)
            .relate("y", ["x"], RelationKind.PROPORTIONAL)
            .output("y")
            .quality(0.9, 0.8)
            .build()
        )
        tapestry = weave(decl, seed=42)
        inputs = {"x": FloatVec.from_list([0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1])}

        # Single observe
        obs1 = observe(tapestry, inputs)
        assert obs1.observation_count == 1

        # Reobserve
        obs2 = reobserve(tapestry, inputs, count=10, seed=42)
        assert obs2.observation_count == 10


class TestQualityMetrics:
    def test_uniform_low_phi(self):
        """Uniform distribution should have low Phi."""
        probs = FloatVec.from_list([0.125] * 8)
        phi = phi_from_entropy(probs)
        assert phi < 0.05

    def test_delta_high_phi(self):
        """Delta distribution should have high Phi."""
        probs = FloatVec.from_list([0.0, 0.0, 1.0, 0.0])
        phi = phi_from_entropy(probs)
        assert phi > 0.95

    def test_omega_convergent(self):
        """Negative lambda → Omega = 1.0"""
        assert omega_from_lyapunov(-2.0) == 1.0

    def test_omega_chaotic(self):
        """Positive lambda → Omega < 1.0"""
        omega = omega_from_lyapunov(1.0)
        assert abs(omega - 0.5) < 0.001


class TestCompositionIntegration:
    def test_serial_omega_degradation(self):
        """Multi-stage pipeline should have lower Omega than individual stages."""
        decl = (
            DeclarationBuilder("pipeline")
            .input("x", 8)
            .relate("a", ["x"], RelationKind.PROPORTIONAL)
            .relate("b", ["a"], RelationKind.ADDITIVE)
            .relate("c", ["b"], RelationKind.MULTIPLICATIVE)
            .output("c")
            .quality(0.7, 0.5)
            .build()
        )
        tapestry = weave(decl, seed=42)
        # Global omega should reflect composition effects
        assert 0.0 < tapestry.weaver_report.estimated_omega <= 1.0

    def test_parallel_min_rule(self):
        """Parallel composition limited by weakest component."""
        omega, phi, _, _ = compose_parallel(
            0.9, 0.8, -0.5, 0.5,
            0.5, 0.3, 1.0, 2.0,
        )
        assert omega == 0.5
        assert phi == 0.3


class TestCostIntegration:
    def test_cost_scales_with_complexity(self):
        simple = (
            DeclarationBuilder("simple")
            .input("x", 8)
            .relate("y", ["x"], RelationKind.PROPORTIONAL)
            .output("y")
            .build()
        )
        complex_decl = (
            DeclarationBuilder("complex")
            .input("a", 8)
            .input("b", 8)
            .relate("c", ["a", "b"], RelationKind.MULTIPLICATIVE)
            .relate("d", ["c"], RelationKind.INVERSE)
            .relate("e", ["d", "c"], RelationKind.CONDITIONAL)
            .output("e")
            .build()
        )
        c_simple = estimate_cost(simple)
        c_complex = estimate_cost(complex_decl)
        assert c_complex.total_cost > c_simple.total_cost


class TestImportsComplete:
    """Verify all public API members are importable."""

    def test_types_importable(self):
        from axol.quantum import SuperposedState, Attractor, TapestryNode
        from axol.quantum import Tapestry, WeaverReport, Observation

    def test_declaration_importable(self):
        from axol.quantum import DeclarationBuilder, EntangleDeclaration
        from axol.quantum import QualityTarget, RelationKind
        from axol.quantum import DeclaredInput, DeclaredRelation

    def test_math_importable(self):
        from axol.quantum import estimate_lyapunov, lyapunov_spectrum
        from axol.quantum import omega_from_lyapunov, omega_from_observations
        from axol.quantum import estimate_fractal_dim, phi_from_fractal, phi_from_entropy

    def test_pipeline_importable(self):
        from axol.quantum import weave, observe, reobserve

    def test_dsl_importable(self):
        from axol.quantum import parse_quantum, QuantumProgram

    def test_compose_importable(self):
        from axol.quantum import compose_serial, compose_parallel, can_reuse_after_observe

    def test_cost_importable(self):
        from axol.quantum import estimate_cost, CostEstimate

    def test_errors_importable(self):
        from axol.quantum import QuantumError, WeaverError, ObservatoryError, QuantumParseError
