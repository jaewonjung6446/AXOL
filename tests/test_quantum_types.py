"""Tests for quantum types and declaration system (Phase 1)."""

import pytest
import numpy as np

from axol.core.types import FloatVec, TransMatrix
from axol.quantum.errors import QuantumError, WeaverError, ObservatoryError, QuantumParseError
from axol.quantum.types import (
    SuperposedState, Attractor, TapestryNode, Tapestry, WeaverReport, Observation,
)
from axol.quantum.declare import (
    RelationKind, QualityTarget, DeclaredInput, DeclaredRelation,
    EntangleDeclaration, DeclarationBuilder,
)


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------

class TestErrors:
    def test_quantum_error_is_base(self):
        assert issubclass(WeaverError, QuantumError)
        assert issubclass(ObservatoryError, QuantumError)
        assert issubclass(QuantumParseError, QuantumError)

    def test_raise_and_catch(self):
        with pytest.raises(QuantumError):
            raise WeaverError("test")


# ---------------------------------------------------------------------------
# SuperposedState
# ---------------------------------------------------------------------------

class TestSuperposedState:
    def test_basic_creation(self):
        amp = FloatVec.from_list([0.6, 0.8])
        state = SuperposedState(name="test", amplitudes=amp)
        assert state.name == "test"
        assert state.dim == 2

    def test_probabilities_born_rule(self):
        amp = FloatVec.from_list([0.6, 0.8])
        state = SuperposedState(name="test", amplitudes=amp)
        probs = state.probabilities
        # |0.6|^2 = 0.36, |0.8|^2 = 0.64 → normalised to 0.36, 0.64
        assert abs(probs.data[0] - 0.36) < 0.01
        assert abs(probs.data[1] - 0.64) < 0.01

    def test_most_probable_index(self):
        amp = FloatVec.from_list([0.1, 0.5, 0.3])
        state = SuperposedState(name="test", amplitudes=amp)
        assert state.most_probable_index == 1

    def test_labels(self):
        amp = FloatVec.from_list([0.1, 0.9])
        labels = {0: "cat", 1: "dog"}
        state = SuperposedState(name="test", amplitudes=amp, labels=labels)
        assert state.most_probable_label == "dog"

    def test_no_label(self):
        amp = FloatVec.from_list([0.1, 0.9])
        state = SuperposedState(name="test", amplitudes=amp)
        assert state.most_probable_label is None


# ---------------------------------------------------------------------------
# Attractor
# ---------------------------------------------------------------------------

class TestAttractor:
    def test_convergent_attractor(self):
        M = TransMatrix(data=np.eye(4, dtype=np.float32) * 0.5)
        att = Attractor(
            phase_space_dim=4,
            embedding_dim=4,
            fractal_dim=0.0,
            lyapunov_spectrum=[-1.0, -1.0, -1.0, -1.0],
            max_lyapunov=-1.0,
            basin_bounds=(-1.0, 1.0),
            trajectory_matrix=M,
        )
        assert not att.is_chaotic
        assert att.omega == 1.0
        assert att.phi == 1.0

    def test_chaotic_attractor(self):
        M = TransMatrix(data=np.eye(4, dtype=np.float32) * 2.0)
        att = Attractor(
            phase_space_dim=4,
            embedding_dim=4,
            fractal_dim=2.0,
            lyapunov_spectrum=[0.9, 0.1, -0.5, -1.0],
            max_lyapunov=0.9,
            basin_bounds=(-2.0, 2.0),
            trajectory_matrix=M,
        )
        assert att.is_chaotic
        assert att.omega < 1.0  # 1/(1+0.9) ≈ 0.526
        assert att.phi < 1.0    # 1/(1+2/4) = 0.667

    def test_omega_formula(self):
        M = TransMatrix(data=np.eye(2, dtype=np.float32))
        att = Attractor(
            phase_space_dim=2, embedding_dim=2, fractal_dim=0.5,
            lyapunov_spectrum=[1.0], max_lyapunov=1.0,
            basin_bounds=(-1.0, 1.0), trajectory_matrix=M,
        )
        assert abs(att.omega - 0.5) < 0.001  # 1/(1+1) = 0.5

    def test_phi_formula(self):
        M = TransMatrix(data=np.eye(4, dtype=np.float32))
        att = Attractor(
            phase_space_dim=4, embedding_dim=4, fractal_dim=2.0,
            lyapunov_spectrum=[0.0], max_lyapunov=0.0,
            basin_bounds=(-1.0, 1.0), trajectory_matrix=M,
        )
        # Phi = 1/(1+2/4) = 1/1.5 ≈ 0.667
        assert abs(att.phi - 1.0 / 1.5) < 0.001


# ---------------------------------------------------------------------------
# QualityTarget
# ---------------------------------------------------------------------------

class TestQualityTarget:
    def test_valid_range(self):
        qt = QualityTarget(omega=0.9, phi=0.7)
        assert qt.omega == 0.9
        assert qt.phi == 0.7

    def test_omega_out_of_range(self):
        with pytest.raises(ValueError):
            QualityTarget(omega=1.5, phi=0.7)

    def test_phi_out_of_range(self):
        with pytest.raises(ValueError):
            QualityTarget(omega=0.9, phi=-0.1)

    def test_boundary_values(self):
        qt = QualityTarget(omega=0.0, phi=1.0)
        assert qt.omega == 0.0
        assert qt.phi == 1.0


# ---------------------------------------------------------------------------
# Declaration + Builder
# ---------------------------------------------------------------------------

class TestDeclarationBuilder:
    def test_simple_declaration(self):
        decl = (
            DeclarationBuilder("search")
            .input("query", 64)
            .input("db", 64)
            .relate("relevance", ["query", "db"], RelationKind.PROPORTIONAL)
            .output("relevance")
            .quality(0.9, 0.7)
            .build()
        )
        assert decl.name == "search"
        assert len(decl.inputs) == 2
        assert len(decl.relations) == 1
        assert decl.quality_target.omega == 0.9

    def test_dependency_graph(self):
        decl = (
            DeclarationBuilder("pipeline")
            .input("a", 8)
            .relate("b", ["a"], RelationKind.PROPORTIONAL)
            .relate("c", ["b"], RelationKind.ADDITIVE)
            .output("c")
            .build()
        )
        graph = decl.dependency_graph()
        assert "a" in graph["b"]
        assert "b" in graph["c"]

    def test_topological_order(self):
        decl = (
            DeclarationBuilder("pipeline")
            .input("a", 8)
            .input("b", 8)
            .relate("c", ["a", "b"], RelationKind.PROPORTIONAL)
            .relate("d", ["c"], RelationKind.ADDITIVE)
            .output("d")
            .build()
        )
        order = decl.topological_order()
        # a and b must come before c, c before d
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("c")
        assert order.index("c") < order.index("d")

    def test_cycle_detection(self):
        # Manually create a cyclic declaration
        decl = EntangleDeclaration(
            name="cycle",
            inputs=[DeclaredInput("a", 8)],
            outputs=["b"],
            relations=[
                DeclaredRelation("b", ["c"], RelationKind.PROPORTIONAL),
                DeclaredRelation("c", ["b"], RelationKind.PROPORTIONAL),
            ],
        )
        with pytest.raises(QuantumError, match="Cycle"):
            decl.topological_order()

    def test_builder_requires_input(self):
        with pytest.raises(QuantumError):
            DeclarationBuilder("empty").relate("a", ["b"], RelationKind.PROPORTIONAL).build()

    def test_builder_requires_relation(self):
        with pytest.raises(QuantumError):
            DeclarationBuilder("empty").input("a", 8).build()

    def test_inferred_outputs(self):
        decl = (
            DeclarationBuilder("test")
            .input("x", 8)
            .relate("y", ["x"], RelationKind.PROPORTIONAL)
            .build()
        )
        assert "y" in decl.outputs

    def test_relation_kinds(self):
        for kind in RelationKind:
            decl = (
                DeclarationBuilder(f"test_{kind.name}")
                .input("x", 8)
                .relate("y", ["x"], kind)
                .output("y")
                .build()
            )
            assert decl.relations[0].kind == kind

    def test_multi_source_relation(self):
        decl = (
            DeclarationBuilder("multi")
            .input("a", 8)
            .input("b", 8)
            .input("c", 8)
            .relate("out", ["a", "b", "c"], RelationKind.ADDITIVE)
            .output("out")
            .build()
        )
        assert len(decl.relations[0].sources) == 3

    def test_labels_on_input(self):
        labels = {0: "cat", 1: "dog", 2: "bird"}
        decl = (
            DeclarationBuilder("classify")
            .input("image", 8, labels=labels)
            .relate("category", ["image"], RelationKind.PROPORTIONAL)
            .output("category")
            .build()
        )
        assert decl.inputs[0].labels == labels
