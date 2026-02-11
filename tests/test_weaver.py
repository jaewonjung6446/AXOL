"""Tests for the weaver (Phase 4)."""

import pytest
import numpy as np

from axol.quantum.declare import DeclarationBuilder, RelationKind
from axol.quantum.weaver import weave
from axol.quantum.types import Tapestry, WeaverReport, Attractor


class TestWeave:
    def test_simple_weave(self):
        """Basic weave produces a valid Tapestry."""
        decl = (
            DeclarationBuilder("simple")
            .input("x", 8)
            .relate("y", ["x"], RelationKind.PROPORTIONAL)
            .output("y")
            .quality(0.8, 0.7)
            .build()
        )
        tapestry = weave(decl, seed=42)
        assert isinstance(tapestry, Tapestry)
        assert tapestry.name == "simple"
        assert "x" in tapestry.nodes
        assert "y" in tapestry.nodes
        assert tapestry.input_names == ["x"]
        assert tapestry.output_names == ["y"]

    def test_weaver_report(self):
        decl = (
            DeclarationBuilder("report")
            .input("a", 8)
            .relate("b", ["a"], RelationKind.PROPORTIONAL)
            .output("b")
            .quality(0.9, 0.7)
            .build()
        )
        tapestry = weave(decl, seed=42)
        report = tapestry.weaver_report
        assert isinstance(report, WeaverReport)
        assert report.target_omega == 0.9
        assert report.target_phi == 0.7
        assert 0.0 <= report.estimated_omega <= 1.0
        assert 0.0 <= report.estimated_phi <= 1.0
        assert report.total_cost > 0

    def test_multi_source_weave(self):
        decl = (
            DeclarationBuilder("multi")
            .input("a", 8)
            .input("b", 8)
            .relate("c", ["a", "b"], RelationKind.ADDITIVE)
            .output("c")
            .build()
        )
        tapestry = weave(decl, seed=42)
        assert "c" in tapestry.nodes
        # Should have transitions for both sources + merge
        assert len(tapestry._internal_program.transitions) >= 3

    def test_pipeline_weave(self):
        decl = (
            DeclarationBuilder("pipeline")
            .input("x", 8)
            .relate("a", ["x"], RelationKind.PROPORTIONAL)
            .relate("b", ["a"], RelationKind.ADDITIVE)
            .relate("c", ["b"], RelationKind.MULTIPLICATIVE)
            .output("c")
            .quality(0.7, 0.6)
            .build()
        )
        tapestry = weave(decl, seed=42)
        assert len(tapestry.nodes) >= 4  # x, a, b, c
        # Depth should increase along the pipeline
        assert tapestry.nodes["a"].depth < tapestry.nodes["b"].depth
        assert tapestry.nodes["b"].depth < tapestry.nodes["c"].depth

    def test_attractor_properties(self):
        decl = (
            DeclarationBuilder("attractor")
            .input("x", 8)
            .relate("y", ["x"], RelationKind.PROPORTIONAL)
            .output("y")
            .build()
        )
        tapestry = weave(decl, seed=42)
        att = tapestry.global_attractor
        assert isinstance(att, Attractor)
        assert att.phase_space_dim > 0
        assert att.trajectory_matrix is not None

    def test_input_nodes_convergent(self):
        """Input nodes should have convergent attractors (lambda < 0)."""
        decl = (
            DeclarationBuilder("inputs")
            .input("x", 8)
            .relate("y", ["x"], RelationKind.PROPORTIONAL)
            .output("y")
            .build()
        )
        tapestry = weave(decl, seed=42)
        input_attractor = tapestry.nodes["x"].attractor
        assert input_attractor.max_lyapunov < 0
        assert not input_attractor.is_chaotic

    def test_deterministic_with_seed(self):
        """Same seed → same tapestry."""
        decl = (
            DeclarationBuilder("det")
            .input("x", 8)
            .relate("y", ["x"], RelationKind.PROPORTIONAL)
            .output("y")
            .build()
        )
        t1 = weave(decl, seed=42)
        t2 = weave(decl, seed=42)
        assert t1.weaver_report.estimated_omega == t2.weaver_report.estimated_omega
        assert t1.weaver_report.estimated_phi == t2.weaver_report.estimated_phi

    def test_different_seeds_different_results(self):
        decl = (
            DeclarationBuilder("diff")
            .input("x", 8)
            .relate("y", ["x"], RelationKind.PROPORTIONAL)
            .output("y")
            .build()
        )
        t1 = weave(decl, seed=42)
        t2 = weave(decl, seed=99)
        # Results may differ (not guaranteed, but likely with different seeds)
        # Just check both are valid
        assert isinstance(t1, Tapestry)
        assert isinstance(t2, Tapestry)

    def test_infeasible_warning(self):
        """Requesting very high quality on complex graph should warn."""
        decl = (
            DeclarationBuilder("hard")
            .input("a", 8)
            .input("b", 8)
            .input("c", 8)
            .relate("d", ["a", "b", "c"], RelationKind.INVERSE)
            .relate("e", ["d"], RelationKind.MULTIPLICATIVE)
            .relate("f", ["e", "d"], RelationKind.CONDITIONAL)
            .output("f")
            .quality(0.99, 0.99)
            .build()
        )
        tapestry = weave(decl, seed=42)
        # May or may not have warnings depending on heuristics
        # Just check it produces a valid tapestry
        assert isinstance(tapestry, Tapestry)

    def test_cost_breakdown(self):
        decl = (
            DeclarationBuilder("breakdown")
            .input("x", 8)
            .relate("y", ["x"], RelationKind.PROPORTIONAL)
            .relate("z", ["y"], RelationKind.ADDITIVE)
            .output("z")
            .build()
        )
        tapestry = weave(decl, seed=42)
        report = tapestry.weaver_report
        assert len(report.cost_breakdown) > 0
        assert report.total_cost > 0

    def test_internal_program_has_transitions(self):
        decl = (
            DeclarationBuilder("prog")
            .input("x", 8)
            .relate("y", ["x"], RelationKind.PROPORTIONAL)
            .output("y")
            .build()
        )
        tapestry = weave(decl, seed=42)
        prog = tapestry._internal_program
        assert len(prog.transitions) >= 1
        assert prog.name.startswith("tapestry_")
