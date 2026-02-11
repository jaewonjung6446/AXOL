"""Tests for entanglement cost estimation (Phase 3)."""

import pytest

from axol.quantum.declare import (
    DeclarationBuilder, RelationKind, QualityTarget,
)
from axol.quantum.cost import estimate_cost, CostEstimate


class TestEstimateCost:
    def test_simple_cost(self):
        decl = (
            DeclarationBuilder("simple")
            .input("x", 8)
            .relate("y", ["x"], RelationKind.PROPORTIONAL)
            .output("y")
            .quality(0.8, 0.7)
            .build()
        )
        cost = estimate_cost(decl)
        assert isinstance(cost, CostEstimate)
        assert cost.total_cost > 0
        assert "y" in cost.per_node_cost
        assert cost.per_node_cost.get("x", 0) == 0  # inputs have zero cost

    def test_complex_costs_more(self):
        """More complex relations should cost more."""
        simple = (
            DeclarationBuilder("s")
            .input("x", 8)
            .relate("y", ["x"], RelationKind.PROPORTIONAL)
            .output("y")
            .build()
        )
        complex_decl = (
            DeclarationBuilder("c")
            .input("x", 8)
            .relate("y", ["x"], RelationKind.CONDITIONAL)
            .output("y")
            .build()
        )
        c_simple = estimate_cost(simple)
        c_complex = estimate_cost(complex_decl)
        assert c_complex.total_cost > c_simple.total_cost

    def test_multi_source_costs_more(self):
        """Multi-source relations should cost more than single-source."""
        single = (
            DeclarationBuilder("s")
            .input("x", 8)
            .relate("y", ["x"], RelationKind.ADDITIVE)
            .output("y")
            .build()
        )
        multi = (
            DeclarationBuilder("m")
            .input("a", 8)
            .input("b", 8)
            .input("c", 8)
            .relate("y", ["a", "b", "c"], RelationKind.ADDITIVE)
            .output("y")
            .build()
        )
        c_single = estimate_cost(single)
        c_multi = estimate_cost(multi)
        assert c_multi.total_cost > c_single.total_cost

    def test_critical_path(self):
        decl = (
            DeclarationBuilder("pipeline")
            .input("a", 8)
            .relate("b", ["a"], RelationKind.PROPORTIONAL)
            .relate("c", ["b"], RelationKind.MULTIPLICATIVE)
            .output("c")
            .build()
        )
        cost = estimate_cost(decl)
        assert len(cost.critical_path) >= 2

    def test_feasibility_check(self):
        """Reasonable targets should be feasible."""
        decl = (
            DeclarationBuilder("simple")
            .input("x", 8)
            .relate("y", ["x"], RelationKind.PROPORTIONAL)
            .output("y")
            .quality(0.5, 0.5)
            .build()
        )
        cost = estimate_cost(decl)
        assert cost.feasible

    def test_max_achievable_metrics(self):
        decl = (
            DeclarationBuilder("test")
            .input("x", 8)
            .relate("y", ["x"], RelationKind.PROPORTIONAL)
            .output("y")
            .build()
        )
        cost = estimate_cost(decl)
        assert 0.0 <= cost.max_achievable_omega <= 1.0
        assert 0.0 <= cost.max_achievable_phi <= 1.0

    def test_pipeline_cost_accumulates(self):
        """Longer pipeline should have higher total cost."""
        short = (
            DeclarationBuilder("short")
            .input("x", 8)
            .relate("y", ["x"], RelationKind.PROPORTIONAL)
            .output("y")
            .build()
        )
        long_decl = (
            DeclarationBuilder("long")
            .input("x", 8)
            .relate("a", ["x"], RelationKind.PROPORTIONAL)
            .relate("b", ["a"], RelationKind.PROPORTIONAL)
            .relate("c", ["b"], RelationKind.PROPORTIONAL)
            .output("c")
            .build()
        )
        c_short = estimate_cost(short)
        c_long = estimate_cost(long_decl)
        assert c_long.total_cost > c_short.total_cost
