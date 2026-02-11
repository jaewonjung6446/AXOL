"""Tests for the quantum DSL parser (Phase 6)."""

import pytest

from axol.quantum.dsl import (
    parse_quantum,
    QuantumProgram,
    ObserveStatement,
    ReobserveStatement,
    ConditionalBlock,
)
from axol.quantum.declare import RelationKind
from axol.quantum.errors import QuantumParseError


class TestParseEntangle:
    def test_basic_entangle(self):
        source = """
entangle search(query: float[64], db: float[64]) @ Omega(0.9) Phi(0.7) {
    relevance <~> similarity(query, db)
    ranking <~> relevance
}
"""
        prog = parse_quantum(source)
        assert isinstance(prog, QuantumProgram)
        assert len(prog.declarations) == 1
        decl = prog.declarations[0]
        assert decl.name == "search"
        assert len(decl.inputs) == 2
        assert decl.inputs[0].name == "query"
        assert decl.inputs[0].dim == 64
        assert decl.quality_target.omega == 0.9
        assert decl.quality_target.phi == 0.7

    def test_all_operators(self):
        source = """
entangle ops(a: float[8]) @ Omega(0.8) Phi(0.6) {
    b <~> a
    c <+> a
    d <*> a
    e <!> a
    f <?> a
}
"""
        prog = parse_quantum(source)
        decl = prog.declarations[0]
        kinds = [r.kind for r in decl.relations]
        assert RelationKind.PROPORTIONAL in kinds
        assert RelationKind.ADDITIVE in kinds
        assert RelationKind.MULTIPLICATIVE in kinds
        assert RelationKind.INVERSE in kinds
        assert RelationKind.CONDITIONAL in kinds

    def test_multiple_sources(self):
        source = """
entangle multi(a: float[8], b: float[8]) @ Omega(0.9) Phi(0.7) {
    c <~> combine(a, b)
}
"""
        prog = parse_quantum(source)
        rel = prog.declarations[0].relations[0]
        assert "a" in rel.sources
        assert "b" in rel.sources

    def test_comments_ignored(self):
        source = """
# This is a comment
entangle test(x: float[4]) @ Omega(0.9) Phi(0.7) {
    // Another comment
    y <~> x
}
"""
        prog = parse_quantum(source)
        assert len(prog.declarations) == 1

    def test_simple_param_names(self):
        source = """
entangle test(x, y) @ Omega(0.9) Phi(0.7) {
    z <~> combine(x, y)
}
"""
        prog = parse_quantum(source)
        assert prog.declarations[0].inputs[0].name == "x"
        assert prog.declarations[0].inputs[1].name == "y"
        # Default dimension
        assert prog.declarations[0].inputs[0].dim == 8


class TestParseObserve:
    def test_basic_observe(self):
        source = """
entangle search(query: float[8]) @ Omega(0.9) Phi(0.7) {
    result <~> query
}

result = observe search(query_vec)
"""
        prog = parse_quantum(source)
        assert len(prog.observations) == 1
        obs = prog.observations[0]
        assert isinstance(obs, ObserveStatement)
        assert obs.result_var == "result"
        assert obs.tapestry_name == "search"
        assert obs.arguments == ["query_vec"]

    def test_reobserve(self):
        source = """
entangle search(query: float[8]) @ Omega(0.9) Phi(0.7) {
    result <~> query
}

result = reobserve search(query_vec) x 10
"""
        prog = parse_quantum(source)
        obs = prog.observations[0]
        assert isinstance(obs, ReobserveStatement)
        assert obs.count == 10

    def test_multiple_args(self):
        source = """
entangle search(query: float[8], db: float[8]) @ Omega(0.9) Phi(0.7) {
    result <~> combine(query, db)
}

result = observe search(q, database)
"""
        prog = parse_quantum(source)
        assert prog.observations[0].arguments == ["q", "database"]


class TestParseConditional:
    def test_basic_conditional(self):
        source = """
entangle search(query: float[8]) @ Omega(0.9) Phi(0.7) {
    result <~> query
}

result = observe search(query_vec)

if result.Omega < 0.95 {
    result = reobserve search(query_vec) x 10
}
"""
        prog = parse_quantum(source)
        assert len(prog.conditionals) == 1
        cond = prog.conditionals[0]
        assert cond.variable == "result"
        assert cond.field_name == "Omega"
        assert cond.operator == "<"
        assert cond.threshold == 0.95
        assert len(cond.body) == 1


class TestParseErrors:
    def test_invalid_entangle(self):
        with pytest.raises(QuantumParseError):
            parse_quantum("entangle {}")

    def test_no_operator_in_relation(self):
        source = """
entangle test(x: float[8]) @ Omega(0.9) Phi(0.7) {
    y = x
}
"""
        with pytest.raises(QuantumParseError):
            parse_quantum(source)

    def test_unexpected_line(self):
        with pytest.raises(QuantumParseError):
            parse_quantum("some random text")


class TestFullProgram:
    def test_full_program(self):
        source = """
entangle search(query: float[64], db: float[64]) @ Omega(0.9) Phi(0.7) {
    relevance <~> similarity(query, db)
    ranking <~> relevance
}

result = observe search(query_vec, db_vec)

if result.Omega < 0.95 {
    result = reobserve search(query_vec, db_vec) x 10
}
"""
        prog = parse_quantum(source)
        assert len(prog.declarations) == 1
        assert len(prog.observations) == 1
        assert len(prog.conditionals) == 1
