"""Tests for the Axol module system — schema, registry, UseOp, compose."""

import os

import numpy as np
import pytest

from axol.core.types import FloatVec, GateVec, TransMatrix, StateBundle
from axol.core.program import (
    Program, Transition, TransformOp, MergeOp, CustomOp, run_program,
)
from axol.core.module import (
    VecSchema, ModuleSchema, Module, ModuleRegistry,
    UseOp, compose, validate_schema, _apply_use_op,
)
from axol.core.dsl import parse

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _make_matrix(rows):
    """Helper to create TransMatrix from nested list."""
    return TransMatrix(data=np.array(rows, dtype=np.float32))


# ═══════════════════════════════════════════════════════════════════════════
# 1. Module Schema
# ═══════════════════════════════════════════════════════════════════════════

class TestModuleSchema:
    def test_create_schema(self):
        schema = ModuleSchema(
            inputs=[VecSchema("atk", "float", 1), VecSchema("def_val", "float", 1)],
            outputs=[VecSchema("dmg", "float", 1)],
        )
        assert len(schema.inputs) == 2
        assert len(schema.outputs) == 1

    def test_empty_schema(self):
        schema = ModuleSchema()
        assert schema.inputs == []
        assert schema.outputs == []

    def test_vec_schema_frozen(self):
        vs = VecSchema("a", "float", 3)
        assert vs.name == "a"
        assert vs.dimensions == 3


# ═══════════════════════════════════════════════════════════════════════════
# 2. Module Registry
# ═══════════════════════════════════════════════════════════════════════════

class TestModuleRegistry:
    def test_register_and_get(self):
        reg = ModuleRegistry()
        prog = Program(
            name="test_mod",
            initial_state=StateBundle(vectors={"v": FloatVec.from_list([0.0])}),
            transitions=[],
        )
        mod = Module(name="test_mod", program=prog)
        reg.register(mod)
        assert reg.get("test_mod") is mod

    def test_get_missing_raises(self):
        reg = ModuleRegistry()
        with pytest.raises(KeyError, match="Module not found"):
            reg.get("nonexistent")

    def test_has(self):
        reg = ModuleRegistry()
        assert not reg.has("foo")
        prog = Program(
            name="foo",
            initial_state=StateBundle(vectors={"v": FloatVec.from_list([0.0])}),
            transitions=[],
        )
        reg.register(Module(name="foo", program=prog))
        assert reg.has("foo")

    def test_load_from_file(self):
        reg = ModuleRegistry()
        path = os.path.join(FIXTURES_DIR, "damage_calc.axol")
        mod = reg.load_from_file(path)
        assert mod.name == "damage_calc"
        assert reg.has("damage_calc")

    def test_load_file_not_found(self):
        reg = ModuleRegistry()
        with pytest.raises(FileNotFoundError):
            reg.load_from_file("/nonexistent/path.axol")

    def test_resolve_import_cached(self):
        reg = ModuleRegistry()
        prog = Program(
            name="cached",
            initial_state=StateBundle(vectors={"v": FloatVec.from_list([0.0])}),
            transitions=[],
        )
        reg.register(Module(name="cached", program=prog))
        resolved = reg.resolve_import("cached")
        assert resolved.name == "cached"

    def test_resolve_import_from_file(self):
        reg = ModuleRegistry()
        fixture_file = os.path.join(FIXTURES_DIR, "heal.axol")
        resolved = reg.resolve_import("heal", relative_to=fixture_file)
        assert resolved.name == "heal"


# ═══════════════════════════════════════════════════════════════════════════
# 3. UseOp
# ═══════════════════════════════════════════════════════════════════════════

class TestUseOp:
    def test_use_op_basic(self):
        """UseOp should execute a sub-module and map outputs back."""
        reg = ModuleRegistry()

        # Sub-module: doubles the input
        sub_prog = Program(
            name="doubler",
            initial_state=StateBundle(vectors={
                "x": FloatVec.from_list([0.0]),
            }),
            transitions=[
                Transition("double", TransformOp(
                    key="x",
                    matrix=_make_matrix([[2.0]]),
                )),
            ],
        )
        reg.register(Module(name="doubler", program=sub_prog))

        op = UseOp(
            module_name="doubler",
            input_mapping={"x": "my_val"},
            output_mapping={"x": "my_val"},
            registry=reg,
        )

        state = StateBundle(vectors={"my_val": FloatVec.from_list([5.0])})
        result = _apply_use_op(op, state)
        assert result["my_val"].to_list() == pytest.approx([10.0])

    def test_use_op_no_registry_raises(self):
        op = UseOp(
            module_name="missing",
            input_mapping={},
            output_mapping={},
            registry=None,
        )
        with pytest.raises(RuntimeError, match="no registry"):
            _apply_use_op(op, StateBundle())


# ═══════════════════════════════════════════════════════════════════════════
# 4. Compose
# ═══════════════════════════════════════════════════════════════════════════

class TestCompose:
    def test_compose_two_programs(self):
        p1 = Program(
            name="p1",
            initial_state=StateBundle(vectors={"a": FloatVec.from_list([1.0])}),
            transitions=[Transition("t1", TransformOp(key="a", matrix=_make_matrix([[2.0]])))],
        )
        p2 = Program(
            name="p2",
            initial_state=StateBundle(vectors={"b": FloatVec.from_list([3.0])}),
            transitions=[Transition("t2", TransformOp(key="b", matrix=_make_matrix([[4.0]])))],
        )
        composed = compose(p1, p2, name="combined")
        assert composed.name == "combined"
        assert "a" in composed.initial_state
        assert "b" in composed.initial_state
        assert len(composed.transitions) == 2

    def test_compose_runs_correctly(self):
        p1 = Program(
            name="p1",
            initial_state=StateBundle(vectors={"v": FloatVec.from_list([2.0])}),
            transitions=[Transition("double", TransformOp(key="v", matrix=_make_matrix([[3.0]])))],
        )
        p2 = Program(
            name="p2",
            initial_state=StateBundle(vectors={}),
            transitions=[Transition("triple", TransformOp(key="v", matrix=_make_matrix([[2.0]])))],
        )
        composed = compose(p1, p2)
        result = run_program(composed)
        # v=2 -> *3=6 -> *2=12
        assert result.final_state["v"].to_list() == pytest.approx([12.0])

    def test_compose_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            compose()


# ═══════════════════════════════════════════════════════════════════════════
# 5. Schema Validation
# ═══════════════════════════════════════════════════════════════════════════

class TestSchemaValidation:
    def test_valid_input(self):
        schema = ModuleSchema(
            inputs=[VecSchema("x", "float", 2)],
            outputs=[],
        )
        mod = Module(name="m", program=Program(
            name="m",
            initial_state=StateBundle(),
            transitions=[],
        ), schema=schema)
        errors = validate_schema(mod, StateBundle(vectors={
            "x": FloatVec.from_list([1.0, 2.0]),
        }))
        assert errors == []

    def test_missing_input(self):
        schema = ModuleSchema(
            inputs=[VecSchema("x", "float", 2)],
            outputs=[],
        )
        mod = Module(name="m", program=Program(
            name="m",
            initial_state=StateBundle(),
            transitions=[],
        ), schema=schema)
        errors = validate_schema(mod, StateBundle())
        assert len(errors) == 1
        assert "Missing input" in errors[0]

    def test_dimension_mismatch(self):
        schema = ModuleSchema(
            inputs=[VecSchema("x", "float", 3)],
            outputs=[],
        )
        mod = Module(name="m", program=Program(
            name="m",
            initial_state=StateBundle(),
            transitions=[],
        ), schema=schema)
        errors = validate_schema(mod, StateBundle(vectors={
            "x": FloatVec.from_list([1.0, 2.0]),  # dim=2, expected 3
        }))
        assert len(errors) == 1
        assert "dim=" in errors[0]


# ═══════════════════════════════════════════════════════════════════════════
# 6. DSL Import / Use
# ═══════════════════════════════════════════════════════════════════════════

class TestDSLImportUse:
    def test_load_fixture_and_use(self):
        """Load a fixture module and use it via registry."""
        reg = ModuleRegistry()
        reg.load_from_file(os.path.join(FIXTURES_DIR, "damage_calc.axol"))
        assert reg.has("damage_calc")

    def test_load_heal_fixture(self):
        reg = ModuleRegistry()
        mod = reg.load_from_file(os.path.join(FIXTURES_DIR, "heal.axol"))
        assert mod.name == "heal"
        result = run_program(mod.program)
        assert "hp" in result.final_state


# ═══════════════════════════════════════════════════════════════════════════
# 7. Error Cases
# ═══════════════════════════════════════════════════════════════════════════

class TestModuleErrors:
    def test_missing_module_in_registry(self):
        reg = ModuleRegistry()
        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_resolve_unresolvable(self):
        reg = ModuleRegistry()
        with pytest.raises(KeyError, match="Cannot resolve"):
            reg.resolve_import("no_such_module")
