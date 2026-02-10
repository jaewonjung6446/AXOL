"""Axol module system — composable programs with schemas and imports.

Modules provide:
  - Schema validation (input/output declarations)
  - Registry for named modules
  - Composition of multiple programs
  - File-based loading with import resolution
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from axol.core.types import FloatVec, StateBundle, _VecBase
from axol.core.program import (
    Program, Transition, TransformOp, CustomOp, OpKind, run_program,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VecSchema:
    """Schema for a single vector port."""
    name: str
    vec_type: str  # "float", "int", "binary", "onehot", "gate"
    dimensions: int


@dataclass(frozen=True)
class ModuleSchema:
    """Schema describing a module's inputs and outputs."""
    inputs: list[VecSchema] = field(default_factory=list)
    outputs: list[VecSchema] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

@dataclass
class Module:
    """A named, reusable Axol program with a schema."""
    name: str
    program: Program
    schema: ModuleSchema = field(default_factory=ModuleSchema)


# ---------------------------------------------------------------------------
# Module Registry
# ---------------------------------------------------------------------------

class ModuleRegistry:
    """Registry for named modules."""

    def __init__(self) -> None:
        self._modules: dict[str, Module] = {}

    def register(self, module: Module) -> None:
        """Register a module by name."""
        self._modules[module.name] = module

    def get(self, name: str) -> Module:
        """Get a module by name."""
        if name not in self._modules:
            raise KeyError(f"Module not found: {name!r}")
        return self._modules[name]

    def has(self, name: str) -> bool:
        return name in self._modules

    def load_from_file(self, path: str, name: str | None = None) -> Module:
        """Load a module from an .axol file.

        Args:
            path: Path to the .axol file.
            name: Optional module name (defaults to filename stem).

        Returns:
            The loaded Module (also registered in this registry).
        """
        from axol.core.dsl import parse

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Module file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            source = f.read()

        prog = parse(source, registry=self, source_path=path)
        mod_name = name or os.path.splitext(os.path.basename(path))[0]
        module = Module(name=mod_name, program=prog)
        self.register(module)
        return module

    def resolve_import(self, name: str, relative_to: str | None = None) -> Module:
        """Resolve an import by name, optionally relative to a file path.

        If already registered, returns the cached module.
        Otherwise attempts to load from file relative to relative_to.
        """
        if self.has(name):
            return self.get(name)

        # Try loading from file
        if relative_to:
            base_dir = os.path.dirname(relative_to)
            candidate = os.path.join(base_dir, f"{name}.axol")
            if os.path.isfile(candidate):
                return self.load_from_file(candidate, name=name)

        raise KeyError(f"Cannot resolve module: {name!r}")


# ---------------------------------------------------------------------------
# UseOp — execute a sub-module
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UseOp:
    """Operation that invokes a sub-module."""
    kind: OpKind = field(default=OpKind.CUSTOM, init=False)
    module_name: str
    input_mapping: dict[str, str]   # module_input_key -> parent_state_key
    output_mapping: dict[str, str]  # module_output_key -> parent_state_key
    registry: ModuleRegistry | None = None


def _apply_use_op(op: UseOp, state: StateBundle) -> StateBundle:
    """Execute a UseOp by running the sub-module's program."""
    if op.registry is None:
        raise RuntimeError(f"UseOp for '{op.module_name}' has no registry")

    module = op.registry.get(op.module_name)
    sub_state = module.program.initial_state.copy()

    # Map parent state into sub-module inputs
    for mod_key, parent_key in op.input_mapping.items():
        if parent_key in state:
            sub_state[mod_key] = state[parent_key]

    # Run the sub-program
    result = run_program(Program(
        name=module.program.name,
        initial_state=sub_state,
        transitions=list(module.program.transitions),
        terminal_key=module.program.terminal_key,
        max_iterations=module.program.max_iterations,
    ))

    # Map sub-module outputs back to parent state
    new_state = state.copy()
    for mod_key, parent_key in op.output_mapping.items():
        if mod_key in result.final_state:
            new_state[parent_key] = result.final_state[mod_key]

    return new_state


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

def compose(*programs: Program, name: str = "composed") -> Program:
    """Compose multiple programs into a single pipeline program.

    The initial state of the composed program is the union of all programs'
    initial states. Transitions are concatenated in order.
    """
    if not programs:
        raise ValueError("compose requires at least one program")

    all_vectors: dict[str, _VecBase] = {}
    all_transitions: list[Transition] = []

    for i, prog in enumerate(programs):
        # Merge initial state (later programs override on key collision)
        for key, vec in prog.initial_state.items():
            all_vectors[key] = vec
        # Prefix transition names to avoid collision
        for t in prog.transitions:
            prefixed_name = f"{prog.name}.{t.name}" if len(programs) > 1 else t.name
            all_transitions.append(Transition(
                name=prefixed_name,
                operation=t.operation,
                metadata=t.metadata,
            ))

    return Program(
        name=name,
        initial_state=StateBundle(vectors=all_vectors),
        transitions=all_transitions,
    )


# ---------------------------------------------------------------------------
# Schema Validation
# ---------------------------------------------------------------------------

def validate_schema(module: Module, input_state: StateBundle) -> list[str]:
    """Validate that input_state satisfies the module's input schema.

    Returns:
        List of error messages (empty = valid).
    """
    errors: list[str] = []

    for spec in module.schema.inputs:
        if spec.name not in input_state:
            errors.append(f"Missing input: {spec.name!r}")
            continue

        vec = input_state[spec.name]
        if vec.data.ndim == 1 and vec.data.shape[0] != spec.dimensions:
            errors.append(
                f"Input {spec.name!r}: expected dim={spec.dimensions}, "
                f"got dim={vec.data.shape[0]}"
            )

    return errors
