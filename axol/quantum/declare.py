"""Declaration AST and builder for the Declare -> Weave -> Observe pipeline.

Provides a fluent API for declaring entangled relationships between variables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np

from axol.core.types import TransMatrix
from axol.quantum.errors import QuantumError


# ---------------------------------------------------------------------------
# Relation kinds (operators)
# ---------------------------------------------------------------------------

class RelationKind(Enum):
    """Kind of relationship between declared variables."""

    PROPORTIONAL = "<~>"    # proportional dependency
    ADDITIVE = "<+>"        # additive combination
    MULTIPLICATIVE = "<*>"  # multiplicative combination
    INVERSE = "<!>"         # inverse relationship
    CONDITIONAL = "<?>"     # conditional dependency


# ---------------------------------------------------------------------------
# Quality target
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QualityTarget:
    """Desired Omega (cohesion) and Phi (clarity)."""

    omega: float = 0.9
    phi: float = 0.7

    def __post_init__(self) -> None:
        if not (0.0 <= self.omega <= 1.0):
            raise ValueError(f"omega must be in [0, 1], got {self.omega}")
        if not (0.0 <= self.phi <= 1.0):
            raise ValueError(f"phi must be in [0, 1], got {self.phi}")


# ---------------------------------------------------------------------------
# Declared components
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeclaredInput:
    """An input variable declaration."""

    name: str
    dim: int
    labels: dict[int, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DeclaredRelation:
    """A relationship between a target and its sources."""

    target: str
    sources: list[str]
    kind: RelationKind
    transform_fn: Callable | None = None
    weight: float = 1.0


# ---------------------------------------------------------------------------
# EntangleDeclaration — the full declaration AST
# ---------------------------------------------------------------------------

@dataclass
class EntangleDeclaration:
    """Complete declaration of an entanglement."""

    name: str
    inputs: list[DeclaredInput]
    outputs: list[str]
    relations: list[DeclaredRelation]
    quality_target: QualityTarget = field(default_factory=QualityTarget)

    @property
    def input_names(self) -> list[str]:
        return [inp.name for inp in self.inputs]

    @property
    def all_names(self) -> set[str]:
        names = set(self.input_names)
        names.update(self.outputs)
        for r in self.relations:
            names.add(r.target)
            names.update(r.sources)
        return names

    def dependency_graph(self) -> dict[str, list[str]]:
        """Return adjacency list: target -> [sources]."""
        graph: dict[str, list[str]] = {}
        for name in self.all_names:
            graph[name] = []
        for r in self.relations:
            graph[r.target] = list(r.sources)
        return graph

    def topological_order(self) -> list[str]:
        """Topological sort of the dependency graph (Kahn's algorithm)."""
        graph = self.dependency_graph()

        # Build in-degree map
        in_degree: dict[str, int] = {name: 0 for name in graph}
        for target, sources in graph.items():
            for src in sources:
                if src in graph:
                    # src is depended upon by target — but in_degree tracks
                    # how many sources a target depends on
                    pass
            in_degree[target] = len(sources)

        # Seed with nodes that have no dependencies
        queue = [n for n in graph if in_degree[n] == 0]
        order: list[str] = []

        while queue:
            queue.sort()  # deterministic order
            node = queue.pop(0)
            order.append(node)
            for target, sources in graph.items():
                if node in sources:
                    in_degree[target] -= 1
                    if in_degree[target] == 0:
                        queue.append(target)

        if len(order) != len(graph):
            raise QuantumError(
                f"Cycle detected in dependency graph. "
                f"Ordered {len(order)} of {len(graph)} nodes."
            )
        return order


# ---------------------------------------------------------------------------
# DeclarationBuilder — fluent API
# ---------------------------------------------------------------------------

class DeclarationBuilder:
    """Fluent builder for constructing EntangleDeclaration."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._inputs: list[DeclaredInput] = []
        self._outputs: list[str] = []
        self._relations: list[DeclaredRelation] = []
        self._quality = QualityTarget()

    def input(self, name: str, dim: int, labels: dict[int, str] | None = None) -> DeclarationBuilder:
        self._inputs.append(DeclaredInput(name=name, dim=dim, labels=labels or {}))
        return self

    def output(self, name: str) -> DeclarationBuilder:
        self._outputs.append(name)
        return self

    def relate(
        self,
        target: str,
        sources: list[str],
        kind: RelationKind = RelationKind.PROPORTIONAL,
        transform_fn: Callable | None = None,
        weight: float = 1.0,
    ) -> DeclarationBuilder:
        self._relations.append(
            DeclaredRelation(
                target=target,
                sources=sources,
                kind=kind,
                transform_fn=transform_fn,
                weight=weight,
            )
        )
        return self

    def quality(self, omega: float = 0.9, phi: float = 0.7) -> DeclarationBuilder:
        self._quality = QualityTarget(omega=omega, phi=phi)
        return self

    def build(self) -> EntangleDeclaration:
        if not self._inputs:
            raise QuantumError("Declaration must have at least one input")
        if not self._relations:
            raise QuantumError("Declaration must have at least one relation")
        if not self._outputs:
            # Infer outputs from relation targets not used as sources
            source_set = set()
            target_set = set()
            for r in self._relations:
                target_set.add(r.target)
                source_set.update(r.sources)
            self._outputs = sorted(target_set - source_set) or sorted(target_set)

        return EntangleDeclaration(
            name=self._name,
            inputs=list(self._inputs),
            outputs=list(self._outputs),
            relations=list(self._relations),
            quality_target=self._quality,
        )
