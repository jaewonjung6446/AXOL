"""Entanglement cost estimation.

E = sum_path [ iterations_to_converge(path) * path_complexity(path) ]
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from axol.core.types import TransMatrix
from axol.quantum.declare import EntangleDeclaration, RelationKind
from axol.quantum.lyapunov import estimate_lyapunov, omega_from_lyapunov
from axol.quantum.fractal import phi_from_fractal


BASE_COST = 10.0


@dataclass(frozen=True)
class CostEstimate:
    """Result of cost estimation for an entanglement declaration."""

    total_cost: float
    per_node_cost: dict[str, float]
    critical_path: list[str]
    max_achievable_omega: float
    max_achievable_phi: float
    feasible: bool
    infeasibility_reason: str | None = None


def _relation_complexity(kind: RelationKind) -> float:
    """Complexity multiplier per relation kind."""
    return {
        RelationKind.PROPORTIONAL: 1.0,
        RelationKind.ADDITIVE: 1.2,
        RelationKind.MULTIPLICATIVE: 1.5,
        RelationKind.INVERSE: 2.0,
        RelationKind.CONDITIONAL: 2.5,
    }[kind]


def _estimate_path_lyapunov(declaration: EntangleDeclaration, path: list[str]) -> float:
    """Estimate the aggregate Lyapunov exponent along a dependency path.

    Each relation adds to the effective lambda based on its complexity.
    """
    total_lambda = 0.0
    relation_map = {r.target: r for r in declaration.relations}

    for node in path:
        if node in relation_map:
            r = relation_map[node]
            # More complex relations tend to be more chaotic
            complexity = _relation_complexity(r.kind)
            # Number of sources adds to instability
            source_factor = math.log2(max(len(r.sources), 1) + 1)
            # Base contribution: mildly convergent for simple, potentially chaotic for complex
            node_lambda = (complexity * source_factor - 1.5) * 0.3
            total_lambda += node_lambda

    return total_lambda


def _find_critical_path(declaration: EntangleDeclaration) -> list[str]:
    """Find the longest dependency path (critical path)."""
    graph = declaration.dependency_graph()
    topo = declaration.topological_order()

    # Dynamic programming for longest path
    dist: dict[str, float] = {n: 0.0 for n in topo}
    prev: dict[str, str | None] = {n: None for n in topo}

    relation_map = {r.target: r for r in declaration.relations}

    for node in topo:
        for target, sources in graph.items():
            if node in sources:
                w = _relation_complexity(relation_map[target].kind) if target in relation_map else 1.0
                if dist[node] + w > dist[target]:
                    dist[target] = dist[node] + w
                    prev[target] = node

    # Find the node with maximum distance
    end_node = max(dist, key=lambda n: dist[n])
    path = []
    current: str | None = end_node
    while current is not None:
        path.append(current)
        current = prev[current]
    path.reverse()
    return path


def estimate_cost(declaration: EntangleDeclaration) -> CostEstimate:
    """Estimate the entanglement cost for a declaration.

    Cost model:
      E = sum_path [iterations_to_converge * path_complexity]

    Also determines whether the quality target is achievable.
    """
    topo = declaration.topological_order()
    relation_map = {r.target: r for r in declaration.relations}
    input_names = set(declaration.input_names)

    # Per-node cost
    per_node: dict[str, float] = {}
    for node in topo:
        if node in input_names:
            per_node[node] = 0.0
            continue

        if node in relation_map:
            r = relation_map[node]
            complexity = _relation_complexity(r.kind)
            n_sources = len(r.sources)
            # Cost = base * complexity * log(sources+1) * weight
            cost = BASE_COST * complexity * math.log2(n_sources + 1) * r.weight
            per_node[node] = cost
        else:
            per_node[node] = BASE_COST

    total_cost = sum(per_node.values())

    # Critical path
    critical_path = _find_critical_path(declaration)
    path_lambda = _estimate_path_lyapunov(declaration, critical_path)

    # Max achievable quality from chaos metrics
    max_omega = omega_from_lyapunov(path_lambda)

    # Estimate fractal dimension from graph structure
    n_relations = len(declaration.relations)
    n_nodes = len(topo)
    # Heuristic: fractal dim ~ number of branching relations / total nodes
    estimated_d = min(n_relations * 0.5, n_nodes)
    max_phi = phi_from_fractal(estimated_d, n_nodes)

    # Feasibility check
    target = declaration.quality_target
    feasible = (target.omega <= max_omega + 0.01) and (target.phi <= max_phi + 0.01)

    reason = None
    if not feasible:
        parts = []
        if target.omega > max_omega + 0.01:
            parts.append(
                f"Omega target {target.omega:.2f} exceeds maximum achievable "
                f"{max_omega:.2f} (lambda={path_lambda:.2f} on path: "
                f"{'->'.join(critical_path)})"
            )
        if target.phi > max_phi + 0.01:
            parts.append(
                f"Phi target {target.phi:.2f} exceeds maximum achievable "
                f"{max_phi:.2f} (D_estimated={estimated_d:.2f})"
            )
        reason = "; ".join(parts)

    return CostEstimate(
        total_cost=total_cost,
        per_node_cost=per_node,
        critical_path=critical_path,
        max_achievable_omega=max_omega,
        max_achievable_phi=max_phi,
        feasible=feasible,
        infeasibility_reason=reason,
    )
