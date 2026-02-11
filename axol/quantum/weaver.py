"""Weaver — builds a Tapestry (strange attractor) from a declaration.

Algorithm:
  1. estimate_cost() -> cost estimate + feasibility
  2. Topological traversal of nodes
  3. Per-node attractor construction:
     - Build trajectory matrix from relation kind
     - Hadamard interference layers
     - Estimate lambda and D
  4. Compose with serial/parallel rules
  5. Assemble axol.core.Program
  6. Build Tapestry with Attractor objects + WeaverReport
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from axol.core.types import FloatVec, TransMatrix, StateBundle
from axol.core.program import (
    Program, Transition,
    TransformOp, MergeOp, MeasureOp,
)
from axol.core import operations as ops

from axol.quantum.errors import WeaverError
from axol.quantum.types import (
    SuperposedState, Attractor, TapestryNode, Tapestry, WeaverReport,
)
from axol.quantum.declare import (
    EntangleDeclaration, DeclaredRelation, RelationKind,
)
from axol.quantum.cost import estimate_cost, BASE_COST
from axol.quantum.lyapunov import estimate_lyapunov, lyapunov_spectrum, omega_from_lyapunov
from axol.quantum.fractal import estimate_fractal_dim, phi_from_fractal
from axol.quantum.compose import compose_serial


def _build_trajectory_matrix(
    dim: int,
    kind: RelationKind,
    n_sources: int,
    weight: float,
    seed: int,
) -> TransMatrix:
    """Build a trajectory matrix based on relation kind.

    Different relation kinds produce matrices with different spectral properties:
    - PROPORTIONAL: near-identity with small perturbations (convergent)
    - ADDITIVE: superposition of random rotations
    - MULTIPLICATIVE: products of random matrices (potentially chaotic)
    - INVERSE: inverse-like structure
    - CONDITIONAL: block-diagonal with random switching
    """
    rng = np.random.default_rng(seed)

    if kind == RelationKind.PROPORTIONAL:
        # Near-identity: spectral radius < 1 → convergent
        base = np.eye(dim, dtype=np.float64)
        perturbation = rng.standard_normal((dim, dim)) * 0.1 * weight
        M = base * 0.8 + perturbation
    elif kind == RelationKind.ADDITIVE:
        # Random orthogonal rotation + small contraction
        A = rng.standard_normal((dim, dim))
        Q, _ = np.linalg.qr(A)
        M = Q * (0.9 - 0.05 * n_sources)
    elif kind == RelationKind.MULTIPLICATIVE:
        # Product structure → can amplify (potentially chaotic)
        M = np.eye(dim, dtype=np.float64)
        for _ in range(n_sources):
            A = rng.standard_normal((dim, dim)) * 0.5
            M = M @ (np.eye(dim) + A * weight)
    elif kind == RelationKind.INVERSE:
        # Inverse-like: can produce large eigenvalues
        A = rng.standard_normal((dim, dim)) * 0.3 * weight
        M = np.eye(dim, dtype=np.float64) + A
        # Try to invert for inverse relationship
        try:
            M = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            M = np.eye(dim, dtype=np.float64) * 1.5
    elif kind == RelationKind.CONDITIONAL:
        # Block-diagonal: different blocks for different conditions
        M = np.zeros((dim, dim), dtype=np.float64)
        block_size = max(dim // max(n_sources, 2), 1)
        for i in range(0, dim, block_size):
            end = min(i + block_size, dim)
            size = end - i
            block = rng.standard_normal((size, size)) * 0.4
            block += np.eye(size) * 0.7
            M[i:end, i:end] = block
    else:
        M = np.eye(dim, dtype=np.float64) * 0.9

    return TransMatrix(data=M.astype(np.float32))


def _apply_interference_layers(
    matrix: TransMatrix,
    num_layers: int,
    seed: int,
) -> TransMatrix:
    """Apply Hadamard-based interference layers to enrich the attractor structure.

    Each layer: M' = H @ M @ H^T (where H is Hadamard-like) then QR-normalise.
    """
    M = matrix.data.astype(np.float64)
    n = M.shape[0]
    rng = np.random.default_rng(seed)

    for layer in range(num_layers):
        # Build a Hadamard-like interference matrix
        # For non-power-of-2 dimensions, use random orthogonal matrix
        A = rng.standard_normal((n, n))
        Q, _ = np.linalg.qr(A)

        # Interference: rotate the dynamics
        M = Q @ M @ Q.T

        # QR normalise to keep eigenvalues bounded
        Q2, R = np.linalg.qr(M)
        # Preserve the signs from R diagonal
        signs = np.sign(np.diag(R))
        signs[signs == 0] = 1.0
        M = Q2 * signs[np.newaxis, :]

        # Scale to maintain spectral properties
        spectral_radius = np.max(np.abs(np.linalg.eigvals(M)))
        if spectral_radius > 0:
            # Slightly contract to maintain stability control
            M = M * (0.95 / max(spectral_radius, 0.95))

    return TransMatrix(data=M.astype(np.float32))


def _generate_attractor_points(
    trajectory_matrix: TransMatrix,
    dim: int,
    n_points: int = 200,
    seed: int = 42,
) -> FloatVec:
    """Generate trajectory points for fractal dimension estimation."""
    M = trajectory_matrix.data.astype(np.float64)
    rng = np.random.default_rng(seed)

    x = rng.standard_normal(dim) * 0.1
    points = []

    # Discard transient
    for _ in range(50):
        x = M @ x
        norm = np.linalg.norm(x)
        if norm > 100.0:
            x = x / norm * 0.1
        elif norm < 1e-10:
            x = rng.standard_normal(dim) * 0.01

    # Collect attractor points
    for _ in range(n_points):
        x = M @ x
        norm = np.linalg.norm(x)
        if norm > 100.0:
            x = x / norm * 0.1
        elif norm < 1e-10:
            x = rng.standard_normal(dim) * 0.01
        points.extend(x.tolist())

    return FloatVec.from_list(points)


def weave(
    declaration: EntangleDeclaration,
    encrypt: bool = False,
    seed: int = 42,
    optimize: bool = True,
) -> Tapestry:
    """Weave a declaration into a Tapestry.

    The weaver:
    1. Estimates cost and feasibility
    2. Builds attractor structure per node
    3. Composes attractors along the dependency graph
    4. Assembles an internal axol.core.Program
    5. Returns a Tapestry with quality report
    """
    # Step 1: Cost estimation
    cost_est = estimate_cost(declaration)
    warnings: list[str] = []

    if not cost_est.feasible:
        warnings.append(
            f"WARNING: target Omega({declaration.quality_target.omega:.2f}) "
            f"Phi({declaration.quality_target.phi:.2f}) may not be achievable. "
            f"Maximum: Omega({cost_est.max_achievable_omega:.2f}) "
            f"Phi({cost_est.max_achievable_phi:.2f}). "
            f"Reason: {cost_est.infeasibility_reason}"
        )

    # Step 2: Topological traversal
    topo_order = declaration.topological_order()
    input_names = set(declaration.input_names)
    relation_map = {r.target: r for r in declaration.relations}

    # Determine working dimension from inputs
    input_dims = {inp.name: inp.dim for inp in declaration.inputs}
    # Use max input dim as working dimension (or default 8)
    work_dim = max(input_dims.values()) if input_dims else 8

    # Step 3: Build nodes with attractor structures
    nodes: dict[str, TapestryNode] = {}
    transitions: list[Transition] = []
    initial_vectors: dict[str, FloatVec] = {}

    # Running composition for global attractor
    global_lambda = 0.0
    global_d = 0.0
    global_omega = 1.0
    global_phi = 1.0
    node_seed = seed

    for node_name in topo_order:
        node_seed += 1

        if node_name in input_names:
            # Input nodes: trivial attractor (fixed point, lambda < 0)
            dim = input_dims[node_name]
            amplitudes = FloatVec(data=np.ones(dim, dtype=np.float32) / np.sqrt(dim))
            state = SuperposedState(
                name=node_name,
                amplitudes=amplitudes,
                labels=next(
                    (inp.labels for inp in declaration.inputs if inp.name == node_name),
                    {},
                ),
            )
            traj = TransMatrix(data=np.eye(dim, dtype=np.float32) * 0.5)
            attractor = Attractor(
                phase_space_dim=dim,
                embedding_dim=dim,
                fractal_dim=0.0,
                lyapunov_spectrum=[-1.0] * dim,
                max_lyapunov=-1.0,
                basin_bounds=(-1.0, 1.0),
                trajectory_matrix=traj,
            )
            nodes[node_name] = TapestryNode(
                name=node_name, state=state, attractor=attractor, depth=0,
            )
            initial_vectors[node_name] = amplitudes
            continue

        # Non-input node: build attractor from relation
        relation = relation_map.get(node_name)
        if relation is None:
            # Isolated node — treat as identity
            amplitudes = FloatVec(data=np.ones(work_dim, dtype=np.float32) / np.sqrt(work_dim))
            state = SuperposedState(name=node_name, amplitudes=amplitudes)
            traj = TransMatrix(data=np.eye(work_dim, dtype=np.float32) * 0.5)
            attractor = Attractor(
                phase_space_dim=work_dim,
                embedding_dim=work_dim,
                fractal_dim=0.0,
                lyapunov_spectrum=[-1.0],
                max_lyapunov=-1.0,
                basin_bounds=(-1.0, 1.0),
                trajectory_matrix=traj,
            )
            nodes[node_name] = TapestryNode(
                name=node_name, state=state, attractor=attractor, depth=0,
            )
            initial_vectors[node_name] = amplitudes
            continue

        # Determine node dimension
        source_dims = []
        for src in relation.sources:
            if src in nodes:
                source_dims.append(nodes[src].state.dim)
            elif src in input_dims:
                source_dims.append(input_dims[src])
            else:
                source_dims.append(work_dim)
        node_dim = max(source_dims) if source_dims else work_dim

        # Budget allocation: proportional to cost estimate
        node_budget = cost_est.per_node_cost.get(node_name, BASE_COST)
        num_layers = max(1, int(math.ceil(math.log2(max(node_budget / BASE_COST, 1.0) + 1))))

        # Build trajectory matrix based on relation kind
        traj_matrix = _build_trajectory_matrix(
            dim=node_dim,
            kind=relation.kind,
            n_sources=len(relation.sources),
            weight=relation.weight,
            seed=node_seed,
        )

        # Apply interference layers
        traj_matrix = _apply_interference_layers(traj_matrix, num_layers, seed=node_seed + 1000)

        # Estimate Lyapunov exponent
        node_lambda = estimate_lyapunov(traj_matrix, steps=50)
        node_spectrum = lyapunov_spectrum(traj_matrix, dim=min(node_dim, 4), steps=50)

        # Generate attractor points and estimate fractal dimension
        attractor_pts = _generate_attractor_points(traj_matrix, node_dim, n_points=100, seed=node_seed)
        node_d = estimate_fractal_dim(attractor_pts, phase_space_dim=node_dim)

        # Build attractor
        attractor = Attractor(
            phase_space_dim=node_dim,
            embedding_dim=node_dim,
            fractal_dim=node_d,
            lyapunov_spectrum=node_spectrum,
            max_lyapunov=node_lambda,
            basin_bounds=(-2.0, 2.0),
            trajectory_matrix=traj_matrix,
        )

        # Build superposed state (initial amplitudes)
        amplitudes = FloatVec(data=np.ones(node_dim, dtype=np.float32) / np.sqrt(node_dim))
        state = SuperposedState(name=node_name, amplitudes=amplitudes)

        # Depth = max source depth + 1
        depth = 0
        incoming_edges: list[tuple[str, TransMatrix]] = []
        for src in relation.sources:
            if src in nodes:
                depth = max(depth, nodes[src].depth + 1)
                incoming_edges.append((src, traj_matrix))

        nodes[node_name] = TapestryNode(
            name=node_name,
            state=state,
            attractor=attractor,
            incoming_edges=incoming_edges,
            allocated_cost=node_budget,
            depth=depth,
        )
        initial_vectors[node_name] = amplitudes

        # Build transitions for the internal program
        if len(relation.sources) == 1:
            # Single source → TransformOp
            src = relation.sources[0]
            transitions.append(Transition(
                name=f"weave_{src}_to_{node_name}",
                operation=TransformOp(key=src, matrix=traj_matrix, out_key=node_name),
            ))
        else:
            # Multiple sources → TransformOp each, then MergeOp
            intermediate_keys = []
            for i, src in enumerate(relation.sources):
                inter_key = f"_inter_{node_name}_{src}"
                transitions.append(Transition(
                    name=f"weave_{src}_to_{inter_key}",
                    operation=TransformOp(key=src, matrix=traj_matrix, out_key=inter_key),
                ))
                intermediate_keys.append(inter_key)
                initial_vectors[inter_key] = FloatVec.zeros(node_dim)

            # Merge intermediates
            n_merge = len(intermediate_keys)
            weights = FloatVec.from_list([1.0 / n_merge] * n_merge)
            transitions.append(Transition(
                name=f"merge_to_{node_name}",
                operation=MergeOp(keys=intermediate_keys, weights=weights, out_key=node_name),
            ))

        # Compose into global metrics (serial for dependency chain)
        global_omega, global_phi, global_lambda, global_d = compose_serial(
            global_omega, global_phi, global_lambda, global_d,
            attractor.omega, attractor.phi, node_lambda, node_d,
        )

    # Step 6: Assemble internal Program
    initial_state = StateBundle(vectors={k: v for k, v in initial_vectors.items()})

    # Add measure ops for outputs
    for out_name in declaration.outputs:
        if out_name in nodes:
            transitions.append(Transition(
                name=f"measure_{out_name}",
                operation=MeasureOp(key=out_name, out_key=f"_prob_{out_name}"),
            ))
            initial_vectors[f"_prob_{out_name}"] = FloatVec.zeros(
                nodes[out_name].state.dim
            )

    initial_state = StateBundle(vectors={k: v for k, v in initial_vectors.items()})

    program = Program(
        name=f"tapestry_{declaration.name}",
        initial_state=initial_state,
        transitions=transitions,
    )

    # Step 7: Build global attractor
    # Use the trajectory matrix of the last output node for the global attractor
    last_output = declaration.outputs[-1] if declaration.outputs else topo_order[-1]
    global_traj = (
        nodes[last_output].attractor.trajectory_matrix
        if last_output in nodes
        else TransMatrix(data=np.eye(work_dim, dtype=np.float32))
    )

    global_attractor = Attractor(
        phase_space_dim=work_dim,
        embedding_dim=work_dim,
        fractal_dim=max(global_d, 0.0),
        lyapunov_spectrum=[global_lambda],
        max_lyapunov=global_lambda,
        basin_bounds=(-2.0, 2.0),
        trajectory_matrix=global_traj,
    )

    # Step 8: WeaverReport
    report = WeaverReport(
        target_omega=declaration.quality_target.omega,
        target_phi=declaration.quality_target.phi,
        estimated_omega=omega_from_lyapunov(global_lambda),
        estimated_phi=phi_from_fractal(global_d, work_dim),
        max_lyapunov=global_lambda,
        fractal_dim=global_d,
        total_cost=cost_est.total_cost,
        cost_breakdown=cost_est.per_node_cost,
        warnings=warnings,
        feasible=cost_est.feasible,
    )

    # Step 9: Return Tapestry
    return Tapestry(
        name=declaration.name,
        nodes=nodes,
        input_names=declaration.input_names,
        output_names=declaration.outputs,
        global_attractor=global_attractor,
        weaver_report=report,
        _internal_program=program,
    )
