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
    TransformOp, MergeOp, MeasureOp, CustomOp,
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
from axol.quantum.koopman import (
    lifted_dim, estimate_koopman_matrix, compose_koopman_chain,
)
from axol.quantum.unitary import (
    estimate_unitary_step, compose_unitary_chain,
    estimate_hybrid_step, compose_hybrid_chain,
)
from axol.core.types import ComplexVec, DensityMatrix


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


def _is_composable_chain(transitions: list) -> bool:
    """Check if transitions form a single composable TransformOp chain.

    Returns True if all non-MeasureOp transitions are TransformOps and
    they form a sequential chain (out_key of one feeds into key of next).
    """
    transform_ops = [t for t in transitions if isinstance(t.operation, TransformOp)]
    non_measure = [t for t in transitions if not isinstance(t.operation, MeasureOp)]

    # All non-measure ops must be TransformOps
    if len(transform_ops) != len(non_measure):
        return False
    if len(transform_ops) == 0:
        return False

    # Check chain connectivity: out_key of op[i] == key of op[i+1]
    for i in range(len(transform_ops) - 1):
        curr_out = transform_ops[i].operation.out_key or transform_ops[i].operation.key
        next_in = transform_ops[i + 1].operation.key
        if curr_out != next_in:
            return False

    return True


def _compose_chain(transitions: list) -> tuple[TransMatrix, dict]:
    """Compose a chain of TransformOps into a single matrix.

    Returns (composed_matrix, chain_info).
    Uses float64 for intermediate computation to preserve precision.
    """
    transform_ops = [t.operation for t in transitions if isinstance(t.operation, TransformOp)]

    # Multiply matrices in order: M_composed = M1 @ M2 @ ... @ MN
    # Since transform does vec @ M, chaining gives vec @ M1 @ M2 @ ... @ MN
    composed = transform_ops[0].matrix.data.astype(np.float64)
    for op in transform_ops[1:]:
        composed = composed @ op.matrix.data.astype(np.float64)

    input_key = transform_ops[0].key
    output_key = transform_ops[-1].out_key or transform_ops[-1].key

    chain_info = {
        "input_key": input_key,
        "output_key": output_key,
        "num_composed": len(transform_ops),
    }
    return TransMatrix(data=composed.astype(np.float32)), chain_info


def _has_nonlinear_step(declaration: EntangleDeclaration) -> bool:
    """Check if any relation in the declaration has a transform_fn (nonlinear)."""
    for rel in declaration.relations:
        if rel.transform_fn is not None:
            return True
    return False


def _build_step_fn(
    traj_matrix: TransMatrix,
    transform_fn: Callable | None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a single step function combining linear transform + optional nonlinearity."""
    M = traj_matrix.data.astype(np.float64)
    if transform_fn is not None:
        def step(x: np.ndarray) -> np.ndarray:
            y = x @ M
            return np.asarray(transform_fn(y), dtype=np.float64)
        return step
    else:
        def step(x: np.ndarray) -> np.ndarray:
            return x @ M
        return step


def _is_koopman_composable_chain(
    transitions: list[Transition],
    relation_map: dict[str, DeclaredRelation],
    has_nonlinear: bool,
) -> bool:
    """Check if transitions form a Koopman-composable chain.

    Requirements:
    - All non-MeasureOp, non-CustomOp transitions are TransformOps (no MergeOp)
    - CustomOps are allowed (they are injected nonlinear transform_fns)
    - Sequential chain connectivity among TransformOps
    - At least one nonlinear step exists
    - All TransformOps share the same dimension
    """
    if not has_nonlinear:
        return False

    transform_ops = [t for t in transitions if isinstance(t.operation, TransformOp)]
    # Filter out MeasureOp and CustomOp (injected nonlinear transforms)
    non_measure_non_custom = [
        t for t in transitions
        if not isinstance(t.operation, (MeasureOp, CustomOp))
    ]

    # All remaining ops must be TransformOps (no MergeOp)
    if len(transform_ops) != len(non_measure_non_custom):
        return False
    if len(transform_ops) == 0:
        return False

    # Check chain connectivity
    for i in range(len(transform_ops) - 1):
        curr_out = transform_ops[i].operation.out_key or transform_ops[i].operation.key
        next_in = transform_ops[i + 1].operation.key
        if curr_out != next_in:
            return False

    # All must have the same matrix dimension
    dims = set()
    for t in transform_ops:
        dims.add(t.operation.matrix.shape[0])
        dims.add(t.operation.matrix.shape[1])
    if len(dims) > 1:
        return False

    return True


def _compose_koopman_chain_from_transitions(
    transitions: list[Transition],
    relation_map: dict[str, DeclaredRelation],
    degree: int = 2,
    n_samples: int = 500,
    seed: int = 42,
    basis: str = "poly",
) -> tuple[TransMatrix, dict]:
    """Estimate and compose Koopman matrices for a chain of transitions.

    For each TransformOp, builds a step function (linear transform + optional
    nonlinear transform_fn), estimates its Koopman matrix via EDMD, then
    composes all Koopman matrices into a single matrix.
    """
    transform_transitions = [t for t in transitions if isinstance(t.operation, TransformOp)]

    koopman_matrices = []
    dim = transform_transitions[0].operation.matrix.shape[0]

    for i, t in enumerate(transform_transitions):
        op = t.operation
        # Find the relation that produced this transition to get its transform_fn
        target_name = op.out_key or op.key
        relation = relation_map.get(target_name)
        tfn = relation.transform_fn if relation is not None else None

        step_fn = _build_step_fn(op.matrix, tfn)
        K = estimate_koopman_matrix(
            step_fn, dim, degree=degree, n_samples=n_samples, seed=seed + i,
            basis=basis,
        )
        koopman_matrices.append(K)

    composed = compose_koopman_chain(koopman_matrices)

    input_key = transform_transitions[0].operation.key
    output_key = transform_transitions[-1].operation.out_key or transform_transitions[-1].operation.key
    ld = lifted_dim(dim, degree, basis)

    chain_info = {
        "input_key": input_key,
        "output_key": output_key,
        "num_composed": len(transform_transitions),
        "original_dim": dim,
        "lifted_dim": ld,
        "degree": degree,
        "basis": basis,
    }
    return composed, chain_info


def _compose_unitary_chain_from_transitions(
    transitions: list[Transition],
    relation_map: dict[str, DeclaredRelation],
    n_samples: int = 500,
    seed: int = 42,
) -> tuple[TransMatrix, dict]:
    """Estimate and compose unitary matrices for a chain of transitions.

    For each TransformOp, builds a step function (linear transform + optional
    nonlinear transform_fn), estimates its nearest unitary matrix, then
    composes all unitary matrices into a single unitary matrix.
    """
    transform_transitions = [t for t in transitions if isinstance(t.operation, TransformOp)]

    unitary_matrices = []
    dim = transform_transitions[0].operation.matrix.shape[0]

    for i, t in enumerate(transform_transitions):
        op = t.operation
        target_name = op.out_key or op.key
        relation = relation_map.get(target_name)
        tfn = relation.transform_fn if relation is not None else None

        step_fn = _build_step_fn(op.matrix, tfn)
        U = estimate_unitary_step(step_fn, dim, n_samples=n_samples, seed=seed + i)
        unitary_matrices.append(U)

    composed = compose_unitary_chain(unitary_matrices)

    input_key = transform_transitions[0].operation.key
    output_key = transform_transitions[-1].operation.out_key or transform_transitions[-1].operation.key

    chain_info = {
        "input_key": input_key,
        "output_key": output_key,
        "num_composed": len(transform_transitions),
        "dim": dim,
    }
    return composed, chain_info


def _compose_hybrid_chain_from_transitions(
    transitions: list[Transition],
    relation_map: dict[str, DeclaredRelation],
    n_samples: int = 500,
    seed: int = 42,
) -> tuple[TransMatrix, np.ndarray, dict]:
    """Estimate raw A matrices, compose, SVD-decompose to (rotation, scales).

    Hybrid = unitary rotation (direction) + singular values (magnitude).
    Maps to open quantum system: unitary gate + decoherence.
    """
    transform_transitions = [t for t in transitions if isinstance(t.operation, TransformOp)]

    raw_matrices = []
    dim = transform_transitions[0].operation.matrix.shape[0]

    for i, t in enumerate(transform_transitions):
        op = t.operation
        target_name = op.out_key or op.key
        relation = relation_map.get(target_name)
        tfn = relation.transform_fn if relation is not None else None

        step_fn = _build_step_fn(op.matrix, tfn)
        A = estimate_hybrid_step(step_fn, dim, n_samples=n_samples, seed=seed + i)
        raw_matrices.append(A)

    composed_matrix, rotation, scales = compose_hybrid_chain(raw_matrices)

    input_key = transform_transitions[0].operation.key
    output_key = transform_transitions[-1].operation.out_key or transform_transitions[-1].operation.key

    chain_info = {
        "input_key": input_key,
        "output_key": output_key,
        "num_composed": len(transform_transitions),
        "dim": dim,
    }
    return composed_matrix, rotation, scales, chain_info


def _build_one_hot(targets: np.ndarray, dim: int) -> np.ndarray:
    """Build one-hot matrix (N, dim) from integer label array (N,)."""
    N = len(targets)
    Y = np.zeros((N, dim), dtype=np.float64)
    for i, t in enumerate(targets):
        Y[i, int(t) % dim] = 1.0
    return Y


def _fit_readout(
    program: Program,
    composed_matrix: TransMatrix | None,
    input_key: str,
    output_key: str,
    dim: int,
    fit_data: dict,
) -> tuple[TransMatrix, dict]:
    """Fit a readout matrix W via lstsq so that argmax(|H @ W|^2) = target.

    Reservoir Computing approach:
    - composed_matrix exists → H = X @ C (fast, single matmul)
    - composed_matrix is None → run program to collect H
    - Y = one-hot(targets, dim)
    - W = lstsq(H, Y) — single shot, no epochs
    """
    X = np.asarray(fit_data["input"], dtype=np.float64)
    targets = np.asarray(fit_data["target"], dtype=np.int64)
    N = X.shape[0]
    n_classes = int(targets.max()) + 1

    # Build reservoir features H
    if composed_matrix is not None:
        C = composed_matrix.data.astype(np.float64)
        H = X @ C
    else:
        # Fallback: run the full program per sample
        from axol.core.program import run_program as _run_program

        H = np.empty((N, dim), dtype=np.float64)
        for i in range(N):
            x_data = X[i].astype(np.float32)
            x_vec = FloatVec(data=x_data)
            state = program.initial_state.copy()
            state[input_key] = x_vec
            injected = Program(
                name=program.name,
                initial_state=state,
                transitions=program.transitions,
            )
            result = _run_program(injected)
            final = result.final_state
            if output_key in final:
                H[i] = final[output_key].data.astype(np.float64)
            else:
                H[i] = x_data.astype(np.float64)

    # Target one-hot
    Y = _build_one_hot(targets, dim)

    # Solve H @ W = Y in least-squares sense
    W, _, _, _ = np.linalg.lstsq(H, Y, rcond=None)
    W = np.nan_to_num(W, nan=0.0, posinf=10.0, neginf=-10.0)

    # Training accuracy (Born rule: argmax(|H @ W|^2))
    scores = H @ W
    probs = scores ** 2
    preds = np.argmax(probs, axis=1)
    accuracy = float(np.mean(preds == targets))

    method = "reservoir" if composed_matrix is not None else "end_to_end"
    fit_info = {
        "n_samples": N,
        "n_classes": n_classes,
        "accuracy": accuracy,
        "method": method,
    }
    return TransMatrix(data=W.astype(np.float32)), fit_info


def _distill_end_to_end(
    program: Program,
    input_key: str,
    output_key: str,
    dim: int,
    n_samples: int = 200,
    seed: int = 42,
) -> tuple[TransMatrix, dict]:
    """Distill an entire program into a single dim x dim matrix via lstsq.

    Runs the full fallback program n_samples times with random inputs,
    collects (input, output) pairs, and fits a single linear map Y = X @ M.
    """
    from axol.core.program import run_program as _run_program

    rng = np.random.default_rng(seed)
    X = np.empty((n_samples, dim), dtype=np.float64)
    Y = np.empty((n_samples, dim), dtype=np.float64)

    for i in range(n_samples):
        x_data = rng.standard_normal(dim).astype(np.float32) * 0.3
        x_vec = FloatVec(data=x_data)

        # Inject input into the program
        state = program.initial_state.copy()
        state[input_key] = x_vec

        injected = Program(
            name=program.name,
            initial_state=state,
            transitions=program.transitions,
        )
        result = _run_program(injected)
        final = result.final_state

        if output_key in final:
            y_data = final[output_key].data.astype(np.float64)
        else:
            y_data = x_data.astype(np.float64)

        X[i] = x_data.astype(np.float64)
        Y[i] = y_data

    # lstsq: find M such that X @ M ≈ Y
    M, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    M = np.nan_to_num(M, nan=0.0, posinf=10.0, neginf=-10.0)

    chain_info = {
        "input_key": input_key,
        "output_key": output_key,
        "dim": dim,
    }
    return TransMatrix(data=M.astype(np.float32)), chain_info


def weave(
    declaration: EntangleDeclaration,
    encrypt: bool = False,
    seed: int = 42,
    optimize: bool = True,
    nonlinear_method: str = "distill",  # "distill" | "hybrid" | "unitary" | "koopman"
    distill_samples: int = 200,
    unitary_samples: int = 500,
    koopman_degree: int = 2,
    koopman_samples: int = 500,
    koopman_basis: str = "poly",
    quantum: bool = False,  # Enable complex amplitudes + density matrix
    fit_data: dict | None = None,  # {"input": (N, dim) ndarray, "target": (N,) int labels}
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
            # Complex amplitudes: uniform magnitude, random phases
            complex_amps = None
            if quantum:
                rng_q = np.random.default_rng(node_seed + 10000)
                phases = rng_q.uniform(0, 2 * np.pi, dim)
                magnitudes = np.ones(dim, dtype=np.float64) / np.sqrt(dim)
                complex_amps = ComplexVec.from_polar(magnitudes, phases)
            node_labels = next(
                (inp.labels for inp in declaration.inputs if inp.name == node_name),
                {},
            )
            state = SuperposedState(
                name=node_name,
                amplitudes=amplitudes,
                labels=node_labels,
                complex_amplitudes=complex_amps,
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
            complex_amps = None
            if quantum:
                rng_q = np.random.default_rng(node_seed + 10000)
                phases = rng_q.uniform(0, 2 * np.pi, work_dim)
                magnitudes = np.ones(work_dim, dtype=np.float64) / np.sqrt(work_dim)
                complex_amps = ComplexVec.from_polar(magnitudes, phases)
            state = SuperposedState(name=node_name, amplitudes=amplitudes, complex_amplitudes=complex_amps)
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
        complex_amps = None
        if quantum:
            rng_q = np.random.default_rng(node_seed + 10000)
            phases = rng_q.uniform(0, 2 * np.pi, node_dim)
            magnitudes = np.ones(node_dim, dtype=np.float64) / np.sqrt(node_dim)
            complex_amps = ComplexVec.from_polar(magnitudes, phases)
        state = SuperposedState(name=node_name, amplitudes=amplitudes, complex_amplitudes=complex_amps)

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
            # Inject CustomOp for nonlinear transform_fn (fallback path only)
            if relation.transform_fn is not None:
                _tfn = relation.transform_fn
                _node_name = node_name

                def _make_custom_fn(fn, key):
                    def custom_fn(state):
                        s = state.copy()
                        v = state[key].data.astype(np.float64)
                        result = np.asarray(fn(v), dtype=np.float64)
                        result = np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)
                        s[key] = FloatVec(data=result.astype(np.float32))
                        return s
                    return custom_fn

                transitions.append(Transition(
                    name=f"nonlinear_{node_name}",
                    operation=CustomOp(
                        fn=_make_custom_fn(_tfn, _node_name),
                        label=f"transform_fn({node_name})",
                    ),
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

    # Step 9: Compose chain if optimizable
    composed_matrix = None
    composed_chain_info = None
    koopman_matrix = None
    koopman_chain_info = None
    unitary_matrix = None
    unitary_chain_info = None
    hybrid_matrix = None
    hybrid_rotation = None
    hybrid_scales = None
    hybrid_chain_info = None
    distilled_matrix = None
    distilled_chain_info = None
    has_nonlinear = _has_nonlinear_step(declaration)

    if optimize:
        if _is_composable_chain(transitions) and not has_nonlinear:
            # Pure linear chain -> existing fast path
            composed_matrix, composed_chain_info = _compose_chain(transitions)
        elif _is_koopman_composable_chain(transitions, relation_map, has_nonlinear):
            if nonlinear_method == "distill":
                # End-to-end distillation via lstsq
                transform_ops = [t for t in transitions if isinstance(t.operation, TransformOp)]
                input_key = transform_ops[0].operation.key
                output_key = transform_ops[-1].operation.out_key or transform_ops[-1].operation.key
                dim = transform_ops[0].operation.matrix.shape[0]
                distilled_matrix, distilled_chain_info = _distill_end_to_end(
                    program, input_key, output_key, dim,
                    n_samples=distill_samples, seed=seed,
                )
            elif nonlinear_method == "hybrid":
                # Nonlinear chain -> Hybrid (unitary rotation + singular-value scales)
                hybrid_matrix, hybrid_rotation, hybrid_scales, hybrid_chain_info = (
                    _compose_hybrid_chain_from_transitions(
                        transitions, relation_map,
                        n_samples=unitary_samples,
                        seed=seed,
                    )
                )
            elif nonlinear_method == "unitary":
                # Nonlinear chain -> Pure unitary (direction only)
                unitary_matrix, unitary_chain_info = _compose_unitary_chain_from_transitions(
                    transitions, relation_map,
                    n_samples=unitary_samples,
                    seed=seed,
                )
            else:
                # Nonlinear chain -> Koopman composition (legacy)
                koopman_matrix, koopman_chain_info = _compose_koopman_chain_from_transitions(
                    transitions, relation_map,
                    degree=koopman_degree,
                    n_samples=koopman_samples,
                    seed=seed,
                    basis=koopman_basis,
                )

    # Step 9b: Fit readout from training data (Reservoir Computing)
    fit_info = None
    if fit_data is not None:
        # Determine input/output keys for fit
        transform_ops = [t for t in transitions if isinstance(t.operation, TransformOp)]
        if transform_ops:
            input_key = transform_ops[0].operation.key
            output_key = transform_ops[-1].operation.out_key or transform_ops[-1].operation.key
        else:
            input_key = declaration.input_names[0]
            output_key = declaration.outputs[0]

        readout, fit_info = _fit_readout(
            program, composed_matrix, input_key, output_key, work_dim, fit_data,
        )
        if composed_matrix is not None:
            # reservoir @ readout → single combined matrix
            final = composed_matrix.data.astype(np.float64) @ readout.data.astype(np.float64)
            composed_matrix = TransMatrix(data=final.astype(np.float32))
        else:
            # end-to-end fit: direct input → target mapping
            X = np.asarray(fit_data["input"], dtype=np.float64)
            Y = _build_one_hot(fit_data["target"], work_dim)
            M, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            composed_matrix = TransMatrix(data=np.nan_to_num(M).astype(np.float32))
            composed_chain_info = {
                "input_key": input_key,
                "output_key": output_key,
                "num_composed": 1,
            }

    # Step 10: Quantum structure (density matrix + Kraus operators)
    density_matrix = None
    kraus_operators = None

    if quantum:
        # Build global density matrix from output node's complex state
        last_output = declaration.outputs[-1] if declaration.outputs else topo_order[-1]
        if last_output in nodes and nodes[last_output].state.complex_amplitudes is not None:
            density_matrix = DensityMatrix.from_pure_state(
                nodes[last_output].state.complex_amplitudes
            )

        # Extract Kraus operators from Hybrid SVD if available
        if hybrid_matrix is not None and hybrid_scales is not None:
            from axol.quantum.density import svd_to_kraus
            try:
                U_h, s_h, Vh_h = np.linalg.svd(
                    hybrid_matrix.data.astype(np.float64), full_matrices=False
                )
                kraus_operators = svd_to_kraus(U_h, s_h, Vh_h, work_dim)
                # Evolve density matrix through the channel
                if density_matrix is not None:
                    from axol.quantum.density import apply_channel
                    density_matrix = apply_channel(density_matrix, kraus_operators)
            except Exception:
                pass  # Kraus extraction failed — proceed without

    # Step 11: Return Tapestry
    return Tapestry(
        name=declaration.name,
        nodes=nodes,
        input_names=declaration.input_names,
        output_names=declaration.outputs,
        global_attractor=global_attractor,
        weaver_report=report,
        _internal_program=program,
        _composed_matrix=composed_matrix,
        _composed_chain_info=composed_chain_info,
        _koopman_matrix=koopman_matrix,
        _koopman_chain_info=koopman_chain_info,
        _unitary_matrix=unitary_matrix,
        _unitary_chain_info=unitary_chain_info,
        _hybrid_matrix=hybrid_matrix,
        _hybrid_rotation=hybrid_rotation,
        _hybrid_scales=hybrid_scales,
        _hybrid_chain_info=hybrid_chain_info,
        _distilled_matrix=distilled_matrix,
        _distilled_chain_info=distilled_chain_info,
        _quantum=quantum,
        _density_matrix=density_matrix,
        _kraus_operators=kraus_operators,
        _fit_info=fit_info,
    )
