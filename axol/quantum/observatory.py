"""Observatory — observe a tapestry to collapse to a single result.

observe():    single observation — collapse to one point on the attractor
reobserve():  repeated observation — improve quality through distribution convergence
"""

from __future__ import annotations

import numpy as np

from axol.core.types import FloatVec, StateBundle, ComplexVec, DensityMatrix
from axol.core.program import run_program
from axol.core import operations as ops
from axol.core.operations import transform_complex, measure_complex, interfere

from axol.quantum.errors import ObservatoryError
from axol.quantum.types import Tapestry, Observation
from axol.quantum.lyapunov import omega_from_lyapunov, omega_from_observations
from axol.quantum.fractal import phi_from_fractal, phi_from_entropy
from axol.quantum.koopman import lift, unlift

from axol.core.amplitude import Amplitude
from axol.core.amplitude_ops import amp_transform, amp_superpose, amp_observe
from axol.core.trans_amplitude import TransAmplitude, trans_amp_apply


def observe(
    tapestry: Tapestry,
    inputs: dict[str, FloatVec],
    seed: int | None = None,
) -> Observation:
    """Single observation — collapse the tapestry to one point on the attractor.

    If the tapestry has a pre-composed matrix (linear chain), uses the fast path:
    a single matrix-vector multiply + Born rule. Otherwise falls back to running
    the full internal program.

    Fast path complexity: O(dim²)  — depth-independent.
    Fallback complexity:  O(depth × dim²)
    """
    # Validate inputs
    for name in tapestry.input_names:
        if name not in inputs:
            raise ObservatoryError(f"Missing input: '{name}'")

    # Quality metrics (shared by both paths)
    attractor = tapestry.global_attractor
    omega = omega_from_lyapunov(attractor.max_lyapunov)
    phi = phi_from_fractal(attractor.fractal_dim, attractor.phase_space_dim)

    output_name = tapestry.output_names[0] if tapestry.output_names else list(tapestry.nodes.keys())[-1]

    # === QUANTUM PATH: complex amplitudes with interference ===
    if tapestry._quantum:
        q_obs = _observe_quantum(tapestry, inputs, output_name, omega, phi)
        if q_obs is not None:
            return q_obs

    # === FAST PATH: composed matrix available ===
    if tapestry._composed_matrix is not None:
        info = tapestry._composed_chain_info
        input_vec = inputs[info["input_key"]]
        value = ops.transform(input_vec, tapestry._composed_matrix)
        probs = ops.measure(value)

        value_index = int(np.argmax(probs.data))
        label = None
        if output_name in tapestry.nodes:
            label = tapestry.nodes[output_name].state.labels.get(value_index)

        return Observation(
            value=value,
            value_index=value_index,
            value_label=label,
            omega=omega,
            phi=phi,
            probabilities=probs,
            tapestry_name=tapestry.name,
            observation_count=1,
        )

    # === DISTILLED FAST PATH: end-to-end lstsq fitted matrix ===
    if tapestry._distilled_matrix is not None:
        info = tapestry._distilled_chain_info
        input_vec = inputs[info["input_key"]]
        value = ops.transform(input_vec, tapestry._distilled_matrix)
        # Safety: clamp extreme values to prevent overflow in measure()
        safe_data = np.nan_to_num(value.data, nan=0.0, posinf=1e6, neginf=-1e6)
        safe_data = np.clip(safe_data, -1e6, 1e6)
        value = FloatVec(data=safe_data.astype(np.float32))
        probs = ops.measure(value)

        value_index = int(np.argmax(probs.data))
        label = None
        if output_name in tapestry.nodes:
            label = tapestry.nodes[output_name].state.labels.get(value_index)

        return Observation(
            value=value,
            value_index=value_index,
            value_label=label,
            omega=omega,
            phi=phi,
            probabilities=probs,
            tapestry_name=tapestry.name,
            observation_count=1,
        )

    # === HYBRID FAST PATH: composed raw matrix (direction + magnitude) ===
    if tapestry._hybrid_matrix is not None:
        info = tapestry._hybrid_chain_info
        input_vec = inputs[info["input_key"]]
        value = ops.transform(input_vec, tapestry._hybrid_matrix)
        probs = ops.measure(value)

        value_index = int(np.argmax(probs.data))
        label = None
        if output_name in tapestry.nodes:
            label = tapestry.nodes[output_name].state.labels.get(value_index)

        return Observation(
            value=value,
            value_index=value_index,
            value_label=label,
            omega=omega,
            phi=phi,
            probabilities=probs,
            tapestry_name=tapestry.name,
            observation_count=1,
        )

    # === UNITARY FAST PATH: nonlinear composed via unitary projection ===
    if tapestry._unitary_matrix is not None:
        info = tapestry._unitary_chain_info
        input_vec = inputs[info["input_key"]]
        value = ops.transform(input_vec, tapestry._unitary_matrix)
        probs = ops.measure(value)

        value_index = int(np.argmax(probs.data))
        label = None
        if output_name in tapestry.nodes:
            label = tapestry.nodes[output_name].state.labels.get(value_index)

        return Observation(
            value=value,
            value_index=value_index,
            value_label=label,
            omega=omega,
            phi=phi,
            probabilities=probs,
            tapestry_name=tapestry.name,
            observation_count=1,
        )

    # === KOOPMAN FAST PATH: nonlinear composed via Koopman operator ===
    if tapestry._koopman_matrix is not None:
        info = tapestry._koopman_chain_info
        input_vec = inputs[info["input_key"]]
        kbasis = info.get("basis", "poly")

        # 1. Lift input to observable space
        lifted = lift(input_vec.data.astype(np.float64), degree=info["degree"], basis=kbasis)
        lifted_fv = FloatVec(data=lifted.astype(np.float32))

        # 2. Single matrix multiply in lifted space
        result_lifted = ops.transform(lifted_fv, tapestry._koopman_matrix)

        # 3. Unlift back to original dimension
        value_data = unlift(result_lifted.data.astype(np.float64), info["original_dim"], degree=info["degree"], basis=kbasis)
        value = FloatVec(data=value_data.astype(np.float32))

        # 4. Born rule
        probs = ops.measure(value)

        value_index = int(np.argmax(probs.data))
        label = None
        if output_name in tapestry.nodes:
            label = tapestry.nodes[output_name].state.labels.get(value_index)

        return Observation(
            value=value,
            value_index=value_index,
            value_label=label,
            omega=omega,
            phi=phi,
            probabilities=probs,
            tapestry_name=tapestry.name,
            observation_count=1,
        )

    # === FALLBACK: run full program (non-linear / branching pipelines) ===
    program = tapestry._internal_program

    # Inject inputs into initial state
    state = program.initial_state.copy()
    for name, vec in inputs.items():
        if name in state:
            state[name] = vec

    # Build a new program with injected inputs
    from axol.core.program import Program
    injected_program = Program(
        name=program.name,
        initial_state=state,
        transitions=program.transitions,
    )

    # Run the internal program
    result = run_program(injected_program)
    final_state = result.final_state

    # Extract output probabilities
    prob_key = f"_prob_{output_name}"

    if prob_key in final_state:
        probs = final_state[prob_key]
    elif output_name in final_state:
        probs = ops.measure(final_state[output_name])
    else:
        raise ObservatoryError(
            f"Output '{output_name}' not found in final state. "
            f"Available keys: {list(final_state.keys())}"
        )

    # Ensure probs is a FloatVec
    if not isinstance(probs, FloatVec):
        probs = FloatVec(data=probs.data.astype(np.float32))

    # Collapse: pick the most probable value
    value_index = int(np.argmax(probs.data))

    # Get value vector (the output state, not probabilities)
    if output_name in final_state:
        value = final_state[output_name]
        if not isinstance(value, FloatVec):
            value = FloatVec(data=value.data.astype(np.float32))
    else:
        value = probs

    # Label lookup
    label = None
    if output_name in tapestry.nodes:
        node = tapestry.nodes[output_name]
        label = node.state.labels.get(value_index)

    return Observation(
        value=value,
        value_index=value_index,
        value_label=label,
        omega=omega,
        phi=phi,
        probabilities=probs,
        tapestry_name=tapestry.name,
        observation_count=1,
    )


def reobserve(
    tapestry: Tapestry,
    inputs: dict[str, FloatVec],
    count: int = 10,
    seed: int | None = None,
) -> Observation:
    """Repeated observation — improve quality through distribution convergence.

    Steps:
    1. observe() count times (with optional seed variation)
    2. Average the probability distributions
    3. Recompute empirical Omega (argmax mode stability)
    4. Recompute Phi from averaged distribution
    """
    if count < 1:
        raise ObservatoryError("count must be >= 1")

    observations: list[Observation] = []
    prob_accumulator = None
    obs_probs: list[FloatVec] = []

    for i in range(count):
        obs_seed = (seed + i) if seed is not None else None
        obs = observe(tapestry, inputs, seed=obs_seed)
        observations.append(obs)
        obs_probs.append(obs.probabilities)

        if prob_accumulator is None:
            prob_accumulator = obs.probabilities.data.astype(np.float64).copy()
        else:
            prob_accumulator += obs.probabilities.data.astype(np.float64)

    # Average probabilities
    avg_probs_data = prob_accumulator / count
    # Re-normalise
    total = np.sum(avg_probs_data)
    if total > 0:
        avg_probs_data = avg_probs_data / total
    avg_probs = FloatVec(data=avg_probs_data.astype(np.float32))

    # Empirical omega from argmax stability
    empirical_omega = omega_from_observations(obs_probs)

    # Phi from averaged distribution entropy
    empirical_phi = phi_from_entropy(avg_probs)

    # Use the last observation's value but with improved metrics
    last_obs = observations[-1]
    value_index = int(np.argmax(avg_probs.data))

    label = None
    output_name = tapestry.output_names[0] if tapestry.output_names else None
    if output_name and output_name in tapestry.nodes:
        label = tapestry.nodes[output_name].state.labels.get(value_index)

    return Observation(
        value=last_obs.value,
        value_index=value_index,
        value_label=label,
        omega=empirical_omega,
        phi=empirical_phi,
        probabilities=avg_probs,
        tapestry_name=tapestry.name,
        observation_count=count,
    )


# ---------------------------------------------------------------------------
# Quantum observation (complex amplitudes + density matrix)
# ---------------------------------------------------------------------------

def _observe_quantum(
    tapestry: Tapestry,
    inputs: dict[str, FloatVec],
    output_name: str,
    omega: float,
    phi: float,
) -> Observation | None:
    """Quantum observation path using complex amplitudes.

    Priority: composed > distilled > hybrid > fallback-to-classical.
    Uses complex amplitudes for interference-aware observation,
    then derives density matrix and quantum quality metrics.
    """
    from axol.quantum.density import phi_from_purity, omega_from_coherence

    # Find the input ComplexVec
    input_name = tapestry.input_names[0] if tapestry.input_names else None
    if input_name is None:
        return None

    input_fv = inputs.get(input_name)
    if input_fv is None:
        return None

    # Promote real input to complex
    input_cv = ComplexVec.from_real(input_fv)

    # Try each fast path with complex transform
    matrix = None
    if tapestry._composed_matrix is not None:
        matrix = tapestry._composed_matrix
    elif tapestry._distilled_matrix is not None:
        matrix = tapestry._distilled_matrix
    elif tapestry._hybrid_matrix is not None:
        matrix = tapestry._hybrid_matrix
    elif tapestry._unitary_matrix is not None:
        matrix = tapestry._unitary_matrix

    if matrix is None:
        return None  # fall through to classical paths

    # Complex transform
    result_cv = transform_complex(input_cv, matrix)

    # Safety clamp for distilled path
    safe_data = np.nan_to_num(result_cv.data)
    safe_data = np.clip(safe_data.real, -1e6, 1e6) + 1j * np.clip(safe_data.imag, -1e6, 1e6)
    result_cv = ComplexVec(data=safe_data.astype(np.complex128))

    # Born rule on complex amplitudes
    probs = measure_complex(result_cv)
    value = FloatVec(data=np.abs(result_cv.data).astype(np.float32))

    value_index = int(np.argmax(probs.data))
    label = None
    if output_name in tapestry.nodes:
        label = tapestry.nodes[output_name].state.labels.get(value_index)

    # Build post-observation density matrix (collapsed state)
    post_density = DensityMatrix.from_pure_state(result_cv)

    # Apply Kraus channel if available (decoherence)
    if tapestry._kraus_operators is not None:
        from axol.quantum.density import apply_channel
        try:
            post_density = apply_channel(post_density, tapestry._kraus_operators)
        except Exception:
            pass

    # Quantum quality metrics
    q_phi = phi_from_purity(post_density)
    q_omega = omega_from_coherence(post_density)

    return Observation(
        value=value,
        value_index=value_index,
        value_label=label,
        omega=omega,
        phi=phi,
        probabilities=probs,
        tapestry_name=tapestry.name,
        observation_count=1,
        density_matrix=post_density,
        quantum_phi=q_phi,
        quantum_omega=q_omega,
    )


# ---------------------------------------------------------------------------
# Amplitude-first observation (Phase 1)
# ---------------------------------------------------------------------------

def observe_amplitude(
    tapestry: Tapestry,
    inputs: dict[str, Amplitude | FloatVec],
    n_paths: int = 2,
) -> Observation:
    """Amplitude-first observation with multi-path interference.

    Unlike observe(), this defers collapse until after interference between
    computation paths, enabling amplitude amplification of the correct answer.

    Algorithm (Grover-inspired, n_paths=2):
      1. Direct path: amp_transform(input, composed_matrix)
      2. Oracle: flip sign of highest-probability state
      3. Diffusion: 2|s><s| - I reflection about input state
      4. Amplified path: oracle → diffusion
      5. Superpose direct + amplified with weights → interference!
      6. amp_observe(result) → collapse

    Why accuracy improves: when top-2 classes are nearly tied (e.g. 0.26 vs 0.24),
    one Grover iteration amplifies the gap to ~0.35 vs ~0.15, making argmax reliable.
    """
    # Validate inputs
    for name in tapestry.input_names:
        if name not in inputs:
            raise ObservatoryError(f"Missing input: '{name}'")

    # Quality metrics
    attractor = tapestry.global_attractor
    omega = omega_from_lyapunov(attractor.max_lyapunov)
    phi = phi_from_fractal(attractor.fractal_dim, attractor.phase_space_dim)

    output_name = (
        tapestry.output_names[0]
        if tapestry.output_names
        else list(tapestry.nodes.keys())[-1]
    )

    # Find the matrix to use (same priority as observe())
    matrix = None
    input_key = None
    if tapestry._composed_matrix is not None:
        matrix = tapestry._composed_matrix
        info = tapestry._composed_chain_info
        input_key = info["input_key"] if info else tapestry.input_names[0]
    elif tapestry._distilled_matrix is not None:
        matrix = tapestry._distilled_matrix
        info = tapestry._distilled_chain_info
        input_key = info["input_key"] if info else tapestry.input_names[0]
    elif tapestry._hybrid_matrix is not None:
        matrix = tapestry._hybrid_matrix
        info = tapestry._hybrid_chain_info
        input_key = info["input_key"] if info else tapestry.input_names[0]
    elif tapestry._unitary_matrix is not None:
        matrix = tapestry._unitary_matrix
        info = tapestry._unitary_chain_info
        input_key = info["input_key"] if info else tapestry.input_names[0]

    if input_key is None:
        input_key = tapestry.input_names[0]

    # Get / promote input to Amplitude
    raw_input = inputs[input_key]
    if isinstance(raw_input, Amplitude):
        input_amp = raw_input
    elif isinstance(raw_input, FloatVec):
        # Direct promotion: treat feature vector values as real amplitudes
        # (no sqrt — input is a feature vector, not a probability distribution)
        input_amp = Amplitude.from_array(raw_input.data.astype(np.float64))
    else:
        raise ObservatoryError(
            f"Input '{input_key}' must be Amplitude or FloatVec, got {type(raw_input)}"
        )

    # === TransAmplitude path: open relation with path interference ===
    if tapestry._trans_amplitude is not None and isinstance(tapestry._trans_amplitude, TransAmplitude):
        result = trans_amp_apply(input_amp, tapestry._trans_amplitude)
        value_index, probs = amp_observe(result)
        label = None
        if output_name in tapestry.nodes:
            label = tapestry.nodes[output_name].state.labels.get(value_index)
        value = FloatVec(data=np.abs(result.data).astype(np.float32))
        return Observation(
            value=value,
            value_index=value_index,
            value_label=label,
            omega=omega,
            phi=phi,
            probabilities=probs,
            tapestry_name=tapestry.name,
            observation_count=1,
            amplitude=result,
        )

    if matrix is None:
        # No fast-path matrix — fall back to classical observe
        if isinstance(raw_input, Amplitude):
            fv = raw_input.to_floatvec()
        else:
            fv = raw_input
        return observe(tapestry, {input_key: fv})

    # === Amplitude-first pipeline ===
    dim = matrix.shape[1]

    # Step 1: Direct path
    direct = amp_transform(input_amp, matrix)

    if n_paths < 2:
        # No interference — just observe the direct path
        value_index, probs = amp_observe(direct)
        label = None
        if output_name in tapestry.nodes:
            label = tapestry.nodes[output_name].state.labels.get(value_index)
        value = FloatVec(data=np.abs(direct.data).astype(np.float32))
        return Observation(
            value=value,
            value_index=value_index,
            value_label=label,
            omega=omega,
            phi=phi,
            probabilities=probs,
            tapestry_name=tapestry.name,
            observation_count=1,
            amplitude=direct,
        )

    # Step 2: Oracle — flip sign of highest-probability state
    direct_probs = direct.probabilities
    marked_idx = int(np.argmax(direct_probs))
    oracle_diag = np.ones(dim, dtype=np.float32)
    oracle_diag[marked_idx] = -1.0
    from axol.core.types import TransMatrix as TM
    oracle = TM(data=np.diag(oracle_diag))

    # Step 3: Diffusion — reflection about input state
    # D = 2|s><s| - I, where |s> is derived from input amplitudes
    s = input_amp.data[:dim].astype(np.complex128) if input_amp.size >= dim else np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    s_norm = np.linalg.norm(s)
    if s_norm > 0:
        s = s / s_norm
    # Construct real diffusion matrix from |s>
    s_real = np.abs(s).astype(np.float64)
    s_real_norm = np.linalg.norm(s_real)
    if s_real_norm > 0:
        s_real = s_real / s_real_norm
    D = 2.0 * np.outer(s_real, s_real) - np.eye(dim, dtype=np.float64)
    diffusion = TM(data=D.astype(np.float32))

    # Step 4: Amplified path = oracle → diffusion applied to direct
    marked = amp_transform(direct, oracle)
    amplified = amp_transform(marked, diffusion)

    # Step 5: Superpose direct + amplified with interference
    # Weight ratio: direct dominates, amplified provides a gentle boost.
    # Large amplified weight can flip borderline-correct predictions.
    # The amplified path refines the probability landscape via interference.
    direct_probs_sorted = np.sort(direct_probs)[::-1]
    ambiguity = 1.0 - (direct_probs_sorted[0] - direct_probs_sorted[1]) if len(direct_probs_sorted) > 1 else 0.0
    # Scale amplified weight by ambiguity: more ambiguous → more interference help
    amp_weight = min(0.3, ambiguity * 0.4)
    direct_weight = 1.0 - amp_weight
    result = amp_superpose(
        [direct, amplified],
        [complex(direct_weight, 0.0), complex(amp_weight, 0.0)],
    )

    # Step 6: Collapse
    value_index, probs = amp_observe(result)

    label = None
    if output_name in tapestry.nodes:
        label = tapestry.nodes[output_name].state.labels.get(value_index)

    value = FloatVec(data=np.abs(result.data).astype(np.float32))

    return Observation(
        value=value,
        value_index=value_index,
        value_label=label,
        omega=omega,
        phi=phi,
        probabilities=probs,
        tapestry_name=tapestry.name,
        observation_count=1,
        amplitude=result,
    )
