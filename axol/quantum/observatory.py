"""Observatory — observe a tapestry to collapse to a single result.

observe():    single observation — collapse to one point on the attractor
reobserve():  repeated observation — improve quality through distribution convergence
"""

from __future__ import annotations

import numpy as np

from axol.core.types import FloatVec, StateBundle
from axol.core.program import run_program
from axol.core import operations as ops

from axol.quantum.errors import ObservatoryError
from axol.quantum.types import Tapestry, Observation
from axol.quantum.lyapunov import omega_from_lyapunov, omega_from_observations
from axol.quantum.fractal import phi_from_fractal, phi_from_entropy


def observe(
    tapestry: Tapestry,
    inputs: dict[str, FloatVec],
    seed: int | None = None,
) -> Observation:
    """Single observation — collapse the tapestry to one point on the attractor.

    Time complexity: O(D) where D is the attractor embedding dimension.

    Steps:
    1. Inject input values into the internal Program's initial state
    2. Run the program
    3. Apply Born rule to the output vector
    4. Compute Omega from global attractor's Lyapunov exponent
    5. Compute Phi from global attractor's fractal dimension
    6. Return Observation
    """
    program = tapestry._internal_program

    # Validate inputs
    for name in tapestry.input_names:
        if name not in inputs:
            raise ObservatoryError(f"Missing input: '{name}'")

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
    output_name = tapestry.output_names[0] if tapestry.output_names else list(tapestry.nodes.keys())[-1]
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

    # Quality metrics
    attractor = tapestry.global_attractor
    omega = omega_from_lyapunov(attractor.max_lyapunov)
    phi = phi_from_fractal(attractor.fractal_dim, attractor.phase_space_dim)

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
