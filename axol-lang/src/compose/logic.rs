//! Logic gates — AND, OR, NOT, IF-THEN-ELSE using Wave variables.
//!
//! Boolean encoding: FALSE = [0.9, 0.1], TRUE = [0.1, 0.9]
//! Gates compose Waves using interference patterns instead of
//! direct amplitude manipulation.

use crate::types::*;
use crate::ops;
use crate::wave::{Wave, InterferencePattern};
use crate::declare::*;
use crate::weaver;
use crate::observatory::{self, Observation};
use crate::errors::{AxolError, Result};

/// Boolean encoding constants (dim=2).
pub const FALSE_VEC: [f32; 2] = [0.9, 0.1];
pub const TRUE_VEC: [f32; 2] = [0.1, 0.9];

/// Encode a boolean into a FloatVec.
pub fn encode_bool(value: bool) -> FloatVec {
    if value {
        FloatVec::new(TRUE_VEC.to_vec())
    } else {
        FloatVec::new(FALSE_VEC.to_vec())
    }
}

/// Encode a boolean into a Wave.
pub fn encode_bool_wave(value: bool) -> Wave {
    Wave::from_classical(&encode_bool(value))
}

/// Decode a Wave to a boolean (dominant index == 1 → TRUE).
pub fn decode_bool_wave(wave: &Wave) -> bool {
    wave.dominant() == 1
}

/// Decode an Observation back to a boolean.
pub fn decode_bool(obs: &Observation) -> bool {
    obs.value_index == 1
}

/// NOT gate: swap amplitudes (destructive interference with self-inverse).
pub fn gate_not(input: &FloatVec) -> Result<Observation> {
    if input.dim() != 2 {
        return Err(AxolError::Compose(format!(
            "NOT gate requires dim=2, got {}", input.dim()
        )));
    }

    // Swap matrix
    let swap = TransMatrix::new(vec![0.0, 1.0, 1.0, 0.0], 2, 2);
    let wave = Wave::from_classical(input).transform(&swap)?;
    let (value_index, collapsed) = wave.observe();
    let probs = FloatVec::new(wave.probabilities().iter().map(|&p| p as f32).collect());

    Ok(Observation {
        value: wave.to_float_vec(),
        value_index,
        value_label: None,
        omega: 1.0,
        phi: 1.0,
        probabilities: probs,
        tapestry_name: "NOT".to_string(),
        observation_count: 1,
        density_matrix: None,
        quantum_phi: None,
        quantum_omega: None,
        n_basins: None,
        chosen_basin: None,
        basin_weights: None,
        collapse_metrics: Some(collapsed.metrics),
        wave: Some(wave),
    })
}

/// NOT gate returning Wave (no collapse).
pub fn wave_not(input: &Wave) -> Result<Wave> {
    if input.dim != 2 {
        return Err(AxolError::Compose(format!(
            "NOT gate requires dim=2, got {}", input.dim
        )));
    }
    let swap = TransMatrix::new(vec![0.0, 1.0, 1.0, 0.0], 2, 2);
    input.transform(&swap)
}

/// AND gate: multiplicative interference (both must be high).
pub fn gate_and(a: &FloatVec, b: &FloatVec) -> Result<Observation> {
    if a.dim() != 2 || b.dim() != 2 {
        return Err(AxolError::Compose("AND gate requires dim=2 inputs".into()));
    }

    let wave_a = Wave::from_classical(a);
    let wave_b = Wave::from_classical(b);
    let wave = Wave::compose(&wave_a, &wave_b, &InterferencePattern::Multiplicative)?;

    let (value_index, collapsed) = wave.observe();
    let probs = FloatVec::new(wave.probabilities().iter().map(|&p| p as f32).collect());

    Ok(Observation {
        value: wave.to_float_vec(),
        value_index,
        value_label: None,
        omega: 1.0,
        phi: 1.0,
        probabilities: probs,
        tapestry_name: "AND".to_string(),
        observation_count: 1,
        density_matrix: None,
        quantum_phi: None,
        quantum_omega: None,
        n_basins: None,
        chosen_basin: None,
        basin_weights: None,
        collapse_metrics: Some(collapsed.metrics),
        wave: Some(wave),
    })
}

/// AND gate returning Wave (no collapse).
pub fn wave_and(a: &Wave, b: &Wave) -> Result<Wave> {
    if a.dim != 2 || b.dim != 2 {
        return Err(AxolError::Compose("AND gate requires dim=2 inputs".into()));
    }
    Wave::compose(a, b, &InterferencePattern::Multiplicative)
}

/// OR gate: constructive interference (at least one high).
pub fn gate_or(a: &FloatVec, b: &FloatVec) -> Result<Observation> {
    if a.dim() != 2 || b.dim() != 2 {
        return Err(AxolError::Compose("OR gate requires dim=2 inputs".into()));
    }

    let wave_a = Wave::from_classical(a);
    let wave_b = Wave::from_classical(b);
    let wave = Wave::compose(&wave_a, &wave_b, &InterferencePattern::Constructive)?;

    let (value_index, collapsed) = wave.observe();
    let probs = FloatVec::new(wave.probabilities().iter().map(|&p| p as f32).collect());

    Ok(Observation {
        value: wave.to_float_vec(),
        value_index,
        value_label: None,
        omega: 1.0,
        phi: 1.0,
        probabilities: probs,
        tapestry_name: "OR".to_string(),
        observation_count: 1,
        density_matrix: None,
        quantum_phi: None,
        quantum_omega: None,
        n_basins: None,
        chosen_basin: None,
        basin_weights: None,
        collapse_metrics: Some(collapsed.metrics),
        wave: Some(wave),
    })
}

/// OR gate returning Wave (no collapse).
pub fn wave_or(a: &Wave, b: &Wave) -> Result<Wave> {
    if a.dim != 2 || b.dim != 2 {
        return Err(AxolError::Compose("OR gate requires dim=2 inputs".into()));
    }
    Wave::compose(a, b, &InterferencePattern::Constructive)
}

/// IF-THEN-ELSE gate: evaluate condition, route to then/else branch.
pub fn eval_if_then_else(
    condition_input: &FloatVec,
    cond_tapestry: &weaver::Tapestry,
    then_tapestry: &weaver::Tapestry,
    else_tapestry: &weaver::Tapestry,
    branch_input: &FloatVec,
) -> Result<Observation> {
    // Observe condition
    let cond_input_name = cond_tapestry.input_names.first()
        .cloned()
        .unwrap_or_else(|| "x".to_string());
    let cond_obs = observatory::observe(
        cond_tapestry,
        &[(&cond_input_name, condition_input)],
    )?;

    let condition_result = decode_bool(&cond_obs);

    // Route to appropriate branch
    let branch_tapestry = if condition_result {
        then_tapestry
    } else {
        else_tapestry
    };

    let branch_input_name = branch_tapestry.input_names.first()
        .cloned()
        .unwrap_or_else(|| "x".to_string());

    observatory::observe(
        branch_tapestry,
        &[(&branch_input_name, branch_input)],
    )
}

/// Create a dim=2 boolean tapestry using the declaration builder.
pub fn make_bool_tapestry(name: &str, omega: f64, phi: f64, seed: u64) -> Result<weaver::Tapestry> {
    let mut builder = DeclarationBuilder::new(name);
    builder
        .input("x", 2)
        .output("result")
        .relate("result", &["x"], RelationKind::Proportional)
        .quality(omega, phi);

    let decl = builder.build();
    weaver::weave(&decl, true, seed)
}
