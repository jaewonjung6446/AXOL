//! Logic gates — AND, OR, NOT, IF-THEN-ELSE (dim=2 boolean encoding).
//!
//! Boolean encoding: FALSE = [0.9, 0.1], TRUE = [0.1, 0.9]
//! Uses dim=2 tapestries for gate operations.

use crate::types::*;
use crate::ops;
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

/// Decode an Observation back to a boolean.
///
/// value_index == 1 → TRUE, else FALSE.
pub fn decode_bool(obs: &Observation) -> bool {
    obs.value_index == 1
}

/// NOT gate: swap matrix [[0,1],[1,0]].
///
/// Trivial — no chaos needed, pure matrix swap.
pub fn gate_not(input: &FloatVec) -> Result<Observation> {
    if input.dim() != 2 {
        return Err(AxolError::Compose(format!(
            "NOT gate requires dim=2, got {}", input.dim()
        )));
    }

    // Swap matrix
    let swap = TransMatrix::new(vec![0.0, 1.0, 1.0, 0.0], 2, 2);
    let value = ops::transform(input, &swap)?;
    let probs = ops::measure(&value);
    let value_index = ops::argmax(&probs);

    Ok(Observation {
        value,
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
    })
}

/// AND gate: both inputs must be TRUE for TRUE output.
///
/// TRUE component = a[1]*b[1] (both-high product).
/// FALSE component = 1 - a[1]*b[1] (complement).
pub fn gate_and(a: &FloatVec, b: &FloatVec) -> Result<Observation> {
    if a.dim() != 2 || b.dim() != 2 {
        return Err(AxolError::Compose("AND gate requires dim=2 inputs".into()));
    }

    // AND: output TRUE only when both TRUE components are high
    let both_true = a.data[1] * b.data[1];
    let combined = FloatVec::new(vec![
        1.0 - both_true, // FALSE component
        both_true,       // TRUE component
    ]);

    let probs = ops::measure(&combined);
    let value_index = ops::argmax(&probs);

    Ok(Observation {
        value: combined,
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
    })
}

/// OR gate: at least one input must be TRUE for TRUE output.
///
/// FALSE component = a[0]*b[0] (both-low product).
/// TRUE component = 1 - a[0]*b[0] (complement: at least one high).
pub fn gate_or(a: &FloatVec, b: &FloatVec) -> Result<Observation> {
    if a.dim() != 2 || b.dim() != 2 {
        return Err(AxolError::Compose("OR gate requires dim=2 inputs".into()));
    }

    // OR: output FALSE only when both FALSE components are high
    let both_false = a.data[0] * b.data[0];
    let combined = FloatVec::new(vec![
        both_false,       // FALSE component
        1.0 - both_false, // TRUE component
    ]);

    let probs = ops::measure(&combined);
    let value_index = ops::argmax(&probs);

    Ok(Observation {
        value: combined,
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
    })
}

/// IF-THEN-ELSE gate: evaluate condition, route to then/else branch.
///
/// 1. Observe condition tapestry with inputs
/// 2. If condition → TRUE (value_index == 1), observe then_tapestry
/// 3. If condition → FALSE (value_index == 0), observe else_tapestry
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
