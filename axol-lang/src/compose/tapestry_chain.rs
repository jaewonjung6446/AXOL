//! Tapestry chain — sequential tapestry composition via matrix multiplication.

use crate::types::*;
use crate::observatory::{self, Observation};
use crate::weaver::Tapestry;
use crate::errors::{AxolError, Result};

/// A chain of tapestries composed sequentially.
#[derive(Clone, Debug)]
pub struct TapestryChain {
    pub name: String,
    pub stages: Vec<Tapestry>,
    pub composed_matrix: TransMatrix,
}

/// Create a chain by composing multiple tapestries via matrix multiplication.
///
/// Validates that dimensions are compatible across stages and composes
/// all transformation matrices into a single composed_matrix.
pub fn chain(name: &str, tapestries: Vec<Tapestry>) -> Result<TapestryChain> {
    if tapestries.is_empty() {
        return Err(AxolError::Compose("Cannot create empty chain".into()));
    }
    if tapestries.len() == 1 {
        let mat = tapestries[0].composed_matrix.clone()
            .ok_or_else(|| AxolError::Compose("Stage 0 has no composed_matrix".into()))?;
        return Ok(TapestryChain {
            name: name.to_string(),
            stages: tapestries,
            composed_matrix: mat,
        });
    }

    // Validate dimensions and compose matrices
    let mut composed = tapestries[0].composed_matrix.clone()
        .ok_or_else(|| AxolError::Compose("Stage 0 has no composed_matrix".into()))?;

    for (i, t) in tapestries.iter().enumerate().skip(1) {
        let mat = t.composed_matrix.as_ref()
            .ok_or_else(|| AxolError::Compose(format!("Stage {} has no composed_matrix", i)))?;

        if composed.cols != mat.rows {
            return Err(AxolError::Compose(format!(
                "Dimension mismatch at stage {}: {} cols vs {} rows",
                i, composed.cols, mat.rows
            )));
        }

        composed = composed.matmul(mat);
    }

    Ok(TapestryChain {
        name: name.to_string(),
        stages: tapestries,
        composed_matrix: composed,
    })
}

/// Observe a chain using the fast path: single composed matrix.
///
/// Constructs a temporary tapestry from the composed matrix and observes it.
pub fn observe_chain(
    chain: &TapestryChain,
    inputs: &[(&str, &FloatVec)],
) -> Result<Observation> {
    if chain.stages.is_empty() {
        return Err(AxolError::Compose("Empty chain".into()));
    }

    // Build a temporary tapestry with the composed matrix
    let base = &chain.stages[0];
    let mut temp = base.clone();
    temp.name = chain.name.clone();
    temp.composed_matrix = Some(chain.composed_matrix.clone());

    observatory::observe(&temp, inputs)
}

/// Observe a chain sequentially: per-stage observe.
///
/// The output of each stage becomes the input of the next.
/// Returns observations from all stages.
pub fn observe_chain_sequential(
    chain: &TapestryChain,
    inputs: &[(&str, &FloatVec)],
) -> Result<Vec<Observation>> {
    if chain.stages.is_empty() {
        return Err(AxolError::Compose("Empty chain".into()));
    }

    let mut observations = Vec::new();
    let mut current_input: Option<FloatVec> = None;

    for (i, stage) in chain.stages.iter().enumerate() {
        let stage_inputs = if i == 0 {
            inputs.to_vec()
        } else if let Some(ref fv) = current_input {
            let input_name = stage.input_names.first()
                .cloned()
                .unwrap_or_else(|| "x".to_string());
            vec![(&*Box::leak(input_name.into_boxed_str()), fv as &FloatVec)]
        } else {
            return Err(AxolError::Compose(format!("No output from stage {}", i - 1)));
        };

        let obs = observatory::observe(stage, &stage_inputs)?;
        current_input = Some(obs.value.clone());
        observations.push(obs);
    }

    Ok(observations)
}

/// Flatten a chain into a single tapestry with the composed matrix.
pub fn flatten(chain: &TapestryChain) -> Result<Tapestry> {
    if chain.stages.is_empty() {
        return Err(AxolError::Compose("Empty chain".into()));
    }

    let mut result = chain.stages[0].clone();
    result.name = chain.name.clone();
    result.composed_matrix = Some(chain.composed_matrix.clone());

    // Use the last stage's output info
    if let Some(last) = chain.stages.last() {
        result.output_names = last.output_names.clone();
    }

    Ok(result)
}
