//! Observatory — compute Waves from tapestries, then observe.
//!
//! The core function is `compute_wave()`: given a tapestry and inputs,
//! produce a Wave (uncollapsed quantum state). Everything else is a
//! wrapper that applies different collapse levels:
//!
//!   compute_wave()  → Wave (t=0.0, pure superposition)
//!   gaze()          → Wave (t=0.0, read probabilities, C=0)
//!   glimpse(gamma)  → Wave (t=gamma, partial collapse, C=gamma)
//!   observe()       → Observation (t=1.0, full collapse, C=1)

use num_complex::Complex64;
use crate::types::*;
use crate::ops;
use crate::density;
use std::collections::HashMap;
use crate::wave::{Wave, InterferencePattern, InterferenceRule};
use crate::collapse::{self, CollapseMetrics};
use crate::weaver::Tapestry;
use crate::errors::{AxolError, Result};

// ---------------------------------------------------------------------------
// Observation result (backward-compatible collapsed output)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Observation {
    pub value: FloatVec,
    pub value_index: usize,
    pub value_label: Option<String>,
    pub omega: f64,
    pub phi: f64,
    pub probabilities: FloatVec,
    pub tapestry_name: String,
    pub observation_count: usize,
    // Quantum extensions
    pub density_matrix: Option<DensityMatrix>,
    pub quantum_phi: Option<f64>,
    pub quantum_omega: Option<f64>,
    // Basin info
    pub n_basins: Option<usize>,
    pub chosen_basin: Option<usize>,
    pub basin_weights: Option<Vec<f64>>,
    // Collapse metrics
    pub collapse_metrics: Option<CollapseMetrics>,
    // Wave (the uncollapsed state that produced this observation)
    pub wave: Option<Wave>,
}

// ---------------------------------------------------------------------------
// compute_wave() — the core: tapestry + inputs → Wave (no collapse)
// ---------------------------------------------------------------------------

/// Compute a Wave from a tapestry and inputs WITHOUT collapsing.
///
/// This is the fundamental operation. The Wave carries the full quantum
/// state (complex amplitudes, phases, interference capability) at t=0.0.
///
/// Collapse cost: C = 0
pub fn compute_wave(tapestry: &Tapestry, inputs: &[(&str, &FloatVec)]) -> Result<Wave> {
    // Validate inputs
    for name in &tapestry.input_names {
        if !inputs.iter().any(|(n, _)| *n == name.as_str()) {
            return Err(AxolError::Observatory(format!("Missing input: '{name}'")));
        }
    }

    // === QUANTUM BASIN PATH ===
    if tapestry.quantum && tapestry.basin_structure.n_basins > 0 {
        let input_waves: HashMap<String, Wave> = inputs.iter()
            .map(|(name, fv)| (name.to_string(), Wave::from_basins(&tapestry.basin_structure, fv)))
            .collect();
        let mut wave = compose_from_rules(&input_waves, &tapestry.interference_rules)?;

        // Apply composed matrix transform (preserves t=0.0)
        if let Some(ref matrix) = tapestry.composed_matrix {
            wave = wave.transform(matrix)?;
        }

        // Apply Kraus operators if present
        if let Some(ref kraus) = tapestry.kraus_operators {
            let rho = wave.to_density();
            let evolved = density::apply_channel(&rho, kraus);
            wave = Wave::from_density(evolved);
        }

        return Ok(wave);
    }

    // === QUANTUM MATRIX PATH (fallback) ===
    if tapestry.quantum {
        if let Some(ref matrix) = tapestry.composed_matrix {
            let input_waves: HashMap<String, Wave> = inputs.iter()
                .map(|(name, fv)| (name.to_string(), Wave::from_classical(fv)))
                .collect();
            let mut wave = compose_from_rules(&input_waves, &tapestry.interference_rules)?;
            wave = wave.transform(matrix)?;

            if let Some(ref kraus) = tapestry.kraus_operators {
                let rho = wave.to_density();
                let evolved = density::apply_channel(&rho, kraus);
                wave = Wave::from_density(evolved);
            }

            return Ok(wave);
        }
    }

    // === CLASSICAL PATH (produces Wave with real amplitudes) ===
    if let Some(ref matrix) = tapestry.composed_matrix {
        let input_waves: HashMap<String, Wave> = inputs.iter()
            .map(|(name, fv)| (name.to_string(), Wave::from_classical(fv)))
            .collect();
        let composed = compose_from_rules(&input_waves, &tapestry.interference_rules)?;
        let result = ops::transform(&composed.to_float_vec(), matrix)?;
        return Ok(Wave::from_classical(&result));
    }

    // === FALLBACK: use output node amplitudes ===
    let output_name = tapestry.output_names.first()
        .cloned()
        .unwrap_or_else(|| tapestry.nodes.keys().last().cloned().unwrap_or_default());

    let output_node = tapestry.nodes.get(&output_name)
        .ok_or_else(|| AxolError::Observatory(format!("Output '{}' not found", output_name)))?;

    Ok(Wave::from_classical(&output_node.amplitudes))
}

// ---------------------------------------------------------------------------
// compose_from_rules() — compose input waves according to interference rules
// ---------------------------------------------------------------------------

/// Compose input waves according to the tapestry's interference rules (DAG evaluation).
///
/// For multi-input models (e.g., `relate y <- x, ctx via <~>`), this composes
/// all input waves using the declared interference pattern, rather than ignoring
/// all but the first input.
fn compose_from_rules(
    input_waves: &HashMap<String, Wave>,
    rules: &[InterferenceRule],
) -> Result<Wave> {
    // Single input: return directly
    if input_waves.len() == 1 {
        return Ok(input_waves.values().next().unwrap().clone());
    }

    // No rules: compose all inputs with Constructive (sorted keys for determinism)
    if rules.is_empty() {
        let mut keys: Vec<&String> = input_waves.keys().collect();
        keys.sort();
        let all: Vec<&Wave> = keys.iter().filter_map(|k| input_waves.get(*k)).collect();
        let pat = InterferencePattern::Constructive;
        let pats: Vec<&InterferencePattern> = vec![&pat; all.len() - 1];
        return Wave::compose_many(&all, &pats);
    }

    // Process rules in DAG order, building intermediate waves
    let mut intermediates: HashMap<String, Wave> = HashMap::new();
    let mut last_output = String::new();

    for rule in rules {
        let sources: Vec<Wave> = rule.sources.iter()
            .filter_map(|name| {
                intermediates.get(name)
                    .or_else(|| input_waves.get(name))
                    .cloned()
            })
            .collect();

        if sources.is_empty() { continue; }

        let result = if sources.len() == 1 {
            sources.into_iter().next().unwrap()
        } else {
            let refs: Vec<&Wave> = sources.iter().collect();
            let pats: Vec<&InterferencePattern> = vec![&rule.pattern; refs.len() - 1];
            Wave::compose_many(&refs, &pats)?
        };

        last_output = rule.output.clone();
        intermediates.insert(rule.output.clone(), result);
    }

    intermediates.remove(&last_output)
        .ok_or_else(|| AxolError::Observatory("No output wave produced from rules".into()))
}

// ---------------------------------------------------------------------------
// gaze() — zero-collapse observation (returns Wave)
// ---------------------------------------------------------------------------

/// Read the Wave without any collapse. C = 0.
///
/// The Wave's probability distribution IS the answer.
/// No information is destroyed.
pub fn gaze(tapestry: &Tapestry, inputs: &[(&str, &FloatVec)]) -> Result<Wave> {
    let mut wave = compute_wave(tapestry, inputs)?;

    // Compute metrics from density matrix
    let rho = wave.to_density();
    wave.metrics.update_from_density(&rho);
    if wave.density.is_none() {
        wave.density = Some(rho);
    }

    Ok(wave)
}

// ---------------------------------------------------------------------------
// glimpse() — partial collapse (returns focused Wave)
// ---------------------------------------------------------------------------

/// Partially collapse a Wave. C = gamma.
///
/// gamma in [0, 1]:
///   0.0 = no change (equivalent to gaze)
///   0.5 = moderate focusing
///   1.0 = full collapse
pub fn glimpse(
    tapestry: &Tapestry,
    inputs: &[(&str, &FloatVec)],
    gamma: f64,
) -> Result<Wave> {
    let wave = compute_wave(tapestry, inputs)?;
    Ok(wave.focus(gamma))
}

// ---------------------------------------------------------------------------
// observe() — full collapse (returns Observation, backward-compatible)
// ---------------------------------------------------------------------------

/// Fully collapse a Wave to a classical result. C = 1.
///
/// This is the boundary between the Wave world and the classical world.
/// The Wave is computed, then collapsed via argmax to produce a definite
/// value_index. All phase and coherence information is destroyed.
pub fn observe(tapestry: &Tapestry, inputs: &[(&str, &FloatVec)]) -> Result<Observation> {
    let wave = compute_wave(tapestry, inputs)?;
    observation_from_wave(tapestry, &wave)
}

/// Convert a Wave to an Observation (full collapse).
pub fn observation_from_wave(tapestry: &Tapestry, wave: &Wave) -> Result<Observation> {
    let omega = tapestry.basin_structure.omega();
    let phi = tapestry.basin_structure.phi();

    let output_name = tapestry.output_names.first()
        .cloned()
        .unwrap_or_else(|| tapestry.nodes.keys().last().cloned().unwrap_or_default());

    let probs = wave.probabilities();
    let probs_f32 = FloatVec::new(probs.iter().map(|&p| p as f32).collect());
    let (value_index, collapsed_wave) = wave.observe();

    let label = tapestry.nodes.get(&output_name)
        .and_then(|n| n.labels.get(&value_index).cloned());

    // Quantum metrics from density matrix
    let rho = wave.to_density();
    let q_phi = density::phi_from_purity(&rho);
    let q_omega = density::omega_from_coherence(&rho);

    Ok(Observation {
        value: wave.to_float_vec(),
        value_index,
        value_label: label,
        omega,
        phi,
        probabilities: probs_f32,
        tapestry_name: tapestry.name.clone(),
        observation_count: 1,
        density_matrix: Some(rho),
        quantum_phi: Some(q_phi),
        quantum_omega: Some(q_omega),
        n_basins: Some(tapestry.basin_structure.n_basins),
        chosen_basin: None,
        basin_weights: None,
        collapse_metrics: Some(collapsed_wave.metrics),
        wave: Some(wave.clone()),
    })
}

// ---------------------------------------------------------------------------
// reobserve() — backward-compatible multi-observation
// ---------------------------------------------------------------------------

pub fn reobserve(tapestry: &Tapestry, inputs: &[(&str, &FloatVec)], count: usize) -> Result<Observation> {
    if count < 1 {
        return Err(AxolError::Observatory("count must be >= 1".into()));
    }

    let mut prob_acc: Option<Vec<f64>> = None;
    let mut last_obs: Option<Observation> = None;

    for _ in 0..count {
        let obs = observe(tapestry, inputs)?;
        let probs_f64: Vec<f64> = obs.probabilities.data.iter().map(|&x| x as f64).collect();

        prob_acc = Some(match prob_acc {
            Some(mut acc) => {
                for (a, p) in acc.iter_mut().zip(probs_f64.iter()) {
                    *a += p;
                }
                acc
            }
            None => probs_f64,
        });
        last_obs = Some(obs);
    }

    let last = last_obs.unwrap();
    let avg: Vec<f32> = prob_acc.unwrap().iter().map(|p| (*p / count as f64) as f32).collect();

    let total: f32 = avg.iter().sum();
    let avg_probs = if total > 0.0 {
        FloatVec::new(avg.iter().map(|p| p / total).collect())
    } else {
        FloatVec::new(avg)
    };

    let value_index = ops::argmax(&avg_probs);
    let output_name = tapestry.output_names.first().cloned().unwrap_or_default();
    let label = tapestry.nodes.get(&output_name)
        .and_then(|n| n.labels.get(&value_index).cloned());

    Ok(Observation {
        value: last.value,
        value_index,
        value_label: label,
        omega: last.omega,
        phi: last.phi,
        probabilities: avg_probs,
        tapestry_name: tapestry.name.clone(),
        observation_count: count,
        density_matrix: last.density_matrix,
        quantum_phi: last.quantum_phi,
        quantum_omega: last.quantum_omega,
        n_basins: last.n_basins,
        chosen_basin: last.chosen_basin,
        basin_weights: last.basin_weights,
        collapse_metrics: None,
        wave: last.wave,
    })
}

// ---------------------------------------------------------------------------
// observe_evolve() — measurement feedback loop
// ---------------------------------------------------------------------------

pub fn observe_evolve(
    tapestry: &mut Tapestry,
    inputs: &[(&str, &FloatVec)],
    iterations: usize,
) -> Result<Observation> {
    use crate::weaver::build_basin_structure;

    let mut last_obs = observe(tapestry, inputs)?;

    for _ in 1..iterations {
        // When basin structure is user-defined (define_basins/from_basins),
        // skip chaos feedback entirely — the structure is fixed by design.
        if !tapestry.preserve_basins {
            if let Some(ref mut chaos) = tapestry.chaos_engine {
                let observed_omega = tapestry.basin_structure.omega();
                let target_omega = tapestry.report.target_omega;

                if observed_omega < target_omega * 0.8 {
                    chaos.r = (chaos.r - 0.02).clamp(3.0, 4.0);
                } else if observed_omega > target_omega * 1.2 {
                    chaos.r = (chaos.r + 0.01).clamp(3.0, 4.0);
                }

                let new_result = chaos.find_attractor(42, 200, 300);
                let new_matrix = chaos.extract_matrix(&new_result);

                tapestry.composed_matrix = Some(new_matrix.clone());
                tapestry.global_attractor.max_lyapunov = new_result.max_lyapunov;
                tapestry.global_attractor.fractal_dim = new_result.fractal_dim;
                tapestry.global_attractor.lyapunov_spectrum = new_result.lyapunov_spectrum.clone();
                tapestry.global_attractor.trajectory_matrix = new_matrix.clone();

                let new_basins = chaos.find_basins(100, 200, 42);
                tapestry.basin_structure = build_basin_structure(
                    &new_basins,
                    tapestry.basin_structure.dim,
                    new_result.fractal_dim,
                    Some(new_matrix),
                );
                tapestry.basins = Some(new_basins);
            }
        }

        last_obs = observe(tapestry, inputs)?;
    }

    last_obs.observation_count = iterations;
    Ok(last_obs)
}
