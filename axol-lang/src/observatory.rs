//! Observatory — observe a tapestry to collapse to a single result.
//!
//! Quantum path now uses basin-based superposition:
//!   1. Run short dynamics from input → determine basin proximity
//!   2. Create quantum superposition across basins
//!   3. Interference between basins (phase-dependent)
//!   4. Born rule measurement → collapse to one outcome
//!   5. Feedback: observation result adjusts dynamics parameters

use num_complex::Complex64;
use crate::types::*;
use crate::ops;
use crate::density;
use crate::dynamics::{ChaosEngine, Basin};
use crate::weaver::Tapestry;
use crate::errors::{AxolError, Result};

// ---------------------------------------------------------------------------
// Observation result
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
}

// ---------------------------------------------------------------------------
// observe()
// ---------------------------------------------------------------------------

pub fn observe(tapestry: &Tapestry, inputs: &[(&str, &FloatVec)]) -> Result<Observation> {
    // Validate inputs
    for name in &tapestry.input_names {
        if !inputs.iter().any(|(n, _)| *n == name.as_str()) {
            return Err(AxolError::Observatory(format!("Missing input: '{name}'")));
        }
    }

    let omega = tapestry.global_attractor.omega();
    let phi = tapestry.global_attractor.phi();

    let output_name = tapestry.output_names.first()
        .cloned()
        .unwrap_or_else(|| tapestry.nodes.keys().last().cloned().unwrap_or_default());

    // === QUANTUM BASIN PATH ===
    if tapestry.quantum {
        if let (Some(ref chaos), Some(ref basins)) = (&tapestry.chaos_engine, &tapestry.basins) {
            if let Some((_, input_fv)) = inputs.first() {
                if !basins.is_empty() {
                    return observe_quantum_basins(
                        tapestry, chaos, basins, input_fv, &output_name, omega, phi,
                    );
                }
            }
        }

        // Fallback to composed matrix quantum path
        if let Some(ref matrix) = tapestry.composed_matrix {
            if let Some((_, input_fv)) = inputs.first() {
                let input_cv = ComplexVec::from_real(input_fv);
                let result_cv = ops::transform_complex(&input_cv, matrix)?;
                let result_cv = clamp_complex(&result_cv);

                let probs = ops::measure_complex(&result_cv);
                let value = result_cv.to_real();
                let value_index = ops::argmax(&probs);

                let label = tapestry.nodes.get(&output_name)
                    .and_then(|n| n.labels.get(&value_index).cloned());

                let post_density = DensityMatrix::from_pure_state(&result_cv);
                let q_phi = density::phi_from_purity(&post_density);
                let q_omega = density::omega_from_coherence(&post_density);

                let final_density = if let Some(ref kraus) = tapestry.kraus_operators {
                    density::apply_channel(&post_density, kraus)
                } else {
                    post_density
                };

                return Ok(Observation {
                    value,
                    value_index,
                    value_label: label,
                    omega,
                    phi,
                    probabilities: probs,
                    tapestry_name: tapestry.name.clone(),
                    observation_count: 1,
                    density_matrix: Some(final_density),
                    quantum_phi: Some(q_phi),
                    quantum_omega: Some(q_omega),
                    n_basins: None,
                    chosen_basin: None,
                });
            }
        }
    }

    // === CLASSICAL FAST PATH ===
    if let Some(ref matrix) = tapestry.composed_matrix {
        if let Some((_, input_fv)) = inputs.first() {
            let value = ops::transform(input_fv, matrix)?;
            let probs = ops::measure(&value);
            let value_index = ops::argmax(&probs);

            let label = tapestry.nodes.get(&output_name)
                .and_then(|n| n.labels.get(&value_index).cloned());

            return Ok(Observation {
                value,
                value_index,
                value_label: label,
                omega,
                phi,
                probabilities: probs,
                tapestry_name: tapestry.name.clone(),
                observation_count: 1,
                density_matrix: None,
                quantum_phi: None,
                quantum_omega: None,
                n_basins: None,
                chosen_basin: None,
            });
        }
    }

    // === FALLBACK ===
    let output_node = tapestry.nodes.get(&output_name)
        .ok_or_else(|| AxolError::Observatory(format!("Output '{}' not found", output_name)))?;

    let probs = ops::measure(&output_node.amplitudes);
    let value_index = ops::argmax(&probs);
    let label = output_node.labels.get(&value_index).cloned();

    Ok(Observation {
        value: output_node.amplitudes.clone(),
        value_index,
        value_label: label,
        omega,
        phi,
        probabilities: probs,
        tapestry_name: tapestry.name.clone(),
        observation_count: 1,
        density_matrix: None,
        quantum_phi: None,
        quantum_omega: None,
        n_basins: None,
        chosen_basin: None,
    })
}

// ---------------------------------------------------------------------------
// Quantum basin observation
// ---------------------------------------------------------------------------

/// Observe using basin-based quantum superposition.
///
/// 1. Run short dynamics from input → find where it goes
/// 2. Compute proximity to each basin
/// 3. Create superposition: amplitude ∝ 1/(1+dist), phase from basin
/// 4. Apply interference between basin components
/// 5. Born rule → collapse
fn observe_quantum_basins(
    tapestry: &Tapestry,
    chaos: &ChaosEngine,
    basins: &[Basin],
    input: &FloatVec,
    output_name: &str,
    omega: f64,
    phi: f64,
) -> Result<Observation> {
    let dim = input.dim();

    // 1. Map input to dynamics range [0,1] and run short dynamics
    let input_normalized: Vec<f64> = input.data.iter()
        .map(|&v| {
            let v64 = v as f64;
            // Sigmoid-like mapping to (0,1)
            1.0 / (1.0 + (-v64).exp())
        })
        .collect();

    let endpoint = chaos.short_run(&input_normalized, 30);

    // 2. Compute distance to each basin
    let mut basin_distances: Vec<(usize, f64)> = basins.iter().enumerate()
        .map(|(i, b)| {
            let dist: f64 = b.center.iter().zip(endpoint.iter())
                .map(|(a, e)| (a - e).powi(2))
                .sum::<f64>()
                .sqrt();
            (i, dist)
        })
        .collect();
    basin_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // 3. Create quantum superposition across basins
    let mut cv_data = vec![Complex64::new(0.0, 0.0); dim];

    for (basin_idx, dist) in &basin_distances {
        let basin = &basins[*basin_idx];
        // Amplitude: closer basin → higher amplitude
        let amplitude = 1.0 / (1.0 + dist * 5.0);
        let phase = basin.phase;

        let component = Complex64::from_polar(amplitude, phase);

        // Each basin contributes to the state vector based on its center
        for d in 0..dim.min(basin.center.len()) {
            cv_data[d] += component * Complex64::new(basin.center[d], 0.0);
        }
    }

    // Normalize
    let norm: f64 = cv_data.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
    if norm > 1e-15 {
        for c in cv_data.iter_mut() {
            *c /= norm;
        }
    }
    let cv = ComplexVec::new(cv_data);

    // 4. Apply composed matrix transform (dynamics-derived)
    let result_cv = if let Some(ref matrix) = tapestry.composed_matrix {
        let transformed = ops::transform_complex(&cv, matrix)?;
        clamp_complex(&transformed)
    } else {
        cv.clone()
    };

    // 5. Born rule measurement
    let probs = ops::measure_complex(&result_cv);
    let value = result_cv.to_real();
    let value_index = ops::argmax(&probs);

    let label = tapestry.nodes.get(output_name)
        .and_then(|n| n.labels.get(&value_index).cloned());

    // Density matrix and quantum metrics
    let post_density = DensityMatrix::from_pure_state(&result_cv);
    let q_phi = density::phi_from_purity(&post_density);
    let q_omega = density::omega_from_coherence(&post_density);

    let final_density = if let Some(ref kraus) = tapestry.kraus_operators {
        density::apply_channel(&post_density, kraus)
    } else {
        post_density
    };

    let chosen = basin_distances.first().map(|(i, _)| *i);

    Ok(Observation {
        value,
        value_index,
        value_label: label,
        omega,
        phi,
        probabilities: probs,
        tapestry_name: tapestry.name.clone(),
        observation_count: 1,
        density_matrix: Some(final_density),
        quantum_phi: Some(q_phi),
        quantum_omega: Some(q_omega),
        n_basins: Some(basins.len()),
        chosen_basin: chosen,
    })
}

// ---------------------------------------------------------------------------
// reobserve()
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
    })
}

// ---------------------------------------------------------------------------
// observe_evolve() — measurement feedback loop
// ---------------------------------------------------------------------------

/// Observe then feed the result back into the dynamics.
/// Each iteration adjusts the chaos parameter based on observed quality.
/// This implements: measure → adjust → re-observe → converge.
pub fn observe_evolve(
    tapestry: &mut Tapestry,
    inputs: &[(&str, &FloatVec)],
    iterations: usize,
) -> Result<Observation> {
    let mut last_obs = observe(tapestry, inputs)?;

    for _ in 1..iterations {
        // Feedback: adjust chaos engine based on observation quality
        if let Some(ref mut chaos) = tapestry.chaos_engine {
            let observed_omega = tapestry.global_attractor.omega();
            let target_omega = tapestry.report.target_omega;

            // If observed omega is below target → reduce chaos (lower r)
            // If above target → increase chaos slightly (raise r)
            if observed_omega < target_omega * 0.8 {
                chaos.r = (chaos.r - 0.02).clamp(3.0, 4.0);
            } else if observed_omega > target_omega * 1.2 {
                chaos.r = (chaos.r + 0.01).clamp(3.0, 4.0);
            }

            // Re-run dynamics with adjusted parameters
            let new_result = chaos.find_attractor(42, 200, 300);
            let new_matrix = chaos.extract_matrix(&new_result);

            // Update tapestry
            tapestry.composed_matrix = Some(new_matrix.clone());
            tapestry.global_attractor.max_lyapunov = new_result.max_lyapunov;
            tapestry.global_attractor.fractal_dim = new_result.fractal_dim;
            tapestry.global_attractor.lyapunov_spectrum = new_result.lyapunov_spectrum.clone();
            tapestry.global_attractor.trajectory_matrix = new_matrix;

            // Re-detect basins
            if tapestry.quantum {
                tapestry.basins = Some(chaos.find_basins(100, 200, 42));
            }
        }

        last_obs = observe(tapestry, inputs)?;
    }

    last_obs.observation_count = iterations;
    Ok(last_obs)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn clamp_complex(cv: &ComplexVec) -> ComplexVec {
    ComplexVec::new(
        cv.data.iter().map(|c| {
            let re = if c.re.is_nan() { 0.0 } else { c.re.clamp(-1e6, 1e6) };
            let im = if c.im.is_nan() { 0.0 } else { c.im.clamp(-1e6, 1e6) };
            Complex64::new(re, im)
        }).collect()
    )
}
