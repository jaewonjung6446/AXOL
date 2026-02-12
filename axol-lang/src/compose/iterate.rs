//! Iterate — convergence-based iteration loop.
//!
//! Generalizes `observe_evolve()` with configurable convergence criteria
//! and feedback strength control.

use crate::types::FloatVec;
use crate::observatory::{self, Observation};
use crate::weaver::Tapestry;
use crate::errors::{AxolError, Result};

/// Convergence criterion for iteration.
#[derive(Clone, Debug)]
pub enum ConvergenceCriterion {
    /// Stop when max probability delta between iterations < threshold.
    ProbabilityDelta(f64),
    /// Stop when value_index is stable for N consecutive iterations.
    StableIndex(usize),
    /// Stop when omega reaches target (within 5%).
    OmegaTarget,
    /// Stop when phi reaches target (within 5%).
    PhiTarget,
    /// Stop when density matrix purity exceeds threshold.
    PurityThreshold,
}

/// Configuration for the iterate loop.
#[derive(Clone, Debug)]
pub struct IterateConfig {
    pub max_iterations: usize,
    pub min_iterations: usize,
    pub convergence: ConvergenceCriterion,
    /// Enable feedback (adjust chaos.r based on quality mismatch).
    pub feedback: bool,
    /// How aggressively r is adjusted: 0.0 = none, 1.0 = aggressive.
    pub feedback_strength: f64,
}

impl Default for IterateConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            min_iterations: 3,
            convergence: ConvergenceCriterion::ProbabilityDelta(0.001),
            feedback: true,
            feedback_strength: 0.5,
        }
    }
}

/// Result of the iterate loop.
#[derive(Clone, Debug)]
pub struct IterateResult {
    pub observation: Observation,
    pub iterations: usize,
    pub converged: bool,
    pub history: Vec<Observation>,
    pub final_delta: f64,
}

/// Run an iterative observation loop with configurable convergence.
///
/// At each iteration:
/// 1. Observe the tapestry
/// 2. Check convergence criterion
/// 3. If feedback enabled, adjust chaos engine parameters
/// 4. Continue until converged or max_iterations reached
pub fn iterate(
    tapestry: &mut Tapestry,
    inputs: &[(&str, &FloatVec)],
    config: &IterateConfig,
) -> Result<IterateResult> {
    if config.max_iterations == 0 {
        return Err(AxolError::Compose("max_iterations must be > 0".into()));
    }

    let mut history: Vec<Observation> = Vec::new();
    let mut converged = false;
    let mut final_delta = f64::MAX;
    let mut stable_count = 0usize;

    for iter in 0..config.max_iterations {
        let obs = observatory::observe(tapestry, inputs)?;

        // Check convergence after min_iterations
        if iter >= config.min_iterations && !history.is_empty() {
            let prev = history.last().unwrap();
            match &config.convergence {
                ConvergenceCriterion::ProbabilityDelta(threshold) => {
                    let delta = max_prob_delta(&obs.probabilities, &prev.probabilities);
                    final_delta = delta;
                    if delta < *threshold {
                        converged = true;
                    }
                }
                ConvergenceCriterion::StableIndex(required) => {
                    if obs.value_index == prev.value_index {
                        stable_count += 1;
                    } else {
                        stable_count = 0;
                    }
                    final_delta = if stable_count >= *required { 0.0 } else { 1.0 };
                    if stable_count >= *required {
                        converged = true;
                    }
                }
                ConvergenceCriterion::OmegaTarget => {
                    let target = tapestry.report.target_omega;
                    let observed = tapestry.global_attractor.omega();
                    final_delta = (observed - target).abs() / target.max(1e-10);
                    if final_delta < 0.05 {
                        converged = true;
                    }
                }
                ConvergenceCriterion::PhiTarget => {
                    let target = tapestry.report.target_phi;
                    let observed = tapestry.global_attractor.phi();
                    final_delta = (observed - target).abs() / target.max(1e-10);
                    if final_delta < 0.05 {
                        converged = true;
                    }
                }
                ConvergenceCriterion::PurityThreshold => {
                    if let Some(ref dm) = obs.density_matrix {
                        let purity = dm.purity();
                        final_delta = 1.0 - purity;
                        if purity > 0.95 {
                            converged = true;
                        }
                    } else {
                        final_delta = 1.0;
                    }
                }
            }
        }

        history.push(obs);

        if converged {
            break;
        }

        // Feedback: adjust chaos engine parameters
        if config.feedback {
            apply_feedback(tapestry, config.feedback_strength);
        }
    }

    let last_obs = history.last().cloned()
        .ok_or_else(|| AxolError::Compose("No observations recorded".into()))?;
    let iterations = history.len();

    Ok(IterateResult {
        observation: last_obs,
        iterations,
        converged,
        history,
        final_delta,
    })
}

/// Compute max absolute difference between two probability vectors.
fn max_prob_delta(a: &FloatVec, b: &FloatVec) -> f64 {
    a.data.iter().zip(b.data.iter())
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .fold(0.0f64, f64::max)
}

/// Apply feedback to adjust chaos engine based on quality mismatch.
fn apply_feedback(tapestry: &mut Tapestry, strength: f64) {
    if let Some(ref mut chaos) = tapestry.chaos_engine {
        let observed_omega = tapestry.global_attractor.omega();
        let target_omega = tapestry.report.target_omega;

        let delta_r = if observed_omega < target_omega * 0.8 {
            -0.02 * strength
        } else if observed_omega > target_omega * 1.2 {
            0.01 * strength
        } else {
            return;
        };

        chaos.r = (chaos.r + delta_r).clamp(3.0, 4.0);

        // Re-run dynamics with adjusted parameters
        let new_result = chaos.find_attractor(42, 200, 300);
        let new_matrix = chaos.extract_matrix(&new_result);

        tapestry.composed_matrix = Some(new_matrix.clone());
        tapestry.global_attractor.max_lyapunov = new_result.max_lyapunov;
        tapestry.global_attractor.fractal_dim = new_result.fractal_dim;
        tapestry.global_attractor.lyapunov_spectrum = new_result.lyapunov_spectrum.clone();
        tapestry.global_attractor.trajectory_matrix = new_matrix;

        if tapestry.quantum {
            tapestry.basins = Some(chaos.find_basins(100, 200, 42));
        }
    }
}
