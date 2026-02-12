//! Confidence observer — majority-vote with early stopping and Wilson score.

use crate::types::FloatVec;
use crate::observatory;
use crate::weaver::Tapestry;
use crate::errors::{AxolError, Result};

/// Configuration for confidence-based observation.
#[derive(Clone, Debug)]
pub struct ConfidenceConfig {
    pub max_observations: usize,
    pub confidence_threshold: f64,
    pub min_observations: usize,
}

impl Default for ConfidenceConfig {
    fn default() -> Self {
        Self {
            max_observations: 100,
            confidence_threshold: 0.95,
            min_observations: 5,
        }
    }
}

/// Result of confidence-based observation.
#[derive(Clone, Debug)]
pub struct ConfidenceResult {
    pub value_index: usize,
    pub confidence: f64,
    pub vote_counts: Vec<usize>,
    pub total_observations: usize,
    pub early_stopped: bool,
    pub avg_probabilities: Vec<f64>,
}

/// Observe a tapestry repeatedly until confidence threshold is met.
///
/// Tracks vote counts per value_index across observations.
/// Uses Wilson score interval for robust small-sample confidence.
/// Early stops when any index exceeds the confidence threshold.
pub fn observe_confident(
    tapestry: &Tapestry,
    inputs: &[(&str, &FloatVec)],
    config: &ConfidenceConfig,
) -> Result<ConfidenceResult> {
    if config.max_observations == 0 {
        return Err(AxolError::Compose("max_observations must be > 0".into()));
    }

    let dim = tapestry.nodes.values().next()
        .map(|n| n.amplitudes.dim())
        .unwrap_or(2);

    let mut vote_counts = vec![0usize; dim];
    let mut prob_acc = vec![0.0f64; dim];
    let mut early_stopped = false;

    for obs_idx in 0..config.max_observations {
        let obs = observatory::observe(tapestry, inputs)?;
        vote_counts[obs.value_index] += 1;

        for (i, &p) in obs.probabilities.data.iter().enumerate() {
            if i < dim {
                prob_acc[i] += p as f64;
            }
        }

        let total = obs_idx + 1;

        // Check early stopping after min_observations
        if total >= config.min_observations {
            let best_idx = vote_counts.iter().enumerate()
                .max_by_key(|(_, &c)| c)
                .map(|(i, _)| i)
                .unwrap_or(0);
            let conf = wilson_lower(vote_counts[best_idx], total);
            if conf >= config.confidence_threshold {
                early_stopped = true;
                break;
            }
        }
    }

    let total = vote_counts.iter().sum::<usize>();
    let best_idx = vote_counts.iter().enumerate()
        .max_by_key(|(_, &c)| c)
        .map(|(i, _)| i)
        .unwrap_or(0);
    let confidence = wilson_lower(vote_counts[best_idx], total);
    let avg_probabilities: Vec<f64> = prob_acc.iter()
        .map(|&p| if total > 0 { p / total as f64 } else { 0.0 })
        .collect();

    Ok(ConfidenceResult {
        value_index: best_idx,
        confidence,
        vote_counts,
        total_observations: total,
        early_stopped,
        avg_probabilities,
    })
}

/// Wilson score lower bound for binomial proportion.
///
/// Provides robust confidence estimate even for small samples.
/// z = 1.96 for 95% confidence level.
fn wilson_lower(successes: usize, total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let n = total as f64;
    let p_hat = successes as f64 / n;
    let z = 1.96; // 95% confidence z-score
    let z2 = z * z;

    let denominator = 1.0 + z2 / n;
    let center = p_hat + z2 / (2.0 * n);
    let spread = z * (p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n)).sqrt();

    ((center - spread) / denominator).clamp(0.0, 1.0)
}
