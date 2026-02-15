//! Shared types and utility functions for the AXOL text pipeline.
//!
//! Provides token prediction types, sampling configuration,
//! and core mathematical utilities (softmax, Omega/Phi metrics)
//! used by the WTE engine, heads, and fingerprint modules.

// ---------------------------------------------------------------------------
// TokenPrediction — result of observing a prediction head
// ---------------------------------------------------------------------------

/// A single token prediction with quality metrics.
#[derive(Clone, Debug)]
pub struct TokenPrediction {
    /// Predicted token ID (basin index after collapse)
    pub token_id: usize,
    /// Predicted word
    pub token: String,
    /// Confidence: how sharply the context points to this token
    pub confidence: f64,
    /// Omega (cohesion): stability of this prediction
    pub omega: f64,
    /// Phi (clarity): precision of this prediction
    pub phi: f64,
    /// Full probability distribution over vocabulary
    pub probabilities: Vec<f64>,
    /// Top-k predictions with probabilities
    pub top_k: Vec<(String, f64)>,
}

// ---------------------------------------------------------------------------
// SamplingConfig
// ---------------------------------------------------------------------------

/// Configuration for sampling during text generation.
#[derive(Clone, Debug)]
pub struct SamplingConfig {
    /// Softmax temperature. <1.0 = sharper, >1.0 = flatter.
    pub temperature: f64,
    /// Top-k filtering. 0 = disabled.
    pub top_k: usize,
    /// Nucleus sampling threshold. 1.0 = disabled.
    pub top_p: f64,
    /// Penalty for already-generated tokens. 1.0 = no penalty.
    pub repetition_penalty: f64,
    /// If true, always pick argmax. If false, sample from distribution.
    pub deterministic: bool,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            deterministic: true,
        }
    }
}

// ---------------------------------------------------------------------------
// GenerationResult
// ---------------------------------------------------------------------------

/// Result of text generation.
#[derive(Clone, Debug)]
pub struct GenerationResult {
    pub prompt: String,
    pub generated: Vec<String>,
    pub full_text: String,
    pub predictions: Vec<TokenPrediction>,
}

impl GenerationResult {
    /// Average omega across all predictions.
    pub fn avg_omega(&self) -> f64 {
        if self.predictions.is_empty() { return 0.0; }
        self.predictions.iter().map(|p| p.omega).sum::<f64>() / self.predictions.len() as f64
    }

    /// Average phi across all predictions.
    pub fn avg_phi(&self) -> f64 {
        if self.predictions.is_empty() { return 0.0; }
        self.predictions.iter().map(|p| p.phi).sum::<f64>() / self.predictions.len() as f64
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Softmax function.
pub fn softmax(scores: &[f64]) -> Vec<f64> {
    if scores.is_empty() { return vec![]; }
    let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum > 0.0 {
        exps.iter().map(|&e| e / sum).collect()
    } else {
        vec![1.0 / scores.len() as f64; scores.len()]
    }
}

/// Compute omega (cohesion) from a probability distribution.
///
/// Omega measures how peaked the distribution is.
/// Ω = 1 when one token dominates (max confidence).
/// Ω → 0 when distribution is uniform (no confidence).
pub fn compute_omega(probs: &[f64]) -> f64 {
    if probs.is_empty() { return 0.0; }
    let n = probs.len() as f64;
    if n <= 1.0 { return 1.0; }

    // Shannon entropy
    let h: f64 = probs.iter()
        .filter(|&&p| p > 1e-15)
        .map(|&p| -p * p.ln())
        .sum();
    let h_max = n.ln();

    if h_max <= 0.0 { return 1.0; }
    (1.0 - h / h_max).clamp(0.0, 1.0)
}

/// Compute phi (clarity) from a probability distribution.
///
/// Phi measures how separated the top prediction is from alternatives.
/// Based on the ratio between top-1 and top-2 probabilities.
pub fn compute_phi(probs: &[f64]) -> f64 {
    if probs.len() < 2 { return 1.0; }

    let mut sorted: Vec<f64> = probs.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let top1 = sorted[0];
    let top2 = sorted[1];

    if top1 <= 0.0 { return 0.0; }

    // Phi = margin between top-1 and top-2
    let margin = (top1 - top2) / top1;
    margin.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let scores = vec![1.0, 2.0, 3.0];
        let probs = softmax(&scores);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_omega_phi() {
        // Peaked distribution → high omega, high phi
        let peaked = vec![0.9, 0.05, 0.03, 0.02];
        assert!(compute_omega(&peaked) > 0.5);
        assert!(compute_phi(&peaked) > 0.5);

        // Uniform distribution → low omega
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        assert!(compute_omega(&uniform) < 0.1);
        assert!(compute_phi(&uniform) < 0.1);
    }
}
