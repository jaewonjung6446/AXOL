//! Collapse metrics — measuring computation by collapses, not time.
//!
//! In AXOL, time is not a fundamental concept. Instead, we measure:
//!   C = collapse count (how many quantum states were forced to definite values)
//!   I = information retained (how much of the quantum state survived)
//!   R = relation count (how many coherences exist between states)
//!
//! A program that achieves the same result with fewer collapses is
//! not "faster" — it is more informationally efficient.

use crate::types::DensityMatrix;
use crate::density;

// ---------------------------------------------------------------------------
// CollapseMetrics — the replacement for time-based measurement
// ---------------------------------------------------------------------------

/// Tracks collapse events and information flow through an AXOL program.
///
/// This replaces time complexity with collapse complexity:
///   - Classical bit operation: 1 full collapse per bit
///   - AXOL observe: 1 full collapse
///   - AXOL glimpse: fractional collapse (0 < gamma < 1)
///   - AXOL gaze: 0 collapses (read density matrix directly)
#[derive(Clone, Debug)]
pub struct CollapseMetrics {
    /// Number of full collapses (Born rule applications)
    pub collapses: usize,
    /// Fractional collapses from partial measurements (sum of gammas)
    pub partial_collapses: f64,
    /// Information retained: 1.0 = pure state (no info lost), 0.0 = maximally mixed
    pub information_retained: f64,
    /// Number of relations (off-diagonal coherences above threshold)
    pub relations: usize,
    /// Effective rank of the density matrix (number of significant eigenvalues)
    pub effective_rank: usize,
}

impl CollapseMetrics {
    pub fn new() -> Self {
        Self {
            collapses: 0,
            partial_collapses: 0.0,
            information_retained: 1.0,
            relations: 0,
            effective_rank: 0,
        }
    }

    /// Total collapse budget spent: full collapses + partial collapses
    pub fn total_collapse_cost(&self) -> f64 {
        self.collapses as f64 + self.partial_collapses
    }

    /// Record a full collapse event (observe)
    pub fn record_collapse(&mut self) {
        self.collapses += 1;
    }

    /// Record a partial collapse event (glimpse)
    pub fn record_glimpse(&mut self, gamma: f64) {
        self.partial_collapses += gamma;
    }

    /// Update information metrics from a density matrix
    pub fn update_from_density(&mut self, rho: &DensityMatrix) {
        self.information_retained = density::phi_from_purity(rho);
        self.relations = count_relations(rho);
        self.effective_rank = effective_rank(rho);
    }
}

impl Default for CollapseMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for CollapseMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "C={:.2}  I={:.3}  R={}  rank={}",
            self.total_collapse_cost(),
            self.information_retained,
            self.relations,
            self.effective_rank,
        )
    }
}

// ---------------------------------------------------------------------------
// Information measures
// ---------------------------------------------------------------------------

/// Count significant off-diagonal relations (coherences above threshold)
pub fn count_relations(rho: &DensityMatrix) -> usize {
    let dim = rho.dim;
    let mut count = 0;
    for i in 0..dim {
        for j in (i + 1)..dim {
            if rho.get(i, j).norm() > 1e-10 {
                count += 1;
            }
        }
    }
    count
}

/// Effective rank: number of eigenvalues above threshold.
/// Lower rank = more collapsed, higher rank = more superposition.
pub fn effective_rank(rho: &DensityMatrix) -> usize {
    let eigenvalues = density::eigenvalues(rho);
    eigenvalues.iter().filter(|&&ev| ev > 1e-10).count()
}

/// Focus probabilities by inverse temperature: gamma ∈ [0, 1]
///   gamma = 0.0 → no change (full superposition)
///   gamma = 0.5 → moderate focusing
///   gamma = 1.0 → full collapse (one-hot on argmax)
///
/// Equivalent to softmax with inverse temperature β = 1/(1-γ).
/// This is the computational analog of partial measurement.
pub fn focus_probabilities(probs: &[f64], gamma: f64) -> Vec<f64> {
    if gamma <= 0.0 {
        return probs.to_vec();
    }
    if gamma >= 1.0 {
        let mut result = vec![0.0; probs.len()];
        let max_idx = probs.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        result[max_idx] = 1.0;
        return result;
    }

    let beta = 1.0 / (1.0 - gamma);
    let powered: Vec<f64> = probs.iter()
        .map(|&p| if p > 0.0 { p.powf(beta) } else { 0.0 })
        .collect();
    let sum: f64 = powered.iter().sum();
    if sum > 0.0 {
        powered.iter().map(|&p| p / sum).collect()
    } else {
        probs.to_vec()
    }
}
