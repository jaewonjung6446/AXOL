//! Relation — a first-class runtime type in AXOL v2.
//!
//! A Relation is a living connection between two Waves (or other Relations).
//! Unlike v1 where relations only existed at declaration time and were converted
//! to Waves, v2 Relations persist at runtime and support:
//!   - observe with expect: path-level confidence landscape via selective dephasing
//!   - widen: reopen possibilities (inverse of focus)
//!   - conflict detection and resolution
//!
//! Key distinction:
//!   "expect" is NOT a result prediction ("the answer is 0").
//!   "expect" IS a path confidence landscape ("this region is worth exploring").
//!   The mechanism is selective dephasing, not constructive interference.

use num_complex::Complex64;
use crate::wave::{Wave, InterferencePattern};
use crate::dsl::parser::RelDirection;
use crate::types::*;
use crate::density;
use crate::errors::{AxolError, Result};

// ---------------------------------------------------------------------------
// Expectation — confidence landscape over state space
// ---------------------------------------------------------------------------

/// A confidence landscape that controls WHERE to explore, not WHAT the answer is.
///
/// This is fundamentally different from a Bayesian prior:
///   - Prior: "I believe the answer is X" → biases outcome
///   - Landscape: "This region is promising" → controls path pruning
///
/// Mechanism: selective dephasing
///   - High landscape values → preserve coherences → keep paths alive
///   - Low landscape values → dephase coherences → prune paths
///   - Strength controls how aggressively to prune
#[derive(Clone, Debug)]
pub struct Expectation {
    pub name: String,
    /// The confidence landscape: a soft distribution over the state space.
    /// Each value indicates how promising that region is for exploration.
    /// Does NOT need to sum to 1 (will be normalized internally).
    pub landscape: Vec<f64>,
    /// How strongly to apply the landscape. 0 = ignore, 1 = full control.
    pub strength: f64,
}

impl Expectation {
    /// Create from a literal distribution.
    pub fn from_distribution(name: &str, landscape: Vec<f64>, strength: f64) -> Self {
        Self {
            name: name.to_string(),
            landscape,
            strength: strength.clamp(0.0, 1.0),
        }
    }

    /// Create from a Wave's probability distribution.
    /// "Explore where this wave points."
    pub fn from_wave(name: &str, wave: &Wave, strength: f64) -> Self {
        Self {
            name: name.to_string(),
            landscape: wave.probabilities(),
            strength: strength.clamp(0.0, 1.0),
        }
    }

    /// Get the normalized landscape (probabilities sum to 1).
    pub fn normalized_landscape(&self) -> Vec<f64> {
        let sum: f64 = self.landscape.iter().sum();
        if sum > 1e-15 {
            self.landscape.iter().map(|&v| v / sum).collect()
        } else {
            vec![1.0 / self.landscape.len() as f64; self.landscape.len()]
        }
    }
}

// ---------------------------------------------------------------------------
// Selective Dephasing — the core mechanism of expect
// ---------------------------------------------------------------------------

/// Apply selective dephasing to a density matrix using a confidence landscape.
///
/// This is the quantum-mechanical implementation of "path pruning":
///   - For each pair of states (i, j):
///     relevance = landscape[i] * landscape[j]
///     If both states are in promising regions, their coherence is preserved.
///     If either is unpromising, their coherence is dephased (path pruned).
///
///   - Strength controls aggressiveness:
///     0.0 = no dephasing (fully open exploration)
///     1.0 = maximum dephasing (only promising paths survive)
///
/// This is NOT interference. It does NOT predict outcomes.
/// It controls which quantum paths (coherences) remain alive for future computation.
pub fn selective_dephase(
    rho: &DensityMatrix,
    landscape: &[f64],
    strength: f64,
) -> DensityMatrix {
    let dim = rho.dim;
    let strength = strength.clamp(0.0, 1.0);

    // Normalize landscape
    let sum: f64 = landscape.iter().take(dim).sum();
    let norm_landscape: Vec<f64> = if sum > 1e-15 {
        landscape.iter().take(dim).map(|&v| v / sum).collect()
    } else {
        vec![1.0 / dim as f64; dim]
    };

    let mut result = rho.data.clone();

    for i in 0..dim {
        for j in 0..dim {
            if i != j {
                // Relevance: geometric mean of both endpoints' landscape values.
                // High when BOTH states are in promising regions.
                // Scale by dim to normalize (uniform landscape → relevance = 1/dim * dim = 1)
                let relevance = (norm_landscape[i] * norm_landscape[j]).sqrt() * dim as f64;
                let relevance = relevance.clamp(0.0, 1.0);

                // Dephasing rate: high when relevance is low (unpromising paths)
                let dephasing = (1.0 - relevance) * strength;

                // Apply: preserve promising coherences, dephase unpromising ones
                result[i * dim + j] *= Complex64::new(1.0 - dephasing, 0.0);
            }
            // Diagonal (populations) are NEVER modified — we don't predict outcomes
        }
    }

    // Hermitian symmetrize
    for i in 0..dim {
        for j in (i + 1)..dim {
            let avg = (result[i * dim + j] + result[j * dim + i].conj()) * 0.5;
            result[i * dim + j] = avg;
            result[j * dim + i] = avg.conj();
        }
    }

    DensityMatrix::new(result, dim)
}

/// Apply an Expectation to a Wave, returning a new Wave with pruned paths.
///
/// The landscape does not change the probability distribution directly.
/// It changes which COHERENCES survive, affecting future interference capability.
pub fn apply_expectation(wave: &Wave, expect: &Expectation) -> Result<Wave> {
    let rho = wave.to_density();
    let dephased = selective_dephase(&rho, &expect.landscape, expect.strength);

    let mut new_wave = Wave::from_density(dephased);
    // t increases slightly — some information was pruned
    new_wave.t = wave.t + (1.0 - wave.t) * expect.strength * 0.3;
    Ok(new_wave)
}

// ---------------------------------------------------------------------------
// Relation — first-class runtime type
// ---------------------------------------------------------------------------

/// A first-class runtime Relation.
///
/// Holds the connection between two entities (Waves or Relations) and
/// the resulting interference pattern. The Relation's own `wave` represents
/// the current state of the relationship.
#[derive(Clone, Debug)]
pub struct Relation {
    pub name: String,
    pub from: String,
    pub to: String,
    pub direction: RelDirection,
    pub pattern: InterferencePattern,
    pub wave: Wave,
    /// Negativity: how open the relation is. 0 = fully decided, 1 = maximally open.
    pub negativity: f64,
}

impl Relation {
    /// Create a new Relation by composing two Waves.
    pub fn new(
        name: &str,
        from: &str,
        to: &str,
        direction: RelDirection,
        from_wave: &Wave,
        to_wave: &Wave,
        pattern: InterferencePattern,
    ) -> Result<Self> {
        let wave = Wave::compose(from_wave, to_wave, &pattern)?;

        // Initial negativity from Bhattacharyya distance between input waves.
        // bc=1 → identical distributions → negativity=0 (fully decided).
        // bc→0 → disjoint distributions → negativity→1 (maximally open).
        let probs_from = from_wave.probabilities();
        let probs_to = to_wave.probabilities();
        let dim = probs_from.len().min(probs_to.len());
        let mut bc = 0.0;
        for i in 0..dim {
            bc += (probs_from[i] * probs_to[i]).sqrt();
        }
        let negativity = (1.0 - bc).clamp(0.0, 1.0);

        // Apply depolarizing noise proportional to negativity.
        // A relation between very different waves starts with more quantum uncertainty.
        let noise = negativity * 0.3;
        let wave = if noise > 1e-6 {
            let rho = wave.to_density();
            let kraus = density::depolarizing_channel(wave.dim, noise);
            let noisy = density::apply_channel(&rho, &kraus);
            Wave::from_density(noisy)
        } else {
            wave
        };

        Ok(Self {
            name: name.to_string(),
            from: from.to_string(),
            to: to.to_string(),
            direction,
            pattern,
            wave,
            negativity,
        })
    }

    /// Gaze — read the relation's probability distribution without collapse. C = 0.
    pub fn gaze(&self) -> Vec<f64> {
        self.wave.gaze()
    }

    /// Apply an expectation landscape to this relation.
    ///
    /// This does NOT bias the result toward a specific outcome.
    /// It prunes unpromising coherences, narrowing which paths remain
    /// available for future interference and observation.
    pub fn apply_expect(&mut self, expect: &Expectation) -> Result<ExpectResult> {
        let old_probs = self.wave.probabilities();
        let old_negativity = self.negativity;

        // Apply selective dephasing
        let new_wave = apply_expectation(&self.wave, expect)?;
        let new_probs = new_wave.probabilities();

        // Measure how much the landscape aligned with the existing state.
        // If the landscape matched where probability already was → paths confirmed → negativity drops.
        // If the landscape conflicts → more uncertainty → negativity may increase.
        let norm_landscape = expect.normalized_landscape();
        let dim = old_probs.len().min(norm_landscape.len());
        let mut alignment = 0.0;
        for i in 0..dim {
            alignment += (old_probs[i] * norm_landscape[i]).sqrt();
        }
        // alignment ∈ [0, 1]: Bhattacharyya coefficient

        // High alignment + high strength → close faster
        // Low alignment + high strength → pruning conflicts with state → stays open
        if alignment > 0.5 {
            self.negativity = (self.negativity * (1.0 - expect.strength * alignment * 0.5)).clamp(0.0, 1.0);
        } else {
            self.negativity = (self.negativity + expect.strength * (1.0 - alignment) * 0.2).clamp(0.0, 1.0);
        }

        // Count how many coherences survived
        let rho_new = new_wave.to_density();
        let surviving_coherences = count_significant_coherences(&rho_new);

        let negativity_delta = self.negativity - old_negativity;
        self.wave = new_wave.clone();

        let (value_index, _) = new_wave.observe();

        Ok(ExpectResult {
            value_index,
            probabilities: new_probs,
            negativity_delta,
            alignment,
            surviving_coherences,
            wave: new_wave,
        })
    }

    /// Widen — reopen possibilities (the inverse of focus).
    pub fn widen(&mut self, amount: f64) -> Result<()> {
        let amount = amount.clamp(0.0, 1.0);
        if amount <= 0.0 {
            return Ok(());
        }

        let new_t = self.wave.t * (1.0 - amount);
        let rho = self.wave.to_density();
        let kraus = density::depolarizing_channel(self.wave.dim, amount);
        let widened_rho = density::apply_channel(&rho, &kraus);

        let mut new_wave = Wave::from_density(widened_rho);
        new_wave.t = new_t;

        self.negativity = (self.negativity + amount * 0.5).clamp(0.0, 1.0);
        self.wave = new_wave;

        Ok(())
    }

    /// Detect conflict between two Relations.
    pub fn conflict_score(a: &Relation, b: &Relation) -> f64 {
        let probs_a = a.wave.probabilities();
        let probs_b = b.wave.probabilities();

        let dim = probs_a.len().min(probs_b.len());
        let mut bc = 0.0;
        for i in 0..dim {
            bc += (probs_a[i] * probs_b[i]).sqrt();
        }

        let distance = if bc > 1e-15 { -bc.ln() } else { 10.0 };
        (distance / 3.0).clamp(0.0, 1.0)
    }

    /// Create a conflict Relation from two existing Relations.
    pub fn conflict(a: &Relation, b: &Relation) -> Result<Relation> {
        let wave = Wave::compose(&a.wave, &b.wave, &InterferencePattern::Destructive)?;

        // Bhattacharyya distance between the two Relations' wave distributions
        let probs_a = a.wave.probabilities();
        let probs_b = b.wave.probabilities();
        let dim = probs_a.len().min(probs_b.len());
        let mut bc = 0.0;
        for i in 0..dim {
            bc += (probs_a[i] * probs_b[i]).sqrt();
        }
        let negativity = (1.0 - bc).clamp(0.0, 1.0);

        let noise = negativity * 0.3;
        let wave = if noise > 1e-6 {
            let rho = wave.to_density();
            let kraus = density::depolarizing_channel(wave.dim, noise);
            let noisy = density::apply_channel(&rho, &kraus);
            Wave::from_density(noisy)
        } else {
            wave
        };

        Ok(Relation {
            name: format!("{}_conflict_{}", a.name, b.name),
            from: a.name.clone(),
            to: b.name.clone(),
            direction: RelDirection::Conflict,
            pattern: InterferencePattern::Destructive,
            wave,
            negativity,
        })
    }
}

/// Result of applying an expectation landscape.
#[derive(Clone, Debug)]
pub struct ExpectResult {
    pub value_index: usize,
    pub probabilities: Vec<f64>,
    /// Change in negativity. Negative = closing, positive = opening.
    pub negativity_delta: f64,
    /// How well the landscape aligned with the existing state [0, 1].
    pub alignment: f64,
    /// Number of coherences that survived the dephasing.
    pub surviving_coherences: usize,
    pub wave: Wave,
}

/// Count significant off-diagonal coherences above threshold.
fn count_significant_coherences(rho: &DensityMatrix) -> usize {
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

// ---------------------------------------------------------------------------
// Resolve strategies
// ---------------------------------------------------------------------------

pub fn resolve_interfere(wave_a: &Wave, wave_b: &Wave) -> Result<Wave> {
    Wave::compose(wave_a, wave_b, &InterferencePattern::Constructive)
}

pub fn resolve_branch(wave_a: &Wave, wave_b: &Wave) -> Result<Wave> {
    Wave::compose(wave_a, wave_b, &InterferencePattern::Additive)
}

pub fn resolve_rebase(wave_a: &Wave, target: &Wave) -> Result<Wave> {
    Wave::compose(wave_a, target, &InterferencePattern::Conditional)
}

pub fn resolve_superpose(wave_a: &Wave, wave_b: &Wave) -> Result<Wave> {
    if wave_a.dim != wave_b.dim {
        return Err(AxolError::DimensionMismatch {
            expected: wave_a.dim,
            got: wave_b.dim,
        });
    }
    let dim = wave_a.dim;

    let rho_a = wave_a.to_density();
    let rho_b = wave_b.to_density();

    let mut mixed_data = vec![Complex64::new(0.0, 0.0); dim * dim];
    for i in 0..dim * dim {
        mixed_data[i] = rho_a.data[i] * 0.5 + rho_b.data[i] * 0.5;
    }
    let mixed_rho = DensityMatrix::new(mixed_data, dim);
    Ok(Wave::from_density(mixed_rho))
}
