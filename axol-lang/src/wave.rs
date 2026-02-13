//! Wave — the native AXOL variable type.
//!
//! A Wave carries a quantum state (complex amplitudes + phases) that can
//! flow through computation without collapsing. Collapse is a choice,
//! not a requirement.
//!
//! Collapse spectrum:
//!   t = 0.0  -> pure superposition (all possibilities, full interference)
//!   t in (0,1) -> partial collapse (some coherence lost)
//!   t = 1.0  -> fully collapsed (classical value)
//!
//! Key insight: on classical hardware, we represent superposition as
//! complex float arrays. This is not simulation of a quantum computer —
//! it is using quantum *mathematics* as a computational tool.
//! The Wave variable gives us interference, phase, and partial collapse
//! that classical variables cannot express.

use num_complex::Complex64;
use crate::types::*;
use crate::ops;
use crate::density;
use crate::collapse::CollapseMetrics;
use crate::errors::{AxolError, Result};

// ---------------------------------------------------------------------------
// InterferencePattern — how relations define Wave composition
// ---------------------------------------------------------------------------

/// Interference pattern determined by the relation declaration.
///
/// This is the core of "relations define computation":
/// the declared relation between variables determines HOW their Waves
/// interfere to produce the output Wave.
#[derive(Clone, Debug, PartialEq)]
pub enum InterferencePattern {
    /// <~> Proportional — constructive interference (aligned phases).
    /// Both waves reinforce each other. Output amplitude = |a + b|.
    Constructive,

    /// <+> Additive — geometric mean of amplitudes.
    /// Both inputs contribute equally. Output_i = sqrt(|a_i| * |b_i|) * exp(i*(θa+θb)/2).
    Additive,

    /// <*> Multiplicative — element-wise amplitude product, phase sum.
    /// Both must be strong for output to be strong. Output = a_i * b_i.
    Multiplicative,

    /// <!> Inverse — destructive interference (pi phase shift).
    /// Waves cancel each other. Output amplitude = |a - b|.
    Destructive,

    /// <?> Conditional — phase coupling.
    /// a's amplitudes are rotated by b's phases. Output = a_i * exp(i * arg(b_i)).
    Conditional,
}

impl InterferencePattern {
    /// Convert from RelationKind to InterferencePattern.
    pub fn from_relation(kind: &crate::declare::RelationKind) -> Self {
        use crate::declare::RelationKind;
        match kind {
            RelationKind::Proportional => Self::Constructive,
            RelationKind::Additive => Self::Additive,
            RelationKind::Multiplicative => Self::Multiplicative,
            RelationKind::Inverse => Self::Destructive,
            RelationKind::Conditional => Self::Conditional,
        }
    }

    /// Phase offset for this pattern.
    pub fn phase_offset(&self) -> f64 {
        match self {
            Self::Constructive => 0.0,
            Self::Additive => 0.0,
            Self::Multiplicative => 0.0,
            Self::Destructive => std::f64::consts::PI,
            Self::Conditional => 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// InterferenceRule — produced by weave from declarations
// ---------------------------------------------------------------------------

/// A single interference rule, derived from a `relate` declaration.
///
/// During computation, source Waves are composed according to the pattern
/// to produce the output Wave.
#[derive(Clone, Debug)]
pub struct InterferenceRule {
    pub output: String,
    pub sources: Vec<String>,
    pub pattern: InterferencePattern,
}

// ---------------------------------------------------------------------------
// Wave — the native AXOL variable
// ---------------------------------------------------------------------------

/// The native AXOL variable.
///
/// Not a classical value (always definite). Not a qubit (physical).
/// A Wave is a mathematical object that carries:
///   - Complex amplitudes (magnitude + phase per dimension)
///   - A collapse level t in [0, 1]
///   - Density matrix (for mixed/partially-collapsed states)
///   - Accumulated collapse metrics
///
/// Waves flow between tapestries without collapsing.
/// Only `observe()` forces a classical output (t -> 1.0).
#[derive(Clone, Debug)]
pub struct Wave {
    /// Complex amplitudes |psi>
    pub amplitudes: ComplexVec,
    /// Collapse level: 0.0 (pure) to 1.0 (collapsed)
    pub t: f64,
    /// Density matrix (Some for mixed states after partial collapse)
    pub density: Option<DensityMatrix>,
    /// Dimension of the Hilbert space
    pub dim: usize,
    /// Accumulated collapse metrics
    pub metrics: CollapseMetrics,
}

impl Wave {
    // =======================================================================
    // Construction
    // =======================================================================

    /// Create a Wave from classical input. Starts at t=0.0 (pure superposition).
    pub fn from_classical(input: &FloatVec) -> Self {
        let cv = ComplexVec::from_real(input).normalized();
        let dim = cv.dim();
        Self {
            amplitudes: cv,
            t: 0.0,
            density: None,
            dim,
            metrics: CollapseMetrics::new(),
        }
    }

    /// Create from complex amplitudes (normalized). t=0.0.
    pub fn from_complex(amplitudes: ComplexVec) -> Self {
        let dim = amplitudes.dim();
        Self {
            amplitudes: amplitudes.normalized(),
            t: 0.0,
            density: None,
            dim,
            metrics: CollapseMetrics::new(),
        }
    }

    /// Create from density matrix (mixed state). t reflects purity.
    pub fn from_density(rho: DensityMatrix) -> Self {
        let dim = rho.dim;
        let purity = rho.purity();
        // t = 0 for pure, approaches 1 for maximally mixed
        let min_purity = 1.0 / dim as f64;
        let t = if purity >= 1.0 - 1e-10 {
            0.0
        } else {
            1.0 - ((purity - min_purity) / (1.0 - min_purity)).clamp(0.0, 1.0)
        };

        // Extract amplitudes from diagonal
        let diag = rho.diagonal();
        let data: Vec<Complex64> = diag.iter()
            .map(|&p| Complex64::new(if p > 0.0 { p.sqrt() } else { 0.0 }, 0.0))
            .collect();

        let mut metrics = CollapseMetrics::new();
        metrics.update_from_density(&rho);

        Self {
            amplitudes: ComplexVec::new(data),
            t,
            density: Some(rho),
            dim,
            metrics,
        }
    }

    /// Create a fully collapsed Wave (t=1.0) from a classical index.
    pub fn collapsed(dim: usize, index: usize) -> Self {
        let mut data = vec![Complex64::new(0.0, 0.0); dim];
        if index < dim {
            data[index] = Complex64::new(1.0, 0.0);
        }
        let mut metrics = CollapseMetrics::new();
        metrics.record_collapse();
        Self {
            amplitudes: ComplexVec::new(data),
            t: 1.0,
            density: None,
            dim,
            metrics,
        }
    }

    // =======================================================================
    // Reading (non-destructive)
    // =======================================================================

    /// Born rule probabilities. Does NOT collapse — just reads.
    pub fn probabilities(&self) -> Vec<f64> {
        if let Some(ref rho) = self.density {
            rho.diagonal()
        } else {
            let norms: Vec<f64> = self.amplitudes.data.iter()
                .map(|c| c.norm_sqr())
                .collect();
            let total: f64 = norms.iter().sum();
            if total > 0.0 {
                norms.iter().map(|&p| p / total).collect()
            } else {
                vec![1.0 / self.dim as f64; self.dim]
            }
        }
    }

    /// Gaze — read the probability distribution without any collapse.
    /// The Wave is unchanged. C = 0.
    pub fn gaze(&self) -> Vec<f64> {
        self.probabilities()
    }

    /// Get density matrix (compute from pure state if needed).
    pub fn to_density(&self) -> DensityMatrix {
        if let Some(ref rho) = self.density {
            rho.clone()
        } else {
            DensityMatrix::from_pure_state(&self.amplitudes)
        }
    }

    /// Is this wave in pure superposition (no collapse)?
    pub fn is_pure(&self) -> bool {
        self.t <= 0.0
    }

    /// Is this wave fully collapsed?
    pub fn is_collapsed(&self) -> bool {
        self.t >= 1.0
    }

    /// Dominant index (highest probability) without collapsing.
    pub fn dominant(&self) -> usize {
        let probs = self.probabilities();
        probs.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    // =======================================================================
    // Operations (collapse-free, C = 0)
    // =======================================================================

    /// Transform through a matrix. Preserves t, no collapse. C = 0.
    pub fn transform(&self, matrix: &TransMatrix) -> Result<Self> {
        let result = ops::transform_complex(&self.amplitudes, matrix)?;
        let result = clamp_complex(&result);

        let new_density = if let Some(ref rho) = self.density {
            Some(ops::evolve_density(rho, matrix)?)
        } else {
            None
        };

        Ok(Self {
            amplitudes: result.normalized(),
            t: self.t,
            density: new_density,
            dim: self.dim,
            metrics: self.metrics.clone(),
        })
    }

    /// Compose two Waves according to an interference pattern. C = 0.
    ///
    /// This is the core operation: relations define how Waves interfere.
    /// The interference pattern (from declare) determines the composition rule.
    pub fn compose(a: &Wave, b: &Wave, pattern: &InterferencePattern) -> Result<Wave> {
        if a.dim != b.dim {
            return Err(AxolError::DimensionMismatch {
                expected: a.dim,
                got: b.dim,
            });
        }
        let dim = a.dim;

        let result_data: Vec<Complex64> = match pattern {
            InterferencePattern::Constructive => {
                // a + b: constructive interference
                a.amplitudes.data.iter()
                    .zip(b.amplitudes.data.iter())
                    .map(|(ai, bi)| ai + bi)
                    .collect()
            }
            InterferencePattern::Additive => {
                // Geometric mean: sqrt(|a|*|b|) with averaged phase
                // Unlike Constructive (a+b), this gives equal weight
                // without one input dominating the other.
                a.amplitudes.data.iter()
                    .zip(b.amplitudes.data.iter())
                    .map(|(ai, bi)| {
                        let mag = (ai.norm() * bi.norm()).sqrt();
                        let phase = (ai.arg() + bi.arg()) / 2.0;
                        Complex64::from_polar(mag, phase)
                    })
                    .collect()
            }
            InterferencePattern::Multiplicative => {
                // a_i * b_i: element-wise product
                a.amplitudes.data.iter()
                    .zip(b.amplitudes.data.iter())
                    .map(|(ai, bi)| ai * bi)
                    .collect()
            }
            InterferencePattern::Destructive => {
                // a - b: destructive interference
                a.amplitudes.data.iter()
                    .zip(b.amplitudes.data.iter())
                    .map(|(ai, bi)| ai - bi)
                    .collect()
            }
            InterferencePattern::Conditional => {
                // a_i * exp(i * arg(b_i)): phase coupling
                a.amplitudes.data.iter()
                    .zip(b.amplitudes.data.iter())
                    .map(|(ai, bi)| {
                        let phase = bi.arg();
                        ai * Complex64::from_polar(1.0, phase)
                    })
                    .collect()
            }
        };

        let result_amps = ComplexVec::new(result_data).normalized();

        // Collapse level = max of inputs (more collapsed dominates)
        let new_t = a.t.max(b.t);

        // Merge metrics (additive)
        let mut metrics = CollapseMetrics::new();
        metrics.collapses = a.metrics.collapses + b.metrics.collapses;
        metrics.partial_collapses = a.metrics.partial_collapses + b.metrics.partial_collapses;

        // Compute density if either input had partial collapse
        let new_density = if new_t > 0.0 {
            Some(DensityMatrix::from_pure_state(&result_amps))
        } else {
            None
        };

        Ok(Wave {
            amplitudes: result_amps,
            t: new_t,
            density: new_density,
            dim,
            metrics,
        })
    }

    /// Compose multiple Waves using a chain of interference patterns.
    /// Reduces left-to-right: compose(compose(a, b, p1), c, p2).
    pub fn compose_many(waves: &[&Wave], patterns: &[&InterferencePattern]) -> Result<Wave> {
        if waves.is_empty() {
            return Err(AxolError::Compose("No waves to compose".into()));
        }
        if waves.len() == 1 {
            return Ok(waves[0].clone());
        }
        if patterns.len() < waves.len() - 1 {
            return Err(AxolError::Compose("Not enough interference patterns".into()));
        }

        let mut result = waves[0].clone();
        for (i, &wave) in waves[1..].iter().enumerate() {
            result = Self::compose(&result, wave, patterns[i])?;
        }
        Ok(result)
    }

    // =======================================================================
    // Collapse operations (irreversible)
    // =======================================================================

    /// Focus — partially collapse the Wave. Irreversible.
    ///
    /// Two effects combine:
    ///   1. **Population sharpening**: p_i → p_i^β / Σ p_j^β  (β = 1/(1-γ))
    ///      Concentrates probability toward the dominant state.
    ///   2. **Dephasing**: off-diagonal coherences decay by (1-γ)
    ///      Reduces interference capability.
    ///
    /// gamma in [0, 1]: degree of additional collapse.
    ///   0.0 = no change (full superposition preserved)
    ///   0.5 = moderate focusing (distribution sharpens, coherence halved)
    ///   1.0 = full collapse (one-hot on argmax, zero coherence)
    ///
    /// New t = old_t + (1 - old_t) * gamma
    /// C = gamma
    pub fn focus(&self, gamma: f64) -> Self {
        let gamma = gamma.clamp(0.0, 1.0);
        if gamma <= 0.0 {
            return self.clone();
        }

        let new_t = self.t + (1.0 - self.t) * gamma;

        // --- Step 1: Population sharpening (changes diagonal / probabilities) ---
        let old_probs = self.probabilities();
        let sharpened = crate::collapse::focus_probabilities(&old_probs, gamma);

        // --- Step 2: Build new amplitudes with sharpened magnitudes + decayed phases ---
        let old_phases = self.amplitudes.phases();
        let new_amps: Vec<Complex64> = sharpened.iter().enumerate()
            .map(|(i, &p)| {
                let mag = if p > 0.0 { p.sqrt() } else { 0.0 };
                // Phase decays with collapse (decoherence)
                let phase = if i < old_phases.len() {
                    old_phases[i] * (1.0 - gamma)
                } else {
                    0.0
                };
                Complex64::from_polar(mag, phase)
            })
            .collect();
        let new_amplitudes = ComplexVec::new(new_amps).normalized();

        // --- Step 3: Build dephased density matrix from sharpened state ---
        let rho_sharp = DensityMatrix::from_pure_state(&new_amplitudes);
        let dephased = density::apply_channel(
            &rho_sharp,
            &density::dephasing_channel(gamma, self.dim),
        );

        // --- Step 4: Update metrics ---
        let mut metrics = self.metrics.clone();
        metrics.record_glimpse(gamma);
        metrics.update_from_density(&dephased);

        Self {
            amplitudes: new_amplitudes,
            t: new_t,
            density: Some(dephased),
            dim: self.dim,
            metrics,
        }
    }

    /// Observe — fully collapse to a classical index. t -> 1.0.
    /// Returns (value_index, collapsed_wave).
    /// C = 1 (one full collapse).
    pub fn observe(&self) -> (usize, Self) {
        let probs = self.probabilities();
        let value_index = probs.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let mut collapsed = Self::collapsed(self.dim, value_index);
        collapsed.metrics = self.metrics.clone();
        collapsed.metrics.record_collapse();

        (value_index, collapsed)
    }

    // =======================================================================
    // Basin interaction
    // =======================================================================

    /// Create a Wave from basin structure (superposition of basin states).
    pub fn from_basins(bs: &BasinStructure, input: &FloatVec) -> Self {
        let dim = input.dim();

        // Map input to basin space via sigmoid
        let input_normalized: Vec<f64> = input.data.iter()
            .map(|&v| 1.0 / (1.0 + (-(v as f64)).exp()))
            .collect();

        // Soft assignment weights
        let basin_weights = bs.soft_assignment(&input_normalized);

        // Build superposition from basin structure
        let mut cv_data = vec![Complex64::new(0.0, 0.0); dim];
        for (i, &weight) in basin_weights.iter().enumerate() {
            if i >= bs.centroids.len() { break; }
            let phase = if i < bs.phases.len() { bs.phases[i] } else { 0.0 };
            let component = Complex64::from_polar(weight.sqrt(), phase);

            for d in 0..dim.min(bs.centroids[i].len()) {
                cv_data[d] += component * Complex64::new(bs.centroids[i][d], 0.0);
            }
        }

        // Normalize
        let norm: f64 = cv_data.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        if norm > 1e-15 {
            for c in cv_data.iter_mut() {
                *c /= norm;
            }
        }

        Self {
            amplitudes: ComplexVec::new(cv_data),
            t: 0.0,
            density: None,
            dim,
            metrics: CollapseMetrics::new(),
        }
    }

    // =======================================================================
    // Conversion to legacy types
    // =======================================================================

    /// Convert to FloatVec (amplitude magnitudes).
    pub fn to_float_vec(&self) -> FloatVec {
        self.amplitudes.to_real()
    }

    /// Convert probabilities to FloatVec.
    pub fn to_prob_vec(&self) -> FloatVec {
        FloatVec::new(self.probabilities().iter().map(|&p| p as f32).collect())
    }
}

impl std::fmt::Display for Wave {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let probs = self.probabilities();
        let probs_str: Vec<String> = probs.iter()
            .take(8)
            .map(|p| format!("{:.3}", p))
            .collect();
        let suffix = if self.dim > 8 { ", ..." } else { "" };
        write!(f, "Wave(dim={}, t={:.2}, [{}{}])",
            self.dim, self.t, probs_str.join(", "), suffix)
    }
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
