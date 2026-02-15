//! Style Fingerprint — zero-shot authorship and style analysis.
//!
//! Builds a style fingerprint from a set of documents by aggregating
//! wave resonance statistics. No training required — the fingerprint
//! is derived directly from the physics of wave propagation.
//!
//! Use cases:
//!   - Authorship identification (compare fingerprints)
//!   - Style change detection (fingerprint drift)
//!   - Anomaly scoring (deviation from expected fingerprint)

use num_complex::Complex64;

use crate::wave::Wave;
use crate::types::ComplexVec;
use crate::collapse::CollapseMetrics;
use super::reservoir::{WaveResonanceReservoir, ReservoirState};
use super::generator::{compute_omega, compute_phi, softmax};
use super::sps::SemanticPhaseSpace;
use super::readout::ReadoutLayer;

// ---------------------------------------------------------------------------
// StyleFingerprint
// ---------------------------------------------------------------------------

/// A style fingerprint capturing the characteristic wave patterns of a text corpus.
///
/// Built by aggregating reservoir statistics across documents.
/// No training required — pure physics-based statistics.
#[derive(Clone, Debug)]
pub struct StyleFingerprint {
    /// Average resonance pattern (wave amplitudes)
    pub signature: Wave,
    /// Average energy distribution across scales [scale_0, scale_1, scale_2]
    pub scale_energies: [f64; 3],
    /// Average phase coherence across documents
    pub coherence: f64,
    /// Omega statistics: (mean, std_dev)
    pub omega_stats: (f64, f64),
    /// Phi statistics: (mean, std_dev)
    pub phi_stats: (f64, f64),
    /// Number of documents used to build the fingerprint
    pub n_documents: usize,
    /// Dimension of the wave space
    pub dim: usize,
}

impl StyleFingerprint {
    /// Build a fingerprint from a collection of reservoir states and their
    /// corresponding prediction probabilities.
    ///
    /// `states`: reservoir states for each document
    /// `probs_list`: prediction probability distributions for each document
    pub fn from_states(
        states: &[ReservoirState],
        probs_list: &[Vec<f64>],
    ) -> Self {
        let n = states.len();
        if n == 0 {
            return Self::empty(64);
        }

        let dim = states[0].merged.dim;

        // Accumulate signature (average merged wave)
        let mut sig_data = vec![Complex64::new(0.0, 0.0); dim];
        let mut total_energies = [0.0f64; 3];
        let mut total_coherence = 0.0;
        let mut omegas = Vec::with_capacity(n);
        let mut phis = Vec::with_capacity(n);

        for (i, state) in states.iter().enumerate() {
            // Accumulate merged wave
            for (j, &amp) in state.merged.amplitudes.data.iter().enumerate() {
                if j < dim {
                    sig_data[j] += amp;
                }
            }

            // Accumulate scale energies
            for (s, scale) in state.scales.iter().enumerate() {
                if s < 3 {
                    let energy: f64 = scale.amplitudes.data.iter()
                        .map(|c| c.norm_sqr())
                        .sum();
                    total_energies[s] += energy;
                }
            }

            // Accumulate coherence
            total_coherence += state.phase_coherence;

            // Compute Ω/Φ from prediction probabilities
            if i < probs_list.len() && !probs_list[i].is_empty() {
                omegas.push(compute_omega(&probs_list[i]));
                phis.push(compute_phi(&probs_list[i]));
            }
        }

        // Average signature
        let n_f64 = n as f64;
        for v in sig_data.iter_mut() {
            *v /= n_f64;
        }

        let signature = Wave {
            amplitudes: ComplexVec::new(sig_data).normalized(),
            t: 0.0,
            density: None,
            dim,
            metrics: CollapseMetrics::new(),
        };

        // Average energies
        let scale_energies = [
            total_energies[0] / n_f64,
            total_energies[1] / n_f64,
            total_energies[2] / n_f64,
        ];

        // Average coherence
        let coherence = total_coherence / n_f64;

        // Omega stats
        let omega_stats = compute_mean_std(&omegas);
        let phi_stats = compute_mean_std(&phis);

        Self {
            signature,
            scale_energies,
            coherence,
            omega_stats,
            phi_stats,
            n_documents: n,
            dim,
        }
    }

    /// Build a fingerprint from raw documents using reservoir processing.
    pub fn from_documents(
        documents: &[&str],
        sps: &SemanticPhaseSpace,
        reservoir: &mut WaveResonanceReservoir,
        readout: Option<&ReadoutLayer>,
    ) -> Self {
        let mut states = Vec::with_capacity(documents.len());
        let mut probs_list = Vec::with_capacity(documents.len());

        for doc in documents {
            // Tokenize and convert to waves
            let words: Vec<&str> = doc.split_whitespace().collect();
            let mut token_ids = Vec::new();
            for word in &words {
                // Simple word lookup in SPS embeddings via position
                let lower = word.to_lowercase();
                // Use a hash-based approach for token ID
                let id = simple_hash(&lower, sps.vocab_size);
                token_ids.push(id);
            }

            let token_waves = sps.tokens_to_waves(&token_ids);
            let state = reservoir.process_sequence(&token_waves);

            // Get prediction probabilities if readout is available
            let probs = if let Some(ro) = readout {
                let features = state.to_feature_vector();
                let scores = ro.forward(&features);
                softmax(&scores)
            } else {
                // Use merged wave probabilities as fallback
                state.merged.probabilities()
            };

            probs_list.push(probs);
            states.push(state);
        }

        Self::from_states(&states, &probs_list)
    }

    /// Compute similarity between two fingerprints.
    ///
    /// Returns a value in [0, 1] where 1 = identical style.
    pub fn similarity(&self, other: &StyleFingerprint) -> f64 {
        let dim = self.dim.min(other.dim);
        if dim == 0 {
            return 0.0;
        }

        // 1. Wave signature cosine similarity
        let mut dot = Complex64::new(0.0, 0.0);
        let mut norm_a = 0.0f64;
        let mut norm_b = 0.0f64;

        for i in 0..dim {
            let a = self.signature.amplitudes.data.get(i)
                .copied().unwrap_or(Complex64::new(0.0, 0.0));
            let b = other.signature.amplitudes.data.get(i)
                .copied().unwrap_or(Complex64::new(0.0, 0.0));
            dot += a * b.conj();
            norm_a += a.norm_sqr();
            norm_b += b.norm_sqr();
        }

        let wave_sim = if norm_a > 1e-10 && norm_b > 1e-10 {
            dot.norm() / (norm_a.sqrt() * norm_b.sqrt())
        } else {
            0.0
        };

        // 2. Scale energy distribution similarity (cosine)
        let e_dot: f64 = self.scale_energies.iter()
            .zip(other.scale_energies.iter())
            .map(|(a, b)| a * b)
            .sum();
        let e_norm_a: f64 = self.scale_energies.iter().map(|x| x * x).sum::<f64>().sqrt();
        let e_norm_b: f64 = other.scale_energies.iter().map(|x| x * x).sum::<f64>().sqrt();
        let energy_sim = if e_norm_a > 1e-10 && e_norm_b > 1e-10 {
            (e_dot / (e_norm_a * e_norm_b)).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // 3. Coherence similarity
        let coherence_sim = 1.0 - (self.coherence - other.coherence).abs();

        // 4. Omega/Phi overlap
        let omega_overlap = 1.0 - (self.omega_stats.0 - other.omega_stats.0).abs().min(1.0);
        let phi_overlap = 1.0 - (self.phi_stats.0 - other.phi_stats.0).abs().min(1.0);

        // Weighted combination (Φ is primary certainty metric)
        let sim = 0.4 * wave_sim
            + 0.2 * energy_sim
            + 0.15 * coherence_sim
            + 0.1 * omega_overlap
            + 0.15 * phi_overlap;

        sim.clamp(0.0, 1.0)
    }

    /// Compute anomaly score for a text relative to this fingerprint.
    ///
    /// Returns a z-score-like value. Higher = more anomalous.
    pub fn anomaly_score(&self, state: &ReservoirState, probs: &[f64]) -> f64 {
        let omega = compute_omega(probs);
        let phi = compute_phi(probs);

        let omega_z = if self.omega_stats.1 > 1e-10 {
            ((omega - self.omega_stats.0) / self.omega_stats.1).abs()
        } else {
            0.0
        };

        let phi_z = if self.phi_stats.1 > 1e-10 {
            ((phi - self.phi_stats.0) / self.phi_stats.1).abs()
        } else {
            0.0
        };

        let coherence_diff = (state.phase_coherence - self.coherence).abs();
        let coherence_z = coherence_diff / self.coherence.max(0.1);

        // Combined score
        (omega_z + phi_z + coherence_z) / 3.0
    }

    /// Create an empty fingerprint.
    pub fn empty(dim: usize) -> Self {
        let data = vec![Complex64::new(1.0 / (dim as f64).sqrt(), 0.0); dim];
        Self {
            signature: Wave {
                amplitudes: ComplexVec::new(data),
                t: 0.0,
                density: None,
                dim,
                metrics: CollapseMetrics::new(),
            },
            scale_energies: [0.0; 3],
            coherence: 0.0,
            omega_stats: (0.0, 1.0),
            phi_stats: (0.0, 1.0),
            n_documents: 0,
            dim,
        }
    }
}

// ---------------------------------------------------------------------------
// AnomalyScore — detailed anomaly analysis
// ---------------------------------------------------------------------------

/// Detailed anomaly analysis result.
#[derive(Clone, Debug)]
pub struct AnomalyScore {
    /// Overall anomaly score (0 = normal, higher = more anomalous)
    pub score: f64,
    /// Is this considered anomalous?
    pub is_anomalous: bool,
    /// Omega z-score
    pub omega_z: f64,
    /// Phi z-score
    pub phi_z: f64,
    /// Coherence deviation
    pub coherence_deviation: f64,
    /// Energy deviation
    pub energy_deviation: f64,
}

impl AnomalyScore {
    /// Compute anomaly score from a reservoir state and fingerprint.
    pub fn compute(
        state: &ReservoirState,
        probs: &[f64],
        fingerprint: &StyleFingerprint,
        threshold: f64,
    ) -> Self {
        let omega = compute_omega(probs);
        let phi = compute_phi(probs);

        let omega_z = if fingerprint.omega_stats.1 > 1e-10 {
            ((omega - fingerprint.omega_stats.0) / fingerprint.omega_stats.1).abs()
        } else {
            0.0
        };

        let phi_z = if fingerprint.phi_stats.1 > 1e-10 {
            ((phi - fingerprint.phi_stats.0) / fingerprint.phi_stats.1).abs()
        } else {
            0.0
        };

        let coherence_deviation = (state.phase_coherence - fingerprint.coherence).abs()
            / fingerprint.coherence.max(0.1);

        let norm_energy = state.resonance_energy.min(10.0) / 10.0;
        let expected_energy: f64 = fingerprint.scale_energies.iter().sum::<f64>() / 3.0;
        let energy_deviation = if expected_energy > 1e-10 {
            (norm_energy - expected_energy).abs() / expected_energy.max(0.1)
        } else {
            0.0
        };

        let score = (omega_z + phi_z + coherence_deviation + energy_deviation) / 4.0;
        let score = score.min(10.0);

        Self {
            score,
            is_anomalous: score > threshold,
            omega_z,
            phi_z,
            coherence_deviation,
            energy_deviation,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute mean and standard deviation.
fn compute_mean_std(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 1.0);
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter()
        .map(|&v| (v - mean) * (v - mean))
        .sum::<f64>() / n;
    let std = variance.sqrt().max(1e-6);
    (mean, std)
}

/// Simple deterministic hash for word → token ID mapping.
fn simple_hash(word: &str, vocab_size: usize) -> usize {
    let mut hash: usize = 5381;
    for byte in word.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(byte as usize);
    }
    (hash % vocab_size.saturating_sub(4)) + 4 // skip special tokens 0-3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fingerprint_self_similarity() {
        let dim = 16;
        let fp = StyleFingerprint {
            signature: Wave {
                amplitudes: ComplexVec::new(
                    (0..dim).map(|i| Complex64::from_polar(1.0 / (dim as f64).sqrt(), i as f64 * 0.3))
                        .collect()
                ).normalized(),
                t: 0.0,
                density: None,
                dim,
                metrics: CollapseMetrics::new(),
            },
            scale_energies: [0.3, 0.5, 0.2],
            coherence: 0.7,
            omega_stats: (0.5, 0.1),
            phi_stats: (0.4, 0.1),
            n_documents: 10,
            dim,
        };

        let sim = fp.similarity(&fp);
        assert!(sim > 0.95, "self-similarity should be ~1.0: {}", sim);
    }

    #[test]
    fn test_fingerprint_different() {
        let dim = 16;
        let fp1 = StyleFingerprint {
            signature: Wave {
                amplitudes: ComplexVec::new(
                    (0..dim).map(|i| Complex64::from_polar(1.0 / (dim as f64).sqrt(), i as f64 * 0.1))
                        .collect()
                ).normalized(),
                t: 0.0,
                density: None,
                dim,
                metrics: CollapseMetrics::new(),
            },
            scale_energies: [0.8, 0.1, 0.1],
            coherence: 0.9,
            omega_stats: (0.8, 0.05),
            phi_stats: (0.7, 0.05),
            n_documents: 10,
            dim,
        };

        let fp2 = StyleFingerprint {
            signature: Wave {
                amplitudes: ComplexVec::new(
                    (0..dim).map(|i| Complex64::from_polar(1.0 / (dim as f64).sqrt(), i as f64 * 2.0))
                        .collect()
                ).normalized(),
                t: 0.0,
                density: None,
                dim,
                metrics: CollapseMetrics::new(),
            },
            scale_energies: [0.1, 0.1, 0.8],
            coherence: 0.2,
            omega_stats: (0.2, 0.3),
            phi_stats: (0.1, 0.3),
            n_documents: 10,
            dim,
        };

        let sim = fp1.similarity(&fp2);
        assert!(sim < 0.8, "different fingerprints should have lower similarity: {}", sim);
    }

    #[test]
    fn test_mean_std() {
        let values = vec![2.0, 4.0, 6.0, 8.0];
        let (mean, std) = compute_mean_std(&values);
        assert!((mean - 5.0).abs() < 1e-6);
        assert!((std - 2.2360679).abs() < 0.01); // sqrt(5)
    }
}
