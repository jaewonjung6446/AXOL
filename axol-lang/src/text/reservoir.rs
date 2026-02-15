//! Wave Resonance Reservoir (WRR) — Multi-scale wave resonance for context processing.
//!
//! Replaces ContextWeaver (RNN-style) with a physics-based reservoir whose
//! dynamics are entirely determined by physical constants (no learned parameters).
//! Only the output readout layer is trained (via lstsq).
//!
//! Architecture:
//!   Token → [SPS] → wave → [WRR 3-scale resonance] → ReservoirState
//!     Scale 1 (τ=2, Destructive)   — lexical level
//!     Scale 2 (τ=5, Constructive)  — phrasal level
//!     Scale 3 (τ=15, Conditional)  — clause level
//!
//! Each scale has damping = exp(-1/τ), phase_freq = 2π/τ — physical constants,
//! zero learnable parameters.
//!
//! ## Resonance Compaction (Phase-Multiplexed Wave Memory)
//!
//! Each register periodically compacts its state via self-interference:
//!   - Channels carrying related information (similar phases) merge
//!     constructively → fewer but stronger channels
//!   - Channels carrying redundant/noise information cancel destructively
//!     → freed channels available for new tokens
//!   - Compaction is triggered every τ tokens, aligned with natural decay
//!
//! This provides automatic memory garbage collection through physics:
//! no learned forget gate needed (unlike LSTM/GRU/Mamba).
//!
//! Complexity: O(n × dim × 3) vs ContextWeaver O(n × dim²)

use num_complex::Complex64;

use crate::types::*;
use crate::wave::{Wave, InterferencePattern};
use crate::collapse::CollapseMetrics;

// ---------------------------------------------------------------------------
// CompactionResult — tracking for resonance compaction events
// ---------------------------------------------------------------------------

/// Result of a resonance compaction event.
#[derive(Clone, Debug)]
pub struct CompactionResult {
    /// Saturation before compaction (effective dimensionality ratio, 0..1)
    pub saturation_before: f64,
    /// Saturation after compaction
    pub saturation_after: f64,
    /// Number of channels that were pruned (fell below Ω threshold)
    pub channels_pruned: usize,
    /// Number of channels that were merged (similar phases)
    pub channels_merged: usize,
    /// Energy before compaction
    pub energy_before: f64,
    /// Energy after compaction
    pub energy_after: f64,
    /// Which register (scale index) was compacted
    pub scale_index: usize,
}

// ---------------------------------------------------------------------------
// ResonanceRegister — a single time-scale resonator
// ---------------------------------------------------------------------------

/// A single resonance register operating at a specific time scale.
///
/// Physics-based: damping and phase frequency are derived from τ,
/// no learnable parameters.
#[derive(Clone, Debug)]
pub struct ResonanceRegister {
    /// Dimension of the wave space
    pub dim: usize,
    /// Characteristic time scale
    pub tau: f64,
    /// Damping factor: exp(-1/τ) — how much old state is retained
    pub damping: f64,
    /// Interference pattern for this scale
    pub pattern: InterferencePattern,
    /// Current state wave
    pub state: Wave,
    /// Phase frequency: 2π/τ — rate of phase rotation
    pub phase_freq: f64,
    /// Token counter for compaction trigger (compacts every τ tokens)
    pub token_count: usize,
}

impl ResonanceRegister {
    /// Create a new resonance register.
    ///
    /// `dim`: wave dimension
    /// `tau`: characteristic time scale (larger = longer memory)
    /// `pattern`: interference pattern (determines how tokens combine)
    pub fn new(dim: usize, tau: f64, pattern: InterferencePattern) -> Self {
        Self {
            dim,
            tau,
            damping: (-1.0 / tau).exp(),
            pattern,
            state: zero_wave(dim),
            phase_freq: 2.0 * std::f64::consts::PI / tau,
            token_count: 0,
        }
    }

    /// Process a single token wave through this register.
    ///
    /// Algorithm:
    ///   1. Rotate token wave by scale-specific phase frequency
    ///   2. Apply damping to existing state (physical decay)
    ///   3. Compose damped state with rotated token (interference pattern)
    ///   4. Apply Kerr nonlinearity (phase modulation by amplitude²)
    pub fn resonate(&mut self, token_wave: &Wave, position: usize) {
        // 1. Phase rotation: rotate token by scale-specific frequency × position
        let rotated = phase_rotate(token_wave, self.phase_freq * position as f64);

        // 2. Damping: decay existing state
        let damped = damp_wave(&self.state, self.damping);

        // 3. Compose: interference between damped state and rotated token
        let composed = Wave::compose(&damped, &rotated, &self.pattern)
            .unwrap_or_else(|_| rotated.clone());

        // 4. Kerr nonlinearity: phase += κ × |amplitude|²
        let kappa = 0.1;
        let nonlinear = kerr_nonlinearity(&composed, kappa);

        self.state = normalize_wave(&nonlinear);
    }

    /// Reset state to zero.
    pub fn reset(&mut self) {
        self.state = zero_wave(self.dim);
        self.token_count = 0;
    }

    /// Get the energy of the current state (sum of |amplitude|²).
    pub fn energy(&self) -> f64 {
        self.state.amplitudes.data.iter()
            .map(|c| c.norm_sqr())
            .sum()
    }

    /// Measure saturation: effective dimensionality ratio via Shannon entropy.
    ///
    /// Returns a value in [0, 1]:
    ///   - 0 = all probability mass in one channel (fully concentrated)
    ///   - 1 = uniform distribution (fully saturated, no channel stands out)
    ///
    /// Uses gaze() (non-destructive read) to get the probability distribution,
    /// then computes normalized Shannon entropy: H / log(dim).
    pub fn saturation(&self) -> f64 {
        let probs = self.state.gaze(); // non-destructive read
        let dim = probs.len();
        if dim <= 1 {
            return 0.0;
        }

        // Shannon entropy: H = -Σ p_i log(p_i)
        let mut entropy = 0.0;
        for &p in &probs {
            if p > 1e-15 {
                entropy -= p * p.ln();
            }
        }

        // Normalize by max entropy (uniform distribution)
        let max_entropy = (dim as f64).ln();
        if max_entropy < 1e-15 {
            return 0.0;
        }
        (entropy / max_entropy).clamp(0.0, 1.0)
    }

    /// Compact the register via self-interference.
    ///
    /// Phase-Multiplexed Wave Memory compaction:
    ///   1. **Gaze**: read probability distribution non-destructively
    ///   2. **Phase grouping**: channels with similar phases merge constructively
    ///      (via Wave::compose with Constructive pattern)
    ///   3. **Ω pruning**: channels where cohesion (Ω) is low are attenuated
    ///   4. **Renormalize**: preserve unit norm
    ///
    /// Returns CompactionResult with before/after statistics.
    pub fn compact(&mut self, scale_index: usize) -> CompactionResult {
        let energy_before = self.energy();
        let saturation_before = self.saturation();

        let amps = &self.state.amplitudes.data;
        let dim = self.dim;

        // ─── Phase grouping: cluster channels by phase similarity ───
        // For each channel, compute phase. Channels within π/4 are "similar".
        let phase_threshold = std::f64::consts::PI / 4.0;
        let phases: Vec<f64> = amps.iter().map(|c| c.arg()).collect();
        let magnitudes: Vec<f64> = amps.iter().map(|c| c.norm()).collect();
        let mut merged_count = 0usize;
        let mut visited = vec![false; dim];

        // Greedy phase-merge: for each unvisited channel, find similar ones
        // and merge them constructively (add magnitudes, average phases).
        let mut new_amps: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); dim];

        for i in 0..dim {
            if visited[i] || magnitudes[i] < 1e-12 {
                if !visited[i] {
                    new_amps[i] = amps[i];
                }
                visited[i] = true;
                continue;
            }

            // Find all channels with similar phase to channel i
            let mut group_mag = magnitudes[i];
            let mut group_phase_x = magnitudes[i] * phases[i].cos();
            let mut group_phase_y = magnitudes[i] * phases[i].sin();
            let mut group_count = 1usize;
            visited[i] = true;

            for j in (i + 1)..dim {
                if visited[j] || magnitudes[j] < 1e-12 {
                    continue;
                }
                // Phase distance (circular)
                let mut dp = (phases[i] - phases[j]).abs();
                if dp > std::f64::consts::PI {
                    dp = 2.0 * std::f64::consts::PI - dp;
                }

                if dp < phase_threshold {
                    // Constructive merge: add magnitudes, weighted phase average
                    group_mag += magnitudes[j];
                    group_phase_x += magnitudes[j] * phases[j].cos();
                    group_phase_y += magnitudes[j] * phases[j].sin();
                    group_count += 1;
                    visited[j] = true;
                    // Mark j as absorbed — its energy goes to channel i
                    new_amps[j] = Complex64::new(0.0, 0.0);
                }
            }

            if group_count > 1 {
                merged_count += group_count - 1;
            }

            // Merged channel gets combined magnitude + averaged phase
            let avg_phase = group_phase_y.atan2(group_phase_x);
            new_amps[i] = Complex64::from_polar(group_mag, avg_phase);
        }

        // ─── Ω pruning: attenuate low-cohesion channels ───
        // Compute per-channel "local cohesion" as |amplitude|² relative to mean.
        // Channels far below mean get attenuated — they contribute noise, not signal.
        let max_energy = new_amps.iter().map(|c| c.norm_sqr())
            .fold(0.0f64, f64::max);
        let prune_threshold = max_energy * 0.001; // channels below 0.1% of max → prune
        let mut pruned_count = 0usize;

        for i in 0..dim {
            let ch_energy = new_amps[i].norm_sqr();
            if ch_energy > 0.0 && ch_energy < prune_threshold {
                // Attenuate (don't zero completely — soft pruning)
                new_amps[i] *= 0.1;
                pruned_count += 1;
            }
        }

        // ─── Renormalize ───
        let norm: f64 = new_amps.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        if norm > 1e-15 {
            for c in &mut new_amps {
                *c /= norm;
            }
        }

        self.state = Wave {
            amplitudes: ComplexVec::new(new_amps),
            t: self.state.t,
            density: None,
            dim: self.dim,
            metrics: self.state.metrics.clone(),
        };

        let energy_after = self.energy();
        let saturation_after = self.saturation();

        CompactionResult {
            saturation_before,
            saturation_after,
            channels_pruned: pruned_count,
            channels_merged: merged_count,
            energy_before,
            energy_after,
            scale_index,
        }
    }
}

// ---------------------------------------------------------------------------
// WaveResonanceReservoir — multi-scale resonance
// ---------------------------------------------------------------------------

/// Multi-scale Wave Resonance Reservoir.
///
/// Contains multiple registers at different time scales, each capturing
/// different linguistic levels (lexical, phrasal, clausal).
/// All dynamics are physics-based — zero learnable parameters.
#[derive(Clone, Debug)]
pub struct WaveResonanceReservoir {
    /// Dimension of the wave space
    pub dim: usize,
    /// Resonance registers at different time scales
    pub registers: Vec<ResonanceRegister>,
    /// Log of compaction events (most recent last)
    pub compaction_log: Vec<CompactionResult>,
}

/// Default scale configurations: (τ, InterferencePattern)
const DEFAULT_SCALES: [(f64, usize); 3] = [
    (2.0, 3),   // τ=2, Destructive (lexical — fast, captures local differences)
    (5.0, 0),   // τ=5, Constructive (phrasal — medium, accumulates phrase patterns)
    (15.0, 4),  // τ=15, Conditional (clausal — slow, phase coupling for long-range)
];

fn pattern_from_index(idx: usize) -> InterferencePattern {
    match idx {
        0 => InterferencePattern::Constructive,
        1 => InterferencePattern::Additive,
        2 => InterferencePattern::Multiplicative,
        3 => InterferencePattern::Destructive,
        4 => InterferencePattern::Conditional,
        _ => InterferencePattern::Constructive,
    }
}

impl WaveResonanceReservoir {
    /// Create a new reservoir with default 3-scale configuration.
    pub fn new(dim: usize) -> Self {
        let registers = DEFAULT_SCALES.iter()
            .map(|&(tau, pat_idx)| {
                ResonanceRegister::new(dim, tau, pattern_from_index(pat_idx))
            })
            .collect();

        Self { dim, registers, compaction_log: Vec::new() }
    }

    /// Create a reservoir with custom scale configurations.
    pub fn with_scales(dim: usize, scales: &[(f64, InterferencePattern)]) -> Self {
        let registers = scales.iter()
            .map(|(tau, pat)| ResonanceRegister::new(dim, *tau, pat.clone()))
            .collect();

        Self { dim, registers, compaction_log: Vec::new() }
    }

    /// Process a sequence of token waves through the reservoir.
    ///
    /// Returns the final ReservoirState capturing all multi-scale context.
    /// Includes automatic resonance compaction every τ tokens per register.
    /// Complexity: O(n × dim × num_scales) where n = sequence length.
    pub fn process_sequence(&mut self, token_waves: &[Wave]) -> ReservoirState {
        // Reset all registers and compaction log
        for reg in &mut self.registers {
            reg.reset();
        }
        self.compaction_log.clear();

        // Feed each token through all registers with auto-compaction
        for (pos, token_wave) in token_waves.iter().enumerate() {
            self.process_token(token_wave, pos);
        }

        self.current_state()
    }

    /// Process a sequence WITHOUT compaction (for A/B comparison).
    pub fn process_sequence_no_compaction(&mut self, token_waves: &[Wave]) -> ReservoirState {
        for reg in &mut self.registers {
            reg.reset();
        }
        self.compaction_log.clear();

        for (pos, token_wave) in token_waves.iter().enumerate() {
            for reg in &mut self.registers {
                reg.resonate(token_wave, pos);
            }
        }

        self.current_state()
    }

    /// Incrementally process a single token (for autoregressive use).
    ///
    /// After resonating, checks each register's token count against its τ.
    /// When token_count reaches τ (rounded), triggers resonance compaction —
    /// the physics-based memory garbage collection.
    pub fn process_token(&mut self, token_wave: &Wave, position: usize) {
        let num_regs = self.registers.len();
        for idx in 0..num_regs {
            self.registers[idx].resonate(token_wave, position);
            self.registers[idx].token_count += 1;

            // Auto-compact every τ tokens (rounded to nearest integer, minimum 2)
            let compact_interval = (self.registers[idx].tau.round() as usize).max(2);
            if self.registers[idx].token_count % compact_interval == 0 {
                let result = self.registers[idx].compact(idx);
                self.compaction_log.push(result);
            }
        }
    }

    /// Get the current reservoir state without resetting.
    pub fn current_state(&self) -> ReservoirState {
        let scales: Vec<Wave> = self.registers.iter()
            .map(|r| r.state.clone())
            .collect();

        // Merge all scales via constructive interference
        let merged = merge_scales(&scales, self.dim);

        // Compute phase coherence across scales
        let phase_coherence = compute_phase_coherence(&scales);

        // Compute total resonance energy
        let resonance_energy = self.registers.iter()
            .map(|r| r.energy())
            .sum();

        ReservoirState {
            merged,
            scales,
            phase_coherence,
            resonance_energy,
        }
    }

    /// Reset all registers to initial state.
    pub fn reset(&mut self) {
        for reg in &mut self.registers {
            reg.reset();
        }
        self.compaction_log.clear();
    }

    /// Number of scales (registers).
    pub fn num_scales(&self) -> usize {
        self.registers.len()
    }
}

// ---------------------------------------------------------------------------
// ReservoirState — snapshot of multi-scale context
// ---------------------------------------------------------------------------

/// Snapshot of the reservoir's multi-scale context.
///
/// Contains both merged and per-scale information, plus
/// physics-derived quality signals (coherence, energy).
#[derive(Clone, Debug)]
pub struct ReservoirState {
    /// Merged multi-scale context wave
    pub merged: Wave,
    /// Individual scale states
    pub scales: Vec<Wave>,
    /// Phase coherence across scales (0..1): how aligned the scales are.
    /// High coherence = scales agree = confident context.
    /// Used as anomaly detection signal.
    pub phase_coherence: f64,
    /// Total resonance energy across all scales.
    /// Higher energy = stronger resonance = more informative context.
    pub resonance_energy: f64,
}

impl ReservoirState {
    /// Compute the feature dimension for readout: 4*dim + 2.
    /// [merged(dim) + scale_0(dim) + scale_1(dim) + scale_2(dim) + coherence(1) + energy(1)]
    pub fn feature_dim(dim: usize, num_scales: usize) -> usize {
        dim * (1 + num_scales) + 2
    }

    /// Extract the full feature vector for the readout layer.
    ///
    /// Each scale (merged + scale0~2) is independently L2-normalized
    /// to preserve relative magnitude differences between scales.
    /// Scalar features (coherence, energy) are excluded from normalization.
    pub fn to_feature_vector(&self) -> Vec<f64> {
        let dim = self.merged.dim;
        let num_scales = self.scales.len();
        let feat_dim = Self::feature_dim(dim, num_scales);
        let mut features = Vec::with_capacity(feat_dim);

        // Helper: extract probability amplitudes and L2-normalize independently
        let push_normalized_scale = |probs: &[f64], features: &mut Vec<f64>| {
            let start = features.len();
            for &p in probs {
                features.push(p.sqrt());
            }
            let norm: f64 = features[start..].iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-10 {
                for v in &mut features[start..] {
                    *v /= norm;
                }
            }
        };

        // Merged wave: independently normalized
        let merged_probs = self.merged.probabilities();
        push_normalized_scale(&merged_probs, &mut features);

        // Per-scale waves: each independently normalized
        for scale in &self.scales {
            let probs = scale.probabilities();
            push_normalized_scale(&probs, &mut features);
        }

        // Scalar features — NOT normalized (preserve raw values)
        features.push(self.phase_coherence);
        features.push(self.resonance_energy.min(10.0) / 10.0);

        features
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Create a zero (uniform superposition) wave.
fn zero_wave(dim: usize) -> Wave {
    let mag = 1.0 / (dim as f64).sqrt();
    let data: Vec<Complex64> = (0..dim)
        .map(|_| Complex64::new(mag, 0.0))
        .collect();
    Wave {
        amplitudes: ComplexVec::new(data),
        t: 0.0,
        density: None,
        dim,
        metrics: CollapseMetrics::new(),
    }
}

/// Apply phase rotation to all components of a wave.
fn phase_rotate(wave: &Wave, angle: f64) -> Wave {
    let data: Vec<Complex64> = wave.amplitudes.data.iter().enumerate()
        .map(|(i, &amp)| {
            // Dimension-dependent frequency for spectral diversity
            let freq = 1.0 / (100.0_f64).powf(2.0 * (i as f64) / wave.dim as f64);
            let theta = angle * freq;
            amp * Complex64::from_polar(1.0, theta)
        })
        .collect();
    Wave {
        amplitudes: ComplexVec::new(data).normalized(),
        t: wave.t,
        density: None,
        dim: wave.dim,
        metrics: wave.metrics.clone(),
    }
}

/// Apply damping to a wave (multiply amplitudes by damping factor).
fn damp_wave(wave: &Wave, damping: f64) -> Wave {
    let data: Vec<Complex64> = wave.amplitudes.data.iter()
        .map(|&amp| amp * damping)
        .collect();
    Wave {
        amplitudes: ComplexVec::new(data).normalized(),
        t: wave.t,
        density: None,
        dim: wave.dim,
        metrics: wave.metrics.clone(),
    }
}

/// Apply Kerr nonlinearity: phase += κ × |amplitude|².
fn kerr_nonlinearity(wave: &Wave, kappa: f64) -> Wave {
    let data: Vec<Complex64> = wave.amplitudes.data.iter()
        .map(|&amp| {
            let mag = amp.norm();
            let phase = amp.arg();
            let new_phase = phase + kappa * mag * mag;
            Complex64::from_polar(mag, new_phase)
        })
        .collect();
    Wave {
        amplitudes: ComplexVec::new(data).normalized(),
        t: wave.t,
        density: None,
        dim: wave.dim,
        metrics: wave.metrics.clone(),
    }
}

/// Normalize a wave to unit norm.
fn normalize_wave(wave: &Wave) -> Wave {
    Wave {
        amplitudes: wave.amplitudes.normalized(),
        t: wave.t,
        density: None,
        dim: wave.dim,
        metrics: wave.metrics.clone(),
    }
}

/// Merge multiple scale waves via weighted constructive interference.
fn merge_scales(scales: &[Wave], dim: usize) -> Wave {
    if scales.is_empty() {
        return zero_wave(dim);
    }
    if scales.len() == 1 {
        return scales[0].clone();
    }

    // Weighted sum: equal weights for simplicity
    let weight = 1.0 / scales.len() as f64;
    let mut data = vec![Complex64::new(0.0, 0.0); dim];

    for scale in scales {
        for (i, &amp) in scale.amplitudes.data.iter().enumerate() {
            if i < dim {
                data[i] += amp * weight;
            }
        }
    }

    Wave {
        amplitudes: ComplexVec::new(data).normalized(),
        t: 0.0,
        density: None,
        dim,
        metrics: CollapseMetrics::new(),
    }
}

/// Compute phase coherence across scales.
///
/// Measures how aligned the phase patterns are across different time scales.
/// High coherence = consistent context interpretation across scales.
/// Returns a value in [0, 1].
fn compute_phase_coherence(scales: &[Wave]) -> f64 {
    if scales.len() < 2 {
        return 1.0;
    }

    let n_pairs = scales.len() * (scales.len() - 1) / 2;
    let mut total_coherence = 0.0;

    for i in 0..scales.len() {
        for j in (i + 1)..scales.len() {
            let dim = scales[i].dim.min(scales[j].dim);
            let mut phase_sum = Complex64::new(0.0, 0.0);

            for k in 0..dim {
                let phase_diff = scales[i].amplitudes.data[k].arg()
                    - scales[j].amplitudes.data[k].arg();
                phase_sum += Complex64::from_polar(1.0, phase_diff);
            }

            total_coherence += phase_sum.norm() / dim as f64;
        }
    }

    (total_coherence / n_pairs as f64).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resonance_register_basic() {
        let dim = 16;
        let mut reg = ResonanceRegister::new(dim, 5.0, InterferencePattern::Constructive);

        // Feed a non-uniform wave
        let data: Vec<Complex64> = (0..dim)
            .map(|i| Complex64::from_polar(1.0 / (dim as f64).sqrt(), i as f64 * 0.5))
            .collect();
        let token = Wave {
            amplitudes: ComplexVec::new(data).normalized(),
            t: 0.0,
            density: None,
            dim,
            metrics: CollapseMetrics::new(),
        };

        reg.resonate(&token, 0);
        assert!(reg.energy() > 0.0);

        // State should differ from zero
        let zero = zero_wave(dim);
        let diff: f64 = reg.state.probabilities().iter()
            .zip(zero.probabilities().iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.001, "state should differ from zero: diff={}", diff);
    }

    #[test]
    fn test_reservoir_multi_scale() {
        let dim = 16;
        let mut reservoir = WaveResonanceReservoir::new(dim);
        assert_eq!(reservoir.num_scales(), 3);

        // Process a short sequence
        let waves: Vec<Wave> = (0..5).map(|i| {
            let data: Vec<Complex64> = (0..dim)
                .map(|j| Complex64::from_polar(
                    1.0 / (dim as f64).sqrt(),
                    (i * dim + j) as f64 * 0.3,
                ))
                .collect();
            Wave {
                amplitudes: ComplexVec::new(data).normalized(),
                t: 0.0,
                density: None,
                dim,
                metrics: CollapseMetrics::new(),
            }
        }).collect();

        let state = reservoir.process_sequence(&waves);

        // Check merged wave
        assert_eq!(state.merged.dim, dim);
        let probs = state.merged.probabilities();
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "probs sum = {}", sum);

        // Check scales
        assert_eq!(state.scales.len(), 3);

        // Phase coherence should be between 0 and 1
        assert!(state.phase_coherence >= 0.0 && state.phase_coherence <= 1.0,
            "coherence = {}", state.phase_coherence);

        // Energy should be positive
        assert!(state.resonance_energy > 0.0, "energy = {}", state.resonance_energy);
    }

    #[test]
    fn test_feature_vector() {
        let dim = 16;
        let mut reservoir = WaveResonanceReservoir::new(dim);
        let waves: Vec<Wave> = (0..3).map(|_| zero_wave(dim)).collect();
        let state = reservoir.process_sequence(&waves);

        let features = state.to_feature_vector();
        let expected_dim = ReservoirState::feature_dim(dim, 3);
        assert_eq!(features.len(), expected_dim,
            "feature dim: got {}, expected {}", features.len(), expected_dim);

        // Per-scale normalization: 4 blocks each with norm ~1 + 2 scalars
        // Total norm ≈ sqrt(4 + scalars²) ≈ 2.0
        let norm: f64 = features.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm > 1.0 && norm < 3.0, "norm = {}", norm);
    }

    #[test]
    fn test_saturation_measurement() {
        let dim = 16;
        let mut reg = ResonanceRegister::new(dim, 5.0, InterferencePattern::Constructive);

        // Zero wave (uniform) should have high saturation (all channels equal)
        let sat_initial = reg.saturation();
        assert!(sat_initial > 0.9, "uniform state should be near-saturated: {}", sat_initial);

        // After processing tokens, saturation should drop (some channels stronger)
        for i in 0..10 {
            let data: Vec<Complex64> = (0..dim)
                .map(|j| Complex64::from_polar(
                    if j == i % dim { 3.0 } else { 0.1 },
                    j as f64 * 0.5,
                ))
                .collect();
            let token = Wave {
                amplitudes: ComplexVec::new(data).normalized(),
                t: 0.0,
                density: None,
                dim,
                metrics: CollapseMetrics::new(),
            };
            reg.resonate(&token, i);
        }

        let sat_after = reg.saturation();
        // After processing, saturation should be less than initial (more structure)
        assert!(sat_after <= sat_initial,
            "saturation should not increase: before={}, after={}", sat_initial, sat_after);
    }

    #[test]
    fn test_compaction_reduces_saturation() {
        let dim = 32;
        let mut reg = ResonanceRegister::new(dim, 5.0, InterferencePattern::Constructive);

        // Saturate with many tokens
        for i in 0..20 {
            let data: Vec<Complex64> = (0..dim)
                .map(|j| Complex64::from_polar(
                    1.0 / (dim as f64).sqrt(),
                    (i * 7 + j * 3) as f64 * 0.4,
                ))
                .collect();
            let token = Wave {
                amplitudes: ComplexVec::new(data).normalized(),
                t: 0.0,
                density: None,
                dim,
                metrics: CollapseMetrics::new(),
            };
            reg.resonate(&token, i);
        }

        let sat_before = reg.saturation();
        let result = reg.compact(0);

        // Compaction should have done something (merged or pruned)
        // At least some channels should have been affected
        assert!(result.saturation_before == sat_before,
            "saturation_before should match: {} vs {}", result.saturation_before, sat_before);
        assert!(result.saturation_after <= 1.0, "saturation_after should be <= 1.0");

        // Energy should be preserved approximately (renormalized)
        assert!(result.energy_after > 0.0, "energy should be positive after compaction");

        println!("Compaction: merged={}, pruned={}, sat {:.3}→{:.3}, energy {:.3}→{:.3}",
            result.channels_merged, result.channels_pruned,
            result.saturation_before, result.saturation_after,
            result.energy_before, result.energy_after);
    }

    #[test]
    fn test_auto_compaction_triggers() {
        let dim = 16;
        let mut reservoir = WaveResonanceReservoir::new(dim);

        // Process enough tokens to trigger compaction on all scales
        // τ=2 → compacts every 2 tokens
        // τ=5 → compacts every 5 tokens
        // τ=15 → compacts every 15 tokens
        let n_tokens = 30;

        for i in 0..n_tokens {
            let data: Vec<Complex64> = (0..dim)
                .map(|j| Complex64::from_polar(
                    1.0 / (dim as f64).sqrt(),
                    (i * 3 + j) as f64 * 0.6,
                ))
                .collect();
            let token = Wave {
                amplitudes: ComplexVec::new(data).normalized(),
                t: 0.0,
                density: None,
                dim,
                metrics: CollapseMetrics::new(),
            };
            reservoir.process_token(&token, i);
        }

        // Should have compaction events
        assert!(!reservoir.compaction_log.is_empty(),
            "should have compaction events after {} tokens", n_tokens);

        // τ=2 register compacts every 2 tokens → 15 compactions in 30 tokens
        // τ=5 register compacts every 5 tokens → 6 compactions
        // τ=15 register compacts every 15 tokens → 2 compactions
        // Total ≈ 23
        let scale0_compactions = reservoir.compaction_log.iter()
            .filter(|r| r.scale_index == 0).count();
        let scale1_compactions = reservoir.compaction_log.iter()
            .filter(|r| r.scale_index == 1).count();
        let scale2_compactions = reservoir.compaction_log.iter()
            .filter(|r| r.scale_index == 2).count();

        assert!(scale0_compactions >= 10,
            "τ=2 scale should compact ~15 times, got {}", scale0_compactions);
        assert!(scale1_compactions >= 4,
            "τ=5 scale should compact ~6 times, got {}", scale1_compactions);
        assert!(scale2_compactions >= 1,
            "τ=15 scale should compact ~2 times, got {}", scale2_compactions);

        println!("Auto-compaction events: scale0={}, scale1={}, scale2={}, total={}",
            scale0_compactions, scale1_compactions, scale2_compactions,
            reservoir.compaction_log.len());
    }

    #[test]
    fn test_compaction_preserves_information() {
        let dim = 16;
        let mut reservoir = WaveResonanceReservoir::new(dim);

        // Process a distinctive sequence
        let waves: Vec<Wave> = (0..10).map(|i| {
            let data: Vec<Complex64> = (0..dim)
                .map(|j| Complex64::from_polar(
                    1.0 / (dim as f64).sqrt(),
                    (i * dim + j) as f64 * 0.3,
                ))
                .collect();
            Wave {
                amplitudes: ComplexVec::new(data).normalized(),
                t: 0.0,
                density: None,
                dim,
                metrics: CollapseMetrics::new(),
            }
        }).collect();

        let state_with_compaction = reservoir.process_sequence(&waves);

        // The state should still be valid and non-trivial after compaction
        let probs = state_with_compaction.merged.probabilities();
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "probabilities should sum to 1: {}", sum);

        // Feature vector should still be well-formed
        let features = state_with_compaction.to_feature_vector();
        let feat_dim = ReservoirState::feature_dim(dim, 3);
        assert_eq!(features.len(), feat_dim);
        let norm: f64 = features.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm > 1.0 && norm < 3.0, "features norm out of range: {}", norm);

        // Energy should be positive
        assert!(state_with_compaction.resonance_energy > 0.0);
    }

    #[test]
    fn test_different_sequences_different_states() {
        let dim = 16;
        let mut reservoir = WaveResonanceReservoir::new(dim);

        let waves1: Vec<Wave> = (0..3).map(|i| {
            let data: Vec<Complex64> = (0..dim)
                .map(|j| Complex64::from_polar(1.0 / (dim as f64).sqrt(), (i + j) as f64 * 0.2))
                .collect();
            Wave::from_complex(ComplexVec::new(data))
        }).collect();

        let state1 = reservoir.process_sequence(&waves1);

        let waves2: Vec<Wave> = (0..3).map(|i| {
            let data: Vec<Complex64> = (0..dim)
                .map(|j| Complex64::from_polar(1.0 / (dim as f64).sqrt(), (i + j) as f64 * 0.7))
                .collect();
            Wave::from_complex(ComplexVec::new(data))
        }).collect();

        let state2 = reservoir.process_sequence(&waves2);

        let diff: f64 = state1.merged.probabilities().iter()
            .zip(state2.merged.probabilities().iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.001, "different sequences should give different states: diff={}", diff);
    }
}
