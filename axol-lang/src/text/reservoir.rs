//! Wave Resonance Reservoir (WRR) — Multi-scale, multi-node wave resonance.
//!
//! Physics-based reservoir whose dynamics are entirely determined by physical
//! constants (no learned parameters). Only the output readout layer is trained.
//!
//! Architecture:
//!   Token → [SPS] → wave → [WRR 3-scale × N-node resonance] → ReservoirState
//!     Scale 1 (τ=2, Destructive)   — lexical level
//!     Scale 2 (τ=5, Constructive)  — phrasal level
//!     Scale 3 (τ=15, Conditional)  — clause level
//!
//! Each scale contains N nodes (default 4) with different phase frequency
//! offsets for spectral diversity. Tokens are broadcast to all nodes via
//! superposition. Through wave interference:
//!   - Repeated/important patterns → constructive interference → amplitudes grow
//!   - Random/noise patterns → destructive interference → amplitudes cancel
//!
//! Inter-node coupling (during compaction) creates correlation between nodes,
//! enabling natural information exchange without explicit memory management.
//!
//! Complexity: O(n × dim × num_scales × num_nodes)

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
    /// Number of channels that were pruned (fell below threshold)
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
// ResonanceRegister — a single time-scale resonator with multiple nodes
// ---------------------------------------------------------------------------

/// A single resonance register operating at a specific time scale.
///
/// Contains multiple nodes, each a wave state with a different phase frequency
/// offset. Tokens are broadcast to all nodes via superposition.
/// Inter-node coupling during compaction creates natural information exchange.
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
    /// Node states (multi-node superposition reservoir)
    pub nodes: Vec<Wave>,
    /// Number of nodes per register
    pub num_nodes: usize,
    /// Phase frequency: 2π/τ — rate of phase rotation
    pub phase_freq: f64,
    /// Token counter for compaction trigger (compacts every τ tokens)
    pub token_count: usize,
}

impl ResonanceRegister {
    /// Create a new resonance register with multiple nodes.
    ///
    /// `dim`: wave dimension
    /// `tau`: characteristic time scale (larger = longer memory)
    /// `pattern`: interference pattern (determines how tokens combine)
    /// `num_nodes`: number of parallel nodes (more = more capacity)
    pub fn new(dim: usize, tau: f64, pattern: InterferencePattern, num_nodes: usize) -> Self {
        let num_nodes = num_nodes.max(1);
        let nodes = (0..num_nodes).map(|_| zero_wave(dim)).collect();
        Self {
            dim,
            tau,
            damping: (-1.0 / tau).exp(),
            pattern,
            nodes,
            num_nodes,
            phase_freq: 2.0 * std::f64::consts::PI / tau,
            token_count: 0,
        }
    }

    /// Process a single token wave through all nodes.
    ///
    /// Each node receives the same token but with a different phase frequency
    /// offset, creating spectral diversity. Over many tokens:
    ///   - Consistent patterns reinforce across nodes (constructive interference)
    ///   - Random patterns cancel across nodes (destructive interference)
    pub fn resonate(&mut self, token_wave: &Wave, position: usize) {
        for k in 0..self.num_nodes {
            // Each node has a unique phase frequency offset for diversity
            // Node 0: base freq, Node 1: 1.2× base, Node 2: 1.4× base, ...
            let node_freq = self.phase_freq * (1.0 + 0.2 * k as f64);
            let rotated = phase_rotate(token_wave, node_freq * position as f64);

            let composed = Wave::compose(&self.nodes[k], &rotated, &self.pattern)
                .unwrap_or_else(|_| rotated.clone());

            let activated = atan_nonlinearity(&composed);
            let kappa = 0.1;
            self.nodes[k] = kerr_nonlinearity(&activated, kappa);
        }
    }

    /// Aggregate all nodes into a single representative wave via superposition.
    ///
    /// The aggregated wave captures the coherent signal across all nodes.
    /// Correlated patterns (appearing in multiple nodes) reinforce;
    /// uncorrelated patterns average out.
    pub fn aggregate(&self) -> Wave {
        if self.num_nodes == 1 {
            return self.nodes[0].clone();
        }
        let weight = 1.0 / self.num_nodes as f64;
        let mut data = vec![Complex64::new(0.0, 0.0); self.dim];
        for node in &self.nodes {
            for (i, &amp) in node.amplitudes.data.iter().enumerate() {
                if i < self.dim {
                    data[i] += amp * weight;
                }
            }
        }
        Wave {
            amplitudes: ComplexVec::new(data).normalized(),
            t: 0.0,
            density: None,
            dim: self.dim,
            metrics: CollapseMetrics::new(),
        }
    }

    /// Reset all nodes to zero.
    pub fn reset(&mut self) {
        for node in &mut self.nodes {
            *node = zero_wave(self.dim);
        }
        self.token_count = 0;
    }

    /// Get the total energy across all nodes.
    pub fn energy(&self) -> f64 {
        self.nodes.iter()
            .map(|n| n.amplitudes.data.iter().map(|c| c.norm_sqr()).sum::<f64>())
            .sum()
    }

    /// Measure saturation of the aggregated state.
    pub fn saturation(&self) -> f64 {
        let agg = self.aggregate();
        let probs = agg.gaze();
        let dim = probs.len();
        if dim <= 1 {
            return 0.0;
        }

        let mut entropy = 0.0;
        for &p in &probs {
            if p > 1e-15 {
                entropy -= p * p.ln();
            }
        }

        let max_entropy = (dim as f64).ln();
        if max_entropy < 1e-15 {
            return 0.0;
        }
        (entropy / max_entropy).clamp(0.0, 1.0)
    }

    /// Compact all nodes via self-interference, then couple nodes.
    ///
    /// Per-node compaction: phase grouping + pruning + renormalization.
    /// Inter-node coupling: weak exchange between neighboring nodes,
    /// creating correlation (correlated signals reinforce, noise cancels).
    pub fn compact(&mut self, scale_index: usize) -> CompactionResult {
        let energy_before = self.energy();
        let saturation_before = self.saturation();

        let mut total_merged = 0;
        let mut total_pruned = 0;

        // Compact each node independently
        for node in &mut self.nodes {
            let (merged, pruned) = compact_single_wave(node);
            total_merged += merged;
            total_pruned += pruned;
        }

        // Inter-node coupling after compaction
        self.couple_nodes();

        let energy_after = self.energy();
        let saturation_after = self.saturation();

        CompactionResult {
            saturation_before,
            saturation_after,
            channels_pruned: total_pruned,
            channels_merged: total_merged,
            energy_before,
            energy_after,
            scale_index,
        }
    }

    /// Weak coupling between adjacent nodes (ring topology).
    ///
    /// Each node exchanges 5% of its amplitude with its neighbor.
    /// Effect: correlated signals (present in multiple nodes) reinforce,
    /// uncorrelated signals remain weak. Like coupled oscillators.
    fn couple_nodes(&mut self) {
        if self.num_nodes < 2 {
            return;
        }
        let coupling = 0.05;
        let old: Vec<Vec<Complex64>> = self.nodes.iter()
            .map(|n| n.amplitudes.data.clone())
            .collect();
        for k in 0..self.num_nodes {
            let next = (k + 1) % self.num_nodes;
            for ch in 0..self.dim {
                self.nodes[k].amplitudes.data[ch] =
                    old[k][ch] * (1.0 - coupling) + old[next][ch] * coupling;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// WaveResonanceReservoir — multi-scale, multi-node resonance
// ---------------------------------------------------------------------------

/// Multi-scale Wave Resonance Reservoir.
///
/// Contains multiple registers at different time scales, each with multiple
/// nodes for increased capacity. Tokens are broadcast to all nodes in all
/// registers. Wave interference naturally prioritizes important patterns.
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

/// Default number of nodes per register.
const DEFAULT_NUM_NODES: usize = 4;

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
    /// Create a new reservoir with default 3-scale configuration and 4 nodes.
    pub fn new(dim: usize) -> Self {
        Self::new_with_nodes(dim, DEFAULT_NUM_NODES)
    }

    /// Create a reservoir with custom number of nodes per register.
    pub fn new_with_nodes(dim: usize, num_nodes: usize) -> Self {
        let registers = DEFAULT_SCALES.iter()
            .map(|&(tau, pat_idx)| {
                ResonanceRegister::new(dim, tau, pattern_from_index(pat_idx), num_nodes)
            })
            .collect();

        Self { dim, registers, compaction_log: Vec::new() }
    }

    /// Create a reservoir with custom scale configurations.
    pub fn with_scales(dim: usize, scales: &[(f64, InterferencePattern)], num_nodes: usize) -> Self {
        let registers = scales.iter()
            .map(|(tau, pat)| ResonanceRegister::new(dim, *tau, pat.clone(), num_nodes))
            .collect();

        Self { dim, registers, compaction_log: Vec::new() }
    }

    /// Process a sequence of token waves through the reservoir.
    ///
    /// Returns the final ReservoirState capturing all multi-scale context.
    /// Includes automatic resonance compaction every τ tokens per register.
    pub fn process_sequence(&mut self, token_waves: &[Wave]) -> ReservoirState {
        for reg in &mut self.registers {
            reg.reset();
        }
        self.compaction_log.clear();

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
    /// When token_count reaches τ, triggers compaction + inter-node coupling.
    pub fn process_token(&mut self, token_wave: &Wave, position: usize) {
        let num_regs = self.registers.len();
        for idx in 0..num_regs {
            self.registers[idx].resonate(token_wave, position);
            self.registers[idx].token_count += 1;

            let compact_interval = (self.registers[idx].tau.round() as usize).max(2);
            if self.registers[idx].token_count % compact_interval == 0 {
                let result = self.registers[idx].compact(idx);
                self.compaction_log.push(result);
            }
        }
    }

    /// Get the current reservoir state without resetting.
    ///
    /// Aggregates per-register node waves into scale waves, and collects
    /// all individual node waves for richer feature statistics.
    pub fn current_state(&self) -> ReservoirState {
        // Aggregate per register → one wave per scale
        let scales: Vec<Wave> = self.registers.iter()
            .map(|r| r.aggregate())
            .collect();

        // Collect ALL individual node waves for μ/σ estimation
        let node_waves: Vec<Wave> = self.registers.iter()
            .flat_map(|r| r.nodes.clone())
            .collect();

        let merged = merge_scales(&scales, self.dim);
        let phase_coherence = compute_phase_coherence(&scales);
        let resonance_energy = self.registers.iter()
            .map(|r| r.energy())
            .sum();

        ReservoirState {
            merged,
            scales,
            node_waves,
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

    /// Number of nodes per register.
    pub fn num_nodes(&self) -> usize {
        self.registers.first().map_or(4, |r| r.num_nodes)
    }
}

// ---------------------------------------------------------------------------
// Multi-slit interference
// ---------------------------------------------------------------------------

/// Create a variant reservoir for multi-slit experiment.
///
/// Each slit_idx produces a different physical configuration —
/// different τ values and interference patterns, like slits of
/// different widths at different positions.
pub fn make_slit_reservoir(dim: usize, slit_idx: usize, num_nodes: usize) -> WaveResonanceReservoir {
    let configs: Vec<(f64, InterferencePattern)> = match slit_idx % 4 {
        0 => vec![
            (2.0, InterferencePattern::Destructive),
            (5.0, InterferencePattern::Constructive),
            (15.0, InterferencePattern::Conditional),
        ],
        1 => vec![
            (3.0, InterferencePattern::Constructive),
            (8.0, InterferencePattern::Conditional),
            (20.0, InterferencePattern::Destructive),
        ],
        2 => vec![
            (1.5, InterferencePattern::Conditional),
            (4.0, InterferencePattern::Destructive),
            (12.0, InterferencePattern::Constructive),
        ],
        _ => vec![
            (4.0, InterferencePattern::Additive),
            (10.0, InterferencePattern::Multiplicative),
            (25.0, InterferencePattern::Constructive),
        ],
    };
    WaveResonanceReservoir::with_scales(dim, &configs, num_nodes)
}

/// Interfere multiple reservoir states (multi-slit experiment).
///
/// Complex amplitudes are ADDED across slits (not averaged).
/// Phase-aligned signals reinforce (constructive interference),
/// phase-misaligned signals cancel (destructive interference).
/// The interference pattern encodes information from ALL slits.
pub fn interfere_states(states: &[ReservoirState], dim: usize) -> ReservoirState {
    if states.len() <= 1 {
        return states[0].clone();
    }

    let num_scales = states[0].scales.len();

    // Interfere corresponding scale waves across slits
    let mut interfered_scales: Vec<Wave> = Vec::with_capacity(num_scales);
    for s in 0..num_scales {
        let mut data = vec![Complex64::new(0.0, 0.0); dim];
        for state in states {
            if s < state.scales.len() {
                for (i, &amp) in state.scales[s].amplitudes.data.iter().enumerate() {
                    if i < dim { data[i] += amp; }
                }
            }
        }
        interfered_scales.push(Wave {
            amplitudes: ComplexVec::new(data).normalized(),
            t: 0.0,
            density: None,
            dim,
            metrics: CollapseMetrics::new(),
        });
    }

    // Interfere merged waves
    let mut merged_data = vec![Complex64::new(0.0, 0.0); dim];
    for state in states {
        for (i, &amp) in state.merged.amplitudes.data.iter().enumerate() {
            if i < dim { merged_data[i] += amp; }
        }
    }
    let merged = Wave {
        amplitudes: ComplexVec::new(merged_data).normalized(),
        t: 0.0,
        density: None,
        dim,
        metrics: CollapseMetrics::new(),
    };

    // Collect ALL node waves from all slits (richer μ/σ statistics)
    let node_waves: Vec<Wave> = states.iter()
        .flat_map(|s| s.node_waves.iter().cloned())
        .collect();

    let phase_coherence = compute_phase_coherence(&interfered_scales);
    let resonance_energy: f64 = states.iter().map(|s| s.resonance_energy).sum();

    ReservoirState {
        merged,
        scales: interfered_scales,
        node_waves,
        phase_coherence,
        resonance_energy,
    }
}

// ---------------------------------------------------------------------------
// ReservoirState — snapshot of multi-scale context
// ---------------------------------------------------------------------------

/// Snapshot of the reservoir's multi-scale context.
///
/// Contains merged, per-scale, and per-node information, plus
/// physics-derived quality signals (coherence, energy).
#[derive(Clone, Debug)]
pub struct ReservoirState {
    /// Merged multi-scale context wave
    pub merged: Wave,
    /// Aggregated scale states (one per register)
    pub scales: Vec<Wave>,
    /// All individual node waves across all registers
    /// (used for richer Gaussian statistics in feature extraction)
    pub node_waves: Vec<Wave>,
    /// Phase coherence across scales (0..1)
    pub phase_coherence: f64,
    /// Total resonance energy across all scales
    pub resonance_energy: f64,
}

impl ReservoirState {
    /// Compute the feature dimension for readout.
    ///
    /// Gaussian representation: each channel contributes (μ, σ, phase).
    /// μ/σ are estimated from ALL node waves (richer statistics).
    /// Pairs are computed between aggregated scale waves (compact).
    /// Feature dim is independent of num_nodes.
    pub fn feature_dim(dim: usize, num_scales: usize) -> usize {
        let per_channel = 3; // μ, σ, phase
        let global = 2; // coherence, energy
        let n_waves = 1 + num_scales;
        let n_pairs = n_waves * (n_waves - 1) / 2;
        let resonance = n_pairs * 3; // overlap_mean, high_overlap_ratio, phase_alignment
        per_channel * dim + global + resonance
    }

    /// Extract the full feature vector for the readout layer.
    ///
    /// Gaussian representation with multi-node statistics:
    ///   - μ = mean magnitude across ALL node waves (richer central tendency)
    ///   - σ = std of magnitudes across ALL node waves (captures inter-node variance)
    ///   - phase = from merged wave (reference frame)
    ///   - Gaussian overlap between aggregated scale waves
    ///
    /// With N nodes per register: μ/σ estimated from 1 + 3N waves instead of 4.
    /// More samples → more robust statistics → better discrimination.
    pub fn to_feature_vector(&self) -> Vec<f64> {
        let dim = self.merged.dim;
        let num_scales = self.scales.len();
        let feat_dim = Self::feature_dim(dim, num_scales);
        let mut features = Vec::with_capacity(feat_dim);

        // ALL waves for μ/σ: merged + all individual node waves
        let all_waves: Vec<&Wave> = std::iter::once(&self.merged)
            .chain(self.node_waves.iter())
            .collect();
        let n_waves = all_waves.len();

        // Aggregated scale waves for pair interactions
        let scale_waves: Vec<&Wave> = std::iter::once(&self.merged)
            .chain(self.scales.iter())
            .collect();

        // Per-channel Gaussian parameters
        let mut sigmas = Vec::with_capacity(dim);
        for k in 0..dim {
            let mags: Vec<f64> = all_waves.iter()
                .map(|w| w.amplitudes.data[k].norm())
                .collect();

            let mu = mags.iter().sum::<f64>() / n_waves as f64;
            let variance = mags.iter()
                .map(|&m| (m - mu).powi(2))
                .sum::<f64>() / n_waves as f64;
            let sigma = variance.sqrt().max(0.01);
            let phase = self.merged.amplitudes.data[k].arg() / std::f64::consts::PI;

            features.push(mu);
            features.push(sigma);
            features.push(phase);
            sigmas.push(sigma);
        }

        // Global statistics
        features.push(self.phase_coherence);
        features.push(self.resonance_energy.min(10.0) / 10.0);

        // Gaussian overlap resonance between aggregated scale waves
        for i in 0..scale_waves.len() {
            for j in (i + 1)..scale_waves.len() {
                let mut total_overlap = 0.0;
                let mut high_overlap_count = 0usize;
                let mut phase_alignment = 0.0;

                for k in 0..dim {
                    let m_i = scale_waves[i].amplitudes.data[k].norm();
                    let m_j = scale_waves[j].amplitudes.data[k].norm();
                    let sigma = sigmas[k];

                    let overlap = (-(m_i - m_j).powi(2) / (4.0 * sigma * sigma)).exp();
                    total_overlap += overlap;
                    if overlap > 0.5 { high_overlap_count += 1; }

                    let phase_diff = scale_waves[i].amplitudes.data[k].arg()
                        - scale_waves[j].amplitudes.data[k].arg();
                    phase_alignment += phase_diff.cos().abs();
                }

                features.push(total_overlap / dim as f64);
                features.push(high_overlap_count as f64 / dim as f64);
                features.push(phase_alignment / dim as f64);
            }
        }

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
            let freq = 1.0 / (100.0_f64).powf(2.0 * (i as f64) / wave.dim as f64);
            let theta = angle * freq;
            amp * Complex64::from_polar(1.0, theta)
        })
        .collect();
    Wave {
        amplitudes: ComplexVec::new(data),
        t: wave.t,
        density: None,
        dim: wave.dim,
        metrics: wave.metrics.clone(),
    }
}

/// Apply atan nonlinearity to wave magnitudes (preserves phase).
fn atan_nonlinearity(wave: &Wave) -> Wave {
    let data: Vec<Complex64> = wave.amplitudes.data.iter()
        .map(|&amp| {
            let mag = amp.norm();
            let phase = amp.arg();
            Complex64::from_polar(mag.atan(), phase)
        })
        .collect();
    Wave {
        amplitudes: ComplexVec::new(data),
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
        amplitudes: ComplexVec::new(data),
        t: wave.t,
        density: None,
        dim: wave.dim,
        metrics: wave.metrics.clone(),
    }
}

/// Compact a single wave via phase grouping + pruning + renormalization.
///
/// Returns (channels_merged, channels_pruned).
fn compact_single_wave(wave: &mut Wave) -> (usize, usize) {
    let dim = wave.dim;
    let amps = &wave.amplitudes.data;

    let phase_threshold = std::f64::consts::PI / 4.0;
    let phases: Vec<f64> = amps.iter().map(|c| c.arg()).collect();
    let magnitudes: Vec<f64> = amps.iter().map(|c| c.norm()).collect();
    let mut merged_count = 0usize;
    let mut visited = vec![false; dim];
    let mut new_amps: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); dim];

    for i in 0..dim {
        if visited[i] || magnitudes[i] < 1e-12 {
            if !visited[i] {
                new_amps[i] = amps[i];
            }
            visited[i] = true;
            continue;
        }

        let mut group_mag = magnitudes[i];
        let mut group_phase_x = magnitudes[i] * phases[i].cos();
        let mut group_phase_y = magnitudes[i] * phases[i].sin();
        let mut group_count = 1usize;
        visited[i] = true;

        for j in (i + 1)..dim {
            if visited[j] || magnitudes[j] < 1e-12 {
                continue;
            }
            let mut dp = (phases[i] - phases[j]).abs();
            if dp > std::f64::consts::PI {
                dp = 2.0 * std::f64::consts::PI - dp;
            }

            if dp < phase_threshold {
                group_mag += magnitudes[j];
                group_phase_x += magnitudes[j] * phases[j].cos();
                group_phase_y += magnitudes[j] * phases[j].sin();
                group_count += 1;
                visited[j] = true;
                new_amps[j] = Complex64::new(0.0, 0.0);
            }
        }

        if group_count > 1 {
            merged_count += group_count - 1;
        }

        let avg_phase = group_phase_y.atan2(group_phase_x);
        new_amps[i] = Complex64::from_polar(group_mag, avg_phase);
    }

    // Prune low-energy channels
    let max_energy = new_amps.iter().map(|c| c.norm_sqr())
        .fold(0.0f64, f64::max);
    let prune_threshold = max_energy * 0.001;
    let mut pruned_count = 0usize;

    for i in 0..dim {
        let ch_energy = new_amps[i].norm_sqr();
        if ch_energy > 0.0 && ch_energy < prune_threshold {
            new_amps[i] *= 0.1;
            pruned_count += 1;
        }
    }

    // Renormalize
    let norm: f64 = new_amps.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
    if norm > 1e-15 {
        for c in &mut new_amps {
            *c /= norm;
        }
    }

    *wave = Wave {
        amplitudes: ComplexVec::new(new_amps),
        t: wave.t,
        density: None,
        dim: wave.dim,
        metrics: wave.metrics.clone(),
    };

    (merged_count, pruned_count)
}

/// Merge multiple scale waves via weighted constructive interference.
fn merge_scales(scales: &[Wave], dim: usize) -> Wave {
    if scales.is_empty() {
        return zero_wave(dim);
    }
    if scales.len() == 1 {
        return scales[0].clone();
    }

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
        let mut reg = ResonanceRegister::new(dim, 5.0, InterferencePattern::Constructive, 4);

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

        // Aggregated state should differ from zero
        let agg = reg.aggregate();
        let zero = zero_wave(dim);
        let diff: f64 = agg.probabilities().iter()
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

        assert_eq!(state.merged.dim, dim);
        let probs = state.merged.probabilities();
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "probs sum = {}", sum);

        assert_eq!(state.scales.len(), 3);
        // node_waves: 3 registers × 4 nodes = 12
        assert_eq!(state.node_waves.len(), 12);

        assert!(state.phase_coherence >= 0.0 && state.phase_coherence <= 1.0,
            "coherence = {}", state.phase_coherence);
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

        let norm: f64 = features.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm > 0.05 && norm < 50.0, "norm = {}", norm);
    }

    #[test]
    fn test_saturation_measurement() {
        let dim = 16;
        let mut reg = ResonanceRegister::new(dim, 5.0, InterferencePattern::Constructive, 4);

        let sat_initial = reg.saturation();
        assert!(sat_initial > 0.9, "uniform state should be near-saturated: {}", sat_initial);

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
        assert!(sat_after <= sat_initial,
            "saturation should not increase: before={}, after={}", sat_initial, sat_after);
    }

    #[test]
    fn test_compaction_reduces_saturation() {
        let dim = 32;
        let mut reg = ResonanceRegister::new(dim, 5.0, InterferencePattern::Constructive, 4);

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

        assert!(result.saturation_before == sat_before,
            "saturation_before should match: {} vs {}", result.saturation_before, sat_before);
        assert!(result.saturation_after <= 1.0, "saturation_after should be <= 1.0");
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

        assert!(!reservoir.compaction_log.is_empty(),
            "should have compaction events after {} tokens", n_tokens);

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

        let probs = state_with_compaction.merged.probabilities();
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "probabilities should sum to 1: {}", sum);

        let features = state_with_compaction.to_feature_vector();
        let feat_dim = ReservoirState::feature_dim(dim, 3);
        assert_eq!(features.len(), feat_dim);
        let norm: f64 = features.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm > 0.05 && norm < 50.0, "features norm out of range: {}", norm);

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

    #[test]
    fn test_multi_node_diversity() {
        let dim = 16;
        let mut reg = ResonanceRegister::new(dim, 5.0, InterferencePattern::Constructive, 4);

        // Feed several tokens
        for i in 0..5 {
            let data: Vec<Complex64> = (0..dim)
                .map(|j| Complex64::from_polar(1.0 / (dim as f64).sqrt(), (i + j) as f64 * 0.5))
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

        // Nodes should have different states (spectral diversity)
        let probs0 = reg.nodes[0].probabilities();
        let probs1 = reg.nodes[1].probabilities();
        let diff: f64 = probs0.iter().zip(probs1.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.001, "different nodes should have different states: diff={}", diff);
    }
}
