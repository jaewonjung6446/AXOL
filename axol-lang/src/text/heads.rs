//! Task-specific heads for the Wave Text Engine.
//!
//! Autocomplete/generation use quantum measurement (Born rule):
//!   |⟨ψ_state|ψ_token⟩|² — no learned parameters, no time axis.
//!
//! Classification uses ReadoutLayer with feature normalization:
//!   z-scored features → ReadoutLayer → softmax (timeless task).
//!
//! Available heads:
//!   - AutocompleteHead: quantum measurement prediction
//!   - SentenceGenerationHead: multi-token generation with quality gate
//!   - ClassificationHead: text classification (with feature normalization)
//!   - AnomalyDetectionHead: anomaly scoring via fingerprint comparison

use num_complex::Complex64;

use super::readout::ReadoutLayer;
use super::reservoir::{WaveResonanceReservoir, ReservoirState};
use super::fingerprint::StyleFingerprint;
use super::generator::{
    TokenPrediction, GenerationResult, softmax, compute_omega, compute_phi,
};
use super::sps::SemanticPhaseSpace;
use super::tokenizer::{Vocabulary, EOS_ID, UNK_ID};

// ---------------------------------------------------------------------------
// AutocompleteHead — quantum measurement prediction
// ---------------------------------------------------------------------------

/// Predicts the next token via quantum measurement (Born rule).
///
/// Pipeline: ReservoirState |ψ⟩ → ⟨ψ|t⟩ for each token t → |⟨ψ|t⟩|² → Φ/Ω
/// No learned parameters — probabilities come directly from wave physics.
#[derive(Clone, Debug)]
pub struct AutocompleteHead {
    pub readout: ReadoutLayer,
    pub vocab_size: usize,
}

impl AutocompleteHead {
    pub fn new(feature_dim: usize, vocab_size: usize, hidden_dim: usize) -> Self {
        Self {
            readout: ReadoutLayer::new(feature_dim, vocab_size, hidden_dim),
            vocab_size,
        }
    }

    /// Compute token probabilities via quantum measurement (Born rule).
    ///
    /// |⟨ψ_merged|ψ_token⟩|² for each token in the vocabulary.
    /// This is the physics-native readout — no learned parameters.
    pub fn quantum_probs(&self, state: &ReservoirState, sps: &SemanticPhaseSpace) -> Vec<f64> {
        let merged = &state.merged;
        let dim = merged.dim;
        let mut raw = Vec::with_capacity(self.vocab_size);

        for tid in 0..self.vocab_size {
            let tw = sps.token_to_wave(tid);
            let mut inner = Complex64::new(0.0, 0.0);
            let d = dim.min(tw.dim);
            for k in 0..d {
                inner += merged.amplitudes.data[k].conj() * tw.amplitudes.data[k];
            }
            raw.push(inner.norm_sqr());
        }

        // Normalize (Born rule)
        let sum: f64 = raw.iter().sum();
        if sum > 1e-10 {
            for p in raw.iter_mut() { *p /= sum; }
        } else {
            let u = 1.0 / self.vocab_size as f64;
            raw.iter_mut().for_each(|p| *p = u);
        }
        raw
    }

    /// Predict the next token from reservoir state via quantum measurement.
    pub fn predict(&self, state: &ReservoirState, sps: &SemanticPhaseSpace, vocab: &Vocabulary) -> TokenPrediction {
        let probs = self.quantum_probs(state, sps);

        let (token_id, &max_prob) = probs.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((UNK_ID, &0.0));

        let omega = compute_omega(&probs);
        let phi = compute_phi(&probs);

        let mut indexed: Vec<(usize, f64)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_k: Vec<(String, f64)> = indexed.iter().take(5)
            .map(|&(id, p)| (vocab.decode_id(id).to_string(), p))
            .collect();

        TokenPrediction {
            token_id,
            token: vocab.decode_id(token_id).to_string(),
            confidence: max_prob,
            omega,
            phi,
            probabilities: probs,
            top_k,
        }
    }

    /// Predict top-n tokens via quantum measurement.
    pub fn predict_top_n(
        &self,
        state: &ReservoirState,
        sps: &SemanticPhaseSpace,
        vocab: &Vocabulary,
        n: usize,
    ) -> Vec<TokenPrediction> {
        let probs = self.quantum_probs(state, sps);
        let omega = compute_omega(&probs);
        let phi = compute_phi(&probs);

        let mut indexed: Vec<(usize, f64)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_k_all: Vec<(String, f64)> = indexed.iter().take(5)
            .map(|&(id, p)| (vocab.decode_id(id).to_string(), p))
            .collect();

        indexed.iter().take(n)
            .map(|&(id, prob)| {
                TokenPrediction {
                    token_id: id,
                    token: vocab.decode_id(id).to_string(),
                    confidence: prob,
                    omega,
                    phi,
                    probabilities: probs.clone(),
                    top_k: top_k_all.clone(),
                }
            })
            .collect()
    }

    /// Train the readout (kept for backward compatibility / classification sharing).
    pub fn train(&mut self, features_batch: &[Vec<f64>], target_ids: &[usize]) {
        let n = features_batch.len();
        if n == 0 || target_ids.len() != n {
            return;
        }

        let targets: Vec<Vec<f64>> = target_ids.iter()
            .map(|&tid| {
                let mut target = vec![0.001 / self.vocab_size as f64; self.vocab_size];
                if tid < self.vocab_size {
                    target[tid] = 0.999;
                }
                let sum: f64 = target.iter().sum();
                for v in target.iter_mut() {
                    *v /= sum;
                }
                target
            })
            .collect();

        self.readout.train(features_batch, &targets);
    }

    pub fn is_trained(&self) -> bool {
        // Quantum measurement doesn't need training — always ready
        true
    }
}

// ---------------------------------------------------------------------------
// SentenceGenerationHead — autoregressive generation with quality gate
// ---------------------------------------------------------------------------

/// Multi-token generation using autoregressive loop with Φ quality gate.
///
/// Generates tokens one at a time, feeding each back through the reservoir.
/// Uses quantum measurement for prediction.
#[derive(Clone, Debug)]
pub struct SentenceGenerationHead {
    pub autocomplete: AutocompleteHead,
    /// Minimum Φ (collapse separation) for continued generation (quality gate)
    pub min_phi: f64,
    /// Maximum tokens to generate
    pub max_tokens: usize,
}

impl SentenceGenerationHead {
    pub fn new(feature_dim: usize, vocab_size: usize, hidden_dim: usize) -> Self {
        Self {
            autocomplete: AutocompleteHead::new(feature_dim, vocab_size, hidden_dim),
            min_phi: 0.0,
            max_tokens: 30,
        }
    }

    /// Generate a sequence of tokens from a prompt.
    pub fn generate(
        &self,
        prompt_ids: &[usize],
        sps: &SemanticPhaseSpace,
        reservoir: &mut WaveResonanceReservoir,
        vocab: &Vocabulary,
        max_tokens: Option<usize>,
    ) -> GenerationResult {
        let max = max_tokens.unwrap_or(self.max_tokens);
        let mut all_ids = prompt_ids.to_vec();
        let mut generated_tokens = Vec::new();
        let mut predictions = Vec::new();

        // Process prompt through reservoir
        let prompt_waves = sps.tokens_to_waves(prompt_ids);
        reservoir.reset();
        for (pos, wave) in prompt_waves.iter().enumerate() {
            reservoir.process_token(wave, pos);
        }

        for step in 0..max {
            let state = reservoir.current_state();
            let pred = self.autocomplete.predict(&state, sps, vocab);

            if pred.token_id == EOS_ID {
                break;
            }
            if pred.phi < self.min_phi && !predictions.is_empty() {
                break;
            }

            let token_wave = sps.token_to_wave(pred.token_id);
            let pos = prompt_ids.len() + step;
            reservoir.process_token(&token_wave, pos);

            all_ids.push(pred.token_id);
            generated_tokens.push(pred.token.clone());
            predictions.push(pred);
        }

        let prompt_text = prompt_ids.iter()
            .filter(|&&id| id != 0 && id != 2 && id != 3)
            .map(|&id| vocab.decode_id(id).to_string())
            .collect::<Vec<_>>()
            .join(" ");

        let full_text = if generated_tokens.is_empty() {
            prompt_text.clone()
        } else {
            format!("{} {}", prompt_text, generated_tokens.join(" "))
        };

        GenerationResult {
            prompt: prompt_text,
            generated: generated_tokens,
            full_text,
            predictions,
        }
    }

    pub fn train(&mut self, features_batch: &[Vec<f64>], target_ids: &[usize]) {
        self.autocomplete.train(features_batch, target_ids);
    }

    pub fn is_trained(&self) -> bool {
        self.autocomplete.is_trained()
    }
}

// ---------------------------------------------------------------------------
// ClassificationHead — text classification with feature normalization
// ---------------------------------------------------------------------------

/// Result of a classification prediction.
#[derive(Clone, Debug)]
pub struct ClassificationResult {
    pub class_id: usize,
    pub class_label: String,
    pub confidence: f64,
    pub probabilities: Vec<f64>,
    pub class_probs: Vec<(String, f64)>,
    pub omega: f64,
    pub phi: f64,
}

/// Text classification head with feature normalization (z-score).
///
/// Maps reservoir features to class probabilities via ReadoutLayer + softmax.
/// Features are z-score normalized using training batch statistics.
#[derive(Clone, Debug)]
pub struct ClassificationHead {
    pub readout: ReadoutLayer,
    pub n_classes: usize,
    pub class_labels: Vec<String>,
    /// Feature means for z-score normalization
    pub feature_means: Vec<f64>,
    /// Feature stds for z-score normalization
    pub feature_stds: Vec<f64>,
}

impl ClassificationHead {
    pub fn new(feature_dim: usize, class_labels: Vec<String>, hidden_dim: usize) -> Self {
        let n_classes = class_labels.len();
        Self {
            readout: ReadoutLayer::new(feature_dim, n_classes, hidden_dim),
            n_classes,
            class_labels,
            feature_means: Vec::new(),
            feature_stds: Vec::new(),
        }
    }

    /// Normalize features using stored training statistics.
    fn normalize(&self, features: &[f64]) -> Vec<f64> {
        if self.feature_means.is_empty() {
            return features.to_vec();
        }
        features.iter().enumerate()
            .map(|(j, &v)| {
                let m = self.feature_means.get(j).copied().unwrap_or(0.0);
                let s = self.feature_stds.get(j).copied().unwrap_or(1.0);
                (v - m) / s
            })
            .collect()
    }

    /// Classify a text from its reservoir state.
    pub fn predict(&self, state: &ReservoirState) -> ClassificationResult {
        let features = state.to_feature_vector();
        let normalized = self.normalize(&features);
        let scores = self.readout.forward(&normalized);
        let probs = softmax(&scores);

        let (class_id, &max_prob) = probs.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));

        let omega = compute_omega(&probs);
        let phi = compute_phi(&probs);

        let class_probs: Vec<(String, f64)> = self.class_labels.iter()
            .zip(probs.iter())
            .map(|(label, &p)| (label.clone(), p))
            .collect();

        ClassificationResult {
            class_id,
            class_label: self.class_labels.get(class_id)
                .cloned()
                .unwrap_or_else(|| format!("class_{}", class_id)),
            confidence: max_prob,
            probabilities: probs,
            class_probs,
            omega,
            phi,
        }
    }

    /// Train with z-score feature normalization.
    pub fn train(&mut self, features_batch: &[Vec<f64>], class_ids: &[usize]) {
        let n = features_batch.len();
        if n == 0 || class_ids.len() != n {
            return;
        }

        // Compute normalization statistics
        let fd = features_batch[0].len();
        let nf = n as f64;
        let mut means = vec![0.0; fd];
        for feat in features_batch {
            for (j, &v) in feat.iter().enumerate() {
                if j < fd { means[j] += v; }
            }
        }
        for v in means.iter_mut() { *v /= nf; }

        let mut stds = vec![0.0; fd];
        for feat in features_batch {
            for (j, &v) in feat.iter().enumerate() {
                if j < fd { stds[j] += (v - means[j]).powi(2); }
            }
        }
        for v in stds.iter_mut() { *v = (*v / nf).sqrt().max(1e-6); }

        self.feature_means = means;
        self.feature_stds = stds;

        // Normalize features
        let normalized: Vec<Vec<f64>> = features_batch.iter()
            .map(|f| self.normalize(f))
            .collect();

        // Build targets
        let targets: Vec<Vec<f64>> = class_ids.iter()
            .map(|&cid| {
                let mut target = vec![0.0; self.n_classes];
                if cid < self.n_classes {
                    target[cid] = 1.0;
                }
                target
            })
            .collect();

        self.readout.train(&normalized, &targets);
    }

    pub fn is_trained(&self) -> bool {
        self.readout.trained()
    }
}

// ---------------------------------------------------------------------------
// AnomalyDetectionHead — anomaly scoring
// ---------------------------------------------------------------------------

/// Result of anomaly detection.
#[derive(Clone, Debug)]
pub struct AnomalyResult {
    pub score: f64,
    pub is_anomalous: bool,
    pub omega_z: f64,
    pub phi_z: f64,
    pub coherence_z: f64,
    pub energy_z: f64,
}

/// Anomaly detection head using StyleFingerprint comparison.
#[derive(Clone, Debug)]
pub struct AnomalyDetectionHead {
    pub fingerprint: Option<StyleFingerprint>,
    pub threshold: f64,
}

impl AnomalyDetectionHead {
    pub fn new(threshold: f64) -> Self {
        Self {
            fingerprint: None,
            threshold,
        }
    }

    pub fn set_fingerprint(&mut self, fp: StyleFingerprint) {
        self.fingerprint = Some(fp);
    }

    pub fn score(&self, state: &ReservoirState, probs: &[f64]) -> AnomalyResult {
        let fp = match &self.fingerprint {
            Some(fp) => fp,
            None => {
                return AnomalyResult {
                    score: 0.0, is_anomalous: false,
                    omega_z: 0.0, phi_z: 0.0, coherence_z: 0.0, energy_z: 0.0,
                };
            }
        };

        let omega = compute_omega(probs);
        let phi = compute_phi(probs);

        let omega_z = if fp.omega_stats.1 > 1e-10 {
            ((omega - fp.omega_stats.0) / fp.omega_stats.1).abs()
        } else { 0.0 };

        let phi_z = if fp.phi_stats.1 > 1e-10 {
            ((phi - fp.phi_stats.0) / fp.phi_stats.1).abs()
        } else { 0.0 };

        let coherence_z = if fp.coherence > 1e-10 {
            ((state.phase_coherence - fp.coherence) / fp.coherence.max(0.1)).abs()
        } else { 0.0 };

        let norm_energy = state.resonance_energy.min(10.0) / 10.0;
        let expected_energy: f64 = fp.scale_energies.iter().sum::<f64>() / 3.0;
        let energy_z = if expected_energy > 1e-10 {
            ((norm_energy - expected_energy) / expected_energy.max(0.1)).abs()
        } else { 0.0 };

        let score = omega_z.max(phi_z).max(coherence_z).max(energy_z).min(10.0);

        AnomalyResult {
            score,
            is_anomalous: score > self.threshold,
            omega_z, phi_z, coherence_z, energy_z,
        }
    }

    pub fn is_trained(&self) -> bool {
        self.fingerprint.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autocomplete_head() {
        let feature_dim = 10;
        let vocab_size = 5;
        let head = AutocompleteHead::new(feature_dim, vocab_size, 0);
        // Quantum measurement is always ready
        assert!(head.is_trained());
    }

    #[test]
    fn test_classification_head() {
        let feature_dim = 10;
        let labels = vec!["positive".into(), "negative".into(), "neutral".into()];
        let mut head = ClassificationHead::new(feature_dim, labels, 0);

        let features = vec![
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let classes = vec![0, 1, 2];

        head.train(&features, &classes);
        assert!(head.is_trained());
        assert_eq!(head.feature_means.len(), 10);
        assert_eq!(head.feature_stds.len(), 10);
    }
}
