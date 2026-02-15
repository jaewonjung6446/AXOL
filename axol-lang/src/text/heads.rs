//! Task-specific heads for the Wave Text Engine.
//!
//! Each head maps reservoir features to a specific task output.
//! All heads use LinearReadout internally, trained via lstsq (one-shot).
//!
//! Available heads:
//!   - AutocompleteHead: next-token prediction
//!   - SentenceGenerationHead: multi-token generation with quality gate
//!   - ClassificationHead: text classification
//!   - AnomalyDetectionHead: anomaly scoring via fingerprint comparison

use super::readout::LinearReadout;
use super::reservoir::{WaveResonanceReservoir, ReservoirState};
use super::fingerprint::StyleFingerprint;
use super::generator::{
    TokenPrediction, GenerationResult, softmax, compute_omega, compute_phi,
};
use super::sps::SemanticPhaseSpace;
use super::tokenizer::{Vocabulary, EOS_ID, UNK_ID};

// ---------------------------------------------------------------------------
// AutocompleteHead — next token prediction
// ---------------------------------------------------------------------------

/// Predicts the next token from reservoir state.
///
/// Pipeline: ReservoirState → features → LinearReadout → softmax → Ω/Φ
#[derive(Clone, Debug)]
pub struct AutocompleteHead {
    pub readout: LinearReadout,
    pub vocab_size: usize,
}

impl AutocompleteHead {
    /// Create an untrained autocomplete head.
    pub fn new(feature_dim: usize, vocab_size: usize) -> Self {
        Self {
            readout: LinearReadout::new(feature_dim, vocab_size),
            vocab_size,
        }
    }

    /// Predict the next token from reservoir state.
    pub fn predict(&self, state: &ReservoirState, vocab: &Vocabulary) -> TokenPrediction {
        let features = state.to_feature_vector();
        let scores = self.readout.forward(&features);
        let probs = softmax(&scores);

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

    /// Predict top-n tokens with probabilities.
    pub fn predict_top_n(
        &self,
        state: &ReservoirState,
        vocab: &Vocabulary,
        n: usize,
    ) -> Vec<TokenPrediction> {
        let features = state.to_feature_vector();
        let scores = self.readout.forward(&features);
        let probs = softmax(&scores);
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

    /// Train the autocomplete head from (feature_vector, target_token_id) pairs.
    pub fn train(&mut self, features_batch: &[Vec<f64>], target_ids: &[usize]) {
        let n = features_batch.len();
        if n == 0 || target_ids.len() != n {
            return;
        }

        // Convert target IDs to one-hot (smoothed) target vectors
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
        self.readout.trained
    }
}

// ---------------------------------------------------------------------------
// SentenceGenerationHead — autoregressive generation with quality gate
// ---------------------------------------------------------------------------

/// Multi-token generation using autoregressive loop with Ω/Φ quality gate.
///
/// Generates tokens one at a time, feeding each back through the reservoir.
/// Automatically stops when Ω drops below threshold (quality gate).
#[derive(Clone, Debug)]
pub struct SentenceGenerationHead {
    pub autocomplete: AutocompleteHead,
    /// Minimum Ω for continued generation (quality gate)
    pub min_omega: f64,
    /// Maximum tokens to generate
    pub max_tokens: usize,
}

impl SentenceGenerationHead {
    /// Create a new sentence generation head.
    pub fn new(feature_dim: usize, vocab_size: usize) -> Self {
        Self {
            autocomplete: AutocompleteHead::new(feature_dim, vocab_size),
            min_omega: 0.0,
            max_tokens: 30,
        }
    }

    /// Generate a sequence of tokens from a prompt.
    ///
    /// Uses the reservoir for context and the autocomplete head for prediction.
    /// Stops on EOS, quality gate (Ω < min_omega), or max_tokens.
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
            let pred = self.autocomplete.predict(&state, vocab);

            // Stop conditions
            if pred.token_id == EOS_ID {
                break;
            }
            if pred.omega < self.min_omega && !predictions.is_empty() {
                break;
            }

            // Feed predicted token back through reservoir
            let token_wave = sps.token_to_wave(pred.token_id);
            let pos = prompt_ids.len() + step;
            reservoir.process_token(&token_wave, pos);

            all_ids.push(pred.token_id);
            generated_tokens.push(pred.token.clone());
            predictions.push(pred);
        }

        let prompt_text = prompt_ids.iter()
            .filter(|&&id| id != 0 && id != 2 && id != 3) // skip PAD, BOS, EOS
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

    /// Train the underlying autocomplete head.
    pub fn train(&mut self, features_batch: &[Vec<f64>], target_ids: &[usize]) {
        self.autocomplete.train(features_batch, target_ids);
    }

    pub fn is_trained(&self) -> bool {
        self.autocomplete.is_trained()
    }
}

// ---------------------------------------------------------------------------
// ClassificationHead — text classification
// ---------------------------------------------------------------------------

/// Result of a classification prediction.
#[derive(Clone, Debug)]
pub struct ClassificationResult {
    /// Predicted class index
    pub class_id: usize,
    /// Predicted class label
    pub class_label: String,
    /// Confidence (probability of predicted class)
    pub confidence: f64,
    /// Probability distribution over all classes
    pub probabilities: Vec<f64>,
    /// All class labels with probabilities
    pub class_probs: Vec<(String, f64)>,
    /// Omega (cohesion) of the prediction
    pub omega: f64,
    /// Phi (clarity) of the prediction
    pub phi: f64,
}

/// Text classification head.
///
/// Maps reservoir features to class probabilities via LinearReadout + softmax.
#[derive(Clone, Debug)]
pub struct ClassificationHead {
    pub readout: LinearReadout,
    pub n_classes: usize,
    pub class_labels: Vec<String>,
}

impl ClassificationHead {
    /// Create a new classification head.
    pub fn new(feature_dim: usize, class_labels: Vec<String>) -> Self {
        let n_classes = class_labels.len();
        Self {
            readout: LinearReadout::new(feature_dim, n_classes),
            n_classes,
            class_labels,
        }
    }

    /// Classify a text from its reservoir state.
    pub fn predict(&self, state: &ReservoirState) -> ClassificationResult {
        let features = state.to_feature_vector();
        let scores = self.readout.forward(&features);
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

    /// Train the classification head from (features, class_id) pairs.
    pub fn train(&mut self, features_batch: &[Vec<f64>], class_ids: &[usize]) {
        let n = features_batch.len();
        if n == 0 || class_ids.len() != n {
            return;
        }

        let targets: Vec<Vec<f64>> = class_ids.iter()
            .map(|&cid| {
                let mut target = vec![0.0; self.n_classes];
                if cid < self.n_classes {
                    target[cid] = 1.0;
                }
                target
            })
            .collect();

        self.readout.train(features_batch, &targets);
    }

    pub fn is_trained(&self) -> bool {
        self.readout.trained
    }
}

// ---------------------------------------------------------------------------
// AnomalyDetectionHead — anomaly scoring
// ---------------------------------------------------------------------------

/// Result of anomaly detection.
#[derive(Clone, Debug)]
pub struct AnomalyResult {
    /// Anomaly score (higher = more anomalous)
    pub score: f64,
    /// Is this considered anomalous? (score > threshold)
    pub is_anomalous: bool,
    /// Component scores
    pub omega_z: f64,
    pub phi_z: f64,
    pub coherence_z: f64,
    pub energy_z: f64,
}

/// Anomaly detection head using StyleFingerprint comparison.
///
/// Computes anomaly score as the deviation of observed Ω/Φ/coherence/energy
/// from expected values (established by the fingerprint).
#[derive(Clone, Debug)]
pub struct AnomalyDetectionHead {
    /// Reference fingerprint (normal behavior)
    pub fingerprint: Option<StyleFingerprint>,
    /// Anomaly threshold
    pub threshold: f64,
}

impl AnomalyDetectionHead {
    /// Create a new anomaly detection head.
    pub fn new(threshold: f64) -> Self {
        Self {
            fingerprint: None,
            threshold,
        }
    }

    /// Set the reference fingerprint.
    pub fn set_fingerprint(&mut self, fp: StyleFingerprint) {
        self.fingerprint = Some(fp);
    }

    /// Compute anomaly score for a reservoir state.
    ///
    /// Uses z-scores of Ω, Φ, coherence, and energy relative to the
    /// fingerprint's statistics.
    pub fn score(&self, state: &ReservoirState, probs: &[f64]) -> AnomalyResult {
        let fp = match &self.fingerprint {
            Some(fp) => fp,
            None => {
                return AnomalyResult {
                    score: 0.0,
                    is_anomalous: false,
                    omega_z: 0.0,
                    phi_z: 0.0,
                    coherence_z: 0.0,
                    energy_z: 0.0,
                };
            }
        };

        let omega = compute_omega(probs);
        let phi = compute_phi(probs);

        // Z-scores
        let omega_z = if fp.omega_stats.1 > 1e-10 {
            ((omega - fp.omega_stats.0) / fp.omega_stats.1).abs()
        } else {
            0.0
        };

        let phi_z = if fp.phi_stats.1 > 1e-10 {
            ((phi - fp.phi_stats.0) / fp.phi_stats.1).abs()
        } else {
            0.0
        };

        let coherence_z = if fp.coherence > 1e-10 {
            ((state.phase_coherence - fp.coherence) / fp.coherence.max(0.1)).abs()
        } else {
            0.0
        };

        let norm_energy = state.resonance_energy.min(10.0) / 10.0;
        let expected_energy: f64 = fp.scale_energies.iter().sum::<f64>() / 3.0;
        let energy_z = if expected_energy > 1e-10 {
            ((norm_energy - expected_energy) / expected_energy.max(0.1)).abs()
        } else {
            0.0
        };

        // Combined score: max of z-scores (most anomalous dimension)
        let score = omega_z.max(phi_z).max(coherence_z).max(energy_z);
        let score = score.min(10.0); // cap at 10

        AnomalyResult {
            score,
            is_anomalous: score > self.threshold,
            omega_z,
            phi_z,
            coherence_z,
            energy_z,
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
        let mut head = AutocompleteHead::new(feature_dim, vocab_size);

        // Train with simple data
        let features = vec![
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let targets = vec![0, 1, 2];

        head.train(&features, &targets);
        assert!(head.is_trained());
    }

    #[test]
    fn test_classification_head() {
        let feature_dim = 10;
        let labels = vec!["positive".into(), "negative".into(), "neutral".into()];
        let mut head = ClassificationHead::new(feature_dim, labels);

        let features = vec![
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let classes = vec![0, 1, 2];

        head.train(&features, &classes);
        assert!(head.is_trained());
    }
}
