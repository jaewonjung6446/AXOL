//! Wave Text Engine (WTE) — SDK interface.
//!
//! The top-level API for AXOL's wave-based text processing.
//! Integrates all components: BPE tokenizer, SPS, WRR reservoir,
//! and task-specific heads.
//!
//! Key properties:
//!   - Zero physics parameters to learn (reservoir is physics-based)
//!   - Only output layer trained via lstsq (one-shot)
//!   - CPU-only, sub-millisecond inference
//!   - Domain adaptation in < 500ms with 10-100 documents

use super::tokenizer::{BpeTokenizer, Vocabulary, EOS_ID};
use super::sps::SemanticPhaseSpace;
use super::reservoir::{WaveResonanceReservoir, ReservoirState};
use super::readout::{LinearReadout, MlpReadout, ReadoutLayer};
use super::heads::{
    AutocompleteHead, SentenceGenerationHead, ClassificationHead,
    AnomalyDetectionHead, ClassificationResult,
};
use super::fingerprint::{StyleFingerprint, AnomalyScore};
use super::generator::{
    TokenPrediction, GenerationResult,
};

// ---------------------------------------------------------------------------
// EngineConfig
// ---------------------------------------------------------------------------

/// Configuration for building a WaveTextEngine.
#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// Phase space dimension (default: 64)
    pub dim: usize,
    /// BPE merge operations (default: 200)
    pub num_merges: usize,
    /// Maximum vocabulary size (default: 2000)
    pub max_vocab: usize,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Anomaly detection threshold (default: 1.5)
    pub anomaly_threshold: f64,
    /// Number of nodes per reservoir register (default: 4)
    /// More nodes = more capacity for superposition, longer effective context
    pub num_nodes: usize,
    /// Hidden dimension for MLP readout (default: 0 = Linear)
    /// When > 0, uses MLP: features → hidden(tanh) → output
    pub hidden_dim: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            dim: 64,
            num_merges: 200,
            max_vocab: 2000,
            seed: 42,
            anomaly_threshold: 1.5,
            num_nodes: 4,
            hidden_dim: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// WaveTextEngine
// ---------------------------------------------------------------------------

/// The Wave Text Engine — a complete text AI SDK.
///
/// Provides autocomplete, generation, classification, anomaly detection,
/// and style fingerprinting. All built on physics-based wave resonance
/// with zero learnable physics parameters.
pub struct WaveTextEngine {
    /// BPE tokenizer
    pub bpe: BpeTokenizer,
    /// Vocabulary (derived from BPE)
    pub vocab: Vocabulary,
    /// Semantic phase space
    pub sps: SemanticPhaseSpace,
    /// Wave resonance reservoir
    pub reservoir: WaveResonanceReservoir,
    /// Configuration
    pub config: EngineConfig,

    // Task-specific heads
    pub autocomplete: Option<AutocompleteHead>,
    pub generator: Option<SentenceGenerationHead>,
    pub classifier: Option<ClassificationHead>,
    pub anomaly_detector: Option<AnomalyDetectionHead>,

    /// Feature dimension for readout layers
    feature_dim: usize,
}

impl WaveTextEngine {
    /// Build a WaveTextEngine from a corpus with default config.
    pub fn from_corpus(documents: &[&str]) -> Self {
        Self::from_corpus_with_config(documents, &EngineConfig::default())
    }

    /// Build a WaveTextEngine from a corpus with custom config.
    pub fn from_corpus_with_config(documents: &[&str], config: &EngineConfig) -> Self {
        // Build BPE tokenizer
        let bpe = BpeTokenizer::from_corpus(documents, config.num_merges, config.max_vocab);
        let vocab = bpe.to_vocabulary();

        // Pre-encode all documents
        let encoded: Vec<Vec<usize>> = documents.iter()
            .map(|s| bpe.encode(s))
            .collect();

        // Build SPS from encoded sequences
        let sps = SemanticPhaseSpace::from_encoded_corpus(
            bpe.size,
            &encoded,
            config.dim,
            config.seed,
        );

        // Build reservoir with multi-node configuration
        let reservoir = WaveResonanceReservoir::new_with_nodes(config.dim, config.num_nodes);

        // Feature dimension: (1 + num_scales) * dim + 2
        let feature_dim = ReservoirState::feature_dim(config.dim, reservoir.num_scales());

        // Build autocomplete head
        let autocomplete = Some(AutocompleteHead::new(feature_dim, bpe.size, config.hidden_dim));

        // Build generation head
        let generator = Some(SentenceGenerationHead::new(feature_dim, bpe.size, config.hidden_dim));

        // Build anomaly detector
        let anomaly_detector = Some(AnomalyDetectionHead::new(config.anomaly_threshold));

        let engine = Self {
            bpe,
            vocab,
            sps,
            reservoir,
            config: config.clone(),
            autocomplete,
            generator,
            classifier: None,
            anomaly_detector,
            feature_dim,
        };

        // No autocomplete training needed — quantum measurement is training-free.
        // Classification and anomaly detection are trained on-demand.

        engine
    }

    // =======================================================================
    // Training
    // =======================================================================

    /// Train the autocomplete and generator heads from a corpus.
    ///
    /// Uses incremental `process_token()` per document for O(n) training
    /// instead of O(n²) `process_sequence()` per prefix.
    fn train_autocomplete(&mut self, documents: &[&str]) {
        let mut features_batch = Vec::new();
        let mut target_ids = Vec::new();

        for doc in documents {
            let ids = self.bpe.encode(doc);
            if ids.len() < 3 {
                continue; // need at least BOS + 1 token + EOS
            }

            // Incremental processing: walk through tokens once per document
            let waves = self.sps.tokens_to_waves(&ids);
            self.reservoir.reset();

            for pos in 0..ids.len() - 1 {
                self.reservoir.process_token(&waves[pos], pos);
                let target = ids[pos + 1];

                if target == EOS_ID {
                    continue;
                }

                let state = self.reservoir.current_state();
                let features = state.to_feature_vector();

                features_batch.push(features);
                target_ids.push(target);
            }
        }

        if features_batch.is_empty() {
            return;
        }

        // Train autocomplete head
        if let Some(ref mut ac) = self.autocomplete {
            ac.train(&features_batch, &target_ids);
        }

        // Generator uses quantum measurement — no readout sharing needed.
    }

    /// Train a classifier on labeled documents.
    ///
    /// `labeled_docs`: slice of (document_text, class_id) pairs
    /// `class_labels`: label names for each class
    pub fn train_classifier(
        &mut self,
        labeled_docs: &[(&str, usize)],
        class_labels: Vec<String>,
    ) {
        let mut features_batch = Vec::new();
        let mut class_ids = Vec::new();

        for (doc, class_id) in labeled_docs {
            let ids = self.bpe.encode(doc);
            let token_waves = self.sps.tokens_to_waves(&ids);
            let state = self.reservoir.process_sequence(&token_waves);
            let features = state.to_feature_vector();

            features_batch.push(features);
            class_ids.push(*class_id);
        }

        let mut head = ClassificationHead::new(self.feature_dim, class_labels, self.config.hidden_dim);
        head.train(&features_batch, &class_ids);
        self.classifier = Some(head);
    }

    /// Train the anomaly detector by building a fingerprint from normal documents.
    pub fn train_anomaly_detector(&mut self, normal_docs: &[&str]) {
        let fp = self.fingerprint(normal_docs);
        if let Some(ref mut ad) = self.anomaly_detector {
            ad.set_fingerprint(fp);
        }
    }

    /// Adapt the engine to new domain documents (incremental training).
    ///
    /// Re-trains the autocomplete head using both the existing model
    /// knowledge and new documents. Typically completes in < 500ms.
    pub fn adapt(&mut self, new_documents: &[&str]) {
        // Simply re-train autocomplete on new documents
        // The reservoir physics doesn't change — only the readout
        self.train_autocomplete(new_documents);
    }

    // =======================================================================
    // Inference APIs
    // =======================================================================

    /// Autocomplete: predict the next token(s) from a prompt.
    pub fn autocomplete(&self, prompt: &str, n: usize) -> Vec<TokenPrediction> {
        let ac = match &self.autocomplete {
            Some(ac) if ac.is_trained() => ac,
            _ => return vec![],
        };

        let ids = self.bpe.encode(prompt);
        let token_waves = self.sps.tokens_to_waves(&ids);

        let mut reservoir = self.reservoir.clone();
        let state = reservoir.process_sequence(&token_waves);

        ac.predict_top_n(&state, &self.sps, &self.vocab, n)
    }

    /// Generate text by completing a prompt.
    pub fn generate(&self, prompt: &str, max_tokens: usize) -> GenerationResult {
        let gen = match &self.generator {
            Some(gen) if gen.is_trained() => gen,
            _ => {
                return GenerationResult {
                    prompt: prompt.to_string(),
                    generated: vec![],
                    full_text: prompt.to_string(),
                    predictions: vec![],
                };
            }
        };

        let ids = self.bpe.encode(prompt);
        // Remove trailing EOS from prompt encoding
        let ids: Vec<usize> = ids.into_iter()
            .filter(|&id| id != EOS_ID)
            .collect();

        let mut reservoir = self.reservoir.clone();
        gen.generate(&ids, &self.sps, &mut reservoir, &self.vocab, Some(max_tokens))
    }

    /// Classify a text.
    pub fn classify(&self, text: &str) -> Option<ClassificationResult> {
        let clf = match &self.classifier {
            Some(clf) if clf.is_trained() => clf,
            _ => return None,
        };

        let ids = self.bpe.encode(text);
        let token_waves = self.sps.tokens_to_waves(&ids);

        let mut reservoir = self.reservoir.clone();
        let state = reservoir.process_sequence(&token_waves);

        Some(clf.predict(&state))
    }

    /// Compute anomaly score for a text.
    pub fn anomaly_score(&self, text: &str) -> Option<AnomalyScore> {
        let ad = match &self.anomaly_detector {
            Some(ad) if ad.is_trained() => ad,
            _ => return None,
        };

        let ids = self.bpe.encode(text);
        let token_waves = self.sps.tokens_to_waves(&ids);

        let mut reservoir = self.reservoir.clone();
        let state = reservoir.process_sequence(&token_waves);

        let fp = ad.fingerprint.as_ref().unwrap();

        // Get prediction probabilities via quantum measurement (Born rule)
        let probs = if let Some(ref ac) = self.autocomplete {
            ac.quantum_probs(&state, &self.sps)
        } else {
            state.merged.probabilities()
        };

        Some(AnomalyScore::compute(
            &state,
            &probs,
            fp,
            self.config.anomaly_threshold,
        ))
    }

    /// Build a style fingerprint from a set of documents.
    pub fn fingerprint(&self, documents: &[&str]) -> StyleFingerprint {
        let mut states = Vec::with_capacity(documents.len());
        let mut probs_list = Vec::with_capacity(documents.len());

        for doc in documents {
            let ids = self.bpe.encode(doc);
            let token_waves = self.sps.tokens_to_waves(&ids);

            let mut reservoir = self.reservoir.clone();
            let state = reservoir.process_sequence(&token_waves);

            // Quantum measurement (Born rule) for probability distributions
            let probs = if let Some(ref ac) = self.autocomplete {
                ac.quantum_probs(&state, &self.sps)
            } else {
                state.merged.probabilities()
            };

            probs_list.push(probs);
            states.push(state);
        }

        StyleFingerprint::from_states(&states, &probs_list)
    }

    // =======================================================================
    // Word-level quantum measurement
    // =======================================================================

    /// Compute word-level resonance via Born rule.
    ///
    /// |⟨ψ_merged | ψ_word⟩|² where ψ_word = superposition of BPE token waves.
    /// Lifts measurement from subword tokens to semantic word units.
    pub fn word_resonance(&self, state: &ReservoirState, word: &str) -> f64 {
        let ids = self.bpe.encode_word(word);
        let word_wave = self.sps.word_to_wave(&ids);

        let merged = &state.merged;
        let dim = merged.dim.min(word_wave.dim);
        let mut inner = num_complex::Complex64::new(0.0, 0.0);
        for k in 0..dim {
            inner += merged.amplitudes.data[k].conj() * word_wave.amplitudes.data[k];
        }
        inner.norm_sqr()
    }

    /// Compute sentence-level resonance via Born rule.
    ///
    /// `|⟨ψ_input | ψ_response⟩|²` where both waves are merged reservoir states.
    /// Used by ChatEngine to select the best response from a pool.
    pub fn sentence_resonance(&self, input_state: &ReservoirState, response: &str) -> f64 {
        let resp_state = self.process_text(response);
        let merged_in = &input_state.merged;
        let merged_resp = &resp_state.merged;
        let dim = merged_in.dim.min(merged_resp.dim);
        let mut inner = num_complex::Complex64::new(0.0, 0.0);
        for k in 0..dim {
            inner += merged_in.amplitudes.data[k].conj() * merged_resp.amplitudes.data[k];
        }
        inner.norm_sqr()
    }

    /// Measure resonance for a list of candidate words against a text context.
    ///
    /// Returns (word, resonance) pairs sorted by descending resonance.
    pub fn word_resonances(&self, context: &str, candidates: &[&str]) -> Vec<(String, f64)> {
        let ids = self.bpe.encode(context);
        let token_waves = self.sps.tokens_to_waves(&ids);
        let mut reservoir = self.reservoir.clone();
        let state = reservoir.process_sequence(&token_waves);

        let mut results: Vec<(String, f64)> = candidates.iter()
            .map(|&w| (w.to_string(), self.word_resonance(&state, w)))
            .collect();

        // Normalize to probabilities
        let sum: f64 = results.iter().map(|(_, r)| r).sum();
        if sum > 1e-10 {
            for (_, r) in results.iter_mut() { *r /= sum; }
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    // =======================================================================
    // Serialization
    // =======================================================================

    /// Serialize the engine to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut data = Vec::new();

        // Magic number + version (WTE4 = quantum measurement + feature normalization)
        data.extend_from_slice(b"WTE4");

        // Config
        push_u64(&mut data, self.config.dim as u64);
        push_u64(&mut data, self.config.num_merges as u64);
        push_u64(&mut data, self.config.max_vocab as u64);
        push_u64(&mut data, self.config.seed);
        push_f64(&mut data, self.config.anomaly_threshold);
        push_u64(&mut data, self.config.num_nodes as u64);
        push_u64(&mut data, self.config.hidden_dim as u64);

        // BPE tokenizer
        push_u64(&mut data, self.bpe.size as u64);
        push_u64(&mut data, self.bpe.merges.len() as u64);
        for merge in &self.bpe.merges {
            push_string(&mut data, &merge.pair.0);
            push_string(&mut data, &merge.pair.1);
            push_string(&mut data, &merge.merged);
            push_u64(&mut data, merge.rank as u64);
        }
        // Token vocabulary
        push_u64(&mut data, self.bpe.id_to_token.len() as u64);
        for token in &self.bpe.id_to_token {
            push_string(&mut data, token);
        }

        // SPS embeddings
        push_u64(&mut data, self.sps.vocab_size as u64);
        push_u64(&mut data, self.sps.dim as u64);
        for emb in &self.sps.embeddings {
            for &v in emb {
                push_f64(&mut data, v);
            }
        }

        // SPS transform matrix
        for &v in &self.sps.transform.data {
            push_f64(&mut data, v as f64);
        }

        // Autocomplete: quantum measurement is training-free, no readout to serialize
        data.push(0u8);

        // Classifier (if trained)
        let has_clf = self.classifier.as_ref().map_or(false, |c| c.is_trained());
        data.push(has_clf as u8);
        if has_clf {
            let clf = self.classifier.as_ref().unwrap();
            push_u64(&mut data, clf.n_classes as u64);
            for label in &clf.class_labels {
                push_string(&mut data, label);
            }
            serialize_readout_layer(&mut data, &clf.readout);
            // Feature normalization stats (WTE4+)
            push_u64(&mut data, clf.feature_means.len() as u64);
            for &v in &clf.feature_means { push_f64(&mut data, v); }
            for &v in &clf.feature_stds { push_f64(&mut data, v); }
        }

        // Anomaly detector fingerprint
        let has_ad = self.anomaly_detector.as_ref().map_or(false, |a| a.is_trained());
        data.push(has_ad as u8);
        if has_ad {
            let ad = self.anomaly_detector.as_ref().unwrap();
            let fp = ad.fingerprint.as_ref().unwrap();
            push_f64(&mut data, fp.coherence);
            push_f64(&mut data, fp.omega_stats.0);
            push_f64(&mut data, fp.omega_stats.1);
            push_f64(&mut data, fp.phi_stats.0);
            push_f64(&mut data, fp.phi_stats.1);
            for &e in &fp.scale_energies {
                push_f64(&mut data, e);
            }
            push_u64(&mut data, fp.n_documents as u64);
        }

        data
    }

    /// Deserialize an engine from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        // Magic number (WTE1=legacy, WTE2=multi-node, WTE3=MLP, WTE4=quantum+normalization)
        if data.len() < 4 {
            return Err("Invalid WTE format".to_string());
        }
        let magic = &data[0..4];
        let version = match magic {
            b"WTE1" => 1,
            b"WTE2" => 2,
            b"WTE3" => 3,
            b"WTE4" => 4,
            _ => return Err("Invalid WTE format".to_string()),
        };
        let mut pos = 4;

        // Config
        let dim = read_u64(data, &mut pos)? as usize;
        let num_merges = read_u64(data, &mut pos)? as usize;
        let max_vocab = read_u64(data, &mut pos)? as usize;
        let seed = read_u64(data, &mut pos)?;
        let anomaly_threshold = read_f64(data, &mut pos)?;
        let num_nodes = if version >= 2 {
            read_u64(data, &mut pos)? as usize
        } else {
            1 // backward compat: single node
        };
        let hidden_dim = if version >= 3 {
            read_u64(data, &mut pos)? as usize
        } else {
            0 // backward compat: Linear readout
        };

        let config = EngineConfig {
            dim,
            num_merges,
            max_vocab,
            seed,
            anomaly_threshold,
            num_nodes,
            hidden_dim,
        };

        // BPE tokenizer
        let bpe_size = read_u64(data, &mut pos)? as usize;
        let n_merges = read_u64(data, &mut pos)? as usize;
        let mut merges = Vec::with_capacity(n_merges);
        for _ in 0..n_merges {
            let left = read_string(data, &mut pos)?;
            let right = read_string(data, &mut pos)?;
            let merged = read_string(data, &mut pos)?;
            let rank = read_u64(data, &mut pos)? as usize;
            merges.push(super::tokenizer::BpeMerge {
                pair: (left, right),
                merged,
                rank,
            });
        }

        let n_tokens = read_u64(data, &mut pos)? as usize;
        let mut id_to_token = Vec::with_capacity(n_tokens);
        let mut token_to_id = std::collections::HashMap::new();
        for i in 0..n_tokens {
            let token = read_string(data, &mut pos)?;
            token_to_id.insert(token.clone(), i);
            id_to_token.push(token);
        }

        let bpe = BpeTokenizer {
            merges,
            token_to_id,
            id_to_token,
            size: bpe_size,
        };
        let vocab = bpe.to_vocabulary();

        // SPS
        let vocab_size = read_u64(data, &mut pos)? as usize;
        let sps_dim = read_u64(data, &mut pos)? as usize;
        let mut embeddings = Vec::with_capacity(vocab_size);
        for _ in 0..vocab_size {
            let mut emb = Vec::with_capacity(sps_dim);
            for _ in 0..sps_dim {
                emb.push(read_f64(data, &mut pos)?);
            }
            embeddings.push(emb);
        }

        let mut transform_data = Vec::with_capacity(sps_dim * sps_dim);
        for _ in 0..sps_dim * sps_dim {
            transform_data.push(read_f64(data, &mut pos)? as f32);
        }

        let sps = SemanticPhaseSpace {
            dim: sps_dim,
            vocab_size,
            embeddings,
            transform: crate::types::TransMatrix::new(transform_data, sps_dim, sps_dim),
        };

        // Reservoir (with configured num_nodes)
        let reservoir = WaveResonanceReservoir::new_with_nodes(dim, num_nodes);
        let feature_dim = ReservoirState::feature_dim(dim, reservoir.num_scales());

        // Autocomplete: quantum measurement is training-free
        let has_ac = read_u8(data, &mut pos)? != 0;
        if has_ac && version <= 3 {
            // WTE3 backward compat: skip serialized readout (unused with quantum measurement)
            let _readout = deserialize_readout_layer(data, &mut pos, version)?;
        }
        let autocomplete = Some(AutocompleteHead::new(feature_dim, bpe_size, hidden_dim));

        // Generator uses quantum measurement — no readout sharing needed
        let generator = Some(SentenceGenerationHead::new(feature_dim, bpe_size, hidden_dim));

        // Classifier
        let has_clf = read_u8(data, &mut pos)? != 0;
        let classifier = if has_clf {
            let n_classes = read_u64(data, &mut pos)? as usize;
            let mut class_labels = Vec::with_capacity(n_classes);
            for _ in 0..n_classes {
                class_labels.push(read_string(data, &mut pos)?);
            }
            let readout = deserialize_readout_layer(data, &mut pos, version)?;
            // Feature normalization stats (WTE4+)
            let (feature_means, feature_stds) = if version >= 4 {
                let fd = read_u64(data, &mut pos)? as usize;
                let mut means = Vec::with_capacity(fd);
                for _ in 0..fd { means.push(read_f64(data, &mut pos)?); }
                let mut stds = Vec::with_capacity(fd);
                for _ in 0..fd { stds.push(read_f64(data, &mut pos)?); }
                (means, stds)
            } else {
                (Vec::new(), Vec::new())
            };
            Some(ClassificationHead { readout, n_classes, class_labels, feature_means, feature_stds })
        } else {
            None
        };

        // Anomaly detector
        let has_ad = read_u8(data, &mut pos)? != 0;
        let anomaly_detector = if has_ad {
            let coherence = read_f64(data, &mut pos)?;
            let omega_mean = read_f64(data, &mut pos)?;
            let omega_std = read_f64(data, &mut pos)?;
            let phi_mean = read_f64(data, &mut pos)?;
            let phi_std = read_f64(data, &mut pos)?;
            let mut scale_energies = [0.0; 3];
            for i in 0..3 {
                scale_energies[i] = read_f64(data, &mut pos)?;
            }
            let n_documents = read_u64(data, &mut pos)? as usize;

            let fp = StyleFingerprint {
                signature: crate::wave::Wave::from_classical(
                    &crate::types::FloatVec::zeros(dim),
                ),
                scale_energies,
                coherence,
                omega_stats: (omega_mean, omega_std),
                phi_stats: (phi_mean, phi_std),
                n_documents,
                dim,
            };
            let mut ad = AnomalyDetectionHead::new(anomaly_threshold);
            ad.set_fingerprint(fp);
            Some(ad)
        } else {
            Some(AnomalyDetectionHead::new(anomaly_threshold))
        };

        Ok(Self {
            bpe,
            vocab,
            sps,
            reservoir,
            config,
            autocomplete,
            generator,
            classifier,
            anomaly_detector,
            feature_dim,
        })
    }

    // =======================================================================
    // Utility
    // =======================================================================

    /// Compute model size in bytes.
    pub fn model_size_bytes(&self) -> usize {
        let sps_size = self.sps.vocab_size * self.sps.dim * 8;
        let transform_size = self.sps.dim * self.sps.dim * 4;

        let ac_size = self.autocomplete.as_ref()
            .map_or(0, |ac| ac.readout.size_bytes());
        let clf_size = self.classifier.as_ref()
            .map_or(0, |c| c.readout.size_bytes());

        sps_size + transform_size + ac_size + clf_size
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.bpe.size
    }

    /// Get the feature dimension.
    pub fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    /// Process a text through the reservoir and return the state.
    pub fn process_text(&self, text: &str) -> ReservoirState {
        let ids = self.bpe.encode(text);
        let token_waves = self.sps.tokens_to_waves(&ids);
        let mut reservoir = self.reservoir.clone();
        reservoir.process_sequence(&token_waves)
    }
}

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

/// Serialize a ReadoutLayer (type_tag + weights).
fn serialize_readout_layer(data: &mut Vec<u8>, layer: &ReadoutLayer) {
    data.push(layer.type_tag()); // 0=Linear, 1=MLP
    match layer {
        ReadoutLayer::Linear(r) => {
            push_u64(data, r.feature_dim as u64);
            push_u64(data, r.output_dim as u64);
            for &w in &r.weights { push_f64(data, w); }
            for &b in &r.bias { push_f64(data, b); }
        }
        ReadoutLayer::Mlp(r) => {
            push_u64(data, r.feature_dim as u64);
            push_u64(data, r.hidden_dim as u64);
            push_u64(data, r.output_dim as u64);
            for &w in &r.w_hidden { push_f64(data, w); }
            for &b in &r.b_hidden { push_f64(data, b); }
            for &w in &r.w_output { push_f64(data, w); }
            for &b in &r.b_output { push_f64(data, b); }
        }
    }
}

/// Deserialize a ReadoutLayer. For WTE1/WTE2 (version < 3), assume Linear.
fn deserialize_readout_layer(data: &[u8], pos: &mut usize, version: u32) -> Result<ReadoutLayer, String> {
    let type_tag = if version >= 3 {
        read_u8(data, pos)?
    } else {
        0 // backward compat: Linear
    };

    match type_tag {
        0 => {
            let feat_dim = read_u64(data, pos)? as usize;
            let out_dim = read_u64(data, pos)? as usize;
            let mut weights = Vec::with_capacity(out_dim * feat_dim);
            for _ in 0..out_dim * feat_dim {
                weights.push(read_f64(data, pos)?);
            }
            let mut bias = Vec::with_capacity(out_dim);
            for _ in 0..out_dim {
                bias.push(read_f64(data, pos)?);
            }
            Ok(ReadoutLayer::Linear(LinearReadout {
                feature_dim: feat_dim,
                output_dim: out_dim,
                weights,
                bias,
                trained: true,
            }))
        }
        1 => {
            let feat_dim = read_u64(data, pos)? as usize;
            let hid_dim = read_u64(data, pos)? as usize;
            let out_dim = read_u64(data, pos)? as usize;
            let mut w_hidden = Vec::with_capacity(hid_dim * feat_dim);
            for _ in 0..hid_dim * feat_dim {
                w_hidden.push(read_f64(data, pos)?);
            }
            let mut b_hidden = Vec::with_capacity(hid_dim);
            for _ in 0..hid_dim {
                b_hidden.push(read_f64(data, pos)?);
            }
            let mut w_output = Vec::with_capacity(out_dim * hid_dim);
            for _ in 0..out_dim * hid_dim {
                w_output.push(read_f64(data, pos)?);
            }
            let mut b_output = Vec::with_capacity(out_dim);
            for _ in 0..out_dim {
                b_output.push(read_f64(data, pos)?);
            }
            Ok(ReadoutLayer::Mlp(MlpReadout {
                feature_dim: feat_dim,
                hidden_dim: hid_dim,
                output_dim: out_dim,
                w_hidden,
                b_hidden,
                w_output,
                b_output,
                trained: true,
            }))
        }
        _ => Err(format!("Unknown readout type tag: {}", type_tag)),
    }
}

fn push_u64(data: &mut Vec<u8>, val: u64) {
    data.extend_from_slice(&val.to_le_bytes());
}

fn push_f64(data: &mut Vec<u8>, val: f64) {
    data.extend_from_slice(&val.to_le_bytes());
}

fn push_string(data: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    push_u64(data, bytes.len() as u64);
    data.extend_from_slice(bytes);
}

fn read_u8(data: &[u8], pos: &mut usize) -> Result<u8, String> {
    if *pos >= data.len() {
        return Err("Unexpected end of data".to_string());
    }
    let val = data[*pos];
    *pos += 1;
    Ok(val)
}

fn read_u64(data: &[u8], pos: &mut usize) -> Result<u64, String> {
    if *pos + 8 > data.len() {
        return Err("Unexpected end of data".to_string());
    }
    let bytes: [u8; 8] = data[*pos..*pos + 8].try_into()
        .map_err(|_| "Failed to read u64")?;
    *pos += 8;
    Ok(u64::from_le_bytes(bytes))
}

fn read_f64(data: &[u8], pos: &mut usize) -> Result<f64, String> {
    if *pos + 8 > data.len() {
        return Err("Unexpected end of data".to_string());
    }
    let bytes: [u8; 8] = data[*pos..*pos + 8].try_into()
        .map_err(|_| "Failed to read f64")?;
    *pos += 8;
    Ok(f64::from_le_bytes(bytes))
}

fn read_string(data: &[u8], pos: &mut usize) -> Result<String, String> {
    let len = read_u64(data, pos)? as usize;
    if *pos + len > data.len() {
        return Err("Unexpected end of data".to_string());
    }
    let s = String::from_utf8(data[*pos..*pos + len].to_vec())
        .map_err(|e| format!("Invalid UTF-8: {}", e))?;
    *pos += len;
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::data::tiny_corpus;

    #[test]
    fn test_engine_construction() {
        let corpus = tiny_corpus();
        let engine = WaveTextEngine::from_corpus(&corpus);

        assert!(engine.autocomplete.is_some());
        assert!(engine.generator.is_some());
        assert!(engine.vocab_size() > 0);
        assert!(engine.feature_dim() > 0);
    }

    #[test]
    fn test_engine_autocomplete() {
        let corpus = tiny_corpus();
        let engine = WaveTextEngine::from_corpus(&corpus);

        let preds = engine.autocomplete("the cat", 3);
        assert!(!preds.is_empty(), "should produce predictions");
        println!("Autocomplete 'the cat': {:?}",
            preds.iter().map(|p| &p.token).collect::<Vec<_>>());
    }

    #[test]
    fn test_engine_generate() {
        let corpus = tiny_corpus();
        let engine = WaveTextEngine::from_corpus(&corpus);

        let result = engine.generate("the cat", 5);
        println!("Generated: '{}'", result.full_text);
        assert!(!result.generated.is_empty(), "should generate tokens");
    }

    #[test]
    fn test_engine_serialization() {
        let corpus = tiny_corpus();
        let engine = WaveTextEngine::from_corpus(&corpus);

        let bytes = engine.to_bytes();
        assert!(!bytes.is_empty());

        let restored = WaveTextEngine::from_bytes(&bytes).unwrap();
        assert_eq!(restored.vocab_size(), engine.vocab_size());
        assert_eq!(restored.config.dim, engine.config.dim);
    }

    #[test]
    fn test_engine_model_size() {
        let corpus = tiny_corpus();
        let config = EngineConfig {
            dim: 64,
            max_vocab: 2000,
            ..EngineConfig::default()
        };
        let engine = WaveTextEngine::from_corpus_with_config(&corpus, &config);

        let size = engine.model_size_bytes();
        println!("Model size: {} bytes ({:.1} KB)", size, size as f64 / 1024.0);
        assert!(size < 2_000_000, "model should be < 2MB: {} bytes", size);
    }
}
