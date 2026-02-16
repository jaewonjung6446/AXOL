//! Chat Engine — Resonance-based dialogue with meta-growth + self-learning.
//!
//! Five-level growth system that evolves with usage:
//!   Level 1: Fitness-weighted selection + decay + pruning
//!   Level 2: Co-resonance → hybrid pool emergence
//!   Level 3: High-fitness response cloning with feature perturbation
//!   Level 4: Self-learning — experience accumulation → classifier retraining
//!   Level 5: Generation evolution — gen_phi tracking → seed selection → fitness boost
//!
//! Pipeline per query (vector-based intent — continuous weighting):
//!   1. Input → wave features + classifier → intent probability vector
//!   2. Score = cosine_sim(input, response) * sqrt(fitness) * exp(W * intent_prob)
//!   3. All pools searched with continuous weighting (no discrete gate)
//!   4. Every N queries → growth_cycle (decay → prune → emerge → replicate)
//!   5. Generation: seed by cosine * sqrt(fitness) * (1+gen_phi) * exp(W*intent_prob)

use super::engine::WaveTextEngine;
use super::reservoir::ReservoirState;
use super::data::{DialogueIntent, chat_classification_data};
use super::tokenizer::EOS_ID;

// ---------------------------------------------------------------------------
// GrowthConfig
// ---------------------------------------------------------------------------

/// Meta-growth configuration parameters.
pub struct GrowthConfig {
    // Level 1: Fitness
    pub initial_fitness: f64,
    pub positive_boost: f64,
    pub negative_penalty: f64,
    pub decay_factor: f64,
    pub prune_threshold: f64,
    pub min_pool_size: usize,
    // Level 2: Emergence
    pub emergence_threshold: f64,
    pub emergence_top_k: usize,
    pub max_emergent_pools: usize,
    // Level 3: Replication
    pub replication_threshold: f64,
    pub mutation_magnitude: f64,
    pub max_clones_per_response: usize,
    pub max_pool_size: usize,
    // Intent vector: exponential weighting from classifier probability distribution.
    // score = cosine * sqrt(fitness) * exp(intent_weight * intent_prob)
    pub intent_weight: f64,
    // Cycle
    pub cycle_interval: usize,
    pub seed: u64,
}

impl Default for GrowthConfig {
    fn default() -> Self {
        Self {
            initial_fitness: 1.0,
            positive_boost: 0.15,
            negative_penalty: 0.10,
            decay_factor: 0.995,
            prune_threshold: 0.2,
            min_pool_size: 3,
            emergence_threshold: 0.7,
            emergence_top_k: 3,
            max_emergent_pools: 5,
            replication_threshold: 1.5,
            mutation_magnitude: 0.05,
            max_clones_per_response: 2,
            max_pool_size: 30,
            intent_weight: 20.0,
            cycle_interval: 10,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// ResponseEntry — individual response with fitness tracking
// ---------------------------------------------------------------------------

/// A single response entry with fitness metadata.
pub struct ResponseEntry {
    pub text: String,
    pub wave: ReservoirState,
    pub features: Vec<f64>,
    pub fitness: f64,
    pub select_count: usize,
    pub is_clone: bool,
    pub generation: usize,
    /// Average Φ quality when used as generation seed (Level 5).
    pub gen_phi: f64,
    /// Number of times used as generation seed.
    pub gen_uses: usize,
}

// ---------------------------------------------------------------------------
// ResponsePool — updated with emergent metadata
// ---------------------------------------------------------------------------

/// A pool of responses for a single intent category.
pub struct ResponsePool {
    pub intent: String,
    pub entries: Vec<ResponseEntry>,
    pub is_emergent: bool,
    pub parent_intents: Option<(String, String)>,
}

// ---------------------------------------------------------------------------
// CoResonanceMatrix
// ---------------------------------------------------------------------------

/// Tracks co-resonance between intent pools for emergence detection.
pub struct CoResonanceMatrix {
    pub labels: Vec<String>,
    pub matrix: Vec<Vec<f64>>,
}

impl CoResonanceMatrix {
    fn new(labels: Vec<String>) -> Self {
        let n = labels.len();
        let matrix = vec![vec![0.0; n]; n];
        Self { labels, matrix }
    }

    pub fn index_of(&self, label: &str) -> Option<usize> {
        self.labels.iter().position(|l| l == label)
    }

    fn add_label(&mut self, label: String) {
        let n = self.labels.len();
        self.labels.push(label);
        for row in &mut self.matrix {
            row.push(0.0);
        }
        self.matrix.push(vec![0.0; n + 1]);
    }

    /// Record co-resonance between two pools.
    fn record(&mut self, a: &str, b: &str, value: f64) {
        if let (Some(i), Some(j)) = (self.index_of(a), self.index_of(b)) {
            if i != j {
                self.matrix[i][j] += value;
                self.matrix[j][i] += value;
            }
        }
    }

    /// Find pairs above the emergence threshold.
    fn above_threshold(&self, threshold: f64) -> Vec<(String, String, f64)> {
        let n = self.labels.len();
        let mut pairs = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if self.matrix[i][j] >= threshold {
                    pairs.push((
                        self.labels[i].clone(),
                        self.labels[j].clone(),
                        self.matrix[i][j],
                    ));
                }
            }
        }
        pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        pairs
    }
}

// ---------------------------------------------------------------------------
// GrowthStats
// ---------------------------------------------------------------------------

/// Statistics tracking for the growth system.
pub struct GrowthStats {
    pub total_queries: usize,
    pub total_cycles: usize,
    pub positive_feedbacks: usize,
    pub negative_feedbacks: usize,
    pub total_pruned: usize,
    pub emergent_pools_created: usize,
    pub total_clones: usize,
    pub classifier_retrains: usize,
    /// Number of gen_phi fitness boosts applied to seed entries (Level 5).
    pub gen_boosts: usize,
}

impl GrowthStats {
    fn new() -> Self {
        Self {
            total_queries: 0,
            total_cycles: 0,
            positive_feedbacks: 0,
            negative_feedbacks: 0,
            total_pruned: 0,
            emergent_pools_created: 0,
            total_clones: 0,
            classifier_retrains: 0,
            gen_boosts: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// ChatResponse
// ---------------------------------------------------------------------------

/// Result of a chat engine query.
pub struct ChatResponse {
    pub response: String,
    pub intent: String,
    pub confidence: f64,
    pub resonance: f64,
    pub pool_id: usize,
    pub response_id: usize,
    /// Whether the response was wave-generated (vs pool-selected).
    pub is_generated: bool,
    /// Average Φ quality of generated tokens (None for pool responses).
    pub generation_quality: Option<f64>,
}

// ---------------------------------------------------------------------------
// Cosine similarity
// ---------------------------------------------------------------------------

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for i in 0..len {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom > 1e-10 { dot / denom } else { 0.0 }
}

// ---------------------------------------------------------------------------
// GenerationConfig — wave-based text generation settings
// ---------------------------------------------------------------------------

/// Seed strategy for wave-based generation.
pub enum SeedMode {
    /// Use the highest-fitness pool response as seed.
    PoolBest,
    /// Generate from input only (no seed).
    InputOnly,
}

/// Configuration for wave-based text generation.
pub struct GenerationConfig {
    /// Maximum tokens to generate (default: 20).
    pub max_tokens: usize,
    /// Minimum Φ quality gate (default: 0.01).
    pub min_phi: f64,
    /// Seed strategy (default: PoolBest).
    pub seed_mode: SeedMode,
    /// Level 5: gen_phi-proportional fitness boost factor (default: 0.5).
    /// On positive feedback for a generated response, the seed entry gets
    /// `fitness += gen_phi * gen_fitness_boost`. Good seeds survive longer.
    pub gen_fitness_boost: f64,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 20,
            min_phi: 0.01,
            seed_mode: SeedMode::PoolBest,
            gen_fitness_boost: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// SelfLearningConfig — Level 4: experience-driven classifier retraining
// ---------------------------------------------------------------------------

/// Configuration for self-learning (Level 4 growth).
pub struct SelfLearningConfig {
    /// Enable self-learning (default: true).
    pub enabled: bool,
    /// Retrain classifier every N accumulated experiences (default: 50).
    pub retrain_interval: usize,
}

impl Default for SelfLearningConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            retrain_interval: 50,
        }
    }
}

// ---------------------------------------------------------------------------
// ChatEngine
// ---------------------------------------------------------------------------

/// Resonance-based chat engine with 5-level meta-growth + generation evolution.
pub struct ChatEngine {
    pub engine: WaveTextEngine,
    pub pools: Vec<ResponsePool>,
    pub growth: GrowthConfig,
    pub generation: GenerationConfig,
    pub self_learning: SelfLearningConfig,
    pub co_resonance: CoResonanceMatrix,
    pub stats: GrowthStats,
    // Self-learning state
    last_input: Option<String>,
    last_class_id: Option<usize>,
    experience: Vec<(String, usize)>,
    retrain_count: usize,
    // Level 5: generation evolution state
    last_gen_text: Option<String>,
    last_gen_phi: f64,
    last_gen_pool_id: Option<usize>,
    // Auto-feedback: system self-judges responses by confidence
    pub auto_feedback: bool,
    pub auto_feedback_threshold: f64,
    // Internal
    query_count: usize,
    mutation_counter: u64,
}

impl ChatEngine {
    /// Build a ChatEngine with default growth config.
    pub fn new(engine: WaveTextEngine, intents: Vec<DialogueIntent>) -> Self {
        Self::with_growth(engine, intents, GrowthConfig::default())
    }

    /// Build a ChatEngine with custom growth config.
    pub fn with_growth(
        engine: WaveTextEngine,
        intents: Vec<DialogueIntent>,
        config: GrowthConfig,
    ) -> Self {
        let initial_fitness = config.initial_fitness;
        let mut pool_map: std::collections::BTreeMap<String, ResponsePool> =
            std::collections::BTreeMap::new();

        for intent in intents {
            let entry = pool_map.entry(intent.intent.clone()).or_insert_with(|| {
                ResponsePool {
                    intent: intent.intent.clone(),
                    entries: Vec::new(),
                    is_emergent: false,
                    parent_intents: None,
                }
            });
            for resp_str in &intent.responses {
                let s = resp_str.to_string();
                let state = engine.process_text(&s);
                let feat = state.to_feature_vector();
                entry.entries.push(ResponseEntry {
                    text: s,
                    wave: state,
                    features: feat,
                    fitness: initial_fitness,
                    select_count: 0,
                    is_clone: false,
                    generation: 0,
                    gen_phi: 0.0,
                    gen_uses: 0,
                });
            }
        }

        let pools: Vec<ResponsePool> = pool_map.into_values().collect();
        let labels: Vec<String> = pools.iter().map(|p| p.intent.clone()).collect();
        let co_resonance = CoResonanceMatrix::new(labels);

        Self {
            engine,
            pools,
            growth: config,
            generation: GenerationConfig::default(),
            self_learning: SelfLearningConfig::default(),
            co_resonance,
            stats: GrowthStats::new(),
            last_input: None,
            last_class_id: None,
            experience: Vec::new(),
            retrain_count: 0,
            last_gen_text: None,
            last_gen_phi: 0.0,
            last_gen_pool_id: None,
            auto_feedback: false,
            auto_feedback_threshold: 0.15,
            query_count: 0,
            mutation_counter: 0,
        }
    }

    /// Respond to user input — vector-weighted search across all pools.
    ///
    /// Intent is a continuous probability vector from the classifier.
    /// Score = cosine_sim(input, response) * sqrt(fitness) * exp(W * intent_prob).
    /// No discrete gate — all pools are searched with continuous weighting.
    pub fn respond(&mut self, input: &str) -> ChatResponse {
        let input_state = self.engine.process_text(input);
        let input_features = input_state.to_feature_vector();

        self.stats.total_queries += 1;
        self.query_count += 1;

        // Intent as continuous vector: classifier probability distribution
        let intent_probs: Vec<(String, f64)> = if let Some(result) = self.engine.classify(input) {
            result.class_probs.clone()
        } else {
            Vec::new()
        };

        let top_confidence = intent_probs.first().map(|(_, p)| *p).unwrap_or(0.0);
        let intent_weight = self.growth.intent_weight;

        // Single-pass weighted search across ALL pools
        let mut best_response = String::new();
        let mut best_intent = String::from("unknown");
        let mut best_score: f64 = f64::NEG_INFINITY;
        let mut best_pool_id: usize = 0;
        let mut best_resp_id: usize = 0;

        let mut pool_max_scores: Vec<(usize, f64)> = Vec::new();

        for (pool_idx, pool) in self.pools.iter().enumerate() {
            // Continuous intent probability for this pool
            let intent_prob = if pool.is_emergent {
                // Emergent pools: max of parent intent probabilities
                pool.parent_intents.as_ref().map_or(0.0, |(pa, pb)| {
                    let p_a = intent_probs.iter().find(|(l, _)| l == pa).map(|(_, p)| *p).unwrap_or(0.0);
                    let p_b = intent_probs.iter().find(|(l, _)| l == pb).map(|(_, p)| *p).unwrap_or(0.0);
                    p_a.max(p_b)
                })
            } else {
                intent_probs.iter()
                    .find(|(l, _)| l == &pool.intent)
                    .map(|(_, p)| *p)
                    .unwrap_or(0.0)
            };

            let mut pool_max: f64 = f64::NEG_INFINITY;

            for (entry_idx, entry) in pool.entries.iter().enumerate() {
                let cosine = cosine_similarity(&input_features, &entry.features);
                let score = cosine * entry.fitness.sqrt() * (intent_weight * intent_prob).exp();

                if cosine > pool_max { pool_max = cosine; }

                if score > best_score {
                    best_score = score;
                    best_response = entry.text.clone();
                    best_intent = pool.intent.clone();
                    best_pool_id = pool_idx;
                    best_resp_id = entry_idx;
                }
            }
            if pool_max > f64::NEG_INFINITY {
                pool_max_scores.push((pool_idx, pool_max));
            }
        }

        // Update select_count
        if best_pool_id < self.pools.len()
            && best_resp_id < self.pools[best_pool_id].entries.len()
        {
            self.pools[best_pool_id].entries[best_resp_id].select_count += 1;
        }

        // Co-resonance update: pairs with both scores >= 0.3
        let co_threshold = 0.3;
        for i in 0..pool_max_scores.len() {
            for j in (i + 1)..pool_max_scores.len() {
                let (idx_a, score_a) = pool_max_scores[i];
                let (idx_b, score_b) = pool_max_scores[j];
                if score_a >= co_threshold && score_b >= co_threshold {
                    let intent_a = self.pools[idx_a].intent.clone();
                    let intent_b = self.pools[idx_b].intent.clone();
                    let value = score_a.min(score_b);
                    self.co_resonance.record(&intent_a, &intent_b, value);
                }
            }
        }

        // Auto growth cycle
        if self.query_count % self.growth.cycle_interval == 0 {
            self.growth_cycle();
        }

        // Self-learning state
        self.last_input = Some(input.to_string());
        self.last_class_id = self.engine.classifier.as_ref()
            .and_then(|c| c.class_labels.iter().position(|l| l == &best_intent));

        // Auto-feedback: high-confidence responses get automatic positive feedback
        if self.auto_feedback && top_confidence >= self.auto_feedback_threshold {
            if best_pool_id < self.pools.len()
                && best_resp_id < self.pools[best_pool_id].entries.len()
            {
                self.pools[best_pool_id].entries[best_resp_id].fitness += self.growth.positive_boost;
                self.stats.positive_feedbacks += 1;

                // Also accumulate self-learning experience
                if self.self_learning.enabled {
                    if let (Some(ref inp), Some(cid)) = (&self.last_input, self.last_class_id) {
                        self.experience.push((inp.clone(), cid));
                        self.maybe_retrain();
                    }
                }
            }
        }

        ChatResponse {
            response: best_response,
            intent: best_intent,
            confidence: top_confidence,
            resonance: best_score,
            pool_id: best_pool_id,
            response_id: best_resp_id,
            is_generated: false,
            generation_quality: None,
        }
    }

    /// Provide feedback on a response. Level 1 direct signal.
    /// On positive feedback, also accumulates experience for self-learning (Level 4).
    pub fn feedback(&mut self, pool_id: usize, response_id: usize, positive: bool) {
        if pool_id >= self.pools.len() {
            return;
        }
        let pool = &mut self.pools[pool_id];
        if response_id >= pool.entries.len() {
            return;
        }

        if positive {
            pool.entries[response_id].fitness += self.growth.positive_boost;
            self.stats.positive_feedbacks += 1;

            // Level 4: accumulate confirmed-correct experience
            if self.self_learning.enabled {
                if let (Some(ref input), Some(class_id)) = (&self.last_input, self.last_class_id) {
                    self.experience.push((input.clone(), class_id));
                    self.maybe_retrain();
                }
            }

            // Level 5: gen_phi fitness boost — good seeds survive longer
            let _gen_text = self.last_gen_text.take();
            let gen_phi = self.last_gen_phi;
            let gen_pool_id = self.last_gen_pool_id.take();
            if let Some(pid) = gen_pool_id {
                if gen_phi > 0.0 && pid < self.pools.len()
                    && response_id < self.pools[pid].entries.len()
                {
                    let boost = gen_phi * self.generation.gen_fitness_boost;
                    self.pools[pid].entries[response_id].fitness += boost;
                    self.stats.gen_boosts += 1;
                }
            }
        } else {
            pool.entries[response_id].fitness =
                (pool.entries[response_id].fitness - self.growth.negative_penalty).max(0.0);
            self.stats.negative_feedbacks += 1;
        }
    }

    /// Level 5: Positive feedback on a generated response → inject it into the pool.
    ///
    /// The generated text becomes a new pool entry that can serve as a seed
    /// for future generation, creating a self-reinforcing evolution loop.
    pub fn feedback_generated(&mut self, pool_id: usize, generated_text: &str, gen_phi: f64) {
        if pool_id >= self.pools.len() {
            return;
        }
        let pool = &mut self.pools[pool_id];
        if pool.entries.len() >= self.growth.max_pool_size {
            return;
        }

        let state = self.engine.process_text(generated_text);
        let features = state.to_feature_vector();
        pool.entries.push(ResponseEntry {
            text: generated_text.to_string(),
            wave: state,
            features,
            fitness: self.growth.initial_fitness,
            select_count: 0,
            is_clone: false,
            generation: 0,
            gen_phi,
            gen_uses: 1,
        });
        self.stats.gen_boosts += 1;
    }

    /// Provide correct intent when the system was wrong (Level 4 self-learning).
    ///
    /// Call after a negative `feedback()` to tell the system what the correct
    /// intent was. This is the key signal that breaks the classifier ceiling.
    pub fn feedback_correct(&mut self, correct_intent: &str) {
        if !self.self_learning.enabled { return; }
        if let Some(ref input) = self.last_input {
            let class_id = self.engine.classifier.as_ref()
                .and_then(|c| c.class_labels.iter().position(|l| l == correct_intent));
            if let Some(cid) = class_id {
                self.experience.push((input.clone(), cid));
                self.maybe_retrain();
            }
        }
    }

    /// Check if experience buffer is large enough to trigger retraining.
    fn maybe_retrain(&mut self) {
        let interval = self.self_learning.retrain_interval;
        if interval > 0 && self.experience.len() % interval == 0 {
            self.retrain_from_experience();
        }
    }

    /// Level 4: Retrain the classifier using original data + accumulated experience.
    pub fn retrain_from_experience(&mut self) {
        let class_labels = match &self.engine.classifier {
            Some(clf) => clf.class_labels.clone(),
            None => return,
        };

        // Original training data
        let (original_labeled, _) = chat_classification_data();

        // Combine with experience
        let exp_refs: Vec<(&str, usize)> = self.experience.iter()
            .map(|(s, id)| (s.as_str(), *id))
            .collect();

        let mut all: Vec<(&str, usize)> = original_labeled;
        all.extend(exp_refs);

        // Retrain classifier
        self.engine.train_classifier(&all, class_labels);
        self.retrain_count += 1;
        self.stats.classifier_retrains += 1;
    }

    /// Number of classifier retrains that have occurred.
    pub fn retrain_count(&self) -> usize {
        self.retrain_count
    }

    /// Number of accumulated experiences.
    pub fn experience_count(&self) -> usize {
        self.experience.len()
    }

    /// Auto-feedback cycle: self-evaluate all pool entries + growth cycle.
    ///
    /// For each pool entry, classify the text. If the classifier agrees
    /// with the pool's intent (confidence >= threshold) → boost fitness.
    /// If it disagrees → penalize. Then run a growth cycle (decay/prune/emerge/replicate).
    /// Returns (boosted, penalized) counts.
    pub fn auto_feedback_cycle(&mut self) -> (usize, usize) {
        let threshold = self.auto_feedback_threshold;
        let boost = self.growth.positive_boost;
        let penalty = self.growth.negative_penalty;
        let mut boosted = 0usize;
        let mut penalized = 0usize;

        // Collect (pool_idx, entry_idx, positive) to avoid borrow issues
        let mut actions: Vec<(usize, usize, bool)> = Vec::new();

        for (pool_idx, pool) in self.pools.iter().enumerate() {
            for (entry_idx, entry) in pool.entries.iter().enumerate() {
                if let Some(result) = self.engine.classify(&entry.text) {
                    let top_intent = &result.class_probs[0].0;
                    let top_conf = result.class_probs[0].1;

                    if top_conf >= threshold {
                        // Classifier is confident — check if it agrees
                        if top_intent == &pool.intent {
                            actions.push((pool_idx, entry_idx, true));
                        } else if !pool.is_emergent {
                            // Disagreement on non-emergent pool → penalize
                            actions.push((pool_idx, entry_idx, false));
                        }
                        // Emergent pools: skip penalize (multi-intent by nature)
                    }
                }
            }
        }

        for (pid, eid, positive) in actions {
            if positive {
                self.pools[pid].entries[eid].fitness += boost;
                boosted += 1;
                self.stats.positive_feedbacks += 1;
            } else {
                self.pools[pid].entries[eid].fitness =
                    (self.pools[pid].entries[eid].fitness - penalty).max(0.0);
                penalized += 1;
                self.stats.negative_feedbacks += 1;
            }
        }

        // Growth cycle after evaluation
        self.growth_cycle();

        (boosted, penalized)
    }

    /// Run N auto-feedback cycles. Returns total (boosted, penalized).
    pub fn auto_feedback_cycles(&mut self, n: usize) -> (usize, usize) {
        let mut total_boosted = 0;
        let mut total_penalized = 0;
        for _ in 0..n {
            let (b, p) = self.auto_feedback_cycle();
            total_boosted += b;
            total_penalized += p;
        }
        (total_boosted, total_penalized)
    }

    /// Execute one growth cycle: decay → prune → emerge → replicate.
    pub fn growth_cycle(&mut self) {
        self.stats.total_cycles += 1;
        self.decay_fitness();
        self.prune_low_fitness();
        self.check_emergence();
        self.replicate_high_fitness();
    }

    /// Stabilization: run N growth cycles with self-learning disabled.
    ///
    /// After active learning (queries + feedback), call this to let pools
    /// equilibrate — strong entries survive decay, weak ones get pruned,
    /// clones compete — without any new experience accumulation or
    /// classifier retraining.
    pub fn stabilize(&mut self, cycles: usize) {
        let was_enabled = self.self_learning.enabled;
        self.self_learning.enabled = false;
        for _ in 0..cycles {
            self.growth_cycle();
        }
        self.self_learning.enabled = was_enabled;
    }

    /// Level 1: decay all fitness values.
    fn decay_fitness(&mut self) {
        for pool in &mut self.pools {
            for entry in &mut pool.entries {
                entry.fitness *= self.growth.decay_factor;
            }
        }
    }

    /// Level 1: prune entries with fitness below threshold (keep min_pool_size).
    fn prune_low_fitness(&mut self) {
        let threshold = self.growth.prune_threshold;
        let min_size = self.growth.min_pool_size;

        for pool in &mut self.pools {
            if pool.entries.len() <= min_size {
                continue;
            }
            let before = pool.entries.len();
            // Sort by fitness descending to keep the best
            pool.entries.sort_by(|a, b| {
                b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal)
            });
            // Keep at least min_size entries, remove those below threshold
            let keep = pool.entries.len().max(min_size);
            pool.entries.truncate(keep);
            // Now remove below-threshold entries beyond min_size
            while pool.entries.len() > min_size {
                if let Some(last) = pool.entries.last() {
                    if last.fitness < threshold {
                        pool.entries.pop();
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            let pruned = before - pool.entries.len();
            self.stats.total_pruned += pruned;
        }
    }

    /// Level 2: check co-resonance matrix for emergence opportunities.
    fn check_emergence(&mut self) {
        let current_emergent = self.pools.iter().filter(|p| p.is_emergent).count();
        if current_emergent >= self.growth.max_emergent_pools {
            return;
        }

        let pairs = self.co_resonance.above_threshold(self.growth.emergence_threshold);

        for (intent_a, intent_b, _score) in pairs {
            if current_emergent + self.pools.iter().filter(|p| p.is_emergent).count()
                >= self.growth.max_emergent_pools
            {
                break;
            }

            let hybrid_name = format!("{}_{}", intent_a, intent_b);

            // Skip if already exists
            if self.pools.iter().any(|p| p.intent == hybrid_name) {
                continue;
            }

            // Collect top-k entries from each parent
            let top_k = self.growth.emergence_top_k;
            let mut hybrid_entries: Vec<ResponseEntry> = Vec::new();

            for parent_intent in [&intent_a, &intent_b] {
                if let Some(parent) = self.pools.iter().find(|p| p.intent == *parent_intent) {
                    let mut sorted_indices: Vec<usize> = (0..parent.entries.len()).collect();
                    sorted_indices.sort_by(|&a, &b| {
                        parent.entries[b].fitness.partial_cmp(&parent.entries[a].fitness)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });

                    for &idx in sorted_indices.iter().take(top_k) {
                        let src = &parent.entries[idx];
                        hybrid_entries.push(ResponseEntry {
                            text: src.text.clone(),
                            wave: src.wave.clone(),
                            features: src.features.clone(),
                            fitness: self.growth.initial_fitness,
                            select_count: 0,
                            is_clone: false,
                            generation: 0,
                            gen_phi: src.gen_phi,
                            gen_uses: 0,
                        });
                    }
                }
            }

            if !hybrid_entries.is_empty() {
                // Register in co-resonance matrix
                self.co_resonance.add_label(hybrid_name.clone());

                self.pools.push(ResponsePool {
                    intent: hybrid_name,
                    entries: hybrid_entries,
                    is_emergent: true,
                    parent_intents: Some((intent_a.clone(), intent_b.clone())),
                });
                self.stats.emergent_pools_created += 1;
            }
        }
    }

    /// Level 3: replicate high-fitness responses with mutation.
    fn replicate_high_fitness(&mut self) {
        let threshold = self.growth.replication_threshold;
        let max_clones = self.growth.max_clones_per_response;
        let max_pool_size = self.growth.max_pool_size;

        for pool_idx in 0..self.pools.len() {
            if self.pools[pool_idx].entries.len() >= max_pool_size {
                continue;
            }

            let mut clones_to_add: Vec<ResponseEntry> = Vec::new();

            // Collect clone candidates first (to satisfy borrow checker)
            let mut candidates: Vec<(String, ReservoirState, Vec<f64>, usize, f64)> = Vec::new();
            for entry in &self.pools[pool_idx].entries {
                if entry.fitness >= threshold {
                    candidates.push((
                        entry.text.clone(),
                        entry.wave.clone(),
                        entry.features.clone(),
                        entry.generation,
                        entry.gen_phi,
                    ));
                }
            }

            for (text, wave, features, generation, parent_gen_phi) in candidates {
                let num_clones = max_clones.min(
                    max_pool_size - self.pools[pool_idx].entries.len() - clones_to_add.len()
                );
                if num_clones == 0 {
                    break;
                }

                for _ in 0..num_clones {
                    let mutated_features = self.perturb_features(&features);
                    clones_to_add.push(ResponseEntry {
                        text: text.clone(),
                        wave: wave.clone(),
                        features: mutated_features,
                        fitness: self.growth.initial_fitness,
                        select_count: 0,
                        is_clone: true,
                        generation: generation + 1,
                        gen_phi: parent_gen_phi,
                        gen_uses: 0,
                    });
                    self.stats.total_clones += 1;
                }
            }

            self.pools[pool_idx].entries.extend(clones_to_add);
        }
    }

    /// Perturb feature vector using deterministic ChaCha8Rng.
    fn perturb_features(&mut self, features: &[f64]) -> Vec<f64> {
        use rand::SeedableRng;
        use rand::Rng;
        use rand_chacha::ChaCha8Rng;

        self.mutation_counter += 1;
        let derived_seed = self.growth.seed ^ self.mutation_counter.wrapping_mul(2654435761);
        let mut rng = ChaCha8Rng::seed_from_u64(derived_seed);

        features.iter()
            .map(|&v| v + rng.gen_range(-1.0..1.0) * self.growth.mutation_magnitude)
            .collect()
    }

    /// Get growth statistics.
    pub fn growth_stats(&self) -> &GrowthStats {
        &self.stats
    }

    /// Pool summary: (intent, size, avg_fitness, is_emergent).
    pub fn pool_summary(&self) -> Vec<(String, usize, f64, bool)> {
        self.pools.iter().map(|pool| {
            let avg_fitness = if pool.entries.is_empty() {
                0.0
            } else {
                pool.entries.iter().map(|e| e.fitness).sum::<f64>() / pool.entries.len() as f64
            };
            (pool.intent.clone(), pool.entries.len(), avg_fitness, pool.is_emergent)
        }).collect()
    }

    // =====================================================================
    // Wave-based generation
    // =====================================================================

    /// Vector-based seed selection: best entry from each pool, return top-K.
    ///
    /// Score: cosine_sim(input, entry) * sqrt(fitness) * (1 + gen_phi) * (1 + W * intent_prob).
    /// Returns one seed per pool, sorted by score descending, up to `k` entries.
    fn top_k_seeds(&self, input: &str, input_features: &[f64], k: usize) -> Vec<(String, usize, usize)> {
        // Intent probability vector for weighting
        let intent_probs: Vec<(String, f64)> = self.engine.classify(input)
            .map(|r| r.class_probs.clone())
            .unwrap_or_default();
        let intent_weight = self.growth.intent_weight;

        let mut pool_bests: Vec<(f64, String, usize, usize)> = Vec::new();

        for (pool_idx, pool) in self.pools.iter().enumerate() {
            let intent_prob = if pool.is_emergent {
                pool.parent_intents.as_ref().map_or(0.0, |(pa, pb)| {
                    let p_a = intent_probs.iter().find(|(l, _)| l == pa).map(|(_, p)| *p).unwrap_or(0.0);
                    let p_b = intent_probs.iter().find(|(l, _)| l == pb).map(|(_, p)| *p).unwrap_or(0.0);
                    p_a.max(p_b)
                })
            } else {
                intent_probs.iter()
                    .find(|(l, _)| l == &pool.intent)
                    .map(|(_, p)| *p)
                    .unwrap_or(0.0)
            };

            let mut best_score = f64::NEG_INFINITY;
            let mut best_text = String::new();
            let mut best_entry_idx = 0;

            for (entry_idx, entry) in pool.entries.iter().enumerate() {
                let cosine = cosine_similarity(input_features, &entry.features);
                let score = cosine * entry.fitness.sqrt()
                    * (1.0 + entry.gen_phi)
                    * (intent_weight * intent_prob).exp();
                if score > best_score {
                    best_score = score;
                    best_text = entry.text.clone();
                    best_entry_idx = entry_idx;
                }
            }

            if best_score > f64::NEG_INFINITY {
                pool_bests.push((best_score, best_text, pool_idx, best_entry_idx));
            }
        }

        pool_bests.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        pool_bests.into_iter()
            .take(k)
            .map(|(_, text, pid, rid)| (text, pid, rid))
            .collect()
    }

    /// Generate a response using wave-based text generation.
    ///
    /// Pipeline (vector-based):
    ///   1. Encode input → wave features + BPE ids
    ///   2. Select top-K seeds by vector similarity across all pools
    ///   3. Condition reservoir: input + seed tokens
    ///   4. Autoregressive generation via quantum measurement
    ///   5. Pick best Φ across seeds
    ///   6. Fallback to seed response if generation fails
    pub fn generate_response(&mut self, input: &str) -> ChatResponse {
        let input_ids = self.engine.bpe.encode(input);
        let input_state = self.engine.process_text(input);
        let input_features = input_state.to_feature_vector();

        // Vector-based seed selection: top-3 from different pools
        let seed_candidates: Vec<(String, usize, usize)> = match self.generation.seed_mode {
            SeedMode::PoolBest => {
                let seeds = self.top_k_seeds(input, &input_features, 3);
                if seeds.is_empty() {
                    vec![(String::new(), 0, 0)]
                } else {
                    seeds
                }
            }
            SeedMode::InputOnly => vec![(String::new(), 0, 0)],
        };

        let primary_intent = seed_candidates.first()
            .and_then(|(_, pid, _)| self.pools.get(*pid))
            .map(|p| p.intent.clone())
            .unwrap_or_else(|| "unknown".to_string());

        let confidence = seed_candidates.first()
            .map(|(_, pid, rid)| {
                if *pid < self.pools.len() && *rid < self.pools[*pid].entries.len() {
                    cosine_similarity(&input_features, &self.pools[*pid].entries[*rid].features)
                } else { 0.0 }
            })
            .unwrap_or(0.0);

        // Generate from each seed, pick best Φ
        let gen = match &self.engine.generator {
            Some(g) => g.clone(),
            None => {
                let (seed_text, pool_id, response_id) = seed_candidates.into_iter().next()
                    .unwrap_or((String::new(), 0, 0));
                return ChatResponse {
                    response: seed_text,
                    intent: primary_intent,
                    confidence,
                    resonance: 0.0,
                    pool_id,
                    response_id,
                    is_generated: false,
                    generation_quality: None,
                };
            }
        };

        let mut best_response = String::new();
        let mut best_phi: f64 = -1.0;
        let mut best_pool_id = 0;
        let mut best_resp_id = 0;
        let mut best_is_generated = false;
        let mut best_seed_text = String::new();

        for (seed_text, pool_id, response_id) in &seed_candidates {
            let mut combined_ids: Vec<usize> = input_ids.iter()
                .copied()
                .filter(|&id| id != EOS_ID)
                .collect();

            if !seed_text.is_empty() {
                let seed_ids = self.engine.bpe.encode(seed_text);
                combined_ids.extend(seed_ids.into_iter().filter(|&id| id != EOS_ID));
            }

            let mut reservoir = self.engine.reservoir.clone();
            let result = gen.generate(
                &combined_ids,
                &self.engine.sps,
                &mut reservoir,
                &self.engine.vocab,
                Some(self.generation.max_tokens),
            );

            let avg_phi = result.avg_phi();
            let is_generated = !result.generated.is_empty();

            if avg_phi > best_phi || (best_phi < 0.0 && is_generated) {
                best_phi = avg_phi;
                best_pool_id = *pool_id;
                best_resp_id = *response_id;
                best_is_generated = is_generated;
                best_seed_text = seed_text.clone();

                if is_generated {
                    let generated_text = result.generated.join(" ");
                    best_response = if seed_text.is_empty() {
                        generated_text
                    } else {
                        format!("{} {}", seed_text, generated_text)
                    };
                } else {
                    best_response = seed_text.clone();
                }
            }
        }

        // Level 5: record gen_phi on the winning seed entry
        if best_is_generated && best_pool_id < self.pools.len()
            && best_resp_id < self.pools[best_pool_id].entries.len()
        {
            let entry = &mut self.pools[best_pool_id].entries[best_resp_id];
            entry.gen_uses += 1;
            let n = entry.gen_uses as f64;
            entry.gen_phi += (best_phi - entry.gen_phi) / n;
        }

        // Fallback
        if best_response.is_empty() {
            best_response = best_seed_text;
        }

        let is_generated = best_is_generated;
        let avg_phi = if best_phi > 0.0 { best_phi } else { 0.0 };

        // Level 5: store generated state
        if is_generated {
            self.last_gen_text = Some(best_response.clone());
            self.last_gen_phi = avg_phi;
            self.last_gen_pool_id = Some(best_pool_id);
        } else {
            self.last_gen_text = None;
            self.last_gen_pool_id = None;
        }

        ChatResponse {
            response: best_response,
            intent: primary_intent,
            confidence,
            resonance: 0.0,
            pool_id: best_pool_id,
            response_id: best_resp_id,
            is_generated,
            generation_quality: if is_generated { Some(avg_phi) } else { None },
        }
    }

    /// Auto-select between pool response and wave-generated response.
    ///
    /// Tries both methods and picks the generated response if its
    /// quality meets the threshold; otherwise falls back to pool response.
    pub fn respond_auto(&mut self, input: &str) -> ChatResponse {
        let pool_response = self.respond(input);
        let gen_response = self.generate_response(input);

        if gen_response.is_generated {
            if let Some(quality) = gen_response.generation_quality {
                if quality >= self.generation.min_phi {
                    return gen_response;
                }
            }
        }

        pool_response
    }

    // =================================================================
    // Save / Load — persist learned state to JSON
    // =================================================================

    /// Save the learned state to a JSON file.
    ///
    /// Persists: pools (fitness, gen_phi, clones), co-resonance matrix,
    /// stats, self-learning experience, and internal counters.
    /// Wave vectors and features are NOT saved — they are recomputed on load.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let data = SaveData {
            pools: self.pools.iter().map(|p| SavePool {
                intent: p.intent.clone(),
                is_emergent: p.is_emergent,
                parent_intents: p.parent_intents.clone(),
                entries: p.entries.iter().map(|e| SaveEntry {
                    text: e.text.clone(),
                    fitness: e.fitness,
                    select_count: e.select_count,
                    is_clone: e.is_clone,
                    generation: e.generation,
                    gen_phi: e.gen_phi,
                    gen_uses: e.gen_uses,
                }).collect(),
            }).collect(),
            co_resonance_labels: self.co_resonance.labels.clone(),
            co_resonance_matrix: self.co_resonance.matrix.clone(),
            stats: SaveStats {
                total_queries: self.stats.total_queries,
                total_cycles: self.stats.total_cycles,
                positive_feedbacks: self.stats.positive_feedbacks,
                negative_feedbacks: self.stats.negative_feedbacks,
                total_pruned: self.stats.total_pruned,
                emergent_pools_created: self.stats.emergent_pools_created,
                total_clones: self.stats.total_clones,
                classifier_retrains: self.stats.classifier_retrains,
                gen_boosts: self.stats.gen_boosts,
            },
            experience: self.experience.iter()
                .map(|(s, id)| (s.clone(), *id))
                .collect(),
            retrain_count: self.retrain_count,
            query_count: self.query_count,
            mutation_counter: self.mutation_counter,
            auto_feedback: self.auto_feedback,
            auto_feedback_threshold: self.auto_feedback_threshold,
        };

        let json = serde_json::to_string_pretty(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    /// Load learned state from a JSON file.
    ///
    /// Replaces all pool data, stats, and experience.
    /// Recomputes wave vectors and features from saved text.
    /// If experience exists, retrains the classifier.
    pub fn load(&mut self, path: &str) -> std::io::Result<()> {
        let json = std::fs::read_to_string(path)?;
        let data: SaveData = serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        // Rebuild pools with recomputed wave/features
        self.pools = data.pools.into_iter().map(|sp| {
            let entries = sp.entries.into_iter().map(|se| {
                let state = self.engine.process_text(&se.text);
                let features = state.to_feature_vector();
                ResponseEntry {
                    text: se.text,
                    wave: state,
                    features,
                    fitness: se.fitness,
                    select_count: se.select_count,
                    is_clone: se.is_clone,
                    generation: se.generation,
                    gen_phi: se.gen_phi,
                    gen_uses: se.gen_uses,
                }
            }).collect();
            ResponsePool {
                intent: sp.intent,
                entries,
                is_emergent: sp.is_emergent,
                parent_intents: sp.parent_intents,
            }
        }).collect();

        // Restore co-resonance
        self.co_resonance = CoResonanceMatrix {
            labels: data.co_resonance_labels,
            matrix: data.co_resonance_matrix,
        };

        // Restore stats
        self.stats = GrowthStats {
            total_queries: data.stats.total_queries,
            total_cycles: data.stats.total_cycles,
            positive_feedbacks: data.stats.positive_feedbacks,
            negative_feedbacks: data.stats.negative_feedbacks,
            total_pruned: data.stats.total_pruned,
            emergent_pools_created: data.stats.emergent_pools_created,
            total_clones: data.stats.total_clones,
            classifier_retrains: data.stats.classifier_retrains,
            gen_boosts: data.stats.gen_boosts,
        };

        // Restore self-learning state
        self.experience = data.experience;
        self.retrain_count = data.retrain_count;
        self.query_count = data.query_count;
        self.mutation_counter = data.mutation_counter;

        // Restore auto-feedback settings
        self.auto_feedback = data.auto_feedback;
        self.auto_feedback_threshold = data.auto_feedback_threshold;

        // Retrain classifier if experience exists
        if !self.experience.is_empty() && self.self_learning.enabled {
            self.retrain_from_experience();
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Serializable types for save/load
// ---------------------------------------------------------------------------

use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct SaveData {
    pools: Vec<SavePool>,
    co_resonance_labels: Vec<String>,
    co_resonance_matrix: Vec<Vec<f64>>,
    stats: SaveStats,
    experience: Vec<(String, usize)>,
    retrain_count: usize,
    query_count: usize,
    mutation_counter: u64,
    #[serde(default)]
    auto_feedback: bool,
    #[serde(default = "default_auto_feedback_threshold")]
    auto_feedback_threshold: f64,
}

fn default_auto_feedback_threshold() -> f64 { 0.15 }

#[derive(Serialize, Deserialize)]
struct SavePool {
    intent: String,
    is_emergent: bool,
    parent_intents: Option<(String, String)>,
    entries: Vec<SaveEntry>,
}

#[derive(Serialize, Deserialize)]
struct SaveEntry {
    text: String,
    fitness: f64,
    select_count: usize,
    is_clone: bool,
    generation: usize,
    gen_phi: f64,
    gen_uses: usize,
}

#[derive(Serialize, Deserialize)]
struct SaveStats {
    total_queries: usize,
    total_cycles: usize,
    positive_feedbacks: usize,
    negative_feedbacks: usize,
    total_pruned: usize,
    emergent_pools_created: usize,
    total_clones: usize,
    classifier_retrains: usize,
    gen_boosts: usize,
}
