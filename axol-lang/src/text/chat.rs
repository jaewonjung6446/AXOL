//! Chat Engine — Resonance-based dialogue with meta-growth.
//!
//! Three-level growth system that evolves with usage:
//!   Level 1: Fitness-weighted selection + decay + pruning
//!   Level 2: Co-resonance → hybrid pool emergence
//!   Level 3: High-fitness response cloning with feature perturbation
//!
//! Pipeline per query:
//!   1. Classifier hard-gate → select intent pool(s)
//!   2. Score = cosine_sim(input, response) * sqrt(fitness)
//!   3. Update co-resonance matrix
//!   4. Every N queries → growth_cycle (decay → prune → emerge → replicate)

use super::engine::WaveTextEngine;
use super::reservoir::ReservoirState;
use super::data::DialogueIntent;

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
    // Gate
    pub soft_gate_threshold: f64,
    pub classifier_boost: f64,
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
            soft_gate_threshold: 0.3,
            classifier_boost: 4.0,
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
// ChatEngine
// ---------------------------------------------------------------------------

/// Resonance-based chat engine with 3-level meta-growth.
pub struct ChatEngine {
    pub engine: WaveTextEngine,
    pub pools: Vec<ResponsePool>,
    pub growth: GrowthConfig,
    pub co_resonance: CoResonanceMatrix,
    pub stats: GrowthStats,
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
            co_resonance,
            stats: GrowthStats::new(),
            query_count: 0,
            mutation_counter: 0,
        }
    }

    /// Respond to user input with fitness-weighted selection.
    pub fn respond(&mut self, input: &str) -> ChatResponse {
        let input_state = self.engine.process_text(input);
        let input_features = input_state.to_feature_vector();

        self.stats.total_queries += 1;
        self.query_count += 1;

        // Step 1: classifier hard-gate
        let (intent_filter, confidence) = if let Some(result) = self.engine.classify(input) {
            (Some(result.class_label.clone()), result.confidence)
        } else {
            (None, 0.0)
        };

        // Step 2: Two-pass selection
        //   Pass 1 (classifier gate): best response from classifier's pool(s)
        //   Pass 2 (global scan):     best response from ALL pools (fitness-weighted)
        //   If global score > classifier score * override_threshold → use global
        //   Otherwise → use classifier (safe default, preserves baseline accuracy)
        let classifier_boost = self.growth.classifier_boost;

        // Classifier-gated pool indices
        let gated_indices: Vec<usize> = if let Some(ref intent) = intent_filter {
            let mut idx: Vec<usize> = self.pools.iter().enumerate()
                .filter(|(_, p)| {
                    p.intent == *intent
                    || p.parent_intents.as_ref().map_or(false,
                        |(a, b)| a == intent || b == intent)
                })
                .map(|(i, _)| i)
                .collect();
            if idx.is_empty() { (0..self.pools.len()).collect() } else { idx.sort(); idx.dedup(); idx }
        } else {
            (0..self.pools.len()).collect()
        };

        // Pass 1: classifier-gated search
        let mut gate_response = String::new();
        let mut gate_intent = String::from("unknown");
        let mut gate_score: f64 = f64::NEG_INFINITY;
        let mut gate_pool_id: usize = 0;
        let mut gate_resp_id: usize = 0;

        for &pool_idx in &gated_indices {
            let pool = &self.pools[pool_idx];
            for (entry_idx, entry) in pool.entries.iter().enumerate() {
                let cosine = cosine_similarity(&input_features, &entry.features);
                let score = cosine * entry.fitness.sqrt();
                if score > gate_score {
                    gate_score = score;
                    gate_response = entry.text.clone();
                    gate_intent = pool.intent.clone();
                    gate_pool_id = pool_idx;
                    gate_resp_id = entry_idx;
                }
            }
        }

        // Pass 2: global scan (all pools, for growth override + co-resonance)
        let mut global_response = String::new();
        let mut global_intent = String::from("unknown");
        let mut global_score: f64 = f64::NEG_INFINITY;
        let mut global_pool_id: usize = 0;
        let mut global_resp_id: usize = 0;

        let mut pool_max_scores: Vec<(usize, f64)> = Vec::new();

        for pool_idx in 0..self.pools.len() {
            let pool = &self.pools[pool_idx];
            let mut pool_max: f64 = f64::NEG_INFINITY;

            for (entry_idx, entry) in pool.entries.iter().enumerate() {
                let cosine = cosine_similarity(&input_features, &entry.features);
                let score = cosine * entry.fitness.sqrt();

                if cosine > pool_max { pool_max = cosine; }

                if score > global_score {
                    global_score = score;
                    global_response = entry.text.clone();
                    global_intent = pool.intent.clone();
                    global_pool_id = pool_idx;
                    global_resp_id = entry_idx;
                }
            }
            if pool_max > f64::NEG_INFINITY {
                pool_max_scores.push((pool_idx, pool_max));
            }
        }

        // Decision: override classifier only if global is significantly better
        let override_threshold = classifier_boost;
        let (best_response, best_intent, best_score, best_pool_id, best_response_id) =
            if gate_score > f64::NEG_INFINITY
                && (global_score <= gate_score * override_threshold
                    || global_intent == gate_intent)
            {
                (gate_response, gate_intent, gate_score, gate_pool_id, gate_resp_id)
            } else {
                (global_response, global_intent, global_score, global_pool_id, global_resp_id)
            };

        // Update select_count
        if !self.pools.is_empty() && best_pool_id < self.pools.len() {
            if best_response_id < self.pools[best_pool_id].entries.len() {
                self.pools[best_pool_id].entries[best_response_id].select_count += 1;
            }
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

        ChatResponse {
            response: best_response,
            intent: best_intent,
            confidence,
            resonance: best_score,
            pool_id: best_pool_id,
            response_id: best_response_id,
        }
    }

    /// Provide feedback on a response. Level 1 direct signal.
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
        } else {
            pool.entries[response_id].fitness =
                (pool.entries[response_id].fitness - self.growth.negative_penalty).max(0.0);
            self.stats.negative_feedbacks += 1;
        }
    }

    /// Execute one growth cycle: decay → prune → emerge → replicate.
    pub fn growth_cycle(&mut self) {
        self.stats.total_cycles += 1;
        self.decay_fitness();
        self.prune_low_fitness();
        self.check_emergence();
        self.replicate_high_fitness();
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
            let mut candidates: Vec<(String, ReservoirState, Vec<f64>, usize)> = Vec::new();
            for entry in &self.pools[pool_idx].entries {
                if entry.fitness >= threshold {
                    candidates.push((
                        entry.text.clone(),
                        entry.wave.clone(),
                        entry.features.clone(),
                        entry.generation,
                    ));
                }
            }

            for (text, wave, features, generation) in candidates {
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
}
