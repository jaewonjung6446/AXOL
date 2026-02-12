//! Learning mechanism — find ChaosEngine parameters that produce correct input→output mappings.
//!
//! AXOL learning searches over chaos dynamics parameters so that basin structures
//! naturally separate inputs into desired output classes.
//! Both axes are preserved:
//!   - Space axis: chaos dynamics still generate the attractor/basins
//!   - Probability axis: Born rule still performs measurement
//!   - Learning finds the RIGHT parameters, not the right outputs
//!
//! Architecture:
//!   Phase 1: Grid search over (r, epsilon) × relation kinds × seeds
//!   Phase 2: Nelder-Mead refinement on (r, epsilon) for top candidates
//!   Phase 3: Full weight optimization via Nelder-Mead (if accuracy < 1.0)

use rand::SeedableRng;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

use crate::dynamics::ChaosEngine;
use crate::weaver::{Tapestry, Attractor, TapestryNode, WeaverReport};
use crate::types::*;
use crate::declare::RelationKind;
use crate::observatory;
use crate::compose::basin_designer::nelder_mead_optimize;
use crate::errors::{AxolError, Result};

// ---------------------------------------------------------------------------
// Training data
// ---------------------------------------------------------------------------

/// A single training example: input vector → expected output class.
#[derive(Clone, Debug)]
pub struct TrainingSample {
    pub input: Vec<f64>,
    pub expected: usize,
}

/// A set of training examples for learning.
#[derive(Clone, Debug)]
pub struct TrainingSet {
    pub samples: Vec<TrainingSample>,
    pub dim: usize,
    pub n_classes: usize,
    pub name: String,
}

impl TrainingSet {
    pub fn new(name: &str, dim: usize, n_classes: usize) -> Self {
        Self {
            samples: Vec::new(),
            dim,
            n_classes,
            name: name.to_string(),
        }
    }

    pub fn add(&mut self, input: Vec<f64>, expected: usize) -> &mut Self {
        self.samples.push(TrainingSample { input, expected });
        self
    }

    /// XOR truth table (non-linearly separable).
    /// Input: concat two dim=2 booleans → dim=4.
    /// Output: 0=FALSE, 1=TRUE.
    pub fn xor() -> Self {
        let mut ts = Self::new("xor", 4, 2);
        ts.add(vec![0.9, 0.1, 0.9, 0.1], 0); // F XOR F = F
        ts.add(vec![0.9, 0.1, 0.1, 0.9], 1); // F XOR T = T
        ts.add(vec![0.1, 0.9, 0.9, 0.1], 1); // T XOR F = T
        ts.add(vec![0.1, 0.9, 0.1, 0.9], 0); // T XOR T = F
        ts
    }

    /// AND truth table.
    pub fn and() -> Self {
        let mut ts = Self::new("and", 4, 2);
        ts.add(vec![0.9, 0.1, 0.9, 0.1], 0);
        ts.add(vec![0.9, 0.1, 0.1, 0.9], 0);
        ts.add(vec![0.1, 0.9, 0.9, 0.1], 0);
        ts.add(vec![0.1, 0.9, 0.1, 0.9], 1);
        ts
    }

    /// OR truth table.
    pub fn or() -> Self {
        let mut ts = Self::new("or", 4, 2);
        ts.add(vec![0.9, 0.1, 0.9, 0.1], 0);
        ts.add(vec![0.9, 0.1, 0.1, 0.9], 1);
        ts.add(vec![0.1, 0.9, 0.9, 0.1], 1);
        ts.add(vec![0.1, 0.9, 0.1, 0.9], 1);
        ts
    }

    /// NOT truth table. Input dim=2, single boolean.
    pub fn not() -> Self {
        let mut ts = Self::new("not", 2, 2);
        ts.add(vec![0.9, 0.1], 1); // NOT F = T
        ts.add(vec![0.1, 0.9], 0); // NOT T = F
        ts
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct LearnConfig {
    pub grid_r_steps: usize,
    pub grid_eps_steps: usize,
    pub top_k: usize,
    pub nelder_mead_iters: usize,
    pub quantum: bool,
    pub seeds: Vec<u64>,
    pub optimize_weights: bool,
}

impl Default for LearnConfig {
    fn default() -> Self {
        Self {
            grid_r_steps: 15,
            grid_eps_steps: 15,
            top_k: 10,
            nelder_mead_iters: 200,
            quantum: true,
            seeds: vec![42, 123, 7],
            optimize_weights: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct LearnResult {
    pub tapestry: Tapestry,
    pub engine: ChaosEngine,
    pub accuracy: f64,
    pub total_evaluations: usize,
    pub best_r: f64,
    pub best_epsilon: f64,
    pub best_relation: RelationKind,
    pub best_seed: u64,
    pub history: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Core learning function
// ---------------------------------------------------------------------------

pub fn learn(training: &TrainingSet, config: &LearnConfig) -> Result<LearnResult> {
    if training.samples.is_empty() {
        return Err(AxolError::Compose("Training set is empty".into()));
    }
    if training.dim == 0 {
        return Err(AxolError::Compose("Training dimension must be > 0".into()));
    }

    // Validate sample dimensions
    for (i, sample) in training.samples.iter().enumerate() {
        if sample.input.len() != training.dim {
            return Err(AxolError::Compose(format!(
                "Sample {} has {} inputs, expected {}", i, sample.input.len(), training.dim
            )));
        }
    }

    let dim = training.dim;
    let relation_kinds = [
        RelationKind::Proportional,
        RelationKind::Additive,
        RelationKind::Multiplicative,
        RelationKind::Inverse,
        RelationKind::Conditional,
    ];

    // =================================================================
    // Phase 1: Grid search over (r, epsilon, relation_kind, seed)
    // =================================================================
    let mut candidates: Vec<(f64, f64, f64, RelationKind, u64)> = Vec::new();

    for &seed in &config.seeds {
        for kind in &relation_kinds {
            let weights = build_weights(dim, kind, seed);
            for ri in 0..config.grid_r_steps {
                let r = 3.0 + (ri as f64 / (config.grid_r_steps - 1).max(1) as f64);
                for ei in 0..config.grid_eps_steps {
                    let eps = 0.01 + (ei as f64 / (config.grid_eps_steps - 1).max(1) as f64) * 0.94;
                    let engine = ChaosEngine {
                        dim,
                        r,
                        epsilon: eps,
                        weights: weights.clone(),
                    };
                    let loss = evaluate_loss(&engine, training, config.quantum, seed);
                    candidates.push((loss, r, eps, kind.clone(), seed));
                }
            }
        }
    }

    candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut history = vec![1.0 - candidates.first().map(|c| c.0).unwrap_or(1.0)];
    let mut total_evals = candidates.len();

    // Perfect accuracy in grid search?
    if let Some(best) = candidates.first() {
        if best.0 < 1e-10 {
            let engine = ChaosEngine {
                dim,
                r: best.1,
                epsilon: best.2,
                weights: build_weights(dim, &best.3, best.4),
            };
            let tapestry = build_tapestry_from_engine(&training.name, &engine, config.quantum, best.4)?;
            let final_accuracy = evaluate(&tapestry, training);
            return Ok(LearnResult {
                tapestry,
                engine,
                accuracy: final_accuracy,
                total_evaluations: total_evals,
                best_r: best.1,
                best_epsilon: best.2,
                best_relation: best.3.clone(),
                best_seed: best.4,
                history,
            });
        }
    }

    // =================================================================
    // Phase 2: Nelder-Mead refinement on (r, epsilon) for top-k
    // =================================================================
    let top_k = config.top_k.min(candidates.len());
    let mut best_loss = candidates[0].0;
    let mut best_r = candidates[0].1;
    let mut best_eps = candidates[0].2;
    let mut best_kind = candidates[0].3.clone();
    let mut best_seed = candidates[0].4;

    for i in 0..top_k {
        let c = candidates[i].clone();
        let weights = build_weights(dim, &c.3, c.4);
        let s = c.4;

        let (opt_params, opt_loss, nm_iters) = nelder_mead_optimize(
            &[c.1, c.2],
            &|params| {
                let r = params[0].clamp(3.0, 4.0);
                let eps = params[1].clamp(0.01, 0.95);
                let engine = ChaosEngine {
                    dim,
                    r,
                    epsilon: eps,
                    weights: weights.clone(),
                };
                evaluate_loss(&engine, training, config.quantum, s)
            },
            config.nelder_mead_iters,
        );

        total_evals += nm_iters;

        if opt_loss < best_loss {
            best_loss = opt_loss;
            best_r = opt_params[0].clamp(3.0, 4.0);
            best_eps = opt_params[1].clamp(0.01, 0.95);
            best_kind = c.3;
            best_seed = c.4;
            history.push(1.0 - opt_loss);
        }

        if best_loss < 1e-10 {
            break;
        }
    }

    // =================================================================
    // Phase 3: Weight optimization via Nelder-Mead (if needed)
    // =================================================================
    let mut best_weights = build_weights(dim, &best_kind, best_seed);

    if config.optimize_weights && best_loss > 1e-10 && dim <= 8 {
        let mut initial = vec![best_r, best_eps];
        for i in 0..dim {
            for j in 0..dim {
                if i != j {
                    initial.push(best_weights[i * dim + j]);
                }
            }
        }

        let s = best_seed;
        let (opt_params, opt_loss, nm_iters) = nelder_mead_optimize(
            &initial,
            &|params| {
                let r = params[0].clamp(3.0, 4.0);
                let eps = params[1].clamp(0.01, 0.95);
                let mut weights = vec![0.0; dim * dim];
                let mut k = 2;
                for i in 0..dim {
                    for j in 0..dim {
                        if i != j {
                            weights[i * dim + j] = params[k];
                            k += 1;
                        }
                    }
                }
                normalize_weights(&mut weights, dim);
                let engine = ChaosEngine { dim, r, epsilon: eps, weights };
                evaluate_loss(&engine, training, config.quantum, s)
            },
            config.nelder_mead_iters * 2,
        );

        total_evals += nm_iters;

        if opt_loss < best_loss {
            // best_loss = opt_loss; // not read after this
            best_r = opt_params[0].clamp(3.0, 4.0);
            best_eps = opt_params[1].clamp(0.01, 0.95);

            let mut weights = vec![0.0; dim * dim];
            let mut k = 2;
            for i in 0..dim {
                for j in 0..dim {
                    if i != j {
                        weights[i * dim + j] = opt_params[k];
                        k += 1;
                    }
                }
            }
            normalize_weights(&mut weights, dim);
            best_weights = weights;
            history.push(1.0 - opt_loss);
        }
    }

    // =================================================================
    // Build final tapestry from best parameters
    // =================================================================
    let engine = ChaosEngine {
        dim,
        r: best_r,
        epsilon: best_eps,
        weights: best_weights,
    };
    let tapestry = build_tapestry_from_engine(&training.name, &engine, config.quantum, best_seed)?;
    let final_accuracy = evaluate(&tapestry, training);

    Ok(LearnResult {
        tapestry,
        engine,
        accuracy: final_accuracy,
        total_evaluations: total_evals,
        best_r,
        best_epsilon: best_eps,
        best_relation: best_kind,
        best_seed,
        history,
    })
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

/// Evaluate accuracy of a tapestry on training data using the full observation pipeline.
pub fn evaluate(tapestry: &Tapestry, training: &TrainingSet) -> f64 {
    if training.samples.is_empty() {
        return 0.0;
    }

    let mut correct = 0usize;
    for sample in &training.samples {
        let input_fv = FloatVec::new(sample.input.iter().map(|&v| v as f32).collect());
        let inputs: Vec<(&str, &FloatVec)> = vec![("x", &input_fv)];

        if let Ok(obs) = observatory::observe(tapestry, &inputs) {
            if obs.value_index == sample.expected {
                correct += 1;
            }
        }
    }

    correct as f64 / training.samples.len() as f64
}

/// Evaluate loss for a given ChaosEngine configuration.
/// Routes through the same observatory::observe pipeline used by evaluate(),
/// ensuring the optimizer sees the exact same loss landscape as final evaluation.
/// Returns value in [0, ~1]: 0 = perfect, 1 = worst.
fn evaluate_loss(
    engine: &ChaosEngine,
    training: &TrainingSet,
    quantum: bool,
    seed: u64,
) -> f64 {
    // Build tapestry using the same pipeline as final evaluation
    let tapestry = match build_tapestry_from_engine(&training.name, engine, quantum, seed) {
        Ok(t) => t,
        Err(_) => return 1.0,
    };

    let n = training.samples.len() as f64;
    let mut total_loss = 0.0;

    for sample in &training.samples {
        let input_fv = FloatVec::new(sample.input.iter().map(|&v| v as f32).collect());
        let inputs: Vec<(&str, &FloatVec)> = vec![("x", &input_fv)];

        match observatory::observe(&tapestry, &inputs) {
            Ok(obs) => {
                // Continuous loss: 1 - probability of correct class
                let expected_prob = obs.probabilities.data
                    .get(sample.expected)
                    .copied()
                    .unwrap_or(0.0) as f64;
                total_loss += 1.0 - expected_prob;
            }
            Err(_) => {
                total_loss += 1.0;
            }
        }
    }

    total_loss / n
}

// ---------------------------------------------------------------------------
// Weight construction
// ---------------------------------------------------------------------------

/// Build coupling weight matrix for a given relation kind.
pub fn build_weights(dim: usize, kind: &RelationKind, seed: u64) -> Vec<f64> {
    let mut weights = vec![0.0f64; dim * dim];

    match kind {
        RelationKind::Proportional => {
            for i in 0..dim {
                for j in 0..dim {
                    if i != j {
                        weights[i * dim + j] = 1.0 / (dim - 1).max(1) as f64;
                    }
                }
            }
        }
        RelationKind::Additive => {
            for i in 0..dim {
                for j in 0..dim {
                    weights[i * dim + j] = 1.0 / dim as f64;
                }
            }
        }
        RelationKind::Multiplicative => {
            for i in 0..dim {
                let j = (i + 1) % dim;
                weights[i * dim + j] += 0.5;
                weights[j * dim + i] += 0.5;
            }
        }
        RelationKind::Inverse => {
            for i in 0..dim {
                for j in 0..dim {
                    if i != j {
                        weights[i * dim + j] = -1.0 / (dim - 1).max(1) as f64;
                    }
                }
            }
        }
        RelationKind::Conditional => {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let n_conn = (dim * dim / 3).max(1);
            for _ in 0..n_conn {
                let i = rng.gen_range(0..dim);
                let j = rng.gen_range(0..dim);
                weights[i * dim + j] += rng.gen::<f64>() * 2.0 - 1.0;
            }
        }
    }

    normalize_weights(&mut weights, dim);
    weights
}

fn normalize_weights(weights: &mut [f64], dim: usize) {
    for i in 0..dim {
        let row_sum: f64 = (0..dim).map(|j| weights[i * dim + j].abs()).sum();
        if row_sum > 1e-15 {
            for j in 0..dim {
                weights[i * dim + j] /= row_sum;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tapestry construction from learned parameters
// ---------------------------------------------------------------------------

/// Build a full Tapestry from a ChaosEngine (for use after learning).
pub fn build_tapestry_from_engine(
    name: &str,
    engine: &ChaosEngine,
    quantum: bool,
    seed: u64,
) -> Result<Tapestry> {
    let dim = engine.dim;

    // Dynamics — consistent parameters used by both training and final evaluation
    let result = engine.find_attractor(seed, 100, 200);
    let matrix = engine.extract_matrix(&result);

    let global_attractor = Attractor {
        phase_space_dim: dim,
        fractal_dim: result.fractal_dim,
        max_lyapunov: result.max_lyapunov,
        lyapunov_spectrum: result.lyapunov_spectrum.clone(),
        trajectory_matrix: matrix.clone(),
    };

    let basins = if quantum {
        Some(engine.find_basins(100, 100, seed))
    } else {
        None
    };

    // Nodes
    let mut nodes = HashMap::new();
    let input_name = "x".to_string();
    let output_name = "y".to_string();

    nodes.insert(input_name.clone(), TapestryNode {
        name: input_name.clone(),
        amplitudes: FloatVec::zeros(dim),
        complex_amplitudes: None,
        labels: HashMap::new(),
        attractor: global_attractor.clone(),
        depth: 0,
    });

    nodes.insert(output_name.clone(), TapestryNode {
        name: output_name.clone(),
        amplitudes: FloatVec::zeros(dim),
        complex_amplitudes: None,
        labels: HashMap::new(),
        attractor: global_attractor.clone(),
        depth: 1,
    });

    let report = WeaverReport {
        target_omega: global_attractor.omega(),
        target_phi: global_attractor.phi(),
        estimated_omega: global_attractor.omega(),
        estimated_phi: global_attractor.phi(),
        max_lyapunov: result.max_lyapunov,
        fractal_dim: result.fractal_dim,
        feasible: true,
        warnings: Vec::new(),
    };

    Ok(Tapestry {
        name: name.to_string(),
        nodes,
        input_names: vec![input_name],
        output_names: vec![output_name],
        global_attractor,
        report,
        composed_matrix: Some(matrix),
        quantum,
        density_matrix: None,
        kraus_operators: None,
        chaos_engine: Some(engine.clone()),
        basins,
    })
}
