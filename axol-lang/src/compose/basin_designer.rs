//! Basin designer — inverse problem: desired basins -> ChaosEngine params.
//!
//! Phase 1: Grid search over (r, epsilon)
//! Phase 2: Nelder-Mead refinement on top candidates

use crate::dynamics::{ChaosEngine, Basin};
use crate::types::BasinStructure;
use crate::weaver::build_basin_structure;
use crate::errors::{AxolError, Result};

/// Specification for desired basin structure.
#[derive(Clone, Debug)]
pub struct BasinSpec {
    pub n_basins: usize,
    pub target_sizes: Vec<f64>,
    pub boundary_hints: Vec<Vec<f64>>,
    pub dim: usize,
}

/// Configuration for basin design optimization.
#[derive(Clone, Debug)]
pub struct BasinDesignConfig {
    pub grid_r_steps: usize,
    pub grid_eps_steps: usize,
    pub top_k: usize,
    pub nelder_mead_iterations: usize,
    pub n_samples: usize,
    pub transient: usize,
}

impl Default for BasinDesignConfig {
    fn default() -> Self {
        Self {
            grid_r_steps: 20,
            grid_eps_steps: 20,
            top_k: 5,
            nelder_mead_iterations: 100,
            n_samples: 200,
            transient: 200,
        }
    }
}

/// Result of basin design.
#[derive(Clone, Debug)]
pub struct BasinDesign {
    pub engine: ChaosEngine,
    pub basins: Vec<Basin>,
    pub basin_structure: BasinStructure,
    pub score: f64,
    pub iterations: usize,
}

/// Design basins to match the given specification.
///
/// Phase 1: Grid search over (r, epsilon) to find candidates.
/// Phase 2: Nelder-Mead refinement on the top-k candidates.
pub fn design_basins(spec: &BasinSpec, config: &BasinDesignConfig) -> Result<BasinDesign> {
    if spec.n_basins == 0 {
        return Err(AxolError::Compose("n_basins must be > 0".into()));
    }
    if spec.dim == 0 {
        return Err(AxolError::Compose("dim must be > 0".into()));
    }

    // Normalize target sizes
    let target_sizes = if spec.target_sizes.is_empty() {
        vec![1.0 / spec.n_basins as f64; spec.n_basins]
    } else {
        let sum: f64 = spec.target_sizes.iter().sum();
        if sum > 0.0 {
            spec.target_sizes.iter().map(|&s| s / sum).collect()
        } else {
            vec![1.0 / spec.n_basins as f64; spec.n_basins]
        }
    };

    // Phase 1: Grid search over (r, epsilon)
    let mut candidates: Vec<(f64, f64, f64)> = Vec::new(); // (score, r, epsilon)

    for ri in 0..config.grid_r_steps {
        let r = 3.0 + (ri as f64 / (config.grid_r_steps - 1).max(1) as f64) * 1.0; // [3.0, 4.0]
        for ei in 0..config.grid_eps_steps {
            let epsilon = 0.01 + (ei as f64 / (config.grid_eps_steps - 1).max(1) as f64) * 0.94; // [0.01, 0.95]

            let weights = uniform_weights(spec.dim);
            let engine = ChaosEngine {
                dim: spec.dim,
                r,
                epsilon,
                weights,
            };

            let basins = engine.find_basins(config.n_samples, config.transient, 42);
            let score = score_basins(spec, &basins, &target_sizes);
            candidates.push((score, r, epsilon));
        }
    }

    // Sort by score (lower is better)
    candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Phase 2: Nelder-Mead refinement on top-k
    let top_k = config.top_k.min(candidates.len());
    let mut best_score = f64::MAX;
    let mut best_engine: Option<ChaosEngine> = None;
    let mut best_basins: Vec<Basin> = Vec::new();
    let mut total_iterations = candidates.len(); // grid evaluations

    for i in 0..top_k {
        let (_, r0, eps0) = candidates[i];

        let (opt_params, opt_score, nm_iters) = nelder_mead_optimize(
            &[r0, eps0],
            &|params| {
                let r = params[0].clamp(3.0, 4.0);
                let eps = params[1].clamp(0.01, 0.95);
                let weights = uniform_weights(spec.dim);
                let engine = ChaosEngine {
                    dim: spec.dim,
                    r,
                    epsilon: eps,
                    weights,
                };
                let basins = engine.find_basins(config.n_samples, config.transient, 42);
                score_basins(spec, &basins, &target_sizes)
            },
            config.nelder_mead_iterations,
        );

        total_iterations += nm_iters;

        if opt_score < best_score {
            best_score = opt_score;
            let r = opt_params[0].clamp(3.0, 4.0);
            let eps = opt_params[1].clamp(0.01, 0.95);
            let weights = uniform_weights(spec.dim);
            let engine = ChaosEngine {
                dim: spec.dim,
                r,
                epsilon: eps,
                weights,
            };
            best_basins = engine.find_basins(config.n_samples, config.transient, 42);
            best_engine = Some(engine);
        }
    }

    let engine = best_engine.ok_or_else(|| AxolError::Compose("No valid basin design found".into()))?;

    let basin_structure = build_basin_structure(&best_basins, spec.dim, 1.0, None);

    Ok(BasinDesign {
        engine,
        basins: best_basins,
        basin_structure,
        score: best_score,
        iterations: total_iterations,
    })
}

/// Score how well detected basins match the spec.
///
/// 0.0 = perfect match. Weights:
/// - Basin count match (heavy weight: 10.0)
/// - Size ratio match (weight: 5.0)
/// - Boundary alignment (weight: 1.0)
pub fn score_basins(spec: &BasinSpec, basins: &[Basin], target_sizes: &[f64]) -> f64 {
    // Basin count penalty
    let count_penalty = (basins.len() as f64 - spec.n_basins as f64).abs() * 10.0;

    // Size ratio match
    let mut size_penalty = 0.0;
    if !basins.is_empty() && !target_sizes.is_empty() {
        let mut sorted_sizes: Vec<f64> = basins.iter().map(|b| b.size).collect();
        sorted_sizes.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let mut sorted_targets = target_sizes.to_vec();
        sorted_targets.sort_by(|a, b| b.partial_cmp(a).unwrap());

        let n = sorted_sizes.len().min(sorted_targets.len());
        for i in 0..n {
            size_penalty += (sorted_sizes[i] - sorted_targets[i]).abs();
        }
        // Penalty for missing or extra basins
        if sorted_sizes.len() > sorted_targets.len() {
            for i in n..sorted_sizes.len() {
                size_penalty += sorted_sizes[i];
            }
        }
        if sorted_targets.len() > sorted_sizes.len() {
            for i in n..sorted_targets.len() {
                size_penalty += sorted_targets[i];
            }
        }
        size_penalty *= 5.0;
    }

    // Boundary alignment (if hints provided)
    let mut boundary_penalty = 0.0;
    if !spec.boundary_hints.is_empty() && !basins.is_empty() {
        for hint in &spec.boundary_hints {
            let min_dist: f64 = basins.iter().map(|b| {
                b.center.iter().zip(hint.iter())
                    .map(|(a, h)| (a - h).powi(2))
                    .sum::<f64>()
                    .sqrt()
            }).fold(f64::MAX, f64::min);
            boundary_penalty += min_dist;
        }
        boundary_penalty *= 1.0;
    }

    count_penalty + size_penalty + boundary_penalty
}

/// Nelder-Mead simplex optimizer (gradient-free, pure Rust).
///
/// Returns (best_params, best_score, iterations_used).
pub fn nelder_mead_optimize(
    initial: &[f64],
    objective: &dyn Fn(&[f64]) -> f64,
    max_iterations: usize,
) -> (Vec<f64>, f64, usize) {
    let n = initial.len();
    let alpha = 1.0;  // reflection
    let gamma = 2.0;  // expansion
    let rho = 0.5;    // contraction
    let sigma = 0.5;  // shrink

    // Initialize simplex: n+1 vertices
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(initial.to_vec());
    for i in 0..n {
        let mut vertex = initial.to_vec();
        vertex[i] += 0.05; // initial step size
        simplex.push(vertex);
    }

    let mut scores: Vec<f64> = simplex.iter().map(|v| objective(v)).collect();
    let mut iters = 0;

    for _ in 0..max_iterations {
        iters += 1;

        // Sort by score
        let mut indices: Vec<usize> = (0..=n).collect();
        indices.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap());

        let best = indices[0];
        let worst = indices[n];
        let second_worst = indices[n - 1];

        // Check convergence
        let score_range = scores[worst] - scores[best];
        if score_range < 1e-10 {
            break;
        }

        // Centroid (excluding worst)
        let mut centroid = vec![0.0; n];
        for &i in &indices[..n] {
            for j in 0..n {
                centroid[j] += simplex[i][j];
            }
        }
        for j in 0..n {
            centroid[j] /= n as f64;
        }

        // Reflection
        let reflected: Vec<f64> = centroid.iter().zip(simplex[worst].iter())
            .map(|(&c, &w)| c + alpha * (c - w))
            .collect();
        let reflected_score = objective(&reflected);

        if reflected_score < scores[second_worst] && reflected_score >= scores[best] {
            simplex[worst] = reflected;
            scores[worst] = reflected_score;
            continue;
        }

        // Expansion
        if reflected_score < scores[best] {
            let expanded: Vec<f64> = centroid.iter().zip(reflected.iter())
                .map(|(&c, &r)| c + gamma * (r - c))
                .collect();
            let expanded_score = objective(&expanded);

            if expanded_score < reflected_score {
                simplex[worst] = expanded;
                scores[worst] = expanded_score;
            } else {
                simplex[worst] = reflected;
                scores[worst] = reflected_score;
            }
            continue;
        }

        // Contraction
        let contracted: Vec<f64> = centroid.iter().zip(simplex[worst].iter())
            .map(|(&c, &w)| c + rho * (w - c))
            .collect();
        let contracted_score = objective(&contracted);

        if contracted_score < scores[worst] {
            simplex[worst] = contracted;
            scores[worst] = contracted_score;
            continue;
        }

        // Shrink
        let best_vertex = simplex[best].clone();
        for i in 0..=n {
            if i != best {
                for j in 0..n {
                    simplex[i][j] = best_vertex[j] + sigma * (simplex[i][j] - best_vertex[j]);
                }
                scores[i] = objective(&simplex[i]);
            }
        }
    }

    // Find best
    let best_idx = scores.iter().enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    (simplex[best_idx].clone(), scores[best_idx], iters)
}

/// Generate uniform coupling weights for a given dimension.
fn uniform_weights(dim: usize) -> Vec<f64> {
    let mut weights = vec![0.0; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            if i != j {
                weights[i * dim + j] = 1.0 / (dim - 1).max(1) as f64;
            }
        }
    }
    weights
}
