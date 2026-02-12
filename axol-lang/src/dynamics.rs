//! Chaos dynamics engine: coupled map lattice, Lyapunov exponents,
//! fractal dimension, basin detection.
//!
//! This replaces the fake random-based attractor generation with
//! real iterative dynamics on a coupled logistic map lattice.

use rand::SeedableRng;
use rand::Rng;
use rand_chacha::ChaCha8Rng;

use crate::declare::*;
use crate::types::*;

// =========================================================================
// Core types
// =========================================================================

/// Coupled map lattice — N-dimensional chaotic dynamical system.
///
/// Each node evolves via the logistic map f(x) = r*x*(1-x),
/// coupled to other nodes via a weight matrix.
///
/// x_i(t+1) = (1-eps)*f(x_i) + eps * sum_j w_ij * f(x_j)
#[derive(Clone, Debug)]
pub struct ChaosEngine {
    pub dim: usize,
    pub r: f64,
    pub epsilon: f64,
    pub weights: Vec<f64>,
}

/// Result of attractor analysis.
#[derive(Clone, Debug)]
pub struct AttractorResult {
    pub lyapunov_spectrum: Vec<f64>,
    pub max_lyapunov: f64,
    pub fractal_dim: f64,
    pub final_state: Vec<f64>,
    pub linearization: Vec<f64>,
}

/// A detected basin of attraction.
#[derive(Clone, Debug)]
pub struct Basin {
    pub center: Vec<f64>,
    pub size: f64,
    pub local_lyapunov: f64,
    pub phase: f64,
}

// =========================================================================
// ChaosEngine
// =========================================================================

impl ChaosEngine {
    /// Build from declaration.
    ///
    /// Quality targets map to dynamics parameters:
    ///   omega → r (control parameter): higher omega = less chaos = lower r
    ///   phi → epsilon (coupling): higher phi = weaker coupling
    ///
    /// Relation kinds define the coupling weight structure.
    pub fn from_declaration(decl: &EntangleDeclaration, seed: u64) -> Self {
        let dim = decl.inputs.first().map(|i| i.dim).unwrap_or(4);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        // omega=0.9 → lambda=0.11 → r≈3.11 (mild chaos)
        // omega=0.5 → lambda=1.0  → r≈3.57 (onset of chaos)
        // omega=0.1 → lambda=9.0  → r≈4.0  (full chaos)
        let target_lambda = (1.0 / decl.quality.omega) - 1.0;
        let r = (3.0 + target_lambda.min(1.0)).clamp(3.0, 4.0);

        let epsilon = (1.0 - decl.quality.phi).clamp(0.01, 0.95);

        let mut weights = vec![0.0f64; dim * dim];

        for rel in &decl.relations {
            match rel.kind {
                RelationKind::Proportional => {
                    // Symmetric diffusive coupling
                    for i in 0..dim {
                        for j in 0..dim {
                            if i != j {
                                weights[i * dim + j] += 1.0 / (dim - 1).max(1) as f64;
                            }
                        }
                    }
                }
                RelationKind::Additive => {
                    // Global mean-field coupling
                    for i in 0..dim {
                        for j in 0..dim {
                            weights[i * dim + j] += 1.0 / dim as f64;
                        }
                    }
                }
                RelationKind::Multiplicative => {
                    // Ring coupling (nearest-neighbor)
                    for i in 0..dim {
                        let j = (i + 1) % dim;
                        weights[i * dim + j] += 0.5;
                        weights[j * dim + i] += 0.5;
                    }
                }
                RelationKind::Inverse => {
                    // Anti-correlated
                    for i in 0..dim {
                        for j in 0..dim {
                            if i != j {
                                weights[i * dim + j] -= 1.0 / (dim - 1).max(1) as f64;
                            }
                        }
                    }
                }
                RelationKind::Conditional => {
                    // Sparse random
                    let n_conn = (dim * dim / 3).max(1);
                    for _ in 0..n_conn {
                        let i = rng.gen_range(0..dim);
                        let j = rng.gen_range(0..dim);
                        weights[i * dim + j] += rng.gen::<f64>() * 2.0 - 1.0;
                    }
                }
            }
        }

        // Normalize each row
        for i in 0..dim {
            let row_sum: f64 = (0..dim).map(|j| weights[i * dim + j].abs()).sum();
            if row_sum > 1e-15 {
                for j in 0..dim {
                    weights[i * dim + j] /= row_sum;
                }
            }
        }

        ChaosEngine { dim, r, epsilon, weights }
    }

    /// Logistic map
    #[inline]
    fn f(&self, x: f64) -> f64 {
        self.r * x * (1.0 - x)
    }

    /// Logistic map derivative
    #[inline]
    fn df(&self, x: f64) -> f64 {
        self.r * (1.0 - 2.0 * x)
    }

    /// One step of the coupled map lattice.
    pub fn step(&self, state: &[f64]) -> Vec<f64> {
        let mut next = vec![0.0; self.dim];
        for i in 0..self.dim {
            let self_term = (1.0 - self.epsilon) * self.f(state[i]);
            let mut coupling = 0.0;
            for j in 0..self.dim {
                coupling += self.weights[i * self.dim + j] * self.f(state[j]);
            }
            next[i] = (self_term + self.epsilon * coupling).clamp(0.001, 0.999);
        }
        next
    }

    /// Jacobian at current state.
    pub fn jacobian(&self, state: &[f64]) -> Vec<f64> {
        let mut jac = vec![0.0; self.dim * self.dim];
        for i in 0..self.dim {
            for j in 0..self.dim {
                let coupling_term = self.epsilon * self.weights[i * self.dim + j] * self.df(state[j]);
                if i == j {
                    jac[i * self.dim + j] = (1.0 - self.epsilon) * self.df(state[i]) + coupling_term;
                } else {
                    jac[i * self.dim + j] = coupling_term;
                }
            }
        }
        jac
    }

    /// Run dynamics to find attractor; compute Lyapunov exponents and fractal dim.
    pub fn find_attractor(&self, seed: u64, transient: usize, record: usize) -> AttractorResult {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut state: Vec<f64> = (0..self.dim)
            .map(|_| rng.gen::<f64>() * 0.8 + 0.1)
            .collect();

        // Transient
        for _ in 0..transient {
            state = self.step(&state);
        }

        // QR-based Lyapunov computation
        let mut q = vec![0.0f64; self.dim * self.dim];
        for i in 0..self.dim {
            q[i * self.dim + i] = 1.0;
        }
        let mut lyap_sums = vec![0.0f64; self.dim];
        let qr_period = 5;
        let mut qr_count = 0;
        let mut jac_sum = vec![0.0f64; self.dim * self.dim];

        // Trajectory for fractal dimension
        let traj_skip = record.max(1) / 500.min(record); // keep ~500 pts
        let traj_skip = traj_skip.max(1);
        let mut trajectory: Vec<Vec<f64>> = Vec::new();

        for t in 0..record {
            if t % traj_skip == 0 {
                trajectory.push(state.clone());
            }

            let jac = self.jacobian(&state);
            for k in 0..self.dim * self.dim {
                jac_sum[k] += jac[k];
            }

            // Q ← J @ Q
            let mut new_q = vec![0.0f64; self.dim * self.dim];
            for i in 0..self.dim {
                for j in 0..self.dim {
                    let mut s = 0.0;
                    for k in 0..self.dim {
                        s += jac[i * self.dim + k] * q[k * self.dim + j];
                    }
                    new_q[i * self.dim + j] = s;
                }
            }
            q = new_q;

            if (t + 1) % qr_period == 0 {
                let (q_new, r_diag) = gram_schmidt_qr(&q, self.dim);
                q = q_new;
                for i in 0..self.dim {
                    if r_diag[i].abs() > 1e-300 {
                        lyap_sums[i] += r_diag[i].abs().ln();
                    }
                }
                qr_count += 1;
            }

            state = self.step(&state);
        }

        // Lyapunov exponents
        let divisor = (qr_count as f64 * qr_period as f64).max(1.0);
        let mut lyapunov_spectrum: Vec<f64> = lyap_sums.iter()
            .map(|&s| s / divisor)
            .collect();
        lyapunov_spectrum.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let max_lyapunov = lyapunov_spectrum[0];

        // Linearization (average Jacobian)
        let linearization: Vec<f64> = jac_sum.iter()
            .map(|&s| s / record as f64)
            .collect();

        // Fractal dimension
        let fractal_dim = correlation_dimension(&trajectory, self.dim);

        AttractorResult {
            lyapunov_spectrum,
            max_lyapunov,
            fractal_dim,
            final_state: state,
            linearization,
        }
    }

    /// Find basins by sampling initial conditions.
    pub fn find_basins(&self, n_samples: usize, transient: usize, seed: u64) -> Vec<Basin> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let mut endpoints: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let mut state: Vec<f64> = (0..self.dim)
                .map(|_| rng.gen::<f64>() * 0.8 + 0.1)
                .collect();
            for _ in 0..transient {
                state = self.step(&state);
            }
            // Average a few more steps for stability
            let avg_n = 20;
            let mut avg = vec![0.0; self.dim];
            for _ in 0..avg_n {
                state = self.step(&state);
                for i in 0..self.dim {
                    avg[i] += state[i];
                }
            }
            for v in avg.iter_mut() {
                *v /= avg_n as f64;
            }
            endpoints.push(avg);
        }

        // Distance-based clustering
        let threshold = 0.15;
        let mut clusters: Vec<(Vec<f64>, usize)> = Vec::new();

        for ep in &endpoints {
            let mut found = false;
            for (center, count) in clusters.iter_mut() {
                let dist: f64 = center.iter().zip(ep.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if dist < threshold {
                    let n = *count as f64;
                    for (c, e) in center.iter_mut().zip(ep.iter()) {
                        *c = (*c * n + *e) / (n + 1.0);
                    }
                    *count += 1;
                    found = true;
                    break;
                }
            }
            if !found {
                clusters.push((ep.clone(), 1));
            }
        }

        let total = n_samples as f64;
        let n_basins = clusters.len().max(1);
        let mut basins: Vec<Basin> = clusters.iter().enumerate().map(|(idx, (center, count))| {
            let jac = self.jacobian(center);
            let sr = spectral_radius(&jac, self.dim);
            let local_lyap = if sr > 1e-15 { sr.ln() } else { -10.0 };

            Basin {
                center: center.clone(),
                size: *count as f64 / total,
                local_lyapunov: local_lyap,
                phase: idx as f64 * 2.0 * std::f64::consts::PI / n_basins as f64,
            }
        }).collect();

        basins.sort_by(|a, b| b.size.partial_cmp(&a.size).unwrap());
        basins
    }

    /// Short run from given input (for observe-time basin proximity).
    pub fn short_run(&self, input: &[f64], steps: usize) -> Vec<f64> {
        let mut state: Vec<f64> = input.iter()
            .map(|&v| v.clamp(0.001, 0.999))
            .collect();
        // Pad or truncate to dim
        state.resize(self.dim, 0.5);

        for _ in 0..steps {
            state = self.step(&state);
        }
        state
    }

    /// Extract transformation matrix from linearization.
    pub fn extract_matrix(&self, result: &AttractorResult) -> TransMatrix {
        let data: Vec<f32> = result.linearization.iter()
            .map(|&v| v as f32)
            .collect();
        TransMatrix::new(data, self.dim, self.dim)
    }
}

// =========================================================================
// Linear algebra helpers
// =========================================================================

/// Gram-Schmidt QR. Returns (Q, R_diagonal).
fn gram_schmidt_qr(a: &[f64], dim: usize) -> (Vec<f64>, Vec<f64>) {
    let mut q = vec![0.0f64; dim * dim];
    let mut r_diag = vec![0.0f64; dim];

    let mut cols: Vec<Vec<f64>> = (0..dim).map(|j| {
        (0..dim).map(|i| a[i * dim + j]).collect()
    }).collect();

    for j in 0..dim {
        for k in 0..j {
            let q_col: Vec<f64> = (0..dim).map(|i| q[i * dim + k]).collect();
            let dot: f64 = cols[j].iter().zip(q_col.iter()).map(|(a, b)| a * b).sum();
            for i in 0..dim {
                cols[j][i] -= dot * q_col[i];
            }
        }
        let norm: f64 = cols[j].iter().map(|x| x * x).sum::<f64>().sqrt();
        r_diag[j] = norm;
        if norm > 1e-15 {
            for i in 0..dim {
                q[i * dim + j] = cols[j][i] / norm;
            }
        }
    }
    (q, r_diag)
}

/// Correlation dimension from trajectory points.
fn correlation_dimension(trajectory: &[Vec<f64>], dim: usize) -> f64 {
    let n = trajectory.len();
    if n < 30 { return 1.0; }

    // Compute sampled pairwise distances
    let max_pairs = 3000;
    let step = ((n * (n - 1) / 2) / max_pairs).max(1);
    let mut distances = Vec::new();
    let mut idx = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            if idx % step == 0 {
                let d: f64 = trajectory[i].iter().zip(trajectory[j].iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if d > 1e-15 {
                    distances.push(d);
                }
            }
            idx += 1;
        }
    }
    if distances.len() < 10 { return 1.0; }

    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let total = distances.len() as f64;
    let p1 = (distances.len() / 100).max(1);
    let p90 = distances.len() * 90 / 100;
    let r_min = distances[p1];
    let r_max = distances[p90.min(distances.len() - 1)];
    if r_min <= 0.0 || r_max <= r_min { return 1.0; }

    let lr_min = r_min.ln();
    let lr_max = r_max.ln();
    let n_scales = 12;

    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for k in 0..n_scales {
        let lr = lr_min + (lr_max - lr_min) * k as f64 / (n_scales - 1) as f64;
        let r = lr.exp();
        let c = distances.iter().filter(|&&d| d < r).count() as f64 / total;
        if c > 1e-15 {
            xs.push(lr);
            ys.push(c.ln());
        }
    }
    if xs.len() < 3 { return 1.0; }

    // Linear regression slope = correlation dimension
    let n_pts = xs.len() as f64;
    let sx: f64 = xs.iter().sum();
    let sy: f64 = ys.iter().sum();
    let sxy: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| x * y).sum();
    let sxx: f64 = xs.iter().map(|x| x * x).sum();
    let slope = (n_pts * sxy - sx * sy) / (n_pts * sxx - sx * sx);
    slope.clamp(0.0, dim as f64)
}

/// Spectral radius via power iteration.
fn spectral_radius(mat: &[f64], dim: usize) -> f64 {
    let mut v: Vec<f64> = vec![0.0; dim];
    v[0] = 1.0;

    for _ in 0..50 {
        let mut mv = vec![0.0; dim];
        for i in 0..dim {
            for j in 0..dim {
                mv[i] += mat[i * dim + j] * v[j];
            }
        }
        let norm: f64 = mv.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-15 {
            for x in mv.iter_mut() { *x /= norm; }
        }
        v = mv;
    }

    let mut mv = vec![0.0; dim];
    for i in 0..dim {
        for j in 0..dim {
            mv[i] += mat[i * dim + j] * v[j];
        }
    }
    v.iter().zip(mv.iter()).map(|(a, b)| a * b).sum::<f64>().abs()
}
