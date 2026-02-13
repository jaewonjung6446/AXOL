//! Core AXOL types: FloatVec, ComplexVec, TransMatrix, DensityMatrix, BasinStructure.

use num_complex::Complex64;
use std::fmt;

// ---------------------------------------------------------------------------
// FloatVec — real-valued state vector (f32)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct FloatVec {
    pub data: Vec<f32>,
}

impl FloatVec {
    pub fn new(data: Vec<f32>) -> Self {
        Self { data }
    }

    pub fn zeros(dim: usize) -> Self {
        Self { data: vec![0.0; dim] }
    }

    pub fn from_slice(s: &[f32]) -> Self {
        Self { data: s.to_vec() }
    }

    pub fn dim(&self) -> usize {
        self.data.len()
    }

    pub fn norm(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
}

impl fmt::Display for FloatVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FloatVec({:?})", &self.data)
    }
}

// ---------------------------------------------------------------------------
// ComplexVec — complex-valued state vector (Complex64)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ComplexVec {
    pub data: Vec<Complex64>,
}

impl ComplexVec {
    pub fn new(data: Vec<Complex64>) -> Self {
        Self { data }
    }

    pub fn zeros(dim: usize) -> Self {
        Self { data: vec![Complex64::new(0.0, 0.0); dim] }
    }

    pub fn from_real(fv: &FloatVec) -> Self {
        Self {
            data: fv.data.iter().map(|&x| Complex64::new(x as f64, 0.0)).collect(),
        }
    }

    pub fn from_polar(magnitudes: &[f64], phases: &[f64]) -> Self {
        assert_eq!(magnitudes.len(), phases.len());
        Self {
            data: magnitudes
                .iter()
                .zip(phases.iter())
                .map(|(&m, &p)| Complex64::from_polar(m, p))
                .collect(),
        }
    }

    pub fn dim(&self) -> usize {
        self.data.len()
    }

    pub fn amplitudes(&self) -> Vec<f64> {
        self.data.iter().map(|c| c.norm()).collect()
    }

    pub fn phases(&self) -> Vec<f64> {
        self.data.iter().map(|c| c.arg()).collect()
    }

    pub fn norm(&self) -> f64 {
        self.data.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt()
    }

    pub fn to_real(&self) -> FloatVec {
        FloatVec {
            data: self.data.iter().map(|c| c.norm() as f32).collect(),
        }
    }

    pub fn normalized(&self) -> Self {
        let n = self.norm();
        if n > 0.0 {
            Self {
                data: self.data.iter().map(|c| c / n).collect(),
            }
        } else {
            self.clone()
        }
    }
}

impl fmt::Display for ComplexVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ComplexVec(dim={})", self.dim())
    }
}

// ---------------------------------------------------------------------------
// TransMatrix — transformation matrix (f32, row-major)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct TransMatrix {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl TransMatrix {
    pub fn new(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols, "data length must equal rows*cols");
        Self { data, rows, cols }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { data: vec![0.0; rows * cols], rows, cols }
    }

    pub fn identity(dim: usize) -> Self {
        let mut data = vec![0.0f32; dim * dim];
        for i in 0..dim {
            data[i * dim + i] = 1.0;
        }
        Self { data, rows: dim, cols: dim }
    }

    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols + col]
    }

    #[inline]
    pub fn set(&mut self, row: usize, col: usize, val: f32) {
        self.data[row * self.cols + col] = val;
    }

    /// Matrix multiply: self @ other
    pub fn matmul(&self, other: &TransMatrix) -> TransMatrix {
        assert_eq!(self.cols, other.rows);
        let mut result = vec![0.0f32; self.rows * other.cols];
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.data[i * self.cols + k];
                if a == 0.0 { continue; }
                for j in 0..other.cols {
                    result[i * other.cols + j] += a * other.data[k * other.cols + j];
                }
            }
        }
        TransMatrix::new(result, self.rows, other.cols)
    }
}

// ---------------------------------------------------------------------------
// DensityMatrix — quantum density matrix (Complex64, dim x dim)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct DensityMatrix {
    pub data: Vec<Complex64>,
    pub dim: usize,
}

impl DensityMatrix {
    pub fn new(data: Vec<Complex64>, dim: usize) -> Self {
        assert_eq!(data.len(), dim * dim);
        Self { data, dim }
    }

    pub fn zeros(dim: usize) -> Self {
        Self { data: vec![Complex64::new(0.0, 0.0); dim * dim], dim }
    }

    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Complex64 {
        self.data[row * self.dim + col]
    }

    #[inline]
    pub fn set(&mut self, row: usize, col: usize, val: Complex64) {
        self.data[row * self.dim + col] = val;
    }

    /// Create density matrix from pure state: rho = |psi><psi|
    pub fn from_pure_state(psi: &ComplexVec) -> Self {
        let dim = psi.dim();
        let mut data = vec![Complex64::new(0.0, 0.0); dim * dim];
        let norm_sq: f64 = psi.data.iter().map(|c| c.norm_sqr()).sum();
        let scale = if norm_sq > 0.0 { 1.0 / norm_sq } else { 1.0 };

        for i in 0..dim {
            for j in 0..dim {
                data[i * dim + j] = psi.data[i] * psi.data[j].conj() * scale;
            }
        }
        Self { data, dim }
    }

    /// Maximally mixed state: rho = I/dim
    pub fn maximally_mixed(dim: usize) -> Self {
        let mut data = vec![Complex64::new(0.0, 0.0); dim * dim];
        let val = Complex64::new(1.0 / dim as f64, 0.0);
        for i in 0..dim {
            data[i * dim + i] = val;
        }
        Self { data, dim }
    }

    /// Trace of the matrix
    pub fn trace(&self) -> Complex64 {
        let mut tr = Complex64::new(0.0, 0.0);
        for i in 0..self.dim {
            tr += self.data[i * self.dim + i];
        }
        tr
    }

    /// Purity: tr(rho^2)
    pub fn purity(&self) -> f64 {
        // rho^2 = rho @ rho, then trace
        let dim = self.dim;
        let mut tr = Complex64::new(0.0, 0.0);
        for i in 0..dim {
            for k in 0..dim {
                tr += self.data[i * dim + k] * self.data[k * dim + i];
            }
        }
        tr.re
    }

    pub fn is_pure(&self) -> bool {
        self.purity() > 0.999
    }

    /// Diagonal elements (populations)
    pub fn diagonal(&self) -> Vec<f64> {
        (0..self.dim).map(|i| self.data[i * self.dim + i].re).collect()
    }

    /// Matrix multiply: self @ other (density matrix multiplication)
    pub fn matmul(&self, other: &DensityMatrix) -> DensityMatrix {
        assert_eq!(self.dim, other.dim);
        let dim = self.dim;
        let mut result = vec![Complex64::new(0.0, 0.0); dim * dim];
        for i in 0..dim {
            for k in 0..dim {
                let a = self.data[i * dim + k];
                if a.norm_sqr() < 1e-30 { continue; }
                for j in 0..dim {
                    result[i * dim + j] += a * other.data[k * dim + j];
                }
            }
        }
        DensityMatrix::new(result, dim)
    }
}

impl fmt::Display for DensityMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DensityMatrix(dim={}, purity={:.4})", self.dim, self.purity())
    }
}

// ---------------------------------------------------------------------------
// BasinStructure — pure spatial + probabilistic object (time-independent)
// ---------------------------------------------------------------------------

/// A time-independent description of basin structure in phase space.
///
/// Captures the geometry (centroids, radii) and probability distribution
/// (volumes) of basins of attraction without any reference to time evolution.
/// All AXOL theory-level quantities (Ω, Φ) are derived from this structure.
#[derive(Clone, Debug)]
pub struct BasinStructure {
    pub dim: usize,
    pub n_basins: usize,
    pub centroids: Vec<Vec<f64>>,
    pub volumes: Vec<f64>,
    pub fractal_dim: f64,
    pub transform: Option<TransMatrix>,
    pub phases: Vec<f64>,
    pub radii: Vec<f64>,
}

impl BasinStructure {
    /// Construct from dynamics results (implementation → theory bridge).
    pub fn from_dynamics(
        dim: usize,
        centroids: Vec<Vec<f64>>,
        volumes: Vec<f64>,
        fractal_dim: f64,
        transform: Option<TransMatrix>,
        phases: Vec<f64>,
        radii: Vec<f64>,
    ) -> Self {
        let n_basins = centroids.len();
        Self { dim, n_basins, centroids, volumes, fractal_dim, transform, phases, radii }
    }

    /// Construct directly from theory (pure definition, no dynamics needed).
    pub fn from_direct(
        dim: usize,
        centroids: Vec<Vec<f64>>,
        volumes: Vec<f64>,
        fractal_dim: f64,
    ) -> Self {
        let n_basins = centroids.len();
        let phases: Vec<f64> = (0..n_basins)
            .map(|i| i as f64 * 2.0 * std::f64::consts::PI / n_basins.max(1) as f64)
            .collect();
        // Default radii estimated from volumes: V ∝ r^dim
        let radii: Vec<f64> = volumes.iter()
            .map(|&v| v.max(1e-15).powf(1.0 / dim.max(1) as f64))
            .collect();
        Self { dim, n_basins, centroids, volumes, fractal_dim, transform: None, phases, radii }
    }

    /// Shannon entropy of the volume distribution.
    pub fn shannon_entropy(&self) -> f64 {
        let total: f64 = self.volumes.iter().sum();
        if total <= 0.0 || self.n_basins <= 1 {
            return 0.0;
        }
        let mut h = 0.0;
        for &v in &self.volumes {
            let p = v / total;
            if p > 1e-15 {
                h -= p * p.ln();
            }
        }
        h
    }

    /// Ω (omega) — coherence measure, time-independent.
    ///
    /// Ω = 1 - H / ln(k) where H = Shannon entropy, k = number of basins.
    /// Ω = 1 when one basin dominates (fully coherent).
    /// Ω = 0 when all basins have equal volume (maximally mixed).
    pub fn omega(&self) -> f64 {
        if self.n_basins <= 1 {
            return 1.0;
        }
        let h = self.shannon_entropy();
        let h_max = (self.n_basins as f64).ln();
        if h_max <= 0.0 {
            return 1.0;
        }
        (1.0 - h / h_max).clamp(0.0, 1.0)
    }

    /// Φ (phi) — structural complexity from fractal dimension.
    pub fn phi(&self) -> f64 {
        if self.dim == 0 {
            return 1.0;
        }
        1.0 / (1.0 + self.fractal_dim / self.dim as f64)
    }

    /// Assign a point to the nearest basin (hard assignment by Euclidean distance).
    /// Returns basin index.
    pub fn assign_basin(&self, point: &[f64]) -> usize {
        if self.centroids.is_empty() {
            return 0;
        }
        let mut best_idx = 0;
        let mut best_dist = f64::MAX;
        for (i, c) in self.centroids.iter().enumerate() {
            let dist: f64 = c.iter().zip(point.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }
        best_idx
    }

    /// Soft assignment: inverse-distance weighted probabilities over basins.
    /// Returns a weight vector (sums to 1.0) representing proximity to each basin.
    pub fn soft_assignment(&self, point: &[f64]) -> Vec<f64> {
        if self.centroids.is_empty() {
            return vec![1.0];
        }
        let mut weights: Vec<f64> = self.centroids.iter()
            .map(|c| {
                let dist: f64 = c.iter().zip(point.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                1.0 / (1.0 + dist * 5.0)
            })
            .collect();
        let total: f64 = weights.iter().sum();
        if total > 1e-15 {
            for w in weights.iter_mut() {
                *w /= total;
            }
        }
        weights
    }

    /// KL divergence between this basin's volume distribution and another.
    /// D_KL(self || other).
    pub fn kl_divergence(&self, other: &BasinStructure) -> f64 {
        let n = self.volumes.len().min(other.volumes.len());
        if n == 0 { return 0.0; }
        let self_total: f64 = self.volumes.iter().sum();
        let other_total: f64 = other.volumes.iter().sum();
        if self_total <= 0.0 || other_total <= 0.0 { return f64::MAX; }

        let mut kl = 0.0;
        for i in 0..n {
            let p = self.volumes[i] / self_total;
            let q = other.volumes[i] / other_total;
            if p > 1e-15 && q > 1e-15 {
                kl += p * (p / q).ln();
            } else if p > 1e-15 {
                kl += 10.0; // large penalty for q=0 where p>0
            }
        }
        kl
    }

    /// Total variation distance between two basin volume distributions.
    /// TV(P, Q) = 0.5 * Σ|p_i - q_i|, in [0, 1].
    pub fn tv_distance(&self, other: &BasinStructure) -> f64 {
        let n = self.volumes.len().max(other.volumes.len());
        if n == 0 { return 0.0; }
        let self_total: f64 = self.volumes.iter().sum();
        let other_total: f64 = other.volumes.iter().sum();

        let mut tv = 0.0;
        for i in 0..n {
            let p = if i < self.volumes.len() && self_total > 0.0 {
                self.volumes[i] / self_total
            } else {
                0.0
            };
            let q = if i < other.volumes.len() && other_total > 0.0 {
                other.volumes[i] / other_total
            } else {
                0.0
            };
            tv += (p - q).abs();
        }
        tv * 0.5
    }
}

impl fmt::Display for BasinStructure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BasinStructure(dim={}, basins={}, Ω={:.4}, Φ={:.4})",
            self.dim, self.n_basins, self.omega(), self.phi())
    }
}
