//! Core AXOL types: FloatVec, ComplexVec, TransMatrix, DensityMatrix.

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
