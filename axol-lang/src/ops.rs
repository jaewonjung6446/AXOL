//! Core operations: transform, measure, Born rule, interference.

use num_complex::Complex64;
use crate::types::*;
use crate::errors::{AxolError, Result};

// ---------------------------------------------------------------------------
// Real-valued operations
// ---------------------------------------------------------------------------

/// Matrix-vector multiply: vec @ matrix (row-vector convention)
pub fn transform(vec: &FloatVec, matrix: &TransMatrix) -> Result<FloatVec> {
    if vec.dim() != matrix.rows {
        return Err(AxolError::DimensionMismatch {
            expected: matrix.rows,
            got: vec.dim(),
        });
    }
    let mut result = vec![0.0f32; matrix.cols];
    for j in 0..matrix.cols {
        let mut sum = 0.0f32;
        for i in 0..matrix.rows {
            sum += vec.data[i] * matrix.data[i * matrix.cols + j];
        }
        result[j] = sum;
    }
    Ok(FloatVec::new(result))
}

/// Born rule on real amplitudes: |alpha_i|^2 / sum
pub fn measure(vec: &FloatVec) -> FloatVec {
    let probs: Vec<f32> = vec.data.iter().map(|x| x * x).collect();
    let total: f32 = probs.iter().sum();
    if total > 0.0 {
        FloatVec::new(probs.iter().map(|p| p / total).collect())
    } else {
        let n = probs.len();
        FloatVec::new(vec![1.0 / n as f32; n])
    }
}

/// Argmax of a float vector
pub fn argmax(vec: &FloatVec) -> usize {
    vec.data
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Complex-valued operations
// ---------------------------------------------------------------------------

/// Complex matrix-vector multiply: vec @ matrix
pub fn transform_complex(vec: &ComplexVec, matrix: &TransMatrix) -> Result<ComplexVec> {
    if vec.dim() != matrix.rows {
        return Err(AxolError::DimensionMismatch {
            expected: matrix.rows,
            got: vec.dim(),
        });
    }
    let mut result = vec![Complex64::new(0.0, 0.0); matrix.cols];
    for j in 0..matrix.cols {
        let mut sum = Complex64::new(0.0, 0.0);
        for i in 0..matrix.rows {
            sum += vec.data[i] * Complex64::new(matrix.data[i * matrix.cols + j] as f64, 0.0);
        }
        result[j] = sum;
    }
    Ok(ComplexVec::new(result))
}

/// Born rule on complex amplitudes: |alpha_i|^2 / sum
pub fn measure_complex(vec: &ComplexVec) -> FloatVec {
    let probs: Vec<f32> = vec.data.iter().map(|c| c.norm_sqr() as f32).collect();
    let total: f32 = probs.iter().sum();
    if total > 0.0 {
        FloatVec::new(probs.iter().map(|p| p / total).collect())
    } else {
        let n = probs.len();
        FloatVec::new(vec![1.0 / n as f32; n])
    }
}

/// Quantum interference: a + exp(i*phase) * b, normalized
pub fn interfere(a: &ComplexVec, b: &ComplexVec, phase: f64) -> Result<ComplexVec> {
    if a.dim() != b.dim() {
        return Err(AxolError::DimensionMismatch {
            expected: a.dim(),
            got: b.dim(),
        });
    }
    let exp_phase = Complex64::from_polar(1.0, phase);
    let result: Vec<Complex64> = a.data.iter()
        .zip(b.data.iter())
        .map(|(ai, bi)| ai + exp_phase * bi)
        .collect();

    let norm: f64 = result.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
    if norm > 0.0 {
        Ok(ComplexVec::new(result.iter().map(|c| c / norm).collect()))
    } else {
        Ok(ComplexVec::new(result))
    }
}

/// Unitary evolution of density matrix: U rho U†
pub fn evolve_density(rho: &DensityMatrix, u: &TransMatrix) -> Result<DensityMatrix> {
    if u.rows != rho.dim || u.cols != rho.dim {
        return Err(AxolError::DimensionMismatch {
            expected: rho.dim,
            got: u.rows,
        });
    }
    let dim = rho.dim;

    // U @ rho
    let mut u_rho = vec![Complex64::new(0.0, 0.0); dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let mut sum = Complex64::new(0.0, 0.0);
            for k in 0..dim {
                sum += Complex64::new(u.get(i, k) as f64, 0.0) * rho.get(k, j);
            }
            u_rho[i * dim + j] = sum;
        }
    }

    // (U @ rho) @ U†
    let mut result = vec![Complex64::new(0.0, 0.0); dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let mut sum = Complex64::new(0.0, 0.0);
            for k in 0..dim {
                sum += u_rho[i * dim + k] * Complex64::new(u.get(j, k) as f64, 0.0); // U† = U^T for real U
            }
            result[i * dim + j] = sum;
        }
    }

    // Hermitian symmetrize
    for i in 0..dim {
        for j in (i + 1)..dim {
            let avg = (result[i * dim + j] + result[j * dim + i].conj()) * 0.5;
            result[i * dim + j] = avg;
            result[j * dim + i] = avg.conj();
        }
    }

    Ok(DensityMatrix::new(result, dim))
}

/// Partial trace: trace out subsystem B from a bipartite system
pub fn partial_trace(rho: &DensityMatrix, dim_a: usize, dim_b: usize, trace_out_b: bool) -> Result<DensityMatrix> {
    if dim_a * dim_b != rho.dim {
        return Err(AxolError::DimensionMismatch {
            expected: dim_a * dim_b,
            got: rho.dim,
        });
    }

    if trace_out_b {
        // Trace out B, keep A (dim_a x dim_a)
        let mut result = vec![Complex64::new(0.0, 0.0); dim_a * dim_a];
        for i in 0..dim_a {
            for j in 0..dim_a {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..dim_b {
                    sum += rho.get(i * dim_b + k, j * dim_b + k);
                }
                result[i * dim_a + j] = sum;
            }
        }
        Ok(DensityMatrix::new(result, dim_a))
    } else {
        // Trace out A, keep B (dim_b x dim_b)
        let mut result = vec![Complex64::new(0.0, 0.0); dim_b * dim_b];
        for i in 0..dim_b {
            for j in 0..dim_b {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..dim_a {
                    sum += rho.get(k * dim_b + i, k * dim_b + j);
                }
                result[i * dim_b + j] = sum;
            }
        }
        Ok(DensityMatrix::new(result, dim_b))
    }
}
