//! Density matrix operations: entropy, fidelity, quantum channels, Kraus operators.

use num_complex::Complex64;
use crate::types::*;
use crate::errors::Result;

// ---------------------------------------------------------------------------
// Entropy and information measures
// ---------------------------------------------------------------------------

/// Von Neumann entropy: S(rho) = -tr(rho log rho)
/// Uses eigenvalue decomposition (Jacobi iteration for Hermitian matrices).
pub fn von_neumann_entropy(rho: &DensityMatrix) -> f64 {
    let eigenvalues = hermitian_eigenvalues(&rho.data, rho.dim);
    let mut entropy = 0.0;
    for &ev in &eigenvalues {
        if ev > 1e-15 {
            entropy -= ev * ev.ln();
        }
    }
    entropy
}

/// Quantum fidelity: F(rho1, rho2) = (tr sqrt(sqrt(rho1) rho2 sqrt(rho1)))^2
/// Simplified for pure states or diagonal-dominant cases.
pub fn fidelity(rho1: &DensityMatrix, rho2: &DensityMatrix) -> f64 {
    assert_eq!(rho1.dim, rho2.dim);
    let dim = rho1.dim;

    // sqrt(rho1) via eigendecomposition
    let (eigvals, eigvecs) = hermitian_eigen(&rho1.data, dim);
    let mut sqrt_rho1 = vec![Complex64::new(0.0, 0.0); dim * dim];
    for i in 0..dim {
        let sqrt_ev = if eigvals[i] > 0.0 { eigvals[i].sqrt() } else { 0.0 };
        for r in 0..dim {
            for c in 0..dim {
                sqrt_rho1[r * dim + c] += Complex64::new(sqrt_ev, 0.0)
                    * eigvecs[r * dim + i]
                    * eigvecs[c * dim + i].conj();
            }
        }
    }

    // product = sqrt_rho1 @ rho2 @ sqrt_rho1
    let temp = mat_mul_complex(&sqrt_rho1, &rho2.data, dim);
    let product = mat_mul_complex(&temp, &sqrt_rho1, dim);

    let product_eigvals = hermitian_eigenvalues(&product, dim);
    let sum_sqrt: f64 = product_eigvals.iter()
        .map(|&ev| if ev > 0.0 { ev.sqrt() } else { 0.0 })
        .sum();

    sum_sqrt * sum_sqrt
}

// ---------------------------------------------------------------------------
// Quantum channels (Kraus operators)
// ---------------------------------------------------------------------------

/// Apply quantum channel: epsilon(rho) = sum_k E_k rho E_k†
pub fn apply_channel(rho: &DensityMatrix, kraus_ops: &[Vec<Complex64>]) -> DensityMatrix {
    let dim = rho.dim;
    let mut result = vec![Complex64::new(0.0, 0.0); dim * dim];

    for e in kraus_ops {
        // E @ rho
        let e_rho = mat_mul_complex(e, &rho.data, dim);
        // (E @ rho) @ E†
        for i in 0..dim {
            for j in 0..dim {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..dim {
                    sum += e_rho[i * dim + k] * e[j * dim + k].conj();
                }
                result[i * dim + j] += sum;
            }
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

    DensityMatrix::new(result, dim)
}

/// Depolarizing channel: epsilon(rho) = (1-p) rho + (p/dim) I
pub fn depolarizing_channel(dim: usize, p: f64) -> Vec<Vec<Complex64>> {
    let mut kraus = Vec::new();

    // E0 = sqrt(1-p) * I
    let mut e0 = vec![Complex64::new(0.0, 0.0); dim * dim];
    let s0 = (1.0 - p).sqrt();
    for i in 0..dim {
        e0[i * dim + i] = Complex64::new(s0, 0.0);
    }
    kraus.push(e0);

    // E_ij = sqrt(p/dim) * |i><j|
    let scale = (p / dim as f64).sqrt();
    for i in 0..dim {
        for j in 0..dim {
            let mut e = vec![Complex64::new(0.0, 0.0); dim * dim];
            e[i * dim + j] = Complex64::new(scale, 0.0);
            kraus.push(e);
        }
    }

    kraus
}

/// Amplitude damping channel
pub fn amplitude_damping_channel(gamma: f64, dim: usize) -> Vec<Vec<Complex64>> {
    let mut kraus = Vec::new();

    // E0: diagonal with 1 at (0,0), sqrt(1-gamma) elsewhere
    let mut e0 = vec![Complex64::new(0.0, 0.0); dim * dim];
    e0[0] = Complex64::new(1.0, 0.0);
    for k in 1..dim {
        e0[k * dim + k] = Complex64::new((1.0 - gamma).sqrt(), 0.0);
    }
    kraus.push(e0);

    // Decay operators
    for k in 1..dim {
        let mut ek = vec![Complex64::new(0.0, 0.0); dim * dim];
        ek[0 * dim + k] = Complex64::new(gamma.sqrt(), 0.0);
        kraus.push(ek);
    }

    kraus
}

/// Dephasing channel: preserves populations, damps coherences by (1-gamma)
pub fn dephasing_channel(gamma: f64, dim: usize) -> Vec<Vec<Complex64>> {
    let mut kraus = Vec::new();

    // E0 = sqrt(1-gamma) * I
    let mut e0 = vec![Complex64::new(0.0, 0.0); dim * dim];
    let s0 = (1.0 - gamma).sqrt();
    for i in 0..dim {
        e0[i * dim + i] = Complex64::new(s0, 0.0);
    }
    kraus.push(e0);

    // E_k = sqrt(gamma) * |k><k|
    let scale = gamma.sqrt();
    for k in 0..dim {
        let mut ek = vec![Complex64::new(0.0, 0.0); dim * dim];
        ek[k * dim + k] = Complex64::new(scale, 0.0);
        kraus.push(ek);
    }

    kraus
}

/// Convert SVD components to Kraus operators (bridge from Hybrid composition)
pub fn svd_to_kraus(u: &[f64], sigma: &[f64], vh: &[f64], dim: usize) -> Vec<Vec<Complex64>> {
    let mut kraus = Vec::new();

    // Clamp sigma to [0, 1]
    let sigma_clamped: Vec<f64> = sigma.iter().take(dim).map(|&s| s.clamp(0.0, 1.0)).collect();

    // Rotation = U @ Vh (top-left dim x dim blocks)
    let mut rotation = vec![Complex64::new(0.0, 0.0); dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let mut sum = 0.0;
            for k in 0..dim {
                sum += u[i * dim + k] * vh[k * dim + j];
            }
            rotation[i * dim + j] = Complex64::new(sum, 0.0);
        }
    }

    // E0 = rotation @ diag(sigma_clamped)
    let mut e0 = vec![Complex64::new(0.0, 0.0); dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            e0[i * dim + j] = rotation[i * dim + j] * Complex64::new(sigma_clamped[j], 0.0);
        }
    }
    kraus.push(e0);

    // Decay operators for damped modes
    for i in 0..dim.min(sigma_clamped.len()) {
        if sigma_clamped[i] < 1.0 - 1e-10 {
            let gamma_i = 1.0 - sigma_clamped[i] * sigma_clamped[i];
            let mut ek_base = vec![Complex64::new(0.0, 0.0); dim * dim];
            ek_base[i * dim + i] = Complex64::new(gamma_i.sqrt(), 0.0);
            // Apply rotation
            let mut ek = vec![Complex64::new(0.0, 0.0); dim * dim];
            for r in 0..dim {
                for c in 0..dim {
                    for k in 0..dim {
                        ek[r * dim + c] += rotation[r * dim + k] * ek_base[k * dim + c];
                    }
                }
            }
            kraus.push(ek);
        }
    }

    kraus
}

// ---------------------------------------------------------------------------
// Quality metrics from quantum state
// ---------------------------------------------------------------------------

/// Phi (clarity) from purity: maps tr(rho^2) in [1/dim, 1] -> [0, 1]
pub fn phi_from_purity(rho: &DensityMatrix) -> f64 {
    let p = rho.purity();
    let dim = rho.dim;
    let min_purity = 1.0 / dim as f64;
    if p <= min_purity {
        return 0.0;
    }
    ((p - min_purity) / (1.0 - min_purity)).min(1.0)
}

/// Omega (cohesion) from off-diagonal coherence
pub fn omega_from_coherence(rho: &DensityMatrix) -> f64 {
    let dim = rho.dim;
    let mut coherence = 0.0;
    for i in 0..dim {
        for j in 0..dim {
            if i != j {
                coherence += rho.get(i, j).norm();
            }
        }
    }
    let max_coherence = (dim as f64 - 1.0).max(1.0);
    (coherence / max_coherence).min(1.0)
}

// ---------------------------------------------------------------------------
// Linear algebra helpers (pure Rust — no external BLAS)
// ---------------------------------------------------------------------------

/// Complex matrix multiply: A @ B (dim x dim)
fn mat_mul_complex(a: &[Complex64], b: &[Complex64], dim: usize) -> Vec<Complex64> {
    let mut result = vec![Complex64::new(0.0, 0.0); dim * dim];
    for i in 0..dim {
        for k in 0..dim {
            let aik = a[i * dim + k];
            if aik.norm_sqr() < 1e-30 { continue; }
            for j in 0..dim {
                result[i * dim + j] += aik * b[k * dim + j];
            }
        }
    }
    result
}

/// Eigenvalues of a Hermitian matrix (Jacobi iteration)
fn hermitian_eigenvalues(data: &[Complex64], dim: usize) -> Vec<f64> {
    let (eigvals, _) = hermitian_eigen(data, dim);
    eigvals
}

/// Eigenvalues and eigenvectors of a Hermitian matrix (Jacobi-like)
/// Returns (eigenvalues, eigenvectors as column-major dim x dim)
fn hermitian_eigen(data: &[Complex64], dim: usize) -> (Vec<f64>, Vec<Complex64>) {
    // For small matrices, use a simple approach:
    // Convert to real symmetric via diagonalization trick, or use QR iteration.
    // For correctness, we use a simple power iteration / deflation approach.

    // Actually, for Hermitian matrices, we'll use Jacobi rotation.
    // For simplicity and correctness, let's use a direct approach for small dims.

    let mut a = data.to_vec();
    let mut v = vec![Complex64::new(0.0, 0.0); dim * dim];
    for i in 0..dim {
        v[i * dim + i] = Complex64::new(1.0, 0.0);
    }

    // Jacobi iteration for Hermitian matrices
    let max_iter = 100 * dim * dim;
    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0;
        let mut p = 0;
        let mut q = 1;
        for i in 0..dim {
            for j in (i + 1)..dim {
                let val = a[i * dim + j].norm();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < 1e-12 {
            break;
        }

        // Compute Jacobi rotation
        let app = a[p * dim + p].re;
        let aqq = a[q * dim + q].re;
        let apq = a[p * dim + q];

        let tau = (aqq - app) / (2.0 * apq.norm());
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };

        let c = 1.0 / (1.0 + t * t).sqrt();
        let s_val = t * c;
        let phase = if apq.norm() > 0.0 { apq / apq.norm() } else { Complex64::new(1.0, 0.0) };
        let s = Complex64::new(s_val, 0.0) * phase.conj();

        // Apply rotation to A
        let mut new_a = a.clone();
        for i in 0..dim {
            if i != p && i != q {
                let aip = a[i * dim + p];
                let aiq = a[i * dim + q];
                new_a[i * dim + p] = Complex64::new(c, 0.0) * aip + s.conj() * aiq;
                new_a[i * dim + q] = Complex64::new(c, 0.0) * aiq - s * aip;
                new_a[p * dim + i] = new_a[i * dim + p].conj();
                new_a[q * dim + i] = new_a[i * dim + q].conj();
            }
        }
        new_a[p * dim + p] = Complex64::new(app + t * apq.norm(), 0.0);
        new_a[q * dim + q] = Complex64::new(aqq - t * apq.norm(), 0.0);
        new_a[p * dim + q] = Complex64::new(0.0, 0.0);
        new_a[q * dim + p] = Complex64::new(0.0, 0.0);
        a = new_a;

        // Apply rotation to eigenvectors
        let mut new_v = v.clone();
        for i in 0..dim {
            let vip = v[i * dim + p];
            let viq = v[i * dim + q];
            new_v[i * dim + p] = Complex64::new(c, 0.0) * vip + s.conj() * viq;
            new_v[i * dim + q] = Complex64::new(c, 0.0) * viq - s * vip;
        }
        v = new_v;
    }

    let eigenvalues: Vec<f64> = (0..dim).map(|i| a[i * dim + i].re).collect();
    (eigenvalues, v)
}
