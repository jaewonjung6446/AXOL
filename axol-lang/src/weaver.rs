//! Weaver — transforms declarations into tapestries via REAL chaos dynamics.
//!
//! Uses ChaosEngine (coupled logistic map lattice) to:
//!   1. Run iterative dynamics → find attractor
//!   2. Compute real Lyapunov exponents and fractal dimension
//!   3. Derive transformation matrix from attractor linearization
//!   4. Detect basins of attraction for quantum branching

use num_complex::Complex64;
use rand::SeedableRng;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

use crate::types::*;
use crate::ops;
use crate::declare::*;
use crate::dynamics::{ChaosEngine, Basin};
use crate::errors::{AxolError, Result};

// ---------------------------------------------------------------------------
// Attractor
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Attractor {
    pub phase_space_dim: usize,
    pub fractal_dim: f64,
    pub max_lyapunov: f64,
    pub lyapunov_spectrum: Vec<f64>,
    pub trajectory_matrix: TransMatrix,
}

impl Attractor {
    pub fn omega(&self) -> f64 {
        1.0 / (1.0 + self.max_lyapunov.max(0.0))
    }

    pub fn phi(&self) -> f64 {
        if self.phase_space_dim == 0 {
            return 1.0;
        }
        1.0 / (1.0 + self.fractal_dim / self.phase_space_dim as f64)
    }
}

// ---------------------------------------------------------------------------
// TapestryNode
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct TapestryNode {
    pub name: String,
    pub amplitudes: FloatVec,
    pub complex_amplitudes: Option<ComplexVec>,
    pub labels: HashMap<usize, String>,
    pub attractor: Attractor,
    pub depth: usize,
}

// ---------------------------------------------------------------------------
// WeaverReport
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct WeaverReport {
    pub target_omega: f64,
    pub target_phi: f64,
    pub estimated_omega: f64,
    pub estimated_phi: f64,
    pub max_lyapunov: f64,
    pub fractal_dim: f64,
    pub feasible: bool,
    pub warnings: Vec<String>,
}

// ---------------------------------------------------------------------------
// Tapestry
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Tapestry {
    pub name: String,
    pub nodes: HashMap<String, TapestryNode>,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    pub global_attractor: Attractor,
    pub report: WeaverReport,
    pub composed_matrix: Option<TransMatrix>,
    pub quantum: bool,
    pub density_matrix: Option<DensityMatrix>,
    pub kraus_operators: Option<Vec<Vec<Complex64>>>,
    // New: real dynamics
    pub chaos_engine: Option<ChaosEngine>,
    pub basins: Option<Vec<Basin>>,
}

// ---------------------------------------------------------------------------
// Weave
// ---------------------------------------------------------------------------

pub fn weave(decl: &EntangleDeclaration, quantum: bool, seed: u64) -> Result<Tapestry> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    if decl.inputs.is_empty() {
        return Err(AxolError::Weaver("No inputs declared".into()));
    }
    if decl.relations.is_empty() {
        return Err(AxolError::Weaver("No relations declared".into()));
    }

    let dim = decl.inputs[0].dim;
    let input_names: Vec<String> = decl.inputs.iter().map(|i| i.name.clone()).collect();
    let output_names = decl.outputs.clone();

    // ===================================================================
    // STEP 1: Build ChaosEngine and run REAL dynamics
    // ===================================================================
    let chaos = ChaosEngine::from_declaration(decl, seed);
    let attractor_result = chaos.find_attractor(seed, 300, 500);

    // Transformation matrix derived from attractor linearization
    let composed_matrix = chaos.extract_matrix(&attractor_result);

    // Global attractor with REAL computed values
    let global_attractor = Attractor {
        phase_space_dim: dim,
        fractal_dim: attractor_result.fractal_dim,
        max_lyapunov: attractor_result.max_lyapunov,
        lyapunov_spectrum: attractor_result.lyapunov_spectrum.clone(),
        trajectory_matrix: composed_matrix.clone(),
    };

    // ===================================================================
    // STEP 2: Create input/output nodes
    // ===================================================================
    let mut nodes: HashMap<String, TapestryNode> = HashMap::new();

    for inp in &decl.inputs {
        let mut amps = vec![0.0f32; inp.dim];
        for v in amps.iter_mut() {
            *v = rng.gen::<f32>() * 2.0 - 1.0;
        }
        let fv = FloatVec::new(amps);

        let complex_amps = if quantum {
            let mags: Vec<f64> = fv.data.iter().map(|&x| (x as f64).abs()).collect();
            let phases: Vec<f64> = (0..inp.dim)
                .map(|_| rng.gen::<f64>() * 2.0 * std::f64::consts::PI)
                .collect();
            Some(ComplexVec::from_polar(&mags, &phases))
        } else {
            None
        };

        nodes.insert(inp.name.clone(), TapestryNode {
            name: inp.name.clone(),
            amplitudes: fv,
            complex_amplitudes: complex_amps,
            labels: inp.labels.clone(),
            attractor: global_attractor.clone(),
            depth: 0,
        });
    }

    // Process relations to create intermediate/output nodes
    for rel in &decl.relations {
        let source_dim = if let Some(src) = rel.sources.first() {
            nodes.get(src).map(|n| n.amplitudes.dim()).unwrap_or(dim)
        } else {
            dim
        };

        let mut combined = FloatVec::zeros(source_dim);
        for src_name in &rel.sources {
            if let Some(src_node) = nodes.get(src_name) {
                if let Ok(transformed) = ops::transform(&src_node.amplitudes, &composed_matrix) {
                    for (i, v) in combined.data.iter_mut().enumerate() {
                        if i < transformed.data.len() {
                            *v += transformed.data[i];
                        }
                    }
                }
            }
        }

        let complex_amps = if quantum {
            let mags: Vec<f64> = combined.data.iter().map(|&x| (x as f64).abs()).collect();
            let phases: Vec<f64> = (0..source_dim)
                .map(|_| rng.gen::<f64>() * 2.0 * std::f64::consts::PI)
                .collect();
            Some(ComplexVec::from_polar(&mags, &phases))
        } else {
            None
        };

        let depth = rel.sources.iter()
            .filter_map(|s| nodes.get(s))
            .map(|n| n.depth)
            .max()
            .unwrap_or(0) + 1;

        nodes.insert(rel.target.clone(), TapestryNode {
            name: rel.target.clone(),
            amplitudes: combined,
            complex_amplitudes: complex_amps,
            labels: HashMap::new(),
            attractor: global_attractor.clone(),
            depth,
        });
    }

    // ===================================================================
    // STEP 3: Basin detection (for quantum branching)
    // ===================================================================
    let basins = if quantum {
        Some(chaos.find_basins(200, 300, seed))
    } else {
        None
    };

    // ===================================================================
    // STEP 4: Quantum structures
    // ===================================================================
    let density_matrix = if quantum {
        // Create superposition from basins
        if let Some(ref basins) = basins {
            if !basins.is_empty() {
                let cv = basin_superposition(basins, dim);
                Some(DensityMatrix::from_pure_state(&cv))
            } else {
                fallback_density(&nodes, &output_names)
            }
        } else {
            fallback_density(&nodes, &output_names)
        }
    } else {
        None
    };

    // ===================================================================
    // STEP 5: Weaver report with REAL metrics
    // ===================================================================
    let report = WeaverReport {
        target_omega: decl.quality.omega,
        target_phi: decl.quality.phi,
        estimated_omega: global_attractor.omega(),
        estimated_phi: global_attractor.phi(),
        max_lyapunov: attractor_result.max_lyapunov,
        fractal_dim: attractor_result.fractal_dim,
        feasible: global_attractor.omega() >= decl.quality.omega * 0.5
            && global_attractor.phi() >= decl.quality.phi * 0.5,
        warnings: Vec::new(),
    };

    Ok(Tapestry {
        name: decl.name.clone(),
        nodes,
        input_names,
        output_names,
        global_attractor,
        report,
        composed_matrix: Some(composed_matrix),
        quantum,
        density_matrix,
        kraus_operators: None,
        chaos_engine: Some(chaos),
        basins,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create quantum superposition from basin structure.
/// Each basin contributes a component with amplitude ∝ sqrt(basin size)
/// and phase from basin geometry.
fn basin_superposition(basins: &[Basin], dim: usize) -> ComplexVec {
    let n_basins = basins.len();
    let mut data = vec![Complex64::new(0.0, 0.0); dim];

    for (idx, basin) in basins.iter().enumerate() {
        let amplitude = basin.size.sqrt();
        let phase = basin.phase;
        let component = Complex64::from_polar(amplitude, phase);

        // Map basin center to state vector dimensions
        for d in 0..dim.min(basin.center.len()) {
            let weight = basin.center[d];
            data[d] += component * Complex64::new(weight, 0.0);
        }
        // For remaining dims, distribute based on basin index
        for d in basin.center.len()..dim {
            let weight = ((idx * dim + d) as f64 * 0.618).sin().abs();
            data[d] += component * Complex64::new(weight, 0.0);
        }
    }

    // Normalize
    let norm: f64 = data.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
    if norm > 1e-15 {
        for c in data.iter_mut() {
            *c /= norm;
        }
    }
    ComplexVec::new(data)
}

/// Fallback: density matrix from output node (if no basins).
fn fallback_density(
    nodes: &HashMap<String, TapestryNode>,
    output_names: &[String],
) -> Option<DensityMatrix> {
    if let Some(output_name) = output_names.first() {
        if let Some(node) = nodes.get(output_name) {
            if let Some(ref cv) = node.complex_amplitudes {
                return Some(DensityMatrix::from_pure_state(cv));
            }
        }
    }
    None
}
