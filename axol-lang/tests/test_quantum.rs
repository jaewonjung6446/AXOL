//! Quantum feature tests for AXOL-Rust
//! "Incomplete logic" style — exploratory verification of each quantum subsystem.

use num_complex::Complex64;
use std::f64::consts::PI;

use axol::types::*;
use axol::ops;
use axol::density;
use axol::declare::*;
use axol::weaver;
use axol::observatory;

// =========================================================================
// 1. ComplexVec basics
// =========================================================================

#[test]
fn complexvec_from_real() {
    let fv = FloatVec::new(vec![3.0, 4.0]);
    let cv = ComplexVec::from_real(&fv);
    assert_eq!(cv.dim(), 2);
    assert!((cv.data[0].re - 3.0).abs() < 1e-10);
    assert!((cv.data[1].re - 4.0).abs() < 1e-10);
    assert!((cv.data[0].im).abs() < 1e-10);
}

#[test]
fn complexvec_from_polar() {
    let mags = vec![1.0, 1.0];
    let phases = vec![0.0, PI / 2.0];
    let cv = ComplexVec::from_polar(&mags, &phases);
    // First: 1*e^{i*0} = 1+0i
    assert!((cv.data[0].re - 1.0).abs() < 1e-10);
    assert!((cv.data[0].im).abs() < 1e-10);
    // Second: 1*e^{i*pi/2} = 0+1i
    assert!((cv.data[1].re).abs() < 1e-10);
    assert!((cv.data[1].im - 1.0).abs() < 1e-10);
}

#[test]
fn complexvec_normalized() {
    let cv = ComplexVec::new(vec![
        Complex64::new(3.0, 0.0),
        Complex64::new(0.0, 4.0),
    ]);
    let n = cv.normalized();
    let norm: f64 = n.data.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
    assert!((norm - 1.0).abs() < 1e-10);
}

// =========================================================================
// 2. Born rule (measure_complex)
// =========================================================================

#[test]
fn born_rule_uniform() {
    // |+> = (1,1)/sqrt(2)  →  probs = (0.5, 0.5)
    let cv = ComplexVec::new(vec![
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
    ]);
    let probs = ops::measure_complex(&cv);
    assert!((probs.data[0] - 0.5).abs() < 1e-5);
    assert!((probs.data[1] - 0.5).abs() < 1e-5);
}

#[test]
fn born_rule_peaked() {
    // |0> = (1,0)  →  probs = (1.0, 0.0)
    let cv = ComplexVec::new(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    ]);
    let probs = ops::measure_complex(&cv);
    assert!((probs.data[0] - 1.0).abs() < 1e-5);
    assert!((probs.data[1]).abs() < 1e-5);
}

#[test]
fn born_rule_phase_invariant() {
    // e^{i*pi/3} * |0>  should give same probs as |0>
    let phase = Complex64::from_polar(1.0, PI / 3.0);
    let cv = ComplexVec::new(vec![phase, Complex64::new(0.0, 0.0)]);
    let probs = ops::measure_complex(&cv);
    assert!((probs.data[0] - 1.0).abs() < 1e-5);
}

// =========================================================================
// 3. Interference
// =========================================================================

#[test]
fn constructive_interference() {
    let a = ComplexVec::new(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);
    let b = ComplexVec::new(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);
    // phase=0 → constructive
    let result = ops::interfere(&a, &b, 0.0).unwrap();
    let probs = ops::measure_complex(&result);
    assert!(probs.data[0] > 0.99, "constructive: first component should dominate");
}

#[test]
fn destructive_interference() {
    // a = (1, 0), b = (0, 1), phase=PI → (1, -1)/sqrt(2) — partial destructive
    let a = ComplexVec::new(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);
    let b = ComplexVec::new(vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]);
    let result = ops::interfere(&a, &b, PI).unwrap();
    let probs = ops::measure_complex(&result);
    // Both components should have equal probability (phases differ but |amplitude| same)
    assert!((probs.data[0] - 0.5).abs() < 0.1, "destructive: equal magnitudes");
    assert!((probs.data[1] - 0.5).abs() < 0.1);
}

#[test]
fn interference_creates_asymmetry() {
    // a = (1, 0), b = (0, 1), phase=0 → (1, 1)/sqrt(2)
    let a = ComplexVec::new(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);
    let b = ComplexVec::new(vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]);
    let result = ops::interfere(&a, &b, 0.0).unwrap();
    let probs = ops::measure_complex(&result);
    // Should be approximately equal
    assert!((probs.data[0] - 0.5).abs() < 0.1);
    assert!((probs.data[1] - 0.5).abs() < 0.1);
}

// =========================================================================
// 4. Hadamard transform (complex transform)
// =========================================================================

#[test]
fn hadamard_transform() {
    // H|+> = |0>
    // |+> = (1,1)/sqrt(2), H = [[1,1],[1,-1]]/sqrt(2)
    let s2 = 2.0_f32.sqrt();
    let plus = ComplexVec::new(vec![
        Complex64::new(1.0 / (s2 as f64), 0.0),
        Complex64::new(1.0 / (s2 as f64), 0.0),
    ]);
    let h = TransMatrix::new(vec![1.0/s2, 1.0/s2, 1.0/s2, -1.0/s2], 2, 2);
    let result = ops::transform_complex(&plus, &h).unwrap();
    let probs = ops::measure_complex(&result);
    assert!(probs.data[0] > 0.9, "H|+> should collapse to |0>");
    assert!(probs.data[1] < 0.1);
}

// =========================================================================
// 5. DensityMatrix
// =========================================================================

#[test]
fn pure_state_density() {
    let psi = ComplexVec::new(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    ]);
    let rho = DensityMatrix::from_pure_state(&psi);
    assert!((rho.trace().re - 1.0).abs() < 1e-10, "tr(rho)=1");
    assert!((rho.purity() - 1.0).abs() < 1e-10, "pure state purity=1");
    assert!(rho.is_pure());
}

#[test]
fn superposition_density() {
    let psi = ComplexVec::new(vec![
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
    ]);
    let rho = DensityMatrix::from_pure_state(&psi);
    assert!((rho.trace().re - 1.0).abs() < 1e-10);
    assert!((rho.purity() - 1.0).abs() < 1e-10, "superposition is still a pure state");
    // Off-diagonal should be nonzero (coherence)
    assert!(rho.get(0, 1).norm() > 0.4, "coherence should exist");
}

#[test]
fn mixed_state_density() {
    let rho = DensityMatrix::maximally_mixed(2);
    assert!((rho.trace().re - 1.0).abs() < 1e-10);
    assert!((rho.purity() - 0.5).abs() < 1e-10, "maximally mixed 2D: purity=0.5");
    assert!(!rho.is_pure());
}

#[test]
fn density_diagonal() {
    let psi = ComplexVec::new(vec![
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
    ]);
    let rho = DensityMatrix::from_pure_state(&psi);
    let diag = rho.diagonal();
    assert!((diag[0] - 0.5).abs() < 1e-10);
    assert!((diag[1] - 0.5).abs() < 1e-10);
}

// =========================================================================
// 6. Quantum channels
// =========================================================================

#[test]
fn dephasing_preserves_trace() {
    let psi = ComplexVec::new(vec![
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
    ]);
    let rho = DensityMatrix::from_pure_state(&psi);

    for &gamma in &[0.0, 0.1, 0.5, 0.9, 1.0] {
        let kraus = density::dephasing_channel(gamma, 2);
        let rho_out = density::apply_channel(&rho, &kraus);
        let tr = rho_out.trace().re;
        assert!(
            (tr - 1.0).abs() < 1e-8,
            "dephasing gamma={gamma}: tr={tr}, expected 1.0"
        );
    }
}

#[test]
fn dephasing_kills_coherence() {
    let psi = ComplexVec::new(vec![
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
    ]);
    let rho = DensityMatrix::from_pure_state(&psi);

    // gamma=1.0 → full dephasing → off-diagonal = 0
    let kraus = density::dephasing_channel(1.0, 2);
    let rho_out = density::apply_channel(&rho, &kraus);
    assert!(
        rho_out.get(0, 1).norm() < 1e-8,
        "full dephasing should kill off-diag coherence"
    );
    // Populations preserved
    assert!((rho_out.get(0, 0).re - 0.5).abs() < 1e-8);
    assert!((rho_out.get(1, 1).re - 0.5).abs() < 1e-8);
}

#[test]
fn depolarizing_preserves_trace() {
    let psi = ComplexVec::new(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    ]);
    let rho = DensityMatrix::from_pure_state(&psi);

    for &p in &[0.0, 0.25, 0.5, 0.75, 1.0] {
        let kraus = density::depolarizing_channel(2, p);
        let rho_out = density::apply_channel(&rho, &kraus);
        let tr = rho_out.trace().re;
        assert!(
            (tr - 1.0).abs() < 1e-8,
            "depolarizing p={p}: tr={tr}"
        );
    }
}

#[test]
fn depolarizing_full_noise_gives_mixed() {
    let psi = ComplexVec::new(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    ]);
    let rho = DensityMatrix::from_pure_state(&psi);

    // p=1.0 → fully depolarized → I/dim
    let kraus = density::depolarizing_channel(2, 1.0);
    let rho_out = density::apply_channel(&rho, &kraus);
    assert!((rho_out.get(0, 0).re - 0.5).abs() < 1e-6, "full depol → 0.5 on diag");
    assert!((rho_out.get(1, 1).re - 0.5).abs() < 1e-6);
}

#[test]
fn amplitude_damping_preserves_trace() {
    let psi = ComplexVec::new(vec![
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    ]);
    let rho = DensityMatrix::from_pure_state(&psi);

    for &gamma in &[0.0, 0.3, 0.7, 1.0] {
        let kraus = density::amplitude_damping_channel(gamma, 2);
        let rho_out = density::apply_channel(&rho, &kraus);
        let tr = rho_out.trace().re;
        assert!(
            (tr - 1.0).abs() < 1e-8,
            "amp damp gamma={gamma}: tr={tr}"
        );
    }
}

#[test]
fn amplitude_damping_decays_to_ground() {
    // |1> with gamma=1.0 → should fully decay to |0>
    let psi = ComplexVec::new(vec![
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    ]);
    let rho = DensityMatrix::from_pure_state(&psi);
    let kraus = density::amplitude_damping_channel(1.0, 2);
    let rho_out = density::apply_channel(&rho, &kraus);
    assert!((rho_out.get(0, 0).re - 1.0).abs() < 1e-8, "|1> → |0> under full damping");
    assert!((rho_out.get(1, 1).re).abs() < 1e-8);
}

// =========================================================================
// 7. Von Neumann entropy
// =========================================================================

#[test]
fn entropy_pure_state_is_zero() {
    let psi = ComplexVec::new(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    ]);
    let rho = DensityMatrix::from_pure_state(&psi);
    let s = density::von_neumann_entropy(&rho);
    assert!(s.abs() < 1e-8, "pure state entropy should be 0, got {s}");
}

#[test]
fn entropy_maximally_mixed() {
    let rho = DensityMatrix::maximally_mixed(2);
    let s = density::von_neumann_entropy(&rho);
    let expected = 2.0_f64.ln(); // ln(2)
    assert!(
        (s - expected).abs() < 0.05,
        "maximally mixed entropy should be ln(2)={expected}, got {s}"
    );
}

// =========================================================================
// 8. Fidelity
// =========================================================================

#[test]
fn fidelity_same_state_is_one() {
    let psi = ComplexVec::new(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    ]);
    let rho = DensityMatrix::from_pure_state(&psi);
    let f = density::fidelity(&rho, &rho);
    assert!((f - 1.0).abs() < 1e-4, "F(rho,rho)=1, got {f}");
}

#[test]
fn fidelity_orthogonal_states_is_zero() {
    let psi0 = ComplexVec::new(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);
    let psi1 = ComplexVec::new(vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]);
    let rho0 = DensityMatrix::from_pure_state(&psi0);
    let rho1 = DensityMatrix::from_pure_state(&psi1);
    let f = density::fidelity(&rho0, &rho1);
    assert!(f.abs() < 1e-4, "F(|0>,|1>)=0, got {f}");
}

// =========================================================================
// 9. Quality metrics (phi, omega from density matrix)
// =========================================================================

#[test]
fn phi_from_pure_is_one() {
    let psi = ComplexVec::new(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);
    let rho = DensityMatrix::from_pure_state(&psi);
    let phi = density::phi_from_purity(&rho);
    assert!((phi - 1.0).abs() < 1e-10, "pure state → phi=1, got {phi}");
}

#[test]
fn phi_from_mixed_is_zero() {
    let rho = DensityMatrix::maximally_mixed(2);
    let phi = density::phi_from_purity(&rho);
    assert!(phi.abs() < 1e-10, "maximally mixed → phi=0, got {phi}");
}

#[test]
fn omega_from_coherent_state() {
    let psi = ComplexVec::new(vec![
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
    ]);
    let rho = DensityMatrix::from_pure_state(&psi);
    let omega = density::omega_from_coherence(&rho);
    assert!(omega > 0.5, "superposition state should have high omega, got {omega}");
}

#[test]
fn omega_from_diagonal_is_zero() {
    // Maximally mixed has zero off-diagonal
    let rho = DensityMatrix::maximally_mixed(2);
    let omega = density::omega_from_coherence(&rho);
    assert!(omega.abs() < 1e-10, "diagonal state → omega=0, got {omega}");
}

// =========================================================================
// 10. Unitary evolution (evolve_density)
// =========================================================================

#[test]
fn identity_evolution_preserves_state() {
    let psi = ComplexVec::new(vec![
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
    ]);
    let rho = DensityMatrix::from_pure_state(&psi);
    let id = TransMatrix::identity(2);
    let rho_evolved = ops::evolve_density(&rho, &id).unwrap();
    assert!((rho_evolved.purity() - rho.purity()).abs() < 1e-10);
    assert!((rho_evolved.trace().re - 1.0).abs() < 1e-10);
}

#[test]
fn hadamard_evolution() {
    // H|0><0|H† should give |+><+|
    let psi0 = ComplexVec::new(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);
    let rho0 = DensityMatrix::from_pure_state(&psi0);
    let s2 = 2.0_f32.sqrt();
    let h = TransMatrix::new(vec![1.0/s2, 1.0/s2, 1.0/s2, -1.0/s2], 2, 2);
    let rho_h = ops::evolve_density(&rho0, &h).unwrap();
    // Diagonal should be (0.5, 0.5)
    let diag = rho_h.diagonal();
    assert!((diag[0] - 0.5).abs() < 1e-4);
    assert!((diag[1] - 0.5).abs() < 1e-4);
    // Purity should remain 1
    assert!((rho_h.purity() - 1.0).abs() < 1e-4);
}

// =========================================================================
// 11. Partial trace
// =========================================================================

#[test]
fn partial_trace_product_state() {
    // |00> = |0>x|0>  → tr_B → |0><0|
    let psi00 = ComplexVec::new(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ]);
    let rho = DensityMatrix::from_pure_state(&psi00);
    let rho_a = ops::partial_trace(&rho, 2, 2, true).unwrap();
    assert_eq!(rho_a.dim, 2);
    assert!((rho_a.get(0, 0).re - 1.0).abs() < 1e-10);
    assert!((rho_a.get(1, 1).re).abs() < 1e-10);
    assert!((rho_a.trace().re - 1.0).abs() < 1e-10);
}

#[test]
fn partial_trace_bell_state_is_mixed() {
    // |Bell> = (|00> + |11>)/sqrt(2)
    let bell = ComplexVec::new(vec![
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
    ]);
    let rho = DensityMatrix::from_pure_state(&bell);
    let rho_a = ops::partial_trace(&rho, 2, 2, true).unwrap();
    // Bell state → reduced state is maximally mixed
    assert!((rho_a.purity() - 0.5).abs() < 1e-8, "Bell reduced: purity=0.5");
    assert!((rho_a.get(0, 0).re - 0.5).abs() < 1e-8);
    assert!((rho_a.get(1, 1).re - 0.5).abs() < 1e-8);
}

// =========================================================================
// 12. SVD → Kraus bridge
// =========================================================================

#[test]
fn svd_to_kraus_preserves_trace() {
    // Identity SVD: U=I, sigma=[1,1], Vh=I → channel should be unitary (no damping)
    let dim = 2;
    let u = vec![1.0, 0.0, 0.0, 1.0];
    let sigma = vec![0.9, 0.8];
    let vh = vec![1.0, 0.0, 0.0, 1.0];
    let kraus = density::svd_to_kraus(&u, &sigma, &vh, dim);

    let psi = ComplexVec::new(vec![
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
    ]);
    let rho = DensityMatrix::from_pure_state(&psi);
    let rho_out = density::apply_channel(&rho, &kraus);
    let tr = rho_out.trace().re;
    assert!(
        (tr - 1.0).abs() < 0.1,
        "SVD Kraus should approximately preserve trace, got {tr}"
    );
}

// =========================================================================
// 13. Full pipeline: Declare → Weave(quantum) → Observe
// =========================================================================

/// Helper to build a simple single-relation quantum declaration.
fn make_decl(name: &str, dim: usize, kind: RelationKind, omega: f64, phi: f64) -> EntangleDeclaration {
    let mut b = DeclarationBuilder::new(name);
    b.input("x", dim);
    b.output("y");
    b.relate("y", &["x"], kind);
    b.quality(omega, phi);
    b.build()
}

#[test]
fn quantum_pipeline_basic() {
    let decl = make_decl("q_test", 4, RelationKind::Proportional, 0.9, 0.7);

    let tapestry = weaver::weave(&decl, true, 42).unwrap();
    assert!(tapestry.quantum, "tapestry should be quantum");
    assert!(tapestry.density_matrix.is_some(), "should have density matrix");
    assert!(tapestry.composed_matrix.is_some(), "should have composed matrix");

    // Density matrix should be valid
    let dm = tapestry.density_matrix.as_ref().unwrap();
    assert!((dm.trace().re - 1.0).abs() < 1e-8, "dm trace=1");
    assert!(dm.purity() > 0.99, "from pure state, purity~1");
}

#[test]
fn quantum_observe_has_quantum_metrics() {
    let decl = make_decl("q_obs", 4, RelationKind::Proportional, 0.9, 0.7);

    let tapestry = weaver::weave(&decl, true, 42).unwrap();
    let input = FloatVec::new(vec![1.0, 0.5, 0.3, 0.1]);
    let obs = observatory::observe(&tapestry, &[("x", &input)]).unwrap();

    // Quantum metrics should be present
    assert!(obs.quantum_phi.is_some(), "should have quantum_phi");
    assert!(obs.quantum_omega.is_some(), "should have quantum_omega");
    assert!(obs.density_matrix.is_some(), "should have density_matrix");

    let q_phi = obs.quantum_phi.unwrap();
    let q_omega = obs.quantum_omega.unwrap();
    assert!(q_phi >= 0.0 && q_phi <= 1.0, "phi in [0,1], got {q_phi}");
    assert!(q_omega >= 0.0 && q_omega <= 1.0, "omega in [0,1], got {q_omega}");

    // Probabilities should sum to 1
    let prob_sum: f32 = obs.probabilities.data.iter().sum();
    assert!((prob_sum - 1.0).abs() < 1e-4, "probs sum to 1");
}

#[test]
fn classical_observe_has_no_quantum_metrics() {
    let decl = make_decl("cl_obs", 4, RelationKind::Additive, 0.9, 0.7);

    let tapestry = weaver::weave(&decl, false, 42).unwrap();
    let input = FloatVec::new(vec![1.0, 0.5, 0.3, 0.1]);
    let obs = observatory::observe(&tapestry, &[("x", &input)]).unwrap();

    assert!(obs.quantum_phi.is_none(), "classical should not have quantum_phi");
    assert!(obs.quantum_omega.is_none(), "classical should not have quantum_omega");
    assert!(obs.density_matrix.is_none(), "classical should not have density_matrix");
}

#[test]
fn quantum_reobserve_stable() {
    let decl = make_decl("q_reobs", 4, RelationKind::Proportional, 0.9, 0.7);

    let tapestry = weaver::weave(&decl, true, 42).unwrap();
    let input = FloatVec::new(vec![1.0, 0.5, 0.3, 0.1]);
    let obs = observatory::reobserve(&tapestry, &[("x", &input)], 5).unwrap();

    assert_eq!(obs.observation_count, 5);
    let prob_sum: f32 = obs.probabilities.data.iter().sum();
    assert!((prob_sum - 1.0).abs() < 1e-3, "reobserve probs sum to 1");
    assert!(obs.quantum_phi.is_some());
}

// =========================================================================
// 14. Multi-chain quantum pipeline
// =========================================================================

#[test]
fn quantum_chain_two_layers() {
    let mut b = DeclarationBuilder::new("q_chain");
    b.input("x", 4);
    b.output("z");
    b.relate("y", &["x"], RelationKind::Proportional);
    b.relate("z", &["y"], RelationKind::Additive);
    b.quality(0.85, 0.6);
    let decl = b.build();

    let tapestry = weaver::weave(&decl, true, 123).unwrap();
    assert!(tapestry.quantum);
    assert!(tapestry.composed_matrix.is_some());

    let input = FloatVec::new(vec![0.7, 0.2, 0.05, 0.05]);
    let obs = observatory::observe(&tapestry, &[("x", &input)]).unwrap();

    assert!(obs.quantum_phi.is_some());
    assert!(obs.density_matrix.is_some());
    let dm = obs.density_matrix.unwrap();
    assert!((dm.trace().re - 1.0).abs() < 1e-6);
}

// =========================================================================
// 15. Dimension sweep — channels stay valid at higher dims
// =========================================================================

#[test]
fn channels_valid_at_dim4() {
    let dim = 4;
    let psi = ComplexVec::new(vec![
        Complex64::new(0.5, 0.0),
        Complex64::new(0.5, 0.0),
        Complex64::new(0.5, 0.0),
        Complex64::new(0.5, 0.0),
    ]);
    let rho = DensityMatrix::from_pure_state(&psi);

    // Dephasing
    let k1 = density::dephasing_channel(0.5, dim);
    let r1 = density::apply_channel(&rho, &k1);
    assert!((r1.trace().re - 1.0).abs() < 1e-7, "dephasing dim=4 trace");

    // Depolarizing
    let k2 = density::depolarizing_channel(dim, 0.3);
    let r2 = density::apply_channel(&rho, &k2);
    assert!((r2.trace().re - 1.0).abs() < 1e-7, "depolarizing dim=4 trace");

    // Amplitude damping
    let k3 = density::amplitude_damping_channel(0.4, dim);
    let r3 = density::apply_channel(&rho, &k3);
    assert!((r3.trace().re - 1.0).abs() < 1e-7, "amp damp dim=4 trace");
}

// =========================================================================
// 16. All RelationKinds with quantum
// =========================================================================

#[test]
fn all_relation_kinds_quantum() {
    let kinds = vec![
        RelationKind::Proportional,
        RelationKind::Additive,
        RelationKind::Multiplicative,
        RelationKind::Inverse,
        RelationKind::Conditional,
    ];

    for kind in kinds {
        let name = format!("q_{}", kind.symbol());
        let decl = make_decl(&name, 4, kind.clone(), 0.9, 0.7);

        let tapestry = weaver::weave(&decl, true, 42).unwrap();
        let input = FloatVec::new(vec![1.0, 0.5, 0.3, 0.1]);
        let obs = observatory::observe(&tapestry, &[("x", &input)]).unwrap();

        assert!(
            obs.quantum_phi.is_some(),
            "kind {:?} should produce quantum_phi",
            kind.symbol()
        );
        let prob_sum: f32 = obs.probabilities.data.iter().sum();
        assert!(
            (prob_sum - 1.0).abs() < 1e-3,
            "kind {:?}: probs sum={prob_sum}",
            kind.symbol()
        );
    }
}
