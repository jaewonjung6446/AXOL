//! Comprehensive benchmark: accuracy, speed, scaling
//! Run: cargo test --test bench_comprehensive --release -- --nocapture

use num_complex::Complex64;
use std::f64::consts::PI;
use std::time::Instant;

use axol::types::*;
use axol::ops;
use axol::density;
use axol::declare::*;
use axol::weaver;
use axol::observatory;

// =========================================================================
// Timing helper
// =========================================================================

fn bench<F: FnMut()>(mut f: F, iters: u64) -> f64 {
    // warmup
    for _ in 0..iters.min(100) {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(&mut f)();
    }
    start.elapsed().as_secs_f64() / iters as f64
}

// =========================================================================
// PART A: ACCURACY (numerical precision)
// =========================================================================

#[test]
fn accuracy_report() {
    println!("\n{}", "=".repeat(60));
    println!("  PART A: ACCURACY (numerical precision)");
    println!("{}\n", "=".repeat(60));

    // A1. Born rule normalization
    println!("[A1] Born rule normalization error");
    for dim in [2, 4, 8, 16, 64, 256, 1024] {
        let data: Vec<Complex64> = (0..dim)
            .map(|i| Complex64::new((i as f64 * 0.7).sin(), (i as f64 * 1.3).cos()))
            .collect();
        let cv = ComplexVec::new(data);
        let probs = ops::measure_complex(&cv);
        let sum: f64 = probs.data.iter().map(|&p| p as f64).sum();
        let err = (sum - 1.0).abs();
        println!("  dim={:>5}: sum={:.15}  err={:.2e}", dim, sum, err);
    }

    // A2. Density matrix trace after channels
    println!("\n[A2] Channel trace preservation error");
    for dim in [2, 4, 8, 16] {
        let psi = ComplexVec::new(
            (0..dim).map(|i| Complex64::new((i as f64 * 0.5).sin(), (i as f64 * 0.3).cos())).collect()
        ).normalized();
        let rho = DensityMatrix::from_pure_state(&psi);

        // Dephasing
        let k1 = density::dephasing_channel(0.5, dim);
        let r1 = density::apply_channel(&rho, &k1);
        let err1 = (r1.trace().re - 1.0).abs();

        // Depolarizing
        let k2 = density::depolarizing_channel(dim, 0.5);
        let r2 = density::apply_channel(&rho, &k2);
        let err2 = (r2.trace().re - 1.0).abs();

        // Amplitude damping
        let k3 = density::amplitude_damping_channel(0.5, dim);
        let r3 = density::apply_channel(&rho, &k3);
        let err3 = (r3.trace().re - 1.0).abs();

        println!("  dim={:>3}: dephase={:.2e}  depolar={:.2e}  amp_damp={:.2e}", dim, err1, err2, err3);
    }

    // A3. Pure state purity
    println!("\n[A3] Pure state purity error (expected 1.0)");
    for dim in [2, 4, 8, 16, 32, 64] {
        let psi = ComplexVec::new(
            (0..dim).map(|i| Complex64::new((i as f64).sin(), (i as f64).cos())).collect()
        ).normalized();
        let rho = DensityMatrix::from_pure_state(&psi);
        let err = (rho.purity() - 1.0).abs();
        println!("  dim={:>3}: purity={:.15}  err={:.2e}", dim, rho.purity(), err);
    }

    // A4. Maximally mixed entropy
    println!("\n[A4] Maximally mixed entropy error (expected ln(dim))");
    for dim in [2, 4, 8, 16] {
        let rho = DensityMatrix::maximally_mixed(dim);
        let s = density::von_neumann_entropy(&rho);
        let expected = (dim as f64).ln();
        let err = (s - expected).abs();
        let rel_err = err / expected;
        println!("  dim={:>3}: S={:.8}  expected={:.8}  err={:.2e}  rel={:.2e}", dim, s, expected, err, rel_err);
    }

    // A5. Fidelity symmetry and bounds
    println!("\n[A5] Fidelity properties");
    for dim in [2, 4, 8] {
        let psi_a = ComplexVec::new(
            (0..dim).map(|i| Complex64::new((i as f64).sin(), 0.0)).collect()
        ).normalized();
        let psi_b = ComplexVec::new(
            (0..dim).map(|i| Complex64::new((i as f64).cos(), 0.0)).collect()
        ).normalized();
        let rho_a = DensityMatrix::from_pure_state(&psi_a);
        let rho_b = DensityMatrix::from_pure_state(&psi_b);

        let f_aa = density::fidelity(&rho_a, &rho_a);
        let f_ab = density::fidelity(&rho_a, &rho_b);
        let f_ba = density::fidelity(&rho_b, &rho_a);
        let sym_err = (f_ab - f_ba).abs();

        println!("  dim={}: F(a,a)={:.8}  F(a,b)={:.8}  symmetry_err={:.2e}  in_bounds={}",
            dim, f_aa, f_ab, sym_err, f_ab >= -1e-10 && f_ab <= 1.0 + 1e-10);
    }

    // A6. Hadamard involution: H*H = I
    println!("\n[A6] Hadamard involution (H*H*|psi> = |psi>)");
    let s2 = 2.0_f32.sqrt();
    let h = TransMatrix::new(vec![1.0/s2, 1.0/s2, 1.0/s2, -1.0/s2], 2, 2);
    let psi = ComplexVec::new(vec![Complex64::new(0.6, 0.0), Complex64::new(0.8, 0.0)]);
    let h1 = ops::transform_complex(&psi, &h).unwrap();
    let h2 = ops::transform_complex(&h1, &h).unwrap();
    let err: f64 = psi.data.iter().zip(h2.data.iter())
        .map(|(a, b)| (a - b).norm())
        .sum();
    println!("  H*H involution error: {:.2e}", err);

    // A7. Interference phase linearity
    println!("\n[A7] Interference: phase 0 vs PI relationship");
    let a = ComplexVec::new(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]).normalized();
    let b = ComplexVec::new(vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]).normalized();

    for &phase in &[0.0, PI/4.0, PI/2.0, PI, 3.0*PI/2.0] {
        let result = ops::interfere(&a, &b, phase).unwrap();
        let probs = ops::measure_complex(&result);
        let p_sum: f32 = probs.data.iter().sum();
        println!("  phase={:.4}: probs=[{:.6}, {:.6}]  sum_err={:.2e}",
            phase, probs.data[0], probs.data[1], (p_sum - 1.0).abs());
    }

    // A8. Partial trace consistency
    println!("\n[A8] Partial trace: tr_B(|psi><psi|) trace = 1");
    for dim_a in [2, 3, 4] {
        let dim_b = 2;
        let total = dim_a * dim_b;
        let psi = ComplexVec::new(
            (0..total).map(|i| Complex64::new((i as f64 * 0.6).sin(), (i as f64 * 0.4).cos())).collect()
        ).normalized();
        let rho = DensityMatrix::from_pure_state(&psi);
        let rho_a = ops::partial_trace(&rho, dim_a, dim_b, true).unwrap();
        let rho_b = ops::partial_trace(&rho, dim_a, dim_b, false).unwrap();
        println!("  {}x{}: tr_A={:.10}  tr_B={:.10}  errs=({:.2e}, {:.2e})",
            dim_a, dim_b, rho_a.trace().re, rho_b.trace().re,
            (rho_a.trace().re - 1.0).abs(), (rho_b.trace().re - 1.0).abs());
    }

    // A9. Channel composition: repeated dephasing converges to diagonal
    println!("\n[A9] Repeated dephasing convergence");
    let psi = ComplexVec::new(vec![
        Complex64::new(1.0/2.0_f64.sqrt(), 0.0),
        Complex64::new(1.0/2.0_f64.sqrt(), 0.0),
    ]);
    let mut rho = DensityMatrix::from_pure_state(&psi);
    let kraus = density::dephasing_channel(0.3, 2);
    for step in [1, 5, 10, 20, 50] {
        for _ in 0..step {
            rho = density::apply_channel(&rho, &kraus);
        }
        let off_diag = rho.get(0, 1).norm();
        let tr_err = (rho.trace().re - 1.0).abs();
        println!("  steps={:>3}: |rho_01|={:.8}  tr_err={:.2e}", step, off_diag, tr_err);
    }

    // A10. Quality metrics range
    println!("\n[A10] Quality metrics range validation");
    for dim in [2, 4, 8, 16] {
        let psi = ComplexVec::new(
            (0..dim).map(|i| Complex64::new((i as f64).sin(), (i as f64).cos())).collect()
        ).normalized();
        let rho_pure = DensityMatrix::from_pure_state(&psi);
        let rho_mixed = DensityMatrix::maximally_mixed(dim);

        let phi_pure = density::phi_from_purity(&rho_pure);
        let phi_mixed = density::phi_from_purity(&rho_mixed);
        let omega_pure = density::omega_from_coherence(&rho_pure);
        let omega_mixed = density::omega_from_coherence(&rho_mixed);

        println!("  dim={:>3}: phi_pure={:.4} phi_mixed={:.4} omega_pure={:.4} omega_mixed={:.4}",
            dim, phi_pure, phi_mixed, omega_pure, omega_mixed);
        assert!(phi_pure >= -1e-10 && phi_pure <= 1.0 + 1e-10);
        assert!(phi_mixed >= -1e-10 && phi_mixed <= 1.0 + 1e-10);
        assert!(omega_pure >= -1e-10 && omega_pure <= 1.0 + 1e-10);
        assert!(omega_mixed >= -1e-10 && omega_mixed <= 1.0 + 1e-10);
    }
}

// =========================================================================
// PART B: SPEED (latency per operation)
// =========================================================================

#[test]
fn speed_report() {
    println!("\n{}", "=".repeat(60));
    println!("  PART B: SPEED (latency per operation)");
    println!("{}\n", "=".repeat(60));

    // B1. Born rule (real)
    println!("[B1] Born rule (real) latency");
    for dim in [4, 8, 16, 64, 256, 1024] {
        let data: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
        let fv = FloatVec::new(data);
        let iters = if dim <= 64 { 500_000 } else { 100_000 };
        let t = bench(|| { std::hint::black_box(ops::measure(&fv)); }, iters);
        println!("  dim={:>5}: {:.3} us  ({:.0} M ops/s)", dim, t * 1e6, 1.0 / t / 1e6);
    }

    // B2. Born rule (complex)
    println!("\n[B2] Born rule (complex) latency");
    for dim in [4, 8, 16, 64, 256, 1024] {
        let data: Vec<Complex64> = (0..dim)
            .map(|i| Complex64::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let cv = ComplexVec::new(data).normalized();
        let iters = if dim <= 64 { 500_000 } else { 100_000 };
        let t = bench(|| { std::hint::black_box(ops::measure_complex(&cv)); }, iters);
        println!("  dim={:>5}: {:.3} us  ({:.0} M ops/s)", dim, t * 1e6, 1.0 / t / 1e6);
    }

    // B3. Matrix-vector multiply (real)
    println!("\n[B3] Transform (real matmul) latency");
    for dim in [4, 8, 16, 64, 256] {
        let mat_data: Vec<f32> = (0..dim*dim).map(|i| ((i as f32)*0.01).sin()).collect();
        let vec_data: Vec<f32> = (0..dim).map(|i| (i as f32).cos()).collect();
        let tm = TransMatrix::new(mat_data, dim, dim);
        let fv = FloatVec::new(vec_data);
        let iters = if dim <= 64 { 200_000 } else { 10_000 };
        let t = bench(|| { std::hint::black_box(ops::transform(&fv, &tm).unwrap()); }, iters);
        println!("  dim={:>5}: {:.3} us  (O(n^2) = {})", dim, t * 1e6, dim*dim);
    }

    // B4. Complex transform
    println!("\n[B4] Transform (complex matmul) latency");
    for dim in [4, 8, 16, 64, 256] {
        let mat_data: Vec<f32> = (0..dim*dim).map(|i| ((i as f32)*0.01).sin()).collect();
        let tm = TransMatrix::new(mat_data, dim, dim);
        let cv = ComplexVec::new(
            (0..dim).map(|i| Complex64::new((i as f64).sin(), (i as f64).cos())).collect()
        ).normalized();
        let iters = if dim <= 64 { 200_000 } else { 10_000 };
        let t = bench(|| { std::hint::black_box(ops::transform_complex(&cv, &tm).unwrap()); }, iters);
        println!("  dim={:>5}: {:.3} us", dim, t * 1e6);
    }

    // B5. Interference
    println!("\n[B5] Interference latency");
    for dim in [4, 8, 16, 64, 256, 1024] {
        let a = ComplexVec::new(
            (0..dim).map(|i| Complex64::new((i as f64).sin(), 0.0)).collect()
        ).normalized();
        let b = ComplexVec::new(
            (0..dim).map(|i| Complex64::new((i as f64).cos(), 0.0)).collect()
        ).normalized();
        let iters = if dim <= 64 { 500_000 } else { 100_000 };
        let t = bench(|| { std::hint::black_box(ops::interfere(&a, &b, 0.3).unwrap()); }, iters);
        println!("  dim={:>5}: {:.3} us", dim, t * 1e6);
    }

    // B6. Density matrix construction
    println!("\n[B6] DensityMatrix::from_pure_state latency");
    for dim in [2, 4, 8, 16, 32, 64] {
        let psi = ComplexVec::new(
            (0..dim).map(|i| Complex64::new((i as f64).sin(), (i as f64).cos())).collect()
        ).normalized();
        let iters = if dim <= 16 { 200_000 } else { 20_000 };
        let t = bench(|| { std::hint::black_box(DensityMatrix::from_pure_state(&psi)); }, iters);
        println!("  dim={:>3}: {:.3} us  (creates {}x{} matrix)", dim, t * 1e6, dim, dim);
    }

    // B7. Purity calculation
    println!("\n[B7] Purity (tr(rho^2)) latency");
    for dim in [2, 4, 8, 16, 32, 64] {
        let psi = ComplexVec::new(
            (0..dim).map(|i| Complex64::new((i as f64).sin(), (i as f64).cos())).collect()
        ).normalized();
        let rho = DensityMatrix::from_pure_state(&psi);
        let iters = if dim <= 16 { 200_000 } else { 20_000 };
        let t = bench(|| { std::hint::black_box(rho.purity()); }, iters);
        println!("  dim={:>3}: {:.3} us", dim, t * 1e6);
    }

    // B8. Quantum channels
    println!("\n[B8] Quantum channel apply latency");
    for dim in [2, 4, 8, 16] {
        let psi = ComplexVec::new(
            (0..dim).map(|i| Complex64::new((i as f64).sin(), (i as f64).cos())).collect()
        ).normalized();
        let rho = DensityMatrix::from_pure_state(&psi);

        let k_deph = density::dephasing_channel(0.3, dim);
        let k_depol = density::depolarizing_channel(dim, 0.3);
        let k_amp = density::amplitude_damping_channel(0.3, dim);

        let iters = if dim <= 8 { 50_000 } else { 5_000 };

        let t1 = bench(|| { std::hint::black_box(density::apply_channel(&rho, &k_deph)); }, iters);
        let t2 = bench(|| { std::hint::black_box(density::apply_channel(&rho, &k_depol)); }, iters);
        let t3 = bench(|| { std::hint::black_box(density::apply_channel(&rho, &k_amp)); }, iters);

        println!("  dim={:>3}: dephase={:.1}us  depolar={:.1}us  amp_damp={:.1}us  (Kraus: {}/{}/{})",
            dim, t1*1e6, t2*1e6, t3*1e6, k_deph.len(), k_depol.len(), k_amp.len());
    }

    // B9. Von Neumann entropy
    println!("\n[B9] Von Neumann entropy latency");
    for dim in [2, 4, 8, 16] {
        let rho = DensityMatrix::maximally_mixed(dim);
        let iters = if dim <= 8 { 50_000 } else { 5_000 };
        let t = bench(|| { std::hint::black_box(density::von_neumann_entropy(&rho)); }, iters);
        println!("  dim={:>3}: {:.1} us", dim, t * 1e6);
    }

    // B10. Fidelity
    println!("\n[B10] Fidelity latency");
    for dim in [2, 4, 8, 16] {
        let psi_a = ComplexVec::new(
            (0..dim).map(|i| Complex64::new((i as f64).sin(), 0.0)).collect()
        ).normalized();
        let psi_b = ComplexVec::new(
            (0..dim).map(|i| Complex64::new((i as f64).cos(), 0.0)).collect()
        ).normalized();
        let rho_a = DensityMatrix::from_pure_state(&psi_a);
        let rho_b = DensityMatrix::from_pure_state(&psi_b);
        let iters = if dim <= 8 { 50_000 } else { 5_000 };
        let t = bench(|| { std::hint::black_box(density::fidelity(&rho_a, &rho_b)); }, iters);
        println!("  dim={:>3}: {:.1} us", dim, t * 1e6);
    }

    // B11. Evolve density (unitary)
    println!("\n[B11] Unitary evolution (U rho U^dag) latency");
    for dim in [2, 4, 8, 16, 32] {
        let psi = ComplexVec::new(
            (0..dim).map(|i| Complex64::new((i as f64).sin(), (i as f64).cos())).collect()
        ).normalized();
        let rho = DensityMatrix::from_pure_state(&psi);
        let u = TransMatrix::identity(dim);
        let iters = if dim <= 8 { 50_000 } else { 5_000 };
        let t = bench(|| { std::hint::black_box(ops::evolve_density(&rho, &u).unwrap()); }, iters);
        println!("  dim={:>3}: {:.1} us  (O(n^3) = {})", dim, t * 1e6, dim*dim*dim);
    }

    // B12. Partial trace
    println!("\n[B12] Partial trace latency");
    for (dim_a, dim_b) in [(2,2), (2,4), (4,4), (4,8)] {
        let total = dim_a * dim_b;
        let psi = ComplexVec::new(
            (0..total).map(|i| Complex64::new((i as f64).sin(), (i as f64).cos())).collect()
        ).normalized();
        let rho = DensityMatrix::from_pure_state(&psi);
        let iters = if total <= 16 { 100_000 } else { 10_000 };
        let t = bench(|| { std::hint::black_box(ops::partial_trace(&rho, dim_a, dim_b, true).unwrap()); }, iters);
        println!("  {}x{} (total={}): {:.1} us", dim_a, dim_b, total, t * 1e6);
    }
}

// =========================================================================
// PART C: FULL PIPELINE SPEED
// =========================================================================

fn make_decl_simple(name: &str, dim: usize, kind: RelationKind) -> EntangleDeclaration {
    let mut b = DeclarationBuilder::new(name);
    b.input("x", dim);
    b.output("y");
    b.relate("y", &["x"], kind);
    b.quality(0.9, 0.7);
    b.build()
}

#[test]
fn pipeline_speed_report() {
    println!("\n{}", "=".repeat(60));
    println!("  PART C: FULL PIPELINE SPEED");
    println!("{}\n", "=".repeat(60));

    // C1. Weave latency (quantum vs classical)
    println!("[C1] Weave latency (quantum vs classical)");
    for dim in [4, 8, 16, 32, 64] {
        let decl = make_decl_simple("bench", dim, RelationKind::Proportional);
        let iters = if dim <= 16 { 50_000 } else { 10_000 };

        let t_q = bench(|| { std::hint::black_box(weaver::weave(&decl, true, 42).unwrap()); }, iters);
        let t_c = bench(|| { std::hint::black_box(weaver::weave(&decl, false, 42).unwrap()); }, iters);

        println!("  dim={:>3}: quantum={:.1}us  classical={:.1}us  overhead={:.1}%",
            dim, t_q*1e6, t_c*1e6, (t_q/t_c - 1.0)*100.0);
    }

    // C2. Observe latency (quantum vs classical)
    println!("\n[C2] Observe latency (quantum vs classical)");
    for dim in [4, 8, 16, 32, 64] {
        let decl = make_decl_simple("bench", dim, RelationKind::Proportional);
        let tap_q = weaver::weave(&decl, true, 42).unwrap();
        let tap_c = weaver::weave(&decl, false, 42).unwrap();
        let input = FloatVec::new((0..dim).map(|i| (i as f32 + 1.0).sin()).collect());
        let iters = if dim <= 16 { 100_000 } else { 20_000 };

        let t_q = bench(|| { std::hint::black_box(observatory::observe(&tap_q, &[("x", &input)]).unwrap()); }, iters);
        let t_c = bench(|| { std::hint::black_box(observatory::observe(&tap_c, &[("x", &input)]).unwrap()); }, iters);

        println!("  dim={:>3}: quantum={:.1}us  classical={:.1}us  overhead={:.1}%",
            dim, t_q*1e6, t_c*1e6, (t_q/t_c - 1.0)*100.0);
    }

    // C3. Reobserve latency
    println!("\n[C3] Reobserve latency (5 observations)");
    for dim in [4, 8, 16, 32] {
        let decl = make_decl_simple("bench", dim, RelationKind::Proportional);
        let tap = weaver::weave(&decl, true, 42).unwrap();
        let input = FloatVec::new((0..dim).map(|i| (i as f32 + 1.0).sin()).collect());
        let iters = if dim <= 16 { 50_000 } else { 10_000 };

        let t = bench(|| { std::hint::black_box(observatory::reobserve(&tap, &[("x", &input)], 5).unwrap()); }, iters);
        println!("  dim={:>3}: {:.1} us  (per-obs: {:.1} us)", dim, t * 1e6, t * 1e6 / 5.0);
    }

    // C4. End-to-end: declare + weave + observe
    println!("\n[C4] End-to-end pipeline (declare+weave+observe)");
    for dim in [4, 8, 16, 32, 64] {
        let iters = if dim <= 16 { 50_000 } else { 10_000 };

        let t = bench(|| {
            let mut b = DeclarationBuilder::new("e2e");
            b.input("x", dim);
            b.output("y");
            b.relate("y", &["x"], RelationKind::Proportional);
            b.quality(0.9, 0.7);
            let d = b.build();
            let tap = weaver::weave(&d, true, 42).unwrap();
            let input = FloatVec::new(vec![1.0; dim]);
            std::hint::black_box(observatory::observe(&tap, &[("x", &input)]).unwrap());
        }, iters);
        println!("  dim={:>3}: {:.1} us", dim, t * 1e6);
    }

    // C5. All RelationKinds speed comparison
    println!("\n[C5] Observe latency by RelationKind (dim=8, quantum)");
    let kinds = vec![
        ("Proportional", RelationKind::Proportional),
        ("Additive", RelationKind::Additive),
        ("Multiplicative", RelationKind::Multiplicative),
        ("Inverse", RelationKind::Inverse),
        ("Conditional", RelationKind::Conditional),
    ];
    let input = FloatVec::new(vec![1.0, 0.5, 0.3, 0.1, 0.8, 0.2, 0.6, 0.4]);
    for (name, kind) in &kinds {
        let decl = make_decl_simple("bench", 8, kind.clone());
        let tap = weaver::weave(&decl, true, 42).unwrap();
        let t = bench(|| { std::hint::black_box(observatory::observe(&tap, &[("x", &input)]).unwrap()); }, 200_000);
        println!("  {:>15}: {:.3} us", name, t * 1e6);
    }
}

// =========================================================================
// PART D: SCALING ANALYSIS
// =========================================================================

#[test]
fn scaling_report() {
    println!("\n{}", "=".repeat(60));
    println!("  PART D: SCALING ANALYSIS");
    println!("{}\n", "=".repeat(60));

    // D1. Born rule scaling (should be O(n))
    println!("[D1] Born rule scaling (expected O(n))");
    let mut prev_t = 0.0;
    for dim in [4, 8, 16, 32, 64, 128, 256, 512, 1024] {
        let cv = ComplexVec::new(
            (0..dim).map(|i| Complex64::new((i as f64).sin(), (i as f64).cos())).collect()
        ).normalized();
        let t = bench(|| { std::hint::black_box(ops::measure_complex(&cv)); }, 200_000);
        let ratio = if prev_t > 0.0 { t / prev_t } else { 0.0 };
        println!("  dim={:>5}: {:.3} us  ratio_vs_prev={:.2}x", dim, t * 1e6, ratio);
        prev_t = t;
    }

    // D2. Density matrix scaling (should be O(n^2))
    println!("\n[D2] DensityMatrix from_pure_state scaling (expected O(n^2))");
    let mut prev_t = 0.0;
    for dim in [2, 4, 8, 16, 32, 64, 128] {
        let psi = ComplexVec::new(
            (0..dim).map(|i| Complex64::new((i as f64).sin(), (i as f64).cos())).collect()
        ).normalized();
        let iters = if dim <= 16 { 200_000 } else if dim <= 64 { 20_000 } else { 2_000 };
        let t = bench(|| { std::hint::black_box(DensityMatrix::from_pure_state(&psi)); }, iters);
        let ratio = if prev_t > 0.0 { t / prev_t } else { 0.0 };
        println!("  dim={:>4}: {:.3} us  ratio={:.2}x  (expected ~4.0x for O(n^2))", dim, t * 1e6, ratio);
        prev_t = t;
    }

    // D3. Channel apply scaling
    println!("\n[D3] Dephasing channel apply scaling");
    let mut prev_t = 0.0;
    for dim in [2, 4, 8, 16, 32] {
        let psi = ComplexVec::new(
            (0..dim).map(|i| Complex64::new((i as f64).sin(), (i as f64).cos())).collect()
        ).normalized();
        let rho = DensityMatrix::from_pure_state(&psi);
        let kraus = density::dephasing_channel(0.3, dim);
        let iters = if dim <= 8 { 50_000 } else { 5_000 };
        let t = bench(|| { std::hint::black_box(density::apply_channel(&rho, &kraus)); }, iters);
        let ratio = if prev_t > 0.0 { t / prev_t } else { 0.0 };
        println!("  dim={:>3}: {:.1} us  ratio={:.2}x  (kraus_ops={})",
            dim, t * 1e6, ratio, kraus.len());
        prev_t = t;
    }

    // D4. Evolve density scaling (should be O(n^3))
    println!("\n[D4] Unitary evolution scaling (expected O(n^3))");
    let mut prev_t = 0.0;
    for dim in [2, 4, 8, 16, 32] {
        let psi = ComplexVec::new(
            (0..dim).map(|i| Complex64::new((i as f64).sin(), (i as f64).cos())).collect()
        ).normalized();
        let rho = DensityMatrix::from_pure_state(&psi);
        let u = TransMatrix::identity(dim);
        let iters = if dim <= 8 { 50_000 } else { 5_000 };
        let t = bench(|| { std::hint::black_box(ops::evolve_density(&rho, &u).unwrap()); }, iters);
        let ratio = if prev_t > 0.0 { t / prev_t } else { 0.0 };
        println!("  dim={:>3}: {:.1} us  ratio={:.2}x  (expected ~8.0x for O(n^3))", dim, t * 1e6, ratio);
        prev_t = t;
    }

    // D5. Von Neumann entropy scaling (dominated by Jacobi eigendecomposition)
    println!("\n[D5] Von Neumann entropy scaling (Jacobi eigendecomp)");
    let mut prev_t = 0.0;
    for dim in [2, 4, 8, 16] {
        let rho = DensityMatrix::maximally_mixed(dim);
        let iters = if dim <= 8 { 50_000 } else { 5_000 };
        let t = bench(|| { std::hint::black_box(density::von_neumann_entropy(&rho)); }, iters);
        let ratio = if prev_t > 0.0 { t / prev_t } else { 0.0 };
        println!("  dim={:>3}: {:.1} us  ratio={:.2}x", dim, t * 1e6, ratio);
        prev_t = t;
    }

    // D6. Pipeline scaling
    println!("\n[D6] Full pipeline scaling (declare+weave+observe, quantum)");
    let mut prev_t = 0.0;
    for dim in [4, 8, 16, 32, 64, 128] {
        let iters = if dim <= 16 { 50_000 } else if dim <= 64 { 10_000 } else { 2_000 };
        let t = bench(|| {
            let mut b = DeclarationBuilder::new("scale");
            b.input("x", dim);
            b.output("y");
            b.relate("y", &["x"], RelationKind::Proportional);
            b.quality(0.9, 0.7);
            let d = b.build();
            let tap = weaver::weave(&d, true, 42).unwrap();
            let input = FloatVec::new(vec![1.0; dim]);
            std::hint::black_box(observatory::observe(&tap, &[("x", &input)]).unwrap());
        }, iters);
        let ratio = if prev_t > 0.0 { t / prev_t } else { 0.0 };
        println!("  dim={:>4}: {:.1} us  ratio={:.2}x", dim, t * 1e6, ratio);
        prev_t = t;
    }
}
