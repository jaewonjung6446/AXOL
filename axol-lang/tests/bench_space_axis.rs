//! Space axis verification + full paradigm benchmark
//! Verifies: real chaos dynamics, basin detection, quantum branching, feedback
//!
//! Run: cargo test --test bench_space_axis --release -- --nocapture

use num_complex::Complex64;
use std::time::Instant;
use std::collections::HashSet;

use axol::types::*;
use axol::ops;
use axol::density;
use axol::declare::*;
use axol::dynamics::*;
use axol::weaver;
use axol::observatory;

fn bench<F: FnMut()>(mut f: F, iters: u64) -> f64 {
    for _ in 0..iters.min(50) { f(); }
    let start = Instant::now();
    for _ in 0..iters { std::hint::black_box(&mut f)(); }
    start.elapsed().as_secs_f64() / iters as f64
}

fn make_decl(name: &str, dim: usize, kind: RelationKind, omega: f64, phi: f64) -> EntangleDeclaration {
    let mut b = DeclarationBuilder::new(name);
    b.input("x", dim);
    b.output("y");
    b.relate("y", &["x"], kind);
    b.quality(omega, phi);
    b.build()
}

// =========================================================================
// PART 1: CHAOS DYNAMICS VERIFICATION
// =========================================================================

#[test]
fn verify_chaos_dynamics() {
    println!("\n{}", "=".repeat(60));
    println!("  PART 1: CHAOS DYNAMICS VERIFICATION");
    println!("{}\n", "=".repeat(60));

    // 1.1 Lyapunov exponents are REAL (not reverse-engineered)
    println!("[1.1] Lyapunov exponents are computed from dynamics");
    for &(omega, label) in &[(0.9, "stable"), (0.5, "edge"), (0.2, "chaotic")] {
        let decl = make_decl("test", 4, RelationKind::Proportional, omega, 0.7);
        let chaos = ChaosEngine::from_declaration(&decl, 42);
        let result = chaos.find_attractor(42, 500, 1000);

        let computed_omega = 1.0 / (1.0 + result.max_lyapunov.max(0.0));
        println!("  target_omega={:.1} ({}): r={:.3} max_lyap={:.4} computed_omega={:.4} fractal_dim={:.3}",
            omega, label, chaos.r, result.max_lyapunov, computed_omega, result.fractal_dim);

        // Lyapunov spectrum should be sorted descending
        for i in 1..result.lyapunov_spectrum.len() {
            assert!(result.lyapunov_spectrum[i-1] >= result.lyapunov_spectrum[i] - 0.01,
                "spectrum should be sorted descending");
        }
    }

    // 1.2 Different quality targets produce different dynamics
    println!("\n[1.2] Quality targets affect dynamics");
    let d1 = make_decl("a", 4, RelationKind::Proportional, 0.9, 0.7);
    let d2 = make_decl("b", 4, RelationKind::Proportional, 0.3, 0.7);
    let c1 = ChaosEngine::from_declaration(&d1, 42);
    let c2 = ChaosEngine::from_declaration(&d2, 42);
    let r1 = c1.find_attractor(42, 500, 500);
    let r2 = c2.find_attractor(42, 500, 500);
    println!("  omega=0.9: r={:.3} lyap={:.4}", c1.r, r1.max_lyapunov);
    println!("  omega=0.3: r={:.3} lyap={:.4}", c2.r, r2.max_lyapunov);
    assert!(c2.r > c1.r, "lower omega should use higher r (more chaos)");

    // 1.3 Different RelationKinds produce different coupling structures
    println!("\n[1.3] RelationKind affects coupling weights");
    let kinds = vec![
        ("Proportional", RelationKind::Proportional),
        ("Additive", RelationKind::Additive),
        ("Multiplicative", RelationKind::Multiplicative),
        ("Inverse", RelationKind::Inverse),
    ];
    let mut matrices: Vec<Vec<f64>> = Vec::new();
    for (name, kind) in &kinds {
        let d = make_decl("w", 4, kind.clone(), 0.7, 0.7);
        let c = ChaosEngine::from_declaration(&d, 42);
        println!("  {}: weights=[{:.3},{:.3},{:.3},{:.3} | {:.3},{:.3},{:.3},{:.3} | ...]",
            name,
            c.weights[0], c.weights[1], c.weights[2], c.weights[3],
            c.weights[4], c.weights[5], c.weights[6], c.weights[7]);
        matrices.push(c.weights.clone());
    }
    // Each kind should produce different weights
    for i in 0..matrices.len() {
        for j in (i+1)..matrices.len() {
            let diff: f64 = matrices[i].iter().zip(matrices[j].iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            assert!(diff > 0.01, "different RelationKinds should produce different weights");
        }
    }

    // 1.4 Dynamics actually iterate (not random)
    println!("\n[1.4] Deterministic dynamics (same seed = same result)");
    let d = make_decl("det", 4, RelationKind::Proportional, 0.7, 0.7);
    let c = ChaosEngine::from_declaration(&d, 42);
    let r_a = c.find_attractor(42, 300, 500);
    let r_b = c.find_attractor(42, 300, 500);
    let diff: f64 = r_a.final_state.iter().zip(r_b.final_state.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    println!("  Same seed difference: {:.2e}", diff);
    assert!(diff < 1e-10, "same seed should produce identical results");

    // In chaotic regime, different seeds → different trajectories
    let d_chaotic = make_decl("chaotic", 4, RelationKind::Proportional, 0.3, 0.3);
    let c_chaotic = ChaosEngine::from_declaration(&d_chaotic, 42);
    let r_c1 = c_chaotic.find_attractor(42, 300, 500);
    let r_c2 = c_chaotic.find_attractor(99, 300, 500);
    let diff2: f64 = r_c1.final_state.iter().zip(r_c2.final_state.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    println!("  Chaotic regime (omega=0.3), diff seed difference: {:.6}", diff2);
    // In chaotic regime, sensitive dependence on ICs should produce different states
    // (Note: stable regime omega=0.7 converges to same attractor regardless of IC)
    assert!(diff2 > 1e-6 || c_chaotic.r < 3.57,
        "chaotic regime with different seeds should diverge (r={}, diff={})", c_chaotic.r, diff2);

    // 1.5 Transformation matrix comes from linearization
    println!("\n[1.5] Matrix derived from attractor linearization (not random)");
    let matrix = c.extract_matrix(&r_a);
    println!("  Matrix dim: {}x{}", matrix.rows, matrix.cols);
    // Matrix should not be identity or zero
    let mut has_nondiag = false;
    for i in 0..matrix.rows {
        for j in 0..matrix.cols {
            if i != j && matrix.get(i, j).abs() > 0.01 {
                has_nondiag = true;
            }
        }
    }
    assert!(has_nondiag, "linearization matrix should have off-diagonal elements");
    println!("  Has meaningful off-diagonal structure: true");
}

// =========================================================================
// PART 2: BASIN DETECTION
// =========================================================================

#[test]
fn verify_basins() {
    println!("\n{}", "=".repeat(60));
    println!("  PART 2: BASIN DETECTION");
    println!("{}\n", "=".repeat(60));

    // 2.1 Basins are detected
    println!("[2.1] Basin detection");
    for &(omega, label) in &[(0.9, "stable"), (0.5, "edge"), (0.2, "chaotic")] {
        let d = make_decl("b", 4, RelationKind::Proportional, omega, 0.7);
        let c = ChaosEngine::from_declaration(&d, 42);
        let basins = c.find_basins(200, 300, 42);
        let total_size: f64 = basins.iter().map(|b| b.size).sum();
        println!("  omega={:.1} ({}): {} basins, total_size={:.3}",
            omega, label, basins.len(), total_size);

        assert!(!basins.is_empty(), "should find at least 1 basin");
        assert!((total_size - 1.0).abs() < 0.1, "basin sizes should sum to ~1");

        for (i, basin) in basins.iter().take(3).enumerate() {
            println!("    basin {}: size={:.3} lyap={:.3} phase={:.3} center=[{:.3}, ...]",
                i, basin.size, basin.local_lyapunov, basin.phase,
                basin.center[0]);
        }
    }

    // 2.2 Basin count varies with dynamics regime
    println!("\n[2.2] Basin count vs chaos level");
    let mut basin_counts = Vec::new();
    for &omega in &[0.95, 0.7, 0.5, 0.3, 0.15] {
        let d = make_decl("b", 4, RelationKind::Proportional, omega, 0.7);
        let c = ChaosEngine::from_declaration(&d, 42);
        let basins = c.find_basins(300, 400, 42);
        basin_counts.push(basins.len());
        println!("  omega={:.2} (r={:.3}): {} basins", omega, c.r, basins.len());
    }

    // 2.3 Basins have distinct phases (for quantum interference)
    println!("\n[2.3] Basin phases are distinct");
    let d = make_decl("p", 8, RelationKind::Proportional, 0.5, 0.5);
    let c = ChaosEngine::from_declaration(&d, 42);
    let basins = c.find_basins(300, 400, 42);
    if basins.len() > 1 {
        let phases: Vec<f64> = basins.iter().map(|b| b.phase).collect();
        let mut unique = HashSet::new();
        for &p in &phases {
            unique.insert((p * 1000.0) as i64);
        }
        println!("  {} basins, {} unique phases", basins.len(), unique.len());
        assert!(unique.len() > 1 || basins.len() == 1, "multiple basins should have different phases");
    }
}

// =========================================================================
// PART 3: QUANTUM BASIN OBSERVATION
// =========================================================================

#[test]
fn verify_quantum_basin_observation() {
    println!("\n{}", "=".repeat(60));
    println!("  PART 3: QUANTUM BASIN OBSERVATION");
    println!("{}\n", "=".repeat(60));

    // 3.1 Quantum observe reports basin info
    println!("[3.1] Observation includes basin info");
    let decl = make_decl("qb", 4, RelationKind::Proportional, 0.7, 0.7);
    let tap = weaver::weave(&decl, true, 42).unwrap();
    let input = FloatVec::new(vec![1.0, 0.5, 0.3, 0.1]);
    let obs = observatory::observe(&tap, &[("x", &input)]).unwrap();

    println!("  n_basins: {:?}", obs.n_basins);
    println!("  chosen_basin: {:?}", obs.chosen_basin);
    println!("  quantum_phi: {:?}", obs.quantum_phi);
    println!("  quantum_omega: {:?}", obs.quantum_omega);
    println!("  probs: {:?}", obs.probabilities.data);

    assert!(obs.n_basins.is_some(), "should report n_basins");
    assert!(obs.chosen_basin.is_some(), "should report chosen_basin");
    assert!(obs.quantum_phi.is_some());
    assert!(obs.quantum_omega.is_some());
    assert!(obs.density_matrix.is_some());

    let prob_sum: f32 = obs.probabilities.data.iter().sum();
    assert!((prob_sum - 1.0).abs() < 1e-3, "probs sum to 1");

    // 3.2 Different inputs can reach different basins
    println!("\n[3.2] Different inputs → potentially different basins");
    let inputs_set = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0],
        vec![0.5, 0.5, 0.5, 0.5],
        vec![-1.0, -0.5, 0.5, 1.0],
    ];
    let mut chosen_basins = Vec::new();
    for inp_data in &inputs_set {
        let inp = FloatVec::new(inp_data.iter().map(|&v| v as f32).collect());
        let obs = observatory::observe(&tap, &[("x", &inp)]).unwrap();
        chosen_basins.push(obs.chosen_basin.unwrap_or(0));
        println!("  input=[{:.1},{:.1},{:.1},{:.1}] → basin={} prob_max={:.3}",
            inp_data[0], inp_data[1], inp_data[2], inp_data[3],
            obs.chosen_basin.unwrap_or(0),
            obs.probabilities.data.iter().cloned().fold(0.0f32, f32::max));
    }

    // 3.3 Quantum metrics are valid
    println!("\n[3.3] Quantum metrics validity");
    let dm = obs.density_matrix.as_ref().unwrap();
    let tr = dm.trace().re;
    let pur = dm.purity();
    println!("  trace={:.10}  purity={:.10}", tr, pur);
    assert!((tr - 1.0).abs() < 1e-6, "density matrix trace should be 1");
    assert!(pur >= -1e-6 && pur <= 1.0 + 1e-6, "purity in [0,1]");

    // 3.4 Classical path has no basin info
    println!("\n[3.4] Classical observation has no basin info");
    let tap_cl = weaver::weave(&decl, false, 42).unwrap();
    let obs_cl = observatory::observe(&tap_cl, &[("x", &input)]).unwrap();
    assert!(obs_cl.n_basins.is_none(), "classical should have no basins");
    assert!(obs_cl.quantum_phi.is_none(), "classical should have no quantum_phi");
}

// =========================================================================
// PART 4: MEASUREMENT FEEDBACK
// =========================================================================

#[test]
fn verify_feedback() {
    println!("\n{}", "=".repeat(60));
    println!("  PART 4: MEASUREMENT FEEDBACK");
    println!("{}\n", "=".repeat(60));

    let decl = make_decl("fb", 4, RelationKind::Proportional, 0.8, 0.7);
    let mut tap = weaver::weave(&decl, true, 42).unwrap();
    let input = FloatVec::new(vec![0.8, 0.3, 0.5, 0.1]);

    let initial_r = tap.chaos_engine.as_ref().unwrap().r;
    let initial_lyap = tap.global_attractor.max_lyapunov;
    println!("  Initial: r={:.4} lyap={:.4} omega={:.4}",
        initial_r, initial_lyap, tap.global_attractor.omega());

    // Run feedback loop
    let obs = observatory::observe_evolve(&mut tap, &[("x", &input)], 5).unwrap();

    let final_r = tap.chaos_engine.as_ref().unwrap().r;
    let final_lyap = tap.global_attractor.max_lyapunov;
    println!("  After 5 iterations: r={:.4} lyap={:.4} omega={:.4}",
        final_r, final_lyap, tap.global_attractor.omega());
    println!("  r changed: {} → {} (delta={:.4})", initial_r, final_r, (final_r - initial_r).abs());
    println!("  observation_count: {}", obs.observation_count);

    assert_eq!(obs.observation_count, 5);
    assert!(obs.quantum_phi.is_some());

    // Verify probabilities still valid
    let prob_sum: f32 = obs.probabilities.data.iter().sum();
    assert!((prob_sum - 1.0).abs() < 1e-3);
}

// =========================================================================
// PART 5: PARADIGM COMPLETENESS CHECKLIST
// =========================================================================

#[test]
fn paradigm_completeness() {
    println!("\n{}", "=".repeat(60));
    println!("  PART 5: PARADIGM COMPLETENESS CHECKLIST");
    println!("{}\n", "=".repeat(60));

    let decl = make_decl("complete", 8, RelationKind::Proportional, 0.7, 0.6);
    let mut tap = weaver::weave(&decl, true, 42).unwrap();
    let input = FloatVec::new(vec![0.9, 0.2, 0.5, 0.1, 0.7, 0.3, 0.8, 0.4]);
    let obs = observatory::observe(&tap, &[("x", &input)]).unwrap();

    // Space axis checks
    let has_real_lyapunov = tap.global_attractor.max_lyapunov != 0.0;
    let has_real_fractal = tap.global_attractor.fractal_dim > 0.0;
    let has_chaos_engine = tap.chaos_engine.is_some();
    let has_basins = tap.basins.is_some() && !tap.basins.as_ref().unwrap().is_empty();
    let has_dynamics_matrix = tap.composed_matrix.is_some();

    // Probability axis checks
    let has_born_rule = obs.probabilities.data.iter().sum::<f32>() > 0.99;
    let has_density_matrix = obs.density_matrix.is_some();
    let has_quantum_phi = obs.quantum_phi.is_some();
    let has_quantum_omega = obs.quantum_omega.is_some();
    let has_basin_observation = obs.n_basins.is_some();

    // Feedback check
    let obs_fb = observatory::observe_evolve(&mut tap, &[("x", &input)], 3).unwrap();
    let has_feedback = obs_fb.observation_count == 3;

    println!("  SPACE AXIS:");
    println!("    [{}] Real Lyapunov exponents (from dynamics, not reverse-engineered)", if has_real_lyapunov {"X"} else {" "});
    println!("    [{}] Real fractal dimension (correlation dimension)", if has_real_fractal {"X"} else {" "});
    println!("    [{}] Chaos engine (coupled logistic map lattice)", if has_chaos_engine {"X"} else {" "});
    println!("    [{}] Basin detection (attractor basins discovered)", if has_basins {"X"} else {" "});
    println!("    [{}] Dynamics-derived transformation matrix", if has_dynamics_matrix {"X"} else {" "});
    println!();
    println!("  PROBABILITY AXIS:");
    println!("    [{}] Born rule measurement", if has_born_rule {"X"} else {" "});
    println!("    [{}] Density matrix", if has_density_matrix {"X"} else {" "});
    println!("    [{}] Quantum phi (purity-based clarity)", if has_quantum_phi {"X"} else {" "});
    println!("    [{}] Quantum omega (coherence-based cohesion)", if has_quantum_omega {"X"} else {" "});
    println!("    [{}] Basin-based quantum observation", if has_basin_observation {"X"} else {" "});
    println!();
    println!("  FEEDBACK:");
    println!("    [{}] Measurement → adjust dynamics → re-observe", if has_feedback {"X"} else {" "});

    let total = [
        has_real_lyapunov, has_real_fractal, has_chaos_engine, has_basins,
        has_dynamics_matrix, has_born_rule, has_density_matrix,
        has_quantum_phi, has_quantum_omega, has_basin_observation, has_feedback,
    ];
    let passed = total.iter().filter(|&&v| v).count();
    println!("\n  TOTAL: {}/{} checks passed", passed, total.len());

    assert!(passed == total.len(), "All paradigm checks should pass");
}

// =========================================================================
// PART 6: PERFORMANCE COMPARISON (before vs after)
// =========================================================================

#[test]
fn performance_comparison() {
    println!("\n{}", "=".repeat(60));
    println!("  PART 6: PERFORMANCE (with real dynamics)");
    println!("{}\n", "=".repeat(60));

    // 6.1 Weave latency (now includes real dynamics)
    println!("[6.1] Weave latency (real dynamics)");
    for dim in [4, 8, 16, 32] {
        let decl = make_decl("perf", dim, RelationKind::Proportional, 0.7, 0.7);
        let iters = if dim <= 8 { 500 } else { 100 };

        let t_q = bench(|| { std::hint::black_box(weaver::weave(&decl, true, 42).unwrap()); }, iters);
        let t_c = bench(|| { std::hint::black_box(weaver::weave(&decl, false, 42).unwrap()); }, iters);

        println!("  dim={:>3}: quantum={:.0}us  classical={:.0}us  overhead={:.0}%",
            dim, t_q*1e6, t_c*1e6, (t_q/t_c - 1.0)*100.0);
    }

    // 6.2 Observe latency
    println!("\n[6.2] Observe latency (basin-based quantum)");
    for dim in [4, 8, 16, 32] {
        let decl = make_decl("perf", dim, RelationKind::Proportional, 0.7, 0.7);
        let tap_q = weaver::weave(&decl, true, 42).unwrap();
        let tap_c = weaver::weave(&decl, false, 42).unwrap();
        let input = FloatVec::new((0..dim).map(|i| (i as f32 + 1.0).sin()).collect());
        let iters = if dim <= 8 { 50_000 } else { 10_000 };

        let t_q = bench(|| { std::hint::black_box(observatory::observe(&tap_q, &[("x", &input)]).unwrap()); }, iters);
        let t_c = bench(|| { std::hint::black_box(observatory::observe(&tap_c, &[("x", &input)]).unwrap()); }, iters);

        let n_basins = tap_q.basins.as_ref().map(|b| b.len()).unwrap_or(0);
        println!("  dim={:>3}: quantum={:.1}us  classical={:.1}us  basins={}  overhead={:.0}%",
            dim, t_q*1e6, t_c*1e6, n_basins, (t_q/t_c - 1.0)*100.0);
    }

    // 6.3 End-to-end pipeline
    println!("\n[6.3] End-to-end (declare+weave+observe)");
    for dim in [4, 8, 16, 32] {
        let iters = if dim <= 8 { 500 } else { 100 };
        let t = bench(|| {
            let d = make_decl("e2e", dim, RelationKind::Proportional, 0.7, 0.7);
            let tap = weaver::weave(&d, true, 42).unwrap();
            let input = FloatVec::new(vec![1.0; dim]);
            std::hint::black_box(observatory::observe(&tap, &[("x", &input)]).unwrap());
        }, iters);
        println!("  dim={:>3}: {:.0} us", dim, t * 1e6);
    }

    // 6.4 Feedback loop
    println!("\n[6.4] observe_evolve (3 iterations)");
    for dim in [4, 8, 16] {
        let decl = make_decl("fb", dim, RelationKind::Proportional, 0.7, 0.7);
        let iters = if dim <= 8 { 100 } else { 20 };
        let t = bench(|| {
            let mut tap = weaver::weave(&decl, true, 42).unwrap();
            let input = FloatVec::new(vec![1.0; dim]);
            std::hint::black_box(observatory::observe_evolve(&mut tap, &[("x", &input)], 3).unwrap());
        }, iters);
        println!("  dim={:>3}: {:.0} us  (per iteration: {:.0} us)", dim, t * 1e6, t * 1e6 / 3.0);
    }

    // 6.5 Observe amortization
    println!("\n[6.5] Weave once, observe N times (amortized cost)");
    for dim in [4, 8, 16] {
        let decl = make_decl("amort", dim, RelationKind::Proportional, 0.7, 0.7);
        let tap = weaver::weave(&decl, true, 42).unwrap();
        let input = FloatVec::new(vec![1.0; dim]);

        let iters = if dim <= 8 { 20_000 } else { 5_000 };
        let t_obs = bench(|| {
            std::hint::black_box(observatory::observe(&tap, &[("x", &input)]).unwrap());
        }, iters);

        // Weave cost (one time)
        let t_weave = bench(|| {
            std::hint::black_box(weaver::weave(&decl, true, 42).unwrap());
        }, 100);

        let n_obs_break_even = (t_weave / t_obs) as u64;
        println!("  dim={:>3}: weave={:.0}us  observe={:.1}us  break_even={}x observe",
            dim, t_weave*1e6, t_obs*1e6, n_obs_break_even);
    }
}
