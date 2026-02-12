//! Compose layer benchmark: confidence, chains, iterate, basin design, logic gates
//! Run: cargo test --test bench_compose --release -- --nocapture

use std::time::Instant;

use axol::types::*;
use axol::declare::*;
use axol::weaver;
use axol::compose::*;

// =========================================================================
// Timing helper
// =========================================================================

fn bench<F: FnMut()>(mut f: F, iters: u64) -> f64 {
    for _ in 0..iters.min(50) { f(); }
    let start = Instant::now();
    for _ in 0..iters { std::hint::black_box(&mut f)(); }
    start.elapsed().as_secs_f64() / iters as f64
}

fn make_decl(name: &str, dim: usize, omega: f64, phi: f64) -> EntangleDeclaration {
    let mut b = DeclarationBuilder::new(name);
    b.input("x", dim);
    b.output("y");
    b.relate("y", &["x"], RelationKind::Proportional);
    b.quality(omega, phi);
    b.build()
}

fn make_tapestry(name: &str, dim: usize, quantum: bool) -> weaver::Tapestry {
    let decl = make_decl(name, dim, 0.9, 0.7);
    weaver::weave(&decl, quantum, 42).unwrap()
}

// =========================================================================
// PART A: LOGIC GATE SPEED
// =========================================================================

#[test]
fn bench_logic_gates() {
    println!("\n{}", "=".repeat(60));
    println!("  PART A: LOGIC GATE SPEED");
    println!("{}\n", "=".repeat(60));

    let f = encode_bool(false);
    let t = encode_bool(true);

    // A1. NOT gate
    println!("[A1] NOT gate latency");
    let t_not = bench(|| { std::hint::black_box(gate_not(&f).unwrap()); }, 500_000);
    println!("  NOT: {:.3} us  ({:.0} M ops/s)", t_not * 1e6, 1.0 / t_not / 1e6);

    // A2. AND gate
    println!("\n[A2] AND gate latency");
    let t_and = bench(|| { std::hint::black_box(gate_and(&t, &f).unwrap()); }, 500_000);
    println!("  AND: {:.3} us  ({:.0} M ops/s)", t_and * 1e6, 1.0 / t_and / 1e6);

    // A3. OR gate
    println!("\n[A3] OR gate latency");
    let t_or = bench(|| { std::hint::black_box(gate_or(&t, &f).unwrap()); }, 500_000);
    println!("  OR:  {:.3} us  ({:.0} M ops/s)", t_or * 1e6, 1.0 / t_or / 1e6);

    // A4. Gate truth table verification + timing
    println!("\n[A4] Full truth table timing");
    let combos = vec![
        (false, false), (false, true), (true, false), (true, true),
    ];
    let start = Instant::now();
    let mut total_ops = 0u64;
    for _ in 0..100_000 {
        for &(a, b) in &combos {
            let va = encode_bool(a);
            let vb = encode_bool(b);
            std::hint::black_box(gate_and(&va, &vb).unwrap());
            std::hint::black_box(gate_or(&va, &vb).unwrap());
            std::hint::black_box(gate_not(&va).unwrap());
            total_ops += 3;
        }
    }
    let elapsed = start.elapsed();
    println!("  {} gate ops in {:.1}ms = {:.1} M ops/s",
        total_ops, elapsed.as_secs_f64() * 1000.0,
        total_ops as f64 / elapsed.as_secs_f64() / 1e6);

    // A5. IF-THEN-ELSE
    println!("\n[A5] IF-THEN-ELSE latency");
    let cond = make_bool_tapestry("cond", 0.9, 0.7, 100).unwrap();
    let then_t = make_bool_tapestry("then", 0.9, 0.7, 200).unwrap();
    let else_t = make_bool_tapestry("else", 0.9, 0.7, 300).unwrap();
    let branch_input = FloatVec::new(vec![0.5, 0.5]);

    let t_ite = bench(|| {
        let cond_in = encode_bool(true);
        std::hint::black_box(eval_if_then_else(&cond_in, &cond, &then_t, &else_t, &branch_input).unwrap());
    }, 50_000);
    println!("  IF-THEN-ELSE: {:.1} us", t_ite * 1e6);
}

// =========================================================================
// PART B: CONFIDENCE OBSERVER SPEED
// =========================================================================

#[test]
fn bench_confidence() {
    println!("\n{}", "=".repeat(60));
    println!("  PART B: CONFIDENCE OBSERVER SPEED");
    println!("{}\n", "=".repeat(60));

    // B1. Confidence with varying max_observations
    println!("[B1] Confidence latency vs max_observations");
    for &max_obs in &[10, 20, 50, 100] {
        let tap = make_tapestry("conf", 4, true);
        let input = FloatVec::new(vec![0.8, 0.2, 0.1, 0.05]);
        let config = ConfidenceConfig {
            max_observations: max_obs,
            confidence_threshold: 0.99, // High threshold → unlikely to early stop
            min_observations: 3,
        };
        let iters = 1000 / max_obs as u64 + 1;

        let t = bench(|| {
            std::hint::black_box(observe_confident(&tap, &[("x", &input)], &config).unwrap());
        }, iters.max(10));
        println!("  max={:>3}: {:.1} us  ({:.1} us/obs)",
            max_obs, t * 1e6, t * 1e6 / max_obs as f64);
    }

    // B2. Confidence with varying dimensions
    println!("\n[B2] Confidence latency vs dimension (max=20)");
    for dim in [2, 4, 8, 16, 32] {
        let tap = make_tapestry("conf", dim, true);
        let input = FloatVec::new((0..dim).map(|i| (i as f32 + 1.0).sin().abs()).collect());
        let config = ConfidenceConfig {
            max_observations: 20,
            confidence_threshold: 0.99,
            min_observations: 3,
        };

        let t = bench(|| {
            std::hint::black_box(observe_confident(&tap, &[("x", &input)], &config).unwrap());
        }, 100);
        println!("  dim={:>3}: {:.1} us  ({:.1} us/obs)", dim, t * 1e6, t * 1e6 / 20.0);
    }

    // B3. Early stopping efficiency
    println!("\n[B3] Early stopping efficiency (threshold=0.5)");
    for dim in [2, 4, 8] {
        let tap = make_tapestry("early", dim, true);
        let input = FloatVec::new((0..dim).map(|i| (i as f32 + 1.0).sin().abs()).collect());
        let config_strict = ConfidenceConfig {
            max_observations: 100,
            confidence_threshold: 0.99,
            min_observations: 3,
        };
        let config_loose = ConfidenceConfig {
            max_observations: 100,
            confidence_threshold: 0.3,
            min_observations: 3,
        };

        let r_strict = observe_confident(&tap, &[("x", &input)], &config_strict).unwrap();
        let r_loose = observe_confident(&tap, &[("x", &input)], &config_loose).unwrap();

        println!("  dim={}: strict(0.99): {} obs early={}  loose(0.30): {} obs early={}",
            dim,
            r_strict.total_observations, r_strict.early_stopped,
            r_loose.total_observations, r_loose.early_stopped);
    }

    // B4. Classical vs quantum confidence
    println!("\n[B4] Classical vs quantum confidence (dim=4, max=20)");
    let input = FloatVec::new(vec![0.8, 0.2, 0.1, 0.05]);
    let config = ConfidenceConfig {
        max_observations: 20,
        confidence_threshold: 0.99,
        min_observations: 3,
    };

    let tap_q = make_tapestry("q", 4, true);
    let tap_c = make_tapestry("c", 4, false);

    let t_q = bench(|| {
        std::hint::black_box(observe_confident(&tap_q, &[("x", &input)], &config).unwrap());
    }, 100);
    let t_c = bench(|| {
        std::hint::black_box(observe_confident(&tap_c, &[("x", &input)], &config).unwrap());
    }, 100);

    println!("  quantum: {:.1} us  classical: {:.1} us  overhead: {:.0}%",
        t_q * 1e6, t_c * 1e6, (t_q / t_c - 1.0) * 100.0);
}

// =========================================================================
// PART C: TAPESTRY CHAIN SPEED
// =========================================================================

#[test]
fn bench_chains() {
    println!("\n{}", "=".repeat(60));
    println!("  PART C: TAPESTRY CHAIN SPEED");
    println!("{}\n", "=".repeat(60));

    // C1. Chain construction (matmul composition)
    println!("[C1] Chain construction latency (matrix composition)");
    for n_stages in [2, 3, 5, 10] {
        let stages: Vec<weaver::Tapestry> = (0..n_stages)
            .map(|i| make_tapestry(&format!("s{}", i), 4, true))
            .collect();

        let t = bench(|| {
            std::hint::black_box(chain("bench_chain", stages.clone()).unwrap());
        }, 10_000);
        println!("  {} stages (dim=4): {:.1} us", n_stages, t * 1e6);
    }

    // C2. Chain vs individual observe
    println!("\n[C2] Chain observe vs individual observe (dim=4, 3 stages)");
    let stages: Vec<weaver::Tapestry> = (0..3)
        .map(|i| make_tapestry(&format!("s{}", i), 4, true))
        .collect();
    let c = chain("bench", stages).unwrap();
    let input = FloatVec::new(vec![0.8, 0.2, 0.1, 0.05]);

    let t_composed = bench(|| {
        std::hint::black_box(observe_chain(&c, &[("x", &input)]).unwrap());
    }, 50_000);
    println!("  Composed (single matrix): {:.1} us", t_composed * 1e6);

    // C3. Flatten latency
    println!("\n[C3] Chain flatten latency");
    for n_stages in [2, 5, 10] {
        let stages: Vec<weaver::Tapestry> = (0..n_stages)
            .map(|i| make_tapestry(&format!("s{}", i), 4, true))
            .collect();
        let c = chain("flat", stages).unwrap();

        let t = bench(|| {
            std::hint::black_box(flatten(&c).unwrap());
        }, 10_000);
        println!("  {} stages: {:.1} us", n_stages, t * 1e6);
    }

    // C4. Chain scaling with dimension
    println!("\n[C4] Chain construction scaling with dimension (2 stages)");
    for dim in [2, 4, 8, 16, 32, 64] {
        let s1 = make_tapestry("a", dim, true);
        let s2 = make_tapestry("b", dim, true);

        let t = bench(|| {
            std::hint::black_box(chain("scale", vec![s1.clone(), s2.clone()]).unwrap());
        }, if dim <= 16 { 10_000 } else { 1_000 });
        println!("  dim={:>3}: {:.1} us  (matmul O(n^3)={})", dim, t * 1e6, dim*dim*dim);
    }
}

// =========================================================================
// PART D: ITERATE LOOP SPEED
// =========================================================================

#[test]
fn bench_iterate() {
    println!("\n{}", "=".repeat(60));
    println!("  PART D: ITERATE LOOP SPEED");
    println!("{}\n", "=".repeat(60));

    // D1. Iterate latency vs max_iterations (no feedback)
    println!("[D1] Iterate latency vs max_iterations (no feedback, dim=4)");
    for &max_iter in &[5, 10, 20, 50] {
        let mut tap = make_tapestry("iter", 4, true);
        let input = FloatVec::new(vec![0.8, 0.2, 0.1, 0.05]);
        let config = IterateConfig {
            max_iterations: max_iter,
            min_iterations: 3,
            convergence: ConvergenceCriterion::ProbabilityDelta(0.0001), // unlikely to converge
            feedback: false,
            feedback_strength: 0.0,
        };

        let start = Instant::now();
        let result = iterate(&mut tap, &[("x", &input)], &config).unwrap();
        let elapsed = start.elapsed();

        println!("  max={:>3}: {:.1} ms  iters={} converged={} ({:.1} us/iter)",
            max_iter,
            elapsed.as_secs_f64() * 1000.0,
            result.iterations, result.converged,
            elapsed.as_secs_f64() * 1e6 / result.iterations as f64);
    }

    // D2. Iterate with feedback
    println!("\n[D2] Iterate latency with feedback (dim=4)");
    for &max_iter in &[5, 10, 20] {
        let mut tap = make_tapestry("iter_fb", 4, true);
        let input = FloatVec::new(vec![0.8, 0.2, 0.1, 0.05]);
        let config = IterateConfig {
            max_iterations: max_iter,
            min_iterations: 3,
            convergence: ConvergenceCriterion::ProbabilityDelta(0.0001),
            feedback: true,
            feedback_strength: 0.5,
        };

        let start = Instant::now();
        let result = iterate(&mut tap, &[("x", &input)], &config).unwrap();
        let elapsed = start.elapsed();

        println!("  max={:>3}: {:.1} ms  iters={} converged={} ({:.1} us/iter)",
            max_iter,
            elapsed.as_secs_f64() * 1000.0,
            result.iterations, result.converged,
            elapsed.as_secs_f64() * 1e6 / result.iterations as f64);
    }

    // D3. Iterate dimension scaling
    println!("\n[D3] Iterate dimension scaling (10 iters, no feedback)");
    for dim in [2, 4, 8, 16, 32] {
        let mut tap = make_tapestry("scale", dim, true);
        let input = FloatVec::new((0..dim).map(|i| (i as f32 + 1.0).sin().abs()).collect());
        let config = IterateConfig {
            max_iterations: 10,
            min_iterations: 3,
            convergence: ConvergenceCriterion::ProbabilityDelta(0.0001),
            feedback: false,
            feedback_strength: 0.0,
        };

        let start = Instant::now();
        let result = iterate(&mut tap, &[("x", &input)], &config).unwrap();
        let elapsed = start.elapsed();

        println!("  dim={:>3}: {:.1} ms  ({:.1} us/iter)",
            dim,
            elapsed.as_secs_f64() * 1000.0,
            elapsed.as_secs_f64() * 1e6 / result.iterations as f64);
    }

    // D4. Convergence criteria comparison (dim=4, 50 max)
    println!("\n[D4] Convergence criteria comparison (dim=4, classical, max=50)");
    let criteria = vec![
        ("ProbDelta(0.01)", ConvergenceCriterion::ProbabilityDelta(0.01)),
        ("ProbDelta(0.001)", ConvergenceCriterion::ProbabilityDelta(0.001)),
        ("StableIndex(3)", ConvergenceCriterion::StableIndex(3)),
        ("StableIndex(5)", ConvergenceCriterion::StableIndex(5)),
    ];
    for (name, criterion) in criteria {
        let mut tap = make_tapestry("crit", 4, false);
        let input = FloatVec::new(vec![0.8, 0.2, 0.1, 0.05]);
        let config = IterateConfig {
            max_iterations: 50,
            min_iterations: 3,
            convergence: criterion,
            feedback: false,
            feedback_strength: 0.0,
        };

        let start = Instant::now();
        let result = iterate(&mut tap, &[("x", &input)], &config).unwrap();
        let elapsed = start.elapsed();

        println!("  {:>20}: iters={:>3} converged={} delta={:.6} ({:.1} ms)",
            name, result.iterations, result.converged, result.final_delta,
            elapsed.as_secs_f64() * 1000.0);
    }
}

// =========================================================================
// PART E: BASIN DESIGNER SPEED
// =========================================================================

#[test]
fn bench_basin_designer() {
    println!("\n{}", "=".repeat(60));
    println!("  PART E: BASIN DESIGNER SPEED");
    println!("{}\n", "=".repeat(60));

    // E1. Basin design latency vs grid size
    println!("[E1] Basin design latency vs grid resolution");
    for &(gr, ge) in &[(5, 5), (10, 10), (15, 15), (20, 20)] {
        let spec = BasinSpec {
            n_basins: 2,
            target_sizes: vec![0.5, 0.5],
            boundary_hints: Vec::new(),
            dim: 2,
        };
        let config = BasinDesignConfig {
            grid_r_steps: gr,
            grid_eps_steps: ge,
            top_k: 3,
            nelder_mead_iterations: 30,
            n_samples: 100,
            transient: 100,
        };

        let start = Instant::now();
        let result = design_basins(&spec, &config).unwrap();
        let elapsed = start.elapsed();

        println!("  grid={}x{}: {:.1} ms  score={:.4} basins={} evals={}",
            gr, ge, elapsed.as_secs_f64() * 1000.0,
            result.score, result.basins.len(), result.iterations);
    }

    // E2. Basin design latency vs dimension
    println!("\n[E2] Basin design latency vs dimension (grid=10x10)");
    for dim in [2, 4, 8] {
        let spec = BasinSpec {
            n_basins: 2,
            target_sizes: vec![0.5, 0.5],
            boundary_hints: Vec::new(),
            dim,
        };
        let config = BasinDesignConfig {
            grid_r_steps: 10,
            grid_eps_steps: 10,
            top_k: 3,
            nelder_mead_iterations: 30,
            n_samples: 100,
            transient: 100,
        };

        let start = Instant::now();
        let result = design_basins(&spec, &config).unwrap();
        let elapsed = start.elapsed();

        println!("  dim={}: {:.1} ms  score={:.4} basins={}",
            dim, elapsed.as_secs_f64() * 1000.0,
            result.score, result.basins.len());
    }

    // E3. Basin design with different target counts
    println!("\n[E3] Basin design vs target basin count (dim=2, grid=10x10)");
    for n_basins in [1, 2, 3, 4, 5] {
        let sizes = vec![1.0 / n_basins as f64; n_basins];
        let spec = BasinSpec {
            n_basins,
            target_sizes: sizes,
            boundary_hints: Vec::new(),
            dim: 2,
        };
        let config = BasinDesignConfig {
            grid_r_steps: 10,
            grid_eps_steps: 10,
            top_k: 3,
            nelder_mead_iterations: 30,
            n_samples: 100,
            transient: 100,
        };

        let start = Instant::now();
        let result = design_basins(&spec, &config).unwrap();
        let elapsed = start.elapsed();

        println!("  target={} basins: {:.1} ms  found={} score={:.4} r={:.3} eps={:.3}",
            n_basins, elapsed.as_secs_f64() * 1000.0,
            result.basins.len(), result.score,
            result.engine.r, result.engine.epsilon);
    }

    // E4. Nelder-Mead standalone benchmark
    println!("\n[E4] Nelder-Mead optimizer (standalone, Rosenbrock function)");
    let rosenbrock = |x: &[f64]| -> f64 {
        let a = 1.0 - x[0];
        let b = x[1] - x[0] * x[0];
        a * a + 100.0 * b * b
    };

    for &max_iter in &[50, 100, 200, 500] {
        let start = Instant::now();
        let (params, score, iters) = nelder_mead_optimize(
            &[0.0, 0.0], &rosenbrock, max_iter,
        );
        let elapsed = start.elapsed();

        println!("  max_iter={:>3}: {:.1} us  score={:.6} params=[{:.4},{:.4}] iters={}",
            max_iter, elapsed.as_secs_f64() * 1e6,
            score, params[0], params[1], iters);
    }
}

// =========================================================================
// PART F: COMPOSE vs EXISTING COMPARISON
// =========================================================================

#[test]
fn bench_compose_vs_raw() {
    println!("\n{}", "=".repeat(60));
    println!("  PART F: COMPOSE vs RAW OPERATIONS");
    println!("{}\n", "=".repeat(60));

    // F1. observe_confident vs reobserve (20 observations)
    println!("[F1] observe_confident vs reobserve (20 obs, dim=4)");
    let tap = make_tapestry("cmp", 4, true);
    let input = FloatVec::new(vec![0.8, 0.2, 0.1, 0.05]);

    let config = ConfidenceConfig {
        max_observations: 20,
        confidence_threshold: 0.99, // no early stop
        min_observations: 3,
    };

    let t_conf = bench(|| {
        std::hint::black_box(observe_confident(&tap, &[("x", &input)], &config).unwrap());
    }, 100);
    let t_reobs = bench(|| {
        std::hint::black_box(axol::observatory::reobserve(&tap, &[("x", &input)], 20).unwrap());
    }, 100);

    println!("  confident:  {:.1} us (with Wilson score + vote counting)", t_conf * 1e6);
    println!("  reobserve:  {:.1} us (simple averaging)", t_reobs * 1e6);
    println!("  overhead:   {:.0}%", (t_conf / t_reobs - 1.0) * 100.0);

    // F2. iterate vs observe_evolve (10 iterations)
    println!("\n[F2] iterate vs observe_evolve (10 iters, dim=4)");
    let t_iterate = {
        let mut tap = make_tapestry("it", 4, true);
        let config = IterateConfig {
            max_iterations: 10,
            min_iterations: 3,
            convergence: ConvergenceCriterion::ProbabilityDelta(0.0001),
            feedback: true,
            feedback_strength: 0.5,
        };
        let start = Instant::now();
        std::hint::black_box(iterate(&mut tap, &[("x", &input)], &config).unwrap());
        start.elapsed().as_secs_f64()
    };
    let t_evolve = {
        let mut tap = make_tapestry("ev", 4, true);
        let start = Instant::now();
        std::hint::black_box(axol::observatory::observe_evolve(&mut tap, &[("x", &input)], 10).unwrap());
        start.elapsed().as_secs_f64()
    };

    println!("  iterate:       {:.1} ms (configurable convergence + history)", t_iterate * 1000.0);
    println!("  observe_evolve:{:.1} ms (fixed feedback)", t_evolve * 1000.0);

    // F3. Chain observe vs sequential observe
    println!("\n[F3] Chain composed vs sequential observe (3 stages, dim=4)");
    let stages: Vec<weaver::Tapestry> = (0..3)
        .map(|i| make_tapestry(&format!("s{}", i), 4, true))
        .collect();
    let c = chain("bench", stages.clone()).unwrap();

    let t_composed = bench(|| {
        std::hint::black_box(observe_chain(&c, &[("x", &input)]).unwrap());
    }, 50_000);

    let t_individual = bench(|| {
        for stage in &stages {
            std::hint::black_box(axol::observatory::observe(stage, &[("x", &input)]).unwrap());
        }
    }, 50_000);

    println!("  Composed (single matrix): {:.1} us", t_composed * 1e6);
    println!("  Individual (3 observes):  {:.1} us", t_individual * 1e6);
    println!("  Speedup:                  {:.1}x", t_individual / t_composed);
}

// =========================================================================
// PART G: END-TO-END COMPOSE PIPELINE
// =========================================================================

#[test]
fn bench_full_compose_pipeline() {
    println!("\n{}", "=".repeat(60));
    println!("  PART G: FULL COMPOSE PIPELINE");
    println!("{}\n", "=".repeat(60));

    // G1. DSL execution with compose commands
    println!("[G1] DSL gate execution");
    let gate_source = r#"
gate not { x = [0.1, 0.9] }
gate and { a = [0.1, 0.9], b = [0.1, 0.9] }
gate or { a = [0.9, 0.1], b = [0.1, 0.9] }
"#;

    let start = Instant::now();
    let mut lexer = axol::dsl::lexer::Lexer::new(gate_source);
    lexer.tokenize();
    let mut parser = axol::dsl::parser::Parser::new(lexer.tokens.clone());
    let program = parser.parse().unwrap();
    let mut rt = axol::dsl::compiler::Runtime::new();
    let output = rt.execute(&program).unwrap();
    let elapsed = start.elapsed();

    for line in &output {
        println!("  {}", line);
    }
    println!("  Total DSL gate time: {:.1} us", elapsed.as_secs_f64() * 1e6);

    // G2. Full pipeline: declare, weave, confident, iterate
    println!("\n[G2] Full DSL pipeline (declare+weave+confident+iterate)");
    let full_source = r#"
declare "bench_pipe" {
    input x(4)
    relate y <- x via <~>
    output y
    quality omega=0.9 phi=0.7
}

weave bench_pipe quantum=true seed=42
confident bench_pipe max=10 threshold=0.5 { x = [0.8, 0.2, 0.1, 0.05] }
iterate bench_pipe max=5 converge=prob_delta value=0.01 { x = [0.8, 0.2, 0.1, 0.05] }
"#;

    let start = Instant::now();
    let mut lexer = axol::dsl::lexer::Lexer::new(full_source);
    lexer.tokenize();
    let mut parser = axol::dsl::parser::Parser::new(lexer.tokens.clone());
    let program = parser.parse().unwrap();
    let mut rt = axol::dsl::compiler::Runtime::new();
    let output = rt.execute(&program).unwrap();
    let elapsed = start.elapsed();

    for line in &output {
        println!("  {}", line);
    }
    println!("  Total pipeline time: {:.1} ms", elapsed.as_secs_f64() * 1000.0);

    // G3. Design + use
    println!("\n[G3] Design basins");
    let design_source = r#"design "binary" { dim 2, basins 2, sizes [0.5, 0.5] }"#;

    let start = Instant::now();
    let mut lexer = axol::dsl::lexer::Lexer::new(design_source);
    lexer.tokenize();
    let mut parser = axol::dsl::parser::Parser::new(lexer.tokens.clone());
    let program = parser.parse().unwrap();
    let mut rt = axol::dsl::compiler::Runtime::new();
    let output = rt.execute(&program).unwrap();
    let elapsed = start.elapsed();

    for line in &output {
        println!("  {}", line);
    }
    println!("  Design time: {:.1} ms", elapsed.as_secs_f64() * 1000.0);

    // G4. Summary table
    println!("\n{}", "=".repeat(60));
    println!("  SUMMARY: COMPOSE OPERATION LATENCIES");
    println!("{}", "=".repeat(60));

    let f = encode_bool(false);
    let t_val = encode_bool(true);
    let tap4 = make_tapestry("sum", 4, true);
    let input4 = FloatVec::new(vec![0.8, 0.2, 0.1, 0.05]);

    let lat_not = bench(|| { std::hint::black_box(gate_not(&f).unwrap()); }, 100_000);
    let lat_and = bench(|| { std::hint::black_box(gate_and(&f, &t_val).unwrap()); }, 100_000);
    let lat_or = bench(|| { std::hint::black_box(gate_or(&f, &t_val).unwrap()); }, 100_000);

    let conf_cfg = ConfidenceConfig { max_observations: 10, confidence_threshold: 0.99, min_observations: 3 };
    let lat_conf = bench(|| {
        std::hint::black_box(observe_confident(&tap4, &[("x", &input4)], &conf_cfg).unwrap());
    }, 100);

    let stages: Vec<weaver::Tapestry> = (0..3)
        .map(|i| make_tapestry(&format!("sum{}", i), 4, true))
        .collect();
    let c = chain("sum", stages).unwrap();
    let lat_chain = bench(|| {
        std::hint::black_box(observe_chain(&c, &[("x", &input4)]).unwrap());
    }, 50_000);

    println!("  NOT gate:          {:.3} us", lat_not * 1e6);
    println!("  AND gate:          {:.3} us", lat_and * 1e6);
    println!("  OR gate:           {:.3} us", lat_or * 1e6);
    println!("  Confidence (10):   {:.1} us", lat_conf * 1e6);
    println!("  Chain observe (3): {:.1} us", lat_chain * 1e6);
    println!("{}", "=".repeat(60));
}
