//! Comprehensive Wave System Benchmark
//!
//! Tests the full Wave variable system across diverse conditions:
//!   - All 5 interference patterns
//!   - Collapse spectrum (gamma 0.0 → 1.0)
//!   - Dimension scaling (2 → 128)
//!   - Input distributions (uniform, peaked, sparse, random)
//!   - Full pipeline: declare → weave → wave → focus → observe
//!   - Wave vs classical path comparison
//!   - Multi-wave composition chains

use std::time::Instant;
use num_complex::Complex64;

use axol::types::*;
use axol::ops;
use axol::density;
use axol::wave::{Wave, InterferencePattern};
use axol::collapse::CollapseMetrics;
use axol::declare::*;
use axol::weaver;
use axol::observatory;

fn main() {
    println!("================================================================");
    println!("  AXOL Wave System — Comprehensive Benchmark Report");
    println!("  Date: 2026-02-12");
    println!("================================================================\n");

    bench_1_compose_patterns();
    bench_2_dimension_scaling();
    bench_3_collapse_spectrum();
    bench_4_input_distributions();
    bench_5_full_pipeline();
    bench_6_wave_vs_classical();
    bench_7_compose_chain();
    bench_8_information_metrics();
    bench_9_density_scaling();
    bench_10_wave_reuse();

    println!("\n================================================================");
    println!("  Benchmark Complete");
    println!("================================================================");
}

// =========================================================================
// [1] Interference Pattern Comparison
// =========================================================================
fn bench_1_compose_patterns() {
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ [1] Interference Pattern Comparison (dim=8)                 │");
    println!("└──────────────────────────────────────────────────────────────┘");
    println!("  Pattern determines HOW two Waves combine. All C=0.\n");

    let dim = 8;
    let data_a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.7).sin().abs() + 0.1).collect();
    let data_b: Vec<f32> = (0..dim).map(|i| (i as f32 * 1.3).cos().abs() + 0.1).collect();
    let wa = Wave::from_classical(&FloatVec::new(data_a));
    let wb = Wave::from_classical(&FloatVec::new(data_b));

    let patterns = [
        ("Constructive <~>", InterferencePattern::Constructive),
        ("Additive     <+>", InterferencePattern::Additive),
        ("Multiplicat. <*>", InterferencePattern::Multiplicative),
        ("Destructive  <!>", InterferencePattern::Destructive),
        ("Conditional  <?>", InterferencePattern::Conditional),
    ];

    println!("  {:>18}  {:>8}  {:>10}  {:>8}  {:>6}", "pattern", "time_us", "dominant", "max_p", "t");
    println!("  {}", "-".repeat(58));

    for (name, pattern) in &patterns {
        let iters = 200_000;
        let start = Instant::now();
        let mut result = Wave::from_classical(&FloatVec::new(vec![0.5; dim]));
        for _ in 0..iters {
            result = std::hint::black_box(Wave::compose(&wa, &wb, pattern).unwrap());
        }
        let us = start.elapsed().as_secs_f64() / iters as f64 * 1e6;

        let probs = result.probabilities();
        let max_p = probs.iter().cloned().fold(0.0_f64, f64::max);
        println!("  {:>18}  {:>7.3}  {:>10}  {:>7.4}  {:>5.2}",
            name, us, result.dominant(), max_p, result.t);
    }

    // Show probability distributions
    println!("\n  Probability distributions:");
    println!("  {:>18}  {}", "pattern", "distribution (top 4)");
    println!("  {}", "-".repeat(58));
    for (name, pattern) in &patterns {
        let result = Wave::compose(&wa, &wb, pattern).unwrap();
        let probs = result.probabilities();
        let mut indexed: Vec<(usize, f64)> = probs.iter().enumerate().map(|(i,&p)| (i,p)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top: Vec<String> = indexed.iter().take(4)
            .map(|(i, p)| format!("[{}]={:.3}", i, p))
            .collect();
        println!("  {:>18}  {}", name, top.join("  "));
    }
    println!();
}

// =========================================================================
// [2] Dimension Scaling
// =========================================================================
fn bench_2_dimension_scaling() {
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ [2] Dimension Scaling                                       │");
    println!("└──────────────────────────────────────────────────────────────┘");
    println!("  How operations scale with Hilbert space dimension.\n");

    println!("  {:>5}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}", "dim", "compose", "gaze", "focus_0.3", "focus_0.7", "observe");
    println!("  {:>5}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}", "", "(us)", "(us)", "(us)", "(us)", "(us)");
    println!("  {}", "-".repeat(65));

    for dim in [2, 4, 8, 16, 32, 64, 128] {
        let data_a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.7).sin()).collect();
        let data_b: Vec<f32> = (0..dim).map(|i| (i as f32 * 1.3).cos()).collect();
        let wa = Wave::from_classical(&FloatVec::new(data_a));
        let wb = Wave::from_classical(&FloatVec::new(data_b));

        let iters = if dim <= 32 { 100_000 } else { 10_000 };

        // compose
        let start = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(Wave::compose(&wa, &wb, &InterferencePattern::Constructive).unwrap());
        }
        let compose_us = start.elapsed().as_secs_f64() / iters as f64 * 1e6;

        // gaze
        let start = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(wa.gaze());
        }
        let gaze_us = start.elapsed().as_secs_f64() / iters as f64 * 1e6;

        // focus 0.3
        let start = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(wa.focus(0.3));
        }
        let focus03_us = start.elapsed().as_secs_f64() / iters as f64 * 1e6;

        // focus 0.7
        let start = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(wa.focus(0.7));
        }
        let focus07_us = start.elapsed().as_secs_f64() / iters as f64 * 1e6;

        // observe
        let start = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(wa.observe());
        }
        let observe_us = start.elapsed().as_secs_f64() / iters as f64 * 1e6;

        println!("  {:>5}  {:>9.3}  {:>9.3}  {:>9.3}  {:>9.3}  {:>9.3}",
            dim, compose_us, gaze_us, focus03_us, focus07_us, observe_us);
    }
    println!();
}

// =========================================================================
// [3] Collapse Spectrum (gamma sweep)
// =========================================================================
fn bench_3_collapse_spectrum() {
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ [3] Collapse Spectrum — gamma sweep (dim=8)                 │");
    println!("└──────────────────────────────────────────────────────────────┘");
    println!("  Continuous transition from superposition to classical.\n");

    let dim = 8;
    // Strongly peaked input
    let mut data = vec![0.1f32; dim];
    data[2] = 0.9;
    data[5] = 0.6;
    let wave = Wave::from_classical(&FloatVec::new(data));

    println!("  {:>6}  {:>5}  {:>8}  {:>8}  {:>8}  {:>28}", "gamma", "t", "dom_idx", "max_p", "entropy", "top-3 probs");
    println!("  {}", "-".repeat(73));

    for gamma_i in 0..=20 {
        let gamma = gamma_i as f64 * 0.05;
        let focused = if gamma == 0.0 { wave.clone() } else { wave.focus(gamma) };
        let probs = focused.probabilities();
        let max_p = probs.iter().cloned().fold(0.0_f64, f64::max);
        let entropy = shannon_entropy(&probs);
        let dom = focused.dominant();

        let mut indexed: Vec<(usize, f64)> = probs.iter().enumerate().map(|(i,&p)| (i,p)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top: Vec<String> = indexed.iter().take(3)
            .map(|(i, p)| format!("[{}]={:.3}", i, p))
            .collect();

        println!("  {:>5.2}  {:>5.3}  {:>7}  {:>7.4}  {:>7.4}  {}",
            gamma, focused.t, dom, max_p, entropy, top.join(" "));
    }
    println!();
}

// =========================================================================
// [4] Input Distribution Comparison
// =========================================================================
fn bench_4_input_distributions() {
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ [4] Input Distribution Comparison (dim=16)                  │");
    println!("└──────────────────────────────────────────────────────────────┘");
    println!("  How input shape affects Wave behavior.\n");

    let dim = 16;

    let distributions: Vec<(&str, Vec<f32>)> = vec![
        ("uniform", vec![1.0 / (dim as f32).sqrt(); dim]),
        ("peaked [0]", {
            let mut d = vec![0.01f32; dim]; d[0] = 1.0; d
        }),
        ("bimodal [2,7]", {
            let mut d = vec![0.01f32; dim]; d[2] = 0.8; d[7] = 0.7; d
        }),
        ("sparse (3 nz)", {
            let mut d = vec![0.0f32; dim]; d[1] = 0.5; d[5] = 0.3; d[12] = 0.8; d
        }),
        ("gradient", (0..dim).map(|i| (i as f32 + 1.0) / dim as f32).collect()),
        ("sine", (0..dim).map(|i| (i as f32 * std::f32::consts::PI / dim as f32).sin()).collect()),
    ];

    println!("  {:>15}  {:>8}  {:>8}  {:>8}  {:>8}  {:>10}", "distribution", "entropy", "max_p", "eff_dim", "focus.3", "compose_us");
    println!("  {}", "-".repeat(65));

    let b_data: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.3).cos().abs() + 0.05).collect();
    let wb = Wave::from_classical(&FloatVec::new(b_data));

    for (name, data) in &distributions {
        let wa = Wave::from_classical(&FloatVec::new(data.clone()));
        let probs = wa.probabilities();
        let entropy = shannon_entropy(&probs);
        let max_p = probs.iter().cloned().fold(0.0_f64, f64::max);
        let eff_dim = effective_dimension(&probs);

        let focused = wa.focus(0.3);
        let focus_max = focused.probabilities().iter().cloned().fold(0.0_f64, f64::max);

        let iters = 100_000;
        let start = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(Wave::compose(&wa, &wb, &InterferencePattern::Constructive).unwrap());
        }
        let compose_us = start.elapsed().as_secs_f64() / iters as f64 * 1e6;

        println!("  {:>15}  {:>7.4}  {:>7.4}  {:>7.2}  {:>7.4}  {:>9.3}",
            name, entropy, max_p, eff_dim, focus_max, compose_us);
    }
    println!();
}

// =========================================================================
// [5] Full Pipeline: declare → weave → wave → focus → observe
// =========================================================================
fn bench_5_full_pipeline() {
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ [5] Full Pipeline: declare → weave → wave → observe        │");
    println!("└──────────────────────────────────────────────────────────────┘");
    println!("  End-to-end timing through the entire AXOL stack.\n");

    let configs = [
        ("binary (2d)", 2, true),
        ("small  (4d)", 4, true),
        ("medium (8d)", 8, true),
        ("large (16d)", 16, true),
        ("xl   (32d)", 32, true),
        ("classical 8d", 8, false),
    ];

    println!("  {:>15}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
        "config", "declare", "weave", "gaze", "glimpse", "observe", "total");
    println!("  {:>15}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
        "", "(us)", "(ms)", "(us)", "(us)", "(us)", "(ms)");
    println!("  {}", "-".repeat(75));

    for (name, dim, quantum) in &configs {
        let total_start = Instant::now();

        // Declare
        let decl_start = Instant::now();
        let mut builder = DeclarationBuilder::new("bench");
        builder
            .input("x", *dim)
            .output("y")
            .relate("y", &["x"], RelationKind::Proportional)
            .quality(0.9, 0.8);
        let decl = builder.build();
        let decl_us = decl_start.elapsed().as_secs_f64() * 1e6;

        // Weave
        let weave_start = Instant::now();
        let tapestry = weaver::weave(&decl, *quantum, 42).unwrap();
        let weave_ms = weave_start.elapsed().as_secs_f64() * 1000.0;

        // Prepare input
        let input_data: Vec<f32> = (0..*dim).map(|i| (i as f32 * 0.5).sin()).collect();
        let input_fv = FloatVec::new(input_data);
        let inputs: Vec<(&str, &FloatVec)> = vec![("x", &input_fv)];

        // Gaze (C=0)
        let gaze_start = Instant::now();
        let gaze_iters = 10_000;
        for _ in 0..gaze_iters {
            std::hint::black_box(observatory::gaze(&tapestry, &inputs).unwrap());
        }
        let gaze_us = gaze_start.elapsed().as_secs_f64() / gaze_iters as f64 * 1e6;

        // Glimpse (C=0.5)
        let glimpse_start = Instant::now();
        let glimpse_iters = 10_000;
        for _ in 0..glimpse_iters {
            std::hint::black_box(observatory::glimpse(&tapestry, &inputs, 0.5).unwrap());
        }
        let glimpse_us = glimpse_start.elapsed().as_secs_f64() / glimpse_iters as f64 * 1e6;

        // Observe (C=1)
        let obs_start = Instant::now();
        let obs_iters = 10_000;
        for _ in 0..obs_iters {
            std::hint::black_box(observatory::observe(&tapestry, &inputs).unwrap());
        }
        let obs_us = obs_start.elapsed().as_secs_f64() / obs_iters as f64 * 1e6;

        let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;

        println!("  {:>15}  {:>7.1}  {:>7.3}  {:>7.3}  {:>7.3}  {:>7.3}  {:>7.1}",
            name, decl_us, weave_ms, gaze_us, glimpse_us, obs_us, total_ms);
    }
    println!();
}

// =========================================================================
// [6] Wave Path vs Classical Path
// =========================================================================
fn bench_6_wave_vs_classical() {
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ [6] Wave Path vs Classical Observe — Same Answer, Less C    │");
    println!("└──────────────────────────────────────────────────────────────┘");
    println!("  Compare: gaze (C=0) vs observe (C=1) vs reobserve x10 (C=10)\n");

    for dim in [4, 8, 16] {
        let mut builder = DeclarationBuilder::new("cmp");
        builder.input("x", dim).output("y")
            .relate("y", &["x"], RelationKind::Proportional)
            .quality(0.9, 0.8);
        let decl = builder.build();
        let tapestry = weaver::weave(&decl, true, 42).unwrap();

        let input_data: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.5).sin()).collect();
        let input_fv = FloatVec::new(input_data);
        let inputs: Vec<(&str, &FloatVec)> = vec![("x", &input_fv)];

        // Gaze
        let wave = observatory::gaze(&tapestry, &inputs).unwrap();
        let gaze_dom = wave.dominant();
        let gaze_probs = wave.probabilities();

        // Observe
        let obs = observatory::observe(&tapestry, &inputs).unwrap();

        // Reobserve x10
        let reobs = observatory::reobserve(&tapestry, &inputs, 10).unwrap();

        println!("  dim={}: gaze_dom={} observe_idx={} reobs_idx={} (all agree={})",
            dim, gaze_dom, obs.value_index, reobs.value_index,
            gaze_dom == obs.value_index && obs.value_index == reobs.value_index);
        println!("    gaze   C=0  : max_p={:.4}", gaze_probs.iter().cloned().fold(0.0_f64, f64::max));
        println!("    observe C=1 : max_p={:.4}", obs.probabilities.data.iter().cloned().fold(0.0_f32, f32::max));
        println!("    reobs  C=10 : max_p={:.4}", reobs.probabilities.data.iter().cloned().fold(0.0_f32, f32::max));
    }
    println!();
}

// =========================================================================
// [7] Multi-Wave Composition Chain
// =========================================================================
fn bench_7_compose_chain() {
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ [7] Multi-Wave Composition Chain                            │");
    println!("└──────────────────────────────────────────────────────────────┘");
    println!("  Composing N waves sequentially. All C=0.\n");

    let dim = 8;

    println!("  {:>8}  {:>10}  {:>10}  {:>8}  {:>8}",
        "n_waves", "total_us", "per_op_us", "dom_idx", "max_p");
    println!("  {}", "-".repeat(52));

    for n in [2, 3, 4, 5, 8, 10, 16] {
        let waves: Vec<Wave> = (0..n).map(|j| {
            let data: Vec<f32> = (0..dim).map(|i| ((i * j + 1) as f32 * 0.37).sin()).collect();
            Wave::from_classical(&FloatVec::new(data))
        }).collect();
        let wave_refs: Vec<&Wave> = waves.iter().collect();
        let patterns: Vec<InterferencePattern> = (0..n-1).map(|j| {
            match j % 3 {
                0 => InterferencePattern::Constructive,
                1 => InterferencePattern::Multiplicative,
                _ => InterferencePattern::Additive,
            }
        }).collect();
        let pattern_refs: Vec<&InterferencePattern> = patterns.iter().collect();

        let iters = 50_000;
        let start = Instant::now();
        let mut result = waves[0].clone();
        for _ in 0..iters {
            result = std::hint::black_box(
                Wave::compose_many(&wave_refs, &pattern_refs).unwrap()
            );
        }
        let total_us = start.elapsed().as_secs_f64() / iters as f64 * 1e6;
        let per_op = total_us / (n - 1) as f64;

        let probs = result.probabilities();
        let max_p = probs.iter().cloned().fold(0.0_f64, f64::max);
        println!("  {:>8}  {:>9.3}  {:>9.3}  {:>8}  {:>7.4}",
            n, total_us, per_op, result.dominant(), max_p);
    }
    println!();
}

// =========================================================================
// [8] Information Metrics Under Composition
// =========================================================================
fn bench_8_information_metrics() {
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ [8] Information Flow Under Composition                      │");
    println!("└──────────────────────────────────────────────────────────────┘");
    println!("  How composition affects information content.\n");

    let dim = 8;
    let data_a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.7).sin().abs() + 0.05).collect();
    let data_b: Vec<f32> = (0..dim).map(|i| (i as f32 * 1.3).cos().abs() + 0.05).collect();
    let wa = Wave::from_classical(&FloatVec::new(data_a));
    let wb = Wave::from_classical(&FloatVec::new(data_b));

    let ha = shannon_entropy(&wa.probabilities());
    let hb = shannon_entropy(&wb.probabilities());
    println!("  input A: entropy={:.4}  eff_dim={:.2}", ha, effective_dimension(&wa.probabilities()));
    println!("  input B: entropy={:.4}  eff_dim={:.2}", hb, effective_dimension(&wb.probabilities()));
    println!();

    println!("  {:>18}  {:>8}  {:>8}  {:>8}  {:>10}", "pattern", "H_out", "delta_H", "eff_dim", "purity");
    println!("  {}", "-".repeat(60));

    let patterns = [
        ("Constructive", InterferencePattern::Constructive),
        ("Additive", InterferencePattern::Additive),
        ("Multiplicative", InterferencePattern::Multiplicative),
        ("Destructive", InterferencePattern::Destructive),
        ("Conditional", InterferencePattern::Conditional),
    ];

    for (name, pattern) in &patterns {
        let result = Wave::compose(&wa, &wb, pattern).unwrap();
        let probs = result.probabilities();
        let h_out = shannon_entropy(&probs);
        let rho = result.to_density();
        let purity = rho.purity();

        println!("  {:>18}  {:>7.4}  {:>+7.4}  {:>7.2}  {:>9.6}",
            name, h_out, h_out - ha, effective_dimension(&probs), purity);
    }

    // Compose then focus
    println!("\n  After compose(Constructive) + focus(gamma):");
    println!("  {:>6}  {:>8}  {:>8}  {:>8}  {:>10}", "gamma", "H", "eff_dim", "max_p", "purity");
    println!("  {}", "-".repeat(48));

    let composed = Wave::compose(&wa, &wb, &InterferencePattern::Constructive).unwrap();
    for gamma_i in [0, 10, 30, 50, 70, 90, 100] {
        let gamma = gamma_i as f64 / 100.0;
        let focused = composed.focus(gamma);
        let probs = focused.probabilities();
        let h = shannon_entropy(&probs);
        let max_p = probs.iter().cloned().fold(0.0_f64, f64::max);
        let rho = focused.to_density();

        println!("  {:>5.2}  {:>7.4}  {:>7.2}  {:>7.4}  {:>9.6}",
            gamma, h, effective_dimension(&probs), max_p, rho.purity());
    }
    println!();
}

// =========================================================================
// [9] Density Matrix Scaling
// =========================================================================
fn bench_9_density_scaling() {
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ [9] Density Matrix Operations — Scaling                     │");
    println!("└──────────────────────────────────────────────────────────────┘");
    println!("  Bottleneck analysis: density matrix dominates focus cost.\n");

    println!("  {:>5}  {:>12}  {:>12}  {:>12}  {:>12}",
        "dim", "from_pure", "purity", "dephase", "von_neumann");
    println!("  {:>5}  {:>12}  {:>12}  {:>12}  {:>12}",
        "", "(us)", "(us)", "(us)", "(us)");
    println!("  {}", "-".repeat(60));

    for dim in [2, 4, 8, 16, 32, 64] {
        let psi = ComplexVec::new(
            (0..dim).map(|i| Complex64::new((i as f64 * 0.7).sin(), (i as f64 * 0.3).cos())).collect()
        ).normalized();

        let iters = if dim <= 16 { 50_000 } else { 5_000 };

        // from_pure_state
        let start = Instant::now();
        let mut rho = DensityMatrix::from_pure_state(&psi);
        for _ in 0..iters {
            rho = std::hint::black_box(DensityMatrix::from_pure_state(&psi));
        }
        let from_pure_us = start.elapsed().as_secs_f64() / iters as f64 * 1e6;

        // purity
        let start = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(rho.purity());
        }
        let purity_us = start.elapsed().as_secs_f64() / iters as f64 * 1e6;

        // dephasing channel
        let kraus = density::dephasing_channel(0.3, dim);
        let start = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(density::apply_channel(&rho, &kraus));
        }
        let dephase_us = start.elapsed().as_secs_f64() / iters as f64 * 1e6;

        // von neumann entropy
        let start = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(density::von_neumann_entropy(&rho));
        }
        let vn_us = start.elapsed().as_secs_f64() / iters as f64 * 1e6;

        println!("  {:>5}  {:>11.3}  {:>11.3}  {:>11.3}  {:>11.3}",
            dim, from_pure_us, purity_us, dephase_us, vn_us);
    }
    println!();
}

// =========================================================================
// [10] Wave Reuse — compose once, read many
// =========================================================================
fn bench_10_wave_reuse() {
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ [10] Wave Reuse — compose once, gaze many (C stays 0)      │");
    println!("└──────────────────────────────────────────────────────────────┘");
    println!("  Key advantage: Wave is computed once, read N times.\n");

    for dim in [4, 8, 16, 32] {
        let data_a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.7).sin()).collect();
        let data_b: Vec<f32> = (0..dim).map(|i| (i as f32 * 1.3).cos()).collect();
        let wa = Wave::from_classical(&FloatVec::new(data_a));
        let wb = Wave::from_classical(&FloatVec::new(data_b));
        let composed = Wave::compose(&wa, &wb, &InterferencePattern::Constructive).unwrap();

        // Time N gaze reads of the same composed wave
        let n_reads = 1_000_000;
        let start = Instant::now();
        for _ in 0..n_reads {
            std::hint::black_box(composed.gaze());
        }
        let total_ms = start.elapsed().as_secs_f64() * 1000.0;
        let per_read_ns = start.elapsed().as_secs_f64() / n_reads as f64 * 1e9;

        // Compare: N observe calls (each would collapse)
        let obs_iters = 100_000;
        let start = Instant::now();
        for _ in 0..obs_iters {
            std::hint::black_box(composed.observe());
        }
        let per_obs_ns = start.elapsed().as_secs_f64() / obs_iters as f64 * 1e9;

        println!("  dim={:>3}: gaze x1M = {:.1}ms ({:.1}ns/read)  vs  observe ({:.1}ns/call)  ratio={:.1}x",
            dim, total_ms, per_read_ns, per_obs_ns, per_obs_ns / per_read_ns);
    }
    println!();
}

// =========================================================================
// Helpers
// =========================================================================

fn shannon_entropy(probs: &[f64]) -> f64 {
    let mut h = 0.0;
    for &p in probs {
        if p > 1e-15 {
            h -= p * p.ln();
        }
    }
    h
}

fn effective_dimension(probs: &[f64]) -> f64 {
    let h = shannon_entropy(probs);
    h.exp()
}
