//! Production benchmark: throughput, latency, scaling, and comparison.
//! Run: cargo test --test bench_production --release -- --nocapture
//!
//! Answers three questions:
//!   A. How fast? (throughput, latency per dim, game-frame budget)
//!   B. How does it scale? (dim 2→256, density matrix cost)
//!   C. How does it compare? (AXOL vs naive k-NN / majority vote)

use std::time::Instant;

use axol::types::*;
use axol::declare::*;
use axol::weaver;
use axol::observatory;
use axol::wave::{Wave, InterferencePattern};
use axol::relation::{Relation, Expectation};
use axol::dsl::parser::RelDirection;

// =========================================================================
// Timing helpers
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

fn make_tapestry(dim: usize, quantum: bool, seed: u64) -> weaver::Tapestry {
    let mut b = DeclarationBuilder::new("bench");
    b.input("x", dim).output("y")
        .relate("y", &["x"], RelationKind::Proportional)
        .quality(0.9, 0.8);
    weaver::weave(&b.build(), quantum, seed).unwrap()
}

fn make_input(dim: usize) -> FloatVec {
    let data: Vec<f32> = (0..dim)
        .map(|i| ((i as f32 * 0.7).sin() + 1.0) / 2.0)
        .collect();
    FloatVec::new(data)
}

// =========================================================================
// PART A: THROUGHPUT & LATENCY
// =========================================================================

#[test]
fn bench_throughput() {
    println!("\n{}", "=".repeat(70));
    println!("  PART A: THROUGHPUT & LATENCY");
    println!("  (Game frame budget: 16.67ms for 60fps, 8.33ms for 120fps)");
    println!("{}\n", "=".repeat(70));

    // A1. observe() throughput across dimensions
    println!("[A1] observatory::observe() latency & throughput");
    println!("  {:>6}  {:>10}  {:>12}  {:>10}  {:>12}", "dim", "latency", "ops/sec", "per frame", "quantum");
    println!("  {}", "-".repeat(60));

    for &dim in &[2, 4, 8, 16, 32, 64, 128] {
        let tapestry = make_tapestry(dim, true, 42);
        let input = make_input(dim);
        let iters = if dim <= 32 { 10_000 } else if dim <= 64 { 5_000 } else { 1_000 };

        let lat = bench(|| {
            std::hint::black_box(observatory::observe(&tapestry, &[("x", &input)]).unwrap());
        }, iters);

        let ops_per_sec = 1.0 / lat;
        let per_frame_60 = (0.01667 / lat) as u64;
        println!("  {:>6}  {:>8.2} us  {:>10.0} /s  {:>8} /fr  quantum=true",
            dim, lat * 1e6, ops_per_sec, per_frame_60);
    }

    // A2. classical vs quantum compare
    println!("\n[A2] Classical vs Quantum observe (dim=8)");
    let dim = 8;
    let input = make_input(dim);

    let tap_c = make_tapestry(dim, false, 42);
    let lat_c = bench(|| {
        std::hint::black_box(observatory::observe(&tap_c, &[("x", &input)]).unwrap());
    }, 50_000);

    let tap_q = make_tapestry(dim, true, 42);
    let lat_q = bench(|| {
        std::hint::black_box(observatory::observe(&tap_q, &[("x", &input)]).unwrap());
    }, 50_000);

    println!("  classical: {:.2} us  ({:.0} /s)", lat_c * 1e6, 1.0 / lat_c);
    println!("  quantum:   {:.2} us  ({:.0} /s)", lat_q * 1e6, 1.0 / lat_q);
    println!("  ratio:     {:.2}x", lat_q / lat_c);

    // A3. Wave operations breakdown
    println!("\n[A3] Wave operation latency breakdown (dim=8)");
    let w = Wave::from_classical(&make_input(8));

    let lat_gaze = bench(|| { std::hint::black_box(w.gaze()); }, 100_000);
    let lat_focus = bench(|| { std::hint::black_box(w.focus(0.5)); }, 100_000);
    let lat_observe = bench(|| { std::hint::black_box(w.observe()); }, 100_000);

    let w2 = Wave::from_classical(&FloatVec::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]));
    let lat_compose = bench(|| {
        std::hint::black_box(Wave::compose(&w, &w2, &InterferencePattern::Constructive).unwrap());
    }, 100_000);

    println!("  gaze:     {:>8.3} us", lat_gaze * 1e6);
    println!("  focus:    {:>8.3} us", lat_focus * 1e6);
    println!("  observe:  {:>8.3} us", lat_observe * 1e6);
    println!("  compose:  {:>8.3} us", lat_compose * 1e6);

    // A4. Relation operations
    println!("\n[A4] Relation operation latency (dim=8)");
    let from_w = Wave::from_classical(&make_input(8));
    let to_w = Wave::from_classical(&FloatVec::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]));

    let lat_rel_new = bench(|| {
        std::hint::black_box(Relation::new(
            "r", "a", "b", RelDirection::Bidir,
            &from_w, &to_w, InterferencePattern::Constructive,
        ).unwrap());
    }, 50_000);

    let mut rel = Relation::new("r", "a", "b", RelDirection::Bidir,
        &from_w, &to_w, InterferencePattern::Constructive).unwrap();
    let expect = Expectation::from_distribution("e", vec![0.3, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05], 0.8);

    let lat_expect = bench(|| {
        let mut r = rel.clone();
        std::hint::black_box(r.apply_expect(&expect).unwrap());
    }, 50_000);

    println!("  new:          {:>8.3} us", lat_rel_new * 1e6);
    println!("  apply_expect: {:>8.3} us", lat_expect * 1e6);

    // A5. Full pipeline: weave + observe
    println!("\n[A5] Full pipeline: weave + observe (dim=8)");
    let input8 = make_input(8);
    let lat_full = bench(|| {
        let tap = make_tapestry(8, true, 42);
        std::hint::black_box(observatory::observe(&tap, &[("x", &input8)]).unwrap());
    }, 1_000);
    println!("  weave+observe: {:.1} us  ({:.0} /s)", lat_full * 1e6, 1.0 / lat_full);

    let per_frame_60 = (0.01667 / lat_full) as u64;
    let per_frame_120 = (0.00833 / lat_full) as u64;
    println!("  per 60fps frame:  {}", per_frame_60);
    println!("  per 120fps frame: {}", per_frame_120);
}

// =========================================================================
// PART B: DIMENSION SCALING
// =========================================================================

#[test]
fn bench_scaling() {
    println!("\n{}", "=".repeat(70));
    println!("  PART B: DIMENSION SCALING (dim 2 → 256)");
    println!("{}\n", "=".repeat(70));

    // B1. Weave scaling
    println!("[B1] weave() latency by dimension");
    println!("  {:>6}  {:>12}  {:>10}", "dim", "latency", "ratio");
    println!("  {}", "-".repeat(35));
    let mut base_weave = 0.0;
    for &dim in &[2, 4, 8, 16, 32, 64, 128, 256] {
        let iters = if dim <= 32 { 500 } else if dim <= 64 { 50 } else if dim <= 128 { 10 } else { 3 };
        let lat = bench(|| {
            std::hint::black_box(make_tapestry(dim, true, 42));
        }, iters);
        if dim == 2 { base_weave = lat; }
        let ratio = lat / base_weave;
        println!("  {:>6}  {:>9.1} us  {:>8.1}x", dim, lat * 1e6, ratio);
    }

    // B2. Observe scaling
    println!("\n[B2] observe() latency by dimension");
    println!("  {:>6}  {:>12}  {:>10}  {:>12}", "dim", "latency", "ratio", "per frame");
    println!("  {}", "-".repeat(50));
    let mut base_obs = 0.0;
    for &dim in &[2, 4, 8, 16, 32, 64, 128, 256] {
        let tapestry = make_tapestry(dim, true, 42);
        let input = make_input(dim);
        let iters = if dim <= 32 { 5_000 } else if dim <= 64 { 1_000 } else if dim <= 128 { 100 } else { 10 };

        let lat = bench(|| {
            std::hint::black_box(observatory::observe(&tapestry, &[("x", &input)]).unwrap());
        }, iters);
        if dim == 2 { base_obs = lat; }
        let ratio = lat / base_obs;
        let per_frame = (0.01667 / lat) as u64;
        println!("  {:>6}  {:>9.1} us  {:>8.1}x  {:>10} /fr", dim, lat * 1e6, ratio, per_frame);
    }

    // B3. Wave operations scaling
    println!("\n[B3] Wave.compose() latency by dimension");
    println!("  {:>6}  {:>12}  {:>10}", "dim", "latency", "ratio");
    println!("  {}", "-".repeat(35));
    let mut base_compose = 0.0;
    for &dim in &[2, 4, 8, 16, 32, 64, 128, 256] {
        let input_a = make_input(dim);
        let data_b: Vec<f32> = (0..dim).map(|i| ((i as f32 * 1.3).cos() + 1.0) / 2.0).collect();
        let input_b = FloatVec::new(data_b);
        let wa = Wave::from_classical(&input_a);
        let wb = Wave::from_classical(&input_b);
        let iters = if dim <= 32 { 100_000 } else if dim <= 128 { 10_000 } else { 1_000 };

        let lat = bench(|| {
            std::hint::black_box(Wave::compose(&wa, &wb, &InterferencePattern::Constructive).unwrap());
        }, iters);
        if dim == 2 { base_compose = lat; }
        println!("  {:>6}  {:>9.3} us  {:>8.1}x", dim, lat * 1e6, lat / base_compose);
    }

    // B4. Density matrix operations scaling (the bottleneck)
    println!("\n[B4] DensityMatrix operations by dimension");
    println!("  {:>6}  {:>12}  {:>12}  {:>12}", "dim", "from_pure", "dephasing", "focus");
    println!("  {}", "-".repeat(55));
    for &dim in &[2, 4, 8, 16, 32, 64, 128, 256] {
        let input = make_input(dim);
        let w = Wave::from_classical(&input);
        let cv = w.amplitudes.clone();
        let iters = if dim <= 32 { 50_000 } else if dim <= 64 { 5_000 } else if dim <= 128 { 200 } else { 20 };

        let lat_pure = bench(|| {
            std::hint::black_box(DensityMatrix::from_pure_state(&cv));
        }, iters);

        let rho = DensityMatrix::from_pure_state(&cv);
        let lat_dephase = bench(|| {
            let k = axol::density::dephasing_channel(0.5, dim);
            std::hint::black_box(axol::density::apply_channel(&rho, &k));
        }, iters);

        let lat_focus = bench(|| {
            std::hint::black_box(w.focus(0.5));
        }, iters);

        println!("  {:>6}  {:>9.1} us  {:>9.1} us  {:>9.1} us",
            dim, lat_pure * 1e6, lat_dephase * 1e6, lat_focus * 1e6);
    }
}

// =========================================================================
// PART C: COMPARISON vs NAIVE METHODS
// =========================================================================

/// Simple k-NN classifier (k=1, Euclidean distance).
fn knn_classify(train: &[(Vec<f32>, usize)], query: &[f32]) -> usize {
    let mut best_dist = f32::MAX;
    let mut best_label = 0;
    for (features, label) in train {
        let dist: f32 = features.iter().zip(query.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        if dist < best_dist {
            best_dist = dist;
            best_label = *label;
        }
    }
    best_label
}

/// Majority vote classifier: pick the most frequent label in training data
/// weighted by inverse distance.
fn weighted_vote_classify(train: &[(Vec<f32>, usize)], query: &[f32], n_classes: usize) -> usize {
    let mut votes = vec![0.0_f64; n_classes];
    for (features, label) in train {
        let dist: f64 = features.iter().zip(query.iter())
            .map(|(a, b)| ((a - b) * (a - b)) as f64)
            .sum::<f64>()
            .sqrt();
        let weight = 1.0 / (dist + 1e-6);
        votes[*label] += weight;
    }
    votes.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// AXOL classifier: declare → weave → observe.
fn axol_classify(tapestry: &weaver::Tapestry, query: &[f32]) -> usize {
    let input = FloatVec::new(query.to_vec());
    let obs = observatory::observe(tapestry, &[("x", &input)]).unwrap();
    obs.value_index
}

#[test]
fn bench_comparison() {
    println!("\n{}", "=".repeat(70));
    println!("  PART C: AXOL vs NAIVE METHODS");
    println!("{}\n", "=".repeat(70));

    // XOR problem: 4D input → 2 classes
    let xor_train: Vec<(Vec<f32>, usize)> = vec![
        (vec![0.9, 0.1, 0.9, 0.1], 0),
        (vec![0.9, 0.1, 0.1, 0.9], 1),
        (vec![0.1, 0.9, 0.9, 0.1], 1),
        (vec![0.1, 0.9, 0.1, 0.9], 0),
    ];
    let xor_test: Vec<(Vec<f32>, usize)> = vec![
        (vec![0.85, 0.15, 0.85, 0.15], 0),
        (vec![0.85, 0.15, 0.15, 0.85], 1),
        (vec![0.15, 0.85, 0.85, 0.15], 1),
        (vec![0.15, 0.85, 0.15, 0.85], 0),
        (vec![0.7, 0.3, 0.8, 0.2], 0),
        (vec![0.8, 0.2, 0.2, 0.8], 1),
        (vec![0.3, 0.7, 0.7, 0.3], 1),
        (vec![0.2, 0.8, 0.3, 0.7], 0),
    ];

    // C1. Accuracy comparison on XOR
    println!("[C1] Accuracy on XOR problem (4D → 2 classes)");

    // k-NN
    let knn_correct: usize = xor_test.iter()
        .filter(|(q, label)| knn_classify(&xor_train, q) == *label)
        .count();

    // Weighted vote
    let wv_correct: usize = xor_test.iter()
        .filter(|(q, label)| weighted_vote_classify(&xor_train, q, 2) == *label)
        .count();

    // AXOL
    let tap = make_tapestry(4, true, 42);
    let axol_correct: usize = xor_test.iter()
        .filter(|(q, label)| axol_classify(&tap, q) == *label)
        .count();

    let total = xor_test.len();
    println!("  k-NN (k=1):       {}/{} ({:.0}%)", knn_correct, total, knn_correct as f64 / total as f64 * 100.0);
    println!("  Weighted vote:    {}/{} ({:.0}%)", wv_correct, total, wv_correct as f64 / total as f64 * 100.0);
    println!("  AXOL (quantum):   {}/{} ({:.0}%)", axol_correct, total, axol_correct as f64 / total as f64 * 100.0);

    // C2. Speed comparison
    println!("\n[C2] Speed comparison on XOR (per-query latency)");
    let query = vec![0.85_f32, 0.15, 0.15, 0.85];

    let lat_knn = bench(|| {
        std::hint::black_box(knn_classify(&xor_train, &query));
    }, 500_000);

    let lat_wv = bench(|| {
        std::hint::black_box(weighted_vote_classify(&xor_train, &query, 2));
    }, 500_000);

    let lat_axol = bench(|| {
        std::hint::black_box(axol_classify(&tap, &query));
    }, 50_000);

    println!("  k-NN:         {:>8.3} us  ({:.0} /s)", lat_knn * 1e6, 1.0 / lat_knn);
    println!("  Weighted vote:{:>8.3} us  ({:.0} /s)", lat_wv * 1e6, 1.0 / lat_wv);
    println!("  AXOL:         {:>8.3} us  ({:.0} /s)", lat_axol * 1e6, 1.0 / lat_axol);
    println!("  AXOL / k-NN:  {:.1}x slower", lat_axol / lat_knn);

    // C3. Scaling comparison: as training data grows
    println!("\n[C3] k-NN scaling vs AXOL scaling (query latency as data grows)");
    println!("  {:>10}  {:>12}  {:>12}  {:>10}", "train size", "k-NN", "AXOL", "ratio");
    println!("  {}", "-".repeat(50));

    for &n in &[4, 16, 64, 256, 1024] {
        // Generate synthetic training data
        let train: Vec<(Vec<f32>, usize)> = (0..n)
            .map(|i| {
                let data: Vec<f32> = (0..4)
                    .map(|j| ((i * 7 + j * 13) as f32 * 0.1).sin().abs())
                    .collect();
                let label = (i % 4) as usize;
                (data, label)
            })
            .collect();

        let query_data = vec![0.5_f32, 0.3, 0.7, 0.1];

        let lat_knn_n = bench(|| {
            std::hint::black_box(knn_classify(&train, &query_data));
        }, if n <= 64 { 100_000 } else { 10_000 });

        // AXOL latency is constant (doesn't depend on training data size)
        let lat_axol_n = bench(|| {
            std::hint::black_box(axol_classify(&tap, &query_data));
        }, 10_000);

        println!("  {:>10}  {:>9.3} us  {:>9.3} us  {:>8.1}x",
            n, lat_knn_n * 1e6, lat_axol_n * 1e6, lat_axol_n / lat_knn_n);
    }

    // C4. Multi-class problem: 8D → 8 classes
    println!("\n[C4] Multi-class problem (8D → 8 classes)");
    let train_8d: Vec<(Vec<f32>, usize)> = (0..64)
        .map(|i| {
            let data: Vec<f32> = (0..8)
                .map(|j| {
                    let base = if j == (i % 8) { 0.8 } else { 0.1 };
                    base + ((i * 17 + j * 31) as f32 * 0.01).sin() * 0.1
                })
                .collect();
            (data, i % 8)
        })
        .collect();

    let test_8d: Vec<(Vec<f32>, usize)> = (0..16)
        .map(|i| {
            let data: Vec<f32> = (0..8)
                .map(|j| {
                    let base = if j == (i % 8) { 0.75 } else { 0.15 };
                    base + ((i * 23 + j * 37) as f32 * 0.01).sin() * 0.05
                })
                .collect();
            (data, i % 8)
        })
        .collect();

    let tap_8 = make_tapestry(8, true, 42);
    let knn_8_correct: usize = test_8d.iter()
        .filter(|(q, label)| knn_classify(&train_8d, q) == *label)
        .count();
    let axol_8_correct: usize = test_8d.iter()
        .filter(|(q, label)| axol_classify(&tap_8, q) == *label)
        .count();

    println!("  k-NN (k=1):     {}/{} ({:.0}%)", knn_8_correct, test_8d.len(),
        knn_8_correct as f64 / test_8d.len() as f64 * 100.0);
    println!("  AXOL (quantum): {}/{} ({:.0}%)", axol_8_correct, test_8d.len(),
        axol_8_correct as f64 / test_8d.len() as f64 * 100.0);

    // Speed on 8D
    let query_8d = vec![0.8_f32, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
    let lat_knn_8d = bench(|| {
        std::hint::black_box(knn_classify(&train_8d, &query_8d));
    }, 100_000);
    let lat_axol_8d = bench(|| {
        std::hint::black_box(axol_classify(&tap_8, &query_8d));
    }, 10_000);
    println!("  k-NN latency:   {:.3} us", lat_knn_8d * 1e6);
    println!("  AXOL latency:   {:.3} us", lat_axol_8d * 1e6);

    // C5. AXOL unique capability: partial observation (gaze/focus)
    println!("\n[C5] AXOL unique: partial observation (no classical equivalent)");
    let w = Wave::from_classical(&make_input(8));
    let probs_gaze = w.gaze();
    let focused_30 = w.focus(0.3);
    let probs_f30 = focused_30.gaze();
    let focused_70 = w.focus(0.7);
    let probs_f70 = focused_70.gaze();
    let (idx, _) = w.observe();

    println!("  gaze (t=0.0):     {:?}", fmt_probs(&probs_gaze));
    println!("  focus 0.3 (t≈0.3):{:?}", fmt_probs(&probs_f30));
    println!("  focus 0.7 (t≈0.7):{:?}", fmt_probs(&probs_f70));
    println!("  observe (t=1.0):  class={}", idx);
    println!("  → Classical methods give only the final class.");
    println!("    AXOL gives probability landscape at every certainty level.");

    // C6. AXOL unique: interference composition
    println!("\n[C6] AXOL unique: interference composition (no classical equivalent)");
    let w_a = Wave::from_classical(&FloatVec::new(vec![0.8, 0.1, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0]));
    let w_b = Wave::from_classical(&FloatVec::new(vec![0.0, 0.0, 0.0, 0.0, 0.05, 0.05, 0.1, 0.8]));
    let patterns = [
        ("Constructive", InterferencePattern::Constructive),
        ("Additive",     InterferencePattern::Additive),
        ("Destructive",  InterferencePattern::Destructive),
    ];
    for (name, pattern) in &patterns {
        let composed = Wave::compose(&w_a, &w_b, pattern).unwrap();
        let (idx, _) = composed.observe();
        println!("  {} → class={}, probs={:?}", name, idx, fmt_probs(&composed.gaze()));
    }
    println!("  → Different composition strategies yield different decisions.");
    println!("    Classical methods have no equivalent to interference patterns.");
}

fn fmt_probs(probs: &[f64]) -> Vec<String> {
    probs.iter().map(|p| format!("{:.3}", p)).collect()
}

// =========================================================================
// PART D: GAME FRAME BUDGET ANALYSIS
// =========================================================================

#[test]
fn bench_frame_budget() {
    println!("\n{}", "=".repeat(70));
    println!("  PART D: GAME FRAME BUDGET ANALYSIS");
    println!("{}\n", "=".repeat(70));

    println!("[D1] How many NPC decisions per frame?");
    println!("  Scenario: Each NPC does one observe() per frame");
    println!("  Budget: 2ms of AI time (typical game allocation)\n");

    println!("  {:>6}  {:>10}  {:>12}  {:>12}  {:>12}", "dim", "latency", "NPCs @2ms", "NPCs @4ms", "NPCs @8ms");
    println!("  {}", "-".repeat(60));

    for &dim in &[4, 8, 16, 32] {
        let tapestry = make_tapestry(dim, true, 42);
        let input = make_input(dim);
        let lat = bench(|| {
            std::hint::black_box(observatory::observe(&tapestry, &[("x", &input)]).unwrap());
        }, 10_000);

        let npcs_2ms = (0.002 / lat) as u64;
        let npcs_4ms = (0.004 / lat) as u64;
        let npcs_8ms = (0.008 / lat) as u64;
        println!("  {:>6}  {:>7.1} us  {:>10}  {:>10}  {:>10}",
            dim, lat * 1e6, npcs_2ms, npcs_4ms, npcs_8ms);
    }

    // D2. Multi-observe per NPC (gaze → focus → observe pipeline)
    println!("\n[D2] Full NPC decision pipeline: gaze → focus → observe");
    for &dim in &[4, 8, 16, 32] {
        let tapestry = make_tapestry(dim, true, 42);
        let input = make_input(dim);

        let lat_pipeline = bench(|| {
            // Step 1: gaze (free look at probabilities)
            let wave = observatory::compute_wave(&tapestry, &[("x", &input)]).unwrap();
            let _probs = wave.gaze();
            // Step 2: partial collapse (narrow options)
            let focused = wave.focus(0.5);
            let _focused_probs = focused.gaze();
            // Step 3: final decision
            let (_idx, _collapsed) = focused.observe();
        }, 10_000);

        let npcs_2ms = (0.002 / lat_pipeline) as u64;
        println!("  dim={:>3}: {:.1} us/npc  → {} NPCs in 2ms AI budget",
            dim, lat_pipeline * 1e6, npcs_2ms);
    }
}
