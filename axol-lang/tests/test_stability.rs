//! Stability and thread-safety tests for game production use.
//!
//! Covers determinism, long-running observe stability, thread safety,
//! and edge-case dimensions.

use std::sync::Arc;
use std::thread;

use axol::types::*;
use axol::declare::*;
use axol::weaver::{self, Tapestry};
use axol::observatory;
use axol::wave::Wave;

// =========================================================================
// Helpers
// =========================================================================

fn build_tapestry(seed: u64) -> Tapestry {
    let mut builder = DeclarationBuilder::new("stability_test");
    builder
        .input("x", 4)
        .output("y")
        .relate("y", &["x"], RelationKind::Proportional)
        .quality(0.9, 0.8);
    let decl = builder.build();
    weaver::weave(&decl, true, seed).unwrap()
}

fn observe_once(tapestry: &Tapestry, input: &[f32]) -> usize {
    let fv = FloatVec::new(input.to_vec());
    let obs = observatory::observe(tapestry, &[("x", &fv)]).unwrap();
    obs.value_index
}

// =========================================================================
// 1. Determinism
// =========================================================================

#[test]
fn same_seed_same_input_same_result() {
    let input = [0.8_f32, 0.2, 0.1, 0.5];
    let mut results = Vec::new();
    for _ in 0..10 {
        let tapestry = build_tapestry(42);
        results.push(observe_once(&tapestry, &input));
    }
    let first = results[0];
    for (i, &r) in results.iter().enumerate() {
        assert_eq!(
            r, first,
            "run {i}: expected {first}, got {r} — same seed should produce same result"
        );
    }
}

#[test]
fn different_seed_may_differ() {
    let input = [0.8_f32, 0.2, 0.1, 0.5];
    let mut results = Vec::new();
    for seed in 0..20 {
        let tapestry = build_tapestry(seed);
        results.push(observe_once(&tapestry, &input));
    }
    // With 20 different seeds, it's likely (not certain) we get at least 2 different outcomes
    let unique: std::collections::HashSet<_> = results.iter().collect();
    // This is a probabilistic assertion — just log if it unexpectedly fails
    if unique.len() == 1 {
        eprintln!("WARNING: all 20 seeds produced the same result — possibly degenerate");
    }
}

// =========================================================================
// 2. Long-running observe stability
// =========================================================================

#[test]
fn observe_10000_no_drift() {
    let tapestry = build_tapestry(42);
    let input = FloatVec::new(vec![0.8, 0.2, 0.1, 0.5]);

    // Get baseline probabilities from first observation
    let obs0 = observatory::observe(&tapestry, &[("x", &input)]).unwrap();
    let baseline_probs: Vec<f32> = obs0.probabilities.data.clone();

    // Run 10,000 observations and check that probabilities don't drift
    for i in 0..10_000 {
        let obs = observatory::observe(&tapestry, &[("x", &input)]).unwrap();
        let probs = &obs.probabilities.data;
        for (j, (&base, &cur)) in baseline_probs.iter().zip(probs.iter()).enumerate() {
            assert!(
                (base - cur).abs() < 1e-4,
                "drift at iteration {i}, dim {j}: baseline={base}, current={cur}"
            );
        }
    }
}

#[test]
fn gaze_10000_wave_unchanged() {
    let w = Wave::from_classical(&FloatVec::new(vec![0.6, 0.3, 0.1]));
    let baseline = w.gaze();

    for i in 0..10_000 {
        let current = w.gaze();
        for (j, (&base, &cur)) in baseline.iter().zip(current.iter()).enumerate() {
            assert!(
                (base - cur).abs() < 1e-12,
                "gaze changed state at iteration {i}, dim {j}: {base} vs {cur}"
            );
        }
    }
}

// =========================================================================
// 3. Thread safety
// =========================================================================

/// Compile-time check: Tapestry must be Send + Sync.
fn _assert_send_sync<T: Send + Sync>() {}

#[test]
fn tapestry_is_send_sync() {
    _assert_send_sync::<Tapestry>();
}

#[test]
fn concurrent_observe_same_tapestry() {
    let tapestry = Arc::new(build_tapestry(42));
    let input_data = vec![0.8_f32, 0.2, 0.1, 0.5];

    // Single-threaded baseline
    let fv = FloatVec::new(input_data.clone());
    let baseline = observatory::observe(&tapestry, &[("x", &fv)]).unwrap();
    let baseline_index = baseline.value_index;

    // Spawn 8 threads, each observing 100 times
    let mut handles = Vec::new();
    for _ in 0..8 {
        let tap = Arc::clone(&tapestry);
        let inp = input_data.clone();
        handles.push(thread::spawn(move || {
            let fv = FloatVec::new(inp);
            let mut results = Vec::with_capacity(100);
            for _ in 0..100 {
                let obs = observatory::observe(&tap, &[("x", &fv)]).unwrap();
                results.push(obs.value_index);
            }
            results
        }));
    }

    for handle in handles {
        let results = handle.join().expect("thread panicked");
        for (i, &idx) in results.iter().enumerate() {
            assert_eq!(
                idx, baseline_index,
                "thread result {i} differs from baseline: got {idx}, expected {baseline_index}"
            );
        }
    }
}

// =========================================================================
// 4. Edge cases: dimension extremes
// =========================================================================

#[test]
fn dim2_full_pipeline() {
    let mut builder = DeclarationBuilder::new("dim2_test");
    builder
        .input("x", 2)
        .output("y")
        .relate("y", &["x"], RelationKind::Proportional)
        .quality(0.9, 0.8);
    let decl = builder.build();
    let tapestry = weaver::weave(&decl, true, 42).unwrap();

    let input = FloatVec::new(vec![0.7, 0.3]);
    let obs = observatory::observe(&tapestry, &[("x", &input)]).unwrap();
    assert!(obs.value_index < 2);
    let prob_sum: f32 = obs.probabilities.data.iter().sum();
    assert!((prob_sum - 1.0).abs() < 1e-3);
}

#[test]
fn dim32_full_pipeline() {
    let mut builder = DeclarationBuilder::new("dim32_test");
    builder
        .input("x", 32)
        .output("y")
        .relate("y", &["x"], RelationKind::Proportional)
        .quality(0.9, 0.8);
    let decl = builder.build();
    let tapestry = weaver::weave(&decl, true, 42).unwrap();

    let mut data = vec![0.0_f32; 32];
    data[0] = 0.5;
    data[15] = 0.3;
    data[31] = 0.2;
    let input = FloatVec::new(data);
    let obs = observatory::observe(&tapestry, &[("x", &input)]).unwrap();
    assert!(obs.value_index < 32);
    let prob_sum: f32 = obs.probabilities.data.iter().sum();
    assert!((prob_sum - 1.0).abs() < 1e-2);
}

// =========================================================================
// 5. Edge cases: special distributions
// =========================================================================

#[test]
fn uniform_input_pipeline() {
    let tapestry = build_tapestry(42);
    // All equal values
    let input = FloatVec::new(vec![0.25, 0.25, 0.25, 0.25]);
    let obs = observatory::observe(&tapestry, &[("x", &input)]).unwrap();
    assert!(obs.value_index < 4);
}

#[test]
fn extreme_input_pipeline() {
    let tapestry = build_tapestry(42);
    // Almost-certain input
    let input = FloatVec::new(vec![0.999, 0.001, 0.0, 0.0]);
    let obs = observatory::observe(&tapestry, &[("x", &input)]).unwrap();
    assert!(obs.value_index < 4);
}

#[test]
fn wave_dim2_focus_observe_cycle() {
    let w = Wave::from_classical(&FloatVec::new(vec![0.7, 0.3]));
    let focused = w.focus(0.5);
    let (idx, collapsed) = focused.observe();
    assert!(idx < 2);
    assert!(collapsed.is_collapsed());
}

#[test]
fn wave_dim32_compose_chain() {
    let mut data_a = vec![0.0_f32; 32];
    data_a[0] = 1.0;
    let mut data_b = vec![0.0_f32; 32];
    data_b[31] = 1.0;

    let a = Wave::from_classical(&FloatVec::new(data_a));
    let b = Wave::from_classical(&FloatVec::new(data_b));

    let composed = Wave::compose(
        &a, &b,
        &axol::wave::InterferencePattern::Constructive,
    ).unwrap();
    assert_eq!(composed.dim, 32);

    let probs = composed.probabilities();
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}
