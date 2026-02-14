//! Wave system unit tests — AXOL v2 core type.
//!
//! Tests construction, probability consistency, gaze (C=0), compose patterns,
//! focus/observe collapse, irreversibility, and edge cases.

use num_complex::Complex64;
use axol::types::*;
use axol::wave::{Wave, InterferencePattern};

// =========================================================================
// Helpers
// =========================================================================

fn assert_prob_sum_one(probs: &[f64]) {
    let sum: f64 = probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "probabilities sum to {sum}, expected 1.0"
    );
}

fn make_wave(values: &[f32]) -> Wave {
    Wave::from_classical(&FloatVec::new(values.to_vec()))
}

// =========================================================================
// 1. Construction
// =========================================================================

#[test]
fn wave_from_classical() {
    let w = make_wave(&[0.8, 0.2, 0.0, 0.0]);
    assert_eq!(w.dim, 4);
    assert!((w.t - 0.0).abs() < 1e-10, "t should be 0.0 for new wave");
    assert!(w.is_pure());
    assert!(!w.is_collapsed());
    assert_prob_sum_one(&w.probabilities());
}

#[test]
fn wave_from_complex() {
    let amps = ComplexVec::new(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 1.0),
    ]);
    let w = Wave::from_complex(amps);
    assert_eq!(w.dim, 2);
    assert!(w.is_pure());
    let probs = w.probabilities();
    assert_prob_sum_one(&probs);
    // Equal magnitudes → equal probabilities
    assert!((probs[0] - probs[1]).abs() < 1e-6);
}

#[test]
fn wave_from_density() {
    // Maximally mixed 2D state
    let data = vec![
        Complex64::new(0.5, 0.0), Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0), Complex64::new(0.5, 0.0),
    ];
    let rho = DensityMatrix::new(data, 2);
    let w = Wave::from_density(rho);
    assert_eq!(w.dim, 2);
    let probs = w.probabilities();
    assert_prob_sum_one(&probs);
    // Should be ~uniform
    assert!((probs[0] - 0.5).abs() < 1e-6);
}

#[test]
fn wave_collapsed() {
    let w = Wave::collapsed(4, 2);
    assert_eq!(w.dim, 4);
    assert!(w.is_collapsed());
    assert!((w.t - 1.0).abs() < 1e-10);
    let probs = w.probabilities();
    assert_prob_sum_one(&probs);
    assert!((probs[2] - 1.0).abs() < 1e-6, "collapsed index should have prob 1.0");
    assert!(probs[0].abs() < 1e-6);
}

#[test]
fn wave_from_basins() {
    let bs = BasinStructure {
        dim: 4,
        n_basins: 2,
        centroids: vec![vec![0.9, 0.1, 0.0, 0.0], vec![0.0, 0.0, 0.1, 0.9]],
        volumes: vec![0.5, 0.5],
        fractal_dim: 1.5,
        transform: None,
        phases: vec![0.0, std::f64::consts::PI / 4.0],
        radii: vec![0.5, 0.5],
    };
    let input = FloatVec::new(vec![0.8, 0.2, 0.1, 0.1]);
    let w = Wave::from_basins(&bs, &input);
    assert_eq!(w.dim, 4);
    assert!(w.is_pure());
    assert_prob_sum_one(&w.probabilities());
}

// =========================================================================
// 2. Probability consistency
// =========================================================================

#[test]
fn probabilities_always_sum_to_one() {
    let inputs: Vec<Vec<f32>> = vec![
        vec![0.8, 0.2],
        vec![0.0, 0.0, 0.0, 1.0],
        vec![0.25, 0.25, 0.25, 0.25],
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.999, 0.001],
    ];
    for vals in &inputs {
        let w = make_wave(vals);
        assert_prob_sum_one(&w.probabilities());
    }
}

// =========================================================================
// 3. Gaze (C=0): read without collapse
// =========================================================================

#[test]
fn gaze_returns_probabilities() {
    let w = make_wave(&[0.8, 0.2, 0.0, 0.0]);
    let g = w.gaze();
    let p = w.probabilities();
    assert_eq!(g.len(), p.len());
    for (a, b) in g.iter().zip(p.iter()) {
        assert!((a - b).abs() < 1e-10);
    }
}

#[test]
fn gaze_does_not_change_state() {
    let w = make_wave(&[0.8, 0.2, 0.1, 0.1]);
    let before = w.probabilities();
    let _ = w.gaze();
    let _ = w.gaze();
    let _ = w.gaze();
    let after = w.probabilities();
    for (a, b) in before.iter().zip(after.iter()) {
        assert!((a - b).abs() < 1e-10, "gaze should not mutate state");
    }
    assert!((w.t - 0.0).abs() < 1e-10, "t should not change after gaze");
}

// =========================================================================
// 4. Compose — 5 interference patterns
// =========================================================================

fn make_ab() -> (Wave, Wave) {
    let a = make_wave(&[0.9, 0.1, 0.0, 0.0]);
    let b = make_wave(&[0.0, 0.0, 0.1, 0.9]);
    (a, b)
}

#[test]
fn compose_constructive() {
    let (a, b) = make_ab();
    let c = Wave::compose(&a, &b, &InterferencePattern::Constructive).unwrap();
    assert_eq!(c.dim, 4);
    assert_prob_sum_one(&c.probabilities());
}

#[test]
fn compose_additive() {
    let (a, b) = make_ab();
    let c = Wave::compose(&a, &b, &InterferencePattern::Additive).unwrap();
    assert_eq!(c.dim, 4);
    assert_prob_sum_one(&c.probabilities());
}

#[test]
fn compose_multiplicative() {
    let (a, b) = make_ab();
    let c = Wave::compose(&a, &b, &InterferencePattern::Multiplicative).unwrap();
    assert_eq!(c.dim, 4);
    assert_prob_sum_one(&c.probabilities());
}

#[test]
fn compose_destructive() {
    let (a, b) = make_ab();
    let c = Wave::compose(&a, &b, &InterferencePattern::Destructive).unwrap();
    assert_eq!(c.dim, 4);
    assert_prob_sum_one(&c.probabilities());
}

#[test]
fn compose_conditional() {
    let (a, b) = make_ab();
    let c = Wave::compose(&a, &b, &InterferencePattern::Conditional).unwrap();
    assert_eq!(c.dim, 4);
    assert_prob_sum_one(&c.probabilities());
}

#[test]
fn compose_patterns_produce_different_distributions() {
    let a = make_wave(&[0.7, 0.3, 0.0, 0.0]);
    let b = make_wave(&[0.0, 0.0, 0.3, 0.7]);

    let patterns = [
        InterferencePattern::Constructive,
        InterferencePattern::Additive,
        InterferencePattern::Multiplicative,
        InterferencePattern::Destructive,
        InterferencePattern::Conditional,
    ];

    let results: Vec<Vec<f64>> = patterns.iter()
        .map(|p| Wave::compose(&a, &b, p).unwrap().probabilities())
        .collect();

    // At least some patterns should produce different results
    let mut found_diff = false;
    for i in 0..results.len() {
        for j in (i + 1)..results.len() {
            let diff: f64 = results[i].iter()
                .zip(results[j].iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            if diff > 1e-6 {
                found_diff = true;
            }
        }
    }
    assert!(found_diff, "at least some patterns should yield different distributions");
}

#[test]
fn compose_dimension_mismatch_returns_error() {
    let a = make_wave(&[0.5, 0.5]);
    let b = make_wave(&[0.3, 0.3, 0.4]);
    let result = Wave::compose(&a, &b, &InterferencePattern::Constructive);
    assert!(result.is_err(), "dimension mismatch should return error");
}

// =========================================================================
// 5. compose_many: 3+ wave chain
// =========================================================================

#[test]
fn compose_many_three_waves() {
    let w1 = make_wave(&[0.8, 0.2, 0.0, 0.0]);
    let w2 = make_wave(&[0.0, 0.5, 0.5, 0.0]);
    let w3 = make_wave(&[0.0, 0.0, 0.2, 0.8]);

    let patterns = [InterferencePattern::Constructive, InterferencePattern::Additive];
    let result = Wave::compose_many(
        &[&w1, &w2, &w3],
        &[&patterns[0], &patterns[1]],
    ).unwrap();

    assert_eq!(result.dim, 4);
    assert_prob_sum_one(&result.probabilities());
}

#[test]
fn compose_many_single_wave() {
    let w = make_wave(&[0.5, 0.5]);
    let result = Wave::compose_many(&[&w], &[]).unwrap();
    let probs_orig = w.probabilities();
    let probs_result = result.probabilities();
    for (a, b) in probs_orig.iter().zip(probs_result.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn compose_many_empty_returns_error() {
    let result = Wave::compose_many(&[], &[]);
    assert!(result.is_err());
}

#[test]
fn compose_many_insufficient_patterns_returns_error() {
    let w1 = make_wave(&[0.5, 0.5]);
    let w2 = make_wave(&[0.3, 0.7]);
    let w3 = make_wave(&[0.6, 0.4]);
    // 3 waves need 2 patterns, provide only 1
    let p = InterferencePattern::Constructive;
    let result = Wave::compose_many(&[&w1, &w2, &w3], &[&p]);
    assert!(result.is_err());
}

// =========================================================================
// 6. Focus (partial collapse)
// =========================================================================

#[test]
fn focus_gamma_zero_no_change() {
    let w = make_wave(&[0.7, 0.3]);
    let focused = w.focus(0.0);
    let probs_before = w.probabilities();
    let probs_after = focused.probabilities();
    for (a, b) in probs_before.iter().zip(probs_after.iter()) {
        assert!((a - b).abs() < 1e-6, "gamma=0 should not change distribution");
    }
    assert!((focused.t - w.t).abs() < 1e-10);
}

#[test]
fn focus_gamma_half_sharpens() {
    let w = make_wave(&[0.6, 0.4]);
    let focused = w.focus(0.5);
    let max_before = w.probabilities().iter().cloned().fold(0.0_f64, f64::max);
    let max_after = focused.probabilities().iter().cloned().fold(0.0_f64, f64::max);
    assert!(
        max_after >= max_before - 1e-6,
        "focus should sharpen: max_before={max_before}, max_after={max_after}"
    );
    assert_prob_sum_one(&focused.probabilities());
}

#[test]
fn focus_gamma_one_collapses_to_one_hot() {
    let w = make_wave(&[0.6, 0.3, 0.1]);
    let focused = w.focus(1.0);
    let probs = focused.probabilities();
    assert_prob_sum_one(&probs);
    let max_p = probs.iter().cloned().fold(0.0_f64, f64::max);
    assert!(max_p > 0.95, "gamma=1.0 should produce near one-hot, got max_p={max_p}");
}

#[test]
fn focus_t_monotonically_increases() {
    let w = make_wave(&[0.6, 0.4]);
    let gammas = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0];
    let mut prev_t = w.t;
    for &g in &gammas {
        let focused = w.focus(g);
        assert!(
            focused.t >= prev_t - 1e-10,
            "t should monotonically increase: prev={prev_t}, curr={}", focused.t
        );
        prev_t = focused.t;
    }
}

// =========================================================================
// 7. Observe (full collapse)
// =========================================================================

#[test]
fn observe_returns_valid_index() {
    let w = make_wave(&[0.1, 0.2, 0.3, 0.4]);
    let (idx, collapsed) = w.observe();
    assert!(idx < w.dim, "index must be < dim");
    assert!(collapsed.is_collapsed());
    assert!((collapsed.t - 1.0).abs() < 1e-10);
    assert_prob_sum_one(&collapsed.probabilities());
}

#[test]
fn observe_selects_dominant() {
    // Deterministic: observe picks argmax
    let w = make_wave(&[0.1, 0.9]);
    let (idx, _) = w.observe();
    assert_eq!(idx, 1, "observe should pick highest probability");
}

#[test]
fn observe_collapsed_is_one_hot() {
    let w = make_wave(&[0.1, 0.2, 0.7]);
    let (idx, collapsed) = w.observe();
    let probs = collapsed.probabilities();
    assert!((probs[idx] - 1.0).abs() < 1e-6);
    for (i, &p) in probs.iter().enumerate() {
        if i != idx {
            assert!(p.abs() < 1e-6);
        }
    }
}

// =========================================================================
// 8. Irreversibility: focus → widen → gaze ≠ original gaze
// =========================================================================

#[test]
fn focus_is_irreversible() {
    let w = make_wave(&[0.6, 0.2, 0.1, 0.1]);
    let original_probs = w.gaze();

    // Focus then try to "undo" by creating a new wave from the focused state
    let focused = w.focus(0.5);
    let focused_probs = focused.gaze();

    // The focused state should differ from original
    let diff: f64 = original_probs.iter()
        .zip(focused_probs.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    // After focus(0.5), distribution should have changed
    assert!(diff > 1e-6, "focus should change the distribution");
}

// =========================================================================
// 9. Edge cases
// =========================================================================

#[test]
fn uniform_distribution_wave() {
    let w = make_wave(&[0.25, 0.25, 0.25, 0.25]);
    let probs = w.probabilities();
    assert_prob_sum_one(&probs);
    for &p in &probs {
        assert!((p - 0.25).abs() < 1e-4, "should be near uniform");
    }
}

#[test]
fn single_dimension_wave() {
    let w = make_wave(&[1.0]);
    assert_eq!(w.dim, 1);
    let probs = w.probabilities();
    assert_eq!(probs.len(), 1);
    assert!((probs[0] - 1.0).abs() < 1e-6);
}

#[test]
fn zero_vector_input() {
    // All zeros → should produce uniform distribution
    let w = make_wave(&[0.0, 0.0, 0.0, 0.0]);
    let probs = w.probabilities();
    assert_prob_sum_one(&probs);
    // Each should be ~0.25
    for &p in &probs {
        assert!((p - 0.25).abs() < 1e-4);
    }
}

#[test]
fn extreme_input_values() {
    let w = make_wave(&[0.999, 0.001]);
    let probs = w.probabilities();
    assert_prob_sum_one(&probs);
    assert!(probs[0] > probs[1], "dominant dimension should have higher probability");
}

#[test]
fn two_dimension_full_pipeline() {
    // dim=2: construct → gaze → focus → observe
    let w = make_wave(&[0.7, 0.3]);
    assert_eq!(w.dim, 2);
    let _ = w.gaze();
    let focused = w.focus(0.5);
    assert_prob_sum_one(&focused.probabilities());
    let (idx, collapsed) = focused.observe();
    assert!(idx < 2);
    assert!(collapsed.is_collapsed());
}
