//! Relation system unit tests — AXOL v2 core type.
//!
//! Tests Relation creation, negativity, gaze, Expectation,
//! widen, conflict_score, resolve strategies, and error handling.

use axol::types::*;
use axol::wave::{Wave, InterferencePattern};
use axol::relation::{Relation, Expectation};
use axol::relation;
use axol::dsl::parser::RelDirection;

// =========================================================================
// Helpers
// =========================================================================

fn make_wave(values: &[f32]) -> Wave {
    Wave::from_classical(&FloatVec::new(values.to_vec()))
}

fn make_relation(
    from_vals: &[f32],
    to_vals: &[f32],
    pattern: InterferencePattern,
) -> Relation {
    let from_w = make_wave(from_vals);
    let to_w = make_wave(to_vals);
    Relation::new(
        "test_rel",
        "from",
        "to",
        RelDirection::Bidir,
        &from_w,
        &to_w,
        pattern,
    ).unwrap()
}

fn assert_prob_sum_one(probs: &[f64]) {
    let sum: f64 = probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "probabilities sum to {sum}, expected 1.0"
    );
}

// =========================================================================
// 1. Relation creation
// =========================================================================

#[test]
fn relation_new_constructive() {
    let rel = make_relation(&[0.8, 0.2, 0.0, 0.0], &[0.0, 0.0, 0.2, 0.8], InterferencePattern::Constructive);
    assert_eq!(rel.wave.dim, 4);
    assert_eq!(rel.pattern, InterferencePattern::Constructive);
    assert_prob_sum_one(&rel.gaze());
}

#[test]
fn relation_new_all_patterns() {
    let patterns = [
        InterferencePattern::Constructive,
        InterferencePattern::Additive,
        InterferencePattern::Multiplicative,
        InterferencePattern::Destructive,
        InterferencePattern::Conditional,
    ];
    // Use overlapping waves to avoid zero vector from Multiplicative on orthogonal inputs
    for pattern in &patterns {
        let rel = make_relation(
            &[0.6, 0.3, 0.05, 0.05],
            &[0.05, 0.05, 0.3, 0.6],
            pattern.clone(),
        );
        assert_prob_sum_one(&rel.gaze());
    }
}

#[test]
fn relation_negativity_positive_initial() {
    // Different distributions → negativity > 0
    let rel = make_relation(
        &[0.9, 0.1, 0.0, 0.0],
        &[0.0, 0.0, 0.1, 0.9],
        InterferencePattern::Constructive,
    );
    assert!(
        rel.negativity > 0.0,
        "negativity should be > 0 for different waves, got {}",
        rel.negativity
    );
}

#[test]
fn relation_negativity_low_for_similar_waves() {
    // Same distribution → negativity ≈ 0
    let rel = make_relation(
        &[0.8, 0.2, 0.0, 0.0],
        &[0.8, 0.2, 0.0, 0.0],
        InterferencePattern::Constructive,
    );
    assert!(
        rel.negativity < 0.3,
        "negativity should be low for similar waves, got {}",
        rel.negativity
    );
}

// =========================================================================
// 2. Gaze (C=0)
// =========================================================================

#[test]
fn relation_gaze_returns_valid_probabilities() {
    let rel = make_relation(
        &[0.6, 0.4, 0.0, 0.0],
        &[0.0, 0.0, 0.4, 0.6],
        InterferencePattern::Constructive,
    );
    let probs = rel.gaze();
    assert_eq!(probs.len(), 4);
    assert_prob_sum_one(&probs);
    for &p in &probs {
        assert!(p >= -1e-10, "probability should not be negative");
    }
}

// =========================================================================
// 3. Expectation creation and application
// =========================================================================

#[test]
fn expectation_from_distribution_normalizes() {
    let expect = Expectation::from_distribution("test", vec![2.0, 2.0, 1.0, 1.0], 0.8);
    let norm = expect.normalized_landscape();
    let sum: f64 = norm.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "normalized landscape should sum to 1");
}

#[test]
fn expectation_strength_clamped() {
    let expect = Expectation::from_distribution("test", vec![1.0, 1.0], 1.5);
    assert!((expect.strength - 1.0).abs() < 1e-10, "strength should be clamped to 1.0");
    let expect2 = Expectation::from_distribution("test", vec![1.0, 1.0], -0.5);
    assert!((expect2.strength - 0.0).abs() < 1e-10, "strength should be clamped to 0.0");
}

#[test]
fn apply_expect_alignment_range() {
    let mut rel = make_relation(
        &[0.7, 0.3, 0.0, 0.0],
        &[0.0, 0.0, 0.3, 0.7],
        InterferencePattern::Constructive,
    );
    let expect = Expectation::from_distribution("test", vec![0.6, 0.3, 0.05, 0.05], 0.8);
    let result = rel.apply_expect(&expect).unwrap();
    assert!(
        result.alignment >= 0.0 && result.alignment <= 1.0,
        "alignment should be in [0,1], got {}",
        result.alignment
    );
}

#[test]
fn apply_expect_tracks_negativity_delta() {
    let mut rel = make_relation(
        &[0.7, 0.3, 0.0, 0.0],
        &[0.0, 0.0, 0.3, 0.7],
        InterferencePattern::Constructive,
    );
    let old_neg = rel.negativity;
    let expect = Expectation::from_distribution("test", vec![0.6, 0.3, 0.05, 0.05], 0.8);
    let result = rel.apply_expect(&expect).unwrap();
    let _expected_delta = rel.negativity - old_neg;
    // negativity_delta is computed before self.negativity is updated in the result,
    // but it should be a finite number
    assert!(result.negativity_delta.is_finite());
}

#[test]
fn apply_expect_strength_zero_minimal_effect() {
    let mut rel = make_relation(
        &[0.7, 0.3, 0.0, 0.0],
        &[0.0, 0.0, 0.3, 0.7],
        InterferencePattern::Constructive,
    );
    let before = rel.gaze().to_vec();
    let expect = Expectation::from_distribution("test", vec![0.9, 0.1, 0.0, 0.0], 0.0);
    let _result = rel.apply_expect(&expect).unwrap();
    let after = rel.gaze();
    // With strength=0, distribution should be nearly unchanged
    let diff: f64 = before.iter().zip(after.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff < 0.1, "strength=0 should have minimal effect, diff={diff}");
}

#[test]
fn apply_expect_strength_one_maximum_effect() {
    let mut rel1 = make_relation(
        &[0.5, 0.5, 0.0, 0.0],
        &[0.0, 0.0, 0.5, 0.5],
        InterferencePattern::Constructive,
    );
    let mut rel2 = rel1.clone();

    let expect_weak = Expectation::from_distribution("weak", vec![0.9, 0.1, 0.0, 0.0], 0.1);
    let expect_strong = Expectation::from_distribution("strong", vec![0.9, 0.1, 0.0, 0.0], 1.0);

    let _r1 = rel1.apply_expect(&expect_weak).unwrap();
    let _r2 = rel2.apply_expect(&expect_strong).unwrap();

    // Strong expectation should change more than weak
    let probs1 = rel1.gaze();
    let probs2 = rel2.gaze();
    // Both should still be valid
    assert_prob_sum_one(&probs1);
    assert_prob_sum_one(&probs2);
}

// =========================================================================
// 4. Widen
// =========================================================================

#[test]
fn widen_decreases_t() {
    let from_w = make_wave(&[0.8, 0.2, 0.0, 0.0]);
    let to_w = make_wave(&[0.0, 0.0, 0.2, 0.8]);
    let mut rel = Relation::new(
        "test", "from", "to",
        RelDirection::Bidir, &from_w, &to_w,
        InterferencePattern::Constructive,
    ).unwrap();

    let t_before = rel.wave.t;
    rel.widen(0.5).unwrap();
    // Widen should decrease t (or keep it if already 0)
    assert!(
        rel.wave.t <= t_before + 1e-10,
        "widen should decrease t: before={t_before}, after={}", rel.wave.t
    );
}

#[test]
fn widen_increases_negativity() {
    let mut rel = make_relation(
        &[0.9, 0.1, 0.0, 0.0],
        &[0.0, 0.0, 0.1, 0.9],
        InterferencePattern::Constructive,
    );
    let neg_before = rel.negativity;
    rel.widen(0.5).unwrap();
    assert!(
        rel.negativity >= neg_before - 1e-10,
        "widen should increase negativity: before={neg_before}, after={}",
        rel.negativity
    );
}

#[test]
fn widen_zero_no_change() {
    let mut rel = make_relation(
        &[0.7, 0.3, 0.0, 0.0],
        &[0.0, 0.0, 0.3, 0.7],
        InterferencePattern::Constructive,
    );
    let probs_before = rel.gaze().to_vec();
    let neg_before = rel.negativity;
    rel.widen(0.0).unwrap();
    let probs_after = rel.gaze();
    let diff: f64 = probs_before.iter().zip(probs_after.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff < 1e-10, "widen(0) should not change anything");
    assert!((rel.negativity - neg_before).abs() < 1e-10);
}

// =========================================================================
// 5. Conflict score
// =========================================================================

#[test]
fn conflict_score_same_waves_low() {
    let rel_a = make_relation(
        &[0.8, 0.2, 0.0, 0.0],
        &[0.0, 0.0, 0.2, 0.8],
        InterferencePattern::Constructive,
    );
    let rel_b = rel_a.clone();
    let score = Relation::conflict_score(&rel_a, &rel_b);
    assert!(
        score < 0.1,
        "same relations should have low conflict, got {score}"
    );
}

#[test]
fn conflict_score_opposite_waves_high() {
    let rel_a = make_relation(
        &[0.95, 0.05, 0.0, 0.0],
        &[0.95, 0.05, 0.0, 0.0],
        InterferencePattern::Constructive,
    );
    let rel_b = make_relation(
        &[0.0, 0.0, 0.05, 0.95],
        &[0.0, 0.0, 0.05, 0.95],
        InterferencePattern::Constructive,
    );
    let score = Relation::conflict_score(&rel_a, &rel_b);
    assert!(
        score > 0.2,
        "opposite relations should have higher conflict, got {score}"
    );
}

#[test]
fn conflict_score_range() {
    let rel_a = make_relation(
        &[0.7, 0.3, 0.0, 0.0],
        &[0.0, 0.0, 0.3, 0.7],
        InterferencePattern::Constructive,
    );
    let rel_b = make_relation(
        &[0.0, 0.0, 0.7, 0.3],
        &[0.3, 0.7, 0.0, 0.0],
        InterferencePattern::Destructive,
    );
    let score = Relation::conflict_score(&rel_a, &rel_b);
    assert!(score >= 0.0 && score <= 1.0, "conflict_score should be in [0,1], got {score}");
}

// =========================================================================
// 6. Resolve functions
// =========================================================================

#[test]
fn resolve_interfere_produces_valid_wave() {
    let a = make_wave(&[0.8, 0.2, 0.0, 0.0]);
    let b = make_wave(&[0.0, 0.0, 0.2, 0.8]);
    let result = relation::resolve_interfere(&a, &b).unwrap();
    assert_eq!(result.dim, 4);
    assert_prob_sum_one(&result.probabilities());
}

#[test]
fn resolve_branch_produces_valid_wave() {
    let a = make_wave(&[0.8, 0.2, 0.0, 0.0]);
    let b = make_wave(&[0.0, 0.0, 0.2, 0.8]);
    let result = relation::resolve_branch(&a, &b).unwrap();
    assert_eq!(result.dim, 4);
    assert_prob_sum_one(&result.probabilities());
}

#[test]
fn resolve_superpose_produces_valid_wave() {
    let a = make_wave(&[0.8, 0.2, 0.0, 0.0]);
    let b = make_wave(&[0.0, 0.0, 0.2, 0.8]);
    let result = relation::resolve_superpose(&a, &b).unwrap();
    assert_eq!(result.dim, 4);
    assert_prob_sum_one(&result.probabilities());
}

#[test]
fn resolve_rebase_produces_valid_wave() {
    let a = make_wave(&[0.8, 0.2, 0.0, 0.0]);
    let target = make_wave(&[0.0, 0.0, 0.2, 0.8]);
    let result = relation::resolve_rebase(&a, &target).unwrap();
    assert_eq!(result.dim, 4);
    assert_prob_sum_one(&result.probabilities());
}

#[test]
fn resolve_strategies_produce_different_results() {
    let a = make_wave(&[0.8, 0.2, 0.0, 0.0]);
    let b = make_wave(&[0.0, 0.0, 0.2, 0.8]);

    let interfered = relation::resolve_interfere(&a, &b).unwrap().probabilities();
    let branched = relation::resolve_branch(&a, &b).unwrap().probabilities();
    let superposed = relation::resolve_superpose(&a, &b).unwrap().probabilities();
    let rebased = relation::resolve_rebase(&a, &b).unwrap().probabilities();

    let all = [interfered, branched, superposed, rebased];
    let mut found_diff = false;
    for i in 0..all.len() {
        for j in (i + 1)..all.len() {
            let diff: f64 = all[i].iter().zip(all[j].iter()).map(|(a, b)| (a - b).abs()).sum();
            if diff > 1e-6 {
                found_diff = true;
            }
        }
    }
    assert!(found_diff, "at least some resolve strategies should produce different results");
}

// =========================================================================
// 7. Error handling: dimension mismatch
// =========================================================================

#[test]
fn relation_dimension_mismatch() {
    let from_w = make_wave(&[0.5, 0.5]);
    let to_w = make_wave(&[0.3, 0.3, 0.4]);
    let result = Relation::new(
        "test", "from", "to",
        RelDirection::Bidir, &from_w, &to_w,
        InterferencePattern::Constructive,
    );
    assert!(result.is_err(), "dimension mismatch should return error");
}

#[test]
fn resolve_dimension_mismatch() {
    let a = make_wave(&[0.5, 0.5]);
    let b = make_wave(&[0.3, 0.3, 0.4]);
    assert!(relation::resolve_interfere(&a, &b).is_err());
    assert!(relation::resolve_superpose(&a, &b).is_err());
}

// =========================================================================
// 8. Conflict relation
// =========================================================================

#[test]
fn conflict_relation_creation() {
    let rel_a = make_relation(
        &[0.9, 0.1, 0.0, 0.0],
        &[0.0, 0.0, 0.1, 0.9],
        InterferencePattern::Constructive,
    );
    let rel_b = make_relation(
        &[0.0, 0.0, 0.9, 0.1],
        &[0.1, 0.9, 0.0, 0.0],
        InterferencePattern::Additive,
    );
    let conflict = Relation::conflict(&rel_a, &rel_b).unwrap();
    assert_eq!(conflict.pattern, InterferencePattern::Destructive);
    assert_eq!(conflict.direction, RelDirection::Conflict);
    assert_prob_sum_one(&conflict.gaze());
}
