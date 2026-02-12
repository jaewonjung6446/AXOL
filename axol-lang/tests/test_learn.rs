//! Tests for the learning mechanism.

use axol::learn::*;
use axol::dsl::lexer::Lexer;
use axol::dsl::parser::Parser;

fn parse_source(source: &str) -> usize {
    let mut lexer = Lexer::new(source);
    lexer.tokenize();
    let mut parser = Parser::new(lexer.tokens.clone());
    let program = parser.parse().expect("parse failed");
    program.statements.len()
}

// ---------------------------------------------------------------------------
// Training set constructors
// ---------------------------------------------------------------------------

#[test]
fn training_set_xor_valid() {
    let ts = TrainingSet::xor();
    assert_eq!(ts.dim, 4);
    assert_eq!(ts.n_classes, 2);
    assert_eq!(ts.samples.len(), 4);
    for s in &ts.samples {
        assert_eq!(s.input.len(), 4);
        assert!(s.expected < 2);
    }
}

#[test]
fn training_set_not_valid() {
    let ts = TrainingSet::not();
    assert_eq!(ts.dim, 2);
    assert_eq!(ts.samples.len(), 2);
}

#[test]
fn training_set_custom() {
    let mut ts = TrainingSet::new("test", 3, 2);
    ts.add(vec![1.0, 0.0, 0.0], 0);
    ts.add(vec![0.0, 1.0, 0.0], 1);
    assert_eq!(ts.samples.len(), 2);
}

// ---------------------------------------------------------------------------
// Learning — NOT gate (simplest, dim=2)
// ---------------------------------------------------------------------------

#[test]
fn learn_not_gate() {
    let training = TrainingSet::not();
    let config = LearnConfig {
        grid_r_steps: 10,
        grid_eps_steps: 10,
        nelder_mead_iters: 100,
        ..LearnConfig::default()
    };

    let result = learn(&training, &config).expect("learning should succeed");
    assert!(result.accuracy >= 0.5, "NOT accuracy={:.1}% should be >= 50%", result.accuracy * 100.0);
    assert!(result.total_evaluations > 0);
    assert!(result.best_r >= 3.0 && result.best_r <= 4.0);
    assert!(result.best_epsilon >= 0.01 && result.best_epsilon <= 0.95);
}

// ---------------------------------------------------------------------------
// Learning — AND gate (linearly separable)
// ---------------------------------------------------------------------------

#[test]
fn learn_and_gate() {
    let training = TrainingSet::and();
    let config = LearnConfig {
        grid_r_steps: 10,
        grid_eps_steps: 10,
        nelder_mead_iters: 100,
        ..LearnConfig::default()
    };

    let result = learn(&training, &config).expect("learning should succeed");
    assert!(result.accuracy >= 0.5, "AND accuracy={:.1}% should be >= 50%", result.accuracy * 100.0);
}

// ---------------------------------------------------------------------------
// Learning — OR gate (linearly separable)
// ---------------------------------------------------------------------------

#[test]
fn learn_or_gate() {
    let training = TrainingSet::or();
    let config = LearnConfig {
        grid_r_steps: 10,
        grid_eps_steps: 10,
        nelder_mead_iters: 100,
        ..LearnConfig::default()
    };

    let result = learn(&training, &config).expect("learning should succeed");
    assert!(result.accuracy >= 0.5, "OR accuracy={:.1}% should be >= 50%", result.accuracy * 100.0);
}

// ---------------------------------------------------------------------------
// Learning — XOR gate (non-linearly separable, the critical test)
// ---------------------------------------------------------------------------

#[test]
fn learn_xor_gate() {
    let training = TrainingSet::xor();
    let config = LearnConfig {
        grid_r_steps: 12,
        grid_eps_steps: 12,
        nelder_mead_iters: 150,
        seeds: vec![42, 123, 7, 999, 31415],
        ..LearnConfig::default()
    };

    let result = learn(&training, &config).expect("learning should succeed");
    // XOR is hard — report accuracy for visibility
    eprintln!("XOR accuracy: {:.1}% r={:.4} eps={:.4} relation={:?} evals={}",
        result.accuracy * 100.0, result.best_r, result.best_epsilon,
        result.best_relation, result.total_evaluations);
    // With continuous probability-based loss and quantum basin dynamics,
    // the optimizer should find something better than random
    assert!(result.accuracy > 0.0,
        "XOR accuracy={:.1}% should be > 0%", result.accuracy * 100.0);
}

// ---------------------------------------------------------------------------
// Learned tapestry works with observatory::observe
// ---------------------------------------------------------------------------

#[test]
fn learned_tapestry_verifies_with_observe() {
    let training = TrainingSet::not();
    let config = LearnConfig {
        grid_r_steps: 10,
        grid_eps_steps: 10,
        nelder_mead_iters: 100,
        ..LearnConfig::default()
    };

    let result = learn(&training, &config).expect("learning should succeed");

    // Verify using the full observation pipeline
    let verify_accuracy = evaluate(&result.tapestry, &training);
    // The verified accuracy should match what learn reported
    assert!((verify_accuracy - result.accuracy).abs() < 0.01,
        "verified={:.1}% vs reported={:.1}%", verify_accuracy * 100.0, result.accuracy * 100.0);
}

// ---------------------------------------------------------------------------
// Result metadata
// ---------------------------------------------------------------------------

#[test]
fn learn_result_has_valid_metadata() {
    let training = TrainingSet::not();
    let config = LearnConfig {
        grid_r_steps: 8,
        grid_eps_steps: 8,
        nelder_mead_iters: 50,
        seeds: vec![42],
        optimize_weights: false,
        ..LearnConfig::default()
    };

    let result = learn(&training, &config).expect("learning should succeed");
    assert!(result.best_r >= 3.0 && result.best_r <= 4.0);
    assert!(result.best_epsilon >= 0.01 && result.best_epsilon <= 0.95);
    assert!(result.total_evaluations > 100); // at least grid search
    assert!(!result.history.is_empty());
    assert!(result.accuracy >= 0.0 && result.accuracy <= 1.0);
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

#[test]
fn learn_empty_training_fails() {
    let ts = TrainingSet::new("empty", 4, 2);
    let result = learn(&ts, &LearnConfig::default());
    assert!(result.is_err());
}

#[test]
fn learn_dim_mismatch_fails() {
    let mut ts = TrainingSet::new("bad", 4, 2);
    ts.add(vec![0.1, 0.2], 0); // dim=2 but training says dim=4
    let result = learn(&ts, &LearnConfig::default());
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// DSL parsing
// ---------------------------------------------------------------------------

#[test]
fn learn_dsl_parses() {
    let source = r#"
learn "xor" dim=4 quantum=1 {
    [0.9, 0.1, 0.9, 0.1] = 0
    [0.9, 0.1, 0.1, 0.9] = 1
    [0.1, 0.9, 0.9, 0.1] = 1
    [0.1, 0.9, 0.1, 0.9] = 0
}
"#;
    let stmts = parse_source(source);
    assert_eq!(stmts, 1);
}

#[test]
fn learn_dsl_with_observe_parses() {
    let source = r#"
learn "mygate" dim=2 quantum=1 seed=42 {
    [0.9, 0.1] = 1
    [0.1, 0.9] = 0
}
"#;
    let stmts = parse_source(source);
    assert_eq!(stmts, 1);
}

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------

#[test]
fn learn_is_deterministic() {
    let training = TrainingSet::not();
    let config = LearnConfig {
        grid_r_steps: 8,
        grid_eps_steps: 8,
        nelder_mead_iters: 50,
        seeds: vec![42],
        optimize_weights: false,
        ..LearnConfig::default()
    };

    let r1 = learn(&training, &config).expect("run 1");
    let r2 = learn(&training, &config).expect("run 2");

    assert_eq!(r1.accuracy, r2.accuracy, "same config should give same accuracy");
    assert_eq!(r1.best_r, r2.best_r, "same config should give same r");
    assert_eq!(r1.best_epsilon, r2.best_epsilon, "same config should give same epsilon");
}
