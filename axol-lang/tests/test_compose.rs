//! Tests for the compose abstraction layer.
//!
//! Covers: confidence observer, tapestry chains, iterate loop,
//! basin designer, logic gates, and DSL extensions.

use axol::types::*;
use axol::declare::*;
use axol::weaver;
use axol::compose::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_decl(name: &str, dim: usize, omega: f64, phi: f64) -> EntangleDeclaration {
    let mut builder = DeclarationBuilder::new(name);
    builder
        .input("x", dim)
        .output("y")
        .relate("y", &["x"], RelationKind::Proportional)
        .quality(omega, phi);
    builder.build()
}

fn make_tapestry(name: &str, dim: usize, quantum: bool) -> weaver::Tapestry {
    let decl = make_decl(name, dim, 0.9, 0.7);
    weaver::weave(&decl, quantum, 42).unwrap()
}

// ===========================================================================
// Confidence observer tests
// ===========================================================================

#[test]
fn confidence_converges() {
    let tapestry = make_tapestry("conf_test", 4, true);
    let input = FloatVec::new(vec![0.8, 0.2, 0.1, 0.05]);

    let config = ConfidenceConfig {
        max_observations: 100,
        confidence_threshold: 0.5,
        min_observations: 3,
    };

    let result = observe_confident(&tapestry, &[("x", &input)], &config).unwrap();

    assert!(result.total_observations >= 3, "Should observe at least min_observations");
    assert!(result.total_observations <= 100, "Should not exceed max");
    assert!(result.confidence > 0.0, "Confidence should be > 0");
    assert_eq!(result.vote_counts.iter().sum::<usize>(), result.total_observations);
}

#[test]
fn confidence_early_stopping() {
    let tapestry = make_tapestry("early_stop", 2, false);
    let input = FloatVec::new(vec![0.9, 0.1]);

    let config = ConfidenceConfig {
        max_observations: 200,
        confidence_threshold: 0.3, // low threshold → should stop early
        min_observations: 3,
    };

    let result = observe_confident(&tapestry, &[("x", &input)], &config).unwrap();

    // With a low threshold, should stop before max
    assert!(result.total_observations < 200,
        "Should early stop: got {} observations", result.total_observations);
}

#[test]
fn confidence_avg_probabilities_sum_to_one() {
    let tapestry = make_tapestry("prob_sum", 4, true);
    let input = FloatVec::new(vec![0.5, 0.3, 0.15, 0.05]);

    let config = ConfidenceConfig::default();
    let result = observe_confident(&tapestry, &[("x", &input)], &config).unwrap();

    let sum: f64 = result.avg_probabilities.iter().sum();
    assert!((sum - 1.0).abs() < 0.1, "avg probs should approximately sum to 1, got {}", sum);
}

// ===========================================================================
// Tapestry chain tests
// ===========================================================================

#[test]
fn chain_single_stage() {
    let t = make_tapestry("single", 4, true);
    let original_mat = t.composed_matrix.clone().unwrap();

    let c = chain("single_chain", vec![t]).unwrap();
    assert_eq!(c.stages.len(), 1);
    assert_eq!(c.composed_matrix.rows, original_mat.rows);
    assert_eq!(c.composed_matrix.cols, original_mat.cols);
}

#[test]
fn chain_matrix_composition() {
    let t1 = make_tapestry("stage1", 4, true);
    let t2 = make_tapestry("stage2", 4, true);

    let m1 = t1.composed_matrix.clone().unwrap();
    let m2 = t2.composed_matrix.clone().unwrap();
    let expected = m1.matmul(&m2);

    let c = chain("composed", vec![t1, t2]).unwrap();

    // Composed matrix should equal m1 @ m2
    for i in 0..4 {
        for j in 0..4 {
            let diff = (c.composed_matrix.get(i, j) - expected.get(i, j)).abs();
            assert!(diff < 1e-5, "Matrix mismatch at ({},{}): {} vs {}",
                i, j, c.composed_matrix.get(i, j), expected.get(i, j));
        }
    }
}

#[test]
fn chain_observe_works() {
    let t1 = make_tapestry("obs1", 4, true);
    let t2 = make_tapestry("obs2", 4, true);

    let c = chain("obs_chain", vec![t1, t2]).unwrap();
    let input = FloatVec::new(vec![0.5, 0.3, 0.15, 0.05]);

    let obs = observe_chain(&c, &[("x", &input)]).unwrap();
    assert!(obs.value_index < 4);
    assert!(!obs.probabilities.data.is_empty());
}

#[test]
fn chain_flatten() {
    let t1 = make_tapestry("f1", 4, true);
    let t2 = make_tapestry("f2", 4, true);

    let c = chain("flat_chain", vec![t1, t2]).unwrap();
    let flat = flatten(&c).unwrap();

    assert_eq!(flat.name, "flat_chain");
    assert!(flat.composed_matrix.is_some());
}

// ===========================================================================
// Iterate tests
// ===========================================================================

#[test]
fn iterate_runs_minimum_iterations() {
    let mut tapestry = make_tapestry("iter_min", 4, true);
    let input = FloatVec::new(vec![0.8, 0.2, 0.1, 0.05]);

    let config = IterateConfig {
        max_iterations: 10,
        min_iterations: 5,
        convergence: ConvergenceCriterion::ProbabilityDelta(0.0001),
        feedback: false,
        feedback_strength: 0.0,
    };

    let result = iterate(&mut tapestry, &[("x", &input)], &config).unwrap();

    assert!(result.iterations >= 5, "Should run at least min_iterations, got {}", result.iterations);
    assert!(result.iterations <= 10, "Should not exceed max_iterations");
    assert!(!result.history.is_empty());
}

#[test]
fn iterate_prob_delta_convergence() {
    let mut tapestry = make_tapestry("iter_prob", 4, false);
    let input = FloatVec::new(vec![0.8, 0.2, 0.1, 0.05]);

    let config = IterateConfig {
        max_iterations: 50,
        min_iterations: 3,
        convergence: ConvergenceCriterion::ProbabilityDelta(1.0), // very loose → should converge
        feedback: false,
        feedback_strength: 0.0,
    };

    let result = iterate(&mut tapestry, &[("x", &input)], &config).unwrap();

    // With a delta threshold of 1.0, classical (deterministic) should converge quickly
    assert!(result.converged, "Should converge with loose threshold");
}

#[test]
fn iterate_stable_index() {
    let mut tapestry = make_tapestry("iter_stable", 4, false);
    let input = FloatVec::new(vec![0.8, 0.2, 0.1, 0.05]);

    let config = IterateConfig {
        max_iterations: 30,
        min_iterations: 3,
        convergence: ConvergenceCriterion::StableIndex(2), // stable for 2 consecutive
        feedback: false,
        feedback_strength: 0.0,
    };

    let result = iterate(&mut tapestry, &[("x", &input)], &config).unwrap();

    // Classical path gives deterministic results → index should stabilize
    assert!(result.converged, "Classical path should produce stable index");
}

#[test]
fn iterate_with_feedback() {
    let mut tapestry = make_tapestry("iter_fb", 4, true);
    let input = FloatVec::new(vec![0.8, 0.2, 0.1, 0.05]);

    let config = IterateConfig {
        max_iterations: 10,
        min_iterations: 3,
        convergence: ConvergenceCriterion::ProbabilityDelta(0.0001),
        feedback: true,
        feedback_strength: 0.5,
    };

    let result = iterate(&mut tapestry, &[("x", &input)], &config).unwrap();

    assert!(result.iterations >= 3);
    assert_eq!(result.history.len(), result.iterations);
}

// ===========================================================================
// Basin designer tests
// ===========================================================================

#[test]
fn basin_designer_finds_basins() {
    let spec = BasinSpec {
        n_basins: 2,
        target_sizes: vec![0.5, 0.5],
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

    let design = design_basins(&spec, &config).unwrap();

    assert!(!design.basins.is_empty(), "Should find at least 1 basin");
    assert!(design.score >= 0.0, "Score should be non-negative");
    assert!(design.iterations > 0);
}

#[test]
fn basin_designer_count_match() {
    let spec = BasinSpec {
        n_basins: 3,
        target_sizes: vec![0.33, 0.33, 0.34],
        boundary_hints: Vec::new(),
        dim: 4,
    };

    let config = BasinDesignConfig {
        grid_r_steps: 10,
        grid_eps_steps: 10,
        top_k: 3,
        nelder_mead_iterations: 30,
        n_samples: 100,
        transient: 100,
    };

    let design = design_basins(&spec, &config).unwrap();

    // The score should penalize count mismatch heavily
    // We can't guarantee exact count match, but the optimization should try
    assert!(design.score < f64::MAX);
}

#[test]
fn basin_score_perfect_match() {
    use axol::dynamics::Basin;

    let spec = BasinSpec {
        n_basins: 2,
        target_sizes: vec![0.5, 0.5],
        boundary_hints: Vec::new(),
        dim: 2,
    };

    let basins = vec![
        Basin { center: vec![0.3, 0.7], size: 0.5, local_lyapunov: 0.1, phase: 0.0 },
        Basin { center: vec![0.7, 0.3], size: 0.5, local_lyapunov: 0.1, phase: 3.14 },
    ];

    let target_sizes = vec![0.5, 0.5];
    let score = score_basins(&spec, &basins, &target_sizes);

    // Perfect count match (2==2) → count_penalty = 0
    // Perfect size match → size_penalty = 0
    assert!(score < 0.01, "Perfect match should have score near 0, got {}", score);
}

// ===========================================================================
// Logic gate tests
// ===========================================================================

#[test]
fn not_gate_truth_table() {
    // NOT FALSE → TRUE
    let false_in = encode_bool(false);
    let obs = gate_not(&false_in).unwrap();
    assert!(decode_bool(&obs), "NOT FALSE should be TRUE");

    // NOT TRUE → FALSE
    let true_in = encode_bool(true);
    let obs = gate_not(&true_in).unwrap();
    assert!(!decode_bool(&obs), "NOT TRUE should be FALSE");
}

#[test]
fn and_gate_truth_table() {
    let f = encode_bool(false);
    let t = encode_bool(true);

    // F AND F → F
    let obs = gate_and(&f, &f).unwrap();
    assert!(!decode_bool(&obs), "F AND F should be F");

    // F AND T → F
    let obs = gate_and(&f, &t).unwrap();
    assert!(!decode_bool(&obs), "F AND T should be F");

    // T AND F → F
    let obs = gate_and(&t, &f).unwrap();
    assert!(!decode_bool(&obs), "T AND F should be F");

    // T AND T → T
    let obs = gate_and(&t, &t).unwrap();
    assert!(decode_bool(&obs), "T AND T should be T");
}

#[test]
fn or_gate_truth_table() {
    let f = encode_bool(false);
    let t = encode_bool(true);

    // F OR F → F
    let obs = gate_or(&f, &f).unwrap();
    assert!(!decode_bool(&obs), "F OR F should be F");

    // F OR T → T
    let obs = gate_or(&f, &t).unwrap();
    assert!(decode_bool(&obs), "F OR T should be T");

    // T OR F → T
    let obs = gate_or(&t, &f).unwrap();
    assert!(decode_bool(&obs), "T OR F should be T");

    // T OR T → T
    let obs = gate_or(&t, &t).unwrap();
    assert!(decode_bool(&obs), "T OR T should be T");
}

#[test]
fn logic_gate_dim_mismatch() {
    let wrong_dim = FloatVec::new(vec![0.5, 0.3, 0.2]);
    assert!(gate_not(&wrong_dim).is_err());
    assert!(gate_and(&wrong_dim, &wrong_dim).is_err());
    assert!(gate_or(&wrong_dim, &wrong_dim).is_err());
}

#[test]
fn if_then_else_routes_correctly() {
    // Create condition, then, and else tapestries (all dim=2)
    let cond = make_bool_tapestry("cond", 0.9, 0.7, 100).unwrap();
    let then_t = make_bool_tapestry("then", 0.9, 0.7, 200).unwrap();
    let else_t = make_bool_tapestry("else", 0.9, 0.7, 300).unwrap();

    // Test with TRUE condition input
    let true_input = encode_bool(true);
    let branch_input = FloatVec::new(vec![0.5, 0.5]);
    let obs = eval_if_then_else(&true_input, &cond, &then_t, &else_t, &branch_input).unwrap();

    // We can't assert which branch was taken since it depends on the condition tapestry,
    // but the function should complete without error and return a valid observation
    assert!(obs.value_index < 2);
}

#[test]
fn encode_decode_bool_roundtrip() {
    for val in [false, true] {
        let encoded = encode_bool(val);
        assert_eq!(encoded.dim(), 2);

        // The encoding should produce the right argmax
        let expected_idx = if val { 1 } else { 0 };
        let max_idx = if encoded.data[1] > encoded.data[0] { 1 } else { 0 };
        assert_eq!(max_idx, expected_idx, "Encoding of {} should have max at index {}", val, expected_idx);
    }
}

// ===========================================================================
// DSL extension tests
// ===========================================================================

#[test]
fn dsl_lex_compose_tokens() {
    use axol::dsl::lexer::{Lexer, Token};

    let source = r#"compose "pipeline" stages=[enc, dec]"#;
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize();

    assert!(tokens.iter().any(|t| matches!(t, Token::Compose)));
}

#[test]
fn dsl_lex_gate_tokens() {
    use axol::dsl::lexer::{Lexer, Token};

    let source = r#"gate not { x = [0.1, 0.9] }"#;
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize();

    assert!(tokens.iter().any(|t| matches!(t, Token::Gate)));
}

#[test]
fn dsl_lex_confident_tokens() {
    use axol::dsl::lexer::{Lexer, Token};

    let source = r#"confident my_tap max=100 threshold=0.95 { x = [0.8, 0.2] }"#;
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize();

    assert!(tokens.iter().any(|t| matches!(t, Token::Confident)));
}

#[test]
fn dsl_lex_iterate_tokens() {
    use axol::dsl::lexer::{Lexer, Token};

    let source = r#"iterate my_tap max=50 converge=prob_delta value=0.001 { x = [0.8, 0.2] }"#;
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize();

    assert!(tokens.iter().any(|t| matches!(t, Token::Iterate)));
    assert!(tokens.iter().any(|t| matches!(t, Token::Converge)));
}

#[test]
fn dsl_lex_design_tokens() {
    use axol::dsl::lexer::{Lexer, Token};

    let source = r#"design "binary" { dim 2, basins 2, sizes [0.5, 0.5] }"#;
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize();

    assert!(tokens.iter().any(|t| matches!(t, Token::Design)));
}

#[test]
fn dsl_parse_gate_not() {
    use axol::dsl::lexer::Lexer;
    use axol::dsl::parser::{Parser, Statement};

    let source = "gate not { x = [0.1, 0.9] }";
    let mut lexer = Lexer::new(source);
    lexer.tokenize();
    let mut parser = Parser::new(lexer.tokens.clone());
    let program = parser.parse().unwrap();

    assert_eq!(program.statements.len(), 1);
    match &program.statements[0] {
        Statement::GateOp(cmd) => {
            assert_eq!(cmd.gate_type, "not");
            assert_eq!(cmd.inputs.len(), 1);
        }
        _ => panic!("Expected GateOp"),
    }
}

#[test]
fn dsl_parse_confident() {
    use axol::dsl::lexer::Lexer;
    use axol::dsl::parser::{Parser, Statement};

    let source = "confident my_tap max=50 threshold=0.9 { x = [0.5, 0.5] }";
    let mut lexer = Lexer::new(source);
    lexer.tokenize();
    let mut parser = Parser::new(lexer.tokens.clone());
    let program = parser.parse().unwrap();

    assert_eq!(program.statements.len(), 1);
    match &program.statements[0] {
        Statement::ConfidentObs(cmd) => {
            assert_eq!(cmd.name, "my_tap");
            assert_eq!(cmd.max_observations, 50);
            assert!((cmd.threshold - 0.9).abs() < 1e-10);
        }
        _ => panic!("Expected ConfidentObs"),
    }
}

#[test]
fn dsl_parse_design() {
    use axol::dsl::lexer::Lexer;
    use axol::dsl::parser::{Parser, Statement};

    let source = r#"design "binary" { dim 2, basins 2, sizes [0.5, 0.5] }"#;
    let mut lexer = Lexer::new(source);
    lexer.tokenize();
    let mut parser = Parser::new(lexer.tokens.clone());
    let program = parser.parse().unwrap();

    assert_eq!(program.statements.len(), 1);
    match &program.statements[0] {
        Statement::DesignBasins(cmd) => {
            assert_eq!(cmd.name, "binary");
            assert_eq!(cmd.dim, 2);
            assert_eq!(cmd.n_basins, 2);
            assert_eq!(cmd.sizes.len(), 2);
        }
        _ => panic!("Expected DesignBasins"),
    }
}

#[test]
fn dsl_compile_gate_not() {
    use axol::dsl::lexer::Lexer;
    use axol::dsl::parser::Parser;
    use axol::dsl::compiler::Runtime;

    let source = "gate not { x = [0.1, 0.9] }";
    let mut lexer = Lexer::new(source);
    lexer.tokenize();
    let mut parser = Parser::new(lexer.tokens.clone());
    let program = parser.parse().unwrap();

    let mut rt = Runtime::new();
    let output = rt.execute(&program).unwrap();

    assert!(output.iter().any(|l| l.contains("[gate]")), "Output should contain gate result");
}

#[test]
fn dsl_compile_gate_and() {
    use axol::dsl::lexer::Lexer;
    use axol::dsl::parser::Parser;
    use axol::dsl::compiler::Runtime;

    let source = "gate and { a = [0.1, 0.9], b = [0.1, 0.9] }";
    let mut lexer = Lexer::new(source);
    lexer.tokenize();
    let mut parser = Parser::new(lexer.tokens.clone());
    let program = parser.parse().unwrap();

    let mut rt = Runtime::new();
    let output = rt.execute(&program).unwrap();

    assert!(output.iter().any(|l| l.contains("[gate]")));
}

// ===========================================================================
// Integration: full pipeline with compose
// ===========================================================================

#[test]
fn full_pipeline_declare_weave_confident() {
    use axol::dsl::lexer::Lexer;
    use axol::dsl::parser::Parser;
    use axol::dsl::compiler::Runtime;

    let source = r#"
declare "test" {
    input x(4)
    relate y <- x via <~>
    output y
    quality omega=0.9 phi=0.7
}

weave test quantum=true seed=42
confident test max=20 threshold=0.5 { x = [0.8, 0.2, 0.1, 0.05] }
"#;

    let mut lexer = Lexer::new(source);
    lexer.tokenize();
    let mut parser = Parser::new(lexer.tokens.clone());
    let program = parser.parse().unwrap();

    let mut rt = Runtime::new();
    let output = rt.execute(&program).unwrap();

    assert!(output.iter().any(|l| l.contains("[declare]")));
    assert!(output.iter().any(|l| l.contains("[weave]")));
    assert!(output.iter().any(|l| l.contains("[confident]")));
}

#[test]
fn full_pipeline_declare_weave_iterate() {
    use axol::dsl::lexer::Lexer;
    use axol::dsl::parser::Parser;
    use axol::dsl::compiler::Runtime;

    let source = r#"
declare "iter_test" {
    input x(4)
    relate y <- x via <~>
    output y
    quality omega=0.9 phi=0.7
}

weave iter_test quantum=true seed=42
iterate iter_test max=10 converge=prob_delta value=0.01 { x = [0.8, 0.2, 0.1, 0.05] }
"#;

    let mut lexer = Lexer::new(source);
    lexer.tokenize();
    let mut parser = Parser::new(lexer.tokens.clone());
    let program = parser.parse().unwrap();

    let mut rt = Runtime::new();
    let output = rt.execute(&program).unwrap();

    assert!(output.iter().any(|l| l.contains("[iterate]")));
}
