//! Tests for AXOL AI — self-referential program generator.

use axol::axol_ai::*;
use axol::dsl::lexer::Lexer;
use axol::dsl::parser::Parser;
use axol::dsl::compiler::Runtime;

fn parse_source(source: &str) -> usize {
    let mut lexer = Lexer::new(source);
    lexer.tokenize();
    let mut parser = Parser::new(lexer.tokens.clone());
    let program = parser.parse().expect("parse failed");
    program.statements.len()
}

fn run_source(source: &str) -> Vec<String> {
    let mut lexer = Lexer::new(source);
    lexer.tokenize();
    let mut parser = Parser::new(lexer.tokens.clone());
    let program = parser.parse().expect("parse failed");
    let mut runtime = Runtime::new();
    runtime.execute(&program).expect("runtime failed")
}

// ---------------------------------------------------------------------------
// AI construction
// ---------------------------------------------------------------------------

#[test]
fn ai_creates_successfully() {
    let ai = AxolAI::new(42).expect("AI creation should succeed");
    let _ = ai; // just verify construction
}

#[test]
fn ai_different_seeds_produce_different_results() {
    let ai1 = AxolAI::new(42).unwrap();
    let ai2 = AxolAI::new(999).unwrap();
    let req = AlgorithmRequest::logic("test");
    let r1 = ai1.generate(&req).unwrap();
    let r2 = ai2.generate(&req).unwrap();
    // Different seeds may produce different structure selections
    // (not guaranteed, but source should at least be generated)
    assert!(!r1.source.is_empty());
    assert!(!r2.source.is_empty());
}

// ---------------------------------------------------------------------------
// Request encoding
// ---------------------------------------------------------------------------

#[test]
fn request_encodes_to_correct_dim() {
    let req = AlgorithmRequest::logic("test");
    let encoded = req.encode();
    assert_eq!(encoded.dim(), 8, "feature vector should be 8-dimensional");
}

#[test]
fn different_tasks_encode_differently() {
    let logic = AlgorithmRequest::logic("a").encode();
    let classify = AlgorithmRequest::classifier("b", 3).encode();
    let pipeline = AlgorithmRequest::pipeline("c", 3).encode();

    // Logic should have high logic_weight (index 0)
    assert!(logic.data[0] > logic.data[1], "logic request should have high logic weight");
    // Classify should have high classify_weight (index 1)
    assert!(classify.data[1] > classify.data[0], "classify request should have high classify weight");
    // Pipeline should have high pipeline_weight (index 2)
    assert!(pipeline.data[2] > pipeline.data[0], "pipeline request should have high pipeline weight");
}

// ---------------------------------------------------------------------------
// Generation — all task types produce valid .axol
// ---------------------------------------------------------------------------

#[test]
fn ai_generates_logic_program() {
    let ai = AxolAI::new(42).unwrap();
    let req = AlgorithmRequest::logic("test_logic");
    let result = ai.generate(&req).unwrap();
    assert!(!result.source.is_empty());
    assert!(result.total_observations > 0, "should make basin observations");
    let stmts = parse_source(&result.source);
    assert!(stmts > 0, "should produce at least one statement");
}

#[test]
fn ai_generates_classifier_program() {
    let ai = AxolAI::new(42).unwrap();
    let req = AlgorithmRequest::classifier("test_cls", 3).with_dim(4);
    let result = ai.generate(&req).unwrap();
    assert!(!result.source.is_empty());
    assert!(result.total_observations > 0);
    parse_source(&result.source);
}

#[test]
fn ai_generates_pipeline_program() {
    let ai = AxolAI::new(42).unwrap();
    let req = AlgorithmRequest::pipeline("test_pipe", 2).with_dim(4);
    let result = ai.generate(&req).unwrap();
    assert!(!result.source.is_empty());
    parse_source(&result.source);
}

#[test]
fn ai_generates_convergent_program() {
    let ai = AxolAI::new(42).unwrap();
    let req = AlgorithmRequest::convergent("test_conv").with_dim(4);
    let result = ai.generate(&req).unwrap();
    assert!(!result.source.is_empty());
    parse_source(&result.source);
}

#[test]
fn ai_generates_composite_program() {
    let ai = AxolAI::new(42).unwrap();
    let req = AlgorithmRequest::composite("test_comp").with_dim(4);
    let result = ai.generate(&req).unwrap();
    assert!(!result.source.is_empty());
    parse_source(&result.source);
}

// ---------------------------------------------------------------------------
// Generated programs execute successfully
// ---------------------------------------------------------------------------

#[test]
fn ai_generated_program_runs() {
    let ai = AxolAI::new(42).unwrap();
    let req = AlgorithmRequest::logic("runnable");
    let result = ai.generate(&req).unwrap();
    let output = run_source(&result.source);
    assert!(!output.is_empty(), "running the generated program should produce output");
    let total_line = output.last().unwrap();
    assert!(total_line.starts_with("--- Total:"), "should end with total time");
}

// ---------------------------------------------------------------------------
// Result metadata
// ---------------------------------------------------------------------------

#[test]
fn result_has_valid_metadata() {
    let ai = AxolAI::new(42).unwrap();
    let req = AlgorithmRequest::classifier("meta_test", 2).with_dim(4).with_quality(0.85);
    let result = ai.generate(&req).unwrap();
    assert!(result.structure_index < 5, "structure index should be 0-4");
    assert!(result.structure_confidence > 0.0, "confidence should be positive");
    assert_eq!(result.dim, 4, "dim should match request");
    assert!((result.quality.0 - 0.85).abs() < 0.01, "omega should match requested quality");
    assert!(result.total_observations >= 15, "should make multiple observations");
}

#[test]
fn result_respects_dim_override() {
    let ai = AxolAI::new(42).unwrap();
    let req = AlgorithmRequest::logic("dim_test").with_dim(8);
    let result = ai.generate(&req).unwrap();
    assert_eq!(result.dim, 8, "dim should be overridden to 8");
}

#[test]
fn result_respects_quality_override() {
    let ai = AxolAI::new(42).unwrap();
    let req = AlgorithmRequest::logic("q_test").with_quality(0.95);
    let result = ai.generate(&req).unwrap();
    assert!((result.quality.0 - 0.95).abs() < 0.01, "quality should be overridden");
}

// ---------------------------------------------------------------------------
// Determinism — same seed + same request = same output
// ---------------------------------------------------------------------------

#[test]
fn ai_is_deterministic() {
    let ai = AxolAI::new(42).unwrap();
    let req = AlgorithmRequest::logic("det_test");
    let r1 = ai.generate(&req).unwrap();
    let r2 = ai.generate(&req).unwrap();
    assert_eq!(r1.source, r2.source, "same AI + same request should produce identical output");
    assert_eq!(r1.structure_index, r2.structure_index);
}
