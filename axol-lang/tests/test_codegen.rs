//! Tests for the codegen module — verifies generated .axol programs parse and run.

use axol::codegen;
use axol::dsl::lexer::Lexer;
use axol::dsl::parser::Parser;
use axol::dsl::compiler::Runtime;

fn parse_and_run(source: &str) -> Vec<String> {
    let mut lexer = Lexer::new(source);
    lexer.tokenize();
    let mut parser = Parser::new(lexer.tokens.clone());
    let program = parser.parse().expect("parse failed");
    let mut runtime = Runtime::new();
    runtime.execute(&program).expect("runtime failed")
}

fn parse_only(source: &str) -> usize {
    let mut lexer = Lexer::new(source);
    lexer.tokenize();
    let mut parser = Parser::new(lexer.tokens.clone());
    let program = parser.parse().expect("parse failed");
    program.statements.len()
}

// ---------------------------------------------------------------------------
// Generator builder tests
// ---------------------------------------------------------------------------

#[test]
fn generator_produces_valid_axol() {
    let mut g = codegen::AxolGenerator::new();
    g.comment("test program");
    g.gate("not", &[("x", &[0.1, 0.9])]);
    let source = g.generate();
    assert!(source.contains("# test program"));
    assert!(source.contains("gate not"));
    let stmts = parse_only(&source);
    assert_eq!(stmts, 1);
}

#[test]
fn generator_declare_weave_observe() {
    let mut g = codegen::AxolGenerator::new();
    g.declare(
        "test",
        &[("x", 4)],
        &["y"],
        &[("y", &["x"], "<~>")],
        Some((0.9, 0.8)),
    );
    g.weave("test", true, 42);
    g.observe("test", &[("x", &[0.5, 0.3, 0.7, 0.2])]);
    let source = g.generate();
    let stmts = parse_only(&source);
    assert_eq!(stmts, 3);
}

// ---------------------------------------------------------------------------
// Template tests — each template must parse successfully
// ---------------------------------------------------------------------------

#[test]
fn template_not_parses() {
    let source = codegen::gen_not(true);
    assert_eq!(parse_only(&source), 1);
}

#[test]
fn template_and_parses() {
    let source = codegen::gen_and(true, false);
    assert_eq!(parse_only(&source), 1);
}

#[test]
fn template_or_parses() {
    let source = codegen::gen_or(false, false);
    assert_eq!(parse_only(&source), 1);
}

#[test]
fn template_xor_parses() {
    let source = codegen::gen_xor(true, false);
    assert_eq!(parse_only(&source), 4); // OR, AND, NOT, AND
}

#[test]
fn template_half_adder_parses() {
    let source = codegen::gen_half_adder(true, true);
    assert_eq!(parse_only(&source), 5); // AND, OR, AND, NOT, AND
}

#[test]
fn template_demo_parses() {
    let source = codegen::gen_full_demo(42);
    let stmts = parse_only(&source);
    assert!(stmts >= 7); // gates + declare + weave + observe + confident + iterate
}

// ---------------------------------------------------------------------------
// End-to-end: generate then run
// ---------------------------------------------------------------------------

#[test]
fn gen_not_runs_correctly() {
    let source = codegen::gen_not(true);
    let output = parse_and_run(&source);
    // NOT(TRUE) -> idx=0 (FALSE)
    let gate_line = output.iter().find(|l| l.starts_with("[gate]")).unwrap();
    assert!(gate_line.contains("idx=0"), "NOT(TRUE) should give idx=0, got: {}", gate_line);
}

#[test]
fn gen_and_runs_correctly() {
    let source = codegen::gen_and(true, true);
    let output = parse_and_run(&source);
    let gate_line = output.iter().find(|l| l.starts_with("[gate]")).unwrap();
    assert!(gate_line.contains("idx=1"), "AND(T,T) should give idx=1, got: {}", gate_line);
}

#[test]
fn gen_xor_runs_correctly() {
    let source = codegen::gen_xor(true, false);
    let output = parse_and_run(&source);
    // Last gate line should be the final XOR result = TRUE (idx=1)
    let gate_lines: Vec<&String> = output.iter().filter(|l| l.starts_with("[gate]")).collect();
    let last = gate_lines.last().unwrap();
    assert!(last.contains("idx=1"), "XOR(T,F) should give idx=1, got: {}", last);
}

#[test]
fn gen_demo_runs_without_error() {
    let source = codegen::gen_full_demo(42);
    let output = parse_and_run(&source);
    assert!(output.len() > 5, "demo should produce multiple output lines");
    let total_line = output.last().unwrap();
    assert!(total_line.starts_with("--- Total:"), "should end with total time");
}

// ---------------------------------------------------------------------------
// Algorithm catalog
// ---------------------------------------------------------------------------

#[test]
fn catalog_covers_all_entries() {
    for (name, _) in codegen::ALGORITHM_CATALOG {
        let result = codegen::generate_algorithm(name, 42);
        assert!(result.is_some(), "catalog entry '{}' should generate code", name);
        let source = result.unwrap();
        assert!(!source.is_empty(), "generated source for '{}' should not be empty", name);
        // Must parse without error
        parse_only(&source);
    }
}

#[test]
fn unknown_algorithm_returns_none() {
    assert!(codegen::generate_algorithm("nonexistent", 42).is_none());
}

#[test]
fn list_algorithms_is_not_empty() {
    let list = codegen::list_algorithms();
    assert!(list.contains("not"));
    assert!(list.contains("xor"));
    assert!(list.contains("demo"));
}
