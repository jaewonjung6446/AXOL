//! Integration tests — verify example .axol files parse and execute without error.
//!
//! Each test reads an example file, tokenizes, parses, and executes it.
//! Asserts no panic/error and non-empty output.

use std::fs;
use axol::dsl::lexer::Lexer;
use axol::dsl::parser::Parser;
use axol::dsl::compiler::Runtime;

// =========================================================================
// Helper
// =========================================================================

fn run_example(filename: &str) {
    let path = format!(
        "{}\\examples\\{}",
        env!("CARGO_MANIFEST_DIR"),
        filename
    );
    let source = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {path}: {e}"));

    let mut lexer = Lexer::new(&source);
    lexer.tokenize();

    let mut parser = Parser::new(lexer.tokens.clone());
    let program = parser.parse()
        .unwrap_or_else(|e| panic!("parse failed for {filename}: {e}"));

    let mut rt = Runtime::new();
    let output = rt.execute(&program)
        .unwrap_or_else(|e| panic!("execution failed for {filename}: {e}"));

    assert!(
        !output.is_empty(),
        "{filename} produced no output"
    );
}

// =========================================================================
// Core examples
// =========================================================================

#[test]
fn example_hello() {
    run_example("hello.axol");
}

#[test]
fn example_hello_v2() {
    run_example("hello_v2.axol");
}

#[test]
fn example_demo() {
    run_example("demo.axol");
}

// =========================================================================
// Classifier & convergence
// =========================================================================

#[test]
fn example_classifier() {
    run_example("classifier.axol");
}

#[test]
fn example_convergent() {
    run_example("convergent.axol");
}

// =========================================================================
// Basin definition
// =========================================================================

#[test]
fn example_define_basins() {
    run_example("define_basins.axol");
}

// =========================================================================
// Logic & learning
// =========================================================================

#[test]
fn example_half_adder() {
    run_example("half_adder.axol");
}

#[test]
fn example_xor() {
    run_example("xor.axol");
}

#[test]
fn example_learn_xor() {
    run_example("learn_xor.axol");
}

#[test]
fn example_ai_logic() {
    run_example("ai_logic.axol");
}

// =========================================================================
// Multi-input
// =========================================================================

#[test]
fn example_test_multi_input() {
    run_example("test_multi_input.axol");
}

// =========================================================================
// Use-case examples
// =========================================================================

#[test]
fn example_usecase_npc_realtime() {
    run_example("usecase_npc_realtime.axol");
}

#[test]
fn example_usecase_perception() {
    run_example("usecase_perception.axol");
}

#[test]
fn example_usecase_dialogue() {
    run_example("usecase_dialogue.axol");
}

#[test]
fn example_usecase_observation_cost() {
    run_example("usecase_observation_cost.axol");
}
