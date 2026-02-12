//! AXOL AI comprehensive benchmark — outputs CSV for visualization.
//! Run: cargo run --release --example bench_ai

use std::time::Instant;
use axol::axol_ai::*;
use axol::dsl::lexer::Lexer;
use axol::dsl::parser::Parser;
use axol::dsl::compiler::Runtime;

fn parse_and_run(source: &str) -> (usize, bool) {
    let mut lexer = Lexer::new(source);
    lexer.tokenize();
    let mut parser = Parser::new(lexer.tokens.clone());
    match parser.parse() {
        Ok(program) => {
            let stmts = program.statements.len();
            let mut runtime = Runtime::new();
            let ok = runtime.execute(&program).is_ok();
            (stmts, ok)
        }
        Err(_) => (0, false),
    }
}

fn main() {
    // === 1. Generation time & observations by task type ===
    println!("CSV:task_type");
    println!("task,gen_time_ms,observations,confidence,dim,stmts,runs_ok,source_bytes");

    let tasks: Vec<(&str, Box<dyn Fn() -> AlgorithmRequest>)> = vec![
        ("logic", Box::new(|| AlgorithmRequest::logic("bench"))),
        ("classify", Box::new(|| AlgorithmRequest::classifier("bench", 3))),
        ("pipeline", Box::new(|| AlgorithmRequest::pipeline("bench", 2))),
        ("converge", Box::new(|| AlgorithmRequest::convergent("bench"))),
        ("composite", Box::new(|| AlgorithmRequest::composite("bench"))),
    ];

    let ai = AxolAI::new(42).expect("AI creation failed");

    for (name, make_req) in &tasks {
        let req = make_req();
        let start = Instant::now();
        let result = ai.generate(&req).expect("generate failed");
        let gen_ms = start.elapsed().as_secs_f64() * 1000.0;

        let (stmts, runs_ok) = parse_and_run(&result.source);

        println!("{},{:.3},{},{:.4},{},{},{},{}",
            name, gen_ms, result.total_observations,
            result.structure_confidence, result.dim,
            stmts, runs_ok, result.source.len());
    }

    // === 2. Effect of seed on generation (logic task) ===
    println!("\nCSV:seed_effect");
    println!("seed,gen_time_ms,observations,confidence,structure_idx,dim");

    let seeds = [1, 7, 42, 100, 123, 256, 500, 777, 999, 1234, 2025, 3141, 5000, 9999, 31415];
    for &seed in &seeds {
        let ai = AxolAI::new(seed).expect("AI creation failed");
        let req = AlgorithmRequest::logic("bench");
        let start = Instant::now();
        let result = ai.generate(&req).expect("generate failed");
        let gen_ms = start.elapsed().as_secs_f64() * 1000.0;

        println!("{},{:.3},{},{:.4},{},{}",
            seed, gen_ms, result.total_observations,
            result.structure_confidence, result.structure_index, result.dim);
    }

    // === 3. AI construction time (weaving internal tapestries) ===
    println!("\nCSV:construction");
    println!("seed,construct_ms");

    for &seed in &seeds {
        let start = Instant::now();
        let _ai = AxolAI::new(seed).expect("AI creation failed");
        let construct_ms = start.elapsed().as_secs_f64() * 1000.0;
        println!("{},{:.3}", seed, construct_ms);
    }

    // === 4. Dimension sweep for each task type ===
    println!("\nCSV:dim_sweep");
    println!("task,dim,gen_time_ms,observations,stmts,source_bytes");

    let ai = AxolAI::new(42).expect("AI creation failed");
    let dim_values = [2, 4, 8, 16];
    let task_names = ["logic", "classify", "pipeline", "converge", "composite"];

    for task_name in &task_names {
        for &dim in &dim_values {
            let req = match *task_name {
                "logic" => AlgorithmRequest::logic("bench").with_dim(dim),
                "classify" => AlgorithmRequest::classifier("bench", 3).with_dim(dim),
                "pipeline" => AlgorithmRequest::pipeline("bench", 2).with_dim(dim),
                "converge" => AlgorithmRequest::convergent("bench").with_dim(dim),
                "composite" => AlgorithmRequest::composite("bench").with_dim(dim),
                _ => unreachable!(),
            };

            let start = Instant::now();
            let result = ai.generate(&req).expect("generate failed");
            let gen_ms = start.elapsed().as_secs_f64() * 1000.0;
            let (stmts, _) = parse_and_run(&result.source);

            println!("{},{},{:.3},{},{},{}",
                task_name, dim, gen_ms, result.total_observations,
                stmts, result.source.len());
        }
    }

    // === 5. Quality sweep ===
    println!("\nCSV:quality_sweep");
    println!("quality,gen_time_ms,observations,omega,phi");

    let ai = AxolAI::new(42).expect("AI creation failed");
    let qualities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95];

    for &q in &qualities {
        let req = AlgorithmRequest::logic("bench").with_quality(q);
        let start = Instant::now();
        let result = ai.generate(&req).expect("generate failed");
        let gen_ms = start.elapsed().as_secs_f64() * 1000.0;

        println!("{:.2},{:.3},{},{:.4},{:.4}",
            q, gen_ms, result.total_observations,
            result.quality.0, result.quality.1);
    }

    // === 6. Repeated generation (latency distribution) ===
    println!("\nCSV:latency_dist");
    println!("run,gen_time_us");

    let ai = AxolAI::new(42).expect("AI creation failed");
    let req = AlgorithmRequest::logic("bench");

    for i in 0..50 {
        let start = Instant::now();
        let _result = ai.generate(&req).expect("generate failed");
        let gen_us = start.elapsed().as_secs_f64() * 1e6;
        println!("{},{:.1}", i, gen_us);
    }

    // === 7. Learn vs AI comparison ===
    println!("\nCSV:learn_comparison");
    println!("gate,learn_time_ms,learn_accuracy,learn_evals,ai_time_ms,ai_observations");

    use axol::learn;

    let gates: Vec<(&str, learn::TrainingSet)> = vec![
        ("not", learn::TrainingSet::not()),
        ("and", learn::TrainingSet::and()),
        ("or", learn::TrainingSet::or()),
        ("xor", learn::TrainingSet::xor()),
    ];

    for (name, training) in &gates {
        let config = learn::LearnConfig {
            grid_r_steps: 10,
            grid_eps_steps: 10,
            nelder_mead_iters: 100,
            ..learn::LearnConfig::default()
        };

        let start = Instant::now();
        let learn_result = learn::learn(training, &config).expect("learn failed");
        let learn_ms = start.elapsed().as_secs_f64() * 1000.0;

        let ai = AxolAI::new(42).expect("AI failed");
        let req = AlgorithmRequest::logic(name);
        let start = Instant::now();
        let ai_result = ai.generate(&req).expect("generate failed");
        let ai_ms = start.elapsed().as_secs_f64() * 1000.0;

        println!("{},{:.1},{:.3},{},{:.3},{}",
            name, learn_ms, learn_result.accuracy,
            learn_result.total_evaluations,
            ai_ms, ai_result.total_observations);
    }

    eprintln!("\n[bench_ai] Done.");
}
