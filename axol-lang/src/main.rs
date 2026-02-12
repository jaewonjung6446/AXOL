//! AXOL CLI — run .axol programs.
//!
//! Usage:
//!   axol run program.axol
//!   axol check program.axol
//!   axol bench

use clap::{Parser, Subcommand};
use std::fs;
use std::time::Instant;

use axol::dsl::lexer::Lexer;
use axol::dsl::parser;
use axol::dsl::compiler::Runtime;
use axol::codegen;
use axol::axol_ai;
use axol::learn;
use axol::types::*;
use axol::ops;
use axol::density;

#[derive(Parser)]
#[command(name = "axol", version, about = "AXOL — Chaos-theory Declare/Weave/Observe language")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run an .axol program
    Run {
        /// Path to .axol file
        file: String,
    },
    /// Check syntax of an .axol program
    Check {
        /// Path to .axol file
        file: String,
    },
    /// Run built-in benchmark
    Bench,
    /// Generate an .axol program from a template
    Generate {
        /// Algorithm name (e.g., xor, half-adder, classifier, demo)
        algorithm: String,
        /// Output file path (prints to stdout if omitted)
        #[arg(short, long)]
        output: Option<String>,
        /// Random seed
        #[arg(long, default_value = "42")]
        seed: u64,
        /// List available algorithms
        #[arg(long)]
        list: bool,
    },
    /// Learn a tapestry from training data (built-in: xor, and, or, not)
    Learn {
        /// Built-in task: xor, and, or, not
        task: String,
        /// Disable quantum path (use classical only)
        #[arg(long)]
        classical: bool,
        /// Random seed
        #[arg(long, default_value = "42")]
        seed: u64,
    },
    /// AI-generate an .axol program using AXOL's own basin observation
    Ai {
        /// Task type: logic, classify, pipeline, converge, composite
        task: String,
        /// Program name
        #[arg(short, long, default_value = "program")]
        name: String,
        /// Output dimension
        #[arg(short, long)]
        dim: Option<usize>,
        /// Number of classes (for classify)
        #[arg(long)]
        classes: Option<usize>,
        /// Number of stages (for pipeline)
        #[arg(long)]
        stages: Option<usize>,
        /// Quality level 0.0..1.0
        #[arg(short, long)]
        quality: Option<f64>,
        /// Output file path
        #[arg(short, long)]
        output: Option<String>,
        /// Random seed for AI tapestries
        #[arg(long, default_value = "42")]
        seed: u64,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Run { file } => cmd_run(&file),
        Commands::Check { file } => cmd_check(&file),
        Commands::Bench => cmd_bench(),
        Commands::Learn { task, classical, seed } => cmd_learn(&task, classical, seed),
        Commands::Generate { algorithm, output, seed, list } => cmd_generate(&algorithm, output.as_deref(), seed, list),
        Commands::Ai { task, name, dim, classes, stages, quality, output, seed } => {
            cmd_ai(&task, &name, dim, classes, stages, quality, output.as_deref(), seed);
        }
    }
}

fn cmd_run(path: &str) {
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading {}: {}", path, e);
            std::process::exit(1);
        }
    };

    let mut lexer = Lexer::new(&source);
    lexer.tokenize();

    let mut parser = parser::Parser::new(lexer.tokens.clone());
    let program = match parser.parse() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Parse error: {}", e);
            std::process::exit(1);
        }
    };

    let mut runtime = Runtime::new();
    match runtime.execute(&program) {
        Ok(lines) => {
            for line in lines {
                println!("{}", line);
            }
        }
        Err(e) => {
            eprintln!("Runtime error: {}", e);
            std::process::exit(1);
        }
    }
}

fn cmd_check(path: &str) {
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading {}: {}", path, e);
            std::process::exit(1);
        }
    };

    let mut lexer = Lexer::new(&source);
    let tokens = lexer.tokenize();
    println!("Tokens: {}", tokens.len());

    let mut parser = parser::Parser::new(lexer.tokens.clone());
    match parser.parse() {
        Ok(program) => {
            println!("OK: {} statements", program.statements.len());
            for stmt in &program.statements {
                match stmt {
                    axol::dsl::parser::Statement::Declare(d) => {
                        println!("  declare '{}': {} inputs, {} relations", d.name, d.inputs.len(), d.relations.len());
                    }
                    axol::dsl::parser::Statement::Weave(w) => {
                        println!("  weave '{}': quantum={}", w.name, w.quantum);
                    }
                    axol::dsl::parser::Statement::Observe(o) => {
                        println!("  observe '{}': {} inputs", o.name, o.inputs.len());
                    }
                    axol::dsl::parser::Statement::Reobserve(r) => {
                        println!("  reobserve '{}' x{}", r.name, r.count);
                    }
                    axol::dsl::parser::Statement::ComposeChain(c) => {
                        println!("  compose '{}': {} stages", c.name, c.stages.len());
                    }
                    axol::dsl::parser::Statement::GateOp(g) => {
                        println!("  gate '{}': {} inputs", g.gate_type, g.inputs.len());
                    }
                    axol::dsl::parser::Statement::ConfidentObs(c) => {
                        println!("  confident '{}': max={}", c.name, c.max_observations);
                    }
                    axol::dsl::parser::Statement::IterateObs(i) => {
                        println!("  iterate '{}': max={}", i.name, i.max_iterations);
                    }
                    axol::dsl::parser::Statement::DesignBasins(d) => {
                        println!("  design '{}': dim={} basins={}", d.name, d.dim, d.n_basins);
                    }
                    axol::dsl::parser::Statement::Learn(l) => {
                        println!("  learn '{}': dim={} samples={}", l.name, l.dim, l.samples.len());
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("Parse error: {}", e);
            std::process::exit(1);
        }
    }
}

fn cmd_learn(task: &str, classical: bool, seed: u64) {
    let training = match task {
        "xor" => learn::TrainingSet::xor(),
        "and" => learn::TrainingSet::and(),
        "or" => learn::TrainingSet::or(),
        "not" => learn::TrainingSet::not(),
        _ => {
            eprintln!("Unknown task: '{}'\n", task);
            eprintln!("Available tasks:");
            eprintln!("  xor   — XOR gate (non-linearly separable)");
            eprintln!("  and   — AND gate");
            eprintln!("  or    — OR gate");
            eprintln!("  not   — NOT gate");
            std::process::exit(1);
        }
    };

    let config = learn::LearnConfig {
        quantum: !classical,
        seeds: vec![seed, seed + 81, seed + 235],
        ..learn::LearnConfig::default()
    };

    eprintln!("[learn] Training '{}' (dim={}, quantum={})...",
        task, training.dim, config.quantum);

    let start = Instant::now();
    let result = match learn::learn(&training, &config) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };
    let elapsed = start.elapsed();

    eprintln!("[learn] Accuracy: {:.1}% ({}/{})",
        result.accuracy * 100.0,
        (result.accuracy * training.samples.len() as f64).round() as usize,
        training.samples.len());
    eprintln!("[learn] Params: r={:.4} eps={:.4} relation={:?}",
        result.best_r, result.best_epsilon, result.best_relation);
    eprintln!("[learn] Evaluations: {} ({:.1}s)",
        result.total_evaluations, elapsed.as_secs_f64());

    // Verify with full observation pipeline
    let verify_accuracy = learn::evaluate(&result.tapestry, &training);
    eprintln!("[learn] Verified accuracy: {:.1}%", verify_accuracy * 100.0);
}

fn cmd_ai(
    task: &str,
    name: &str,
    dim: Option<usize>,
    classes: Option<usize>,
    stages: Option<usize>,
    quality: Option<f64>,
    output: Option<&str>,
    seed: u64,
) {
    let task_kind = match task {
        "logic" => axol_ai::TaskKind::Logic,
        "classify" | "classifier" => axol_ai::TaskKind::Classify,
        "pipeline" | "chain" => axol_ai::TaskKind::Pipeline,
        "converge" | "convergent" | "iterate" => axol_ai::TaskKind::Converge,
        "composite" | "full" | "demo" => axol_ai::TaskKind::Composite,
        _ => {
            eprintln!("Unknown task type: '{}'\n", task);
            eprintln!("Available tasks:");
            eprintln!("  logic       — Logic gates/circuits (NOT, AND, OR, XOR, ...)");
            eprintln!("  classify    — Binary or multi-class classifier");
            eprintln!("  pipeline    — Sequential tapestry chain");
            eprintln!("  converge    — Convergent iteration");
            eprintln!("  composite   — Combined logic + classify + iterate");
            std::process::exit(1);
        }
    };

    let mut request = axol_ai::AlgorithmRequest::new(name, task_kind);
    request.dim = dim;
    request.n_classes = classes;
    request.n_stages = stages;
    request.quality = quality;

    // Weave the AI tapestries
    eprintln!("[AI] Weaving generator tapestries (seed={})...", seed);
    let ai = match axol_ai::AxolAI::new(seed) {
        Ok(ai) => ai,
        Err(e) => {
            eprintln!("Error creating AI: {}", e);
            std::process::exit(1);
        }
    };

    // Generate via basin observation
    eprintln!("[AI] Observing basins to generate program...");
    let result = match ai.generate(&request) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error generating: {}", e);
            std::process::exit(1);
        }
    };

    eprintln!("[AI] Structure: type={} (confidence={:.2})", result.structure_index, result.structure_confidence);
    eprintln!("[AI] Params: dim={} quality=({:.2}, {:.2})", result.dim, result.quality.0, result.quality.1);
    eprintln!("[AI] Total basin observations: {}", result.total_observations);

    match output {
        Some(path) => {
            match fs::write(path, &result.source) {
                Ok(_) => eprintln!("[AI] Written: {} ({} bytes)", path, result.source.len()),
                Err(e) => {
                    eprintln!("Error writing {}: {}", path, e);
                    std::process::exit(1);
                }
            }
        }
        None => print!("{}", result.source),
    }
}

fn cmd_generate(algorithm: &str, output: Option<&str>, seed: u64, list: bool) {
    if list || algorithm == "list" {
        print!("{}", codegen::list_algorithms());
        return;
    }

    let source = match codegen::generate_algorithm(algorithm, seed) {
        Some(s) => s,
        None => {
            eprintln!("Unknown algorithm: '{}'", algorithm);
            eprintln!();
            eprint!("{}", codegen::list_algorithms());
            std::process::exit(1);
        }
    };

    match output {
        Some(path) => {
            match fs::write(path, &source) {
                Ok(_) => println!("Generated: {} ({} bytes)", path, source.len()),
                Err(e) => {
                    eprintln!("Error writing {}: {}", path, e);
                    std::process::exit(1);
                }
            }
        }
        None => print!("{}", source),
    }
}

fn cmd_bench() {
    use num_complex::Complex64;

    println!("=== AXOL Rust Native Benchmark ===\n");

    // 1. Born rule
    println!("[1] Born rule (measure)");
    for dim in [8, 64, 256, 1024] {
        let data: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
        let fv = FloatVec::new(data);

        let start = Instant::now();
        let iters = 100_000;
        for _ in 0..iters {
            std::hint::black_box(ops::measure(&fv));
        }
        let us = start.elapsed().as_secs_f64() / iters as f64 * 1e6;
        println!("  dim={:>5}: {:.3} us", dim, us);
    }

    // 2. Matrix-vector multiply
    println!("\n[2] Matrix-vector multiply (transform)");
    for dim in [8, 64, 256] {
        let mat_data: Vec<f32> = (0..dim * dim).map(|i| ((i as f32) * 0.01).sin()).collect();
        let vec_data: Vec<f32> = (0..dim).map(|i| (i as f32).cos()).collect();
        let tm = TransMatrix::new(mat_data, dim, dim);
        let fv = FloatVec::new(vec_data);

        let start = Instant::now();
        let iters = 100_000;
        for _ in 0..iters {
            std::hint::black_box(ops::transform(&fv, &tm).unwrap());
        }
        let us = start.elapsed().as_secs_f64() / iters as f64 * 1e6;
        println!("  dim={:>5}: {:.3} us", dim, us);
    }

    // 3. Complex Born rule
    println!("\n[3] Complex Born rule (measure_complex)");
    for dim in [8, 64, 256, 1024] {
        let data: Vec<Complex64> = (0..dim)
            .map(|i| Complex64::new((i as f64).sin(), (i as f64).cos()))
            .collect();
        let cv = ComplexVec::new(data).normalized();

        let start = Instant::now();
        let iters = 100_000;
        for _ in 0..iters {
            std::hint::black_box(ops::measure_complex(&cv));
        }
        let us = start.elapsed().as_secs_f64() / iters as f64 * 1e6;
        println!("  dim={:>5}: {:.3} us", dim, us);
    }

    // 4. Interference
    println!("\n[4] Interference");
    for dim in [8, 64, 256, 1024] {
        let a = ComplexVec::new(
            (0..dim).map(|i| Complex64::new((i as f64).sin(), 0.0)).collect()
        ).normalized();
        let b = ComplexVec::new(
            (0..dim).map(|i| Complex64::new((i as f64).cos(), 0.0)).collect()
        ).normalized();

        let start = Instant::now();
        let iters = 100_000;
        for _ in 0..iters {
            std::hint::black_box(ops::interfere(&a, &b, 0.3).unwrap());
        }
        let us = start.elapsed().as_secs_f64() / iters as f64 * 1e6;
        println!("  dim={:>5}: {:.3} us", dim, us);
    }

    // 5. Density matrix ops
    println!("\n[5] Density matrix (from_pure + purity)");
    for dim in [4, 8, 16, 32] {
        let psi = ComplexVec::new(
            (0..dim).map(|i| Complex64::new((i as f64 * 0.7).sin(), (i as f64 * 0.3).cos())).collect()
        ).normalized();

        let start = Instant::now();
        let iters = 50_000;
        for _ in 0..iters {
            let rho = DensityMatrix::from_pure_state(&psi);
            std::hint::black_box(rho.purity());
        }
        let us = start.elapsed().as_secs_f64() / iters as f64 * 1e6;
        println!("  dim={:>5}: {:.3} us", dim, us);
    }

    // 6. Quantum channel
    println!("\n[6] Dephasing channel (apply)");
    for dim in [4, 8, 16, 32] {
        let psi = ComplexVec::new(
            (0..dim).map(|i| Complex64::new((i as f64).sin(), (i as f64).cos())).collect()
        ).normalized();
        let rho = DensityMatrix::from_pure_state(&psi);
        let kraus = density::dephasing_channel(0.3, dim);

        let start = Instant::now();
        let iters = 10_000;
        for _ in 0..iters {
            std::hint::black_box(density::apply_channel(&rho, &kraus));
        }
        let us = start.elapsed().as_secs_f64() / iters as f64 * 1e6;
        println!("  dim={:>5}: {:.3} us ({} Kraus ops)", dim, us, kraus.len());
    }

    // 7. Von Neumann entropy
    println!("\n[7] Von Neumann entropy");
    for dim in [4, 8, 16, 32] {
        let psi = ComplexVec::new(
            (0..dim).map(|i| Complex64::new((i as f64).sin(), (i as f64).cos())).collect()
        ).normalized();
        let rho = DensityMatrix::from_pure_state(&psi);

        let start = Instant::now();
        let iters = 10_000;
        for _ in 0..iters {
            std::hint::black_box(density::von_neumann_entropy(&rho));
        }
        let us = start.elapsed().as_secs_f64() / iters as f64 * 1e6;
        println!("  dim={:>5}: {:.3} us  S={:.6}", dim, us, density::von_neumann_entropy(&rho));
    }

    println!("\n=== Done ===");
}
