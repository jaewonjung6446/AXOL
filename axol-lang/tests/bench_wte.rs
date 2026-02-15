//! Benchmarks for the Wave Text Engine (WTE).
//!
//! Measures speed, model size, and quality metrics.
//! Run with: cargo test --test bench_wte -- --show-output

use std::time::Instant;

use axol::text::engine::*;
use axol::text::data::medium_corpus;
use axol::text::reservoir::ReservoirState;

// ---------------------------------------------------------------------------
// Speed benchmarks
// ---------------------------------------------------------------------------

#[test]
fn bench_wte_autocomplete_speed() {
    let corpus = medium_corpus();
    let config = EngineConfig {
        dim: 64,
        num_merges: 200,
        max_vocab: 2000,
        seed: 42,
        anomaly_threshold: 1.5,
        ..EngineConfig::default()
    };

    println!("\n=== WTE Autocomplete Speed ===");

    let start = Instant::now();
    let engine = WaveTextEngine::from_corpus_with_config(&corpus, &config);
    let build_time = start.elapsed();
    println!("Build time: {:.1}ms", build_time.as_secs_f64() * 1000.0);

    // Warmup
    let _ = engine.autocomplete("the cat", 1);

    // Benchmark single-token autocomplete
    let n_iters = 100;
    let start = Instant::now();
    for _ in 0..n_iters {
        let _ = engine.autocomplete("the cat sat on", 1);
    }
    let elapsed = start.elapsed();
    let per_call = elapsed.as_secs_f64() * 1000.0 / n_iters as f64;

    println!("Autocomplete (1 token): {:.2}ms per call ({} iters)", per_call, n_iters);
    println!("Target: < 1ms");
}

#[test]
fn bench_wte_generation_speed() {
    let corpus = medium_corpus();
    let config = EngineConfig {
        dim: 64,
        num_merges: 200,
        max_vocab: 2000,
        seed: 42,
        anomaly_threshold: 1.5,
        ..EngineConfig::default()
    };

    let engine = WaveTextEngine::from_corpus_with_config(&corpus, &config);

    println!("\n=== WTE Generation Speed ===");

    // Benchmark 20-token generation
    let n_iters = 20;
    let start = Instant::now();
    for _ in 0..n_iters {
        let _ = engine.generate("the cat", 20);
    }
    let elapsed = start.elapsed();
    let per_call = elapsed.as_secs_f64() * 1000.0 / n_iters as f64;

    println!("Generate (20 tokens): {:.1}ms per call ({} iters)", per_call, n_iters);
    println!("Target: < 20ms");
}

#[test]
fn bench_wte_classification_speed() {
    let corpus = medium_corpus();
    let config = EngineConfig {
        dim: 64,
        num_merges: 200,
        max_vocab: 2000,
        seed: 42,
        anomaly_threshold: 1.5,
        ..EngineConfig::default()
    };

    let mut engine = WaveTextEngine::from_corpus_with_config(&corpus, &config);

    // Train classifier
    let labeled: Vec<(&str, usize)> = vec![
        ("the cat sat on the mat", 0),
        ("the dog ate the bone", 0),
        ("the bird flew over the tree", 0),
        ("the boy went to school", 1),
        ("the girl read a book", 1),
        ("the man walked to the store", 1),
        ("the sun shone in the sky", 2),
        ("the rain fell on the roof", 2),
        ("the wind blew through trees", 2),
    ];
    let labels = vec!["animals".into(), "people".into(), "weather".into()];
    engine.train_classifier(&labeled, labels);

    println!("\n=== WTE Classification Speed ===");

    let n_iters = 100;
    let start = Instant::now();
    for _ in 0..n_iters {
        let _ = engine.classify("the cat ran fast");
    }
    let elapsed = start.elapsed();
    let per_call = elapsed.as_secs_f64() * 1000.0 / n_iters as f64;

    println!("Classify: {:.2}ms per call ({} iters)", per_call, n_iters);
    println!("Target: < 2ms");
}

#[test]
fn bench_wte_anomaly_speed() {
    let corpus = medium_corpus();
    let config = EngineConfig {
        dim: 64,
        num_merges: 200,
        max_vocab: 2000,
        seed: 42,
        anomaly_threshold: 1.5,
        ..EngineConfig::default()
    };

    let mut engine = WaveTextEngine::from_corpus_with_config(&corpus, &config);

    let normal_docs: Vec<&str> = corpus[..20].iter().copied().collect();
    engine.train_anomaly_detector(&normal_docs);

    println!("\n=== WTE Anomaly Detection Speed ===");

    let n_iters = 100;
    let start = Instant::now();
    for _ in 0..n_iters {
        let _ = engine.anomaly_score("the cat sat on the mat");
    }
    let elapsed = start.elapsed();
    let per_call = elapsed.as_secs_f64() * 1000.0 / n_iters as f64;

    println!("Anomaly score: {:.2}ms per call ({} iters)", per_call, n_iters);
    println!("Target: < 5ms");
}

// ---------------------------------------------------------------------------
// Model size benchmark
// ---------------------------------------------------------------------------

#[test]
fn bench_wte_model_size() {
    println!("\n=== WTE Model Size ===");

    let corpus = medium_corpus();

    let configs = vec![
        ("dim=32, vocab=500", EngineConfig {
            dim: 32, num_merges: 100, max_vocab: 500, seed: 42, anomaly_threshold: 1.5,
            ..EngineConfig::default()
        }),
        ("dim=64, vocab=1000", EngineConfig {
            dim: 64, num_merges: 200, max_vocab: 1000, seed: 42, anomaly_threshold: 1.5,
            ..EngineConfig::default()
        }),
        ("dim=64, vocab=2000", EngineConfig {
            dim: 64, num_merges: 200, max_vocab: 2000, seed: 42, anomaly_threshold: 1.5,
            ..EngineConfig::default()
        }),
    ];

    for (name, config) in &configs {
        let engine = WaveTextEngine::from_corpus_with_config(&corpus, config);
        let size = engine.model_size_bytes();
        let serial_size = engine.to_bytes().len();
        println!("  {} → model={:.1}KB, serialized={:.1}KB",
            name, size as f64 / 1024.0, serial_size as f64 / 1024.0);
    }

    println!("Target: < 2MB for dim=64, vocab=2K");
}

// ---------------------------------------------------------------------------
// Training speed benchmark
// ---------------------------------------------------------------------------

#[test]
fn bench_wte_training_speed() {
    let corpus = medium_corpus();

    println!("\n=== WTE Training Speed ===");

    let config = EngineConfig {
        dim: 64,
        num_merges: 200,
        max_vocab: 2000,
        seed: 42,
        anomaly_threshold: 1.5,
        ..EngineConfig::default()
    };

    let start = Instant::now();
    let _engine = WaveTextEngine::from_corpus_with_config(&corpus, &config);
    let elapsed = start.elapsed();

    println!("Full training (medium corpus, {} sentences): {:.0}ms",
        corpus.len(), elapsed.as_secs_f64() * 1000.0);
    println!("Target: < 5000ms");

    // Domain adaptation speed
    let engine_for_adapt = WaveTextEngine::from_corpus_with_config(&corpus, &config);
    let new_docs = vec![
        "the robot cleaned the room",
        "the computer processed the data",
        "the machine learned the pattern",
    ];

    let start = Instant::now();
    let mut engine_mut = engine_for_adapt;
    engine_mut.adapt(&new_docs);
    let adapt_elapsed = start.elapsed();

    println!("Domain adaptation (3 docs): {:.1}ms", adapt_elapsed.as_secs_f64() * 1000.0);
    println!("Target: < 500ms");
}

// ---------------------------------------------------------------------------
// Feature dimension check
// ---------------------------------------------------------------------------

#[test]
fn bench_wte_feature_dimensions() {
    println!("\n=== WTE Feature Dimensions ===");

    for dim in [16, 32, 64, 128] {
        let feat_dim = ReservoirState::feature_dim(dim, 3);
        let readout_params = feat_dim * 500; // example vocab_size=500
        println!("  dim={:<4} → features={:<6} readout_params={:<8} ({:.1}KB)",
            dim, feat_dim, readout_params, readout_params as f64 * 8.0 / 1024.0);
    }
}

// ---------------------------------------------------------------------------
// LLM comparison (conceptual)
// ---------------------------------------------------------------------------

#[test]
fn bench_wte_vs_llm_efficiency() {
    println!("\n=== WTE vs LLM Efficiency (Conceptual) ===");
    println!("{:<20} {:<15} {:<15}", "", "WTE", "GPT-2 (small)");
    println!("{:<20} {:<15} {:<15}", "Parameters", "~260K", "124M");
    println!("{:<20} {:<15} {:<15}", "Training", "lstsq (1 shot)", "SGD (300K steps)");
    println!("{:<20} {:<15} {:<15}", "GPU required", "No", "Yes");
    println!("{:<20} {:<15} {:<15}", "Training data", "10-100 docs", "8M web pages");
    println!("{:<20} {:<15} {:<15}", "Model size", "< 2MB", "500MB");
    println!("{:<20} {:<15} {:<15}", "Inference", "< 1ms/token", "~50ms/token");
    println!("{:<20} {:<15} {:<15}", "Anomaly detection", "Built-in", "Separate model");
    println!("{:<20} {:<15} {:<15}", "Style analysis", "Zero-shot", "Fine-tuning");
    println!("{:<20} {:<15} {:<15}", "Context mechanism", "Physics O(nd)", "Attention O(n²d)");
    println!("{:<20} {:<15} {:<15}", "Quality metrics", "Ω/Φ built-in", "External");
}
