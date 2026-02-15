//! MLP vs Linear readout 비교 벤치마크.
//!
//! Run with: cargo test --release --test bench_mlp_vs_linear -- --show-output

use std::time::Instant;
use axol::text::engine::*;
use axol::text::data::medium_corpus;
use axol::text::generator::softmax;

fn classification_data() -> (Vec<(&'static str, usize)>, Vec<String>) {
    let labeled = vec![
        ("the cat sat on the mat", 0),
        ("the dog ate the bone", 0),
        ("the bird flew over the tree", 0),
        ("the cat ate the fish", 0),
        ("the dog ran in the park", 0),
        ("the bird sang in the tree", 0),
        ("the boy went to school", 1),
        ("the girl read a book", 1),
        ("the man walked to the store", 1),
        ("the woman cooked dinner", 1),
        ("the boy ran home", 1),
        ("the girl played in the yard", 1),
        ("the sun shone in the sky", 2),
        ("the rain fell on the roof", 2),
        ("the wind blew through trees", 2),
        ("the snow covered the ground", 2),
        ("the moon rose at night", 2),
        ("the stars shone brightly", 2),
    ];
    let labels = vec!["animals".into(), "people".into(), "nature".into()];
    (labeled, labels)
}

fn classification_test_cases() -> Vec<(&'static str, &'static str)> {
    vec![
        ("the cat ran fast", "animals"),
        ("the dog played fetch", "animals"),
        ("the bird nested", "animals"),
        ("the boy ate lunch", "people"),
        ("the girl went home", "people"),
        ("the man sat down", "people"),
        ("the rain poured down", "nature"),
        ("the wind howled", "nature"),
        ("the sun set slowly", "nature"),
    ]
}

fn autocomplete_test_cases() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        ("the cat sat on", vec!["the", "mat", "a"]),
        ("the dog ate", vec!["the", "a", "his"]),
        ("the boy went to", vec!["the", "school", "a"]),
        ("the bird flew over", vec!["the", "tree", "a"]),
        ("the girl read a", vec!["book", "the", "a"]),
        ("the cat", vec!["sat", "ate", "ran"]),
        ("the dog sat on", vec!["the", "mat", "a"]),
        ("the boy", vec!["went", "ran", "walked"]),
    ]
}

#[test]
fn bench_mlp_vs_linear() {
    println!("\n{}", "=".repeat(70));
    println!("  MLP vs Linear Readout 비교");
    println!("{}", "=".repeat(70));

    let corpus = medium_corpus();

    for (label, hidden_dim) in [
        ("Linear (hidden=0)", 0),
        ("MLP (hidden=64)", 64),
        ("MLP (hidden=128)", 128),
        ("MLP (hidden=256)", 256),
    ] {
        println!("\n--- {} ---", label);

        let config = EngineConfig {
            dim: 64,
            num_merges: 200,
            max_vocab: 2000,
            seed: 42,
            hidden_dim,
            ..EngineConfig::default()
        };

        let t0 = Instant::now();
        let mut engine = WaveTextEngine::from_corpus_with_config(&corpus, &config);
        let train_ms = t0.elapsed().as_millis();
        println!("  Train time: {}ms", train_ms);

        // Model size
        let size = engine.model_size_bytes();
        println!("  Model size: {:.1} KB", size as f64 / 1024.0);

        // Autocomplete accuracy
        let cases = autocomplete_test_cases();
        let mut top1_hits = 0;
        let mut top5_hits = 0;
        let mut total_phi = 0.0;
        let mut total_max_uniform = 0.0;

        for (prompt, expected) in &cases {
            let preds = engine.autocomplete(prompt, 5);
            if !preds.is_empty() {
                total_phi += preds[0].phi;
                let max_p = preds[0].confidence;
                let uniform = 1.0 / engine.vocab_size() as f64;
                total_max_uniform += max_p / uniform;

                if expected.contains(&preds[0].token.as_str()) {
                    top1_hits += 1;
                }
                if preds.iter().any(|p| expected.contains(&p.token.as_str())) {
                    top5_hits += 1;
                }
            }
        }

        let n = cases.len() as f64;
        println!("  Autocomplete Top-1: {}/{} ({:.1}%)", top1_hits, cases.len(), top1_hits as f64 / n * 100.0);
        println!("  Autocomplete Top-5: {}/{} ({:.1}%)", top5_hits, cases.len(), top5_hits as f64 / n * 100.0);
        println!("  Avg Φ: {:.4}", total_phi / n);
        println!("  Avg max/uniform: {:.2}×", total_max_uniform / n);

        // Classification
        let (labeled, labels) = classification_data();
        engine.train_classifier(&labeled, labels);
        let test_cases = classification_test_cases();
        let mut clf_correct = 0;
        for (text, expected) in &test_cases {
            if let Some(result) = engine.classify(text) {
                if result.class_label == *expected {
                    clf_correct += 1;
                }
            }
        }
        println!("  Classification: {}/{} ({:.1}%)", clf_correct, test_cases.len(),
            clf_correct as f64 / test_cases.len() as f64 * 100.0);

        // Anomaly detection
        let normal = &corpus[..corpus.len()/2];
        engine.train_anomaly_detector(normal);
        let anomalous = vec![
            "quantum entanglement photon laser beam optics",
            "blockchain cryptocurrency mining hash rate",
            "neural network gradient descent backpropagation",
        ];
        let mut normal_scores = Vec::new();
        let mut anomaly_scores = Vec::new();
        for doc in normal {
            if let Some(score) = engine.anomaly_score(doc) {
                normal_scores.push(score.score);
            }
        }
        for doc in &anomalous {
            if let Some(score) = engine.anomaly_score(doc) {
                anomaly_scores.push(score.score);
            }
        }
        let avg_normal: f64 = normal_scores.iter().sum::<f64>() / normal_scores.len().max(1) as f64;
        let avg_anomaly: f64 = anomaly_scores.iter().sum::<f64>() / anomaly_scores.len().max(1) as f64;
        println!("  Anomaly normal avg: {:.3}, anomaly avg: {:.3}, gap: {:.3}",
            avg_normal, avg_anomaly, avg_anomaly - avg_normal);

        // Serialization roundtrip
        let bytes = engine.to_bytes();
        let restored = WaveTextEngine::from_bytes(&bytes).unwrap();
        let preds_orig = engine.autocomplete("the cat", 3);
        let preds_rest = restored.autocomplete("the cat", 3);
        let match_ok = preds_orig.len() == preds_rest.len()
            && preds_orig.iter().zip(preds_rest.iter())
                .all(|(a, b)| a.token == b.token);
        println!("  Serialization roundtrip: {}", if match_ok { "OK" } else { "MISMATCH" });
    }

    println!("\n{}", "=".repeat(70));
}
