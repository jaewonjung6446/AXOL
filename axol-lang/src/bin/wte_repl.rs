//! WTE Interactive REPL — 실전 테스트용 대화형 인터페이스.
//!
//! Run: cargo run --release --bin wte_repl
//!
//! Commands:
//!   > text here       — autocomplete (top-5 예측)
//!   gen> text here    — 문장 생성 (최대 15토큰)
//!   cls> text here    — 분류
//!   ano> text here    — 이상 탐지 점수
//!   info              — 엔진 정보
//!   quit / exit       — 종료

use std::io::{self, Write, BufRead};

use axol::text::engine::*;
use axol::text::data::medium_corpus;

fn main() {
    eprintln!("=== AXOL WTE REPL ===");
    eprintln!("Loading corpus and training...");

    let config = EngineConfig {
        dim: 64,
        num_merges: 200,
        max_vocab: 2000,
        seed: 42,
        ..EngineConfig::default()
    };

    let corpus = medium_corpus();
    let mut engine = WaveTextEngine::from_corpus_with_config(&corpus, &config);

    // Train classifier
    let (labeled, labels) = classification_data();
    engine.train_classifier(&labeled, labels);

    // Train anomaly detector
    let normal: Vec<&str> = corpus.iter().copied().take(100).collect();
    engine.train_anomaly_detector(&normal);

    eprintln!("Ready! vocab={}, features={}, model={:.1}KB",
        engine.vocab_size(), engine.feature_dim(),
        engine.model_size_bytes() as f64 / 1024.0);
    eprintln!();
    eprintln!("Commands:");
    eprintln!("  > text        autocomplete");
    eprintln!("  gen> text     generate");
    eprintln!("  cls> text     classify");
    eprintln!("  ano> text     anomaly score");
    eprintln!("  info          engine info");
    eprintln!("  quit          exit");
    eprintln!();

    let stdin = io::stdin();
    loop {
        eprint!("wte> ");
        io::stderr().flush().unwrap();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap() == 0 {
            break; // EOF
        }
        let line = line.trim();
        if line.is_empty() { continue; }

        match line {
            "quit" | "exit" => break,
            "info" => {
                println!("  Vocab size:    {}", engine.vocab_size());
                println!("  Feature dim:   {}", engine.feature_dim());
                println!("  Model size:    {:.1} KB", engine.model_size_bytes() as f64 / 1024.0);
                println!("  Config dim:    {}", engine.config.dim);
                println!("  Num nodes:     {}", engine.config.num_nodes);
                println!("  Hidden dim:    {}", engine.config.hidden_dim);
            }
            _ if line.starts_with("gen>") => {
                let prompt = line[4..].trim();
                if prompt.is_empty() {
                    eprintln!("  (empty prompt)");
                    continue;
                }
                let result = engine.generate(prompt, 15);
                println!("  prompt:    \"{}\"", result.prompt);
                println!("  generated: \"{}\"", result.generated.join(" "));
                println!("  full:      \"{}\"", result.full_text);
                if !result.predictions.is_empty() {
                    println!("  tokens:    {}", result.predictions.len());
                    let avg_phi: f64 = result.predictions.iter()
                        .map(|p| p.phi).sum::<f64>() / result.predictions.len() as f64;
                    println!("  avg Φ:     {:.4}", avg_phi);
                }
            }
            _ if line.starts_with("cls>") => {
                let text = line[4..].trim();
                if text.is_empty() {
                    eprintln!("  (empty text)");
                    continue;
                }
                match engine.classify(text) {
                    Some(r) => {
                        println!("  class: {} (conf={:.3}, Φ={:.4})", r.class_label, r.confidence, r.phi);
                        for (label, prob) in &r.class_probs {
                            println!("    {}: {:.3}", label, prob);
                        }
                    }
                    None => println!("  (classifier not trained)"),
                }
            }
            _ if line.starts_with("ano>") => {
                let text = line[4..].trim();
                if text.is_empty() {
                    eprintln!("  (empty text)");
                    continue;
                }
                match engine.anomaly_score(text) {
                    Some(s) => {
                        println!("  score: {:.3} {}", s.score,
                            if s.is_anomalous { "(ANOMALY)" } else { "(normal)" });
                        println!("  omega_z={:.3}, phi_z={:.3}, coherence={:.3}, energy={:.3}",
                            s.omega_z, s.phi_z, s.coherence_deviation, s.energy_deviation);
                    }
                    None => println!("  (anomaly detector not trained)"),
                }
            }
            _ => {
                // Default: autocomplete
                let prompt = if line.starts_with(">") {
                    line[1..].trim()
                } else {
                    line
                };
                if prompt.is_empty() {
                    eprintln!("  (empty prompt)");
                    continue;
                }
                let preds = engine.autocomplete(prompt, 5);
                if preds.is_empty() {
                    println!("  (no predictions)");
                } else {
                    println!("  \"{}\" →", prompt);
                    for (i, p) in preds.iter().enumerate() {
                        println!("    {}. \"{}\" (conf={:.3}, Φ={:.4})", i + 1, p.token, p.confidence, p.phi);
                    }
                }
            }
        }
    }

    eprintln!("Bye!");
}

fn classification_data() -> (Vec<(&'static str, usize)>, Vec<String>) {
    let labeled = vec![
        ("the cat sat on the mat", 0),
        ("the dog ate the bone", 0),
        ("the bird flew over the tree", 0),
        ("the cat ate the fish", 0),
        ("the dog ran in the park", 0),
        ("the bird sang in the tree", 0),
        ("the fish swam in the water", 0),
        ("the rabbit hopped across the field", 0),
        ("the boy went to school", 1),
        ("the girl read a book", 1),
        ("the man walked to the store", 1),
        ("the woman cooked dinner", 1),
        ("the boy ran home", 1),
        ("the girl played in the yard", 1),
        ("the teacher taught the students", 1),
        ("the doctor helped the sick man", 1),
        ("the sun shone in the sky", 2),
        ("the rain fell on the roof", 2),
        ("the wind blew through trees", 2),
        ("the snow covered the ground", 2),
        ("the moon rose at night", 2),
        ("the stars shone brightly", 2),
        ("the river flowed to the sea", 2),
        ("the flowers bloomed in spring", 2),
    ];
    let labels = vec!["animals".into(), "people".into(), "nature".into()];
    (labeled, labels)
}
