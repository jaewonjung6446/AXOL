//! Feature vector similarity measurement.
//! Run with: cargo test --release --test measure_features -- --show-output --nocapture

use axol::text::engine::*;
use axol::text::data::medium_corpus;

fn cosine_sim(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na < 1e-15 || nb < 1e-15 { return 0.0; }
    dot / (na * nb)
}

#[test]
fn measure_feature_similarity() {
    let corpus = medium_corpus();
    let config = EngineConfig { dim: 64, ..EngineConfig::default() };
    let corpus_refs: Vec<&str> = corpus.iter().map(|s| *s).collect();
    let engine = WaveTextEngine::from_corpus_with_config(&corpus_refs, &config);

    let prompts = vec![
        "the cat sat on",
        "the dog ate the",
        "the boy went to",
        "the bird flew over",
        "the sun shone in",
        "the rain fell on",
        "the girl read a",
        "the man walked to",
        "quantum entanglement drives",
        "the cat",
        "the dog",
    ];

    let mut features: Vec<(String, Vec<f64>)> = Vec::new();
    for p in &prompts {
        let state = engine.process_text(p);
        let feat = state.to_feature_vector();
        features.push((p.to_string(), feat));
    }

    let dim = features[0].1.len();
    println!("\n=== Feature Vector Analysis ===");
    println!("Feature dim: {}", dim);
    println!("Num prompts: {}", prompts.len());

    // Feature norms
    println!("\n--- Feature Norms ---");
    for (name, feat) in &features {
        let norm: f64 = feat.iter().map(|x| x * x).sum::<f64>().sqrt();
        let max: f64 = feat.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min: f64 = feat.iter().cloned().fold(f64::INFINITY, f64::min);
        println!("  {:35} norm={:.4}  max={:.4}  min={:.4}", name, norm, max, min);
    }

    // Cosine similarity matrix
    println!("\n--- Cosine Similarity Matrix ---");
    let short_names: Vec<String> = prompts.iter().map(|p| {
        let words: Vec<&str> = p.split_whitespace().collect();
        if words.len() >= 2 { format!("{} {}", words[0], words[1]) }
        else { words[0].to_string() }
    }).collect();

    print!("{:12}", "");
    for name in &short_names { print!("{:>10}", name); }
    println!();

    let mut sims = Vec::new();
    let mut normal_sims = Vec::new();
    let mut anomaly_sims = Vec::new();
    for (i, (_, feat_a)) in features.iter().enumerate() {
        print!("{:12}", short_names[i]);
        for (j, (_, feat_b)) in features.iter().enumerate() {
            let sim = cosine_sim(feat_a, feat_b);
            print!("{:10.4}", sim);
            if i != j {
                sims.push(sim);
                if i < 8 && j < 8 { normal_sims.push(sim); }
                if (i < 8 && j == 8) || (i == 8 && j < 8) { anomaly_sims.push(sim); }
            }
        }
        println!();
    }

    let avg_sim: f64 = sims.iter().sum::<f64>() / sims.len() as f64;
    let avg_normal: f64 = if normal_sims.is_empty() { 0.0 } else { normal_sims.iter().sum::<f64>() / normal_sims.len() as f64 };
    let avg_anomaly: f64 = if anomaly_sims.is_empty() { 0.0 } else { anomaly_sims.iter().sum::<f64>() / anomaly_sims.len() as f64 };
    let min_sim: f64 = sims.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_sim: f64 = sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("\n--- Summary ---");
    println!("  Avg pairwise cosine (all):       {:.4}", avg_sim);
    println!("  Avg cosine (normal vs normal):    {:.4}", avg_normal);
    println!("  Avg cosine (normal vs anomaly):   {:.4}", avg_anomaly);
    println!("  Min pairwise cosine:              {:.4}", min_sim);
    println!("  Max pairwise cosine (non-self):   {:.4}", max_sim);

    // Per-channel variance
    println!("\n--- Per-Channel Variance (top 20) ---");
    let mut variances: Vec<(usize, f64)> = Vec::new();
    for ch in 0..dim {
        let vals: Vec<f64> = features.iter().map(|f| f.1[ch]).collect();
        let mean: f64 = vals.iter().sum::<f64>() / vals.len() as f64;
        let var: f64 = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
        variances.push((ch, var));
    }
    variances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (ch, var) in variances.iter().take(20) {
        println!("  ch[{:3}] var={:.8}", ch, var);
    }

    let total_var: f64 = variances.iter().map(|v| v.1).sum();
    let top20_var: f64 = variances.iter().take(20).map(|v| v.1).sum();
    let zero_var: usize = variances.iter().filter(|v| v.1 < 1e-10).count();
    println!("\n  Total variance:  {:.6}", total_var);
    println!("  Top-20 hold:     {:.1}% of total", 100.0 * top20_var / total_var.max(1e-15));
    println!("  Near-zero var:   {}/{} channels", zero_var, dim);
}
