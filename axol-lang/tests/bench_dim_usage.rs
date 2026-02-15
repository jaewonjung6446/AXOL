//! 차원 활용도 진단 — SPS 임베딩이 64차원을 얼마나 쓰는가?
//!
//! 가설: 코퍼스가 작으면 실제 유효 차원이 64보다 훨씬 낮아서
//!       Born rule 확률이 평탄해진다.
//!
//! Run: cargo test --release --test bench_dim_usage -- --show-output

use axol::text::engine::*;
use axol::text::data::medium_corpus;

#[test]
fn diagnose_dim_usage() {
    println!("\n{}", "═".repeat(70));
    println!("  차원 활용도 진단");
    println!("{}", "═".repeat(70));

    let corpus = medium_corpus();
    let config = EngineConfig {
        dim: 64, num_merges: 200, max_vocab: 2000, seed: 42,
        ..EngineConfig::default()
    };
    let engine = WaveTextEngine::from_corpus_with_config(&corpus, &config);

    // 1. SPS 임베딩의 차원별 분산
    let vocab_size = engine.sps.vocab_size;
    let dim = engine.sps.dim;

    println!("\n  [1] SPS 임베딩 차원별 분산 (vocab={}, dim={})", vocab_size, dim);

    let mut dim_variances = vec![0.0f64; dim];
    let mut dim_means = vec![0.0f64; dim];

    // 평균
    for emb in &engine.sps.embeddings {
        for (j, &v) in emb.iter().enumerate() {
            if j < dim { dim_means[j] += v; }
        }
    }
    for v in dim_means.iter_mut() { *v /= vocab_size as f64; }

    // 분산
    for emb in &engine.sps.embeddings {
        for (j, &v) in emb.iter().enumerate() {
            if j < dim {
                dim_variances[j] += (v - dim_means[j]).powi(2);
            }
        }
    }
    for v in dim_variances.iter_mut() { *v /= vocab_size as f64; }

    // 정렬 (내림차순)
    let mut sorted_vars: Vec<(usize, f64)> = dim_variances.iter()
        .copied().enumerate().collect();
    sorted_vars.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let total_var: f64 = dim_variances.iter().sum();
    let mut cumulative = 0.0;
    let mut effective_dims = 0;

    println!("    차원  |  분산      |  누적 비율");
    println!("    ──────────────────────────────");
    for (i, (dim_idx, var)) in sorted_vars.iter().enumerate() {
        cumulative += var;
        let pct = cumulative / total_var * 100.0;
        if i < 10 || (i < 20 && *var > 0.001) {
            println!("    {:>3}   |  {:.6}  |  {:.1}%", dim_idx, var, pct);
        }
        if pct < 95.0 {
            effective_dims = i + 1;
        }
    }

    let dims_90 = sorted_vars.iter()
        .scan(0.0, |acc, (_, v)| { *acc += v; Some(*acc) })
        .position(|c| c / total_var >= 0.90)
        .unwrap_or(dim) + 1;
    let dims_95 = sorted_vars.iter()
        .scan(0.0, |acc, (_, v)| { *acc += v; Some(*acc) })
        .position(|c| c / total_var >= 0.95)
        .unwrap_or(dim) + 1;
    let dims_99 = sorted_vars.iter()
        .scan(0.0, |acc, (_, v)| { *acc += v; Some(*acc) })
        .position(|c| c / total_var >= 0.99)
        .unwrap_or(dim) + 1;

    println!("\n  ▶ 총 분산: {:.6}", total_var);
    println!("  ▶ 분산 90% 도달 차원: {}/{}", dims_90, dim);
    println!("  ▶ 분산 95% 도달 차원: {}/{}", dims_95, dim);
    println!("  ▶ 분산 99% 도달 차원: {}/{}", dims_99, dim);
    println!("  ▶ 최대 분산: {:.6} (차원 {})", sorted_vars[0].1, sorted_vars[0].0);
    println!("  ▶ 최소 분산: {:.6} (차원 {})", sorted_vars.last().unwrap().1, sorted_vars.last().unwrap().0);
    let ratio = sorted_vars[0].1 / sorted_vars.last().unwrap().1.max(1e-15);
    println!("  ▶ 최대/최소 비율: {:.1}×", ratio);

    // 2. 파동 겹침 분포 — 무작위 토큰 쌍의 |⟨ψ_a|ψ_b⟩|² 분포
    println!("\n  [2] 토큰 파동 겹침 분포 |⟨ψ_a|ψ_b⟩|²");

    let mut overlaps = Vec::new();
    let sample = vocab_size.min(100);
    for i in 4..sample {  // skip special tokens
        for j in (i+1)..sample {
            let wa = engine.sps.token_to_wave(i);
            let wb = engine.sps.token_to_wave(j);
            let d = wa.dim.min(wb.dim);
            let mut inner = num_complex::Complex64::new(0.0, 0.0);
            for k in 0..d {
                inner += wa.amplitudes.data[k].conj() * wb.amplitudes.data[k];
            }
            overlaps.push(inner.norm_sqr());
        }
    }

    overlaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = overlaps.len();
    let mean_overlap = overlaps.iter().sum::<f64>() / n as f64;
    let min_overlap = overlaps[0];
    let max_overlap = overlaps[n - 1];
    let median_overlap = overlaps[n / 2];
    let p10 = overlaps[n / 10];
    let p90 = overlaps[n * 9 / 10];

    println!("    쌍 수: {}", n);
    println!("    평균:  {:.6}", mean_overlap);
    println!("    중앙값: {:.6}", median_overlap);
    println!("    최소:  {:.6}", min_overlap);
    println!("    최대:  {:.6}", max_overlap);
    println!("    10%tile: {:.6}", p10);
    println!("    90%tile: {:.6}", p90);
    println!("    범위(90-10): {:.6}", p90 - p10);

    let ideal_overlap = 1.0 / dim as f64;
    println!("\n  ▶ 이상적 겹침 (직교 시): {:.6}", ideal_overlap);
    println!("  ▶ 실제 평균 겹침:        {:.6}", mean_overlap);
    println!("  ▶ 겹침/이상 비율:        {:.1}×", mean_overlap / ideal_overlap);
    if mean_overlap > ideal_overlap * 3.0 {
        println!("  ▶ 진단: 파동이 너무 비슷함 — 유효 차원이 낮음");
    } else {
        println!("  ▶ 진단: 파동 다양성 양호");
    }

    // 3. dim을 줄이면 어떻게 되는지 비교
    println!("\n  [3] dim 변화에 따른 의미 유사도 분리");

    for test_dim in [8, 16, 32, 64] {
        let cfg = EngineConfig {
            dim: test_dim, num_merges: 200, max_vocab: 2000, seed: 42,
            ..EngineConfig::default()
        };
        let eng = WaveTextEngine::from_corpus_with_config(&corpus, &cfg);

        // 같은 도메인
        let s1 = wave_sim(&eng, "the cat sat on the mat", "the dog sat on the log");
        let s2 = wave_sim(&eng, "the boy went to school", "the girl went to school");
        let same_avg = (s1 + s2) / 2.0;

        // 다른 도메인
        let d1 = wave_sim(&eng, "the cat sat on the mat", "the sun shone in the sky");
        let d2 = wave_sim(&eng, "the dog ate the bone", "the boy went to school");
        let diff_avg = (d1 + d2) / 2.0;

        let gap = same_avg - diff_avg;
        println!("    dim={:>2}: 같은={:.3} 다른={:.3} gap={:.3}", test_dim, same_avg, diff_avg, gap);
    }

    // 4. 단어 공명 확률 분포의 엔트로피
    println!("\n  [4] 단어 공명 확률 분포 엔트로피");

    let test_words: Vec<&str> = vec![
        "cat", "dog", "bird", "fish", "boy", "girl", "man", "woman",
        "sun", "rain", "wind", "snow", "tree", "school", "book", "house",
        "sat", "ate", "flew", "went", "read", "walked", "fell", "ran",
    ];

    for context in ["the cat sat on the mat", "the boy went to school", "the sun shone in the sky"] {
        let ranked = eng_word_resonances(&engine, context, &test_words);
        let entropy = -ranked.iter()
            .map(|(_, p)| if *p > 1e-15 { p * p.ln() } else { 0.0 })
            .sum::<f64>();
        let max_entropy = (test_words.len() as f64).ln();
        let norm_entropy = entropy / max_entropy;

        println!("    \"{}\"", context);
        println!("      엔트로피: {:.3} / {:.3} (정규화: {:.3})",
            entropy, max_entropy, norm_entropy);
        let top3: Vec<String> = ranked.iter().take(3)
            .map(|(w, p)| format!("{}({:.3})", w, p))
            .collect();
        println!("      상위3: {}", top3.join(", "));
        if norm_entropy > 0.95 {
            println!("      → 거의 균등 분포 (정보 없음)");
        } else if norm_entropy > 0.85 {
            println!("      → 약한 선호도 (정보 부족)");
        } else {
            println!("      → 명확한 선호도 (정보 있음)");
        }
    }

    println!("\n{}", "═".repeat(70));
}

fn wave_sim(engine: &WaveTextEngine, a: &str, b: &str) -> f64 {
    let sa = engine.process_text(a);
    let sb = engine.process_text(b);
    let dim = sa.merged.dim.min(sb.merged.dim);
    let mut dot = num_complex::Complex64::new(0.0, 0.0);
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..dim {
        let ca = sa.merged.amplitudes.data[i];
        let cb = sb.merged.amplitudes.data[i];
        dot += ca.conj() * cb;
        na += ca.norm_sqr();
        nb += cb.norm_sqr();
    }
    if na > 1e-10 && nb > 1e-10 { dot.norm() / (na.sqrt() * nb.sqrt()) } else { 0.0 }
}

fn eng_word_resonances<'a>(engine: &WaveTextEngine, context: &str, words: &[&'a str]) -> Vec<(&'a str, f64)> {
    let ids = engine.bpe.encode(context);
    let token_waves = engine.sps.tokens_to_waves(&ids);
    let mut reservoir = engine.reservoir.clone();
    let state = reservoir.process_sequence(&token_waves);

    let mut results: Vec<(&str, f64)> = words.iter()
        .map(|&w| (w, engine.word_resonance(&state, w)))
        .collect();

    let sum: f64 = results.iter().map(|(_, r)| r).sum();
    if sum > 1e-10 {
        for (_, r) in results.iter_mut() { *r /= sum; }
    }
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    results
}
