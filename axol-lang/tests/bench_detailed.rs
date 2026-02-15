//! 상세 실측 벤치마크: 실제 사용 시나리오 품질 진단.
//!
//! bench_accuracy의 수치가 "실제로 쓸 수 있는 수준"인지 검증.
//! Run: cargo test --release --test bench_detailed -- --show-output

use axol::text::engine::*;
use axol::text::data::medium_corpus;
use axol::text::generator::{compute_omega, compute_phi};

// ---------------------------------------------------------------------------
// 1. 자동완성: "the" 편향 제거한 실질 정확도
// ---------------------------------------------------------------------------

#[test]
fn bench_autocomplete_real() {
    println!("\n{}", "=".repeat(70));
    println!("  자동완성 실질 정확도 (\"the\" 편향 분리)");
    println!("{}", "=".repeat(70));

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

    // 정답이 "the"인 경우 vs 아닌 경우 분리
    let cases_the: Vec<(&str, Vec<&str>)> = vec![
        ("the cat sat on", vec!["the"]),
        ("the dog ate", vec!["the"]),
        ("the rain fell on", vec!["the"]),
        ("the boy went to", vec!["the"]),
        ("the bird flew over", vec!["the"]),
        ("the man walked to", vec!["the"]),
    ];

    let cases_content: Vec<(&str, Vec<&str>)> = vec![
        ("the cat", vec!["sat", "ate", "ran", "chased"]),
        ("the dog", vec!["sat", "ate", "ran", "chased"]),
        ("the boy", vec!["went", "ran", "kicked", "walked"]),
        ("the girl", vec!["read", "drew", "went", "played"]),
        ("the cat sat on the", vec!["mat", "bed", "floor"]),
        ("the dog sat on the", vec!["log", "mat", "floor"]),
        ("the boy went to", vec!["school", "the"]),
        ("the sun shone", vec!["brightly", "in"]),
        ("the rain fell on the", vec!["roof", "ground"]),
        ("the wind blew through", vec!["the", "trees"]),
    ];

    println!("\n--- \"the\" 예측 (기능어) ---");
    let mut the_hits = 0;
    for (prompt, expected) in &cases_the {
        let preds = engine.autocomplete(prompt, 5);
        let top1 = preds.first().map(|p| p.token.as_str()).unwrap_or("?");
        let hit = expected.contains(&top1);
        if hit { the_hits += 1; }
        let conf = preds.first().map(|p| p.confidence).unwrap_or(0.0);
        let omega = preds.first().map(|p| p.omega).unwrap_or(0.0);
        println!("  {:<30} → {:<10} {:<5} conf={:.3} Ω={:.4}",
            prompt, top1, if hit {"O"} else {"X"}, conf, omega);
    }
    println!("  정확도: {}/{} ({:.0}%)", the_hits, cases_the.len(),
        the_hits as f64 / cases_the.len() as f64 * 100.0);

    println!("\n--- 내용어 예측 (핵심) ---");
    let mut content_top1 = 0;
    let mut content_top5 = 0;
    for (prompt, expected) in &cases_content {
        let preds = engine.autocomplete(prompt, 5);
        let top1 = preds.first().map(|p| p.token.as_str()).unwrap_or("?");
        let top5: Vec<&str> = preds.iter().map(|p| p.token.as_str()).collect();
        let hit1 = expected.contains(&top1);
        let hit5 = top5.iter().any(|t| expected.contains(t));
        if hit1 { content_top1 += 1; }
        if hit5 { content_top5 += 1; }
        let conf = preds.first().map(|p| p.confidence).unwrap_or(0.0);
        let omega = preds.first().map(|p| p.omega).unwrap_or(0.0);
        println!("  {:<30} → {:<10} top1={:<3} top5={:<3} conf={:.3} Ω={:.4}  {:?}",
            prompt, top1,
            if hit1 {"O"} else {"X"},
            if hit5 {"O"} else {"X"},
            conf, omega,
            &top5[..top5.len().min(5)]);
    }
    println!("  Top-1: {}/{} ({:.0}%)", content_top1, cases_content.len(),
        content_top1 as f64 / cases_content.len() as f64 * 100.0);
    println!("  Top-5: {}/{} ({:.0}%)", content_top5, cases_content.len(),
        content_top5 as f64 / cases_content.len() as f64 * 100.0);

    println!("\n--- 종합 ---");
    let total = cases_the.len() + cases_content.len();
    let total_top1 = the_hits + content_top1;
    println!("  전체 Top-1: {}/{} ({:.0}%)", total_top1, total,
        total_top1 as f64 / total as f64 * 100.0);
}

// ---------------------------------------------------------------------------
// 2. 생성 품질: 반복률, 고유 토큰 비율, Ω 분포
// ---------------------------------------------------------------------------

#[test]
fn bench_generation_quality() {
    println!("\n{}", "=".repeat(70));
    println!("  문장 생성 품질 진단");
    println!("{}", "=".repeat(70));

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

    let prompts = vec![
        "the cat",
        "the dog ate",
        "the boy went to",
        "the sun shone",
        "the bird flew over",
    ];

    println!("\n{:<20} {:<5} {:<5} {:<6} {:<8} {:<8} {}",
        "프롬프트", "len", "uniq", "uniq%", "avg_Ω", "avg_Φ", "생성 텍스트");

    let mut total_tokens = 0;
    let mut total_unique = 0;
    let mut total_omega = 0.0;
    let mut total_phi = 0.0;
    let mut total_preds = 0;

    for prompt in &prompts {
        let result = engine.generate(prompt, 20);
        let gen = &result.generated;
        let n = gen.len();

        // 고유 토큰 수
        let mut unique: Vec<&String> = gen.iter().collect();
        unique.sort();
        unique.dedup();
        let n_unique = unique.len();

        let unique_pct = if n > 0 { n_unique as f64 / n as f64 * 100.0 } else { 0.0 };

        let avg_omega = if !result.predictions.is_empty() {
            result.predictions.iter().map(|p| p.omega).sum::<f64>() / result.predictions.len() as f64
        } else { 0.0 };

        let avg_phi = if !result.predictions.is_empty() {
            result.predictions.iter().map(|p| p.phi).sum::<f64>() / result.predictions.len() as f64
        } else { 0.0 };

        total_tokens += n;
        total_unique += n_unique;
        total_omega += avg_omega * n as f64;
        total_phi += avg_phi * n as f64;
        total_preds += n;

        let display = if result.full_text.len() > 50 {
            format!("{}...", &result.full_text[..47])
        } else {
            result.full_text.clone()
        };

        println!("{:<20} {:<5} {:<5} {:<6.1} {:<8.4} {:<8.4} {}",
            prompt, n, n_unique, unique_pct, avg_omega, avg_phi, display);

        // 토큰별 상세
        if result.predictions.len() <= 5 || true {
            for (i, pred) in result.predictions.iter().take(5).enumerate() {
                println!("    [{:2}] {:<10} Ω={:.4} Φ={:.4} conf={:.4}  top3: {:?}",
                    i, pred.token, pred.omega, pred.phi, pred.confidence,
                    &pred.top_k[..pred.top_k.len().min(3)]);
            }
            if result.predictions.len() > 5 {
                println!("    ... ({} more)", result.predictions.len() - 5);
            }
        }
    }

    let overall_unique_pct = if total_tokens > 0 {
        total_unique as f64 / total_tokens as f64 * 100.0
    } else { 0.0 };
    let overall_omega = if total_preds > 0 { total_omega / total_preds as f64 } else { 0.0 };
    let overall_phi = if total_preds > 0 { total_phi / total_preds as f64 } else { 0.0 };

    println!("\n--- 종합 ---");
    println!("  총 생성 토큰: {}", total_tokens);
    println!("  고유 토큰 비율: {:.1}% ({}/{})", overall_unique_pct, total_unique, total_tokens);
    println!("  평균 Ω: {:.6}", overall_omega);
    println!("  평균 Φ: {:.4}", overall_phi);
    println!("\n  ※ Ω ≈ 0 = readout이 근사 균일 분포 → 생성 품질 저하");
    println!("  ※ 고유 비율 < 30% = 심각한 반복");
}

// ---------------------------------------------------------------------------
// 3. readout 출력 분포 진단
// ---------------------------------------------------------------------------

#[test]
fn bench_readout_distribution() {
    println!("\n{}", "=".repeat(70));
    println!("  Readout 출력 분포 진단");
    println!("{}", "=".repeat(70));

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

    let prompts = vec![
        "the cat sat on the",
        "the dog ate the",
        "the boy went to",
        "the cat",
        "the sun shone in the",
    ];

    for prompt in &prompts {
        let preds = engine.autocomplete(prompt, 10);
        if preds.is_empty() { continue; }

        let probs = &preds[0].probabilities;
        let omega = preds[0].omega;
        let phi = preds[0].phi;

        // 분포 통계
        let max_p = probs.iter().cloned().fold(0.0f64, f64::max);
        let min_p = probs.iter().cloned().fold(f64::MAX, f64::min);
        let mean_p = probs.iter().sum::<f64>() / probs.len() as f64;
        let uniform_p = 1.0 / probs.len() as f64;

        // 엔트로피
        let entropy: f64 = probs.iter()
            .filter(|&&p| p > 1e-15)
            .map(|&p| -p * p.ln())
            .sum();
        let max_entropy = (probs.len() as f64).ln();
        let norm_entropy = entropy / max_entropy;

        // top-10 확률 합
        let mut sorted: Vec<f64> = probs.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let top10_sum: f64 = sorted.iter().take(10).sum();
        let top1_p = sorted[0];

        println!("\n  \"{}\"", prompt);
        println!("    vocab_size={}, Ω={:.6}, Φ={:.4}", probs.len(), omega, phi);
        println!("    max_p={:.6}, min_p={:.8}, mean_p={:.8}, uniform={:.8}",
            max_p, min_p, mean_p, uniform_p);
        println!("    max/uniform={:.2}x, 정규 엔트로피={:.4} (1.0=완전 균일)",
            max_p / uniform_p, norm_entropy);
        println!("    top-1={:.4}%, top-10={:.2}%",
            top1_p * 100.0, top10_sum * 100.0);
        println!("    top-10: {:?}",
            preds.iter().take(10).map(|p| format!("{}({:.3}%)", p.token, p.confidence * 100.0))
                .collect::<Vec<_>>());
    }
}

// ---------------------------------------------------------------------------
// 4. 분류 상세 (confusion matrix)
// ---------------------------------------------------------------------------

#[test]
fn bench_classification_detail() {
    println!("\n{}", "=".repeat(70));
    println!("  분류 상세 진단 (확장 테스트셋)");
    println!("{}", "=".repeat(70));

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

    // 학습 데이터 (확장)
    let train: Vec<(&str, usize)> = vec![
        ("the cat sat on the mat", 0), ("the dog ate the bone", 0),
        ("the bird flew over the tree", 0), ("the cat ate the fish", 0),
        ("the dog ran in the park", 0), ("the bird sang in the tree", 0),
        ("the cat chased the mouse", 0), ("the dog chased the cat", 0),
        ("the rabbit hopped across the field", 0), ("the horse ran across the field", 0),
        ("the boy went to school", 1), ("the girl read a book", 1),
        ("the man walked to the store", 1), ("the woman cooked dinner", 1),
        ("the boy ran home", 1), ("the girl played in the yard", 1),
        ("the man wrote a letter", 1), ("the woman built a garden", 1),
        ("the teacher taught the students", 1), ("the doctor helped the sick man", 1),
        ("the sun shone brightly in the sky", 2), ("the rain fell on the roof", 2),
        ("the wind blew through trees", 2), ("the snow covered the ground", 2),
        ("the moon rose at night", 2), ("the stars shone brightly", 2),
        ("the storm came from the north", 2), ("the fog covered the city", 2),
        ("the clouds covered the sky", 2), ("the thunder roared across the sky", 2),
    ];
    let labels = vec!["animals".into(), "people".into(), "nature".into()];
    engine.train_classifier(&train, labels.clone());

    // 테스트
    let test: Vec<(&str, usize)> = vec![
        ("the cat ran fast", 0),
        ("the dog played fetch", 0),
        ("the bird nested in the tree", 0),
        ("the fish swam in the water", 0),
        ("the boy ate lunch", 1),
        ("the girl went home", 1),
        ("the man sat down", 1),
        ("the woman read a newspaper", 1),
        ("the rain poured down", 2),
        ("the wind howled loudly", 2),
        ("the sun set slowly", 2),
        ("the snow fell gently", 2),
    ];

    let n_classes = labels.len();
    let mut confusion = vec![vec![0usize; n_classes]; n_classes];  // [actual][predicted]

    println!("\n{:<35} {:<10} {:<10} {:<6} {:<8}",
        "텍스트", "예측", "정답", "맞음?", "conf");

    for (text, expected) in &test {
        if let Some(result) = engine.classify(text) {
            let correct = result.class_id == *expected;
            confusion[*expected][result.class_id] += 1;

            println!("{:<35} {:<10} {:<10} {:<6} {:<8.3}",
                text, result.class_label, labels[*expected],
                if correct {"O"} else {"X"}, result.confidence);
        }
    }

    // Confusion matrix
    println!("\n  Confusion Matrix (행=정답, 열=예측):");
    print!("  {:<12}", "");
    for l in &labels { print!("{:<10}", l); }
    println!();
    for (i, l) in labels.iter().enumerate() {
        print!("  {:<12}", l);
        for j in 0..n_classes {
            print!("{:<10}", confusion[i][j]);
        }
        println!();
    }

    // Per-class F1
    let mut macro_f1 = 0.0;
    println!();
    for c in 0..n_classes {
        let tp = confusion[c][c] as f64;
        let fp: f64 = (0..n_classes).filter(|&i| i != c).map(|i| confusion[i][c] as f64).sum();
        let fn_: f64 = (0..n_classes).filter(|&j| j != c).map(|j| confusion[c][j] as f64).sum();
        let p = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        let r = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
        let f1 = if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 };
        macro_f1 += f1;
        println!("  {}: P={:.2} R={:.2} F1={:.2}", labels[c], p, r, f1);
    }
    macro_f1 /= n_classes as f64;

    let total_correct: usize = (0..n_classes).map(|c| confusion[c][c]).sum();
    let total: usize = test.len();
    println!("\n  정확도: {}/{} ({:.1}%)", total_correct, total,
        total_correct as f64 / total as f64 * 100.0);
    println!("  Macro F1: {:.4}", macro_f1);
}

// ---------------------------------------------------------------------------
// 5. 이상탐지 상세 (score 분포 + threshold 분석)
// ---------------------------------------------------------------------------

#[test]
fn bench_anomaly_detail() {
    println!("\n{}", "=".repeat(70));
    println!("  이상탐지 상세 진단");
    println!("{}", "=".repeat(70));

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
    let normal_docs: Vec<&str> = corpus[..30].iter().copied().collect();
    engine.train_anomaly_detector(&normal_docs);

    let normal_tests = vec![
        "the cat sat on the mat",
        "the dog ate the bone",
        "the boy went to school",
        "the bird flew over the tree",
        "the girl read a book",
        "the man walked to the store",
        "the rain fell on the roof",
        "the sun shone in the sky",
    ];

    let anomaly_tests = vec![
        "quantum entanglement drives blockchain synergy",
        "the recursive algorithm optimizes neural gradients",
        "cryptocurrency mining consumes excessive kilowatts",
        "photosynthesis converts carbon dioxide into glucose",
        "microprocessor architectures leverage parallel pipelines",
        "deep learning transformers attend to all positions",
        "the API endpoint returns a JSON payload",
        "kubernetes orchestrates containerized microservices",
    ];

    println!("\n{:<55} {:<8} {:<8}", "텍스트", "점수", "판정");

    let mut normal_scores = Vec::new();
    println!("\n  [정상]");
    for text in &normal_tests {
        if let Some(r) = engine.anomaly_score(text) {
            normal_scores.push(r.score);
            println!("  {:<55} {:<8.4} {}",
                text, r.score, if r.is_anomalous {"[!] 이상"} else {"[ ] 정상"});
        }
    }

    let mut anomaly_scores = Vec::new();
    println!("\n  [이상]");
    for text in &anomaly_tests {
        if let Some(r) = engine.anomaly_score(text) {
            anomaly_scores.push(r.score);
            println!("  {:<55} {:<8.4} {}",
                text, r.score, if r.is_anomalous {"[!] 이상"} else {"[ ] 정상"});
        }
    }

    // 통계
    let n_mean = normal_scores.iter().sum::<f64>() / normal_scores.len() as f64;
    let a_mean = anomaly_scores.iter().sum::<f64>() / anomaly_scores.len() as f64;
    let n_max = normal_scores.iter().cloned().fold(0.0f64, f64::max);
    let a_min = anomaly_scores.iter().cloned().fold(f64::MAX, f64::min);

    // AUROC
    let mut correct_pairs = 0;
    let mut total_pairs = 0;
    for &ns in &normal_scores {
        for &as_ in &anomaly_scores {
            total_pairs += 1;
            if as_ > ns { correct_pairs += 1; }
            else if (as_ - ns).abs() < 1e-10 { correct_pairs += 1; } // tie = 0.5 simplified
        }
    }
    let auroc = correct_pairs as f64 / total_pairs as f64;

    // threshold별 정밀도/재현율
    println!("\n  --- threshold 분석 ---");
    for threshold in &[0.5, 0.7, 1.0, 1.5, 2.0] {
        let tp = anomaly_scores.iter().filter(|&&s| s > *threshold).count();
        let fp = normal_scores.iter().filter(|&&s| s > *threshold).count();
        let fn_ = anomaly_scores.iter().filter(|&&s| s <= *threshold).count();
        let p = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
        let r = if tp + fn_ > 0 { tp as f64 / (tp + fn_) as f64 } else { 0.0 };
        let f1 = if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 };
        println!("    θ={:.1}: P={:.2} R={:.2} F1={:.2} (TP={} FP={} FN={})",
            threshold, p, r, f1, tp, fp, fn_);
    }

    println!("\n  --- 종합 ---");
    println!("  정상 평균: {:.4}, 최대: {:.4}", n_mean, n_max);
    println!("  이상 평균: {:.4}, 최소: {:.4}", a_mean, a_min);
    println!("  분리 갭: {:.4}", a_min - n_max);
    println!("  AUROC: {:.4}", auroc);
}
