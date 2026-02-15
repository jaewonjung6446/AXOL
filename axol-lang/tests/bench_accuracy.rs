//! AXOL WTE 정확도 + 속도 종합 실측 테스트.
//!
//! 자동완성 top-K 정확도, 문장 생성 품질, 분류 F1, 이상탐지 분리도,
//! 그리고 각 작업의 실측 속도를 측정한다.
//! Run with: cargo test --release --test bench_accuracy -- --show-output

use std::time::Instant;

use axol::text::engine::*;
use axol::text::data::medium_corpus;

// ---------------------------------------------------------------------------
// 테스트 데이터
// ---------------------------------------------------------------------------

/// 자동완성 정확도 측정용: (프롬프트, 기대 정답 목록)
fn autocomplete_test_cases() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        ("the cat sat on", vec!["the", "mat", "a"]),
        ("the dog ate", vec!["the", "a", "his"]),
        ("the boy went to", vec!["the", "school", "a"]),
        ("the bird flew over", vec!["the", "tree", "a"]),
        ("the girl read a", vec!["book", "the", "a"]),
        ("the man walked to", vec!["the", "store", "a"]),
        ("the sun shone in", vec!["the", "sky", "a"]),
        ("the rain fell on", vec!["the", "roof", "a"]),
        ("the cat", vec!["sat", "ate", "ran"]),
        ("the dog sat on", vec!["the", "mat", "a"]),
        ("the wind blew", vec!["through", "the", "over"]),
        ("the boy", vec!["went", "ran", "walked"]),
    ]
}

/// 분류 학습 데이터
fn classification_data() -> (Vec<(&'static str, usize)>, Vec<String>) {
    let labeled = vec![
        // 동물 (0)
        ("the cat sat on the mat", 0),
        ("the dog ate the bone", 0),
        ("the bird flew over the tree", 0),
        ("the cat ate the fish", 0),
        ("the dog ran in the park", 0),
        ("the bird sang in the tree", 0),
        // 사람 (1)
        ("the boy went to school", 1),
        ("the girl read a book", 1),
        ("the man walked to the store", 1),
        ("the woman cooked dinner", 1),
        ("the boy ran home", 1),
        ("the girl played in the yard", 1),
        // 자연 (2)
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

/// 분류 테스트 데이터: (텍스트, 기대 클래스)
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

// ---------------------------------------------------------------------------
// 1. 자동완성 정확도 (Top-1, Top-3, Top-5)
// ---------------------------------------------------------------------------

#[test]
fn bench_accuracy_autocomplete() {
    println!("\n{}", "=".repeat(70));
    println!("  AXOL WTE 자동완성 정확도 측정");
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
    let cases = autocomplete_test_cases();

    let mut top1_hits = 0;
    let mut top3_hits = 0;
    let mut top5_hits = 0;
    let total = cases.len();

    println!("\n{:<25} {:<12} {:<8} {:<8} {:<8} {:<20}",
        "프롬프트", "Top-1", "hit1?", "hit3?", "hit5?", "top-5 예측");

    for (prompt, expected) in &cases {
        let preds = engine.autocomplete(prompt, 5);
        let pred_tokens: Vec<&str> = preds.iter().map(|p| p.token.as_str()).collect();

        let hit1 = pred_tokens.get(0).map(|t| expected.contains(t)).unwrap_or(false);
        let hit3 = pred_tokens.iter().take(3).any(|t| expected.contains(t));
        let hit5 = pred_tokens.iter().take(5).any(|t| expected.contains(t));

        if hit1 { top1_hits += 1; }
        if hit3 { top3_hits += 1; }
        if hit5 { top5_hits += 1; }

        let mark1 = if hit1 { "O" } else { "X" };
        let mark3 = if hit3 { "O" } else { "X" };
        let mark5 = if hit5 { "O" } else { "X" };

        println!("{:<25} {:<12} {:<8} {:<8} {:<8} {:?}",
            prompt,
            pred_tokens.first().unwrap_or(&"?"),
            mark1, mark3, mark5,
            &pred_tokens[..pred_tokens.len().min(5)]);
    }

    let top1_pct = top1_hits as f64 / total as f64 * 100.0;
    let top3_pct = top3_hits as f64 / total as f64 * 100.0;
    let top5_pct = top5_hits as f64 / total as f64 * 100.0;

    println!("\n--- 결과 ---");
    println!("Top-1 정확도: {}/{} ({:.1}%)", top1_hits, total, top1_pct);
    println!("Top-3 정확도: {}/{} ({:.1}%)", top3_hits, total, top3_pct);
    println!("Top-5 정확도: {}/{} ({:.1}%)", top5_hits, total, top5_pct);
}

// ---------------------------------------------------------------------------
// 2. 문장 생성 품질 측정
// ---------------------------------------------------------------------------

#[test]
fn bench_accuracy_generation() {
    println!("\n{}", "=".repeat(70));
    println!("  AXOL WTE 문장 생성 품질 측정");
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
        "the dog",
        "the boy went",
        "the sun",
        "the bird flew",
        "the girl",
        "the man walked",
        "the rain",
    ];

    println!("\n{:<20} {:<6} {:<50} {:<8} {:<8}",
        "프롬프트", "토큰수", "생성 결과", "avg_Ω", "avg_Φ");

    let mut total_tokens = 0;
    let mut total_omega = 0.0;
    let mut total_phi = 0.0;
    let mut total_predictions = 0;

    for prompt in &prompts {
        let result = engine.generate(prompt, 20);
        let n_gen = result.generated.len();
        total_tokens += n_gen;

        let avg_omega = if !result.predictions.is_empty() {
            result.predictions.iter().map(|p| p.omega).sum::<f64>() / result.predictions.len() as f64
        } else { 0.0 };

        let avg_phi = if !result.predictions.is_empty() {
            result.predictions.iter().map(|p| p.phi).sum::<f64>() / result.predictions.len() as f64
        } else { 0.0 };

        total_omega += avg_omega * n_gen as f64;
        total_phi += avg_phi * n_gen as f64;
        total_predictions += n_gen;

        // 40자로 잘라서 표시
        let display = if result.full_text.len() > 48 {
            format!("{}...", &result.full_text[..45])
        } else {
            result.full_text.clone()
        };

        println!("{:<20} {:<6} {:<50} {:<8.4} {:<8.4}",
            prompt, n_gen, display, avg_omega, avg_phi);
    }

    let overall_omega = if total_predictions > 0 { total_omega / total_predictions as f64 } else { 0.0 };
    let overall_phi = if total_predictions > 0 { total_phi / total_predictions as f64 } else { 0.0 };

    println!("\n--- 결과 ---");
    println!("총 생성 토큰: {}", total_tokens);
    println!("프롬프트당 평균: {:.1} 토큰", total_tokens as f64 / prompts.len() as f64);
    println!("전체 평균 Ω: {:.4}", overall_omega);
    println!("전체 평균 Φ: {:.4}", overall_phi);
}

// ---------------------------------------------------------------------------
// 3. 분류 F1 측정
// ---------------------------------------------------------------------------

#[test]
fn bench_accuracy_classification() {
    println!("\n{}", "=".repeat(70));
    println!("  AXOL WTE 분류 F1 측정");
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

    let (labeled, labels) = classification_data();
    let mut engine = WaveTextEngine::from_corpus_with_config(&corpus, &config);
    engine.train_classifier(&labeled, labels.clone());

    let test_cases = classification_test_cases();

    println!("\n{:<30} {:<12} {:<12} {:<10} {:<10}",
        "텍스트", "예측", "정답", "일치?", "confidence");

    let mut correct = 0;
    let total = test_cases.len();
    let n_classes = labels.len();

    // per-class TP, FP, FN
    let mut tp = vec![0usize; n_classes];
    let mut fp = vec![0usize; n_classes];
    let mut fn_ = vec![0usize; n_classes];

    for (text, expected) in &test_cases {
        if let Some(result) = engine.classify(text) {
            let expected_id = labels.iter().position(|l| l == expected).unwrap_or(999);
            let is_correct = result.class_label == *expected;
            if is_correct {
                correct += 1;
                tp[result.class_id] += 1;
            } else {
                fp[result.class_id] += 1;
                if expected_id < n_classes {
                    fn_[expected_id] += 1;
                }
            }

            println!("{:<30} {:<12} {:<12} {:<10} {:<10.4}",
                text, result.class_label, expected,
                if is_correct { "O" } else { "X" },
                result.confidence);
        }
    }

    // Macro F1
    let mut f1_sum = 0.0;
    let mut f1_count = 0;
    println!("\n--- 클래스별 ---");
    for (i, label) in labels.iter().enumerate() {
        let precision = if tp[i] + fp[i] > 0 { tp[i] as f64 / (tp[i] + fp[i]) as f64 } else { 0.0 };
        let recall = if tp[i] + fn_[i] > 0 { tp[i] as f64 / (tp[i] + fn_[i]) as f64 } else { 0.0 };
        let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
        f1_sum += f1;
        f1_count += 1;
        println!("  {}: P={:.2} R={:.2} F1={:.2}", label, precision, recall, f1);
    }

    let accuracy = correct as f64 / total as f64 * 100.0;
    let macro_f1 = f1_sum / f1_count as f64;

    println!("\n--- 결과 ---");
    println!("정확도: {}/{} ({:.1}%)", correct, total, accuracy);
    println!("Macro F1: {:.4}", macro_f1);
}

// ---------------------------------------------------------------------------
// 4. 이상탐지 분리도 측정
// ---------------------------------------------------------------------------

#[test]
fn bench_accuracy_anomaly_detection() {
    println!("\n{}", "=".repeat(70));
    println!("  AXOL WTE 이상탐지 분리도 측정");
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

    // 정상 문장
    let normal_tests = vec![
        "the cat sat on the mat",
        "the dog ate the bone",
        "the boy went to school",
        "the bird flew over the tree",
        "the girl read a book",
    ];

    // 이상 문장 (코퍼스와 무관한 문장들)
    let anomaly_tests = vec![
        "quantum entanglement drives blockchain synergy",
        "the recursive algorithm optimizes neural gradients",
        "cryptocurrency mining consumes excessive kilowatts",
        "photosynthesis converts carbon dioxide into glucose",
        "microprocessor architectures leverage parallel pipelines",
    ];

    println!("\n[정상 문장]");
    let mut normal_scores = Vec::new();
    for text in &normal_tests {
        if let Some(result) = engine.anomaly_score(text) {
            normal_scores.push(result.score);
            println!("  {:.4} {} \"{}\"",
                result.score,
                if result.is_anomalous { "[!]" } else { "[ ]" },
                text);
        }
    }

    println!("\n[이상 문장]");
    let mut anomaly_scores = Vec::new();
    for text in &anomaly_tests {
        if let Some(result) = engine.anomaly_score(text) {
            anomaly_scores.push(result.score);
            println!("  {:.4} {} \"{}\"",
                result.score,
                if result.is_anomalous { "[!]" } else { "[ ]" },
                text);
        }
    }

    let normal_avg = if !normal_scores.is_empty() {
        normal_scores.iter().sum::<f64>() / normal_scores.len() as f64
    } else { 0.0 };
    let anomaly_avg = if !anomaly_scores.is_empty() {
        anomaly_scores.iter().sum::<f64>() / anomaly_scores.len() as f64
    } else { 0.0 };

    let normal_max = normal_scores.iter().cloned().fold(0.0_f64, f64::max);
    let anomaly_min = anomaly_scores.iter().cloned().fold(f64::MAX, f64::min);
    let separation = anomaly_min - normal_max;

    // 간단 AUROC 근사: 정상/이상 쌍 비교
    let mut correct_pairs = 0;
    let mut total_pairs = 0;
    for &ns in &normal_scores {
        for &as_ in &anomaly_scores {
            total_pairs += 1;
            if as_ > ns { correct_pairs += 1; }
        }
    }
    let auroc = if total_pairs > 0 { correct_pairs as f64 / total_pairs as f64 } else { 0.0 };

    println!("\n--- 결과 ---");
    println!("정상 평균 점수: {:.4}", normal_avg);
    println!("이상 평균 점수: {:.4}", anomaly_avg);
    println!("분리 갭 (이상min - 정상max): {:.4}", separation);
    println!("AUROC (근사): {:.4}", auroc);
}

// ---------------------------------------------------------------------------
// 5. 종합 속도 측정 (WTE)
// ---------------------------------------------------------------------------

#[test]
fn bench_accuracy_speed_summary() {
    println!("\n{}", "=".repeat(70));
    println!("  AXOL WTE 종합 속도 측정");
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

    // 빌드 시간
    let start = Instant::now();
    let engine = WaveTextEngine::from_corpus_with_config(&corpus, &config);
    let build_time = start.elapsed();

    // Warmup
    let _ = engine.autocomplete("the cat", 1);
    let _ = engine.generate("the cat", 5);

    // 자동완성 속도
    let n = 50;
    let start = Instant::now();
    for _ in 0..n {
        let _ = engine.autocomplete("the cat sat on", 3);
    }
    let autocomplete_time = start.elapsed();

    // 생성 속도
    let start = Instant::now();
    for _ in 0..n {
        let _ = engine.generate("the cat", 20);
    }
    let gen_time = start.elapsed();

    // 분류 속도
    let (labeled, labels) = classification_data();
    let mut engine_cls = WaveTextEngine::from_corpus_with_config(&corpus, &config);
    engine_cls.train_classifier(&labeled, labels);

    let start = Instant::now();
    for _ in 0..n {
        let _ = engine_cls.classify("the cat ran fast");
    }
    let cls_time = start.elapsed();

    // 이상탐지 속도
    let normal_docs: Vec<&str> = corpus[..20].iter().copied().collect();
    let mut engine_anom = WaveTextEngine::from_corpus_with_config(&corpus, &config);
    engine_anom.train_anomaly_detector(&normal_docs);

    let start = Instant::now();
    for _ in 0..n {
        let _ = engine_anom.anomaly_score("the cat sat on the mat");
    }
    let anom_time = start.elapsed();

    // 모델 크기
    let model_bytes = engine.model_size_bytes();
    let serial_bytes = engine.to_bytes().len();

    println!("\n{:<25} {:<15} {:<15}", "항목", "측정값", "목표");
    println!("{:-<55}", "");
    println!("{:<25} {:<15.0} {:<15}",
        "빌드 시간 (ms)", build_time.as_secs_f64() * 1000.0, "< 5,000");
    println!("{:<25} {:<15.3} {:<15}",
        "자동완성 (ms/call)", autocomplete_time.as_secs_f64() * 1000.0 / n as f64, "< 1");
    println!("{:<25} {:<15.2} {:<15}",
        "생성 20tok (ms/call)", gen_time.as_secs_f64() * 1000.0 / n as f64, "< 20");
    println!("{:<25} {:<15.3} {:<15}",
        "분류 (ms/call)", cls_time.as_secs_f64() * 1000.0 / n as f64, "< 2");
    println!("{:<25} {:<15.3} {:<15}",
        "이상탐지 (ms/call)", anom_time.as_secs_f64() * 1000.0 / n as f64, "< 5");
    println!("{:<25} {:<15.1} {:<15}",
        "모델 크기 (KB)", model_bytes as f64 / 1024.0, "< 2,048");
    println!("{:<25} {:<15.1} {:<15}",
        "직렬화 크기 (KB)", serial_bytes as f64 / 1024.0, "< 2,048");
}

// ---------------------------------------------------------------------------
// 6. WTE 정확도 측정
// ---------------------------------------------------------------------------

#[test]
fn bench_accuracy_version_comparison() {
    println!("\n{}", "=".repeat(70));
    println!("  AXOL WTE 정확도 측정");
    println!("{}", "=".repeat(70));

    let corpus = medium_corpus();

    let test_prompts: Vec<(&str, Vec<&str>)> = vec![
        ("the cat sat on", vec!["the", "mat", "a"]),
        ("the dog ate", vec!["the", "a", "bone"]),
        ("the boy went to", vec!["the", "school", "a"]),
        ("the bird flew over", vec!["the", "tree", "a"]),
        ("the sun shone in", vec!["the", "sky", "a"]),
    ];

    // WTE
    let config = EngineConfig {
        dim: 64,
        num_merges: 200,
        max_vocab: 2000,
        seed: 42,
        anomaly_threshold: 1.5,
        ..EngineConfig::default()
    };
    let wte = WaveTextEngine::from_corpus_with_config(&corpus, &config);
    let mut wte_hits = 0;
    for (prompt, expected) in &test_prompts {
        let preds = wte.autocomplete(prompt, 3);
        let hit = preds.iter().any(|p| expected.contains(&p.token.as_str()));
        if hit { wte_hits += 1; }
    }

    let n = test_prompts.len();
    println!("\n{:<10} {:<15} {:<15}", "모델", "정답 수", "Top-3 정확도");
    println!("{:-<40}", "");
    println!("{:<10} {:<15} {:.1}%", "WTE", format!("{}/{}", wte_hits, n), wte_hits as f64 / n as f64 * 100.0);

    // 세부 예측 출력
    println!("\n{:<25} {:<10}", "프롬프트", "WTE");
    println!("{:-<35}", "");

    for (prompt, _expected) in &test_prompts {
        let wte_tok = wte.autocomplete(prompt, 1)
            .first().map(|p| p.token.clone()).unwrap_or_else(|| "?".into());

        println!("{:<25} {:<10}", prompt, wte_tok);
    }
}
