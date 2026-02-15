//! AXOL 고유 평가 프레임워크 — 시간축 없는 파동 기반 평가.
//!
//! 기존 NLP 평가(다음 토큰 정확도, perplexity)는 시간축 전제.
//! AXOL은 공간+확률축 기반이므로 고유한 평가 기준이 필요하다.
//!
//! 평가 항목:
//!   1. 의미장 공명 — 공명하는 토큰이 맥락에 속하는가?
//!   2. 의미 유사도 — 비슷한 텍스트 → 비슷한 파동 상태?
//!   3. 분류 정확도 — 전체 상태 → 범주 (시간축 없는 작업)
//!   4. 이상 탐지 분리도 — 정상 vs 이상 점수 차이
//!   5. Φ 붕괴 분리도 — 확률 분포의 결정성
//!   6. 핑거프린트 자기일관성 — 같은 도메인 → 높은 유사도
//!
//! Run: cargo test --release --test bench_axol_eval -- --show-output

use axol::text::engine::*;
use axol::text::data::medium_corpus;
use num_complex::Complex64;

// ═══════════════════════════════════════════════════════════════════
// 의미장 정의 (Semantic Fields)
// ═══════════════════════════════════════════════════════════════════

/// 의미장: 특정 맥락에 "속하는" 토큰 집합.
/// 시간 순서 무관 — 이 단어들이 같은 의미 공간에 존재하는가?
fn semantic_fields() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        ("동물", vec![
            "cat", "dog", "bird", "fish", "mouse", "rabbit", "horse",
            "cow", "sheep", "frog", "duck", "owl", "hen", "pig",
            "sat", "ate", "flew", "swam", "chased", "ran", "slept",
            "jumped", "climbed", "hopped", "sang", "built", "nest",
            "mat", "bone", "tree", "water", "hole", "bed", "field",
        ]),
        ("사람", vec![
            "boy", "girl", "man", "woman", "child", "children", "baby",
            "king", "queen", "teacher", "doctor", "farmer", "old",
            "went", "read", "walked", "cooked", "played", "taught",
            "helped", "school", "book", "store", "home", "dinner",
            "yard", "garden", "student", "friend",
        ]),
        ("자연", vec![
            "sun", "moon", "rain", "wind", "snow", "star", "sky",
            "river", "sea", "lake", "mountain", "flower", "spring",
            "winter", "summer", "night", "cloud", "storm", "thunder",
            "shone", "fell", "blew", "covered", "rose", "bloomed",
            "flowed", "bright", "cold", "warm", "dark",
        ]),
    ]
}

// ═══════════════════════════════════════════════════════════════════
// 테스트 1: 의미장 공명 (Semantic Field Resonance)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_1_semantic_field_resonance() {
    println!("\n{}", "═".repeat(70));
    println!("  테스트 1: 의미장 공명 (Semantic Field Resonance)");
    println!("  질문: 단어 수준 양자 측정이 맥락의 의미장을 찾는가?");
    println!("  측정: |⟨ψ_merged | ψ_word⟩|² (서브워드 중첩 → 단어 파동)");
    println!("{}", "═".repeat(70));

    let corpus = medium_corpus();
    let config = EngineConfig {
        dim: 64, num_merges: 200, max_vocab: 2000, seed: 42,
        ..EngineConfig::default()
    };
    let engine = WaveTextEngine::from_corpus_with_config(&corpus, &config);
    let fields = semantic_fields();

    // 모든 의미장 단어를 후보로 합침
    let all_candidates: Vec<&str> = fields.iter()
        .flat_map(|(_, words)| words.iter().copied())
        .collect();

    let test_cases: Vec<(&str, &str)> = vec![
        ("the cat sat on the mat", "동물"),
        ("the dog chased the cat", "동물"),
        ("the bird flew over the tree", "동물"),
        ("the fish swam in the water", "동물"),
        ("the boy went to school", "사람"),
        ("the girl read a book", "사람"),
        ("the man walked to the store", "사람"),
        ("the woman cooked dinner", "사람"),
        ("the sun shone in the sky", "자연"),
        ("the rain fell on the roof", "자연"),
        ("the wind blew through trees", "자연"),
        ("the snow covered the ground", "자연"),
    ];

    let top_k = 10;
    let mut total_resonance = 0.0;
    let mut total_cases = 0;

    for (text, expected_field) in &test_cases {
        let ranked = engine.word_resonances(text, &all_candidates);
        let field_words: &Vec<&str> = &fields.iter()
            .find(|(name, _)| name == expected_field)
            .unwrap().1;

        // 상위 k개 중 올바른 의미장에 속하는 단어 수
        let top = &ranked[..ranked.len().min(top_k)];
        let hits: Vec<&str> = top.iter()
            .filter(|(w, _)| field_words.contains(&w.as_str()))
            .map(|(w, _)| w.as_str())
            .collect();
        let misses: Vec<&str> = top.iter()
            .filter(|(w, _)| !field_words.contains(&w.as_str()))
            .take(5)
            .map(|(w, _)| w.as_str())
            .collect();

        let resonance = hits.len() as f64 / top.len() as f64;
        total_resonance += resonance;
        total_cases += 1;

        println!("  \"{}\" [{}]", text, expected_field);
        println!("    공명률: {}/{} ({:.0}%)  적중: {:?}",
            hits.len(), top.len(), resonance * 100.0, hits);
        if !misses.is_empty() {
            println!("    비적중: {:?}", misses);
        }
        // 상위 3개 + 확률 표시
        let top3: Vec<String> = ranked.iter().take(3)
            .map(|(w, p)| format!("{}({:.3})", w, p))
            .collect();
        println!("    상위3: {}", top3.join(", "));
    }

    let avg_resonance = total_resonance / total_cases as f64;
    let n_per_field = fields[0].1.len() as f64;
    let random_baseline = n_per_field / all_candidates.len() as f64;
    println!("\n  ▶ 평균 의미장 공명률: {:.1}%", avg_resonance * 100.0);
    println!("    (랜덤 기준: ~{:.1}%)", random_baseline * 100.0);
    println!("    의미: 상위 {} 공명 단어 중 올바른 의미장 비율", top_k);
    if avg_resonance > random_baseline + 0.05 {
        println!("    판정: ✓ 의미장 공명이 랜덤 이상");
    } else if avg_resonance > random_baseline {
        println!("    판정: △ 의미장 공명이 약하지만 랜덤 이상");
    } else {
        println!("    판정: ✗ 의미장 공명이 랜덤 이하");
    }
}

// ═══════════════════════════════════════════════════════════════════
// 테스트 2: 의미 유사도 (Semantic Similarity)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_2_semantic_similarity() {
    println!("\n{}", "═".repeat(70));
    println!("  테스트 2: 의미 유사도 (Semantic Similarity)");
    println!("  질문: 비슷한 문장 → 비슷한 파동 상태?");
    println!("{}", "═".repeat(70));

    let corpus = medium_corpus();
    let config = EngineConfig {
        dim: 64, num_merges: 200, max_vocab: 2000, seed: 42,
        ..EngineConfig::default()
    };
    let engine = WaveTextEngine::from_corpus_with_config(&corpus, &config);

    // 같은 도메인 쌍 (유사해야 함)
    let same_domain = vec![
        ("the cat sat on the mat", "the dog sat on the log"),
        ("the cat ate the fish", "the dog ate the bone"),
        ("the boy went to school", "the girl went to school"),
        ("the sun shone in the sky", "the moon rose at night"),
    ];

    // 다른 도메인 쌍 (달라야 함)
    let diff_domain = vec![
        ("the cat sat on the mat", "the sun shone in the sky"),
        ("the dog ate the bone", "the boy went to school"),
        ("the bird flew over the tree", "the man walked to the store"),
        ("the fish swam in the water", "the wind blew through trees"),
    ];

    let mut same_sims = Vec::new();
    let mut diff_sims = Vec::new();

    println!("\n  [같은 도메인 쌍]");
    for (a, b) in &same_domain {
        let state_a = engine.process_text(a);
        let state_b = engine.process_text(b);
        let sim = wave_cosine_similarity(&state_a, &state_b);
        same_sims.push(sim);
        println!("    \"{}\"\n    \"{}\" → 유사도: {:.4}", a, b, sim);
    }

    println!("\n  [다른 도메인 쌍]");
    for (a, b) in &diff_domain {
        let state_a = engine.process_text(a);
        let state_b = engine.process_text(b);
        let sim = wave_cosine_similarity(&state_a, &state_b);
        diff_sims.push(sim);
        println!("    \"{}\"\n    \"{}\" → 유사도: {:.4}", a, b, sim);
    }

    let avg_same = same_sims.iter().sum::<f64>() / same_sims.len() as f64;
    let avg_diff = diff_sims.iter().sum::<f64>() / diff_sims.len() as f64;
    let gap = avg_same - avg_diff;

    println!("\n  ▶ 같은 도메인 평균: {:.4}", avg_same);
    println!("  ▶ 다른 도메인 평균: {:.4}", avg_diff);
    println!("  ▶ 분리도 (gap):     {:.4}", gap);
    println!("    의미: gap > 0이면 같은 도메인이 더 유사한 파동 상태를 가짐");
    if gap > 0.01 {
        println!("    판정: ✓ 의미 유사도가 도메인을 구분함");
    } else if gap > 0.0 {
        println!("    판정: △ 약한 분리 — 파동 표현이 의미를 약하게 인코딩");
    } else {
        println!("    판정: ✗ 분리 실패 — 파동 상태가 의미를 구분 못함");
    }
}

/// 두 ReservoirState의 merged wave 코사인 유사도.
fn wave_cosine_similarity(
    a: &axol::text::reservoir::ReservoirState,
    b: &axol::text::reservoir::ReservoirState,
) -> f64 {
    let dim = a.merged.dim.min(b.merged.dim);
    let mut dot = Complex64::new(0.0, 0.0);
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for i in 0..dim {
        let ca = a.merged.amplitudes.data[i];
        let cb = b.merged.amplitudes.data[i];
        dot += ca.conj() * cb;
        norm_a += ca.norm_sqr();
        norm_b += cb.norm_sqr();
    }

    if norm_a > 1e-10 && norm_b > 1e-10 {
        dot.norm() / (norm_a.sqrt() * norm_b.sqrt())
    } else {
        0.0
    }
}

// ═══════════════════════════════════════════════════════════════════
// 테스트 3: 분류 정확도 (Classification — 시간축 없는 작업)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_3_classification() {
    println!("\n{}", "═".repeat(70));
    println!("  테스트 3: 분류 정확도 (Classification)");
    println!("  질문: 전체 파동 상태 → 올바른 범주? (시간축 없는 순수 작업)");
    println!("{}", "═".repeat(70));

    let corpus = medium_corpus();
    let config = EngineConfig {
        dim: 64, num_merges: 200, max_vocab: 2000, seed: 42,
        ..EngineConfig::default()
    };
    let mut engine = WaveTextEngine::from_corpus_with_config(&corpus, &config);

    let labeled = vec![
        ("the cat sat on the mat", 0), ("the dog ate the bone", 0),
        ("the bird flew over the tree", 0), ("the cat ate the fish", 0),
        ("the dog ran in the park", 0), ("the bird sang in the tree", 0),
        ("the fish swam in the water", 0), ("the rabbit hopped across the field", 0),
        ("the boy went to school", 1), ("the girl read a book", 1),
        ("the man walked to the store", 1), ("the woman cooked dinner", 1),
        ("the boy ran home", 1), ("the girl played in the yard", 1),
        ("the teacher taught the students", 1), ("the doctor helped the sick man", 1),
        ("the sun shone in the sky", 2), ("the rain fell on the roof", 2),
        ("the wind blew through trees", 2), ("the snow covered the ground", 2),
        ("the moon rose at night", 2), ("the stars shone brightly", 2),
        ("the river flowed to the sea", 2), ("the flowers bloomed in spring", 2),
    ];
    let labels = vec!["동물".into(), "사람".into(), "자연".into()];
    engine.train_classifier(&labeled, labels);

    let test_cases = vec![
        ("the cat ran fast", "동물"),
        ("the dog played fetch", "동물"),
        ("the bird nested", "동물"),
        ("the fish jumped", "동물"),
        ("the boy ate lunch", "사람"),
        ("the girl went home", "사람"),
        ("the man sat down", "사람"),
        ("the woman sang", "사람"),
        ("the rain poured down", "자연"),
        ("the wind howled", "자연"),
        ("the sun set slowly", "자연"),
        ("the snow melted", "자연"),
    ];

    let mut correct = 0;
    let mut total_phi = 0.0;
    let mut total_conf = 0.0;

    for (text, expected) in &test_cases {
        let result = engine.classify(text).unwrap();
        let ok = result.class_label == *expected;
        if ok { correct += 1; }
        total_phi += result.phi;
        total_conf += result.confidence;

        let mark = if ok { "✓" } else { "✗" };
        println!("  {} \"{}\" → {} (신뢰도={:.3}, Φ={:.4}) [정답: {}]",
            mark, text, result.class_label, result.confidence, result.phi, expected);
    }

    let acc = correct as f64 / test_cases.len() as f64;
    let avg_phi = total_phi / test_cases.len() as f64;
    let avg_conf = total_conf / test_cases.len() as f64;

    println!("\n  ▶ 정확도: {}/{} ({:.1}%)", correct, test_cases.len(), acc * 100.0);
    println!("  ▶ 평균 신뢰도: {:.3}", avg_conf);
    println!("  ▶ 평균 Φ: {:.4}", avg_phi);
    println!("    의미: 분류는 시간축 없는 작업 — AXOL 아키텍처에 적합");
    println!("    랜덤 기준: 33.3%");
    if acc >= 0.5 {
        println!("    판정: ✓ 랜덤 이상의 분류 성능");
    } else {
        println!("    판정: △ 분류 성능 개선 필요");
    }
}

// ═══════════════════════════════════════════════════════════════════
// 테스트 4: 이상 탐지 분리도 (Anomaly Separation)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_4_anomaly_separation() {
    println!("\n{}", "═".repeat(70));
    println!("  테스트 4: 이상 탐지 분리도 (Anomaly Separation)");
    println!("  질문: 정상 텍스트 vs 이상 텍스트의 점수 차이가 있는가?");
    println!("{}", "═".repeat(70));

    let corpus = medium_corpus();
    let config = EngineConfig {
        dim: 64, num_merges: 200, max_vocab: 2000, seed: 42,
        ..EngineConfig::default()
    };
    let mut engine = WaveTextEngine::from_corpus_with_config(&corpus, &config);

    let normal_docs: Vec<&str> = corpus.iter().copied().take(100).collect();
    engine.train_anomaly_detector(&normal_docs);

    // 정상 텍스트 (코퍼스와 유사)
    let normal_test = vec![
        "the cat sat on the mat",
        "the dog ate the bone",
        "the boy went to school",
        "the sun shone brightly",
        "the girl read a book",
        "the rain fell on the ground",
    ];

    // 이상 텍스트 (코퍼스와 완전히 다른 도메인)
    let anomalous = vec![
        "quantum entanglement photon laser beam optics",
        "blockchain cryptocurrency mining hash rate",
        "neural network gradient descent backpropagation",
        "semiconductor transistor silicon wafer lithography",
        "algorithm complexity recursive binary search",
        "database index query optimization cache",
    ];

    let mut normal_scores = Vec::new();
    let mut anomaly_scores = Vec::new();

    println!("\n  [정상 텍스트]");
    for text in &normal_test {
        if let Some(score) = engine.anomaly_score(text) {
            println!("    {:.3} {} \"{}\"",
                score.score, if score.is_anomalous { "⚠" } else { "○" }, text);
            normal_scores.push(score.score);
        }
    }

    println!("\n  [이상 텍스트]");
    for text in &anomalous {
        if let Some(score) = engine.anomaly_score(text) {
            println!("    {:.3} {} \"{}\"",
                score.score, if score.is_anomalous { "⚠" } else { "○" }, text);
            anomaly_scores.push(score.score);
        }
    }

    let avg_normal = normal_scores.iter().sum::<f64>() / normal_scores.len().max(1) as f64;
    let avg_anomaly = anomaly_scores.iter().sum::<f64>() / anomaly_scores.len().max(1) as f64;
    let gap = avg_anomaly - avg_normal;

    println!("\n  ▶ 정상 평균 점수: {:.3}", avg_normal);
    println!("  ▶ 이상 평균 점수: {:.3}", avg_anomaly);
    println!("  ▶ 분리도 (gap):   {:.3}", gap);
    println!("    의미: gap > 0이면 이상 텍스트를 더 높은 점수로 감지");
    if gap > 0.2 {
        println!("    판정: ✓ 이상 탐지 분리가 명확함");
    } else if gap > 0.0 {
        println!("    판정: △ 약한 분리 — 임계값 조정 필요");
    } else {
        println!("    판정: ✗ 분리 실패 — 정상과 이상을 구분 못함");
    }
}

// ═══════════════════════════════════════════════════════════════════
// 테스트 5: Φ 붕괴 분리도 (Collapse Separation)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_5_phi_collapse() {
    println!("\n{}", "═".repeat(70));
    println!("  테스트 5: Φ 붕괴 분리도 (Collapse Separation)");
    println!("  질문: 의미적으로 명확한 맥락에서 Φ가 더 높은가?");
    println!("{}", "═".repeat(70));

    let corpus = medium_corpus();
    let config = EngineConfig {
        dim: 64, num_merges: 200, max_vocab: 2000, seed: 42,
        ..EngineConfig::default()
    };
    let engine = WaveTextEngine::from_corpus_with_config(&corpus, &config);

    // 명확한 맥락 (특정 의미장에 강하게 속함)
    let clear_contexts = vec![
        "the cat sat on the mat",           // 매우 구체적
        "the dog ate the bone",             // 매우 구체적
        "the boy went to school",           // 매우 구체적
    ];

    // 모호한 맥락 (의미장이 혼합됨)
    let ambiguous_contexts = vec![
        "the",                              // 거의 무정보
        "a",                                // 거의 무정보
        "on the",                           // 약한 맥락
    ];

    let mut clear_phis = Vec::new();
    let mut ambig_phis = Vec::new();

    println!("\n  [명확한 맥락]");
    for text in &clear_contexts {
        let preds = engine.autocomplete(text, 5);
        let phi = if !preds.is_empty() { preds[0].phi } else { 0.0 };
        let max_conf = if !preds.is_empty() { preds[0].confidence } else { 0.0 };
        clear_phis.push(phi);
        println!("    Φ={:.4} conf={:.4} \"{}\"", phi, max_conf, text);
        if !preds.is_empty() {
            let tokens: Vec<&str> = preds.iter().take(3).map(|p| p.token.as_str()).collect();
            println!("      상위: {:?}", tokens);
        }
    }

    println!("\n  [모호한 맥락]");
    for text in &ambiguous_contexts {
        let preds = engine.autocomplete(text, 5);
        let phi = if !preds.is_empty() { preds[0].phi } else { 0.0 };
        let max_conf = if !preds.is_empty() { preds[0].confidence } else { 0.0 };
        ambig_phis.push(phi);
        println!("    Φ={:.4} conf={:.4} \"{}\"", phi, max_conf, text);
        if !preds.is_empty() {
            let tokens: Vec<&str> = preds.iter().take(3).map(|p| p.token.as_str()).collect();
            println!("      상위: {:?}", tokens);
        }
    }

    let avg_clear = clear_phis.iter().sum::<f64>() / clear_phis.len().max(1) as f64;
    let avg_ambig = ambig_phis.iter().sum::<f64>() / ambig_phis.len().max(1) as f64;
    let gap = avg_clear - avg_ambig;

    println!("\n  ▶ 명확한 맥락 평균 Φ: {:.4}", avg_clear);
    println!("  ▶ 모호한 맥락 평균 Φ: {:.4}", avg_ambig);
    println!("  ▶ Φ 분리도:          {:.4}", gap);
    println!("    의미: Φ는 확률 분포의 결정성. 명확한 맥락에서 더 높아야 함");
    if gap > 0.01 {
        println!("    판정: ✓ 맥락 정보가 Φ에 반영됨");
    } else {
        println!("    판정: △ Φ가 맥락 정보를 반영하지 못함");
    }
}

// ═══════════════════════════════════════════════════════════════════
// 테스트 6: 핑거프린트 도메인 분리 (Fingerprint Domain Separation)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_6_fingerprint_domains() {
    println!("\n{}", "═".repeat(70));
    println!("  테스트 6: 핑거프린트 도메인 분리 (Fingerprint Domain Separation)");
    println!("  질문: 같은 도메인 → 유사한 핑거프린트?");
    println!("{}", "═".repeat(70));

    let corpus = medium_corpus();
    let config = EngineConfig {
        dim: 64, num_merges: 200, max_vocab: 2000, seed: 42,
        ..EngineConfig::default()
    };
    let engine = WaveTextEngine::from_corpus_with_config(&corpus, &config);

    // 3개 도메인 핑거프린트
    let animal_docs: Vec<&str> = vec![
        "the cat sat on the mat", "the dog ate the bone",
        "the bird flew over the tree", "the fish swam in the water",
        "the cat chased the mouse", "the rabbit hopped across the field",
    ];
    let people_docs: Vec<&str> = vec![
        "the boy went to school", "the girl read a book",
        "the man walked to the store", "the woman cooked dinner",
        "the teacher taught the students", "the children played in the yard",
    ];
    let nature_docs: Vec<&str> = vec![
        "the sun shone in the sky", "the rain fell on the roof",
        "the wind blew through trees", "the snow covered the ground",
        "the moon rose at night", "the river flowed to the sea",
    ];

    let fp_animal = engine.fingerprint(&animal_docs);
    let fp_people = engine.fingerprint(&people_docs);
    let fp_nature = engine.fingerprint(&nature_docs);

    // 자기 유사도
    let self_sim_a = fp_animal.similarity(&fp_animal);
    let self_sim_p = fp_people.similarity(&fp_people);
    let self_sim_n = fp_nature.similarity(&fp_nature);

    // 교차 유사도
    let cross_ap = fp_animal.similarity(&fp_people);
    let cross_an = fp_animal.similarity(&fp_nature);
    let cross_pn = fp_people.similarity(&fp_nature);

    println!("\n  [자기 유사도 (1.0이어야 함)]");
    println!("    동물-동물: {:.4}", self_sim_a);
    println!("    사람-사람: {:.4}", self_sim_p);
    println!("    자연-자연: {:.4}", self_sim_n);

    println!("\n  [교차 유사도 (자기 유사도보다 낮아야 함)]");
    println!("    동물-사람: {:.4}", cross_ap);
    println!("    동물-자연: {:.4}", cross_an);
    println!("    사람-자연: {:.4}", cross_pn);

    let avg_self = (self_sim_a + self_sim_p + self_sim_n) / 3.0;
    let avg_cross = (cross_ap + cross_an + cross_pn) / 3.0;
    let gap = avg_self - avg_cross;

    println!("\n  ▶ 자기 유사도 평균: {:.4}", avg_self);
    println!("  ▶ 교차 유사도 평균: {:.4}", avg_cross);
    println!("  ▶ 도메인 분리도:   {:.4}", gap);
    println!("    의미: gap > 0이면 핑거프린트가 도메인을 구분함");
    if gap > 0.05 {
        println!("    판정: ✓ 핑거프린트가 도메인을 명확히 구분");
    } else if gap > 0.0 {
        println!("    판정: △ 약한 도메인 분리");
    } else {
        println!("    판정: ✗ 도메인 분리 실패");
    }
}

// ═══════════════════════════════════════════════════════════════════
// 종합 평가 요약
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_0_summary() {
    println!("\n{}", "═".repeat(70));
    println!("  AXOL WTE 종합 평가 요약");
    println!("  평가 패러다임: 공간 + 확률축 (시간축 배제)");
    println!("{}", "═".repeat(70));
    println!();
    println!("  평가 항목              | 측정 내용                    | 시간축?");
    println!("  ─────────────────────────────────────────────────────────────");
    println!("  1. 의미장 공명          | 공명 토큰이 맥락에 속하는가  | ✗ 없음");
    println!("  2. 의미 유사도          | 비슷한 문장 → 비슷한 파동    | ✗ 없음");
    println!("  3. 분류 정확도          | 전체 상태 → 범주             | ✗ 없음");
    println!("  4. 이상 탐지 분리도     | 정상 vs 이상 점수 차이       | ✗ 없음");
    println!("  5. Φ 붕괴 분리도       | 맥락 명확성 → Φ 크기         | ✗ 없음");
    println!("  6. 핑거프린트 도메인    | 같은 도메인 → 높은 유사도    | ✗ 없음");
    println!();
    println!("  ※ 기존 NLP 평가(다음 토큰 정확도, perplexity, BLEU)는");
    println!("    시간축 전제이므로 AXOL 평가에서 제외됨.");
    println!();
    println!("  각 테스트를 개별 실행하려면:");
    println!("  cargo test --release --test bench_axol_eval test_1 -- --show-output");
    println!("{}", "═".repeat(70));
}
