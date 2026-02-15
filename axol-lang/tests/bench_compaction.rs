//! 공명 정리 (Resonance Compaction) 실측 벤치마크.
//!
//! 압축 ON/OFF A/B 비교, 포화도 추이, 예측 품질, 속도 영향을 측정한다.
//! Run with: cargo test --test bench_compaction -- --show-output

use std::time::Instant;

use axol::text::engine::*;
use axol::text::reservoir::*;
use axol::text::sps::SemanticPhaseSpace;
use axol::text::tokenizer::{BpeTokenizer, Vocabulary};
use axol::text::data::{medium_corpus, tiny_corpus};

// ---------------------------------------------------------------------------
// 1. A/B: 압축 ON vs OFF — 포화도 · 위상 정합도 · 에너지 비교
// ---------------------------------------------------------------------------

#[test]
fn bench_compaction_ab_reservoir_metrics() {
    println!("\n=== 공명 정리 A/B: 저수지 메트릭 비교 ===");
    println!("{:<12} {:<12} {:<12} {:<12} {:<12}",
        "모드", "포화도(avg)", "정합도", "에너지", "시간(ms)");

    let corpus = medium_corpus();
    let bpe = BpeTokenizer::from_corpus(&corpus, 200, 2000);
    let encoded: Vec<Vec<usize>> = corpus.iter()
        .map(|s| bpe.encode(s))
        .collect();

    let sps = SemanticPhaseSpace::from_encoded_corpus(
        bpe.size, &encoded, 64, 42,
    );

    // 여러 문장에 대해 측정
    let test_sentences = &corpus[..20];

    // ─── OFF ───
    let mut res_off = WaveResonanceReservoir::new(64);
    let mut sat_sum_off = 0.0;
    let mut coh_sum_off = 0.0;
    let mut ene_sum_off = 0.0;

    let start = Instant::now();
    for sent in test_sentences {
        let ids = bpe.encode(sent);
        let waves = sps.tokens_to_waves(&ids);
        let state = res_off.process_sequence_no_compaction(&waves);
        // 포화도: 3개 스케일의 평균
        let sat_avg: f64 = res_off.registers.iter()
            .map(|r| r.saturation()).sum::<f64>() / 3.0;
        sat_sum_off += sat_avg;
        coh_sum_off += state.phase_coherence;
        ene_sum_off += state.resonance_energy;
    }
    let time_off = start.elapsed();

    let n = test_sentences.len() as f64;
    println!("{:<12} {:<12.4} {:<12.4} {:<12.4} {:<12.2}",
        "OFF",
        sat_sum_off / n,
        coh_sum_off / n,
        ene_sum_off / n,
        time_off.as_secs_f64() * 1000.0);

    // ─── ON ───
    let mut res_on = WaveResonanceReservoir::new(64);
    let mut sat_sum_on = 0.0;
    let mut coh_sum_on = 0.0;
    let mut ene_sum_on = 0.0;

    let start = Instant::now();
    for sent in test_sentences {
        let ids = bpe.encode(sent);
        let waves = sps.tokens_to_waves(&ids);
        let state = res_on.process_sequence(&waves);
        let sat_avg: f64 = res_on.registers.iter()
            .map(|r| r.saturation()).sum::<f64>() / 3.0;
        sat_sum_on += sat_avg;
        coh_sum_on += state.phase_coherence;
        ene_sum_on += state.resonance_energy;
    }
    let time_on = start.elapsed();

    println!("{:<12} {:<12.4} {:<12.4} {:<12.4} {:<12.2}",
        "ON",
        sat_sum_on / n,
        coh_sum_on / n,
        ene_sum_on / n,
        time_on.as_secs_f64() * 1000.0);

    let sat_reduction = (1.0 - (sat_sum_on / sat_sum_off)) * 100.0;
    println!("\n포화도 감소: {:.1}%", sat_reduction);
    println!("압축 이벤트 총: {}", res_on.compaction_log.len());

    // 압축이 포화도를 줄여야 함
    assert!(sat_sum_on <= sat_sum_off,
        "compaction should reduce saturation: ON={:.4} vs OFF={:.4}",
        sat_sum_on / n, sat_sum_off / n);
}

// ---------------------------------------------------------------------------
// 2. A/B: 압축 ON vs OFF — 자동완성 예측 품질 비교
// ---------------------------------------------------------------------------

#[test]
fn bench_compaction_ab_prediction_quality() {
    println!("\n=== 공명 정리 A/B: 예측 품질 비교 ===");

    let corpus = medium_corpus();
    let config = EngineConfig {
        dim: 64,
        num_merges: 200,
        max_vocab: 2000,
        seed: 42,
        anomaly_threshold: 1.5,
        ..EngineConfig::default()
    };

    // 엔진은 내부적으로 compaction ON 상태로 학습됨
    let engine = WaveTextEngine::from_corpus_with_config(&corpus, &config);

    let test_prompts = vec![
        "the cat sat on",
        "the dog ate",
        "the boy went to",
        "the bird flew over",
        "the sun shone in",
        "the girl read a",
        "the man walked to",
        "the rain fell on",
    ];

    println!("{:<25} {:<10} {:<10} {:<10} {:<15}",
        "프롬프트", "예측", "Ω", "Φ", "top-3");

    let mut omega_sum = 0.0;
    let mut phi_sum = 0.0;
    let mut confidence_sum = 0.0;

    for prompt in &test_prompts {
        let preds = engine.autocomplete(prompt, 3);
        if let Some(p) = preds.first() {
            omega_sum += p.omega;
            phi_sum += p.phi;
            confidence_sum += p.confidence;

            let top3: Vec<&str> = preds.iter().map(|p| p.token.as_str()).collect();
            println!("{:<25} {:<10} {:<10.4} {:<10.4} {:?}",
                prompt, p.token, p.omega, p.phi, top3);
        }
    }

    let n = test_prompts.len() as f64;
    println!("\n평균 Ω: {:.4}", omega_sum / n);
    println!("평균 Φ: {:.4}", phi_sum / n);
    println!("평균 confidence: {:.4}", confidence_sum / n);
}

// ---------------------------------------------------------------------------
// 3. 포화도 추이: 토큰별 포화도 변화 관찰
// ---------------------------------------------------------------------------

#[test]
fn bench_compaction_saturation_timeline() {
    println!("\n=== 공명 정리: 토큰별 포화도 추이 ===");

    let corpus = medium_corpus();
    let bpe = BpeTokenizer::from_corpus(&corpus, 200, 2000);

    let encoded: Vec<Vec<usize>> = corpus.iter()
        .map(|s| bpe.encode(s))
        .collect();

    let sps = SemanticPhaseSpace::from_encoded_corpus(
        bpe.size, &encoded, 64, 42,
    );

    // 긴 문장 하나로 토큰별 추이 관찰
    let long_text = "the cat sat on the mat and the dog sat on the log \
                     and the bird flew over the tree and the boy went to \
                     school and the girl read a book and the man walked \
                     to the store and the sun shone in the sky";
    let ids = bpe.encode(long_text);
    let waves = sps.tokens_to_waves(&ids);

    println!("토큰 수: {}", ids.len());
    println!("{:<6} {:<10} {:<10} {:<10} {:<10}",
        "토큰#", "τ=2 sat", "τ=5 sat", "τ=15 sat", "정합도");

    let mut reservoir = WaveResonanceReservoir::new(64);
    reservoir.reset();

    for (pos, wave) in waves.iter().enumerate() {
        reservoir.process_token(wave, pos);

        // 매 토큰마다 스냅샷
        if pos % 3 == 0 || pos == waves.len() - 1 {
            let sats: Vec<f64> = reservoir.registers.iter()
                .map(|r| r.saturation())
                .collect();
            let state = reservoir.current_state();

            println!("{:<6} {:<10.4} {:<10.4} {:<10.4} {:<10.4}",
                pos, sats[0], sats[1], sats[2], state.phase_coherence);
        }
    }

    // 압축 이벤트 요약
    println!("\n압축 이벤트: {}", reservoir.compaction_log.len());
    let scale_counts: Vec<usize> = (0..3)
        .map(|s| reservoir.compaction_log.iter().filter(|r| r.scale_index == s).count())
        .collect();
    println!("  τ=2:  {} events", scale_counts[0]);
    println!("  τ=5:  {} events", scale_counts[1]);
    println!("  τ=15: {} events", scale_counts[2]);

    // 평균 병합/프루닝 수
    if !reservoir.compaction_log.is_empty() {
        let avg_merged: f64 = reservoir.compaction_log.iter()
            .map(|r| r.channels_merged as f64).sum::<f64>()
            / reservoir.compaction_log.len() as f64;
        let avg_pruned: f64 = reservoir.compaction_log.iter()
            .map(|r| r.channels_pruned as f64).sum::<f64>()
            / reservoir.compaction_log.len() as f64;
        let avg_sat_drop: f64 = reservoir.compaction_log.iter()
            .map(|r| r.saturation_before - r.saturation_after).sum::<f64>()
            / reservoir.compaction_log.len() as f64;

        println!("\n이벤트당 평균:");
        println!("  병합 채널: {:.1}", avg_merged);
        println!("  프루닝 채널: {:.1}", avg_pruned);
        println!("  포화도 감소: {:.4}", avg_sat_drop);
    }
}

// ---------------------------------------------------------------------------
// 4. 속도 영향: 압축 ON vs OFF 처리 속도
// ---------------------------------------------------------------------------

#[test]
fn bench_compaction_speed_impact() {
    println!("\n=== 공명 정리: 속도 영향 측정 ===");

    let corpus = medium_corpus();
    let bpe = BpeTokenizer::from_corpus(&corpus, 200, 2000);

    let encoded: Vec<Vec<usize>> = corpus.iter()
        .map(|s| bpe.encode(s))
        .collect();

    let sps = SemanticPhaseSpace::from_encoded_corpus(
        bpe.size, &encoded, 64, 42,
    );

    let test_docs: Vec<Vec<usize>> = corpus[..30].iter()
        .map(|s| bpe.encode(s))
        .collect();
    let all_waves: Vec<Vec<_>> = test_docs.iter()
        .map(|ids| sps.tokens_to_waves(ids))
        .collect();

    // ─── OFF ───
    let mut res_off = WaveResonanceReservoir::new(64);
    let start = Instant::now();
    for waves in &all_waves {
        let _ = res_off.process_sequence_no_compaction(waves);
    }
    let time_off = start.elapsed();

    // ─── ON ───
    let mut res_on = WaveResonanceReservoir::new(64);
    let start = Instant::now();
    for waves in &all_waves {
        let _ = res_on.process_sequence(waves);
    }
    let time_on = start.elapsed();

    let off_ms = time_off.as_secs_f64() * 1000.0;
    let on_ms = time_on.as_secs_f64() * 1000.0;
    let overhead_pct = ((on_ms - off_ms) / off_ms) * 100.0;

    println!("문서 30개 처리:");
    println!("  OFF: {:.2}ms", off_ms);
    println!("  ON:  {:.2}ms", on_ms);
    println!("  오버헤드: {:.1}% ({:.2}ms)", overhead_pct, on_ms - off_ms);
    println!("  압축 이벤트: {}", res_on.compaction_log.len());
}

// ---------------------------------------------------------------------------
// 5. 스케일별 압축 효과 분석
// ---------------------------------------------------------------------------

#[test]
fn bench_compaction_per_scale_analysis() {
    println!("\n=== 공명 정리: 스케일별 효과 분석 ===");

    let corpus = medium_corpus();
    let bpe = BpeTokenizer::from_corpus(&corpus, 200, 2000);

    let encoded: Vec<Vec<usize>> = corpus.iter()
        .map(|s| bpe.encode(s))
        .collect();

    let sps = SemanticPhaseSpace::from_encoded_corpus(
        bpe.size, &encoded, 64, 42,
    );

    // 다양한 길이의 문장으로 테스트
    let short = "the cat sat";
    let medium = "the cat sat on the mat and the dog sat on the log";
    let long = "the cat sat on the mat and the dog sat on the log \
                and the bird flew over the tree and the boy went to \
                school and the girl read a book and the man walked \
                to the store and the sun shone in the sky and the \
                wind blew through the leaves";

    let tests = vec![("짧은(3tok)", short), ("중간(~15tok)", medium), ("긴(~40tok)", long)];

    println!("{:<15} {:<8} {:<10} {:<10} {:<10} {:<10}",
        "길이", "스케일", "이벤트", "병합(avg)", "포화전", "포화후");

    for (label, text) in &tests {
        let ids = bpe.encode(text);
        let waves = sps.tokens_to_waves(&ids);

        let mut reservoir = WaveResonanceReservoir::new(64);
        let _ = reservoir.process_sequence(&waves);

        for scale_idx in 0..3 {
            let scale_events: Vec<&CompactionResult> = reservoir.compaction_log.iter()
                .filter(|r| r.scale_index == scale_idx)
                .collect();

            let tau_label = match scale_idx {
                0 => "τ=2",
                1 => "τ=5",
                2 => "τ=15",
                _ => "?",
            };

            if scale_events.is_empty() {
                println!("{:<15} {:<8} {:<10} {:<10} {:<10} {:<10}",
                    label, tau_label, 0, "-", "-", "-");
            } else {
                let avg_merged: f64 = scale_events.iter()
                    .map(|r| r.channels_merged as f64).sum::<f64>()
                    / scale_events.len() as f64;
                let avg_sat_before: f64 = scale_events.iter()
                    .map(|r| r.saturation_before).sum::<f64>()
                    / scale_events.len() as f64;
                let avg_sat_after: f64 = scale_events.iter()
                    .map(|r| r.saturation_after).sum::<f64>()
                    / scale_events.len() as f64;

                println!("{:<15} {:<8} {:<10} {:<10.1} {:<10.4} {:<10.4}",
                    label, tau_label, scale_events.len(), avg_merged,
                    avg_sat_before, avg_sat_after);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 6. 엔진 수준 A/B: 전체 파이프라인 품질 비교
// ---------------------------------------------------------------------------

#[test]
fn bench_compaction_engine_quality_comparison() {
    println!("\n=== 공명 정리 A/B: 엔진 품질 비교 ===");

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

    // 자동완성 테스트
    let prompts = vec![
        "the cat",
        "the dog sat",
        "the boy went",
        "the sun",
        "the bird flew over",
    ];

    println!("\n[자동완성]");
    println!("{:<25} {:<12} {:<10} {:<10}",
        "프롬프트", "예측", "Ω", "confidence");

    for prompt in &prompts {
        let preds = engine.autocomplete(prompt, 1);
        if let Some(p) = preds.first() {
            println!("{:<25} {:<12} {:<10.4} {:<10.4}",
                prompt, p.token, p.omega, p.confidence);
        }
    }

    // 생성 테스트
    println!("\n[생성]");
    for prompt in &["the cat", "the boy"] {
        let result = engine.generate(prompt, 15);
        println!("  \"{}\" → \"{}\"", prompt, result.full_text);
        if let Some(last) = result.predictions.last() {
            println!("    (마지막 토큰 Ω={:.4}, Φ={:.4})", last.omega, last.phi);
        }
    }

    // 분류 테스트
    println!("\n[분류]");
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
    let mut engine_cls = WaveTextEngine::from_corpus_with_config(&corpus, &config);
    engine_cls.train_classifier(&labeled, labels);

    let test_cases = vec![
        ("the cat ran fast", "animals"),
        ("the boy ran home", "people"),
        ("the rain poured down", "weather"),
    ];

    for (text, expected) in &test_cases {
        if let Some(result) = engine_cls.classify(text) {
            println!("  \"{}\" → {} (expected: {}, Ω={:.4}, conf={:.4})",
                text, result.class_label, expected, result.omega, result.confidence);
        }
    }

    // 이상탐지 테스트
    println!("\n[이상탐지]");
    let normal_docs: Vec<&str> = corpus[..20].iter().copied().collect();
    let mut engine_anom = WaveTextEngine::from_corpus_with_config(&corpus, &config);
    engine_anom.train_anomaly_detector(&normal_docs);

    let normal_text = "the cat sat on the mat";
    let anomaly_text = "quantum entanglement drives blockchain synergy";

    if let Some(sn) = engine_anom.anomaly_score(normal_text) {
        println!("  정상: \"{}\" → score={:.4}, anomalous={}",
            normal_text, sn.score, sn.is_anomalous);
    }
    if let Some(sa) = engine_anom.anomaly_score(anomaly_text) {
        println!("  이상: \"{}\" → score={:.4}, anomalous={}",
            anomaly_text, sa.score, sa.is_anomalous);
    }

    // 지문 테스트
    println!("\n[지문]");
    let fp1 = engine.fingerprint(&corpus[..10]);
    let fp2 = engine.fingerprint(&corpus[..10]);
    let fp3 = engine.fingerprint(&corpus[50..60]);
    println!("  같은 문서 유사도: {:.4}", fp1.similarity(&fp2));
    println!("  다른 문서 유사도: {:.4}", fp1.similarity(&fp3));
}

// ---------------------------------------------------------------------------
// 7. 차원별 압축 효과 비교
// ---------------------------------------------------------------------------

#[test]
fn bench_compaction_dimension_comparison() {
    println!("\n=== 공명 정리: 차원별 효과 비교 ===");
    println!("{:<8} {:<12} {:<12} {:<12} {:<12} {:<12}",
        "dim", "이벤트수", "병합(avg)", "포화감소", "ON(ms)", "OFF(ms)");

    let corpus = tiny_corpus();

    for dim in [16, 32, 64] {
        let bpe = BpeTokenizer::from_corpus(&corpus, 100, 500);
        let encoded: Vec<Vec<usize>> = corpus.iter()
            .map(|s| bpe.encode(s))
            .collect();
        let sps = SemanticPhaseSpace::from_encoded_corpus(
            bpe.size, &encoded, dim, 42,
        );

        let all_waves: Vec<Vec<_>> = corpus.iter()
            .map(|s| {
                let ids = bpe.encode(s);
                sps.tokens_to_waves(&ids)
            })
            .collect();

        // ON
        let mut res_on = WaveResonanceReservoir::new(dim);
        let start = Instant::now();
        let mut sat_on = 0.0;
        for waves in &all_waves {
            let _ = res_on.process_sequence(waves);
            sat_on += res_on.registers.iter()
                .map(|r| r.saturation()).sum::<f64>() / 3.0;
        }
        let time_on = start.elapsed();
        let total_events = res_on.compaction_log.len();
        let avg_merged = if total_events > 0 {
            res_on.compaction_log.iter()
                .map(|r| r.channels_merged as f64).sum::<f64>()
                / total_events as f64
        } else { 0.0 };

        // OFF
        let mut res_off = WaveResonanceReservoir::new(dim);
        let start = Instant::now();
        let mut sat_off = 0.0;
        for waves in &all_waves {
            let _ = res_off.process_sequence_no_compaction(waves);
            sat_off += res_off.registers.iter()
                .map(|r| r.saturation()).sum::<f64>() / 3.0;
        }
        let time_off = start.elapsed();

        let sat_reduction = if sat_off > 0.0 {
            (1.0 - sat_on / sat_off) * 100.0
        } else { 0.0 };

        println!("{:<8} {:<12} {:<12.1} {:<12.1}% {:<12.2} {:<12.2}",
            dim, total_events, avg_merged, sat_reduction,
            time_on.as_secs_f64() * 1000.0,
            time_off.as_secs_f64() * 1000.0);
    }
}
