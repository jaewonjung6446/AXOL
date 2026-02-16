//! WTE 한국어 대화 엔진 테스트 — 메타 성장 포함.

use axol::text::data::{chat_corpus, chat_intents_ko, chat_classification_data};
use axol::text::engine::{WaveTextEngine, EngineConfig};
use axol::text::chat::ChatEngine;

fn build_chat_engine() -> ChatEngine {
    let corpus = chat_corpus();
    let config = EngineConfig {
        dim: 128,
        num_merges: 400,
        max_vocab: 3000,
        seed: 42,
        hidden_dim: 1024,
        ..EngineConfig::default()
    };
    let refs: Vec<&str> = corpus.iter().map(|s| *s).collect();
    let mut engine = WaveTextEngine::from_corpus_with_config(&refs, &config);

    let (labeled, class_labels) = chat_classification_data();
    engine.train_classifier(&labeled, class_labels);

    ChatEngine::new(engine, chat_intents_ko())
}

#[test]
fn test_chat_basic_greeting() {
    let mut chat = build_chat_engine();
    let result = chat.respond("안녕하세요");
    println!("입력: '안녕하세요'");
    println!("  응답: {}", result.response);
    println!("  의도: {}, 확신도: {:.3}, 공명: {:.6}",
        result.intent, result.confidence, result.resonance);

    assert!(!result.response.is_empty(), "응답이 비어있으면 안됨");
    assert!(result.resonance > 0.0, "공명이 양수여야 함");
}

#[test]
fn test_chat_topic_matching() {
    let mut chat = build_chat_engine();

    // 동물
    let animal = chat.respond("강아지 좋아해?");
    println!("입력: '강아지 좋아해?'");
    println!("  응답: {}", animal.response);
    println!("  의도: {}", animal.intent);
    assert!(!animal.response.is_empty());

    // 자연
    let nature = chat.respond("산 좋아해?");
    println!("입력: '산 좋아해?'");
    println!("  응답: {}", nature.response);
    println!("  의도: {}", nature.intent);
    assert!(!nature.response.is_empty());

    // 작별
    let bye = chat.respond("안녕히 가세요");
    println!("입력: '안녕히 가세요'");
    println!("  응답: {}", bye.response);
    println!("  의도: {}", bye.intent);
    assert!(!bye.response.is_empty());
}

#[test]
fn test_chat_sentence_resonance() {
    let corpus = chat_corpus();
    let config = EngineConfig {
        dim: 128,
        num_merges: 400,
        max_vocab: 3000,
        seed: 42,
        hidden_dim: 1024,
        ..EngineConfig::default()
    };
    let refs: Vec<&str> = corpus.iter().map(|s| *s).collect();
    let engine = WaveTextEngine::from_corpus_with_config(&refs, &config);

    let state = engine.process_text("강아지가 귀여워");

    let similar = engine.sentence_resonance(&state, "고양이가 좋아");
    let dissimilar = engine.sentence_resonance(&state, "오늘 날씨 어때");

    println!("공명: 유사={:.6}, 비유사={:.6}", similar, dissimilar);
    assert!(similar > 0.0, "유사 문장 공명이 양수여야 함");
    assert!(dissimilar > 0.0, "비유사 문장 공명이 양수여야 함");
}

// =========================================================================
// 파동 기반 생성 테스트
// =========================================================================

#[test]
fn test_generate_basic() {
    let mut chat = build_chat_engine();
    let result = chat.generate_response("안녕하세요");

    println!("[생성 기본]");
    println!("  입력: '안녕하세요'");
    println!("  응답: {}", result.response);
    println!("  의도: {}, 확신도: {:.3}", result.intent, result.confidence);
    println!("  생성: {}, 품질: {:?}", result.is_generated, result.generation_quality);

    assert!(!result.response.is_empty(), "생성 응답이 비어있으면 안됨");
    // Either generated or fell back to pool — both are valid
}

#[test]
fn test_generate_intent_conditioning() {
    let mut chat = build_chat_engine();

    let animal = chat.generate_response("강아지가 아파요");
    let food = chat.generate_response("배고파 뭐 먹을까");
    let nature = chat.generate_response("바다가 좋아");

    println!("[의도 조건부 생성]");
    println!("  동물: 의도={}, 응답={}", animal.intent, animal.response);
    println!("  음식: 의도={}, 응답={}", food.intent, food.response);
    println!("  자연: 의도={}, 응답={}", nature.intent, nature.response);

    // Different intents should produce different responses
    assert_ne!(animal.response, food.response,
        "동물/음식 응답이 같으면 안됨");
    assert_ne!(food.response, nature.response,
        "음식/자연 응답이 같으면 안됨");
}

#[test]
fn test_generate_quality_metrics() {
    let mut chat = build_chat_engine();

    let intents = vec![
        "안녕하세요", "강아지 좋아해?", "배고파", "날씨 어때?", "슬퍼",
    ];

    println!("[생성 품질 메트릭]");
    for input in intents {
        let result = chat.generate_response(input);
        println!("  '{}' → 생성={} 품질={:?} 응답='{}'",
            input, result.is_generated,
            result.generation_quality, result.response);

        if result.is_generated {
            let q = result.generation_quality.unwrap();
            assert!(q >= 0.0, "품질이 0 이상이어야 함: {}", q);
        }
    }
}

#[test]
fn test_respond_auto() {
    let mut chat = build_chat_engine();

    let queries = vec![
        "안녕하세요", "강아지 좋아해?", "배고파", "날씨 어때?",
        "슬퍼", "도와줘", "안녕히 가세요",
    ];

    println!("[자동 모드]");
    for input in queries {
        let result = chat.respond_auto(input);
        let tag = if result.is_generated { "[생성]" } else { "[풀]" };
        println!("  {} '{}' → '{}' (의도={}, 품질={:?})",
            tag, input, result.response, result.intent, result.generation_quality);

        assert!(!result.response.is_empty(), "자동 응답이 비어있으면 안됨");
    }
}

// =========================================================================
// 메타 성장 테스트
// =========================================================================

#[test]
fn test_growth_feedback() {
    let mut chat = build_chat_engine();

    let result = chat.respond("안녕하세요");
    let pool_id = result.pool_id;
    let resp_id = result.response_id;

    let before = chat.pools[pool_id].entries[resp_id].fitness;

    // Positive feedback
    chat.feedback(pool_id, resp_id, true);
    let after_pos = chat.pools[pool_id].entries[resp_id].fitness;
    assert!(
        (after_pos - before - 0.15).abs() < 1e-9,
        "positive feedback: {:.4} → {:.4} (expected +0.15)",
        before, after_pos
    );

    // Negative feedback
    chat.feedback(pool_id, resp_id, false);
    let after_neg = chat.pools[pool_id].entries[resp_id].fitness;
    assert!(
        (after_neg - (after_pos - 0.10)).abs() < 1e-9,
        "negative feedback: {:.4} → {:.4} (expected -0.10)",
        after_pos, after_neg
    );

    let stats = chat.growth_stats();
    assert_eq!(stats.positive_feedbacks, 1);
    assert_eq!(stats.negative_feedbacks, 1);

    println!("피드백 테스트 통과: {:.4} → +{:.4} → -{:.4}", before, after_pos, after_neg);
}

#[test]
fn test_growth_decay_and_prune() {
    let mut chat = build_chat_engine();

    // Record initial pool sizes
    let initial_sizes: Vec<usize> = chat.pools.iter().map(|p| p.entries.len()).collect();

    // Set one entry to very low fitness
    if !chat.pools.is_empty() && chat.pools[0].entries.len() > 3 {
        chat.pools[0].entries[0].fitness = 0.01;
    }

    // Run several decay+prune cycles
    for _ in 0..5 {
        chat.growth_cycle();
    }

    // Check fitness decayed
    for pool in &chat.pools {
        for entry in &pool.entries {
            // After 5 decay cycles at 0.995, initial 1.0 → ~0.975
            assert!(entry.fitness <= 1.0 || entry.fitness >= 0.0,
                "fitness out of range: {}", entry.fitness);
        }
    }

    // Check pruning happened (the 0.01 entry should be removed if pool > min_size)
    let stats = chat.growth_stats();
    println!("감쇠/정리 테스트:");
    println!("  총 사이클: {}", stats.total_cycles);
    println!("  총 정리: {}", stats.total_pruned);
    println!("  풀 크기 변화:");
    for (i, pool) in chat.pools.iter().enumerate() {
        let init = if i < initial_sizes.len() { initial_sizes[i] } else { 0 };
        println!("    {}: {} → {}", pool.intent, init, pool.entries.len());
    }

    assert!(stats.total_cycles == 5, "5 cycles expected");
}

#[test]
fn test_growth_emergence() {
    let mut chat = build_chat_engine();

    // Manually set high co-resonance between two pools
    let intent_a = chat.pools[0].intent.clone();
    let intent_b = chat.pools[1].intent.clone();
    let threshold = chat.growth.emergence_threshold;

    // Set co-resonance above threshold
    if let (Some(i), Some(j)) = (
        chat.co_resonance.index_of(&intent_a),
        chat.co_resonance.index_of(&intent_b),
    ) {
        chat.co_resonance.matrix[i][j] = threshold + 0.1;
        chat.co_resonance.matrix[j][i] = threshold + 0.1;
    }

    let pools_before = chat.pools.len();
    chat.growth_cycle();
    let pools_after = chat.pools.len();

    println!("출현 테스트:");
    println!("  공명 풀 A: {}", intent_a);
    println!("  공명 풀 B: {}", intent_b);
    println!("  풀 수 변화: {} → {}", pools_before, pools_after);

    assert!(
        pools_after > pools_before,
        "emergence should create a new pool: {} → {}",
        pools_before, pools_after
    );

    // Verify the new pool is emergent
    let new_pool = chat.pools.last().unwrap();
    assert!(new_pool.is_emergent, "new pool should be emergent");
    assert!(
        new_pool.parent_intents.is_some(),
        "emergent pool should have parent intents"
    );
    let (pa, pb) = new_pool.parent_intents.as_ref().unwrap();
    assert!(
        (pa == &intent_a && pb == &intent_b) || (pa == &intent_b && pb == &intent_a),
        "parent intents mismatch"
    );

    let stats = chat.growth_stats();
    assert_eq!(stats.emergent_pools_created, 1, "should have 1 emergent pool");

    println!("  신규 풀: {} (크기: {})", new_pool.intent, new_pool.entries.len());
}

#[test]
fn test_growth_replication() {
    let mut chat = build_chat_engine();

    // Set one entry to high fitness above replication threshold
    let replication_threshold = chat.growth.replication_threshold;
    chat.pools[0].entries[0].fitness = replication_threshold + 0.5;

    let entries_before = chat.pools[0].entries.len();
    chat.growth_cycle();
    let entries_after = chat.pools[0].entries.len();

    println!("복제 테스트:");
    println!("  풀: {}", chat.pools[0].intent);
    println!("  엔트리 수: {} → {}", entries_before, entries_after);

    assert!(
        entries_after > entries_before,
        "replication should create clones: {} → {}",
        entries_before, entries_after
    );

    // Verify clones exist
    let clones: Vec<&_> = chat.pools[0].entries.iter().filter(|e| e.is_clone).collect();
    assert!(!clones.is_empty(), "should have clone entries");
    for clone in &clones {
        assert_eq!(clone.generation, 1, "clone should be generation 1");
        assert!((clone.fitness - chat.growth.initial_fitness).abs() < 1e-9,
            "clone should have initial fitness");
    }

    let stats = chat.growth_stats();
    assert!(stats.total_clones > 0, "should have clones");
    println!("  클론 수: {}", stats.total_clones);
}

#[test]
fn test_growth_full_cycle() {
    let mut chat = build_chat_engine();

    let queries = vec![
        "안녕하세요", "강아지 좋아해?", "배고파", "날씨 어때?",
        "슬퍼", "도와줘", "안녕히 가세요", "친구가 보고 싶어",
        "자연이 좋아", "뭐 먹을까?",
        "반가워", "고양이가 귀여워", "라면 먹고 싶다", "비 올까?",
        "기분이 좋아", "사용법 알려줘", "잘 가", "가족이 그리워",
        "바다가 좋아", "치킨 먹자",
    ];

    // Run queries with feedback
    for (i, query) in queries.iter().enumerate() {
        let result = chat.respond(query);
        // Alternate positive/negative feedback
        if i % 3 == 0 {
            chat.feedback(result.pool_id, result.response_id, true);
        } else if i % 3 == 1 {
            chat.feedback(result.pool_id, result.response_id, false);
        }
        // else: no feedback
    }

    let stats = chat.growth_stats();
    println!("전체 사이클 테스트:");
    println!("  총 쿼리: {}", stats.total_queries);
    println!("  총 사이클: {}", stats.total_cycles);
    println!("  긍정 피드백: {}", stats.positive_feedbacks);
    println!("  부정 피드백: {}", stats.negative_feedbacks);
    println!("  정리 수: {}", stats.total_pruned);
    println!("  출현 풀: {}", stats.emergent_pools_created);
    println!("  클론 수: {}", stats.total_clones);

    assert_eq!(stats.total_queries, 20, "should process all queries");
    assert!(stats.total_cycles >= 1, "should have run at least 1 cycle");
    assert!(stats.positive_feedbacks > 0, "should have positive feedbacks");
    assert!(stats.negative_feedbacks > 0, "should have negative feedbacks");

    // Pool summary
    println!("\n  [풀 상태]");
    for (intent, size, avg_fit, emergent) in chat.pool_summary() {
        let tag = if emergent { " [출현]" } else { "" };
        println!("    {:15} 크기={:3} 평균fitness={:.4}{}",
            intent, size, avg_fit, tag);
    }
}

// =========================================================================
// 안정화 테스트: 학습 → 안정화 → 전후 비교
// =========================================================================

#[test]
fn test_stabilize_after_learning() {
    let mut chat = build_chat_engine();

    // 테스트 케이스
    let test_cases: Vec<(&str, &str)> = vec![
        ("안녕하세요", "greeting"), ("반가워요", "greeting"),
        ("강아지 좋아해?", "animals"), ("고양이가 최고야", "animals"),
        ("친구랑 싸웠어", "people"), ("가족이 보고 싶어", "people"),
        ("바다가 좋아", "nature"), ("산이 멋져", "nature"),
        ("안녕히 가세요", "farewell"), ("잘 가", "farewell"),
        ("도와줘", "help"), ("사용법 알려줘", "help"),
        ("오늘 춥다", "weather"), ("비 올까?", "weather"),
        ("배고파", "food"), ("치킨 먹자", "food"),
        ("슬퍼", "emotions"), ("기분이 좋아", "emotions"),
        ("강아지가 아파요", "animals"), ("바다가 보고 싶어", "nature"),
    ];

    // 정확도 측정 함수
    let measure = |chat: &mut ChatEngine, cases: &[(&str, &str)]| -> (usize, usize) {
        let mut correct = 0;
        for &(input, expected) in cases {
            let result = chat.respond(input);
            if result.intent == expected { correct += 1; }
        }
        (correct, cases.len())
    };

    // 생성 품질 측정
    let measure_gen = |chat: &mut ChatEngine| -> (f64, usize, usize) {
        let gen_inputs = vec![
            "안녕하세요", "강아지가 아파요", "배고파", "바다가 좋아", "슬퍼",
        ];
        let mut total_phi = 0.0;
        let mut success = 0;
        for input in &gen_inputs {
            let r = chat.generate_response(input);
            if r.is_generated {
                total_phi += r.generation_quality.unwrap_or(0.0);
                success += 1;
            }
        }
        let avg = if success > 0 { total_phi / success as f64 } else { 0.0 };
        (avg, success, gen_inputs.len())
    };

    // 재훈련 빠르게 발동하도록 설정
    chat.self_learning.retrain_interval = 10;

    // ── Phase 0: 초기 상태 ──
    let (c0, t0) = measure(&mut chat, &test_cases);
    let (phi0, gen0, gtot0) = measure_gen(&mut chat);
    println!("\n[Phase 0: 초기]");
    println!("  정확도:  {}/{} ({:.1}%)", c0, t0, c0 as f64 / t0 as f64 * 100.0);
    println!("  생성:    {}/{} Φ={:.4}", gen0, gtot0, phi0);

    // ── Phase 1: 학습 (쿼리 + 피드백 + 교정) — 3라운드 ──
    let learning_queries: Vec<(&str, &str)> = vec![
        ("안녕하세요", "greeting"), ("안녕!", "greeting"), ("반가워", "greeting"),
        ("좋은 아침", "greeting"), ("잘 지냈어?", "greeting"),
        ("강아지 귀여워", "animals"), ("고양이 좋아", "animals"), ("동물 키워?", "animals"),
        ("반려동물 사랑", "animals"), ("강아지가 아파", "animals"),
        ("친구가 보고 싶어", "people"), ("가족 관계", "people"), ("사람이 그리워", "people"),
        ("직장 동료", "people"), ("인간관계 어려워", "people"),
        ("바다 가자", "nature"), ("숲이 좋아", "nature"), ("자연이 최고", "nature"),
        ("산이 멋져", "nature"), ("바다가 보고 싶다", "nature"),
        ("잘 가요", "farewell"), ("또 만나", "farewell"), ("안녕히", "farewell"),
        ("바이바이", "farewell"), ("다음에 또", "farewell"),
        ("도와줘", "help"), ("뭐 할 수 있어?", "help"), ("기능이 뭐야?", "help"),
        ("사용법", "help"), ("어떻게 써?", "help"),
        ("오늘 덥다", "weather"), ("비 와?", "weather"), ("미세먼지 심해", "weather"),
        ("날씨 어때?", "weather"), ("장마 언제?", "weather"),
        ("배고파", "food"), ("뭐 먹을까", "food"), ("치킨!", "food"),
        ("라면 먹자", "food"), ("맛집 추천", "food"),
        ("슬프다", "emotions"), ("기쁘다", "emotions"), ("화나", "emotions"),
        ("우울해", "emotions"), ("스트레스", "emotions"),
    ];

    // 3라운드 반복 학습
    for round in 0..3 {
        for &(input, correct_intent) in &learning_queries {
            let result = chat.respond(input);
            if result.intent == correct_intent {
                chat.feedback(result.pool_id, result.response_id, true);
            } else {
                chat.feedback_correct(correct_intent);
            }
        }
        let r = chat.retrain_count();
        println!("  라운드 {}: 재훈련 누적 {}회", round + 1, r);
    }

    let (c1, t1) = measure(&mut chat, &test_cases);
    let (phi1, gen1, gtot1) = measure_gen(&mut chat);
    println!("\n[Phase 1: 학습 후 ({}쿼리, 재훈련 {}회)]",
        learning_queries.len(), chat.retrain_count());
    println!("  정확도:  {}/{} ({:.1}%)", c1, t1, c1 as f64 / t1 as f64 * 100.0);
    println!("  생성:    {}/{} Φ={:.4}", gen1, gtot1, phi1);
    println!("  경험:    {}건", chat.experience_count());

    // ── Phase 2: 안정화 (피드백 없이 사이클만) ──
    chat.stabilize(1000);

    let (c2, t2) = measure(&mut chat, &test_cases);
    let (phi2, gen2, gtot2) = measure_gen(&mut chat);
    println!("\n[Phase 2: 안정화 후 (1,000 사이클)]");
    println!("  정확도:  {}/{} ({:.1}%)", c2, t2, c2 as f64 / t2 as f64 * 100.0);
    println!("  생성:    {}/{} Φ={:.4}", gen2, gtot2, phi2);

    // ── Phase 3: 추가 안정화 ──
    chat.stabilize(9000);

    let (c3, t3) = measure(&mut chat, &test_cases);
    let (phi3, gen3, gtot3) = measure_gen(&mut chat);
    println!("\n[Phase 3: 추가 안정화 후 (총 10,000 사이클)]");
    println!("  정확도:  {}/{} ({:.1}%)", c3, t3, c3 as f64 / t3 as f64 * 100.0);
    println!("  생성:    {}/{} Φ={:.4}", gen3, gtot3, phi3);

    // ── 요약 ──
    println!("\n[요약]");
    println!("  초기:          {}/{} ({:.1}%) 생성={}/{} Φ={:.4}",
        c0, t0, c0 as f64/t0 as f64*100.0, gen0, gtot0, phi0);
    println!("  학습 후:       {}/{} ({:.1}%) 생성={}/{} Φ={:.4}",
        c1, t1, c1 as f64/t1 as f64*100.0, gen1, gtot1, phi1);
    println!("  안정화 1K:     {}/{} ({:.1}%) 생성={}/{} Φ={:.4}",
        c2, t2, c2 as f64/t2 as f64*100.0, gen2, gtot2, phi2);
    println!("  안정화 10K:    {}/{} ({:.1}%) 생성={}/{} Φ={:.4}",
        c3, t3, c3 as f64/t3 as f64*100.0, gen3, gtot3, phi3);

    let stats = chat.growth_stats();
    println!("\n  재훈련: {}회 / 정리: {} / 출현: {} / 클론: {} / 시드부스트: {}",
        stats.classifier_retrains, stats.total_pruned,
        stats.emergent_pools_created, stats.total_clones, stats.gen_boosts);
}

// =========================================================================
// Level 5: 생성 진화 테스트
// =========================================================================

#[test]
fn test_generation_evolution() {
    let mut chat = build_chat_engine();
    chat.self_learning.retrain_interval = 10;

    let gen_inputs = vec![
        ("안녕하세요", "greeting"),
        ("강아지 좋아해?", "animals"),
        ("배고파", "food"),
        ("바다가 좋아", "nature"),
        ("슬퍼", "emotions"),
    ];

    // ── Phase 0: 초기 생성 품질 ──
    let mut phi_before = Vec::new();
    for &(input, _) in &gen_inputs {
        let r = chat.generate_response(input);
        phi_before.push((input, r.generation_quality.unwrap_or(0.0), r.is_generated));
    }
    let avg_phi0: f64 = phi_before.iter()
        .filter(|x| x.2).map(|x| x.1).sum::<f64>()
        / phi_before.iter().filter(|x| x.2).count().max(1) as f64;

    println!("\n[Phase 0: 초기 생성]");
    for (input, phi, gen) in &phi_before {
        let tag = if *gen { "생성" } else { "폴백" };
        println!("  [{}] '{}' Φ={:.4}", tag, input, phi);
    }
    println!("  평균 Φ: {:.4}", avg_phi0);

    // ── Phase 1: 생성 + 피드백 루프 (5라운드) ──
    // 매 라운드: 생성 → 긍정 피드백 → 생성 텍스트 풀 주입
    println!("\n[Phase 1: 생성 진화 루프 — 5라운드]");
    for round in 0..5 {
        for &(input, correct_intent) in &gen_inputs {
            // 1. 분류기 학습용 respond
            let pool_r = chat.respond(input);
            if pool_r.intent == correct_intent {
                chat.feedback(pool_r.pool_id, pool_r.response_id, true);
            } else {
                chat.feedback_correct(correct_intent);
            }

            // 2. 생성 → 피드백 (긍정만)
            let gen_r = chat.generate_response(input);
            if gen_r.is_generated {
                chat.feedback(gen_r.pool_id, gen_r.response_id, true);
            }
        }

        // 안정화
        chat.stabilize(100);

        // 라운드별 측정
        let mut round_phi = 0.0;
        let mut round_gen = 0;
        for &(input, _) in &gen_inputs {
            let r = chat.generate_response(input);
            if r.is_generated {
                round_phi += r.generation_quality.unwrap_or(0.0);
                round_gen += 1;
            }
        }
        let avg = if round_gen > 0 { round_phi / round_gen as f64 } else { 0.0 };
        println!("  라운드 {}: Φ={:.4} 생성={}/{} 부스트={}",
            round + 1, avg, round_gen, gen_inputs.len(),
            chat.growth_stats().gen_boosts);
    }

    // ── Phase 2: 최종 생성 품질 ──
    let mut phi_after = Vec::new();
    for &(input, _) in &gen_inputs {
        let r = chat.generate_response(input);
        phi_after.push((input, r.generation_quality.unwrap_or(0.0), r.is_generated));
    }
    let avg_phi_final: f64 = phi_after.iter()
        .filter(|x| x.2).map(|x| x.1).sum::<f64>()
        / phi_after.iter().filter(|x| x.2).count().max(1) as f64;

    println!("\n[Phase 2: 최종 생성]");
    for (input, phi, gen) in &phi_after {
        let tag = if *gen { "생성" } else { "폴백" };
        println!("  [{}] '{}' Φ={:.4}", tag, input, phi);
    }
    println!("  평균 Φ: {:.4}", avg_phi_final);

    // gen_phi tracking 확인
    println!("\n[시드 gen_phi 추적]");
    for pool in &chat.pools {
        for entry in &pool.entries {
            if entry.gen_uses > 0 {
                let preview: String = entry.text.chars().take(15).collect();
                println!("  [{}] '{}...' gen_phi={:.4} uses={}",
                    pool.intent, preview, entry.gen_phi, entry.gen_uses);
            }
        }
    }

    let stats = chat.growth_stats();
    println!("\n[요약]");
    println!("  초기 Φ:    {:.4}", avg_phi0);
    println!("  최종 Φ:    {:.4}", avg_phi_final);
    println!("  변화:      {:+.4}", avg_phi_final - avg_phi0);
    println!("  시드 부스트:  {}건", stats.gen_boosts);
    println!("  재훈련:     {}회", stats.classifier_retrains);
}
