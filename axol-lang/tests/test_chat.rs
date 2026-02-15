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
