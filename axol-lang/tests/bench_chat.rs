//! WTE 한국어 대화 엔진 — 정확도 & 속도 & 메타 성장 벤치마크.

use axol::text::data::{chat_corpus, chat_intents_ko, chat_classification_data};
use axol::text::engine::{WaveTextEngine, EngineConfig};
use axol::text::chat::{ChatEngine, GrowthConfig};
use std::time::Instant;

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

fn korean_test_cases() -> Vec<(&'static str, &'static str)> {
    vec![
        // greeting
        ("안녕하세요", "greeting"),
        ("반가워요", "greeting"),
        ("좋은 아침이에요", "greeting"),
        ("잘 지냈어?", "greeting"),
        ("오랜만이야", "greeting"),
        ("처음 뵙겠습니다", "greeting"),
        ("안녕!", "greeting"),
        // animals
        ("강아지 좋아해?", "animals"),
        ("고양이가 최고야", "animals"),
        ("동물에 대해 알려줘", "animals"),
        ("반려동물 키워?", "animals"),
        ("동물원 가본 적 있어?", "animals"),
        ("우리 집 강아지가 아파", "animals"),
        ("고양이 사료 추천해줘", "animals"),
        // people
        ("친구랑 싸웠어", "people"),
        ("사람 관계가 어려워", "people"),
        ("인간관계 조언 좀 해줘", "people"),
        ("가족 관계가 힘들어", "people"),
        ("사회생활이 힘들어", "people"),
        ("직장 동료와 갈등이 있어", "people"),
        ("부모님이 보고 싶어", "people"),
        // nature
        ("자연이 좋아", "nature"),
        ("산 좋아해?", "nature"),
        ("바다 좋아하세요?", "nature"),
        ("별 보러 가고 싶어", "nature"),
        ("한라산 올라가 봤어?", "nature"),
        ("숲속이 정말 좋다", "nature"),
        ("꽃이 예쁘게 폈어", "nature"),
        // farewell
        ("안녕히 가세요", "farewell"),
        ("잘 가", "farewell"),
        ("바이바이", "farewell"),
        ("또 봐요", "farewell"),
        ("잘 자요", "farewell"),
        ("다음에 또 이야기해요", "farewell"),
        ("오늘 재미있었어", "farewell"),
        // help
        ("도와줘", "help"),
        ("뭐 할 수 있어?", "help"),
        ("사용법 알려줘", "help"),
        ("너는 누구야?", "help"),
        ("어떻게 시작하면 돼?", "help"),
        ("기능이 뭐가 있어?", "help"),
        ("설명 좀 해줘", "help"),
        // weather
        ("오늘 날씨 어때?", "weather"),
        ("비 올까?", "weather"),
        ("오늘 춥다", "weather"),
        ("미세먼지 심해?", "weather"),
        ("장마 언제야?", "weather"),
        ("내일 날씨 알려줘", "weather"),
        ("오늘 덥다", "weather"),
        // food
        ("배고파", "food"),
        ("뭐 먹을까?", "food"),
        ("치킨 좋아해?", "food"),
        ("라면 먹고 싶다", "food"),
        ("맛집 알려줘", "food"),
        ("점심 뭐 먹지?", "food"),
        ("디저트 추천해줘", "food"),
        // emotions
        ("기분이 좋아", "emotions"),
        ("슬퍼", "emotions"),
        ("화가 나", "emotions"),
        ("스트레스 받아", "emotions"),
        ("힘들어", "emotions"),
        ("오늘 기분이 최고야", "emotions"),
        ("우울해", "emotions"),
        // unknown
        ("ㅁㄴㅇㄹ", "unknown"),
        ("asdfghjkl", "unknown"),
    ]
}

#[test]
fn bench_chat_accuracy_and_speed() {
    println!("\n======================================================================");
    println!("  WTE 한국어 대화 엔진 — 정확도 & 속도 & 메타 성장 벤치마크");
    println!("======================================================================\n");

    // ── 빌드 ──
    let t0 = Instant::now();
    let mut chat = build_chat_engine();
    let build_time = t0.elapsed();
    println!("[빌드]");
    println!("  엔진 빌드 시간:  {:.1}ms", build_time.as_secs_f64() * 1000.0);
    println!("  응답 풀:         {}", chat.pools.len());
    let total_responses: usize = chat.pools.iter().map(|p| p.entries.len()).sum();
    println!("  총 응답:         {}", total_responses);
    println!();

    // ── 정확도 ──
    let cases = korean_test_cases();
    let mut correct = 0;
    let mut total = 0;
    let mut times = Vec::new();
    let mut details = Vec::new();

    println!("[정확도]");
    for (input, expected) in &cases {
        let t = Instant::now();
        let result = chat.respond(input);
        let elapsed = t.elapsed();
        times.push(elapsed.as_secs_f64() * 1000.0);

        let ok = result.intent == *expected;
        if ok { correct += 1; }
        total += 1;

        let mark = if ok { "OK" } else { "MISS" };
        details.push(format!(
            "  [{:4}] {:30} → 의도={:10} 기대={:10} 확신={:.3} 공명={:.4} ({:.1}ms)",
            mark, input, result.intent, expected, result.confidence, result.resonance,
            elapsed.as_secs_f64() * 1000.0,
        ));
    }
    for d in &details { println!("{}", d); }
    let acc = correct as f64 / total as f64 * 100.0;
    println!("  ─────────────────────────────────────────────");
    println!("  정확도: {}/{} ({:.1}%)", correct, total, acc);
    println!();

    // ── 속도 ──
    let avg = times.iter().sum::<f64>() / times.len() as f64;
    let min = times.iter().cloned().fold(f64::MAX, f64::min);
    let max = times.iter().cloned().fold(0.0_f64, f64::max);
    let mut sorted = times.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = sorted[sorted.len() / 2];
    let p95 = sorted[(sorted.len() as f64 * 0.95) as usize];
    let p99 = sorted[(sorted.len() as f64 * 0.99) as usize];

    println!("[속도]");
    println!("  쿼리 수:     {}", times.len());
    println!("  평균 지연:   {:.2}ms", avg);
    println!("  최소 지연:   {:.2}ms", min);
    println!("  최대 지연:   {:.2}ms", max);
    println!("  P50 지연:    {:.2}ms", p50);
    println!("  P95 지연:    {:.2}ms", p95);
    println!("  P99 지연:    {:.2}ms", p99);
    println!("  처리량:      {:.0} queries/sec", 1000.0 / avg);
    println!();

    // ── 의도별 정확도 ──
    println!("[의도별 정확도]");
    let intents = ["greeting", "animals", "people", "nature", "farewell",
                    "help", "weather", "food", "emotions", "unknown"];
    for intent in &intents {
        let intent_total = cases.iter().filter(|(_, e)| e == intent).count();
        let intent_correct = cases.iter().zip(details.iter()).filter(|((_, e), d)| {
            e == intent && d.contains("[OK  ]")
        }).count();
        if intent_total > 0 {
            println!("  {:12} {}/{} ({:.0}%)",
                intent, intent_correct, intent_total,
                intent_correct as f64 / intent_total as f64 * 100.0);
        }
    }
    println!();

    // ── 분류기 단독 정확도 ──
    println!("[분류기 단독 정확도]");
    let mut clf_correct = 0;
    let mut clf_total = 0;
    for (input, expected) in &cases {
        if let Some(result) = chat.engine.classify(input) {
            if result.class_label == *expected {
                clf_correct += 1;
            }
        }
        clf_total += 1;
    }
    println!("  분류기 정확도: {}/{} ({:.1}%)",
        clf_correct, clf_total, clf_correct as f64 / clf_total as f64 * 100.0);
    println!();

    // ── 공명 변별력 ──
    println!("[공명 변별력]");
    let pairs = vec![
        ("안녕하세요 반갑습니다", "좋은 아침이에요", "ㅁㄴㅇㄹ asdf"),
        ("강아지가 귀여워", "고양이가 좋아", "오늘 날씨 어때"),
        ("배고파 뭐 먹을까", "치킨 먹고 싶다", "친구랑 싸웠어"),
    ];
    for (a, similar, dissimilar) in &pairs {
        let state = chat.engine.process_text(a);
        let sim_r = chat.engine.sentence_resonance(&state, similar);
        let dis_r = chat.engine.sentence_resonance(&state, dissimilar);
        let ratio = if dis_r > 1e-10 { sim_r / dis_r } else { f64::INFINITY };
        println!("  '{}' vs 유사={:.4} 비유사={:.4} 비율={:.2}x",
            a, sim_r, dis_r, ratio);
    }
    println!();

    // ══════════════════════════════════════════════════════════════════════
    // 파동 기반 생성 벤치마크
    // ══════════════════════════════════════════════════════════════════════

    println!("======================================================================");
    println!("  [생성]");
    println!("======================================================================\n");

    let gen_inputs: Vec<(&str, &str)> = vec![
        ("안녕하세요", "greeting"),
        ("강아지가 아파요", "animals"),
        ("친구랑 싸웠어", "people"),
        ("바다가 좋아", "nature"),
        ("안녕히 가세요", "farewell"),
        ("도와줘", "help"),
        ("오늘 날씨 어때?", "weather"),
        ("배고파", "food"),
        ("기분이 좋아", "emotions"),
        ("ㅁㄴㅇㄹ", "unknown"),
    ];

    println!("[의도별 생성]");
    let mut gen_times = Vec::new();
    let mut gen_count = 0;
    let mut gen_quality_sum = 0.0;

    for (input, expected) in &gen_inputs {
        let t = Instant::now();
        let result = chat.generate_response(input);
        let elapsed = t.elapsed().as_secs_f64() * 1000.0;
        gen_times.push(elapsed);

        let tag = if result.is_generated { "생성" } else { "폴백" };
        let q = result.generation_quality.unwrap_or(0.0);
        if result.is_generated {
            gen_count += 1;
            gen_quality_sum += q;
        }

        println!("  [{:4}] {:20} → 의도={:10} 기대={:10} Φ={:.4} ({:.1}ms) '{}'",
            tag, input, result.intent, expected, q, elapsed,
            if result.response.len() > 40 {
                format!("{}...", &result.response[..result.response.char_indices()
                    .nth(40).map_or(result.response.len(), |(i, _)| i)])
            } else {
                result.response.clone()
            });
    }

    let gen_avg_time = gen_times.iter().sum::<f64>() / gen_times.len() as f64;
    let gen_avg_quality = if gen_count > 0 { gen_quality_sum / gen_count as f64 } else { 0.0 };

    println!();
    println!("[생성 통계]");
    println!("  생성 성공:     {}/{}", gen_count, gen_inputs.len());
    println!("  평균 Φ 품질:   {:.4}", gen_avg_quality);
    println!("  평균 생성 시간: {:.2}ms", gen_avg_time);
    println!();

    // 생성 vs 풀선택 비교
    println!("[생성 vs 풀선택 비교]");
    for (input, _) in gen_inputs.iter().take(5) {
        let t_pool = Instant::now();
        let pool_result = chat.respond(input);
        let pool_ms = t_pool.elapsed().as_secs_f64() * 1000.0;

        let t_gen = Instant::now();
        let gen_result = chat.generate_response(input);
        let gen_ms = t_gen.elapsed().as_secs_f64() * 1000.0;

        let gen_tag = if gen_result.is_generated { "생성" } else { "폴백" };
        println!("  '{}': 풀={:.1}ms vs 생성={:.1}ms [{}]",
            input, pool_ms, gen_ms, gen_tag);
        println!("    풀: '{}'", pool_result.response);
        println!("    생성: '{}'", gen_result.response);
    }
    println!();

    // ══════════════════════════════════════════════════════════════════════
    // 메타 성장 벤치마크
    // ══════════════════════════════════════════════════════════════════════

    println!("======================================================================");
    println!("  [메타 성장]");
    println!("======================================================================\n");

    // 성장 전 정확도 기록
    let acc_before = acc;

    // 100회 쿼리 + 랜덤 피드백 시뮬레이션
    let growth_queries: Vec<&str> = vec![
        "안녕하세요", "반가워요", "강아지 좋아해?", "고양이가 최고야",
        "친구랑 싸웠어", "사람 관계가 어려워", "자연이 좋아", "산 좋아해?",
        "안녕히 가세요", "잘 가", "도와줘", "뭐 할 수 있어?",
        "오늘 날씨 어때?", "비 올까?", "배고파", "뭐 먹을까?",
        "기분이 좋아", "슬퍼", "좋은 아침이에요", "잘 지냈어?",
        "동물에 대해 알려줘", "반려동물 키워?", "인간관계 조언 좀 해줘",
        "가족 관계가 힘들어", "바다 좋아하세요?", "별 보러 가고 싶어",
        "바이바이", "또 봐요", "사용법 알려줘", "너는 누구야?",
        "오늘 춥다", "미세먼지 심해?", "치킨 좋아해?", "라면 먹고 싶다",
        "화가 나", "스트레스 받아", "오랜만이야", "처음 뵙겠습니다",
        "동물원 가본 적 있어?", "우리 집 강아지가 아파",
        "사회생활이 힘들어", "직장 동료와 갈등이 있어",
        "한라산 올라가 봤어?", "숲속이 정말 좋다",
        "잘 자요", "다음에 또 이야기해요", "어떻게 시작하면 돼?",
        "기능이 뭐가 있어?", "장마 언제야?", "내일 날씨 알려줘",
        "맛집 알려줘", "점심 뭐 먹지?", "힘들어", "오늘 기분이 최고야",
        "안녕!", "고양이 사료 추천해줘", "부모님이 보고 싶어",
        "꽃이 예쁘게 폈어", "오늘 재미있었어", "설명 좀 해줘",
        "오늘 덥다", "디저트 추천해줘", "우울해",
        // 반복 (100까지)
        "안녕하세요", "강아지 좋아해?", "배고파", "날씨 어때?",
        "슬퍼", "도와줘", "안녕히 가세요", "친구가 보고 싶어",
        "자연이 좋아", "뭐 먹을까?", "반가워", "고양이가 귀여워",
        "라면 먹고 싶다", "비 올까?", "기분이 좋아", "사용법 알려줘",
        "잘 가", "가족이 그리워", "바다가 좋아", "치킨 먹자",
        "안녕하세요", "강아지 좋아해?", "배고파", "날씨 어때?",
        "슬퍼", "도와줘", "안녕히 가세요", "친구가 보고 싶어",
        "자연이 좋아", "뭐 먹을까?", "반가워", "고양이가 귀여워",
        "라면 먹고 싶다", "비 올까?", "기분이 좋아", "사용법 알려줘",
        "잘 가",
    ];

    let t_growth = Instant::now();

    for (i, query) in growth_queries.iter().enumerate() {
        let result = chat.respond(query);
        let is_positive = i % 5 != 4; // 80% positive
        chat.feedback(result.pool_id, result.response_id, is_positive);
    }

    // 마지막 growth_cycle 시간 측정
    let t_cycle = Instant::now();
    chat.growth_cycle();
    let growth_cycle_time = t_cycle.elapsed();

    let total_growth_time = t_growth.elapsed();

    // Snapshot stats before further mutable calls
    let s_queries = chat.growth_stats().total_queries;
    let s_cycles = chat.growth_stats().total_cycles;
    let s_pos = chat.growth_stats().positive_feedbacks;
    let s_neg = chat.growth_stats().negative_feedbacks;
    let s_pruned = chat.growth_stats().total_pruned;
    let s_emerged = chat.growth_stats().emergent_pools_created;
    let s_clones = chat.growth_stats().total_clones;

    println!("[성장 통계]");
    println!("  총 쿼리:         {}", s_queries);
    println!("  총 사이클:       {}", s_cycles);
    println!("  긍정 피드백:     {}", s_pos);
    println!("  부정 피드백:     {}", s_neg);
    println!("  정리된 응답:     {}", s_pruned);
    println!("  출현 풀 생성:    {}", s_emerged);
    println!("  클론 생성:       {}", s_clones);
    println!("  성장 시뮬 시간:  {:.1}ms", total_growth_time.as_secs_f64() * 1000.0);
    println!("  growth_cycle:    {:.3}ms", growth_cycle_time.as_secs_f64() * 1000.0);
    println!();

    // 성장 후 정확도 측정
    let mut correct_after = 0;
    let mut total_after = 0;
    for (input, expected) in &cases {
        let result = chat.respond(input);
        if result.intent == *expected { correct_after += 1; }
        total_after += 1;
    }
    let acc_after = correct_after as f64 / total_after as f64 * 100.0;
    println!("[정확도 변화]");
    println!("  성장 전: {:.1}%", acc_before);
    println!("  성장 후: {:.1}%", acc_after);
    let delta = acc_after - acc_before;
    let arrow = if delta > 0.0 { "↑" } else if delta < 0.0 { "↓" } else { "→" };
    println!("  변화:    {:+.1}% {}", delta, arrow);
    println!();

    // ── 풀 상태 ──
    println!("[풀 상태]");
    for (intent, size, avg_fit, emergent) in chat.pool_summary() {
        let tag = if emergent { " [출현]" } else { "" };
        println!("  {:20} 크기={:3} 평균fitness={:.4}{}",
            intent, size, avg_fit, tag);
    }
    println!();

    // ── 출현 패턴 ──
    println!("[출현 패턴]");
    let emergent_pools: Vec<_> = chat.pools.iter()
        .filter(|p| p.is_emergent)
        .collect();
    if emergent_pools.is_empty() {
        println!("  (출현 풀 없음 — 더 많은 쿼리가 필요할 수 있음)");
    } else {
        for pool in &emergent_pools {
            let (pa, pb) = pool.parent_intents.as_ref().unwrap();
            println!("  {} (부모: {} + {}, 크기: {})",
                pool.intent, pa, pb, pool.entries.len());
        }
    }
    println!();

    // ── 요약 ──
    println!("[요약]");
    println!("  정확도:      {}/{} ({:.1}%)", correct, total, acc);
    println!("  성장 후:     {}/{} ({:.1}%)", correct_after, total_after, acc_after);
    println!("  평균 지연:   {:.2}ms", avg);
    println!("  처리량:      {:.0} queries/sec", 1000.0 / avg);
    println!("  성장 사이클: {}", s_cycles);
    println!("  출현 풀:     {}", s_emerged);
    println!("  클론:        {}", s_clones);
    println!();
}

#[test]
fn bench_deep_reservoir() {
    println!("\n======================================================================");
    println!("  Deep Reservoir — depth별 정확도 & 속도 비교");
    println!("======================================================================\n");

    let corpus = chat_corpus();
    let engine_config = EngineConfig {
        dim: 128,
        num_merges: 400,
        max_vocab: 3000,
        seed: 42,
        hidden_dim: 1024,
        ..EngineConfig::default()
    };
    let refs: Vec<&str> = corpus.iter().map(|s| *s).collect();

    let cases = korean_test_cases();

    // Depth 비교 (slits=1)
    println!("[Depth 비교 (slits=1)]");
    for depth in [1, 2, 3] {
        let mut engine = WaveTextEngine::from_corpus_with_config(&refs, &engine_config);
        let (labeled, class_labels) = chat_classification_data();
        engine.train_classifier(&labeled, class_labels);

        let growth_config = GrowthConfig {
            depth,
            num_slits: 1,
            ..GrowthConfig::default()
        };
        let mut chat = ChatEngine::with_growth(engine, chat_intents_ko(), growth_config);

        let t0 = Instant::now();
        let mut correct = 0;
        let mut total = 0;
        for (input, expected) in &cases {
            let result = chat.respond(input);
            if result.intent == *expected { correct += 1; }
            total += 1;
        }
        let elapsed = t0.elapsed();
        let acc = correct as f64 / total as f64 * 100.0;
        let avg_ms = elapsed.as_secs_f64() * 1000.0 / total as f64;

        println!("  depth={} slits=1: 정확도 {}/{} ({:.1}%)  평균 {:.2}ms/쿼리",
            depth, correct, total, acc, avg_ms);
    }
    println!();

    // Multi-slit 비교 (depth=2)
    println!("[Multi-slit 비교 (depth=2)]");
    for num_slits in [1, 2, 3, 4] {
        let mut engine = WaveTextEngine::from_corpus_with_config(&refs, &engine_config);
        let (labeled, class_labels) = chat_classification_data();
        engine.train_classifier(&labeled, class_labels);

        let growth_config = GrowthConfig {
            depth: 2,
            num_slits,
            ..GrowthConfig::default()
        };
        let mut chat = ChatEngine::with_growth(engine, chat_intents_ko(), growth_config);

        let t0 = Instant::now();
        let mut correct = 0;
        let mut total = 0;
        for (input, expected) in &cases {
            let result = chat.respond(input);
            if result.intent == *expected { correct += 1; }
            total += 1;
        }
        let elapsed = t0.elapsed();
        let acc = correct as f64 / total as f64 * 100.0;
        let avg_ms = elapsed.as_secs_f64() * 1000.0 / total as f64;

        println!("  depth=2 slits={}: 정확도 {}/{} ({:.1}%)  평균 {:.2}ms/쿼리",
            num_slits, correct, total, acc, avg_ms);
    }
    println!();
}

#[test]
fn bench_resonance_map() {
    println!("\n======================================================================");
    println!("  [공명 맵] — O(1) 파동 서명 탐색 벤치마크");
    println!("======================================================================\n");

    let mut chat = build_chat_engine();
    let cases = korean_test_cases();

    // ── 공명 맵 구축 시간 ──
    let t_build = Instant::now();
    // Force rebuild by triggering dirty + respond
    chat.growth_cycle();
    let _ = chat.respond("테스트");
    let build_ms = t_build.elapsed().as_secs_f64() * 1000.0;
    println!("[공명 맵 구축]");
    println!("  구축 시간 (growth_cycle + rebuild): {:.3}ms", build_ms);
    println!("  풀 수:         {}", chat.pools.len());
    let total_entries: usize = chat.pools.iter().map(|p| p.entries.len()).sum();
    println!("  총 엔트리:     {}", total_entries);
    println!();

    // ── O(1) 쿼리 속도 ──
    println!("[O(1) 공명 탐색 속도]");
    let mut times = Vec::new();
    let mut correct = 0;
    let mut total = 0;

    for (input, expected) in &cases {
        let t = Instant::now();
        let result = chat.respond(input);
        let elapsed_ms = t.elapsed().as_secs_f64() * 1000.0;
        times.push(elapsed_ms);

        if result.intent == *expected {
            correct += 1;
        } else {
            println!("    FAIL: \"{}\" → 예측={}, 정답={}", input, result.intent, expected);
        }
        total += 1;
    }

    let avg = times.iter().sum::<f64>() / times.len() as f64;
    let mut sorted = times.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = sorted[sorted.len() / 2];
    let p95 = sorted[(sorted.len() as f64 * 0.95) as usize];
    let min = sorted[0];
    let max = *sorted.last().unwrap();

    println!("  정확도:      {}/{} ({:.1}%)", correct, total,
        correct as f64 / total as f64 * 100.0);
    println!("  평균 지연:   {:.2}ms", avg);
    println!("  최소 지연:   {:.2}ms", min);
    println!("  최대 지연:   {:.2}ms", max);
    println!("  P50 지연:    {:.2}ms", p50);
    println!("  P95 지연:    {:.2}ms", p95);
    println!("  처리량:      {:.0} queries/sec", 1000.0 / avg);
    println!();

    // ── 반복 쿼리 (맵 이미 구축, dirty=false) ──
    println!("[반복 쿼리 — 맵 캐시 상태]");
    let repeat_inputs = vec![
        "안녕하세요", "강아지 좋아해?", "배고파", "날씨 어때?",
        "슬퍼", "도와줘", "안녕히 가세요", "친구가 보고 싶어",
        "자연이 좋아", "뭐 먹을까?",
    ];
    let mut repeat_times = Vec::new();
    for input in &repeat_inputs {
        let t = Instant::now();
        let _ = chat.respond(input);
        repeat_times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let repeat_avg = repeat_times.iter().sum::<f64>() / repeat_times.len() as f64;
    println!("  반복 쿼리 수:  {}", repeat_times.len());
    println!("  평균 지연:     {:.2}ms (맵 재구축 없음)", repeat_avg);
    println!("  처리량:        {:.0} queries/sec", 1000.0 / repeat_avg);
    println!();

    // ── 키워드 분류기 vs 파동 클러스터링 비교 ──
    println!("[키워드 분류기 vs 파동 클러스터링]");
    {
        // 방법 1: 키워드 분류기 (현재 방식)
        let mut clf_correct = 0;
        for (input, expected) in &cases {
            if let Some(result) = chat.engine.classify(input) {
                if result.class_label == *expected { clf_correct += 1; }
            }
        }
        println!("  키워드 분류기:    {}/{} ({:.1}%)",
            clf_correct, cases.len(), clf_correct as f64 / cases.len() as f64 * 100.0);

        // 방법 2: 풀 중심 파동 (cavity wave) — 키워드 없이 응답 파동만으로
        // 각 풀의 적합도 가중 중심 특징 벡터 계산
        let pool_centroids: Vec<Vec<f64>> = chat.pools.iter().map(|pool| {
            let dim = pool.entries[0].features.len();
            let total_fitness: f64 = pool.entries.iter().map(|e| e.fitness).sum();
            if total_fitness < 1e-10 {
                return vec![0.0; dim];
            }
            let mut centroid = vec![0.0; dim];
            for entry in &pool.entries {
                let w = entry.fitness / total_fitness;
                for (i, &f) in entry.features.iter().enumerate() {
                    centroid[i] += f * w;
                }
            }
            centroid
        }).collect();

        fn cos_sim(a: &[f64], b: &[f64]) -> f64 {
            let len = a.len().min(b.len());
            let (mut d, mut na, mut nb) = (0.0, 0.0, 0.0);
            for i in 0..len { d += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
            let dn = na.sqrt() * nb.sqrt();
            if dn > 1e-10 { d / dn } else { 0.0 }
        }

        let mut wave_correct = 0;
        for (input, expected) in &cases {
            let state = chat.engine.process_text_deep(input, 2, 1);
            let feat = state.to_feature_vector();
            let mut best_pool = 0;
            let mut best_sim = f64::NEG_INFINITY;
            for (pi, centroid) in pool_centroids.iter().enumerate() {
                let sim = cos_sim(&feat, centroid);
                if sim > best_sim { best_sim = sim; best_pool = pi; }
            }
            if chat.pools[best_pool].intent == *expected { wave_correct += 1; }
        }
        println!("  파동 클러스터링:  {}/{} ({:.1}%) ← 키워드 없이 순수 파동",
            wave_correct, cases.len(), wave_correct as f64 / cases.len() as f64 * 100.0);

        // 방법 3: 파동 + 적합도 가중 (sqrt(fitness))
        let mut wave_fit_correct = 0;
        for (input, expected) in &cases {
            let state = chat.engine.process_text_deep(input, 2, 1);
            let feat = state.to_feature_vector();
            let mut best_pool = 0;
            let mut best_score = f64::NEG_INFINITY;
            for (pi, pool) in chat.pools.iter().enumerate() {
                for entry in &pool.entries {
                    let sim = cos_sim(&feat, &entry.features);
                    let score = sim * entry.fitness.sqrt();
                    if score > best_score { best_score = score; best_pool = pi; }
                }
            }
            if chat.pools[best_pool].intent == *expected { wave_fit_correct += 1; }
        }
        println!("  파동+적합도:      {}/{} ({:.1}%) ← 엔트리별 cos×√fitness",
            wave_fit_correct, cases.len(), wave_fit_correct as f64 / cases.len() as f64 * 100.0);

        // 방법 4: 의미 투영 (SemanticProjection — 입력 공간 Fisher 판별)
        // respond()가 이제 의미 투영을 사용하므로 직접 측정
        let mut sem_correct = 0;
        for (input, expected) in &cases {
            let result = chat.respond(input);
            if result.intent == *expected { sem_correct += 1; }
        }
        println!("  의미 투영:        {}/{} ({:.1}%) ← 입력 파동의 의미 축 투영",
            sem_correct, cases.len(), sem_correct as f64 / cases.len() as f64 * 100.0);

        // 방법 5: 파동 클러스터링 + 분류기 앙상블
        let mut ensemble_correct = 0;
        for (input, expected) in &cases {
            let state = chat.engine.process_text_deep(input, 2, 1);
            let feat = state.to_feature_vector();

            // 파동 점수
            let mut wave_scores: Vec<(usize, f64)> = pool_centroids.iter().enumerate()
                .map(|(pi, c)| (pi, cos_sim(&feat, c)))
                .collect();
            wave_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let wave_top = wave_scores[0].0;

            // 분류기 점수
            let clf_top = if let Some(result) = chat.engine.classify(input) {
                chat.pools.iter().position(|p| p.intent == result.class_label)
                    .unwrap_or(0)
            } else { 0 };

            // 앙상블: 둘이 같으면 확신, 다르면 분류기 우선
            let pick = if wave_top == clf_top { wave_top } else { clf_top };
            if chat.pools[pick].intent == *expected { ensemble_correct += 1; }
        }
        println!("  앙상블(파동+분류기): {}/{} ({:.1}%)",
            ensemble_correct, cases.len(), ensemble_correct as f64 / cases.len() as f64 * 100.0);
    }
    println!();

    // ── 공명 해상도(dims×bands) 비교 ──
    println!("[공명 해상도 비교]");
    println!("  {:>6} {:>6} {:>10} {:>8} {:>12}",
        "dims", "bands", "buckets", "정확도", "평균ms");
    for &(dims, bands) in &[
        (4, 2), (4, 3), (6, 3), (6, 4),
        (8, 3), (8, 4), (10, 4), (12, 4), (16, 4),
    ] {
        let corpus = chat_corpus();
        let config = EngineConfig {
            dim: 128, num_merges: 400, max_vocab: 3000,
            seed: 42, hidden_dim: 1024,
            ..EngineConfig::default()
        };
        let refs: Vec<&str> = corpus.iter().map(|s| *s).collect();
        let mut engine = WaveTextEngine::from_corpus_with_config(&refs, &config);
        let (labeled, class_labels) = chat_classification_data();
        engine.train_classifier(&labeled, class_labels);

        let growth_config = GrowthConfig {
            resonance_dims: dims,
            resonance_bands: bands,
            ..GrowthConfig::default()
        };
        let mut chat_r = ChatEngine::with_growth(engine, chat_intents_ko(), growth_config);

        let t0 = Instant::now();
        let mut c = 0;
        let mut t = 0;
        for (input, expected) in &cases {
            let result = chat_r.respond(input);
            if result.intent == *expected { c += 1; }
            t += 1;
        }
        let elapsed = t0.elapsed();
        let acc = c as f64 / t as f64 * 100.0;
        let avg_ms = elapsed.as_secs_f64() * 1000.0 / t as f64;
        let total_buckets = (bands as u64).pow(dims as u32);

        println!("  {:>6} {:>6} {:>10} {:>7.1}% {:>10.2}ms",
            dims, bands, total_buckets, acc, avg_ms);
    }
    println!();
}

// ══════════════════════════════════════════════════════════════════════════
// 정확 경향성 테스트 — AXOL 시간복잡도 독립 × 수렴 검증
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn bench_convergence_tendency() {
    println!("\n======================================================================");
    println!("  [정확 경향성] — 피드백 반복에 따른 수렴 + 비용 일정성 검증");
    println!("======================================================================\n");

    let mut chat = build_chat_engine();
    chat.self_learning.retrain_interval = 20;

    let eval_cases = korean_test_cases();

    // 학습에 사용할 쿼리 (정답 의도 포함)
    let train_queries: Vec<(&str, &str)> = vec![
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

    // 평가 함수: (정확도, 평균 지연ms, 총 엔트리 수, 풀 수)
    let measure = |chat: &mut ChatEngine, cases: &[(&str, &str)]| -> (f64, f64, usize, usize) {
        let mut correct = 0;
        let mut total_ms = 0.0;
        for &(input, expected) in cases {
            let t = Instant::now();
            let result = chat.respond(input);
            total_ms += t.elapsed().as_secs_f64() * 1000.0;
            if result.intent == expected { correct += 1; }
        }
        let acc = correct as f64 / cases.len() as f64 * 100.0;
        let avg_ms = total_ms / cases.len() as f64;
        let total_entries: usize = chat.pools.iter().map(|p| p.entries.len()).sum();
        let num_pools = chat.pools.len();
        (acc, avg_ms, total_entries, num_pools)
    };

    // ── 체크포인트 기록 ──
    struct Checkpoint {
        round: usize,
        acc: f64,
        avg_ms: f64,
        entries: usize,
        pools: usize,
        retrains: usize,
    }
    let mut checkpoints: Vec<Checkpoint> = Vec::new();

    // ── Round 0: 초기 상태 ──
    let (acc, avg_ms, entries, pools) = measure(&mut chat, &eval_cases);
    checkpoints.push(Checkpoint { round: 0, acc, avg_ms, entries, pools, retrains: 0 });

    // ── Round 1..20: 반복 학습 ──
    let total_rounds = 20;
    for round in 1..=total_rounds {
        // 1라운드 = train_queries 전체를 한 번 순회하며 피드백
        for &(input, correct_intent) in &train_queries {
            let result = chat.respond(input);
            if result.intent == correct_intent {
                chat.feedback(result.pool_id, result.response_id, true);
            } else {
                chat.feedback(result.pool_id, result.response_id, false);
                chat.feedback_correct(correct_intent);
            }
        }

        // 안정화 (50 사이클)
        chat.stabilize(50);

        // 평가
        let (acc, avg_ms, entries, pools) = measure(&mut chat, &eval_cases);
        checkpoints.push(Checkpoint {
            round, acc, avg_ms, entries, pools,
            retrains: chat.retrain_count(),
        });
    }

    // ── 결과 출력 ──
    println!("{:>6} {:>8} {:>10} {:>8} {:>6} {:>8}",
        "라운드", "정확도%", "지연ms", "엔트리", "풀수", "재훈련");
    println!("  {}", "─".repeat(56));
    for cp in &checkpoints {
        // 경향성 화살표
        let trend = if cp.round == 0 { " " }
            else if cp.acc > checkpoints[cp.round - 1].acc { "↑" }
            else if cp.acc < checkpoints[cp.round - 1].acc { "↓" }
            else { "→" };
        println!("  {:>4}   {:>6.1}% {:<2} {:>7.2}ms {:>7} {:>5} {:>6}",
            cp.round, cp.acc, trend, cp.avg_ms, cp.entries, cp.pools, cp.retrains);
    }

    // ── 수렴 분석 ──
    let initial_acc = checkpoints[0].acc;
    let final_acc = checkpoints.last().unwrap().acc;
    let initial_ms = checkpoints[0].avg_ms;
    let final_ms = checkpoints.last().unwrap().avg_ms;
    let initial_entries = checkpoints[0].entries;
    let final_entries = checkpoints.last().unwrap().entries;

    // 정확도가 개선된 라운드 수
    let improving_rounds = checkpoints.windows(2)
        .filter(|w| w[1].acc > w[0].acc).count();
    let declining_rounds = checkpoints.windows(2)
        .filter(|w| w[1].acc < w[0].acc).count();
    let stable_rounds = checkpoints.windows(2)
        .filter(|w| (w[1].acc - w[0].acc).abs() < 0.01).count();

    println!("\n[수렴 분석]");
    println!("  초기 정확도:     {:.1}%", initial_acc);
    println!("  최종 정확도:     {:.1}%", final_acc);
    println!("  정확도 변화:     {:+.1}%", final_acc - initial_acc);
    println!("  개선 라운드:     {}", improving_rounds);
    println!("  하락 라운드:     {}", declining_rounds);
    println!("  안정 라운드:     {}", stable_rounds);

    println!("\n[비용 일정성]");
    println!("  초기 지연:       {:.2}ms (엔트리: {})", initial_ms, initial_entries);
    println!("  최종 지연:       {:.2}ms (엔트리: {})", final_ms, final_entries);
    println!("  엔트리 증가율:   {:.1}x", final_entries as f64 / initial_entries as f64);
    let ms_change = (final_ms - initial_ms) / initial_ms * 100.0;
    println!("  지연 변화율:     {:+.1}%", ms_change);

    // ── 피드백 수당 정확도 효율 ──
    let total_feedback = train_queries.len() * total_rounds;
    let acc_per_feedback = (final_acc - initial_acc) / total_feedback as f64;
    println!("\n[효율]");
    println!("  총 피드백 수:    {}", total_feedback);
    println!("  피드백당 정확도: {:+.3}%/feedback", acc_per_feedback);
    println!("  경향성 비율:     {:.0}% 개선 / {:.0}% 하락",
        improving_rounds as f64 / total_rounds as f64 * 100.0,
        declining_rounds as f64 / total_rounds as f64 * 100.0);

    // ── ASCII 수렴 그래프 ──
    println!("\n[수렴 그래프]");
    let min_acc = checkpoints.iter().map(|c| c.acc).fold(f64::MAX, f64::min);
    let max_acc = checkpoints.iter().map(|c| c.acc).fold(f64::MIN, f64::max);
    let range = (max_acc - min_acc).max(1.0);
    let bar_width = 40;

    for cp in &checkpoints {
        let normalized = (cp.acc - min_acc) / range;
        let filled = (normalized * bar_width as f64) as usize;
        let bar: String = "█".repeat(filled) + &"░".repeat(bar_width - filled);
        println!("  R{:>2} |{}| {:.1}%", cp.round, bar, cp.acc);
    }

    // ── 지연 그래프 ──
    println!("\n[지연 그래프 (엔트리 증가에도 일정한가?)]");
    let min_ms = checkpoints.iter().map(|c| c.avg_ms).fold(f64::MAX, f64::min);
    let max_ms = checkpoints.iter().map(|c| c.avg_ms).fold(f64::MIN, f64::max);
    let ms_range = (max_ms - min_ms).max(0.01);

    for cp in &checkpoints {
        let normalized = (cp.avg_ms - min_ms) / ms_range;
        let filled = (normalized * bar_width as f64) as usize;
        let bar: String = "█".repeat(filled) + &"░".repeat(bar_width - filled);
        println!("  R{:>2} |{}| {:.2}ms (E={})",
            cp.round, bar, cp.avg_ms, cp.entries);
    }

    // ── 결론 ──
    println!("\n[결론]");
    let tendency_valid = improving_rounds > declining_rounds;
    let cost_stable = ms_change.abs() < 50.0; // 지연 변화 50% 이내면 안정

    if tendency_valid && cost_stable {
        println!("  ✓ 정확 경향성 유효: 피드백에 따라 정확도 수렴, 비용 일정");
    } else if tendency_valid {
        println!("  △ 정확 경향성 유효하나 비용 증가 (지연 {:+.1}%)", ms_change);
    } else if cost_stable {
        println!("  △ 비용은 일정하나 정확 경향성 불명확");
    } else {
        println!("  ✗ 경향성 + 비용 일정성 모두 미충족");
    }

    let stats = chat.growth_stats();
    println!("\n[성장 통계]");
    println!("  총 쿼리:       {}", stats.total_queries);
    println!("  총 사이클:     {}", stats.total_cycles);
    println!("  재훈련:        {}회", stats.classifier_retrains);
    println!("  정리:          {}", stats.total_pruned);
    println!("  출현 풀:       {}", stats.emergent_pools_created);
    println!("  클론:          {}", stats.total_clones);
    println!("  시드부스트:    {}", stats.gen_boosts);
}
