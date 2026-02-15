//! WTE 한국어 대화 엔진 — 정확도 & 속도 & 메타 성장 벤치마크.

use axol::text::data::{chat_corpus, chat_intents_ko, chat_classification_data};
use axol::text::engine::{WaveTextEngine, EngineConfig};
use axol::text::chat::ChatEngine;
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
