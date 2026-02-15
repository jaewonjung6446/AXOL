//! 1000회 성장 사이클 전후 성능 비교 (소프트 게이트 + 출현 활성화).

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

fn eval_cases() -> Vec<(&'static str, &'static str)> {
    vec![
        ("안녕하세요", "greeting"), ("반가워요", "greeting"),
        ("좋은 아침이에요", "greeting"), ("잘 지냈어?", "greeting"),
        ("오랜만이야", "greeting"), ("처음 뵙겠습니다", "greeting"),
        ("안녕!", "greeting"),
        ("강아지 좋아해?", "animals"), ("고양이가 최고야", "animals"),
        ("동물에 대해 알려줘", "animals"), ("반려동물 키워?", "animals"),
        ("동물원 가본 적 있어?", "animals"), ("우리 집 강아지가 아파", "animals"),
        ("고양이 사료 추천해줘", "animals"),
        ("친구랑 싸웠어", "people"), ("사람 관계가 어려워", "people"),
        ("인간관계 조언 좀 해줘", "people"), ("가족 관계가 힘들어", "people"),
        ("사회생활이 힘들어", "people"), ("직장 동료와 갈등이 있어", "people"),
        ("부모님이 보고 싶어", "people"),
        ("자연이 좋아", "nature"), ("산 좋아해?", "nature"),
        ("바다 좋아하세요?", "nature"), ("별 보러 가고 싶어", "nature"),
        ("한라산 올라가 봤어?", "nature"), ("숲속이 정말 좋다", "nature"),
        ("꽃이 예쁘게 폈어", "nature"),
        ("안녕히 가세요", "farewell"), ("잘 가", "farewell"),
        ("바이바이", "farewell"), ("또 봐요", "farewell"),
        ("잘 자요", "farewell"), ("다음에 또 이야기해요", "farewell"),
        ("오늘 재미있었어", "farewell"),
        ("도와줘", "help"), ("뭐 할 수 있어?", "help"),
        ("사용법 알려줘", "help"), ("너는 누구야?", "help"),
        ("어떻게 시작하면 돼?", "help"), ("기능이 뭐가 있어?", "help"),
        ("설명 좀 해줘", "help"),
        ("오늘 날씨 어때?", "weather"), ("비 올까?", "weather"),
        ("오늘 춥다", "weather"), ("미세먼지 심해?", "weather"),
        ("장마 언제야?", "weather"), ("내일 날씨 알려줘", "weather"),
        ("오늘 덥다", "weather"),
        ("배고파", "food"), ("뭐 먹을까?", "food"),
        ("치킨 좋아해?", "food"), ("라면 먹고 싶다", "food"),
        ("맛집 알려줘", "food"), ("점심 뭐 먹지?", "food"),
        ("디저트 추천해줘", "food"),
        ("기분이 좋아", "emotions"), ("슬퍼", "emotions"),
        ("화가 나", "emotions"), ("스트레스 받아", "emotions"),
        ("힘들어", "emotions"), ("오늘 기분이 최고야", "emotions"),
        ("우울해", "emotions"),
        ("ㅁㄴㅇㄹ", "unknown"), ("asdfghjkl", "unknown"),
    ]
}

/// 학습 쿼리와 기대 의도 (정답 피드백용)
fn training_data() -> Vec<(&'static str, &'static str)> {
    vec![
        ("안녕하세요", "greeting"), ("반가워요", "greeting"),
        ("좋은 아침이에요", "greeting"), ("잘 지냈어?", "greeting"),
        ("오랜만이야", "greeting"), ("처음 뵙겠습니다", "greeting"),
        ("안녕!", "greeting"),
        ("강아지 좋아해?", "animals"), ("고양이가 최고야", "animals"),
        ("동물에 대해 알려줘", "animals"), ("반려동물 키워?", "animals"),
        ("동물원 가본 적 있어?", "animals"), ("우리 집 강아지가 아파", "animals"),
        ("고양이 사료 추천해줘", "animals"),
        ("친구랑 싸웠어", "people"), ("사람 관계가 어려워", "people"),
        ("인간관계 조언 좀 해줘", "people"), ("가족 관계가 힘들어", "people"),
        ("사회생활이 힘들어", "people"), ("직장 동료와 갈등이 있어", "people"),
        ("부모님이 보고 싶어", "people"),
        ("자연이 좋아", "nature"), ("산 좋아해?", "nature"),
        ("바다 좋아하세요?", "nature"), ("별 보러 가고 싶어", "nature"),
        ("한라산 올라가 봤어?", "nature"), ("숲속이 정말 좋다", "nature"),
        ("꽃이 예쁘게 폈어", "nature"),
        ("안녕히 가세요", "farewell"), ("잘 가", "farewell"),
        ("바이바이", "farewell"), ("또 봐요", "farewell"),
        ("잘 자요", "farewell"), ("다음에 또 이야기해요", "farewell"),
        ("오늘 재미있었어", "farewell"),
        ("도와줘", "help"), ("뭐 할 수 있어?", "help"),
        ("사용법 알려줘", "help"), ("너는 누구야?", "help"),
        ("어떻게 시작하면 돼?", "help"), ("기능이 뭐가 있어?", "help"),
        ("설명 좀 해줘", "help"),
        ("오늘 날씨 어때?", "weather"), ("비 올까?", "weather"),
        ("오늘 춥다", "weather"), ("미세먼지 심해?", "weather"),
        ("장마 언제야?", "weather"), ("내일 날씨 알려줘", "weather"),
        ("오늘 덥다", "weather"),
        ("배고파", "food"), ("뭐 먹을까?", "food"),
        ("치킨 좋아해?", "food"), ("라면 먹고 싶다", "food"),
        ("맛집 알려줘", "food"), ("점심 뭐 먹지?", "food"),
        ("디저트 추천해줘", "food"),
        ("기분이 좋아", "emotions"), ("슬퍼", "emotions"),
        ("화가 나", "emotions"), ("스트레스 받아", "emotions"),
        ("힘들어", "emotions"), ("오늘 기분이 최고야", "emotions"),
        ("우울해", "emotions"),
    ]
}

fn measure_accuracy(chat: &mut ChatEngine, cases: &[(&str, &str)]) -> (usize, usize, f64, Vec<(String, String, bool)>) {
    let mut correct = 0;
    let total = cases.len();
    let mut details = Vec::new();
    for (input, expected) in cases {
        let result = chat.respond(input);
        let ok = result.intent == *expected;
        if ok { correct += 1; }
        details.push((input.to_string(), result.intent.clone(), ok));
    }
    let acc = correct as f64 / total as f64 * 100.0;
    (correct, total, acc, details)
}

#[test]
fn test_growth_1000_cycles() {
    println!("\n======================================================================");
    println!("  1000회 성장 사이클 — 소프트 게이트 + 출현 활성화");
    println!("======================================================================\n");

    let mut chat = build_chat_engine();
    let cases = eval_cases();
    let train = training_data();

    // ── 성장 전 측정 ──
    let (c0, t0, acc0, details0) = measure_accuracy(&mut chat, &cases);
    let pool_before: Vec<(String, usize, f64, bool)> = chat.pool_summary();
    let total_entries_before: usize = pool_before.iter().map(|(_, s, _, _)| s).sum();

    println!("[성장 전]");
    println!("  정확도: {}/{} ({:.1}%)", c0, t0, acc0);
    println!("  풀 수: {}, 총 응답: {}", pool_before.len(), total_entries_before);

    // MISS 목록
    let misses0: Vec<_> = details0.iter().filter(|(_, _, ok)| !ok).collect();
    println!("  MISS: {}개", misses0.len());
    for (input, got, _) in &misses0 {
        let expected = cases.iter().find(|(i, _)| *i == input.as_str()).unwrap().1;
        println!("    '{}' → {} (기대: {})", input, got, expected);
    }
    println!();

    // ── 시뮬레이션: 10라운드 학습 + 1000 사이클 ──
    println!("[시뮬레이션]");
    let t_sim = Instant::now();

    // 10라운드 학습 (650회 쿼리 + 정답 기반 피드백)
    for _round in 0..10 {
        for (query, expected) in &train {
            let result = chat.respond(query);
            let is_correct = result.intent == *expected;
            chat.feedback(result.pool_id, result.response_id, is_correct);
        }
    }

    let queries_done = chat.growth_stats().total_queries;
    let auto_cycles = chat.growth_stats().total_cycles;
    println!("  학습 쿼리: {}회 (자동 사이클 {}회)", queries_done - (c0 + t0), auto_cycles);

    // 1000회 수동 사이클
    let t_cycles = Instant::now();
    for _ in 0..1000 {
        chat.growth_cycle();
    }
    let cycle_time = t_cycles.elapsed();
    let sim_time = t_sim.elapsed();

    println!("  수동 사이클: 1000회 (총 {}회)", chat.growth_stats().total_cycles);
    println!("  시뮬 시간:   {:.1}ms", sim_time.as_secs_f64() * 1000.0);
    println!("  1000사이클:  {:.1}ms", cycle_time.as_secs_f64() * 1000.0);
    println!();

    // ── 성장 후 측정 ──
    let (c1, t1, acc1, details1) = measure_accuracy(&mut chat, &cases);
    let pool_after: Vec<(String, usize, f64, bool)> = chat.pool_summary();
    let total_entries_after: usize = pool_after.iter().map(|(_, s, _, _)| s).sum();

    println!("[성장 후]");
    println!("  정확도: {}/{} ({:.1}%)", c1, t1, acc1);
    println!("  풀 수: {}, 총 응답: {}", pool_after.len(), total_entries_after);

    // MISS 목록
    let misses1: Vec<_> = details1.iter().filter(|(_, _, ok)| !ok).collect();
    println!("  MISS: {}개", misses1.len());
    for (input, got, _) in &misses1 {
        let expected = cases.iter().find(|(i, _)| *i == input.as_str()).unwrap().1;
        println!("    '{}' → {} (기대: {})", input, got, expected);
    }
    println!();

    // ── 개선/퇴보 분석 ──
    println!("[변화 분석]");
    let mut fixed = Vec::new();
    let mut broken = Vec::new();
    for i in 0..cases.len() {
        let (ref inp, _, ok0) = details0[i];
        let (_, _, ok1) = details1[i];
        if !ok0 && ok1 {
            fixed.push(inp.clone());
        } else if ok0 && !ok1 {
            broken.push(inp.clone());
        }
    }
    if !fixed.is_empty() {
        println!("  개선 (MISS→OK): {}개", fixed.len());
        for f in &fixed { println!("    + '{}'", f); }
    }
    if !broken.is_empty() {
        println!("  퇴보 (OK→MISS): {}개", broken.len());
        for b in &broken { println!("    - '{}'", b); }
    }
    if fixed.is_empty() && broken.is_empty() {
        println!("  변화 없음");
    }
    println!();

    // ── 성장 통계 ──
    let s = chat.growth_stats();
    println!("[성장 통계]");
    println!("  총 쿼리:       {}", s.total_queries);
    println!("  총 사이클:     {}", s.total_cycles);
    println!("  긍정 피드백:   {}", s.positive_feedbacks);
    println!("  부정 피드백:   {}", s.negative_feedbacks);
    println!("  정리된 응답:   {}", s.total_pruned);
    println!("  출현 풀 생성:  {}", s.emergent_pools_created);
    println!("  클론 생성:     {}", s.total_clones);
    println!();

    // ── 풀 상태 비교 ──
    println!("[풀 상태 비교]");
    println!("  {:25} {:>6} {:>6} {:>10} {:>10} {:>4}",
        "의도", "전", "후", "전fitness", "후fitness", "출현");
    println!("  {}", "-".repeat(67));
    for (intent, size_a, fit_a, emrg_a) in &pool_after {
        let (size_b, fit_b) = pool_before.iter()
            .find(|(i, _, _, _)| i == intent)
            .map(|(_, s, f, _)| (*s, *f))
            .unwrap_or((0, 0.0));
        let tag = if *emrg_a { " *" } else { "" };
        println!("  {:25} {:>6} {:>6} {:>10.4} {:>10.4}{}",
            intent, size_b, size_a, fit_b, fit_a, tag);
    }
    println!();

    // ── 출현 패턴 ──
    let emergent: Vec<_> = chat.pools.iter().filter(|p| p.is_emergent).collect();
    if !emergent.is_empty() {
        println!("[출현 패턴]");
        for pool in &emergent {
            let (pa, pb) = pool.parent_intents.as_ref().unwrap();
            println!("  {} (부모: {} + {}, 크기: {})", pool.intent, pa, pb, pool.entries.len());
        }
        println!();
    }

    // ── 최종 비교 ──
    let delta = acc1 - acc0;
    let arrow = if delta > 0.5 { "↑" } else if delta < -0.5 { "↓" } else { "→" };
    println!("======================================================================");
    println!("  정확도:  {:.1}% → {:.1}% ({:+.1}%) {}", acc0, acc1, delta, arrow);
    println!("  풀 수:   {} → {}", pool_before.len(), pool_after.len());
    println!("  응답 수: {} → {}", total_entries_before, total_entries_after);
    println!("  개선:    {} / 퇴보: {}", fixed.len(), broken.len());
    println!("======================================================================\n");
}
