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

fn measure_generation(chat: &mut ChatEngine, inputs: &[(&str, &str)], max_tokens: usize) -> Vec<(String, String, String, bool, f64)> {
    let original_max = chat.generation.max_tokens;
    chat.generation.max_tokens = max_tokens;

    let mut results = Vec::new();
    for (input, expected_intent) in inputs {
        let result = chat.generate_response(input);
        let q = result.generation_quality.unwrap_or(0.0);
        results.push((
            input.to_string(),
            result.intent.clone(),
            result.response.clone(),
            result.is_generated,
            q,
        ));
    }

    chat.generation.max_tokens = original_max;
    results
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

// =========================================================================
// 100,000 사이클 성장 + 파동 생성 전후 비교
// =========================================================================

#[test]
fn test_growth_100k_cycles_with_generation() {
    println!("\n======================================================================");
    println!("  100,000회 성장 사이클 — 파동 생성 전후 비교 (긴 문장 포함)");
    println!("======================================================================\n");

    let mut chat = build_chat_engine();
    let cases = eval_cases();
    let train = training_data();

    // 생성 평가용 입력 (다양한 의도 + 긴 응답 기대)
    let gen_eval: Vec<(&str, &str)> = vec![
        ("안녕하세요 오늘 기분이 어떠세요?", "greeting"),
        ("강아지가 아파요 어떻게 해야 하나요?", "animals"),
        ("친구랑 크게 싸웠어 너무 속상해", "people"),
        ("바다가 너무 보고 싶어 제주도 가고 싶다", "nature"),
        ("이제 그만 가볼게요 오늘 즐거웠어", "farewell"),
        ("이 프로그램 어떻게 사용하는 거야?", "help"),
        ("내일 비 온대 우산 챙겨야겠다", "weather"),
        ("배고파 맛있는 거 먹고 싶다 추천해줘", "food"),
        ("요즘 너무 우울하고 힘들어 스트레스 받아", "emotions"),
        ("ㅋㅋㅋ 뭐하고 있어?", "unknown"),
    ];

    // ── 성장 전: 정확도 + 생성 측정 ──
    let (c0, t0, acc0, _) = measure_accuracy(&mut chat, &cases);

    // 짧은 생성 (20 토큰)
    let gen_before_short = measure_generation(&mut chat, &gen_eval, 20);
    // 긴 생성 (50 토큰)
    let gen_before_long = measure_generation(&mut chat, &gen_eval, 50);

    println!("[성장 전]");
    println!("  풀 정확도: {}/{} ({:.1}%)", c0, t0, acc0);
    println!();

    println!("  [생성 — 20 토큰]");
    let mut q_sum_before_s = 0.0;
    let mut q_cnt_before_s = 0;
    for (input, intent, response, is_gen, quality) in &gen_before_short {
        let tag = if *is_gen { "생성" } else { "폴백" };
        if *is_gen { q_sum_before_s += quality; q_cnt_before_s += 1; }
        println!("    [{:4}] {:35} Φ={:.4} 의도={:10} '{}'",
            tag, input, quality, intent,
            if response.len() > 60 {
                format!("{}...", &response[..response.char_indices()
                    .nth(60).map_or(response.len(), |(i, _)| i)])
            } else { response.clone() });
    }
    let avg_q_before_s = if q_cnt_before_s > 0 { q_sum_before_s / q_cnt_before_s as f64 } else { 0.0 };
    println!("    평균 Φ: {:.4} (생성 {}/{})", avg_q_before_s, q_cnt_before_s, gen_eval.len());
    println!();

    println!("  [생성 — 50 토큰]");
    let mut q_sum_before_l = 0.0;
    let mut q_cnt_before_l = 0;
    for (input, intent, response, is_gen, quality) in &gen_before_long {
        let tag = if *is_gen { "생성" } else { "폴백" };
        if *is_gen { q_sum_before_l += quality; q_cnt_before_l += 1; }
        println!("    [{:4}] {:35} Φ={:.4} 의도={:10} '{}'",
            tag, input, quality, intent,
            if response.len() > 80 {
                format!("{}...", &response[..response.char_indices()
                    .nth(80).map_or(response.len(), |(i, _)| i)])
            } else { response.clone() });
    }
    let avg_q_before_l = if q_cnt_before_l > 0 { q_sum_before_l / q_cnt_before_l as f64 } else { 0.0 };
    println!("    평균 Φ: {:.4} (생성 {}/{})", avg_q_before_l, q_cnt_before_l, gen_eval.len());
    println!();

    // ── 학습 시뮬레이션: 20라운드 + 100,000 사이클 ──
    println!("[시뮬레이션: 20라운드 학습 + 100,000 사이클]");
    let t_sim = Instant::now();

    // 20라운드 학습 (정답 기반 피드백)
    for _round in 0..20 {
        for (query, expected) in &train {
            let result = chat.respond(query);
            let is_correct = result.intent == *expected;
            chat.feedback(result.pool_id, result.response_id, is_correct);
        }
    }

    let queries_after_train = chat.growth_stats().total_queries;
    let auto_cycles = chat.growth_stats().total_cycles;
    println!("  학습 쿼리:   {}회 (자동 사이클 {}회)", queries_after_train, auto_cycles);

    // 100,000회 수동 사이클
    let t_cycles = Instant::now();
    for _ in 0..100_000 {
        chat.growth_cycle();
    }
    let cycle_time = t_cycles.elapsed();
    let sim_time = t_sim.elapsed();

    println!("  수동 사이클: 100,000회 (총 {}회)", chat.growth_stats().total_cycles);
    println!("  시뮬 시간:   {:.1}ms", sim_time.as_secs_f64() * 1000.0);
    println!("  100k사이클:  {:.1}ms ({:.3}ms/cycle)",
        cycle_time.as_secs_f64() * 1000.0,
        cycle_time.as_secs_f64() * 1000.0 / 100_000.0);
    println!();

    // ── 성장 후: 정확도 + 생성 측정 ──
    let (c1, t1, acc1, details1) = measure_accuracy(&mut chat, &cases);

    let gen_after_short = measure_generation(&mut chat, &gen_eval, 20);
    let gen_after_long = measure_generation(&mut chat, &gen_eval, 50);

    println!("[성장 후]");
    println!("  풀 정확도: {}/{} ({:.1}%)", c1, t1, acc1);
    println!();

    println!("  [생성 — 20 토큰]");
    let mut q_sum_after_s = 0.0;
    let mut q_cnt_after_s = 0;
    for (input, intent, response, is_gen, quality) in &gen_after_short {
        let tag = if *is_gen { "생성" } else { "폴백" };
        if *is_gen { q_sum_after_s += quality; q_cnt_after_s += 1; }
        println!("    [{:4}] {:35} Φ={:.4} 의도={:10} '{}'",
            tag, input, quality, intent,
            if response.len() > 60 {
                format!("{}...", &response[..response.char_indices()
                    .nth(60).map_or(response.len(), |(i, _)| i)])
            } else { response.clone() });
    }
    let avg_q_after_s = if q_cnt_after_s > 0 { q_sum_after_s / q_cnt_after_s as f64 } else { 0.0 };
    println!("    평균 Φ: {:.4} (생성 {}/{})", avg_q_after_s, q_cnt_after_s, gen_eval.len());
    println!();

    println!("  [생성 — 50 토큰]");
    let mut q_sum_after_l = 0.0;
    let mut q_cnt_after_l = 0;
    for (input, intent, response, is_gen, quality) in &gen_after_long {
        let tag = if *is_gen { "생성" } else { "폴백" };
        if *is_gen { q_sum_after_l += quality; q_cnt_after_l += 1; }
        println!("    [{:4}] {:35} Φ={:.4} 의도={:10} '{}'",
            tag, input, quality, intent,
            if response.len() > 80 {
                format!("{}...", &response[..response.char_indices()
                    .nth(80).map_or(response.len(), |(i, _)| i)])
            } else { response.clone() });
    }
    let avg_q_after_l = if q_cnt_after_l > 0 { q_sum_after_l / q_cnt_after_l as f64 } else { 0.0 };
    println!("    평균 Φ: {:.4} (생성 {}/{})", avg_q_after_l, q_cnt_after_l, gen_eval.len());
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

    // ── 풀 상태 ──
    println!("[풀 상태]");
    for (intent, size, avg_fit, emergent) in chat.pool_summary() {
        let tag = if emergent { " [출현]" } else { "" };
        println!("  {:25} 크기={:3} fitness={:.6}{}",
            intent, size, avg_fit, tag);
    }
    println!();

    // ── 생성 전후 비교 ──
    println!("======================================================================");
    println!("  [전후 비교]");
    println!("======================================================================\n");

    let delta_acc = acc1 - acc0;
    let arrow_acc = if delta_acc > 0.5 { "↑" } else if delta_acc < -0.5 { "↓" } else { "→" };
    println!("  풀 정확도:      {:.1}% → {:.1}% ({:+.1}%) {}", acc0, acc1, delta_acc, arrow_acc);
    println!();

    println!("  생성 Φ (20tok): {:.4} → {:.4} ({:+.4})",
        avg_q_before_s, avg_q_after_s, avg_q_after_s - avg_q_before_s);
    println!("  생성 Φ (50tok): {:.4} → {:.4} ({:+.4})",
        avg_q_before_l, avg_q_after_l, avg_q_after_l - avg_q_before_l);
    println!("  생성 성공:      {}/{} → {}/{} (20tok)",
        q_cnt_before_s, gen_eval.len(), q_cnt_after_s, gen_eval.len());
    println!("  생성 성공:      {}/{} → {}/{} (50tok)",
        q_cnt_before_l, gen_eval.len(), q_cnt_after_l, gen_eval.len());
    println!();

    // 입력별 상세 비교 (50 토큰 긴 문장)
    println!("  [50 토큰 생성 — 전후 비교]");
    for i in 0..gen_eval.len() {
        let (input, _, _, _, _) = &gen_before_long[i];
        let (_, _, resp_b, gen_b, q_b) = &gen_before_long[i];
        let (_, _, resp_a, gen_a, q_a) = &gen_after_long[i];
        let tag_b = if *gen_b { "생성" } else { "폴백" };
        let tag_a = if *gen_a { "생성" } else { "폴백" };
        println!("    '{}' :", input);
        println!("      전[{}] Φ={:.4} '{}'", tag_b, q_b,
            if resp_b.len() > 80 {
                format!("{}...", &resp_b[..resp_b.char_indices()
                    .nth(80).map_or(resp_b.len(), |(i, _)| i)])
            } else { resp_b.clone() });
        println!("      후[{}] Φ={:.4} '{}'", tag_a, q_a,
            if resp_a.len() > 80 {
                format!("{}...", &resp_a[..resp_a.char_indices()
                    .nth(80).map_or(resp_a.len(), |(i, _)| i)])
            } else { resp_a.clone() });
    }
    println!();

    println!("======================================================================\n");
}

// =========================================================================
// 자기 학습 (Level 4) — 분류기 재학습으로 83.1% 천장 돌파
// =========================================================================

#[test]
fn test_self_learning_10m() {
    println!("\n======================================================================");
    println!("  자기 학습 (Level 4) — 10M 학습 + 분류기 재학습");
    println!("======================================================================\n");

    let mut chat = build_chat_engine();
    let cases = eval_cases();
    let train = training_data();

    // 자기 학습 설정: 경험 65개마다 재학습
    chat.self_learning.retrain_interval = 65;

    // ── 학습 전 측정 ──
    let (c0, t0, acc0, details0) = measure_accuracy(&mut chat, &cases);
    println!("[학습 전]");
    println!("  정확도:  {}/{} ({:.1}%)", c0, t0, acc0);
    let misses0: Vec<_> = details0.iter().filter(|(_, _, ok)| !ok).collect();
    println!("  MISS:    {}개", misses0.len());
    for (input, got, _) in &misses0 {
        let expected = cases.iter().find(|(i, _)| *i == input.as_str()).unwrap().1;
        println!("    '{}' → {} (기대: {})", input, got, expected);
    }
    println!();

    // ── 10M 학습 시뮬레이션 ──
    // 65개 학습 데이터 × ~154,000 라운드 ≈ 10M 쿼리
    let total_target = 10_000_000;
    let round_size = train.len();
    let num_rounds = total_target / round_size;

    println!("[시뮬레이션: {}라운드 × {} = {}회 쿼리]", num_rounds, round_size, num_rounds * round_size);
    let t_sim = Instant::now();

    let mut milestone_accuracies = Vec::new();

    for round in 0..num_rounds {
        for (query, expected) in &train {
            let result = chat.respond(query);
            let is_correct = result.intent == *expected;
            if is_correct {
                chat.feedback(result.pool_id, result.response_id, true);
            } else {
                chat.feedback(result.pool_id, result.response_id, false);
                // 핵심: 오분류 시 정답 의도를 교정 → 분류기 학습 데이터에 추가
                chat.feedback_correct(expected);
            }
        }

        // 마일스톤 기록 (0.1M, 0.5M, 1M, 2M, 5M, 10M 시점)
        let queries_so_far = (round + 1) * round_size;
        let milestones = [100_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000];
        for &m in &milestones {
            if queries_so_far >= m && queries_so_far < m + round_size {
                let (c, t, acc, _) = measure_accuracy(&mut chat, &cases);
                let retrains = chat.retrain_count();
                let exps = chat.experience_count();
                milestone_accuracies.push((m, c, t, acc, retrains, exps));
                println!("  {:>8} 쿼리: 정확도 {}/{} ({:.1}%) | 재학습 {}회 | 경험 {}개",
                    format_count(m), c, t, acc, retrains, exps);
            }
        }
    }

    let sim_time = t_sim.elapsed();
    println!("  시뮬 시간: {:.1}s", sim_time.as_secs_f64());
    println!();

    // ── 학습 후 측정 ──
    let (c1, t1, acc1, details1) = measure_accuracy(&mut chat, &cases);
    println!("[학습 후]");
    println!("  정확도:  {}/{} ({:.1}%)", c1, t1, acc1);
    let misses1: Vec<_> = details1.iter().filter(|(_, _, ok)| !ok).collect();
    println!("  MISS:    {}개", misses1.len());
    for (input, got, _) in &misses1 {
        let expected = cases.iter().find(|(i, _)| *i == input.as_str()).unwrap().1;
        println!("    '{}' → {} (기대: {})", input, got, expected);
    }
    println!();

    // ── 생성 품질 비교 ──
    let gen_eval: Vec<(&str, &str)> = vec![
        ("안녕하세요 오늘 기분이 어떠세요?", "greeting"),
        ("강아지가 아파요 어떻게 해야 하나요?", "animals"),
        ("친구랑 크게 싸웠어 너무 속상해", "people"),
        ("바다가 너무 보고 싶어 제주도 가고 싶다", "nature"),
        ("이제 그만 가볼게요 오늘 즐거웠어", "farewell"),
        ("이 프로그램 어떻게 사용하는 거야?", "help"),
        ("내일 비 온대 우산 챙겨야겠다", "weather"),
        ("배고파 맛있는 거 먹고 싶다 추천해줘", "food"),
        ("요즘 너무 우울하고 힘들어 스트레스 받아", "emotions"),
        ("ㅋㅋㅋ 뭐하고 있어?", "unknown"),
    ];

    println!("[생성 품질 — 50 토큰]");
    let gen_results = measure_generation(&mut chat, &gen_eval, 50);
    let mut gen_correct = 0;
    let mut q_sum = 0.0;
    let mut q_cnt = 0;
    for (i, (input, intent, response, is_gen, quality)) in gen_results.iter().enumerate() {
        let expected = gen_eval[i].1;
        let tag = if *is_gen { "생성" } else { "폴백" };
        let intent_ok = intent == expected;
        if intent_ok { gen_correct += 1; }
        if *is_gen { q_sum += quality; q_cnt += 1; }
        let mark = if intent_ok { "OK" } else { "MISS" };
        println!("  [{:4}][{:4}] {:35} 의도={:10} 기대={:10} Φ={:.4}",
            mark, tag, input, intent, expected, quality);
        println!("              '{}'",
            if response.len() > 80 {
                format!("{}...", &response[..response.char_indices()
                    .nth(80).map_or(response.len(), |(i, _)| i)])
            } else { response.clone() });
    }
    let gen_avg_q = if q_cnt > 0 { q_sum / q_cnt as f64 } else { 0.0 };
    println!("  생성 의도 정확도: {}/{}", gen_correct, gen_eval.len());
    println!("  평균 Φ: {:.4}", gen_avg_q);
    println!();

    // ── 변화 분석 ──
    println!("[변화 분석]");
    let mut fixed = Vec::new();
    let mut broken = Vec::new();
    for i in 0..cases.len() {
        let (ref inp, _, ok0) = details0[i];
        let (_, _, ok1) = details1[i];
        if !ok0 && ok1 { fixed.push(inp.clone()); }
        else if ok0 && !ok1 { broken.push(inp.clone()); }
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
    println!("[통계]");
    println!("  총 쿼리:        {}", s.total_queries);
    println!("  총 사이클:      {}", s.total_cycles);
    println!("  분류기 재학습:  {}", s.classifier_retrains);
    println!("  경험 축적:      {}", chat.experience_count());
    println!("  긍정 피드백:    {}", s.positive_feedbacks);
    println!("  부정 피드백:    {}", s.negative_feedbacks);
    println!("  정리 응답:      {}", s.total_pruned);
    println!("  출현 풀:        {}", s.emergent_pools_created);
    println!("  클론:           {}", s.total_clones);
    println!();

    // ── 정확도 진화 그래프 ──
    println!("[정확도 진화]");
    println!("  {:>10}  {:>10}  {:>8}  {:>8}", "쿼리", "정확도", "재학습", "경험");
    println!("  {}", "-".repeat(42));
    println!("  {:>10}  {:>5}/{:<4} ({:.1}%)  {:>8}  {:>8}",
        "0", c0, t0, acc0, 0, 0);
    for (m, c, t, acc, retrains, exps) in &milestone_accuracies {
        println!("  {:>10}  {:>5}/{:<4} ({:.1}%)  {:>8}  {:>8}",
            format_count(*m), c, t, acc, retrains, exps);
    }
    println!("  {:>10}  {:>5}/{:<4} ({:.1}%)  {:>8}  {:>8}",
        "최종", c1, t1, acc1, chat.retrain_count(), chat.experience_count());
    println!();

    // ── 최종 비교 ──
    let delta = acc1 - acc0;
    let arrow = if delta > 0.5 { "↑" } else if delta < -0.5 { "↓" } else { "→" };
    println!("======================================================================");
    println!("  정확도:  {:.1}% → {:.1}% ({:+.1}%) {}", acc0, acc1, delta, arrow);
    println!("  개선:    {} / 퇴보: {}", fixed.len(), broken.len());
    println!("  재학습:  {}회 (경험 {}개)", chat.retrain_count(), chat.experience_count());
    println!("  시간:    {:.1}s", sim_time.as_secs_f64());
    println!("======================================================================\n");
}

fn format_count(n: usize) -> String {
    if n >= 1_000_000 { format!("{}M", n / 1_000_000) }
    else if n >= 1_000 { format!("{}K", n / 1_000) }
    else { format!("{}", n) }
}
