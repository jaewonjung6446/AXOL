//! WTE Chat — 공명 기반 한국어 대화 엔진 (메타 성장).
//!
//! Usage: `cargo run --release --bin wte_chat`

use axol::text::data::{chat_corpus, chat_intents_ko, chat_classification_data};
use axol::text::engine::{WaveTextEngine, EngineConfig};
use axol::text::chat::ChatEngine;

use std::io::{self, BufRead, Write};

fn main() {
    println!("=== WTE 한국어 대화 엔진 (메타 성장) ===");
    println!("엔진 빌드 중...");

    let corpus = chat_corpus();
    let config = EngineConfig {
        dim: 128,
        num_merges: 400,
        max_vocab: 3000,
        seed: 42,
        hidden_dim: 1024,
        ..EngineConfig::default()
    };

    let mut engine = WaveTextEngine::from_corpus_with_config(
        &corpus.iter().map(|s| *s).collect::<Vec<&str>>(),
        &config,
    );
    println!("  어휘 크기: {}", engine.vocab_size());
    println!("  특징 차원: {}", engine.feature_dim());

    // 의도 분류기 학습
    println!("의도 분류기 학습 중...");
    let (labeled, class_labels) = chat_classification_data();
    engine.train_classifier(&labeled, class_labels);
    println!("  분류기 학습 완료 ({} 패턴)", labeled.len());

    // 응답 파동 사전 계산
    println!("응답 파동 사전 계산 중...");
    let mut chat = ChatEngine::new(engine, chat_intents_ko());
    let total_entries: usize = chat.pools.iter().map(|p| p.entries.len()).sum();
    println!("  {} 응답 풀 ({} 응답) 준비 완료", chat.pools.len(), total_entries);

    println!("\n준비 완료! 메시지를 입력하세요.");
    println!("  '종료' — 종료");
    println!("  '성장' — 성장 통계 + 풀 상태");
    println!("  '사이클' / '사이클 N' — 성장 사이클 수동 실행 (N회)");
    println!("  응답 후: [+] 좋아요 / [-] 별로 / Enter: 계속\n");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    let mut last_pool_id: Option<usize> = None;
    let mut last_response_id: Option<usize> = None;

    loop {
        print!("나> ");
        stdout.flush().unwrap();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap() == 0 {
            break;
        }
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        if input == "종료" || input == "quit" || input == "exit" {
            println!("안녕히 가세요!");
            break;
        }

        // 성장 통계 명령
        if input == "성장" || input == "growth" {
            let stats = chat.growth_stats();
            println!("\n[성장 통계]");
            println!("  총 쿼리:       {}", stats.total_queries);
            println!("  총 사이클:     {}", stats.total_cycles);
            println!("  긍정 피드백:   {}", stats.positive_feedbacks);
            println!("  부정 피드백:   {}", stats.negative_feedbacks);
            println!("  정리된 응답:   {}", stats.total_pruned);
            println!("  출현 풀 생성:  {}", stats.emergent_pools_created);
            println!("  클론 생성:     {}", stats.total_clones);
            println!("\n[풀 상태]");
            for (intent, size, avg_fit, emergent) in chat.pool_summary() {
                let tag = if emergent { " [출현]" } else { "" };
                println!("  {:20} 크기={:3} fitness={:.4}{}",
                    intent, size, avg_fit, tag);
            }
            println!();
            continue;
        }

        // 사이클 수동 실행
        if input == "사이클" || input == "cycle" {
            chat.growth_cycle();
            let s = chat.growth_stats();
            println!("  → 성장 사이클 1회 실행 (총 {}회)", s.total_cycles);
            println!("    정리: {} / 출현: {} / 클론: {}",
                s.total_pruned, s.emergent_pools_created, s.total_clones);
            println!();
            continue;
        }
        if let Some(rest) = input.strip_prefix("사이클 ").or_else(|| input.strip_prefix("cycle ")) {
            if let Ok(n) = rest.trim().parse::<usize>() {
                let before_cycles = chat.growth_stats().total_cycles;
                let before_pruned = chat.growth_stats().total_pruned;
                let before_emerged = chat.growth_stats().emergent_pools_created;
                let before_clones = chat.growth_stats().total_clones;
                for _ in 0..n {
                    chat.growth_cycle();
                }
                let s = chat.growth_stats();
                println!("  → 성장 사이클 {}회 실행 (총 {}회)", n, s.total_cycles);
                println!("    이번 정리: {} / 출현: {} / 클론: {}",
                    s.total_pruned - before_pruned,
                    s.emergent_pools_created - before_emerged,
                    s.total_clones - before_clones);
                println!();
                continue;
            }
        }

        // 피드백 처리
        if (input == "+" || input == "좋아요") && last_pool_id.is_some() {
            chat.feedback(last_pool_id.unwrap(), last_response_id.unwrap(), true);
            println!("  → 긍정 피드백 반영!");
            println!();
            continue;
        }
        if (input == "-" || input == "별로") && last_pool_id.is_some() {
            chat.feedback(last_pool_id.unwrap(), last_response_id.unwrap(), false);
            println!("  → 부정 피드백 반영!");
            println!();
            continue;
        }

        let result = chat.respond(input);
        println!("봇> {}", result.response);
        println!("    [의도: {}, 확신도: {:.3}, 공명: {:.6}]",
            result.intent, result.confidence, result.resonance);
        println!("    [+] 좋아요 / [-] 별로 / Enter: 계속");
        println!();

        last_pool_id = Some(result.pool_id);
        last_response_id = Some(result.response_id);
    }
}
