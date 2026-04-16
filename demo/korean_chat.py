"""한국어 실전 테스트: Q&A 암기 + 프랙탈 변주.

AXOL의 CharTokenizer는 원래 유니코드 기반이므로 한글을 있는 그대로
처리합니다. 여기서 검증하는 것:

  1. Q&A 쌍 완벽 recall (snap decoder)
  2. 새 프롬프트의 가장 가까운 문장 retrieval
  3. 프랙탈 노이즈로 창조적 변주
  4. 한국 속담에서의 동작

실행 방법
---------
    python demo/korean_chat.py
    python demo/korean_chat.py --repl        (대화형 모드)
"""

from __future__ import annotations

import argparse
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from axol.quantum.conversation import vocab_from_texts  # noqa: E402
from axol.quantum.fractal_text import FractalTextGenerator  # noqa: E402
from axol.quantum.sentence_decoder import SentenceDecoderLanguageModel  # noqa: E402


# ---------------------------------------------------------------------------
# 대화 샘플 (FAQ 스타일)
# ---------------------------------------------------------------------------

QA_PAIRS: list[tuple[str, str]] = [
    ("안녕",           "안녕하세요 반갑습니다"),
    ("이름이 뭐야",    "저는 악솔 입니다"),
    ("잘 지내",        "네 덕분에 잘 지냅니다"),
    ("뭐 해",          "지금 대화를 나누고 있습니다"),
    ("고마워",         "천만에요 도움이 되어 기쁩니다"),
    ("미안해",         "괜찮습니다 신경 쓰지 마세요"),
    ("몇 살",          "저는 방금 만들어졌습니다"),
    ("날씨",           "저는 날씨를 알 수 없습니다"),
    ("잘 가",          "안녕히 가세요 또 만나요"),
    ("사랑해",         "저도 당신을 좋아합니다"),
    ("배고파",         "맛있는 것을 드세요"),
    ("슬퍼",           "무슨 일인지 이야기해 보세요"),
]


# ---------------------------------------------------------------------------
# 한국 속담 (창조적 변주용)
# ---------------------------------------------------------------------------

PROVERBS: list[str] = [
    "가는 말이 고와야 오는 말이 곱다",
    "세 살 버릇 여든까지 간다",
    "등잔 밑이 어둡다",
    "티끌 모아 태산",
    "백지장도 맞들면 낫다",
    "발 없는 말이 천리 간다",
    "우물 안 개구리",
    "고생 끝에 낙이 온다",
    "호랑이도 제 말 하면 온다",
    "벼는 익을수록 고개를 숙인다",
]


# ---------------------------------------------------------------------------
# 모델 학습
# ---------------------------------------------------------------------------

def build_qa_model() -> SentenceDecoderLanguageModel:
    all_text = [q for q, _ in QA_PAIRS] + [a for _, a in QA_PAIRS]
    vocab = vocab_from_texts(all_text)
    m = SentenceDecoderLanguageModel(
        vocab=vocab, intent_dim=22, intent_window=6,
        regularization=1e-4, seed=0,
    )
    m.train_pairs(QA_PAIRS, epochs=5)
    return m


def build_proverb_model() -> SentenceDecoderLanguageModel:
    # 속담을 (첫 두 어절 → 전체) 쌍으로 학습
    pairs = [(" ".join(p.split()[:2]), p) for p in PROVERBS]
    all_text = [q for q, _ in pairs] + [a for _, a in pairs]
    vocab = vocab_from_texts(all_text)
    m = SentenceDecoderLanguageModel(
        vocab=vocab, intent_dim=24, intent_window=6,
        regularization=1e-4, seed=0,
    )
    m.train_pairs(pairs, epochs=6)
    return m


# ---------------------------------------------------------------------------
# 평가 섹션들
# ---------------------------------------------------------------------------

def eval_qa_recall(m: SentenceDecoderLanguageModel) -> None:
    print("=" * 75)
    print("1) Q&A 완벽 recall 테스트 (snap decoder)")
    print("=" * 75)
    print(f"{'프롬프트':12s}  {'정답':28s}  {'응답':28s}  conf")
    print("-" * 75)
    hits = 0
    for q, expected in QA_PAIRS:
        res = m.generate(q)
        ok = res.text == expected
        hits += int(ok)
        mark = "✓" if ok else "✗"
        print(f"  {mark} {q:12s}  {expected:28s}  {res.text:28s}  {res.confidence:.3f}")
    print(f"\n  암기 성공: {hits}/{len(QA_PAIRS)}")
    print(f"  intent Ω={m.omega:.3f}   dict_size={m.dictionary_size}\n")


def eval_novel_prompts(m: SentenceDecoderLanguageModel) -> None:
    print("=" * 75)
    print("2) 학습 안 한 프롬프트 — 가장 가까운 학습 문장으로 snap")
    print("=" * 75)
    novels = ["반가워", "안녕히", "뭐야", "오늘 어때", "도와줘"]
    for q in novels:
        res = m.generate(q, top_k=3)
        print(f"  프롬프트: {q!r}")
        print(f"    응답:   {res.text!r}  (conf={res.confidence:.3f})")
        for alt, score in res.alternatives[1:]:
            print(f"    대안:   {alt!r}  ({score:.3f})")
        print()


def eval_proverb_recall(m: SentenceDecoderLanguageModel) -> None:
    print("=" * 75)
    print("3) 한국 속담 recall")
    print("=" * 75)
    hits = 0
    for p in PROVERBS:
        prompt = " ".join(p.split()[:2])
        res = m.generate(prompt)
        ok = res.text == p
        hits += int(ok)
        mark = "✓" if ok else "✗"
        print(f"  {mark} {prompt:10s} → {res.text}")
    print(f"\n  속담 암기: {hits}/{len(PROVERBS)}\n")


def eval_fractal_variations(m: SentenceDecoderLanguageModel) -> None:
    print("=" * 75)
    print("4) 프랙탈 노이즈로 창조적 변주 (속담 mix-and-match)")
    print("=" * 75)
    fg = FractalTextGenerator(m)

    print("[noise 강도 sweep — 프롬프트 '가는', seed=42]")
    for ns in [0.0, 0.3, 0.6, 1.0]:
        r = fg.compose("가는", noise_strength=ns, chunk_scale=0.5,
                        k_anchors=6, seed=42)
        print(f"  noise={ns:.1f}  sub={r.substitution_rate:.2f}  {r.text!r}")

    print("\n[여러 시드 변주 — 프롬프트 '세 살', noise=0.7]")
    variants = fg.variations("세 살", n=5, base_seed=123,
                              noise_strength=0.7, chunk_scale=0.4,
                              k_anchors=5)
    for i, v in enumerate(variants, 1):
        print(f"  v{i}: {v.text!r}  (sub={v.substitution_rate:.2f})")

    print("\n[chunk_scale sweep — 프롬프트 '티끌 모아', noise=0.9]")
    for cs in [0.2, 0.5, 1.2, 2.5]:
        r = fg.compose("티끌 모아", noise_strength=0.9, chunk_scale=cs,
                        k_anchors=5, seed=7)
        print(f"  chunk={cs:3.1f}  {r.text!r}")


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

def run_repl(qa_model: SentenceDecoderLanguageModel,
             proverb_model: SentenceDecoderLanguageModel) -> None:
    fg_q = FractalTextGenerator(qa_model)
    fg_p = FractalTextGenerator(proverb_model)
    current = "qa"

    print("\nAXOL 한국어 대화.  /help 입력 시 명령어 보기.")
    while True:
        try:
            line = input(f"[{current}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/quit", "/exit"):
            break
        if line == "/help":
            print("  /qa               Q&A 모델로 전환")
            print("  /proverb          속담 모델로 전환")
            print("  /mix <프롬프트>    noise=0.7로 프랙탈 변주 생성")
            print("  /quit             종료")
            print("  그 외             프롬프트로 응답")
            continue
        if line == "/qa":
            current = "qa"
            continue
        if line == "/proverb":
            current = "proverb"
            continue
        if line.startswith("/mix"):
            prompt = line[len("/mix"):].strip()
            if not prompt:
                print("  사용법: /mix 프롬프트")
                continue
            fg = fg_q if current == "qa" else fg_p
            r = fg.compose(prompt, noise_strength=0.7, chunk_scale=0.5,
                           k_anchors=5)
            print(f"  변주: {r.text!r}  (sub={r.substitution_rate:.2f})")
            continue

        model = qa_model if current == "qa" else proverb_model
        res = model.generate(line, top_k=2)
        print(f"  응답: {res.text}   [conf={res.confidence:.3f}]")
        if len(res.alternatives) > 1:
            alt, score = res.alternatives[1]
            print(f"  2순위: {alt}   ({score:.3f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AXOL 한국어 실전 테스트")
    parser.add_argument("--repl", action="store_true",
                        help="REPL(대화형 모드) 실행")
    parser.add_argument("--skip-eval", action="store_true",
                        help="평가 섹션 건너뛰고 바로 REPL")
    args = parser.parse_args()

    qa_model = build_qa_model()
    proverb_model = build_proverb_model()

    if not args.skip_eval:
        eval_qa_recall(qa_model)
        eval_novel_prompts(qa_model)
        eval_proverb_recall(proverb_model)
        eval_fractal_variations(proverb_model)

    if args.repl:
        run_repl(qa_model, proverb_model)


if __name__ == "__main__":
    main()
