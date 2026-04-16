"""한국어 AXOL 대화 — 직접 사용 가능한 CLI.

실행
----
    # 내장 corpus 107쌍으로 바로 대화
    python demo/korean_chat.py

    # 저장된 모델 불러오기
    python demo/korean_chat.py --load mybot

    # 평가만 (REPL 없이)
    python demo/korean_chat.py --eval

REPL 명령어
-----------
    /help               명령어 목록
    /teach 입력 -> 출력  실시간으로 새 대화 가르치기
    /forget 0.5         기억 50% 약화
    /save PATH          모델 저장 (PATH.npz + PATH.json)
    /load PATH          저장된 모델 불러오기
    /mode snap|hybrid|creative   응답 전략 전환
    /alt                방금 응답의 2~3순위 대안 보기
    /report             학습 상태 보고
    /vary on|off        동일 프롬프트에 조금씩 변주 (hybrid 모드)
    /quit               종료
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
from axol.quantum.sentence_decoder import (  # noqa: E402
    HybridResponder,
    SentenceDecoderLanguageModel,
)

from demo.korean_corpus import CORPUS, all_texts  # noqa: E402


# ---------------------------------------------------------------------------
# 모델 구성
# ---------------------------------------------------------------------------

def build_model() -> SentenceDecoderLanguageModel:
    vocab = vocab_from_texts(all_texts())
    return SentenceDecoderLanguageModel(
        vocab=vocab,
        intent_dim=28,           # 한국어 + corpus 100+ 쌍을 위해 확장
        intent_window=8,
        regularization=1e-4,
        seed=0,
    )


def train_on_corpus(m: SentenceDecoderLanguageModel, epochs: int = 4) -> None:
    m.train_pairs(CORPUS, epochs=epochs)


# ---------------------------------------------------------------------------
# 간단한 평가
# ---------------------------------------------------------------------------

def quick_eval(m: SentenceDecoderLanguageModel) -> None:
    print("=" * 70)
    print(f"암기 검증 — 학습된 {len(CORPUS)}쌍 중 샘플 20쌍")
    print("=" * 70)
    import random
    rng = random.Random(0)
    sample = rng.sample(CORPUS, min(20, len(CORPUS)))
    hits = 0
    for q, expected in sample:
        res = m.generate(q)
        ok = res.text == expected
        hits += int(ok)
        mark = "✓" if ok else "✗"
        print(f"  {mark} {q:18s}  →  {res.text}")
    print(f"\n  {hits}/{len(sample)} 정확 재현  (intent Ω={m.omega:.3f})")
    print()

    print("=" * 70)
    print("학습 안 한 프롬프트 — 의미적 검색")
    print("=" * 70)
    novel = ["반가워요", "기분이 좋아", "너무 힘드네", "안녕히 가", "나 집에 갈래",
             "기쁘다", "피곤하네", "뭘 물어볼까"]
    for q in novel:
        res = m.generate(q, top_k=2)
        print(f"  {q!r:16} → {res.text!r:30} (conf={res.confidence:.3f})")
    print()


# ---------------------------------------------------------------------------
# Hybrid + 프랙탈 변주 데모
# ---------------------------------------------------------------------------

def show_hybrid_modes(m: SentenceDecoderLanguageModel) -> None:
    print("=" * 70)
    print("Hybrid 응답 모드 비교")
    print("=" * 70)
    hr_strict = HybridResponder(m, snap_threshold=0.95, blend_threshold=0.60)
    hr_vary = HybridResponder(m, snap_threshold=0.95, blend_threshold=0.60,
                              vary_known=True, vary_strength=0.25)
    hr_creative = HybridResponder(m, snap_threshold=1.01, blend_threshold=0.40,
                                   vary_known=True, vary_strength=0.3,
                                   blend_strength=0.7)

    prompts = [
        "안녕",                # 학습됨 → snap
        "오랜만이야",           # 학습됨 → snap
        "뭐 하고 놀까",          # 비슷한 게 있음 → blended
        "오늘 기분이 좀 그래",   # medium
        "빨간 코끼리가 춤춰",    # 완전 out-of-dist
    ]
    print(f"{'prompt':22s}  {'strict':20s}  {'vary':20s}  {'creative':20s}")
    print("-" * 90)
    for p in prompts:
        s = hr_strict.respond(p, seed=7).text
        v = hr_vary.respond(p, seed=7).text
        c = hr_creative.respond(p, seed=7).text
        print(f"  {p:20s}  {s[:18]:20s}  {v[:18]:20s}  {c[:18]:20s}")
    print()


def show_fractal_blends(m: SentenceDecoderLanguageModel) -> None:
    print("=" * 70)
    print("프랙탈 노이즈 변주 — 학습된 조각들의 창조적 조합")
    print("=" * 70)
    fg = FractalTextGenerator(m)
    for prompt in ["사랑해", "슬퍼", "고마워"]:
        print(f"[{prompt!r}]")
        for ns in [0.0, 0.4, 0.8]:
            r = fg.compose(prompt, noise_strength=ns,
                            chunk_scale=0.5, k_anchors=6, seed=42)
            print(f"  noise={ns:.1f}  {r.text!r}")
        print()


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

class ChatSession:
    """Stateful interactive session wrapping model + hybrid + history."""

    def __init__(self, m: SentenceDecoderLanguageModel) -> None:
        self.m = m
        self.fg = FractalTextGenerator(m)
        self.hybrid = HybridResponder(
            m, snap_threshold=0.92, blend_threshold=0.55,
            vary_known=True, vary_strength=0.2, blend_strength=0.6,
        )
        self.mode = "hybrid"    # "snap" | "hybrid" | "creative"
        self.last_response = None
        self.turn = 0

    def _respond_snap(self, prompt: str):
        return self.m.generate(prompt, top_k=3)

    def _respond_hybrid(self, prompt: str, seed: int | None):
        return self.hybrid.respond(prompt, seed=seed)

    def _respond_creative(self, prompt: str, seed: int | None):
        # Always apply fractal blend, even for known prompts.
        r = self.fg.compose(prompt, noise_strength=0.75, chunk_scale=0.5,
                            k_anchors=6, seed=seed)
        snap = self.m.generate(prompt, top_k=3)
        return type("X", (), {
            "text": r.text or snap.text,
            "confidence": snap.confidence,
            "mode": "creative",
            "alternatives": snap.alternatives,
            "substitution_rate": r.substitution_rate,
        })()

    def respond(self, prompt: str) -> None:
        self.turn += 1
        seed = self.turn * 997   # each turn gets a different seed
        if self.mode == "snap":
            res = self._respond_snap(prompt)
            text = res.text
            tag = f"snap conf={res.confidence:.2f}"
        elif self.mode == "creative":
            res = self._respond_creative(prompt, seed)
            text = res.text
            tag = f"creative sub={res.substitution_rate:.2f} conf={res.confidence:.2f}"
        else:
            res = self._respond_hybrid(prompt, seed)
            text = res.text
            tag = f"{res.mode} conf={res.confidence:.2f}"
            if getattr(res, "substitution_rate", 0.0) > 0:
                tag += f" sub={res.substitution_rate:.2f}"
        self.last_response = res
        print(f"AXOL: {text}")
        print(f"      [{tag}]")


def run_repl(session: ChatSession) -> None:
    print(f"\n한국어 대화 시작 — 모드 '{session.mode}'. /help 로 명령어 보기, /quit 종료.")
    while True:
        try:
            line = input(f"[{session.mode}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/quit", "/exit"):
            break

        if line == "/help":
            print("  /help                       명령어 목록")
            print("  /teach 입력 -> 출력          실시간 가르치기")
            print("  /forget 0.5                 기억 절반 약화 (0..1)")
            print("  /save PATH                  모델 저장")
            print("  /load PATH                  모델 불러오기")
            print("  /mode snap|hybrid|creative  응답 전략")
            print("  /vary on|off                동일 프롬프트 변주 on/off")
            print("  /alt                        2~3순위 대안 보기")
            print("  /report                     상태 보고")
            print("  /quit                       종료")
            continue

        if line.startswith("/teach"):
            body = line[len("/teach"):].strip()
            if "->" not in body:
                print("  사용법: /teach 입력 -> 출력")
                continue
            left, right = [x.strip() for x in body.split("->", 1)]
            if not left or not right:
                print("  입력과 출력 모두 있어야 합니다")
                continue
            session.m.teach(left, right)
            print(f"  [배웠음] {left!r} → {right!r}  (dict size={session.m.dictionary_size})")
            continue

        if line.startswith("/forget"):
            body = line[len("/forget"):].strip()
            try:
                f = float(body)
            except ValueError:
                print("  사용법: /forget 0.5")
                continue
            session.m.forget(f)
            print(f"  [망각] factor={f}")
            continue

        if line.startswith("/save"):
            body = line[len("/save"):].strip()
            if not body:
                print("  사용법: /save PATH")
                continue
            session.m.save(body)
            print(f"  [저장] {body}.npz + {body}.json")
            continue

        if line.startswith("/load"):
            body = line[len("/load"):].strip()
            if not body:
                print("  사용법: /load PATH")
                continue
            try:
                new_m = SentenceDecoderLanguageModel.load(body)
            except Exception as e:
                print(f"  [오류] {e}")
                continue
            session.m = new_m
            session.fg = FractalTextGenerator(new_m)
            session.hybrid = HybridResponder(
                new_m, snap_threshold=0.92, blend_threshold=0.55,
                vary_known=True, vary_strength=0.2, blend_strength=0.6,
            )
            print(f"  [불러옴] {body}  dict size={new_m.dictionary_size}  "
                  f"pairs={new_m.pairs_taught}")
            continue

        if line.startswith("/mode"):
            body = line[len("/mode"):].strip()
            if body not in ("snap", "hybrid", "creative"):
                print("  사용법: /mode snap|hybrid|creative")
                continue
            session.mode = body
            print(f"  [모드] {body}")
            continue

        if line.startswith("/vary"):
            body = line[len("/vary"):].strip()
            if body == "on":
                session.hybrid.vary_known = True
                print("  [변주 켜짐]")
            elif body == "off":
                session.hybrid.vary_known = False
                print("  [변주 꺼짐]")
            else:
                print("  사용법: /vary on|off")
            continue

        if line == "/alt":
            if session.last_response is None:
                print("  아직 응답이 없습니다")
                continue
            alts = getattr(session.last_response, "alternatives", [])
            if not alts:
                print("  대안이 없습니다")
                continue
            for i, (text, score) in enumerate(alts[:3], 1):
                print(f"  {i}순위 ({score:.3f}): {text}")
            continue

        if line == "/report":
            print(f"  dict size       : {session.m.dictionary_size}")
            print(f"  pairs taught    : {session.m.pairs_taught}")
            print(f"  intent Ω        : {session.m.omega:.3f}")
            print(f"  intent Φ        : {session.m.phi:.3f}")
            print(f"  mode            : {session.mode}")
            print(f"  vary_known      : {session.hybrid.vary_known}")
            continue

        # 일반 프롬프트 → 응답
        session.respond(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AXOL 한국어 실전 대화")
    parser.add_argument("--load", metavar="PATH", help="저장된 모델 불러오기")
    parser.add_argument("--save", metavar="PATH",
                        help="학습 후 모델 저장")
    parser.add_argument("--eval", action="store_true",
                        help="평가만 출력하고 REPL 건너뛰기")
    parser.add_argument("--no-train", action="store_true",
                        help="내장 corpus 학습 없이 실행 (fresh model)")
    parser.add_argument("--epochs", type=int, default=4,
                        help="corpus 학습 epoch 수")
    args = parser.parse_args()

    if args.load:
        m = SentenceDecoderLanguageModel.load(args.load)
        print(f"[load] {args.load}  dict size={m.dictionary_size}  "
              f"pairs={m.pairs_taught}")
    else:
        m = build_model()
        if not args.no_train:
            print(f"[train] corpus {len(CORPUS)}쌍 × {args.epochs} epochs ...")
            train_on_corpus(m, epochs=args.epochs)
            print(f"[train] done  dict={m.dictionary_size}  "
                  f"intent Ω={m.omega:.3f}")

    if args.save:
        m.save(args.save)
        print(f"[save] {args.save}.npz + {args.save}.json")

    if args.eval:
        quick_eval(m)
        show_hybrid_modes(m)
        show_fractal_blends(m)
        return

    session = ChatSession(m)
    run_repl(session)


if __name__ == "__main__":
    main()
