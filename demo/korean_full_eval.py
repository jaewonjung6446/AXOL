"""한국어 실전 테스트 — 8개 섹션 상세 평가.

이 스크립트는 사용자가 구축한 한국어 AXOL 시스템의 실제 성능을
여러 각도에서 측정합니다.  각 섹션은 독립적이며 JSON 결과를
``korean_eval_results.json`` 에 저장합니다.

섹션:
    1. 암기 recall (샘플 40쌍, 전체 cut-off)
    2. 학습 안 한 prompt semantic retrieval (25 프롬프트)
    3. 어미 변형 일반화 (자모 vs 음절 비교)
    4. Hybrid 모드 3종 (strict/hybrid/creative) 출력 비교
    5. 프랙탈 블렌드 vs n-gram 필터 효과
    6. Save/load round-trip (완전 일치 확인)
    7. 실시간 학습 (10쌍 추가 후 즉시 recall)
    8. Stress: 100 무작위 prompt의 confidence 분포

실행
----
    python demo/korean_full_eval.py                 # syllable 모드
    python demo/korean_full_eval.py --jamo          # 자모 모드
    python demo/korean_full_eval.py --compare       # 두 모드 동시 비교
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from collections import Counter
from statistics import mean, median, stdev

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from axol.quantum import jamo  # noqa: E402
from axol.quantum.conversation import vocab_from_texts  # noqa: E402
from axol.quantum.fractal_text import FractalTextGenerator  # noqa: E402
from axol.quantum.ngram_filter import NgramFilter  # noqa: E402
from axol.quantum.sentence_decoder import (  # noqa: E402
    HybridResponder,
    SentenceDecoderLanguageModel,
)

from demo.korean_corpus import CORPUS, all_texts  # noqa: E402


# ---------------------------------------------------------------------------
# 모델 빌드 (자모 / 음절)
# ---------------------------------------------------------------------------

def build(use_jamo: bool, epochs: int = 4) -> SentenceDecoderLanguageModel:
    if use_jamo:
        texts = [jamo.decompose(t) for t in all_texts()]
        pairs = jamo.decompose_pairs(CORPUS)
    else:
        texts = all_texts()
        pairs = CORPUS
    vocab = vocab_from_texts(texts)
    m = SentenceDecoderLanguageModel(
        vocab=vocab, intent_dim=28, intent_window=8,
        regularization=1e-4, seed=0,
    )
    t0 = time.perf_counter()
    m.train_pairs(pairs, epochs=epochs)
    return m, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Helpers — input/output conversion per mode
# ---------------------------------------------------------------------------

def _in(text: str, use_jamo: bool) -> str:
    return jamo.decompose(text) if use_jamo else text


def _out(text: str, use_jamo: bool) -> str:
    return jamo.compose_safely(text) if use_jamo else text


# ---------------------------------------------------------------------------
# 섹션 1: 암기
# ---------------------------------------------------------------------------

def section_memorisation(m, use_jamo: bool, results: dict) -> None:
    print("=" * 78)
    print("[1] 암기 recall — 학습한 쌍을 얼마나 정확히 돌려주는가")
    print("=" * 78)
    rng = random.Random(0)
    sample = rng.sample(CORPUS, min(40, len(CORPUS)))
    hits = 0
    mismatches = []
    for q, expected in sample:
        res = m.generate(_in(q, use_jamo))
        got = _out(res.text, use_jamo)
        if got == expected:
            hits += 1
        else:
            mismatches.append((q, expected, got, res.confidence))
    rate = hits / len(sample)
    print(f"  {hits}/{len(sample)} = {rate:.1%}")
    if mismatches:
        print(f"  mismatches ({len(mismatches)}):")
        for q, exp, got, conf in mismatches[:5]:
            print(f"    {q!r:14} expected={exp!r}")
            print(f"                    got={got!r}  (conf={conf:.3f})")
    results["memorisation"] = {
        "n": len(sample), "hits": hits, "rate": rate,
        "mismatches": [
            {"q": q, "expected": e, "got": g, "conf": c}
            for q, e, g, c in mismatches
        ],
    }


# ---------------------------------------------------------------------------
# 섹션 2: novel prompt 의미 검색
# ---------------------------------------------------------------------------

NOVEL_PROMPTS = [
    "반가워요", "오랜만입니다", "너무 고마워요", "죄송해요 진짜",
    "기분이 꿀꿀해", "마음이 아파요", "지금 너무 행복해", "화가 많이 나",
    "날씨 어떠냐", "오늘 몇 시쯤이야", "나 집에 갈까", "지금 뭐 할까",
    "아 짜증", "진짜 싫어", "졸려 죽겠다", "커피 한 잔 할까",
    "너 뭐 할 줄 알아", "내 이름 기억해", "안녕히 주무세요",
    "새해 목표", "오늘 피곤", "회의 늦었어", "스트레스 풀고 싶어",
    "사랑하는 사람", "외롭지 않아?",
]


def section_retrieval(m, use_jamo: bool, results: dict) -> None:
    print("=" * 78)
    print("[2] 학습 안 한 프롬프트 — 의미 검색")
    print("=" * 78)
    records = []
    conf_list = []
    for p in NOVEL_PROMPTS:
        res = m.generate(_in(p, use_jamo), top_k=3)
        got = _out(res.text, use_jamo)
        conf_list.append(res.confidence)
        records.append({"prompt": p, "reply": got, "conf": res.confidence})
        print(f"  {p!r:20} → {got!r:40} (conf={res.confidence:.3f})")
    print(f"\n  confidence  mean={mean(conf_list):.3f}  "
          f"median={median(conf_list):.3f}  "
          f"min={min(conf_list):.3f}  max={max(conf_list):.3f}")
    results["retrieval"] = {
        "records": records,
        "stats": {
            "mean_conf": mean(conf_list),
            "median_conf": median(conf_list),
            "min_conf": min(conf_list),
            "max_conf": max(conf_list),
        },
    }


# ---------------------------------------------------------------------------
# 섹션 3: 어미 변형 일반화 (자모 vs 음절 직접 비교)
# ---------------------------------------------------------------------------

SUFFIX_TESTS = [
    # (학습된 표현, 어미가 변형된 prompt, 기대 응답과 비슷해야 함)
    ("안녕",   "안녕하세요",   "안녕"),
    ("안녕",   "안녕히",       "안녕"),
    ("고마워", "고마워요",     "천만"),     # contains "천만"
    ("미안해", "미안합니다",   "괜찮"),
    ("기뻐",   "기뻐요",       "기쁨"),
    ("슬퍼",   "슬프다",       "이야기"),
    ("잘 가", "잘 가세요",     "안녕"),
]


def section_suffix_generalisation(results: dict) -> None:
    print("=" * 78)
    print("[3] 어미 변형 일반화 — 자모 모드 vs 음절 모드 직접 비교")
    print("=" * 78)

    m_syl, _ = build(use_jamo=False)
    m_jam, _ = build(use_jamo=True)

    syl_hits = jam_hits = 0
    per_case = []
    print(f"  {'원형':8s}  {'변형 프롬프트':15s}  "
          f"  syllable response             jamo response")
    for trained, variant, substring_hint in SUFFIX_TESTS:
        r_s = m_syl.generate(variant, top_k=3)
        r_j = m_jam.generate(jamo.decompose(variant), top_k=3)
        out_s = r_s.text
        out_j = jamo.compose_safely(r_j.text)
        # "성공" 여부는 응답이 hint substring을 포함하는가로 얼추 측정
        ok_s = substring_hint in out_s
        ok_j = substring_hint in out_j
        syl_hits += int(ok_s)
        jam_hits += int(ok_j)
        per_case.append({
            "trained": trained, "variant": variant, "hint": substring_hint,
            "syllable": out_s, "syl_hit": ok_s, "syl_conf": r_s.confidence,
            "jamo": out_j,     "jam_hit": ok_j, "jam_conf": r_j.confidence,
        })
        mark_s = "✓" if ok_s else "✗"
        mark_j = "✓" if ok_j else "✗"
        print(f"  {trained:8s}  {variant:15s}  {mark_s} {out_s[:28]:28s}  "
              f"{mark_j} {out_j[:28]:28s}")

    print(f"\n  syllable hits: {syl_hits}/{len(SUFFIX_TESTS)}")
    print(f"  jamo     hits: {jam_hits}/{len(SUFFIX_TESTS)}")
    results["suffix_generalisation"] = {
        "syllable_hits": syl_hits,
        "jamo_hits": jam_hits,
        "total": len(SUFFIX_TESTS),
        "per_case": per_case,
    }


# ---------------------------------------------------------------------------
# 섹션 4: Hybrid 3모드 비교
# ---------------------------------------------------------------------------

HYBRID_PROMPTS = [
    ("안녕", "trained"),
    ("고마워", "trained"),
    ("너무 힘드네", "novel"),
    ("기분 좋아", "novel"),
    ("빨간 코끼리가 춤춰", "out-of-dist"),
    ("asdf zxcv", "out-of-dist"),
]


def section_hybrid_modes(m, use_jamo: bool, results: dict) -> None:
    print("=" * 78)
    print("[4] Hybrid 3모드 — strict / hybrid / creative")
    print("=" * 78)
    hr_strict = HybridResponder(m, snap_threshold=0.95, blend_threshold=0.60,
                                 vary_known=False)
    hr_hybrid = HybridResponder(m, snap_threshold=0.92, blend_threshold=0.55,
                                 vary_known=True, vary_strength=0.2,
                                 blend_strength=0.6)
    hr_creative = HybridResponder(m, snap_threshold=1.01, blend_threshold=0.40,
                                   vary_known=True, vary_strength=0.3,
                                   blend_strength=0.7)
    records = []
    print(f"  {'prompt':22s}  {'kind':12s}  {'strict':22s}  {'hybrid':22s}  {'creative':22s}")
    for p, kind in HYBRID_PROMPTS:
        p_in = _in(p, use_jamo)
        rs = hr_strict.respond(p_in, seed=7)
        rh = hr_hybrid.respond(p_in, seed=7)
        rc = hr_creative.respond(p_in, seed=7)
        ts = _out(rs.text, use_jamo)
        th = _out(rh.text, use_jamo)
        tc = _out(rc.text, use_jamo)
        records.append({
            "prompt": p, "kind": kind,
            "strict":   {"text": ts, "mode": rs.mode, "conf": rs.confidence},
            "hybrid":   {"text": th, "mode": rh.mode, "conf": rh.confidence,
                         "sub": rh.substitution_rate},
            "creative": {"text": tc, "mode": rc.mode, "conf": rc.confidence,
                         "sub": rc.substitution_rate},
        })
        print(f"  {p:22s}  {kind:12s}  {ts[:20]:22s}  {th[:20]:22s}  {tc[:20]:22s}")
    results["hybrid_modes"] = records


# ---------------------------------------------------------------------------
# 섹션 5: 프랙탈 with vs without n-gram 필터
# ---------------------------------------------------------------------------

def section_fractal_vs_filter(m, use_jamo: bool, results: dict) -> None:
    print("=" * 78)
    print("[5] 프랙탈 블렌드 — n-gram 필터 on/off")
    print("=" * 78)
    pairs = jamo.decompose_pairs(CORPUS) if use_jamo else CORPUS
    filt = NgramFilter([a for _, a in pairs], n=2)
    fg = FractalTextGenerator(m)

    probes = ["사랑해", "힘들어", "고마워", "미안해"]
    records = []
    for p in probes:
        p_in = _in(p, use_jamo)
        # raw variant
        cands = fg.variations(p_in, n=6, base_seed=100,
                               noise_strength=0.7, chunk_scale=0.5,
                               k_anchors=6)
        cand_texts = [c.text for c in cands]
        raw = cand_texts[0]
        best, best_score = filt.pick_best(cand_texts)
        raw_out = _out(raw, use_jamo)
        best_out = _out(best, use_jamo)
        raw_score = filt.score(raw)
        records.append({
            "prompt": p,
            "raw": raw_out, "raw_ngram_score": raw_score,
            "filtered": best_out, "filtered_ngram_score": best_score,
            "variants": [_out(t, use_jamo) for t in cand_texts],
        })
        print(f"  prompt {p!r}")
        print(f"    raw      (score={raw_score:.2f})  {raw_out!r}")
        print(f"    filtered (score={best_score:.2f})  {best_out!r}")
    results["fractal_vs_filter"] = records


# ---------------------------------------------------------------------------
# 섹션 6: save/load round-trip
# ---------------------------------------------------------------------------

def section_save_load(m, use_jamo: bool, results: dict) -> None:
    print("=" * 78)
    print("[6] save / load round-trip")
    print("=" * 78)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model")
        m.save(path)
        loaded = SentenceDecoderLanguageModel.load(path)

        probes = random.Random(7).sample(CORPUS, min(10, len(CORPUS)))
        hits = 0
        for q, expected in probes:
            q_in = _in(q, use_jamo)
            exp_in = _in(expected, use_jamo)
            res = loaded.generate(q_in)
            if res.text == exp_in:
                hits += 1
    ok = hits == len(probes)
    print(f"  save/load 후 {hits}/{len(probes)} 정확 재현  ({'✓ PASS' if ok else '✗ FAIL'})")
    results["save_load"] = {"n": len(probes), "hits": hits, "passed": ok}


# ---------------------------------------------------------------------------
# 섹션 7: 실시간 학습
# ---------------------------------------------------------------------------

RUNTIME_TEACH = [
    ("이거 뭐야",    "설명해 드릴게요"),
    ("나 삐졌어",   "제가 기분 풀어드릴게요"),
    ("산책 갈까",    "상쾌한 바람이 좋을 거예요"),
    ("치킨 먹을까",  "오늘은 치맥 어떠세요"),
    ("졸음 쏟아져",  "잠깐 눈 감고 쉬어 보세요"),
    ("회의 준비",    "잘 준비하실 수 있을 거예요"),
    ("발표 잘했어", "정말 대단하세요 축하드려요"),
    ("친구 만났어", "즐거운 시간이었겠네요"),
    ("영화 봤어",   "어떤 영화였는지 궁금해요"),
    ("비행기 타",    "즐거운 여행 되세요"),
]


def section_runtime_teaching(m, use_jamo: bool, results: dict) -> None:
    print("=" * 78)
    print("[7] 실시간 학습 — 10쌍 추가 후 즉시 recall")
    print("=" * 78)
    before = m.dictionary_size
    for q, a in RUNTIME_TEACH:
        m.teach(_in(q, use_jamo), _in(a, use_jamo))
    after = m.dictionary_size
    hits = 0
    for q, a in RUNTIME_TEACH:
        res = m.generate(_in(q, use_jamo))
        if res.text == _in(a, use_jamo):
            hits += 1
    print(f"  dict size: {before} → {after}")
    print(f"  새 쌍 recall: {hits}/{len(RUNTIME_TEACH)}")
    results["runtime_teaching"] = {
        "before": before, "after": after,
        "recalled": hits, "total": len(RUNTIME_TEACH),
    }


# ---------------------------------------------------------------------------
# 섹션 8: confidence 분포
# ---------------------------------------------------------------------------

def section_confidence_distribution(m, use_jamo: bool, results: dict) -> None:
    print("=" * 78)
    print("[8] 100 랜덤 프롬프트 confidence 분포")
    print("=" * 78)
    rng = random.Random(42)
    # corpus의 prompt 반 + 일부러 섞인 무작위 문자열 반
    known = [q for q, _ in CORPUS]
    junk = [
        "".join(rng.choice("가나다라마바사아자차카타파하 ") for _ in range(rng.randint(4, 10)))
        for _ in range(50)
    ]
    prompts = rng.sample(known, 50) + junk
    rng.shuffle(prompts)

    confs = []
    for p in prompts:
        res = m.generate(_in(p, use_jamo))
        confs.append(res.confidence)

    def bucket(c: float) -> str:
        if c >= 0.95: return "0.95+"
        if c >= 0.80: return "0.80-0.95"
        if c >= 0.60: return "0.60-0.80"
        if c >= 0.40: return "0.40-0.60"
        return "< 0.40"

    counts = Counter(bucket(c) for c in confs)
    order = ["0.95+", "0.80-0.95", "0.60-0.80", "0.40-0.60", "< 0.40"]
    for b in order:
        c = counts.get(b, 0)
        bar = "█" * c
        print(f"  {b:12s}  {c:3d}  {bar}")
    print(f"\n  overall  mean={mean(confs):.3f}  "
          f"median={median(confs):.3f}  "
          f"stdev={stdev(confs):.3f}")
    results["confidence_distribution"] = {
        "n": len(prompts),
        "buckets": {b: counts.get(b, 0) for b in order},
        "mean": mean(confs), "median": median(confs),
        "stdev": stdev(confs),
    }


# ---------------------------------------------------------------------------
# 메인 — 하나의 mode 실행
# ---------------------------------------------------------------------------

def run_full(use_jamo: bool, epochs: int = 4) -> dict:
    mode = "자모 (jamo)" if use_jamo else "음절 (syllable)"
    print(f"\n{'#' * 78}")
    print(f"#  AXOL 한국어 상세 평가 — 모드: {mode}")
    print(f"{'#' * 78}\n")

    m, train_time = build(use_jamo=use_jamo, epochs=epochs)
    print(f"[build] corpus {len(CORPUS)}쌍 × {epochs} epochs  "
          f"→ train {train_time:.2f}s  dict={m.dictionary_size}  "
          f"intent Ω={m.omega:.3f}\n")

    results: dict = {
        "mode": "jamo" if use_jamo else "syllable",
        "train_time_s": train_time,
        "corpus_size": len(CORPUS),
        "epochs": epochs,
        "dict_size": m.dictionary_size,
        "intent_omega": m.omega,
    }

    section_memorisation(m, use_jamo, results)
    print()
    section_retrieval(m, use_jamo, results)
    print()
    section_hybrid_modes(m, use_jamo, results)
    print()
    section_fractal_vs_filter(m, use_jamo, results)
    print()
    section_save_load(m, use_jamo, results)
    print()
    section_runtime_teaching(m, use_jamo, results)
    print()
    section_confidence_distribution(m, use_jamo, results)
    print()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="한국어 AXOL 상세 평가")
    parser.add_argument("--jamo", action="store_true",
                        help="자모 모드로 평가 (기본: 음절)")
    parser.add_argument("--compare", action="store_true",
                        help="음절 모드와 자모 모드 동시 평가")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--out", metavar="PATH",
                        default="korean_eval_results.json",
                        help="결과 JSON 저장 경로")
    args = parser.parse_args()

    # 섹션 3(어미 일반화)은 독립적이므로 한 번만 실행
    suffix_results: dict = {}
    section_suffix_generalisation(suffix_results)
    print()

    if args.compare:
        syl_results = run_full(use_jamo=False, epochs=args.epochs)
        jam_results = run_full(use_jamo=True, epochs=args.epochs)
        output = {
            "syllable": syl_results,
            "jamo": jam_results,
            "suffix_generalisation": suffix_results["suffix_generalisation"],
        }
    else:
        output = run_full(use_jamo=args.jamo, epochs=args.epochs)
        output["suffix_generalisation"] = (
            suffix_results["suffix_generalisation"]
        )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
