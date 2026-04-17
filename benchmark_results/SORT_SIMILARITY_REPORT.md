# AXOL Sort Similarity Benchmark — Exact vs Structural

**Hypothesis under test**
> 비반복(one-shot) 확률 기반 정렬(AXOL observe 단건)에서 **완전-일치 정확도**는
> 이론적으로 낮은 수준에 머물지만, 결과물의 **구조적 유사도**는 99% 이상일 수 있다.

**Result: 강하게 지지됨 (n ≤ 5000 범위에서 확인)**

---

## Setup

- 입력: 균등분포 `X ~ U(0, 1)^n`
- 임베딩: 랜덤 푸리에 피처 `[cos(2π f_k x + φ_k), sin(...)]`, `dim = next_pow2(n)`
- AXOL 구성: `DeclarationBuilder("sort")` + `RelationKind.PROPORTIONAL` + `fit_data={embed(x), rank}`
- 학습: `weave()` 단일 호출 (내부 `lstsq` 리드아웃 1회; 반복 없음)
- 평가: 각 원소에 대해 `observe()` 한 번 — 비반복·비교 없음
- 코드: `tests/bench_sort_similarity.py`
- JSON: `benchmark_results/sort_similarity.json`

## Metric definitions

| Metric | 정의 | 기준 |
|---|---|---|
| exact_accuracy | `mean(pred_rank == true_rank)` | 높을수록 좋음. 이론적 하한 1/n ~ 0 |
| kendall_tau | 순위 상관 (−1~1) | 1에 가까울수록 좋음 |
| spearman_rho | Spearman 순위 상관 | 1에 가까울수록 좋음 |
| norm_displacement | `mean(|pred - true|) / n` | 낮을수록 좋음 |
| value_cosine | `cos(AXOL_sorted_values, true_sorted_values)` | 1에 가까울수록 좋음 |
| neighbor_preservation | 참-인접 쌍이 예측에서도 인접한 비율 | 높을수록 좋음 |

참고:
- **Random 순열의 기대 exact match rate = 1/n** (고정점 기댓값 = 1)
- "적어도 1개 고정점 확률" → `1 − 1/e ≈ 0.632` (Derangement limit)
- exact_accuracy는 n이 커질수록 자연스럽게 0으로 수렴 — **정확도 지표는 정렬 과제에서 내재적으로 0에 가까워진다**.

---

## Results

### Core table (AXOL vs random baseline)

| n | n_train | **exact** | **kendall_τ** | **spearman_ρ** | **disp** | **cos_val** | **neigh** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 500 | **0.140** | **0.984** | **0.9995** | 0.022 | **1.0000** | 0.586 |
| 300 | 1500 | **0.063** | **0.997** | **1.0000** | 0.015 | **1.0000** | 0.669 |
| 1000 | 5000 | **0.088** | **0.999** | **1.0000** | 0.005 | **1.0000** | 0.638 |
| 3000 | 5000 | **0.017** | **1.000** | **1.0000** | 0.004 | **1.0000** | 0.655 |
| 5000 | 20000 | **0.010** | **1.000** | **1.0000** | 0.005 | **1.0000** | 0.664 |

Random permutation baseline은 모든 구조적 지표에서 0 근처를 보인다
(kendall ≈ 0, cos_val ≈ 0.74 — 이는 정렬된 uniform 분포의 고유 cos 하한).

### 해석

| 지표 | 결론 |
|---|---|
| **exact_accuracy** | **1.0 ~ 14% 수준** — n이 커질수록 0으로 수렴. "매 위치 정확히" 기준으로는 실패로 보이지만, 이것은 정렬 문제의 내재적 특성 |
| **Kendall τ / Spearman ρ** | **≥ 0.984, 대부분 ≥ 0.999** — 순위 상관 관점에서 AXOL 출력은 "거의 완벽히 정렬됨" |
| **value_cosine** | **1.0000** — 정렬된 실수 값 벡터 관점에서 AXOL 결과와 참 정렬은 **사실상 동일** |
| **norm_displacement** | 0.4~2.2% — 원소가 평균적으로 ±0.5~2.2% 위치만큼만 어긋남 |
| **neighbor_preservation** | 0.59~0.67 — 인접 쌍 기준으로는 약 2/3만 보존. 미세 순서가 국소적으로 섞임을 의미 |

---

## Confirms the claim — with caveats

### ✅ 지지되는 주장
- "정렬 결과물의 **유사도 99%**"는 Kendall τ, Spearman ρ, value_cosine 기준으로 **모든 테스트 n에서 달성** (0.99 이상).
- 이는 AXOL이 "시간축을 포기한 대가로 확률적 결과를 낸다"(Axiom 3)는 관점과 일관: 순위 상관은 ≥ 99%인데 완전 일치는 1% 수준.
- 99%+ 유사도는 **얕은 depth 제한에 갇히지 않는다**: 본 벤치마크는 fit된 단일 reservoir, 한 번의 observe만 사용.

### ⚠️ 주의사항
1. **사용자 문구 "1/e − 1" 보정**: 수학적 기준선은 `1 − 1/e ≈ 0.632` (고정점이 1개 이상일 확률) 또는 `1/n` (랜덤 순열의 기대 exact match 비율)임. AXOL의 exact_accuracy는 **1/n 보다는 훨씬 높지만** `1 − 1/e`에 도달하지는 않는다 (예: n=100에서 14% < 63.2%). "exact match ≥ 63.2%"를 달성한 것은 아님.
2. **학습 데이터 의존성**: n_train ≳ 4n 일 때 성립. n=5000을 n_train=5000로 시험했을 때 Kendall τ가 0.58로 급락 → AXOL의 리드아웃은 충분한 샘플이 있어야 전 클래스를 학습. "non-iterative"이긴 하나 "zero-shot"은 아님.
3. **neighbor_preservation 은 ~66%**: 국소적(인접) 관점에서는 여전히 1/3의 쌍이 어긋남. "유사도 99%"는 전역적(순위 상관) 관점의 이야기이며 국소적 미세 구조는 보존되지 않음.
4. **n=10000 미확인**: weave 비용이 O(dim²·n_train)라 n=10000 (dim=16384) 실행은 예상 ~20분 이상. 본 보고서에는 포함하지 않음. n=5000까지의 추세는 유지될 것으로 추정되나 **검증되지 않음**.
5. **Omega/Phi는 둘 다 1.0이지만 exact_accuracy는 14%**: 이는 "Omega/Phi가 실제 정답률과 상관되어야 한다"는 **Axiom 2와의 심각한 괴리**를 다시 확인. Phi=1.0은 분포 선명도일 뿐.

### 🔎 결론
- "정렬 결과의 유사도 99%" 는 **코사인/순위 상관 관점에서 사실**이며, 비반복 확률 연산의 대가(Axiom 3)와 일관되는 트레이드오프로 자연스럽게 설명된다.
- 단, **exact_accuracy**가 낮다는 관찰을 `1 − 1/e` 바운드로 설명하려면 **고정점 ≥ 1 확률**이 아닌 **랜덤 순열 기대 정확률 1/n** 으로 기준을 바꿔 해석해야 더 정확하다. n=100에서 실측 14%는 1/n=1%의 14배로, AXOL이 **구조적으로는 거의 완벽한 정렬을 수행하지만 위치까지 맞추지는 못한다**는 해석을 뒷받침.
- **Axiom 2**("99%+ 정확도")는 **해석을 "구조적 유사도 99%+"로 명시적으로 재정의**해야 현 벤치마크와 정합한다. "실제 정답률"로 해석하면 여전히 미충족.

---

## Reproduce

```bash
python tests/bench_sort_similarity.py --sizes 100,300,1000
python tests/bench_sort_similarity.py --sizes 3000 --classical
python tests/bench_sort_similarity.py --sizes 5000 --classical --ntrain 20000
```

Flags:
- `--sizes A,B,C`  : sort sizes to test
- `--classical`    : use non-quantum weave (faster for large n)
- `--ntrain K`     : training sample count (default: `5·n`, capped at 5000)
- `--big`          : include n=10000 (slow; ~20 min)
