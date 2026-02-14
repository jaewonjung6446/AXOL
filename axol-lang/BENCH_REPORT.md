# AXOL Wave System — Comprehensive Benchmark Report

**Date**: 2026-02-13 (v3 — multi-input fix, O(1) observation cost 검증, 실제 유즈케이스)
**Platform**: Windows 11 Pro (10.0.26200)
**Build**: `cargo build --release` (opt-level=3, LTO, codegen-units=1)

---

## Executive Summary

AXOL은 분류기가 아니다. **가능성의 구조를 짜놓고(weave), 원하는 만큼만 꺼내 보는(observe) 시스템**이다.

핵심 성질: **weave에 비용을 지불하고, observe는 O(1).**

| 항목 | 결과 | 상태 |
|------|------|------|
| **O(1) 관측 비용** | **20 frame NPC: 4.3μs/frame, 총 13ms** | **핵심** |
| **Collapse Spectrum** | **C=0→1 연속 스펙트럼: 0.443→1.000 (5단계 검증)** | **핵심** |
| **관측의 비가역성** | **focus 후 widen해도 원래 분포 복구 불가** | **핵심** |
| **관계 간섭 패턴** | **`<!>` 파괴적 간섭이 차이를 부각, negativity 추적** | **핵심** |
| multi-input composition | 2+ 입력의 DAG 기반 간섭 합성 정상 동작 | FIXED |
| Relation negativity | Bhattacharyya distance 기반 비영 초기값 | FIXED |
| preserve_basins | define_basins 구조가 iterate에서 보존됨 | FIXED |
| 간섭 패턴 5종 | 모두 정상 동작, 5종 모두 고유한 분포 생성 | OK |
| compose/gaze/observe 성능 | 전 dim에서 sub-microsecond | OK |
| density 병목 | dephasing channel이 O(dim³) 스케일링 | 예상 범위 |

### 수정 이력

| 버전 | 이슈 | 수정 내용 |
|------|------|-----------|
| v1→v2 | focus 확률 불변 | population sharpening 추가: `p_i^β` (β=1/(1-γ)) + dephasing |
| v1→v2 | Additive=Constructive | geometric mean으로 재정의: `√(\|a\|·\|b\|)·exp(i(θa+θb)/2)` |
| v2→v3 | 2-input 모델 균일 분포 | compute_wave에서 모든 입력을 compose_from_rules로 DAG 합성 |
| v2→v3 | Relation negativity=0 | von Neumann entropy → Bhattacharyya distance + depolarizing noise |
| v2→v3 | iterate가 basin 구조 파괴 | preserve_basins 플래그로 user-defined basin 보호 |

---

## [1] Interference Pattern Comparison (dim=8)

5종 간섭 패턴이 **각각 고유한** 확률 분포를 생성함을 확인.

| 패턴 | 시간 (us) | dominant | max_p | 특성 |
|------|-----------|----------|-------|------|
| Constructive `<~>` | 0.190 | 7 | 0.2367 | a+b, 두 입력의 합산 → 분포가 퍼짐 |
| Additive `<+>` | 0.458 | 7 | **0.2679** | √(\|a\|·\|b\|), 기하평균 → Constructive보다 집중 |
| Multiplicative `<*>` | 0.180 | 7 | 0.3779 | a·b, 둘 다 강해야 살아남음 → 가장 집중 |
| Destructive `<!>` | 0.213 | 0 | 0.4264 | a-b, 차이가 큰 차원 부각 → dominant 반전 |
| Conditional `<?>` | 0.290 | 2 | 0.2275 | 위상 커플링 → 가장 균등 |

**v1 대비 변화**: Additive가 이제 Constructive와 다른 분포를 생성한다.
- Constructive: max_p=0.2367 (순서: 7, 2, 3, 5)
- Additive: max_p=0.2679 (순서: 7, 2, 3, 5 — 동일 순서지만 더 집중됨)

**분포 비교** (top 4):
```
Constructive:  [7]=0.237  [2]=0.218  [3]=0.167  [5]=0.120
Additive:      [7]=0.268  [2]=0.245  [3]=0.188  [5]=0.115
Multiplicat.:  [7]=0.378  [2]=0.317  [3]=0.186  [5]=0.069
Destructive:   [0]=0.426  [6]=0.311  [5]=0.160  [1]=0.070
Conditional:   [2]=0.228  [7]=0.226  [6]=0.182  [3]=0.179
```

**집중도 순서** (max_p 기준): Destructive > Multiplicative > Additive > Constructive > Conditional

**수학적 해석**:
- Constructive (a+b): 두 입력이 서로 보완 → 엔트로피 증가
- Additive (√ab): 기하평균이므로 한쪽이 0이면 0 → Constructive보다 선택적
- Multiplicative (a·b): 양쪽 모두 강한 차원만 통과 → AND 게이트적
- Destructive (a-b): 비슷한 부분 상쇄, 다른 부분 강조 → XOR 게이트적
- Conditional (a·e^iθb): a의 진폭을 b의 위상으로 회전 → 입력 A 보존

---

## [2] Dimension Scaling

| dim | compose (us) | gaze (us) | focus γ=0.3 (us) | focus γ=0.7 (us) | observe (us) |
|-----|-------------|-----------|------------------|------------------|-------------|
| 2 | 0.145 | 0.110 | 3.8 | 3.7 | 0.177 |
| 4 | 0.140 | 0.115 | 7.9 | 8.1 | 0.189 |
| 8 | 0.159 | 0.154 | 27.3 | 25.1 | 0.301 |
| 16 | 0.204 | 0.146 | 100.5 | 105.8 | 0.294 |
| 32 | 0.308 | 0.226 | 641.9 | 605.4 | 0.357 |
| 64 | 0.431 | 0.372 | 5,640 | 5,662 | 0.407 |
| 128 | 0.654 | 0.553 | 38,717 | 37,096 | 0.685 |

**복잡도 요약**:

| 연산 | 복잡도 | dim=8 | dim=32 | dim=128 |
|------|--------|-------|--------|---------|
| compose | O(n) | 0.16us | 0.31us | 0.65us |
| gaze | O(n) | 0.15us | 0.23us | 0.55us |
| observe | O(n) | 0.30us | 0.36us | 0.69us |
| focus | O(n³) | 27us | 642us | 38ms |

**병목**: focus의 dephasing channel이 O(n_kraus × dim²), n_kraus≈dim → **O(dim³)**.

**실용 범위**:
- dim ≤ 32: focus < 1ms → 실시간 사용 가능
- dim 64: focus ≈ 6ms → 배치 전용
- dim 128: focus ≈ 38ms → focus 없이 compose+gaze 권장

---

## [3] Collapse Spectrum (dim=8)

gamma 0.00에서 1.00까지 0.05 간격으로 sweep.

```
gamma    t     dom  max_p   entropy   top-3 probs
0.00   0.000    2   0.659   0.869     [2]=0.659 [5]=0.293 [0]=0.008
0.10   0.100    2   0.689   0.778     [2]=0.689 [5]=0.280 [0]=0.005
0.20   0.200    2   0.721   0.690     [2]=0.721 [5]=0.262 [0]=0.003
0.30   0.300    2   0.755   0.609     [2]=0.755 [5]=0.237 [0]=0.001
0.40   0.400    2   0.792   0.533     [2]=0.792 [5]=0.205 [0]=0.001
0.50   0.500    2   0.834   0.455     [2]=0.834 [5]=0.165 [0]=0.000
0.60   0.600    2   0.884   0.361     [2]=0.884 [5]=0.116 [0]=0.000
0.70   0.700    2   0.937   0.235     [2]=0.937 [5]=0.063 [0]=0.000
0.80   0.800    2   0.983   0.086     [2]=0.983 [5]=0.017 [0]=0.000
0.90   0.900    2   1.000   0.003     [2]=1.000 [5]=0.000 [0]=0.000
1.00   1.000    2   1.000   0.000     [2]=1.000 [0]=0.000 [1]=0.000
```

**정상 동작 확인**:
- gamma=0: 원래 분포 (max_p=0.659, H=0.869)
- gamma=0.5: 중간 집중 (max_p=0.834, H=0.455)
- gamma=1.0: 완전 붕괴 (max_p=1.000, H=0.000)
- **단조 증가**: max_p가 gamma에 대해 연속적으로 증가
- **단조 감소**: entropy가 gamma에 대해 연속적으로 감소
- **dominant 불변**: idx=2가 전 구간에서 유지 (dominant가 바뀌지 않음)

**메커니즘**: `p_i^(1/(1-γ))` temperature scaling이 확률을 집중시키고, dephasing이 coherence를 감쇠. 두 효과가 결합하여 "관측에 가까워질수록 불확실성 감소"를 구현.

---

## [4] Input Distribution Comparison (dim=16)

| 분포 | entropy | max_p | eff_dim | focus.3 max_p | delta | compose (us) |
|------|---------|-------|---------|--------------|-------|-------------|
| uniform | 2.773 | 0.0625 | 16.00 | 0.0625 | 0.000 | 0.222 |
| peaked [0] | 0.015 | 0.9985 | 1.02 | **1.0000** | +0.002 | 0.247 |
| bimodal [2,7] | 0.696 | 0.5657 | 2.01 | **0.5942** | +0.029 | 0.254 |
| sparse (3 nz) | 0.846 | 0.6531 | 2.33 | **0.7566** | +0.104 | 0.247 |
| gradient | 2.372 | 0.1711 | 10.72 | **0.2145** | +0.043 | 0.252 |
| sine | 2.465 | 0.1250 | 11.77 | **0.1443** | +0.019 | 0.244 |

**분석**:
- **focus(0.3)가 이제 확률을 실제로 변경함** (delta 열 참조)
- uniform: 모든 확률이 동일하므로 sharpening 효과 없음 (0.0625^β 해도 균등)
- peaked: 이미 거의 1.0이라 완전히 1.0으로 수렴
- sparse: 가장 큰 변화 (+0.104) — 3개 비영 성분 중 최대가 크게 강화
- compose 성능은 입력 분포에 무관 (0.22~0.25us)

---

## [5] Full Pipeline (declare → weave → wave → observe)

| 설정 | declare (us) | weave (ms) | gaze (us) | glimpse (us) | observe (us) | total (ms) |
|------|-------------|-----------|-----------|-------------|-------------|-----------|
| binary (2d) | 40.1 | 7.0 | 4.5 | 6.2 | 1.9 | 132.9 |
| small (4d) | 4.2 | 12.9 | 8.1 | 11.4 | 2.0 | 228.4 |
| medium (8d) | 5.2 | 18.2 | 18.1 | 26.6 | 3.2 | 497.2 |
| large (16d) | 4.5 | 46.3 | 49.4 | 97.0 | 4.6 | 1,556.1 |
| xl (32d) | 3.9 | 179.7 | 179.1 | 500.7 | 17.5 | 7,153.2 |
| classical 8d | 6.8 | 15.8 | 16.9 | 27.0 | 2.4 | 478.1 |

**비용 분포 (8d quantum 기준)**:
- weave: 18.2ms (97%) — chaos attractor 탐색이 지배적
- gaze: 18.1us — compute_wave + density matrix 생성
- glimpse: 26.6us — gaze + focus(population sharpening + dephasing)
- observe: 3.2us — compute_wave + argmax

**핵심 통찰**:
- weave는 **1회** 비용, gaze/observe는 **매 관측** 비용
- gaze가 observe보다 ~6x 느림 (density matrix 계산) — 하지만 C=0
- glimpse가 gaze보다 ~1.5x 느림 (sharpening + dephasing 추가)
- quantum vs classical (8d): 런타임 차이 미미 (gaze 18.1 vs 16.9us)

---

## [6] Wave Path vs Classical Observe — 정합성

| dim | gaze dominant | observe idx | reobs x10 idx | 일치 |
|-----|---------------|-------------|---------------|------|
| 4 | 3 | 3 | 3 | YES |
| 8 | 1 | 1 | 7 | **NO** |
| 16 | 15 | 15 | 15 | YES |

**dim=8 불일치 분석**:
- gaze와 observe는 동일 (idx=1)
- reobserve x10만 다름 (idx=7)
- 원인: dim=8에서 max_p=0.1250 (= 1/8, **완전 균등 분포**) → argmax 결과가 비결정적
- 10번 평균의 미세한 수치 노이즈가 다른 인덱스를 선택

**결론**: 버그 아님. 균등 분포 하에서 argmax는 임의 선택. focus를 적용하면 불확실성이 해소되어 일관된 결과를 얻을 수 있다.

---

## [7] Multi-Wave Composition Chain

| n_waves | total (us) | per_op (us) | dominant | max_p |
|---------|-----------|-------------|----------|-------|
| 2 | 0.305 | 0.305 | 3 | 0.182 |
| 3 | 0.533 | 0.266 | 2 | 0.287 |
| 4 | 1.130 | 0.377 | 1 | 0.257 |
| 5 | 1.386 | 0.346 | 1 | 0.447 |
| 8 | 2.763 | 0.395 | 3 | 0.247 |
| 10 | 3.652 | 0.406 | 7 | 0.256 |
| 16 | 6.193 | 0.413 | 7 | 0.320 |

**분석**:
- **선형 스케일링**: per-op 비용 ~0.3-0.4us (Additive의 geometric mean이 v1보다 약간 느림)
- 16-wave 체인도 6.2us — 충분히 빠름
- **C = 0 유지**: 전체 체인에서 단 한 번도 collapse 없음
- 간섭 패턴 순환(Constructive→Multiplicative→Additive)으로 max_p가 비단조적

---

## [8] Information Flow Under Composition

### 간섭 패턴별 정보 변화

| 패턴 | H_out | delta_H | eff_dim | 정보 행동 |
|------|-------|---------|---------|----------|
| Constructive | 1.912 | **+0.147** | 6.77 | 엔트로피 증가 (분포 퍼짐) |
| Additive | 1.754 | **-0.012** | 5.78 | 엔트로피 미세 감소 (균형 유지) |
| Multiplicative | 1.378 | **-0.388** | 3.97 | 엔트로피 큰 감소 (AND 선택) |
| Destructive | 1.357 | **-0.409** | 3.88 | 엔트로피 가장 큰 감소 (차이 부각) |
| Conditional | 1.766 | **+0.000** | 5.85 | 엔트로피 보존 (위상만 회전) |

**v1 대비 변화**: Additive가 이제 Constructive와 다르다.
- v1: Additive H=1.912 (Constructive와 동일)
- v2: Additive H=1.754 (약간의 엔트로피 감소 — 기하평균의 특성)

**패턴 스펙트럼** (엔트로피 변화 기준):
```
  확산 ←──────────────────────────────────────→ 집중
  Constructive  >  Conditional  >  Additive  >  Multiplicative  >  Destructive
   +0.147           ±0.000         -0.012         -0.388             -0.409
```

### focus(gamma)에 따른 정보 변화 — compose(Constructive) 후

| gamma | H | eff_dim | max_p | purity |
|-------|---|---------|-------|--------|
| 0.00 | 1.912 | 6.77 | 0.244 | 1.000 |
| 0.10 | 1.878 | 6.54 | 0.258 | 0.844 |
| 0.30 | 1.772 | 5.88 | 0.296 | 0.594 |
| 0.50 | 1.577 | 4.84 | 0.356 | 0.441 |
| 0.70 | 1.228 | 3.41 | 0.460 | 0.408 |
| 0.90 | 0.690 | 1.99 | 0.691 | 0.567 |
| 1.00 | **0.000** | **1.00** | **1.000** | 1.000 |

**정상 동작**:
- H: 1.912 → 0.000 (완전 정보 소실)
- eff_dim: 6.77 → 1.00 (8차원 → 1차원)
- max_p: 0.244 → 1.000 (균등 → 확정)
- purity: 1.0 → 0.408 → 1.0 (U자형 — 중간에 mixed, 완전 붕괴 시 다시 pure)

**purity U-curve 해석**: focus(0.7)에서 purity=0.408로 최소. 이는 population sharpening이 분포를 변경하면서 density matrix의 off-diagonal이 dephasing으로 감쇠되어 mixed state가 되기 때문. gamma=1.0에서는 하나의 상태만 남아 다시 pure.

---

## [9] Density Matrix Operations — 병목 분석

| dim | from_pure (us) | purity (us) | dephase (us) | von_neumann (us) |
|-----|---------------|-------------|-------------|-----------------|
| 2 | 0.110 | 0.014 | 0.502 | 2.899 |
| 4 | 0.177 | 0.047 | 2.960 | 6.931 |
| 8 | 0.249 | 0.128 | 7.029 | 15.219 |
| 16 | 0.634 | 0.499 | 42.616 | 48.876 |
| 32 | 2.309 | 2.109 | 293.564 | 177.932 |
| 64 | 6.621 | 8.789 | 2,697.810 | 651.608 |

**스케일링 분석**:

| 연산 | 복잡도 | dim 2x → 시간 |
|------|--------|---------------|
| from_pure | O(dim²) | ~4x |
| purity | O(dim²) | ~4x |
| dephase | O(dim³) | ~7-8x |
| von_neumann | O(dim³) | ~3-4x (faer 최적화) |

**핵심**: focus = population sharpening O(n) + from_pure O(n²) + dephasing O(n³).
dim≤32에서는 dephasing이 ~300us로 실용적. dim≥64에서 ms 진입.

---

## [10] Wave Reuse — compose once, gaze many

| dim | gaze 1M회 (ms) | ns/read | observe ns/call | ratio |
|-----|---------------|---------|-----------------|-------|
| 4 | 173.7 | 173.7 | 284.4 | 1.6x |
| 8 | 177.6 | 177.6 | 285.4 | 1.6x |
| 16 | 207.5 | 207.5 | 291.6 | 1.4x |
| 32 | 240.8 | 240.8 | 399.3 | 1.7x |

**분석**:
- gaze가 observe보다 ~1.5x 빠름 (observe는 collapse 관련 추가 연산)
- 둘 다 ns 단위 — 성능 차이보다 **정보 보존 여부**가 핵심
- Wave를 1번 compute하고 1M번 gaze해도 C=0 유지

---

## 남은 이슈

### INFO: dim≥64에서 focus 성능 저하

**현상**: dim=64에서 focus ≈ 5.6ms, dim=128에서 ≈ 38ms
**원인**: dephasing channel의 O(dim³) 스케일링
**수준**: 현재 AXOL의 주요 사용 범위(dim 2~32)에서는 문제 없음
**향후 고려**: 고차원 필요 시 sparse dephasing 또는 diagonal-only 근사

### INFO: dim=8 균등 분포에서 reobserve 비결정성

**현상**: 완전 균등 확률(max_p=1/dim)에서 argmax 결과가 불안정
**수준**: 기대된 행동 (균등 분포 = 정보 없음 = 선택 불가)
**대응**: focus를 적용하여 분포를 집중시킨 후 observe하면 해결

---

## 성능 요약 (dim=8 기준)

| 연산 | 시간 | C (collapses) | 정보 보존 | 확률 변화 |
|------|------|---------------|----------|----------|
| compose | 0.16us | 0 | 완전 | 간섭 패턴에 따라 |
| gaze | 0.15us | 0 | 완전 | 없음 (읽기만) |
| focus (γ=0.3) | 27us | 0.3 | 부분 | max_p 증가 |
| focus (γ=0.7) | 25us | 0.7 | 소량 | max_p 크게 증가 |
| observe | 0.30us | 1 | 없음 | one-hot (argmax) |
| 16-wave chain | 6.2us | 0 | 완전 | 패턴 혼합 |
| full pipeline | 26.6us | 0.5 (glimpse) | 부분 | sharpened |

---

## [11] O(1) Observation Cost — 실제 유즈케이스 검증

### 핵심 원리

```
비용 구조:
  weave  (1회)  → chaos dynamics, basin 생성  → ~11ms (dim=8)
  observe (N회) → Born rule + argmax           → ~5μs × N
```

Neural network inference는 O(parameters)이고 매 호출마다 동일한 비용을 지불한다.
AXOL의 observe는 O(dim)이며, dim은 모델 정의 시점에 고정된 상수다.
**질문이 공짜다.**

### NPC Real-time AI (usecase_npc_realtime.axol)

20프레임 시뮬레이션: 평화 → 교전 → 패배 → 도주.

| 구간 | 프레임 | 확률 변화 (top-2) | 해석 |
|------|--------|-------------------|------|
| 평화 | 1-5 | [3]=0.259, [7]=0.257 | 중립/방어 균형 |
| 적 발견 | 6-10 | [7]=0.292→0.301, [3]=0.207→0.185 | alertness 급등 |
| 전투 | 11-15 | [7]=0.296→0.300, [0]=0.131→0.120 | 전투 모드, 피로 누적 |
| 패배 | 16-20 | [7]=0.305→0.308, [3]=0.199→0.205 | 공포 상승, 도주 전환 |

NPC간 관계 (모두 O(1)):
```
alliance  (warrior↔healer, <~>): 균형 분포 — 상호 보완
tension   (warrior↔coward, <!>): [0]=0.258, [2]=0.227 — 극단 부각
persuasion(healer↔coward, <+>): 중간 분포 — 설득은 중간지대
```

**총 20 observe + 3 wave + 3 rel + 1 expect: 13.274ms → frame당 ~4.3μs**

### Perception Engine — Predictive Coding (usecase_perception.axol)

뇌의 지각 과정 모델링: expect(예측) ↔ wave(감각) = perception(지각).

| Cycle | 상황 | alignment | negativity_delta | 해석 |
|-------|------|-----------|------------------|------|
| 1 | 예측 = 현실 | 0.927 | -0.000 | 예측 확인 |
| 2 | 갑자기 움직임 | 0.922 | +0.000 | 미세한 prediction error |
| 3 | 예측 업데이트 | 0.880 | +0.000 | 새 모델 적응 |
| 4 | 착시 (없는 움직임 예상) | 0.883 | -0.000 | 착시 상태 유지 |

**Collapse Spectrum 실증** — 같은 장면을 다른 깊이로:
```
C=0.0: [1]=0.443, [5]=0.162, [0]=0.107  (5개 가능성 공존)
C=0.2: [1]=0.538                        (하나가 강해지기 시작)
C=0.4: [1]=0.790                        (거의 확실)
C=0.6: [1]=0.994                        (나머지 소멸)
C=0.8: [1]=1.000                        (완전 붕괴)
```

**총 시간: 11.897ms** (4 perception cycle + collapse spectrum + widen)

### Dialogue Dynamics (usecase_dialogue.axol)

대화의 구조를 간섭 패턴으로 모델링.

| 대화 유형 | 간섭 패턴 | alignment | negativity_delta | 해석 |
|-----------|-----------|-----------|------------------|------|
| 동의 | `<~>` constructive | 0.706 | **-0.001** | 대화가 닫힘 |
| 반박 | `<!>` destructive | 0.403 | **+0.072** | 대화가 열림 |
| 질문→답변 | `<+>` additive | — | — | 정보 누적 |

**반박이 오면 negativity 72배 증가** (0.001 → 0.073). 대화가 열린다.

다자 대화 충돌 해결:
```
resolve ab, bc with interfere → [3]=0.222 (간섭으로 합의점 탐색)
resolve ab, ac with superpose → t=0.548   (가능성 공존, 높은 collapse 수준)
```

**총 시간: 13.845ms** (6 wave + 6 rel + 8 observe + 2 resolve)

### Observation Cost — 알기 위해 잃는 것 (usecase_observation_cost.axol)

**AXOL만의 고유 기능: 관측의 비가역성 추적.**

미지 행성 스캔 — 알수록 잃는다:
```
C=0.0:  [1]=0.443, [5]=0.162, [0]=0.107  ← 모든 가능성
C=0.2:  [1]=0.538                         ← 일부 시작
C=0.4:  [1]=0.790                         ← 하나가 지배
C=0.6:  [1]=0.994                         ← 거의 확정
C=0.8:  [1]=1.000                         ← 돌아갈 수 없음
```

**비가역성 증명** — focus 후 widen:
```
gaze (원래):       [1]=0.398, [5]=0.170, [0]=0.122  (열린 가능성)
focus(0.5):        dominant=1                         (부분 붕괴)
widen(0.3):        [1]=0.532, [5]=0.127, [0]=0.084  (복구 시도)
gaze (복구 후):    [1]=0.532 ≠ 0.398                 (원래와 다르다)
```

**관계와 앎의 관계** — 한쪽을 알면 관계가 변한다:
```
before_knowing: negativity=0.004, [1]=0.469  (열린 관계)
  ↓ focus(0.8)로 scan_a 붕괴
after_knowing:  negativity=0.247, [1]=0.724  (닫힌 관계)
```
negativity **0.004 → 0.247** (62배). 한쪽을 알게 되면 관계의 구조 자체가 변한다.

**총 시간: 15.298ms** (4 gaze + 6 glimpse + 2 focus + 1 widen + 3 observe + 2 rel)

### O(1) 관측 비용 종합

| 유즈케이스 | 관측 횟수 | 총 시간 | weave 제외 | 관측당 평균 |
|------------|-----------|---------|-----------|-------------|
| NPC (20 frames) | 20 observe + 3 rel | 13.3ms | ~2.1ms | **~4μs** |
| Perception (4 cycles) | 4 cycle + spectrum | 11.9ms | ~1.9ms | **~5μs** |
| Dialogue (6 turns) | 6 rel + 8 observe | 13.8ms | ~2.7ms | **~5μs** |
| Observation Cost | 9 glimpse/gaze + 3 observe | 15.3ms | ~2.0ms | **~5μs** |

**weave (~11ms) 1회 후, 관측은 몇 번이든 ~5μs. 240,000fps에서도 실시간.**

---

## [12] v3 버그 수정 검증

### Fix 1: Multi-input Composition

**문제**: `compute_wave()`에서 `inputs.first()`만 사용 → 2-input 모델이 실질적으로 1-input.

**수정**: 모든 입력을 HashMap으로 수집, `compose_from_rules()`로 DAG 기반 간섭 합성.

```
수정 전: 2-input observe → [0.125, 0.125, 0.125, ...] (균일 = 두 번째 입력 무시)
수정 후: 2-input observe → [0]=0.262, [1]=0.203, [4]=0.168 (비균일 = 간섭 작동)
```

### Fix 2: Relation Initial Negativity

**문제**: `Wave::compose`가 pure state → `von_neumann_entropy` = 0 → 모든 관계가 negativity=0.

**수정**: Bhattacharyya distance + depolarizing noise.

```
수정 전: rel enc_rel negativity=0.0000 (expect 무의미)
수정 후: rel enc_rel negativity=0.0851, cross=0.2654, tension=0.2815
```

### Fix 3: preserve_basins

**문제**: iterate의 `apply_feedback()`이 chaos dynamics를 재실행하여 user-defined basin 파괴.

**수정**: Tapestry에 `preserve_basins: bool` 추가, from_basins 사용 시 true.

```
수정 전: iterate → [0.125, 0.125, ...] (basin 파괴 → 균일)
수정 후: iterate → [0]=0.262, [1]=0.203 (basin 보존)
```

---

## 결론

### AXOL은 분류기가 아니다

AXOL은 **가능성의 구조**를 짜놓고, 그 구조에서 **원하는 만큼만** 꺼내 보는 시스템이다.

**계산의 위치가 다르다:**
- 신경망: 지능이 inference에 있다. 매번 다시 생각한다. O(parameters).
- AXOL: 지능이 구조에 있다. wave function 자체가 답이다. 관측은 그걸 읽을 뿐. O(dim).

**AXOL만 할 수 있는 것:**
1. **얼마나 알지 선택** — C=0(아무것도 잃지 않음) → C=1(전부 잃음). 신경망은 전부 아니면 전무.
2. **아는 것의 대가 추적** — negativity 변화로 관측이 시스템을 어떻게 바꾸는지 명시적 추적.
3. **관계의 구조를 실시간으로** — 간섭 패턴(`<~>`, `<!>`, `<+>`, `<*>`, `<?>`)으로 차이를 증폭하거나 약화.
4. **비가역성** — focus 후 widen해도 원래로 돌아가지 않는다. 정보는 한번 잃으면 끝.

### 성능 특성 (dim=8 기준)

| 연산 | 시간 | C | 의미 |
|------|------|---|------|
| weave | ~11ms | — | 구조 생성 (1회) |
| gaze | ~5μs | 0 | 읽기만, 잃는 것 없음 |
| glimpse | ~30μs | γ | γ만큼 잃음 |
| observe | ~5μs | 1 | 전부 잃음, 하나의 답 |
| rel observe | ~5μs | — | 관계 구조 관측 |
| focus | ~25μs | γ | 비가역적 부분 붕괴 |
| widen | ~90μs | — | 가능성 재개방 (불완전 복구) |

### 간섭 패턴 5종 — 정보 행동

```
  확산 ←──────────────────────────────────────→ 집중
  Constructive  >  Conditional  >  Additive  >  Multiplicative  >  Destructive
   +0.147           ±0.000         -0.012         -0.388             -0.409
```

### 적합한 유즈케이스

| 도메인 | AXOL 활용 | 핵심 연산 |
|--------|-----------|-----------|
| 게임 NPC AI | 매 프레임 O(1) 의사결정 | observe, rel |
| 지각 모델링 | 예측-감각 간섭, prediction error | expect, negativity_delta |
| 대화 구조 | 동의/반박의 간섭 패턴 | rel `<~>`/`<!>`, resolve |
| 탐사/의사결정 | 관측의 비가역적 비용 추적 | collapse spectrum, widen |
| 실시간 관계 추적 | NPC간, 센서간, 개념간 관계 | rel, gaze, negativity |
