# AXOL AI Home-Ground Benchmark — Digit Classification

**Task**: sklearn `digits` (MNIST-like) — 8×8 이미지 분류
**Dataset**: 1,257 train / 540 test, 10 classes, 64 features
**Motivation**: 정렬은 AXOL의 원정 경기. 본 벤치는 **AXOL 홈그라운드**(고정 입력/출력 dim, 분류)에서의 경쟁력 측정.

---

## Results

| Method | Accuracy | Train | **Infer/item** | Omega | Phi |
|---|---:|---:|---:|---:|---:|
| LogReg | **0.9815** | 40 ms | 75.7 μs | — | — |
| MLP (64→32→10) | 0.9759 | 766 ms | 205.9 μs | — | — |
| Random Forest (100) | 0.9685 | 340 ms | 7,293 μs | — | — |
| kNN (k=5) | 0.9704 | **1 ms** | 13,916 μs | — | — |
| **AXOL (dim=64)** | 0.8963 | **28 ms** | **17.6 μs** ★ | 1.000 | 0.999 |
| AXOL (dim=128) | 0.8963 | 65 ms | 20.8 μs | 1.000 | 1.000 |
| AXOL (dim=256) | 0.8963 | 144 ms | 23.6 μs | 1.000 | 1.000 |
| AXOL (dim=512) | 0.8963 | 589 ms | 82.5 μs | 1.000 | 1.000 |

★ = 전 방식 중 **추론 최속**

---

## Key Findings

### 1. 홈그라운드에서 AXOL은 **추론 속도 1위**

- **17.6 μs/item** — 2위 LogReg(75.7 μs) 대비 **4.3× 빠름**, MLP(206 μs) 대비 **11.7× 빠름**
- 정렬 벤치에서 numpy 대비 5~6 자릿수 느렸던 AXOL이 **분류 홈그라운드에서는 압도적 속도 우위**
- 원인: 분류는 `dim = max(n_features, n_classes) = 64`로 task-fixed. 정렬은 `dim = next_pow2(n)`으로 입력 크기에 묶임.
- **Axiom 3 ("시간축 포기 = 속도 확보")의 첫 명확한 실증**

### 2. 정확도는 89.6%에서 **하드 실링**

- dim을 64 → 512까지 8배 늘려도 **정확도 동일 (0.8963)**
- 원인: `fit_readout`은 **linear lstsq + MSE objective**
  - `W = lstsq(H, one_hot(Y))`
  - 분류용 cross-entropy가 아니라 회귀용 MSE
  - 비선형 표현력 없음 (MLP의 hidden layer 같은 기제 부재)
- LogReg(98.15%)도 linear 모델이지만 cross-entropy + L2 regularization → 8.5pp 차이

### 3. Phi = 1.000 vs 실제 정확도 89.6% — 또다시 괴리

- 모든 AXOL 구성에서 Phi ≈ 1.000
- 실제 정확도는 89.6% → **Phi는 신뢰도 지표이지 정답률이 아님**
- Axiom 2 ("Phi가 실제 정답률과 상관") 다시 한 번 반증

### 4. dim 증가 = 속도만 느려짐

| dim | infer/item |
|---:|---:|
| 64 | 17.6 μs |
| 128 | 20.8 μs |
| 256 | 23.6 μs |
| 512 | 82.5 μs |

- 정확도 개선 없음, 속도만 손해
- 최적: **dim = task가 요구하는 최소값** (여기서는 max(features, classes))

---

## Speed Trade-off Plot

```
Accuracy
  1.00 │   LogReg ●
       │   MLP ●
  0.97 │         RF ●      kNN ●
       │
  0.90 │                                AXOL(d=64) ●
       │                                AXOL(d=128) ●
       │                                AXOL(d=256) ●
  0.85 │
       └─────────────────────────────────────────────→ Infer latency (log)
           10μs      100μs      1ms       10ms      100ms
```

- **Pareto 최적**: AXOL은 "낮은 정확도 + 매우 빠른 추론", LogReg는 "높은 정확도 + 빠른 추론"
- 용도별:
  - **고정확도 필수**: LogReg 선택 (98.15%, 75μs)
  - **초고속 필수**: AXOL (89.63%, 18μs) — 4× 속도 이득이 8.5pp 정확도 손실을 상쇄하면 의미 있음
  - **지연시간 민감 엣지/스트리밍**: AXOL 적합

---

## AI 응용 관점에서의 의미

### ✅ AXOL이 실제로 의미있는 영역

1. **실시간 스트리밍 분류** (>10k QPS 요구): 18 μs/item = 55K QPS 가능
2. **엣지 추론** (단일 코어 CPU, Python 가능): 경량 Python 구현으로 MLP 수준 속도
3. **후보 필터링 단계**: 정확한 모델 앞단의 빠른 프리필터로 활용
4. **Confidence-aware 분류**: Omega/Phi로 uncertainty 자동 제공

### ⚠️ 경쟁력 없는 영역

1. **정확도 > 95% 요구**: linear MSE 한계로 달성 불가
2. **비선형 결정경계**: hidden layer가 없어 XOR-like 문제 해결 못 함
3. **Few-shot 학습**: 샘플 부족 시 lstsq 불안정
4. **출력 공간이 큰 분류** (k > 1000): output dim 스케일 문제

---

## AXOL Axiom 재평가 (홈그라운드 기준)

| Axiom | 이번 벤치 결과 | 판정 |
|---|---|---|
| 1 (시간축 부정) | 추론 17.6 μs — epoch/iteration 없음 | ✅ 실증 |
| 2 (99%+ 정확도) | 89.63% (하드 실링) | ❌ 미달 |
| 3 (확률이 대가) | 정확도 -8.5pp 희생 → 속도 4× 획득 | ✅ 실증 |
| 4 (상용 수준) | 속도는 프로덕션급, 정확도는 도메인 한정 | ⚠️ 조건부 |

---

## 개선 가능성

1. **Cross-entropy readout 지원** → 예상 정확도 95%+ (LogReg와 동격)
2. **비선형 reservoir 활성화 함수** (tanh, ReLU) → 예상 96~98%
3. **Regularization (Ridge 도입)** → 안정성 향상, 소규모 데이터에서 +1~2pp
4. **Output dim ≠ Input dim**: 현재 동일해야 하는 제약 완화 → 속도 더 확보

위 1-2만 구현해도 AXOL이 **"LogReg 정확도 + 4× 빠른 추론"**이라는 실질적 경쟁력을 획득.

---

## 결론

**AXOL은 홈그라운드(고정 dim 분류)에서 명확한 존재 이유를 가진다**:
- 추론 속도 1위 (17.6 μs, 경쟁 방법 대비 4~800× 빠름)
- 정확도 89.6%는 현 구현의 하드 실링 (linear MSE)
- 속도-정확도 Pareto 상에서 "**ultra-fast, decent-accuracy**" 포지션 차지
- 정렬(원정) 벤치와 달리 Axiom 3이 실제로 작동 — "시간축 포기 → 속도 확보"가 수치로 입증

현재 구현은 "speed-focused linear classifier" 로 포지셔닝하면 실제 AI 파이프라인에서 **프리필터, 엣지 추론, 대용량 배치 분류** 에 유의미하게 쓰일 수 있다. 정확도 95%+ 달성하려면 cross-entropy readout 및 비선형 reservoir 추가가 필수.

---

## Reproduce

```bash
pip install scikit-learn numpy
python tests/bench_ai_classification.py
```

Output: `benchmark_results/ai_classification.json`
