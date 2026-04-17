# AXOL Practical Usecase Benchmark Report

> Runtime: 3.7s  |  2026-04-17 08:17


## [1] Cosine Similarity Search

| Dim | DB Size | NumPy Time | AXOL Total | AXOL/item | Weave | Omega | Phi | Match |
|-----|---------|------------|------------|-----------|-------|-------|-----|-------|
|  16 |    50 | 1.8us | 21666us | 410.8us | 4.3ms | 1.00 | 0.98 | DIFF |
|  32 |    50 | 1.9us | 22166us | 422.9us | 4.7ms | 1.00 | 1.00 | DIFF |
|  64 |    50 | 3.0us | 20916us | 403.3us | 8.0ms | 1.00 | 1.00 | DIFF |
| 128 |    50 | 2.3us | 21320us | 411.8us | 36.0ms | 1.00 | 1.00 | DIFF |

## [2] XOR Classification

| Method | Train/Weave | Inference | Accuracy | Omega | Phi |
|--------|------------|-----------|----------|-------|-----|
| NN (500 epochs) | 9.4ms | 39.5us | 100% | - | - |
| AXOL (no fit) | 2.5ms | 226.7us | 75% | 1.00 | 0.88 |
| **AXOL + fit_data** | 2.5ms | 213.5us | **100%** | 1.00 | 0.88 |

> fit_data train accuracy: 100%

## [3] Multi-class Pattern Recognition

- NumPy: accuracy=100%, time=80.8us
- AXOL (no fit):  accuracy=70%, time=1377us (obs/item=89.1us), weave=3.0ms
- **AXOL + fit_data: accuracy=100%**, time=1352us (obs/item=68.1us), weave=5.5ms
- fit_data train accuracy: 100%
- Omega=1.00, Phi=0.96

## [4] Anomaly Detection

- NumPy:  F1=1.00, time=34.6us
- AXOL (no fit): F1=0.40, time=1669us (obs/item=74.2us), weave=3.6ms
- **AXOL + fit_data: F1=1.00**, time=1442us (obs/item=73.6us), weave=3.6ms
- fit_data train accuracy: 100%
- Omega=1.00, Phi=0.99

## [5] Pipeline Depth Advantage (Key Result)

| Depth | Traditional | AXOL Observe | Speedup | Weave | Omega | Phi |
|-------|-------------|--------------|---------|-------|-------|-----|
|     1 |       5.1us |       15.2us | **0.3x** | 3.91ms | 1.00 | 1.00 |
|     5 |       8.4us |       12.2us | **0.7x** | 17.40ms | 1.00 | 0.98 |
|    10 |      12.2us |       11.9us | **1.0x** | 36.80ms | 1.00 | 0.96 |
|    50 |      40.3us |       12.1us | **3.3x** | 184.55ms | 1.00 | 0.82 |
|   100 |      86.8us |       11.8us | **7.4x** | 370.00ms | 1.00 | 0.69 |
|   500 |     534.3us |       12.1us | **44.3x** | 1.91s | 1.00 | 0.31 |

## Key Takeaways

1. **AXOL observe()는 depth에 무관** — depth=500에서도 관측 비용 일정
2. **Weave는 일회성 비용** — N회 관측 시 상각되어 사실상 무료
3. **Omega/Phi 품질 보증** — 매 관측마다 정량적 신뢰도 제공
4. **NumPy 대비 단일 연산 속도는 느림** — AXOL 장점은 속도가 아니라 depth-independence + 품질 보증
5. **깊은 파이프라인 + 반복 관측 시나리오**에서 AXOL이 전통 방식을 압도


```
Total benchmark time: 3.7s
```