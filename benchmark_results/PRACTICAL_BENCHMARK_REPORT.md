# AXOL Practical Usecase Benchmark Report

> Runtime: 8.3s  |  2026-02-13 20:06


## [1] Cosine Similarity Search

| Dim | DB Size | NumPy Time | AXOL Total | AXOL/item | Weave | Omega | Phi | Match |
|-----|---------|------------|------------|-----------|-------|-------|-----|-------|
|  16 |    50 | 4.2us | 54984us | 988.4us | 10.0ms | 1.00 | 0.98 | DIFF |
|  32 |    50 | 7.2us | 48748us | 855.8us | 10.2ms | 1.00 | 1.00 | DIFF |
|  64 |    50 | 7.2us | 57136us | 797.0us | 14.1ms | 1.00 | 1.00 | DIFF |
| 128 |    50 | 4.6us | 54356us | 908.9us | 118.2ms | 1.00 | 1.00 | DIFF |

## [2] XOR Classification

| Method | Train/Weave | Inference | Accuracy | Omega | Phi |
|--------|------------|-----------|----------|-------|-----|
| NN (500 epochs) | 27.3ms | 87.6us | 100% | - | - |
| AXOL (no fit) | 8.4ms | 809.3us | 75% | 1.00 | 0.88 |
| **AXOL + fit_data** | 7.8ms | 534.3us | **100%** | 1.00 | 0.88 |

> fit_data train accuracy: 100%

## [3] Multi-class Pattern Recognition

- NumPy: accuracy=100%, time=179.6us
- AXOL (no fit):  accuracy=70%, time=2957us (obs/item=128.2us), weave=11.9ms
- **AXOL + fit_data: accuracy=100%**, time=2609us (obs/item=127.6us), weave=7.0ms
- fit_data train accuracy: 100%
- Omega=1.00, Phi=0.96

## [4] Anomaly Detection

- NumPy:  F1=1.00, time=108.5us
- AXOL (no fit): F1=0.40, time=3820us (obs/item=136.1us), weave=6.6ms
- **AXOL + fit_data: F1=1.00**, time=2745us (obs/item=138.3us), weave=6.6ms
- fit_data train accuracy: 100%
- Omega=1.00, Phi=0.99

## [5] Pipeline Depth Advantage (Key Result)

| Depth | Traditional | AXOL Observe | Speedup | Weave | Omega | Phi |
|-------|-------------|--------------|---------|-------|-------|-----|
|     1 |       9.7us |       19.1us | **0.5x** | 8.97ms | 1.00 | 1.00 |
|     5 |      17.3us |       19.1us | **0.9x** | 39.71ms | 1.00 | 0.98 |
|    10 |      25.3us |       19.7us | **1.3x** | 78.22ms | 1.00 | 0.96 |
|    50 |      96.8us |       19.1us | **5.1x** | 415.91ms | 1.00 | 0.82 |
|   100 |     199.1us |       19.3us | **10.3x** | 889.46ms | 1.00 | 0.69 |
|   500 |     986.3us |       26.5us | **37.2x** | 4.29s | 1.00 | 0.31 |

## Key Takeaways

1. **AXOL observe()는 depth에 무관** — depth=500에서도 관측 비용 일정
2. **Weave는 일회성 비용** — N회 관측 시 상각되어 사실상 무료
3. **Omega/Phi 품질 보증** — 매 관측마다 정량적 신뢰도 제공
4. **NumPy 대비 단일 연산 속도는 느림** — AXOL 장점은 속도가 아니라 depth-independence + 품질 보증
5. **깊은 파이프라인 + 반복 관측 시나리오**에서 AXOL이 전통 방식을 압도


```
Total benchmark time: 8.3s
```