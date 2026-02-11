# AXOL Extreme Performance Report

> Generated: 2026-02-11
> Platform: Windows 11 Pro, Python 3.13.5, NumPy 2.3.5

---

## 1. Core Operation Scaling

### 1.1 Transform (Matrix-Vector Multiply)

| Dim | Avg | Min | Max | GFLOP/s |
|----:|----:|----:|----:|--------:|
| 64 | 5.5 us | 4.4 us | 16.5 us | 1.49 |
| 256 | 8.5 us | 8.1 us | 10.4 us | 15.38 |
| 1,024 | 218 us | 159 us | 294 us | 9.62 |
| 4,096 | 2.80 ms | 2.47 ms | 3.18 ms | 11.97 |
| **10,000** | **14.2 ms** | **13.2 ms** | **15.2 ms** | **14.09** |
| 50,000 | 259 ms | 252 ms | 267 ms | 19.27 |

- dim=10,000: 2억 FLOP을 14ms 안에 처리, **14 GFLOP/s**
- dim=50,000: 50억 FLOP을 259ms 안에 처리, **19.3 GFLOP/s** (최대 처리량)

### 1.2 Merge (Weighted Sum)

| N vectors | Avg | Per-vector |
|----------:|----:|-----------:|
| 2 | 22.8 us | 11.4 us |
| 8 | 56.2 us | 7.0 us |
| 32 | 179 us | 5.6 us |
| 128 | 722 us | 5.6 us |
| 512 | 3.49 ms | 6.8 us |

- 512개 벡터(dim=1024) merge: 3.49ms, **선형 확장 확인 (O(N))**

### 1.3 Distance (Metric Comparison)

| Dim | Euclidean | Cosine | Dot |
|----:|----------:|-------:|----:|
| 256 | 11.4 us | 19.2 us | 5.5 us |
| 1,024 | 11.9 us | 22.8 us | 7.5 us |
| **10,000** | **15.5 us** | **17.7 us** | **7.2 us** |
| 50,000 | 48.6 us | 50.9 us | 31.5 us |
| 100,000 | 81.7 us | 93.8 us | 52.6 us |

- dim=10,000에서 dot product: **7.2us** (1.39 GDOT/s)

### 1.4 Element-wise Functions

| Dim | relu | sigmoid | square | sqrt |
|----:|-----:|--------:|-------:|-----:|
| 1,024 | 11.8 us | 16.6 us | 7.5 us | 12.6 us |
| **10,000** | **28.2 us** | **48.3 us** | **13.6 us** | **34.8 us** |
| 100,000 | 142 us | 387 us | 57.6 us | 210 us |
| 1,000,000 | 4.42 ms | 9.68 ms | 3.64 ms | 6.28 ms |

- **square 최대 처리량: 1,736 M elem/s** (dim=100K)
- dim=10,000 sigmoid: 48us = **207 M elem/s**

### 1.5 Pipeline Depth

| Depth | Total | Per-step |
|------:|------:|---------:|
| 5 | 432 us | 86.5 us |
| 10 | 845 us | 84.5 us |
| 25 | 1.95 ms | 78.1 us |
| 50 | 4.44 ms | 88.8 us |
| 100 | 8.49 ms | 84.9 us |
| 200 | 18.0 ms | 90.1 us |

- 200단계 파이프라인(dim=256): 18ms, **단계당 ~85us 일정** (완전 선형)

---

## 2. Python Pure Loop vs AXOL

### 2.1 MatMul (O(N^2) 연산)

| Dim | Python | AXOL | Speedup |
|----:|-------:|-----:|--------:|
| 16 | 35.0 us | 5.9 us | 5.9x |
| 64 | 528 us | 6.7 us | **78.6x** |
| 256 | 9.54 ms | 14.4 us | **664x** |
| 1,024 | 150 ms | 269 us | **557x** |
| 4,096 | 2.554 s | 2.33 ms | **1,097x** |
| **10,000** | **14.87 s** | **14.81 ms** | **1,004x** |

```
  Python dim=10000:  14.87초 (약 15초)
  AXOL   dim=10000:  14.81밀리초
  -----------------------------------------------
  AXOL이 Python보다 1,004배 빠름
```

### 2.2 Sigmoid (Element-wise)

| Dim | Python | AXOL | Speedup |
|----:|-------:|-----:|--------:|
| 1,000 | 197 us | 17.3 us | 11.4x |
| **10,000** | **1.87 ms** | **154 us** | **12.2x** |
| 100,000 | 18.1 ms | 496 us | 36.6x |
| 1,000,000 | 214 ms | 9.81 ms | 21.8x |
| 10,000,000 | 2.52 s | 87.0 ms | **29.0x** |

### 2.3 Euclidean Distance

| Dim | Python | AXOL | Speedup |
|----:|-------:|-----:|--------:|
| 100 | 20.3 us | 6.7 us | 3.0x |
| 1,000 | 267 us | 9.1 us | 29.5x |
| **10,000** | **2.57 ms** | **27.4 us** | **93.5x** |
| 100,000 | 24.4 ms | 98.8 us | **247x** |
| 1,000,000 | 254 ms | 3.95 ms | **64.4x** |

### 2.4 Grover Search (Algorithm Complexity)

| N | Grover Iters | Python (brute) | AXOL (Grover) | Note |
|--:|-------------:|---------------:|--------------:|:-----|
| 16 | 3 | 563 ns | 793 us | Python trivial win |
| 64 | 6 | 1.7 us | 1.33 ms | Python trivial win |
| 256 | 12 | 4.6 us | 2.28 ms | Python trivial win |
| 1,024 | 25 | 16.5 us | 20.5 ms | Python trivial win |
| 4,096 | 50 | 68.8 us | 256 ms | Python trivial win |

> Python의 선형 탐색(`for i in range(N)`)은 trivial하여 ns 단위이지만,
> AXOL은 실제 양자 시뮬레이션(행렬 연산)을 수행하므로 오버헤드가 있음.
> 진짜 양자 하드웨어에서는 AXOL의 O(sqrt(N)) 구조가 지수적 우위를 가짐.

---

## 3. Quantum Pipeline (Declare -> Weave -> Observe)

### 3.1 Weave Chain Scaling

| Nodes | Avg | Per-node |
|------:|----:|---------:|
| 2 | 22.3 ms | 11.2 ms |
| 5 | 47.5 ms | 9.5 ms |
| 10 | 97.5 ms | 9.7 ms |
| 20 | 195 ms | 9.8 ms |
| 50 | 503 ms | **10.1 ms** |

- **Weave는 노드 수에 선형 비례 (O(N))**, 노드당 ~10ms

### 3.2 Weave Dimension Scaling

| Dim | Avg |
|----:|----:|
| 4 | 38.7 ms |
| 8 | 44.9 ms |
| 16 | 52.4 ms |
| 32 | 61.4 ms |
| 64 | 82.1 ms |
| 128 | 407 ms |

- dim 64->128에서 급등: 행렬 연산 O(D^2) + QR 분해 비용

### 3.3 Observation Throughput

| Dim | Single Observe | obs/sec | Reobserve x10 | obs/sec |
|----:|---------------:|--------:|---------------:|--------:|
| 4 | 1.00 ms | 995 | 9.62 ms | 1,040 |
| 8 | 1.06 ms | 944 | 9.56 ms | 1,046 |
| 16 | 1.04 ms | 961 | 10.9 ms | 919 |
| 32 | 1.07 ms | 936 | 12.4 ms | 807 |
| 64 | 1.00 ms | 996 | 11.9 ms | 843 |

- **~1,000 observations/sec** (Weave 이후 관측 단계)
- "얽힘은 한 번, 관측은 무한 번" 원칙 실증

### 3.4 Wide Topology

| N inputs | Weave | Observe |
|---------:|------:|--------:|
| 2 | 9.88 ms | 1.20 ms |
| 4 | 10.1 ms | 2.98 ms |
| 8 | 11.8 ms | 7.39 ms |
| 16 | 11.3 ms | 25.2 ms |
| 32 | 14.8 ms | 86.0 ms |

- Weave: 입력 수에 거의 무관 (~10ms)
- Observe: 입력 수에 비례 (중간 벡터 생성 비용)

### 3.5 Relation Kind Comparison

| Kind | Weave | Observe | Omega | Phi |
|:-----|------:|--------:|------:|----:|
| PROPORTIONAL | 41.5 ms | 1.72 ms | 1.000 | 0.926 |
| ADDITIVE | 48.0 ms | 2.26 ms | 1.000 | 0.939 |
| MULTIPLICATIVE | 49.6 ms | 1.60 ms | 1.000 | 0.937 |
| INVERSE | 41.2 ms | 2.07 ms | 1.000 | 0.942 |
| CONDITIONAL | 56.4 ms | 1.75 ms | 1.000 | 0.936 |

- 모든 관계 종류에서 **Omega = 1.0** (완전 수렴)
- Phi > 0.92 (높은 선명도)

---

## 4. Chaos Theory Computation

### 4.1 Lyapunov Estimation

| Dim | 50 steps | 200 steps | 500 steps |
|----:|---------:|----------:|----------:|
| 4 | 788 us | 2.21 ms | 6.27 ms |
| 16 | 747 us | 2.34 ms | 6.41 ms |
| 64 | 726 us | 2.87 ms | 7.01 ms |
| 128 | 822 us | 3.94 ms | 10.1 ms |
| 256 | 1.69 ms | 5.61 ms | 13.8 ms |

### 4.2 Fractal Dimension

| N points | PSD=4 | PSD=16 | PSD=32 |
|---------:|------:|-------:|-------:|
| 100 | 1.60 ms | 2.46 ms | 3.28 ms |
| 500 | 6.66 ms | 11.3 ms | 15.9 ms |
| 1,000 | 14.9 ms | 27.0 ms | 31.3 ms |
| 5,000 | 72.4 ms | 131 ms | 167 ms |

---

## 5. DSL Parse + Execute

### 5.1 Parse Scaling

| Steps | Source chars | Parse time |
|------:|------------:|-----------:|
| 5 | 370 | 294 us |
| 10 | 716 | 507 us |
| 25 | 1,766 | 1.17 ms |
| 50 | 3,516 | 2.20 ms |
| 100 | 7,017 | 3.93 ms |

- **완전 선형: O(L)** - 1KB당 ~0.6ms

### 5.2 Automaton Scaling

| N states | Iterations | Parse | Execute |
|---------:|-----------:|------:|--------:|
| 8 | 7 | 132 us | 1.42 ms |
| 16 | 15 | 176 us | 3.49 ms |
| 32 | 31 | 329 us | 8.07 ms |
| 64 | 63 | 326 us | 15.9 ms |
| 128 | 127 | 1.02 ms | 31.2 ms |
| 256 | 255 | 1.48 ms | 68.8 ms |

---

## 6. Memory Efficiency

### 6.1 Vector Types (dim=100,000)

| Type | Total | Per element | vs Python float (~28B) |
|:-----|------:|------------:|-----------------------:|
| FloatVec | 391 KB | **4.0 B** | **7x savings** |
| IntVec | 782 KB | 8.0 B | 3.5x savings |
| BinaryVec | 392 KB | 4.0 B | 7x savings |

### 6.2 StateBundle

| N keys | Total | Per element |
|-------:|------:|------------:|
| 10 | 42.5 KB | 4.3 B |
| 50 | 213 KB | 4.3 B |
| 100 | 426 KB | 4.3 B |
| 500 | 2.1 MB | 4.3 B |

### 6.3 Tapestry

| Nodes | Total | Per node |
|------:|------:|---------:|
| 3 | 149 KB | 49.6 KB |
| 5 | 154 KB | 30.8 KB |
| 10 | 166 KB | 16.6 KB |
| 20 | 191 KB | 9.5 KB |

- 노드가 많을수록 공유 구조 덕분에 **노드당 비용 감소**

---

## 7. Token Efficiency (cl100k_base)

### 7.1 Program Comparison

| Program | Python | C# | AXOL | vs Python | vs C# |
|:--------|-------:|---:|-----:|----------:|------:|
| Counter | 32 | 58 | 32 | 0% | **-45%** |
| StateMachine | 66 | 95 | 46 | **-30%** | **-52%** |
| Combat | 103 | 109 | 60 | **-42%** | **-45%** |
| HP Decay | 37 | 64 | 50 | +35% | **-22%** |
| NeuralLayer | 77 | 152 | 181 | +135% | +19% |
| SearchSort | 58 | 95 | 106 | +83% | +12% |

- 제어 흐름/로직 중심: AXOL이 **30-52% 절약**
- 행렬 리터럴이 큰 경우: AXOL이 더 많은 토큰 사용 (숫자 데이터 자체의 비용)

### 7.2 Automaton Scaling

| N | Python | C# | AXOL | AXOL/Python |
|--:|-------:|---:|-----:|------------:|
| 5 | 63 | 73 | 66 | 1.05x |
| 50 | 378 | 343 | 336 | 0.89x |
| 100 | 728 | 643 | 636 | **0.87x** |
| 500 | 3,528 | 3,043 | 3,036 | **0.86x** |

---

## 8. Summary

### dim=10,000 Highlight

| Metric | Python | AXOL | Ratio |
|:-------|-------:|-----:|------:|
| MatMul (10Kx10K) | **14.87 s** | **14.81 ms** | **1,004x** |
| Sigmoid (10K) | 1.87 ms | 154 us | 12.2x |
| Distance (10K) | 2.57 ms | 27.4 us | 93.5x |
| Transform GFLOP/s | - | 14.09 | - |

### Strengths

1. **Vector Operations**: dim=10,000 MatMul에서 Python 대비 **1,004배** 속도 향상
2. **Memory**: Python float 대비 **7배** 메모리 효율 (4B vs ~28B per element)
3. **Quality Metrics**: 모든 RelationKind에서 Omega=1.0, Phi>0.92 자동 산출
4. **Observe Throughput**: Weave 후 **~1,000 obs/sec** - "얽힘은 한 번, 관측은 무한 번"
5. **Linear Scaling**: Pipeline depth, merge count, parse time 모두 선형 확장

### Weaknesses

1. **Small dim overhead**: dim<32에서 NumPy 호출 오버헤드로 Python 루프보다 이점 적음
2. **Weave cost**: 노드당 ~10ms 사전 구축 필요 (단, 한 번만 수행)
3. **Matrix literals**: 큰 행렬을 DSL로 표기할 때 토큰 효율 감소
4. **Grover on classical HW**: 고전 하드웨어에서는 양자 시뮬레이션 오버헤드 존재

### Complexity Summary

| Operation | Complexity | Verified |
|:----------|:-----------|:---------|
| Transform | O(D^2) | GFLOP/s scales with D |
| Merge | O(N*D) | Linear in vector count |
| Distance | O(D) | Sub-linear growth observed |
| Map (elem-wise) | O(D) | 100-1700 M elem/s |
| Pipeline | O(depth * D^2) | Per-step constant ~85us |
| Weave | O(N * D^2) | Per-node constant ~10ms |
| Observe | O(transitions * D^2) | ~1ms per observation |
| DSL Parse | O(L) | ~0.6ms per KB |
| Token scaling | O(N) | AXOL constant < Python/C# |
