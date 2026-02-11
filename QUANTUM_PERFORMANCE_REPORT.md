# AXOL Quantum Module Performance Report

Auto-generated benchmark results for the chaos-theory-based quantum module.

## 1. Accuracy

### 1.1 Lyapunov Estimation Accuracy

| System | Expected | Measured | Abs Error |
|--------|----------|----------|-----------|
| Contracting (0.5*I, dim=4) | -0.6931 | -0.6931 | 0.0000 |
| Expanding (2.0*I, dim=4) | 0.6931 | 0.6931 | 0.0000 |
| Identity (dim=4) | 0.0000 | 0.0000 | 0.0000 |
| Lorenz-like (lambda~0.91, dim=3) | 0.9100 | 0.9090 | 0.0010 |
| Strong contraction (0.01*I) | -4.6052 | -4.6052 | 0.0000 |
| Pure rotation (2D) | 0.0000 | -0.0000 | 0.0000 |
| **Average** | | | **0.0002** |

### 1.2 Fractal Dimension Estimation Accuracy

| Geometry | Expected | Measured | Abs Error |
|----------|----------|----------|-----------|
| Point cluster (2D) | 0.000 | 0.934 | 0.934 |
| Line segment (2D) | 1.000 | 1.000 | 0.000 |
| Filled square (2D) | 2.000 | 1.384 | 0.616 |
| Filled cube (3D) | 3.000 | 1.072 | 1.928 |
| Cantor-like set | 0.631 | 0.722 | 0.091 |
| Sierpinski triangle | 1.585 | 1.537 | 0.048 |
| **Average** | | | **0.603** |

### 1.3 Omega Formula Verification

`Omega = 1/(1+max(lambda,0))`

| lambda | Expected | Actual | Error |
|--------|----------|--------|-------|
| -5.00 | 1.000000 | 1.000000 | 0.00e+00 |
| -2.00 | 1.000000 | 1.000000 | 0.00e+00 |
| -1.00 | 1.000000 | 1.000000 | 0.00e+00 |
| -0.50 | 1.000000 | 1.000000 | 0.00e+00 |
| 0.00 | 1.000000 | 1.000000 | 0.00e+00 |
| 0.50 | 0.666667 | 0.666667 | 0.00e+00 |
| 0.91 | 0.523560 | 0.523560 | 0.00e+00 |
| 1.00 | 0.500000 | 0.500000 | 0.00e+00 |
| 2.00 | 0.333333 | 0.333333 | 0.00e+00 |
| 5.00 | 0.166667 | 0.166667 | 0.00e+00 |
| 10.00 | 0.090909 | 0.090909 | 0.00e+00 |

### 1.4 Phi Formula Verification

`Phi = 1/(1+D/D_max)`

| D | D_max | Expected | Actual | Error |
|---|-------|----------|--------|-------|
| 0.00 | 4 | 1.000000 | 1.000000 | 0.00e+00 |
| 1.00 | 4 | 0.800000 | 0.800000 | 0.00e+00 |
| 2.00 | 4 | 0.666667 | 0.666667 | 0.00e+00 |
| 4.00 | 4 | 0.500000 | 0.500000 | 0.00e+00 |
| 2.06 | 3 | 0.592885 | 0.592885 | 0.00e+00 |
| 0.00 | 1 | 1.000000 | 1.000000 | 0.00e+00 |
| 10.00 | 10 | 0.500000 | 0.500000 | 0.00e+00 |

### 1.5 Phi from Entropy

| Distribution | Expected | Actual |
|-------------|----------|--------|
| Delta [0,0,1,0] | 1.0000 | 1.0000 |
| Uniform [.25]*4 | 0.0000 | 0.0000 |
| Peaked [.01,.01,.97,.01] | ~ | 0.8790 |
| Binary [.5,.5,0,0] | ~ | 0.5000 |
| Uniform [.125]*8 | 0.0000 | 0.0000 |
| Delta 1-of-8 | 1.0000 | 1.0000 |

### 1.6 Composition Rules Verification

| Test | Metrics | Result |
|------|---------|--------|
| Serial: 2 convergent | lambda=-1.50, Omega=1.0000, Phi=0.7200, D=0.80 | PASS |
| Serial: 2 chaotic | lambda=0.80, Omega=0.5556, Phi=0.4200, D=1.50 | PASS |
| Parallel: weak link | lambda=1.00, Omega=0.5000, Phi=0.3000, D=2.00 | PASS |
| Serial 5-stage (lambda=0.1 each) | lambda=0.50, Omega=0.6667, Phi=0.5905, D=1.50 | PASS |

### 1.7 Observation Consistency

| Test | Value | Result |
|------|-------|--------|
| Single observe x50 (argmax stability) | 1.0000 | PASS |
| Reobserve x20 (Omega) | 1.0000 | PASS |
| Reobserve x20 (Phi) | 0.8104 | PASS |
| Probability normalisation | 1.0000 | PASS |
| Different inputs -> different outputs | 1.8611 | PASS |

## 2. Speed Benchmarks

### 2.1 Lyapunov Estimation Speed

| Dimension | Steps | Avg Time (us) |
|-----------|-------|---------------|
| 4 | 50 | 810.8 |
| 4 | 200 | 3073.5 |
| 8 | 50 | 976.7 |
| 8 | 200 | 3210.8 |
| 16 | 50 | 750.8 |
| 16 | 200 | 2864.1 |
| 32 | 50 | 699.0 |
| 32 | 200 | 3033.4 |
| 64 | 50 | 845.4 |
| 64 | 200 | 3048.0 |
| 128 | 50 | 938.9 |
| 128 | 200 | 3524.1 |

### 2.2 Fractal Dimension Estimation Speed

| Points | Phase Dim | Box-counting (us) | Correlation (us) |
|--------|-----------|-------------------|------------------|
| 50 | 2 | 1068.2 | 10527.0 |
| 50 | 4 | 931.7 | 11917.6 |
| 200 | 2 | 2957.7 | 156530.3 |
| 200 | 4 | 2955.4 | 156331.8 |
| 500 | 2 | 5147.6 | 969928.3 |
| 500 | 4 | 7651.1 | 979772.9 |
| 1000 | 2 | 15854.8 | 1097654.1 |
| 1000 | 4 | 23120.4 | 1004130.3 |
| 2000 | 2 | 27803.3 | 992649.0 |
| 2000 | 4 | 37362.2 | 958928.4 |

### 2.3 Weave Speed

| Nodes | Dimension | Avg Time (ms) |
|-------|-----------|---------------|
| 1 | 8 | 11.72 |
| 2 | 8 | 22.54 |
| 4 | 8 | 40.27 |
| 8 | 8 | 75.21 |
| 16 | 8 | 155.99 |
| 2 | 4 | 19.20 |
| 2 | 8 | 23.06 |
| 2 | 16 | 24.27 |
| 2 | 32 | 32.66 |
| 2 | 64 | 37.31 |

### 2.4 Observe Speed

| Dimension | Single Observe (us) | Reobserve x10 (us) |
|-----------|--------------------|--------------------|
| 4 | 39.6 | 654.5 |
| 8 | 22.1 | 588.3 |
| 16 | 36.2 | 605.2 |
| 32 | 34.2 | 555.3 |
| 64 | 44.2 | 644.1 |

### 2.5 DSL Parse Speed

| Source | Characters | Avg Time (us) |
|--------|------------|---------------|
| simple (1 relation) | 65 | 28.7 |
| medium (3 relations) | 114 | 72.3 |
| full program | 267 | 100.0 |

### 2.6 End-to-End Pipeline (dim=16, 2 relations)

| Phase | Avg Time |
|-------|----------|
| Parse DSL | 73.4 us |
| Estimate cost | 68.8 us |
| Weave (build tapestry) | 22.83 ms |
| Observe (single) | 2.18 ms |
| Reobserve x10 | 21.46 ms |
| Full pipeline (parse->weave->observe) | 27.17 ms |

## 3. Token Efficiency

Quantum DSL vs equivalent Python (tiktoken cl100k_base)

| Program | Python Tokens | Python Lines | DSL Tokens | DSL Lines | Saving |
|---------|--------------|-------------|------------|-----------|--------|
| search | 173 | 16 | 57 | 5 | 67% |
| classify | 129 | 12 | 39 | 4 | 70% |
| pipeline | 210 | 22 | 73 | 9 | 65% |
| multi_input | 191 | 15 | 74 | 6 | 61% |
| reobserve_pattern | 131 | 16 | 62 | 7 | 53% |
| **TOTAL** | **834** | | **305** | | **63%** |
