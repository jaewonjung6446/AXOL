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
| 4 | 50 | 595.7 |
| 4 | 200 | 2549.2 |
| 8 | 50 | 619.7 |
| 8 | 200 | 2043.1 |
| 16 | 50 | 722.0 |
| 16 | 200 | 2125.7 |
| 32 | 50 | 504.8 |
| 32 | 200 | 2281.0 |
| 64 | 50 | 754.4 |
| 64 | 200 | 2342.6 |
| 128 | 50 | 687.6 |
| 128 | 200 | 3049.1 |

### 2.2 Fractal Dimension Estimation Speed

| Points | Phase Dim | Box-counting (us) | Correlation (us) |
|--------|-----------|-------------------|------------------|
| 50 | 2 | 668.0 | 9072.5 |
| 50 | 4 | 772.0 | 8217.5 |
| 200 | 2 | 2815.5 | 116239.6 |
| 200 | 4 | 2471.4 | 110732.8 |
| 500 | 2 | 5060.2 | 710308.6 |
| 500 | 4 | 5305.4 | 686375.5 |
| 1000 | 2 | 10829.4 | 696612.8 |
| 1000 | 4 | 11943.3 | 697649.6 |
| 2000 | 2 | 19524.0 | 711464.1 |
| 2000 | 4 | 23749.5 | 715662.8 |

### 2.3 Weave Speed

| Nodes | Dimension | Avg Time (ms) |
|-------|-----------|---------------|
| 1 | 8 | 8.83 |
| 2 | 8 | 16.58 |
| 4 | 8 | 32.84 |
| 8 | 8 | 70.81 |
| 16 | 8 | 136.75 |
| 2 | 4 | 15.93 |
| 2 | 8 | 16.31 |
| 2 | 16 | 18.75 |
| 2 | 32 | 24.14 |
| 2 | 64 | 33.02 |

### 2.4 Observe Speed

| Dimension | Single Observe (us) | Reobserve x10 (us) |
|-----------|--------------------|--------------------|
| 4 | 29.7 | 348.2 |
| 8 | 31.7 | 514.8 |
| 16 | 40.0 | 524.5 |
| 32 | 45.8 | 377.2 |
| 64 | 27.4 | 664.5 |

### 2.5 DSL Parse Speed

| Source | Characters | Avg Time (us) |
|--------|------------|---------------|
| simple (1 relation) | 65 | 21.1 |
| medium (3 relations) | 114 | 57.3 |
| full program | 267 | 64.0 |

### 2.6 End-to-End Pipeline (dim=16, 2 relations)

| Phase | Avg Time |
|-------|----------|
| Parse DSL | 43.4 us |
| Estimate cost | 39.2 us |
| Weave (build tapestry) | 18.40 ms |
| Observe (single) | 1.69 ms |
| Reobserve x10 | 16.31 ms |
| Full pipeline (parse->weave->observe) | 23.07 ms |

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
