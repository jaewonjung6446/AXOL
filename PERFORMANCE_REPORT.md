# Axol Performance Report

Auto-generated benchmark results.

## 1. Token Cost Comparison

| Program | Python | C# | Axol | Saving vs Python |
|---------|--------|----|------|------------------|
| counter | 27 | 22 | 32 | -19% |
| hp_decay | 30 | 36 | 50 | -67% |
| state_machine | 38 | 43 | 47 | -24% |

## 2. Runtime Benchmarks

| Dimension | Avg Time (μs) |
|-----------|---------------|
| 4 | 107.6 |
| 100 | 90.9 |
| 1000 | 2661.8 |

## 3. Optimizer Effect

- Original: 2 transitions, 93.7 μs
- Optimized: 1 transitions, 54.2 μs
- Speedup: 1.73x

## 4. Encryption Overhead

- Plaintext: 456.8 μs
- Encrypted: 224889.5 μs
- Overhead: 492.36x

## 5. Scaling Analysis (N-State Automaton)

| N States | Avg Time (μs) |
|----------|---------------|
| 5 | 1160.6 |
| 10 | 2052.0 |
| 20 | 3477.1 |
| 50 | 9230.1 |
| 100 | 17068.2 |

## 6. Backend Comparison

| Backend | Avg Time (μs) |
|---------|---------------|
| numpy | 45.9 |
