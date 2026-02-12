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
| 4 | 158.4 |
| 100 | 122.6 |
| 1000 | 3047.1 |

## 3. Optimizer Effect

- Original: 2 transitions, 142.8 μs
- Optimized: 1 transitions, 71.9 μs
- Speedup: 1.98x

## 4. Encryption Overhead

- Plaintext: 507.2 μs
- Encrypted: 216669.0 μs
- Overhead: 427.19x

## 5. Scaling Analysis (N-State Automaton)

| N States | Avg Time (μs) |
|----------|---------------|
| 5 | 800.0 |
| 10 | 1576.4 |
| 20 | 3414.3 |
| 50 | 9626.4 |
| 100 | 20808.1 |

## 6. Backend Comparison

| Backend | Avg Time (μs) |
|---------|---------------|
| numpy | 68.5 |
