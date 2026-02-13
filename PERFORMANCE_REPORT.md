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
| 4 | 89.2 |
| 100 | 83.7 |
| 1000 | 2601.0 |

## 3. Optimizer Effect

- Original: 2 transitions, 92.8 μs
- Optimized: 1 transitions, 48.2 μs
- Speedup: 1.93x

## 4. Encryption Overhead

- Plaintext: 381.0 μs
- Encrypted: 184242.6 μs
- Overhead: 483.54x

## 5. Scaling Analysis (N-State Automaton)

| N States | Avg Time (μs) |
|----------|---------------|
| 5 | 682.8 |
| 10 | 1468.1 |
| 20 | 3101.2 |
| 50 | 8678.1 |
| 100 | 16035.4 |

## 6. Backend Comparison

| Backend | Avg Time (μs) |
|---------|---------------|
| numpy | 48.9 |
