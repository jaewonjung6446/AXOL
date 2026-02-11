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
| 4 | 75.5 |
| 100 | 90.6 |
| 1000 | 2832.5 |

## 3. Optimizer Effect

- Original: 2 transitions, 91.9 μs
- Optimized: 1 transitions, 59.4 μs
- Speedup: 1.55x

## 4. Encryption Overhead

- Plaintext: 519.5 μs
- Encrypted: 229824.6 μs
- Overhead: 442.42x

## 5. Scaling Analysis (N-State Automaton)

| N States | Avg Time (μs) |
|----------|---------------|
| 5 | 802.1 |
| 10 | 2199.5 |
| 20 | 5395.3 |
| 50 | 11553.8 |
| 100 | 20138.5 |

## 6. Backend Comparison

| Backend | Avg Time (μs) |
|---------|---------------|
| numpy | 97.0 |
