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
| 4 | 76.9 |
| 100 | 81.8 |
| 1000 | 2410.2 |

## 3. Optimizer Effect

- Original: 2 transitions, 108.8 μs
- Optimized: 1 transitions, 54.3 μs
- Speedup: 2.00x

## 4. Encryption Overhead

- Plaintext: 431.5 μs
- Encrypted: 238707.9 μs
- Overhead: 553.17x

## 5. Scaling Analysis (N-State Automaton)

| N States | Avg Time (μs) |
|----------|---------------|
| 5 | 830.7 |
| 10 | 1703.0 |
| 20 | 4107.2 |
| 50 | 9046.5 |
| 100 | 21823.3 |

## 6. Backend Comparison

| Backend | Avg Time (μs) |
|---------|---------------|
| numpy | 61.8 |
