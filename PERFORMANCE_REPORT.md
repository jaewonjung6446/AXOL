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
| 4 | 92.5 |
| 100 | 97.9 |
| 1000 | 2200.9 |

## 3. Optimizer Effect

- Original: 2 transitions, 100.9 μs
- Optimized: 1 transitions, 52.1 μs
- Speedup: 1.94x

## 4. Encryption Overhead

- Plaintext: 425.3 μs
- Encrypted: 240449.3 μs
- Overhead: 565.32x

## 5. Scaling Analysis (N-State Automaton)

| N States | Avg Time (μs) |
|----------|---------------|
| 5 | 932.9 |
| 10 | 1979.9 |
| 20 | 4219.6 |
| 50 | 12276.6 |
| 100 | 24403.2 |

## 6. Backend Comparison

| Backend | Avg Time (μs) |
|---------|---------------|
| numpy | 63.9 |
