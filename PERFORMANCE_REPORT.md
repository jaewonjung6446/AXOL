# Axol Performance Report

Auto-generated benchmark results.

## 1. Token Cost Comparison

| Program | Python | C# | Axol | Saving vs Python |
|---------|--------|----|------|------------------|
| counter | 8 | 5 | 11 | -38% |
| hp_decay | 9 | 7 | 14 | -56% |
| state_machine | 9 | 6 | 14 | -56% |

## 2. Runtime Benchmarks

| Dimension | Avg Time (μs) |
|-----------|---------------|
| 4 | 122.9 |
| 100 | 144.1 |
| 1000 | 2601.4 |

## 3. Optimizer Effect

- Original: 2 transitions, 136.4 μs
- Optimized: 1 transitions, 85.7 μs
- Speedup: 1.59x

## 4. Encryption Overhead

- Plaintext: 567.3 μs
- Encrypted: 240582.6 μs
- Overhead: 424.08x

## 5. Scaling Analysis (N-State Automaton)

| N States | Avg Time (μs) |
|----------|---------------|
| 5 | 1116.0 |
| 10 | 1983.4 |
| 20 | 4365.4 |
| 50 | 11326.7 |
| 100 | 22438.6 |

## 6. Backend Comparison

| Backend | Avg Time (μs) |
|---------|---------------|
| numpy | 73.0 |
