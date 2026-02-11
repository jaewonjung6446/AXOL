<p align="center">
  <h1 align="center">AXOL</h1>
  <p align="center">
    <strong>Spatial-Probabilistic Computing Language Based on Chaos Theory</strong>
  </p>
  <p align="center">
    <a href="#"><img src="https://img.shields.io/badge/status-experimental-orange" alt="Status: Experimental"></a>
    <a href="#"><img src="https://img.shields.io/badge/version-0.2.0-blue" alt="Version 0.2.0"></a>
    <a href="#"><img src="https://img.shields.io/badge/python-3.11%2B-brightgreen" alt="Python 3.11+"></a>
    <a href="#"><img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="License: MIT"></a>
  </p>
  <p align="center">
    <a href="README.md">English</a> |
    <a href="README.ko.md">한국어</a> |
    <a href="README.ja.md">日本語</a> |
    <a href="README.zh.md">中文</a>
  </p>
</p>

---

> **WARNING: This project is in an early experimental stage.**
> APIs, DSL syntax, and internal architecture are subject to breaking changes without notice. Not recommended for production use. Contributions and feedback are welcome.

---

## What is AXOL?

**AXOL** is a programming language that **denies the time-axis** (sequential execution) as the foundation of computation and replaces it with two alternative axes:

- **Spatial axis** — relationships between nodes determine computation
- **Probability axis** — likelihood of outcomes determines results

Instead of "do this, then do that," AXOL asks: "what relates to what, and how strongly?" The result is a fundamentally different execution model based on **chaos theory**, where computation is the act of building and observing **strange attractors**.

```
Traditional:  instruction 1 → instruction 2 → instruction 3  (time-sequential)

AXOL:
  [Spatial]     NodeA ──relation── NodeB     "where" determines computation
  [Probability] state = { alpha|possibility1> + beta|possibility2> }  "how likely" determines results
```

### Key Properties

- **Three-phase execution**: Declare → Weave → Observe (not compile → run)
- **Chaos theory foundation**: Tapestry = Strange Attractor, quality measured by Lyapunov exponents and fractal dimensions
- **Dual quality metrics**: Omega (Cohesion) + Phi (Clarity) — rigorous, measurable, composable
- **63% average token savings** vs equivalent Python (quantum DSL)
- **Infeasibility detection** — warns before computation when targets are mathematically unachievable
- **Lyapunov estimation accuracy**: average error 0.0002
- **9 primitive operations** in the foundation layer: `transform`, `gate`, `merge`, `distance`, `route` (encrypted) + `step`, `branch`, `clamp`, `map` (plaintext)
- **Matrix-level encryption** — similarity transformation makes programs cryptographically unreadable
- **NumPy backend** with optional GPU acceleration (CuPy/JAX)

---

## Table of Contents

- [The Paradigm Shift](#the-paradigm-shift)
- [Three-Phase Execution Model](#three-phase-execution-model)
- [Quality Metrics](#quality-metrics)
- [Chaos Theory Foundation](#chaos-theory-foundation)
- [Composition Rules](#composition-rules)
- [Quantum DSL](#quantum-dsl)
- [Performance](#performance)
- [Foundation Layer](#foundation-layer)
  - [9 Primitive Operations](#9-primitive-operations)
  - [Matrix Encryption (Shadow AI)](#matrix-encryption-shadow-ai)
  - [Plaintext Operations & Security Classification](#plaintext-operations--security-classification)
  - [Compiler Optimizer](#compiler-optimizer)
  - [GPU Backend](#gpu-backend)
  - [Module System](#module-system)
  - [Quantum Interference (Phase 6)](#quantum-interference-phase-6)
  - [Client-Server Architecture](#client-server-architecture)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Test Suite](#test-suite)
- [Roadmap](#roadmap)

---

## The Paradigm Shift

### What We Deny

Every modern programming language is built on the **time-axis** (sequential execution):

| Paradigm | Time-Axis Dependency |
|----------|---------------------|
| Imperative (C, Python) | "Do this first, then that" — explicit time order |
| Functional (Haskell, Lisp) | Declarative, but evaluation order exists |
| Parallel (Go, Rust async) | Multiple time-axes simultaneously — still time-bound |
| Declarative (SQL, HTML) | Describes "what," but the engine processes on a time-axis |

This is because the Von Neumann architecture operates on clock cycles — a time-axis.

### What We Propose

AXOL replaces the time-axis with two alternative axes:

| Axis | What It Determines | Analogy |
|------|-------------------|---------|
| **Spatial axis** (relationships) | Nodes connected by relations determine computation | "Where something is" matters, not "when" |
| **Probability axis** (possibilities) | Superposed states collapse to most-probable outcomes | "How likely" matters, not "exact" |

The tradeoff: **we sacrifice exactness for the elimination of time-bottlenecks.**

```
Exactness ↑  →  Entanglement cost ↑  →  Build time ↑
Exactness ↓  →  Entanglement cost ↓  →  Build time ↓
               but observation is always instant
```

### Why This Matters

| Property | Traditional Compilation | AXOL Entanglement |
|----------|----------------------|-------------------|
| Preparation | Code → machine translation | Build probabilistic correlations between logic |
| Execution | Sequential machine instructions | Observation (input) → instant collapse |
| Bottleneck | Proportional to execution path length | Depends only on entanglement depth |
| Analogy | "Building a fast road" | "Already being at the destination" |

---

## Three-Phase Execution Model

### Phase 1: Declare

Define **what relates to what** and set quality targets. No computation happens yet.

```python
from axol.quantum import DeclarationBuilder, RelationKind

decl = (
    DeclarationBuilder("search")
    .input("query", 64)
    .input("db", 64)
    .relate("relevance", ["query", "db"], RelationKind.PROPORTIONAL)
    .output("relevance")
    .quality(omega=0.9, phi=0.7)   # quality targets
    .build()
)
```

Or in DSL:

```
entangle search(query: float[64], db: float[64]) @ Omega(0.9) Phi(0.7) {
    relevance <~> similarity(query, db)
    ranking <~> relevance
}
```

### Phase 2: Weave

Build the **strange attractor** (Tapestry). This is where computation cost is paid. The weaver:

1. Estimates entanglement cost
2. Detects infeasibility (warns if targets are mathematically unachievable)
3. Builds attractor structures per node (trajectory matrices, Hadamard interference)
4. Estimates Lyapunov exponents and fractal dimensions
5. Assembles the internal `Program` for execution

```python
from axol.quantum import weave

tapestry = weave(decl, seed=42)
print(tapestry.weaver_report)
# target:   Omega(0.90) Phi(0.70)
# achieved: Omega(0.95) Phi(0.82)
# feasible: True
```

Infeasibility detection example:

```
> weave predict_weather: WARNING
>   target:   Omega(0.99) Phi(0.99)
>   maximum:  Omega(0.71) Phi(0.68)
>   reason:   chaotic dependency (lambda=2.16 on path: input->atmosphere->prediction)
>   attractor_dim: D=2.06 (Lorenz-class)
```

### Phase 3: Observe

Input values → **instant collapse** to a point on the attractor. Time complexity: O(D) where D is the attractor's embedding dimension.

```python
from axol.quantum import observe, reobserve
from axol.core.types import FloatVec

# Single observation
result = observe(tapestry, {
    "query": FloatVec.from_list([1.0] * 64),
    "db": FloatVec.from_list([0.5] * 64),
})
print(f"Omega: {result.omega:.2f}, Phi: {result.phi:.2f}")

# Repeated observation for quality improvement
result = reobserve(tapestry, inputs, count=10)
# Averages probability distributions, recalculates empirical Omega
```

### Full Pipeline (DSL)

```
entangle search(query: float[64], db: float[64]) @ Omega(0.9) Phi(0.7) {
    relevance <~> similarity(query, db)
    ranking <~> relevance
}

result = observe search(query_vec, db_vec)

if result.Omega < 0.95 {
    result = reobserve search(query_vec, db_vec) x 10
}
```

---

## Quality Metrics

AXOL measures computation quality on two independent axes:

```
        Phi (Clarity)
        ^
   1.0  |  Sharp but unstable    Ideal (strong entanglement)
        |
   0.0  |  Noise                 Stable but blurry
        +-----------------------------> Omega (Cohesion)
       0.0                             1.0
```

### Omega — Cohesion (How stable?)

Derived from the **maximum Lyapunov exponent** (lambda):

```
Omega = 1 / (1 + max(lambda, 0))
```

| lambda | Meaning | Omega |
|--------|---------|-------|
| lambda < 0 | Converging system (stable) | 1.0 |
| lambda = 0 | Neutral stability | 1.0 |
| lambda = 0.91 | Lorenz-class chaos | 0.52 |
| lambda = 2.0 | Strong chaos | 0.33 |

**Interpretation**: Omega = 1.0 means repeated observations always give the same result. Omega < 1.0 means chaotic sensitivity — small input changes cause different outputs.

### Phi — Clarity (How sharp?)

Derived from the **fractal dimension** (D) of the attractor:

```
Phi = 1 / (1 + D / D_max)
```

| D | D_max | Meaning | Phi |
|---|-------|---------|-----|
| 0 | any | Point (delta distribution) | 1.0 |
| 1 | 4 | Line attractor | 0.80 |
| 2.06 | 3 | Lorenz attractor | 0.59 |
| D_max | D_max | Fills entire phase space | 0.50 |

**Interpretation**: Phi = 1.0 means the output is a sharp, definite value. Phi → 0.0 means the output is spread across many possibilities (noise).

### Both Metrics Are Composable

Quality metrics propagate through composition — see [Composition Rules](#composition-rules).

---

## Chaos Theory Foundation

AXOL's theoretical foundation maps its concepts to well-established chaos theory:

| AXOL Concept | Chaos Theory | Mathematical Object |
|---|---|---|
| Tapestry | Strange Attractor | Compact invariant set in phase space |
| Omega (Cohesion) | Lyapunov Stability | `1/(1+max(lambda,0))` |
| Phi (Clarity) | Fractal Dimension Inverse | `1/(1+D/D_max)` |
| Weave | Attractor Construction | Iterative map's trajectory matrix |
| Observe | Point Collapse on Attractor | Time complexity O(D) |
| Entanglement Range | Basin of Attraction | Boundary of convergence region |
| Entanglement Cost | Convergence Iterations | `E = sum_path(iterations * complexity)` |
| Reuse After Observe | Attractor Stability | lambda < 0: reusable, lambda > 0: re-weave |

### Lyapunov Exponent Estimation

Uses the **Benettin QR decomposition method** to estimate the maximum Lyapunov exponent from trajectory matrices.

- **Contracting systems** (lambda < 0): predictable, Omega approaches 1.0
- **Neutral systems** (lambda = 0): edge of chaos
- **Chaotic systems** (lambda > 0): sensitive to initial conditions, Omega < 1.0

Estimation accuracy verified against known systems (average error: 0.0002).

### Fractal Dimension Estimation

Two methods available:

- **Box-counting**: Grid-based, regression of ln(N) vs ln(1/epsilon)
- **Correlation dimension** (Grassberger-Procaccia): Pairwise distance analysis

Verified against known geometries: line segments (D=1), Cantor sets (D~0.63), Sierpinski triangles (D~1.58).

### Full Theory Documents

- [THEORY.md](THEORY.md) — Foundational theory (time-axis denial, entanglement-based computation)
- [THEORY_MATH.md](THEORY_MATH.md) — Chaos theory formalization (Lyapunov, fractal, composition proofs)

---

## Composition Rules

When combining multiple tapestries, quality metrics propagate according to strict mathematical rules:

### Serial Composition (A → B)

```
lambda_total = lambda_A + lambda_B          (exponents accumulate)
Omega_total  = 1/(1+max(lambda_total, 0))   (Omega degrades)
D_total      = D_A + D_B                    (dimension sums)
Phi_total    = Phi_A * Phi_B                (Phi multiplies — always degrades)
```

### Parallel Composition (A || B)

```
lambda_total = max(lambda_A, lambda_B)      (weakest link)
Omega_total  = min(Omega_A, Omega_B)        (weakest link)
D_total      = max(D_A, D_B)               (most complex)
Phi_total    = min(Phi_A, Phi_B)            (least clear)
```

### Reuse Rule

```
lambda < 0  →  reusable after observation (attractor is stable)
lambda > 0  →  must re-weave after observation (chaotic — attractor disrupted)
```

### Summary Table

| Mode | lambda | Omega | D | Phi |
|------|--------|-------|---|-----|
| Serial | sum | 1/(1+max(sum,0)) | sum | product |
| Parallel | max | min | max | min |

---

## Quantum DSL

### Syntax Overview

```
entangle NAME(PARAM: TYPE[DIM], ...) @ Omega(X) Phi(Y) {
    TARGET <OP> SOURCE_EXPRESSION
    ...
}

result = observe NAME(args...)
result = reobserve NAME(args...) x COUNT

if result.FIELD OP VALUE {
    ...
}
```

### Relation Operators

| Operator | Name | Meaning |
|----------|------|---------|
| `<~>` | Proportional | Linear correlation |
| `<+>` | Additive | Weighted sum |
| `<*>` | Multiplicative | Product relationship |
| `<!>` | Inverse | Inverse correlation |
| `<?>` | Conditional | Context-dependent |

### Examples

#### Simple Search

```
entangle search(query: float[64], db: float[64]) @ Omega(0.9) Phi(0.7) {
    relevance <~> similarity(query, db)
    ranking <~> relevance
}
result = observe search(query_vec, db_vec)
```

#### Multi-Stage Pipeline

```
entangle analyze(data: float[128], model: float[128]) @ Omega(0.85) Phi(0.8) {
    features <~> extract(data)
    prediction <~> apply(features, model)
    confidence <+> validate(prediction, data)
}

result = observe analyze(data_vec, model_vec)

if result.Omega < 0.9 {
    result = reobserve analyze(data_vec, model_vec) x 20
}
```

#### Classification

```
entangle classify(input: float[32]) @ Omega(0.95) Phi(0.9) {
    category <~> classify(input)
}
result = observe classify(input_vec)
```

---

## Performance

### Token Efficiency — Quantum DSL vs Python

Measured with `tiktoken` cl100k_base tokenizer.

| Program | Python Tokens | DSL Tokens | Saving |
|---------|:------------:|:----------:|:------:|
| search | 173 | 57 | **67%** |
| classify | 129 | 39 | **70%** |
| pipeline | 210 | 73 | **65%** |
| multi_input | 191 | 74 | **61%** |
| reobserve_pattern | 131 | 62 | **53%** |
| **Total** | **834** | **305** | **63%** |

### Token Efficiency — Foundation DSL vs Python vs C#

| Program | Python | C# | Axol DSL | vs Python | vs C# |
|---------|:------:|:--:|:--------:|:---------:|:-----:|
| Counter | 32 | 61 | 33 | -3% | 46% |
| State Machine | 67 | 147 | 48 | 28% | 67% |
| Combat Pipeline | 145 | 203 | 66 | 55% | 68% |
| 100-State Automaton | 739 | 869 | 636 | 14% | 27% |

### Accuracy

| Metric | Value |
|--------|-------|
| Lyapunov estimation average error | **0.0002** |
| Omega formula error | **0** (exact) |
| Phi formula error | **0** (exact) |
| Composition rules | **All PASS** |
| Observation consistency (50 repeats) | **1.0000** |

### Speed

| Operation | Time |
|-----------|------|
| DSL parse (simple) | ~25 us |
| DSL parse (full program) | ~62 us |
| Cost estimation | ~40 us |
| Single observation | ~300 us |
| Weave (2 nodes, dim=8) | ~14 ms |
| Reobserve x10 | ~14 ms |
| **Full pipeline** (parse → weave → observe, dim=16) | **~17 ms** |

### Scaling

| Nodes | Dimension | Weave Time |
|:-----:|:---------:|:----------:|
| 1 | 8 | 9 ms |
| 4 | 8 | 25 ms |
| 16 | 8 | 108 ms |
| 2 | 4 | 12 ms |
| 2 | 64 | 39 ms |

Full benchmark data: [QUANTUM_PERFORMANCE_REPORT.md](QUANTUM_PERFORMANCE_REPORT.md) | [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md)

---

## Foundation Layer

The quantum module (`axol/quantum/`) is built on top of the foundation layer (`axol/core/`) without modifying it. The foundation provides the mathematical engine: vector types, matrix operations, program execution, encryption, and optimization.

### 9 Primitive Operations

| Operation | Security | Mathematical Basis | Description |
|-----------|:--------:|-------------------|-------------|
| `transform` | **E** | Matrix multiplication: `v @ M` | Linear state transformation |
| `gate` | **E** | Hadamard product: `v * g` | Conditional masking |
| `merge` | **E** | Weighted sum: `sum(v_i * w_i)` | Vector combination |
| `distance` | **E** | L2 / cosine / dot | Similarity measurement |
| `route` | **E** | `argmax(v @ R)` | Discrete branching |
| `step` | **P** | `where(v >= t, 1, 0)` | Threshold to binary gate |
| `branch` | **P** | `where(g, then, else)` | Conditional vector select |
| `clamp` | **P** | `clip(v, min, max)` | Value range restriction |
| `map` | **P** | `f(v)` element-wise | Nonlinear activation |

The 5 **E** (Encrypted) operations can run on encrypted data via similarity transformation. The 4 **P** (Plaintext) operations add nonlinear expressiveness.

### Matrix Encryption (Shadow AI)

All computation in Axol reduces to matrix multiplication (`v @ M`). This enables **similarity transformation encryption**:

```
M' = K^(-1) @ M @ K     (encrypted operation matrix)
state' = state @ K       (encrypted initial state)
result = result' @ K^(-1)(decrypted output)
```

- The encrypted program runs correctly in the encrypted domain
- All business logic is hidden — matrices appear as random noise
- Different from obfuscation — this is cryptographic transformation
- All 5 E operations verified (21 tests in `tests/test_encryption.py`)

### Plaintext Operations & Security Classification

Every operation carries a `SecurityLevel` (E or P). The built-in analyzer reports encryption coverage:

```python
from axol.core import parse, analyze

result = analyze(program)
print(result.summary())
# Program: damage_calc
# Transitions: 3 total, 1 encrypted (E), 2 plaintext (P)
# Coverage: 33.3%
```

### Compiler Optimizer

Three-pass optimization: transform fusion, dead state elimination, constant folding.

```python
from axol.core import parse, optimize, run_program

program = parse(source)
optimized = optimize(program)
result = run_program(optimized)
```

### GPU Backend

Pluggable array backend: `numpy` (default), `cupy` (NVIDIA GPU), `jax`.

```python
from axol.core import set_backend
set_backend("cupy")   # NVIDIA GPU
set_backend("jax")    # Google JAX
```

### Module System

Reusable, composable programs with schemas, imports, and sub-module execution.

```
@main
import damage_calc from "damage_calc.axol"
s atk=[50] def_val=[10]
: calc=use(damage_calc;in=atk,def_val;out=dmg)
```

### Quantum Interference (Phase 6)

Phase 6 introduced quantum interference — Grover search, quantum walk — achieving **100% encryption coverage** for quantum programs. Hadamard, Oracle, and Diffusion generate `TransMatrix` objects used via `TransformOp` (E-class), so the existing optimizer and encryption module work automatically.

```
@grover_4
s state=[0.5 0.5 0.5 0.5]
: oracle=oracle(state;marked=[3];n=4)
: diffuse=diffuse(state;n=4)
? found state[3]>=0.9
```

### Client-Server Architecture

Encrypt on client, compute on untrusted server:

```
Client (key)              Server (no key)
  Program ─── encrypt ──► Encrypted Program
  pad_and_encrypt()       run_program() on noise
                    ◄──── Encrypted Result
  decrypt_result()
  ──► Result
```

Key components: `KeyFamily(seed)`, `fn_to_matrix()`, `pad_and_encrypt()`, `AxolClient` SDK.

---

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │            axol/quantum/                     │
                    │                                             │
  Quantum DSL ──►  │  dsl.py ──► declare.py ──► weaver.py ──►   │
  (entangle,       │               │              │   │          │
   observe,        │  types.py   cost.py    lyapunov.py          │
   reobserve)      │               │        fractal.py           │
                    │           compose.py                        │
                    │               │                             │
                    │           observatory.py ──► Observation    │
                    └──────────────┬──────────────────────────────┘
                                   │ reuses
                    ┌──────────────┴──────────────────────────────┐
                    │            axol/core/                        │
                    │                                             │
  Foundation DSL ►  │  dsl.py ──► program.py ──► operations.py   │
  (@prog,           │              │              │               │
   s/:/?)           │  types.py  optimizer.py  backend.py        │
                    │              │              (numpy/cupy/jax) │
                    │  encryption.py  analyzer.py  module.py      │
                    └──────────────┬──────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────────────────────┐
                    │            axol/api/ + axol/server/          │
                    │  Tool-Use API    FastAPI + HTML/JS debugger  │
                    └─────────────────────────────────────────────┘
```

### Internal Engine Reuse

The quantum module reuses `axol/core` without modification:

| Quantum Concept | Core Implementation |
|----------------|-------------------|
| Attractor amplitudes/trajectories | `FloatVec` |
| Attractor correlation matrix | `TransMatrix` |
| Tapestry internal execution | `Program` + `run_program()` |
| Born rule probabilities | `operations.measure()` |
| Weave transform construction | `TransformOp`, `MergeOp` |
| Attractor exploration diffusion | `hadamard_matrix()`, `diffusion_matrix()` |

---

## Quick Start

### Installation

```bash
git clone https://github.com/your-username/AXOL.git
cd AXOL
pip install -e ".[dev]"
```

### Requirements

- Python 3.11+
- NumPy >= 1.24.0
- pytest >= 7.4.0 (dev)
- tiktoken >= 0.5.0 (dev, for token analysis)
- fastapi >= 0.100.0, uvicorn >= 0.23.0 (optional, for web frontend)
- cupy-cuda12x >= 12.0.0 (optional, for GPU)
- jax[cpu] >= 0.4.0 (optional, for JAX backend)

### Hello World — Quantum DSL (Declare → Weave → Observe)

```python
from axol.quantum import (
    DeclarationBuilder, RelationKind,
    weave, observe, parse_quantum,
)
from axol.core.types import FloatVec

# Option 1: Python API
decl = (
    DeclarationBuilder("hello")
    .input("x", 4)
    .relate("y", ["x"], RelationKind.PROPORTIONAL)
    .output("y")
    .quality(0.9, 0.8)
    .build()
)

tapestry = weave(decl, seed=42)
result = observe(tapestry, {"x": FloatVec.from_list([1, 0, 0, 0])})
print(f"Omega: {result.omega:.2f}, Phi: {result.phi:.2f}")

# Option 2: DSL
program = parse_quantum("""
entangle hello(x: float[4]) @ Omega(0.9) Phi(0.8) {
    y <~> transform(x)
}
""")
```

### Hello World — Foundation DSL (Vector Operations)

```python
from axol.core import parse, run_program

source = """
@counter
s count=[0] one=[1]
: increment=merge(count one;w=[1 1])->count
? done count>=5
"""

program = parse(source)
result = run_program(program)
print(f"Final count: {result.final_state['count'].to_list()}")  # [5.0]
```

---

## API Reference

### Quantum Module (`axol.quantum`)

```python
# Declaration
DeclarationBuilder(name)           # Fluent API for building declarations
  .input(name, dim, labels?)       # Add input
  .output(name)                    # Mark output
  .relate(target, sources, kind)   # Add relation
  .quality(omega, phi)             # Set quality targets
  .build() -> EntangleDeclaration

# Weaving
weave(declaration, encrypt?, seed?, optimize?) -> Tapestry

# Observation
observe(tapestry, inputs, seed?) -> Observation
reobserve(tapestry, inputs, count, seed?) -> Observation

# DSL
parse_quantum(source) -> QuantumProgram

# Lyapunov
estimate_lyapunov(trajectory_matrix, steps?) -> float
lyapunov_spectrum(trajectory_matrix, dim, steps?) -> list[float]
omega_from_lyapunov(lyapunov) -> float

# Fractal
estimate_fractal_dim(attractor_points, method?, phase_space_dim?) -> float
phi_from_fractal(fractal_dim, phase_space_dim) -> float
phi_from_entropy(probs) -> float

# Composition
compose_serial(omega_a, phi_a, lambda_a, d_a, ...) -> tuple
compose_parallel(omega_a, phi_a, lambda_a, d_a, ...) -> tuple
can_reuse_after_observe(lyapunov) -> bool

# Cost
estimate_cost(declaration) -> CostEstimate
```

### Core Types

| Type | Description |
|------|-------------|
| `SuperposedState` | Named state with amplitudes, labels, Born-rule probabilities |
| `Attractor` | Strange attractor with Lyapunov spectrum, fractal dim, trajectory matrix |
| `Tapestry` | Graph of `TapestryNode`s with global attractor and weaver report |
| `Observation` | Collapsed result with value, Omega, Phi, probabilities |
| `WeaverReport` | Target vs achieved quality, feasibility, cost breakdown |
| `CostEstimate` | Per-node cost, critical path, max achievable Omega/Phi |
| `FloatVec` | 32-bit float vector |
| `TransMatrix` | M x N float32 matrix |
| `StateBundle` | Named collection of vectors |
| `Program` | Executable sequence of transitions |

### Foundation Module (`axol.core`)

```python
parse(source) -> Program
run_program(program) -> ExecutionResult
optimize(program) -> Program
set_backend(name)    # "numpy" | "cupy" | "jax"
analyze(program) -> AnalysisResult
dispatch(request) -> dict    # Tool-Use API
```

---

## Examples

### 1. Declare → Weave → Observe (Full Pipeline)

```python
from axol.quantum import *
from axol.core.types import FloatVec

decl = (
    DeclarationBuilder("search")
    .input("query", 64)
    .input("db", 64)
    .relate("relevance", ["query", "db"], RelationKind.PROPORTIONAL)
    .relate("ranking", ["relevance"], RelationKind.PROPORTIONAL)
    .output("ranking")
    .quality(0.9, 0.7)
    .build()
)

tapestry = weave(decl, seed=42)
result = observe(tapestry, {
    "query": FloatVec.zeros(64),
    "db": FloatVec.ones(64),
})
print(f"Omega={result.omega:.2f}, Phi={result.phi:.2f}")
```

### 2. Quantum DSL Round-Trip

```python
from axol.quantum import parse_quantum, weave, observe
from axol.core.types import FloatVec

prog = parse_quantum("""
entangle classify(input: float[32]) @ Omega(0.95) Phi(0.9) {
    category <~> classify(input)
}
result = observe classify(input_vec)
""")

tapestry = weave(prog.declarations[0], seed=0)
result = observe(tapestry, {"input": FloatVec.zeros(32)})
```

### 3. State Machine (Foundation DSL)

```
@state_machine
s state=onehot(0,3)
: advance=transform(state;M=[0 1 0;0 0 1;0 0 1])
? done state[2]>=1
```

### 4. Grover Search (Quantum Interference)

```
@grover_4
s state=[0.5 0.5 0.5 0.5]
: oracle=oracle(state;marked=[3];n=4)
: diffuse=diffuse(state;n=4)
? found state[3]>=0.9
```

### 5. Encrypted Execution

```python
from axol.core import parse, run_program
from axol.core.encryption import encrypt_program, decrypt_state

program = parse("@test\ns v=[1 0 0]\n: t=transform(v;M=[0 1 0;0 0 1;1 0 0])")
encrypted, key = encrypt_program(program)
result = run_program(encrypted)
decrypted = decrypt_state(result.final_state, key)
```

---

## Test Suite

```bash
# Full test suite (545 tests)
pytest tests/ -v

# Quantum module tests (101 tests)
pytest tests/test_quantum_*.py tests/test_lyapunov.py tests/test_fractal.py tests/test_compose.py -v

# Performance benchmarks (generates reports)
pytest tests/test_quantum_performance.py -v -s
pytest tests/test_performance_report.py -v -s

# Core tests
pytest tests/test_types.py tests/test_operations.py tests/test_program.py tests/test_dsl.py -v

# Encryption tests (21 tests)
pytest tests/test_encryption.py -v -s

# Quantum interference tests (37 tests)
pytest tests/test_quantum.py -v -s

# API + Server tests
pytest tests/test_api.py tests/test_server.py -v

# Start web frontend
python -m axol.server   # http://localhost:8080
```

Current: **545 tests passed**, 0 failed, 4 skipped (cupy/jax not installed).

---

## Roadmap

- [x] Phase 1: Type system (7 vector types + StateBundle) + 5 primitive ops + execution engine
- [x] Phase 2: DSL parser + sparse matrix notation + token benchmarks + encryption PoC
- [x] Phase 3: Compiler optimizer (fusion, elimination, folding) + GPU backend
- [x] Phase 4: Tool-Use API + encryption module
- [x] Phase 5: Module system (registry, import/use, compose, schemas)
- [x] Frontend: FastAPI + HTML/JS visual debugger
- [x] Phase 6: Quantum interference (Hadamard/Oracle/Diffusion, 100% E-class coverage)
- [x] Phase 7: KeyFamily, rectangular encryption, fn_to_matrix, padding, branch compilation, AxolClient SDK
- [x] Phase 8: Chaos theory quantum module — Declare → Weave → Observe pipeline
- [x] Phase 8: Lyapunov exponent estimation (Benettin QR) + Omega = 1/(1+max(lambda,0))
- [x] Phase 8: Fractal dimension estimation (box-counting/correlation) + Phi = 1/(1+D/D_max)
- [x] Phase 8: Weaver, Observatory, Composition rules, Cost estimation, DSL parser
- [x] Phase 8: 101 new tests (545 total, 0 failed)
- [ ] Phase 9: Complex amplitudes (a+bi) for Shor, QPE, QFT — full phase interference
- [ ] Phase 10: Distributed tapestry weaving across multiple nodes
- [ ] Phase 11: Adaptive quality — dynamic Omega/Phi adjustment during observation

---

## License

MIT License. See [LICENSE](LICENSE) for details.
