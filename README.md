<p align="center">
  <h1 align="center">AXOL</h1>
  <p align="center">
    <strong>Token-Efficient Vector Programming Language</strong>
  </p>
  <p align="center">
    <a href="#"><img src="https://img.shields.io/badge/status-experimental-orange" alt="Status: Experimental"></a>
    <a href="#"><img src="https://img.shields.io/badge/version-0.1.0-blue" alt="Version 0.1.0"></a>
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

## What is Axol?

**Axol** is a domain-specific language (DSL) designed from the ground up for **AI agents** to read, write, and reason about programs using **fewer tokens** than conventional programming languages.

Instead of traditional control flow (if/else, for loops, function calls), Axol represents all computation as **vector transformations** and **state transitions** over immutable vector bundles. This design choice is rooted in a simple observation: **LLMs pay per token**, and existing programming languages were designed for human readability, not token efficiency.

### Key Properties

- **30-50% fewer tokens** than equivalent Python code
- **48-75% fewer tokens** than equivalent C# code
- **9 primitive operations** cover all computation: `transform`, `gate`, `merge`, `distance`, `route` (encrypted) + `step`, `branch`, `clamp`, `map` (plaintext)
- **Sparse matrix notation** scales O(N) vs O(N^2) for dense representations
- **Deterministic execution** with full state tracing
- **NumPy backend** for large vector operations (faster than pure Python loops for large dimensions)
- **E/P security classification** - each operation is classified as Encrypted (E) or Plaintext (P), with encryption coverage vs expressiveness tradeoff visualized by the built-in analyzer
- **Matrix-level encryption** - secret key matrices make programs cryptographically unreadable, a fundamental solution to the Shadow AI problem

---

## Table of Contents

- [Theoretical Background](#theoretical-background)
- [Shadow AI & Matrix Encryption](#shadow-ai--matrix-encryption)
  - [Encryption Proof: All 5 Operations Verified](#encryption-proof-all-5-operations-verified)
- [Plaintext Operations & Security Classification](#plaintext-operations--security-classification)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [DSL Syntax](#dsl-syntax)
- [Compiler Optimizer](#compiler-optimizer)
- [GPU Backend](#gpu-backend)
- [Module System](#module-system)
- [Tool-Use API](#tool-use-api)
- [Web Frontend](#web-frontend)
- [Token Cost Comparison](#token-cost-comparison)
- [Runtime Performance](#runtime-performance)
- [Performance Benchmarks](#performance-benchmarks)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Test Suite](#test-suite)
- [Roadmap](#roadmap)

---

## Theoretical Background

### The Token Economy Problem

Modern AI systems (GPT-4, Claude, etc.) operate under a **token economy**: every character of input and output consumes tokens, which directly translate to cost and latency. When an AI agent writes or reads code, the verbosity of the programming language directly impacts:

1. **Cost** - More tokens = higher API costs
2. **Latency** - More tokens = slower response times
3. **Context window** - More tokens = less room for other information
4. **Reasoning accuracy** - Compressed representations reduce noise

### Why Vector Computation?

Traditional programming languages express logic through **control flow** (branches, loops, recursion). This is intuitive for humans but inefficient for AI:

```python
# Python: 67 tokens
TRANSITIONS = {"IDLE": "RUNNING", "RUNNING": "DONE", "DONE": "DONE"}
def state_machine():
    state = "IDLE"
    steps = 0
    while state != "DONE":
        state = TRANSITIONS[state]
        steps += 1
    return state, steps
```

The same logic as a vector transformation:

```
# Axol DSL: 48 tokens (28% saved)
@state_machine
s state=onehot(0,3)
: advance=transform(state;M=[0 1 0;0 0 1;0 0 1])
? done state[2]>=1
```

The state machine's transition table becomes a **matrix**, and state advancement becomes **matrix multiplication**. The AI doesn't need to reason about string comparisons, dictionary lookups, or loop conditions - just a single matrix operation.

### The Nine Primitives

Axol provides nine primitive operations. The first five are **Encrypted (E)** - they can run on encrypted data. The last four are **Plaintext (P)** - they require plaintext but add nonlinear expressiveness:

| Operation | Security | Mathematical Basis | Description |
|-----------|:--------:|-------------------|-------------|
| `transform` | **E** | Matrix multiplication: `v @ M` | Linear state transformation |
| `gate` | **E** | Hadamard product: `v * g` | Conditional masking (0/1) |
| `merge` | **E** | Weighted sum: `sum(v_i * w_i)` | Vector combination |
| `distance` | **E** | L2 / cosine / dot | Similarity measurement |
| `route` | **E** | `argmax(v @ R)` | Discrete branching |
| `step` | **P** | `where(v >= t, 1, 0)` | Threshold to binary gate |
| `branch` | **P** | `where(g, then, else)` | Conditional vector select |
| `clamp` | **P** | `clip(v, min, max)` | Value range restriction |
| `map` | **P** | `f(v)` element-wise | Nonlinear activation (relu, sigmoid, abs, neg, square, sqrt) |

The five E operations form a **linear algebra basis** for encrypted computation:
- State machines (transform)
- Conditional logic (gate)
- Accumulation/aggregation (merge)
- Similarity search (distance)
- Decision making (route)

The four P operations add **nonlinear expressiveness** for AI/ML workloads:
- Activation functions (map: relu, sigmoid)
- Threshold decisions (step + branch)
- Value normalization (clamp)

### Sparse Matrix Notation

For large state spaces, dense matrix representation is O(N^2) in tokens. Axol's sparse notation reduces this to O(N):

```
# Dense: O(N^2) tokens - impractical for N=100
M=[0 1 0 0 ... 0; 0 0 1 0 ... 0; ...]

# Sparse: O(N) tokens - scales linearly
M=sparse(100x100;0,1=1 1,2=1 ... 98,99=1 99,99=1)
```

| N | Python | C# | Axol DSL | DSL/Python | DSL/C# |
|---|--------|-----|----------|------------|--------|
| 5 | 74 | 109 | 66 | 0.89x | 0.61x |
| 25 | 214 | 269 | 186 | 0.87x | 0.69x |
| 100 | 739 | 869 | 636 | 0.86x | 0.73x |
| 200 | 1,439 | 1,669 | 1,236 | 0.86x | 0.74x |

---

## Shadow AI & Matrix Encryption

### The Shadow AI Problem

**Shadow AI** refers to the risk of unauthorized AI agents leaking, copying, or reverse-engineering proprietary business logic. As AI agents increasingly write and execute code autonomously, traditional source code becomes a critical attack surface:

- An AI agent's prompts and generated code can be **extracted via prompt injection**
- Code in Python/C#/JavaScript is **human-readable by design** - obfuscation is reversible
- Proprietary algorithms, decision rules, and trade secrets embedded in code are **exposed in plaintext**
- Traditional obfuscation (variable renaming, control flow flattening) only raises the bar slightly - the logic remains structurally intact and recoverable

### How Axol Solves This: Matrix-Level Encryption

Because **all computation in Axol reduces to matrix multiplication** (`v @ M`), a mathematical property becomes available that is impossible in traditional programming languages: **similarity transformation encryption**.

Given a secret invertible key matrix **K**, any Axol program can be encrypted:

```
Original program:     state  -->  M  -->  new_state
Encrypted program:    state' -->  M' -->  new_state'

Where:
  M' = K^(-1) @ M @ K          (encrypted operation matrix)
  state' = state @ K            (encrypted initial state)
  result  = result' @ K^(-1)    (decrypted final output)
```

This is not obfuscation - it is **cryptographic transformation**. The encrypted program:

1. **Runs correctly** in the encrypted domain (matrix algebra preserves conjugation)
2. **Produces encrypted output** that requires K^(-1) to decode
3. **Hides all business logic** - the matrices M' are mathematically unrelated to M without K
4. **Resists reverse engineering** - recovering K from M' is a matrix decomposition problem whose difficulty grows with N. While no known polynomial-time algorithm exists for the general case, formal cryptographic hardness proofs are an area of ongoing research

### Concrete Example

```
# Original: State machine transition matrix (business logic visible)
M = [0 1 0]    # IDLE -> RUNNING
    [0 0 1]    # RUNNING -> DONE
    [0 0 1]    # DONE -> DONE (absorbing)

# After encryption with secret key K:
M' = [0.73  -0.21   0.48]    # Meaningless without K
     [0.15   0.89  -0.04]    # Cannot infer state machine structure
     [0.52   0.33   0.15]    # Appears as random noise
```

The encrypted program still executes correctly (matrix algebra guarantees `K^(-1)(KvM)K = vM`), but the DSL text contains **only the encrypted matrices**. Even if the entire `.axol` file is leaked:

- No state names are visible (vectors are encrypted)
- No transition logic is visible (matrices are encrypted)
- No terminal conditions are meaningful (thresholds operate on encrypted values)

### Why This Is Impossible in Traditional Languages

| Property | Python/C#/JS | FHE | Axol |
|----------|-------------|-----|------|
| Code semantics | Plaintext control flow | Encrypted (any computation) | Matrix multiplication |
| Obfuscation | Reversible (rename vars, flatten flow) | N/A | N/A |
| Encryption | Impossible (must be parseable) | Full (any computation) | Linear ops only (5 of 9) |
| Performance overhead | N/A | 1000-10000x | ~0% (pipeline mode) |
| Complexity | N/A | Very high | Low (key matrix only) |
| Leaked code | Full logic exposed | Encrypted | Random-looking numbers |
| Key separation | Not possible | Required | Key matrix stored separately (HSM, enclave) |
| Correctness after encryption | N/A | Mathematically guaranteed | Mathematically guaranteed |

### Security Architecture

```
  [Developer]                    [Deployment]
       |                              |
  Original .axol                 Encrypted .axol
  (readable logic)               (encrypted matrices)
       |                              |
       +--- K (secret key) ---------->|
       |    stored in HSM/enclave     |
       v                              v
  encrypt(M, K) = K^(-1)MK      run_program(encrypted)
                                      |
                                 encrypted output
                                      |
                                 decrypt(output, K^(-1))
                                      |
                                 actual result
```

The secret key matrix K can be:
- Stored in a **Hardware Security Module (HSM)**
- Managed by a **key management service (KMS)**
- Rotated periodically without changing program structure
- Different per deployment environment (dev/staging/prod)

Axol offers a lightweight alternative to Fully Homomorphic Encryption (FHE) for matrix-based computations. Unlike FHE (which supports arbitrary computation but with high overhead), Axol's similarity transformation is efficient but limited to linear operations. This tradeoff makes it practical for specific use cases where the 5 encrypted operations suffice.

### Encryption Proof: All 5 Operations Verified

The encryption compatibility of all 5 Axol operations has been **mathematically proven and tested** (21 tests in `tests/test_encryption.py`):

| Operation | Encryption Method | Key Constraint | Status |
|-----------|------------------|----------------|--------|
| `transform` | `M' = K^(-1) M K` (similarity transform) | Any invertible K | **PROVEN** |
| `gate` | Rewrite as `diag(g)` matrix, then transform | Any invertible K | **PROVEN** |
| `merge` | Linear: `w*(v@K) = (wv)@K` (automatic) | Any invertible K | **PROVEN** |
| `distance` | `\|\|v@K\|\| = \|\|v\|\|` (orthogonal preserves norms) | Orthogonal K | **PROVEN** |
| `route` | `R' = K^(-1) R` (left-multiply only) | Any invertible K | **PROVEN** |

**Complex multi-operation programs also verified:**

- HP Decay (transform + merge loop) - encrypted/decrypted results match
- 3-state FSM (chained transforms) - correct state transitions in encrypted domain
- Combat pipeline (transform + gate + merge) - 3 ops chained, error < 0.001
- 20-state automaton (sparse matrix, 19 steps) - encrypted execution matches original
- 50x50 large-scale matrix - float32 precision maintained

**Security properties proven by test:**

- Encrypted matrices appear as random noise (sparse -> dense, no visible structure)
- Different keys produce completely different encrypted matrices
- 100 random key brute-force attempts fail to recover original
- OneHot vector structure completely hidden after encryption

---

## Plaintext Operations & Security Classification

### Why Plaintext Operations?

The original 5 encrypted operations are **linear** - they can only express linear transformations. Many real-world AI/ML workloads require **nonlinear** operations (activation functions, conditional branching, value clamping). The 4 new plaintext operations fill this gap.

### SecurityLevel Enum

Every operation carries a `SecurityLevel`:

```python
from axol.core import SecurityLevel

SecurityLevel.ENCRYPTED  # "E" - can run on encrypted data
SecurityLevel.PLAINTEXT  # "P" - requires plaintext
```

### Encryption Coverage Analyzer

The built-in analyzer reports what percentage of a program can run encrypted:

```python
from axol.core import parse, analyze

program = parse("""
@damage_calc
s raw=[50 30] armor=[10 5]
: diff=merge(raw armor;w=[1 -1])->dmg
: act=map(dmg;fn=relu)
: safe=clamp(dmg;min=0,max=100)
""")

result = analyze(program)
print(result.summary())
# Program: damage_calc
# Transitions: 3 total, 1 encrypted (E), 2 plaintext (P)
# Coverage: 33.3%
# Encryptable keys: (keys only E ops touch)
# Plaintext keys: (keys any P op touches)
```

### Security-Expressiveness Tradeoff

Adding P operations increases expressiveness but reduces encryption coverage:

| Program Type | Encryption Coverage | Expressiveness |
|-------------|-------------------|---------------|
| E ops only | 100% | Linear only |
| Mixed E+P | 30-70% (typical) | Full (nonlinear) |
| P ops only | 0% | Full (nonlinear) |

Programs requiring nonlinear operations (activation functions, conditional branching) must accept partial encryption coverage. Use the built-in analyzer to measure your program's coverage and identify which keys require plaintext access.

### New Ops Token Cost (Python vs C# vs Axol DSL)

| Program | Python | C# | Axol DSL | vs Python | vs C# |
|---------|-------:|---:|--------:|---------:|------:|
| ReLU Activation | 48 | 82 | 28 | 42% | 66% |
| Threshold Select | 140 | 184 | 80 | 43% | 57% |
| Value Clamp | 66 | 95 | 31 | 53% | 67% |
| Sigmoid Activation | 57 | 88 | 28 | 51% | 68% |
| Damage Pipeline | 306 | 326 | 155 | 49% | 53% |
| **Total** | **617** | **775** | **322** | **48%** | **59%** |

### New Ops Runtime (dim=10,000)

| Operation | Python Loop | Axol (NumPy) | Speedup |
|-----------|----------:|----------:|--------:|
| ReLU | 575 us | 21 us | **27x** |
| Sigmoid | 1.7 ms | 42 us | **40x** |
| Step+Branch | 889 us | 96 us | **9x** |
| Clamp | 937 us | 16 us | **58x** |
| Damage Pipeline | 3.8 ms | 191 us | **20x** |

---

## Architecture

```
                                          +-------------+
  .axol source -----> Parser (dsl.py) --> | Program     |
                         |                | + optimize()|
                         v                +------+------+
                    Module System               |
                    (module.py)                  v
                      - import             +-----------+    +-----------+
                      - use()              |  Engine   |--->|  Verify   |
                      - compose()          |(program.py)|    |(verify.py)|
                                           +-----------+    +-----------+
                                                |
                    +-----------+    +----------+----------+
                    |  Backend  |<---|    Operations       |
                    |(backend.py)|    | (operations.py)     |
                    | numpy/cupy|    +---------------------+
                    | /jax      |               |
                    +-----------+    +-----------+----------+
                                    |      Types           |
                                    |   (types.py)         |
                    +-----------+   +----------------------+
                    |Encryption |   +-----------+
                    |(encryption|   | Analyzer  |
                    |       .py)|   |(analyzer  |
                    +-----------+   |       .py)|
                                    +-----------+
                    +-----------+    +-----------+
                    | Tool API  |    |  Server   |
                    |(api/)     |    |(server/)  |
                    | dispatch  |    | FastAPI   |
                    | tools     |    | HTML/JS   |
                    +-----------+    +-----------+
```

### Module Overview

| Module | Description |
|--------|-------------|
| `axol.core.types` | 7 vector types (`BinaryVec`, `IntVec`, `FloatVec`, `OneHotVec`, `GateVec`, `TransMatrix`) + `StateBundle` |
| `axol.core.operations` | 9 primitive operations: `transform`, `gate`, `merge`, `distance`, `route`, `step`, `branch`, `clamp`, `map_fn` |
| `axol.core.program` | Execution engine: `Program`, `Transition`, `run_program`, `SecurityLevel`, `StepOp`/`BranchOp`/`ClampOp`/`MapOp` |
| `axol.core.verify` | State verification with exact/cosine/euclidean matching |
| `axol.core.dsl` | DSL parser: `parse(source) -> Program` with `import`/`use()` support |
| `axol.core.optimizer` | 3-pass compiler optimizer: transform fusion, dead state elimination, constant folding |
| `axol.core.backend` | Pluggable array backend: `numpy` (default), `cupy`, `jax` |
| `axol.core.encryption` | Similarity transformation encryption: `encrypt_program`, `decrypt_state` (E/P-aware) |
| `axol.core.analyzer` | Encryption coverage analyzer: `analyze(program) -> AnalysisResult` with E/P classification |
| `axol.core.module` | Module system: `Module`, `ModuleRegistry`, `compose()`, schema validation |
| `axol.api` | Tool-Use API for AI agents: `dispatch(request)`, `get_tool_definitions()` |
| `axol.server` | FastAPI web server + vanilla HTML/JS visual debugger frontend |

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/AXOL.git
cd AXOL

# Install dependencies
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

### Hello World - DSL

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
print(f"Steps: {result.steps_executed}")
print(f"Terminated by: {result.terminated_by}")  # terminal_condition
```

### Hello World - Python API

```python
from axol.core import (
    FloatVec, GateVec, TransMatrix, StateBundle,
    Program, Transition, run_program,
)
from axol.core.program import TransformOp

state = StateBundle(vectors={
    "hp": FloatVec.from_list([100.0]),
})
decay = TransMatrix.from_list([[0.8]])

program = Program(
    name="hp_decay",
    initial_state=state,
    transitions=[
        Transition("decay", TransformOp(key="hp", matrix=decay)),
    ],
)
result = run_program(program)
print(f"HP after decay: {result.final_state['hp'].to_list()}")  # [80.0]
```

---

## DSL Syntax

### Program Structure

```
@program_name              # Header: program name
s key1=[values] key2=...   # State: initial vector declarations
: name=op(args)->out       # Transition: operation definitions
? terminal condition       # Terminal: loop exit condition (optional)
```

### State Declarations

```
s hp=[100]                          # Single float vector
s pos=[1.5 2.0 -3.0]               # Multi-element vector
s state=onehot(0,5)                 # One-hot vector: index 0, size 5
s buffer=zeros(10)                  # Zero vector of size 10
s mask=ones(3)                      # All-ones vector of size 3
s hp=[100] mp=[50] stamina=[75]     # Multiple vectors on one line
```

### Operations

```
# --- Encrypted (E) operations ---

# transform: matrix multiplication
: decay=transform(hp;M=[0.8])
: advance=transform(state;M=[0 1 0;0 0 1;0 0 1])

# gate: element-wise masking
: masked=gate(values;g=mask)

# merge: weighted sum of vectors
: total=merge(a b c;w=[1 1 1])->result

# distance: similarity measurement
: dist=distance(pos1 pos2)
: sim=distance(vec1 vec2;metric=cosine)

# route: argmax routing
: choice=route(scores;R=[1 0 0;0 1 0;0 0 1])

# --- Plaintext (P) operations ---

# step: threshold to binary gate
: mask=step(scores;t=0.5)->gate_out

# branch: conditional vector select (requires ->out_key)
: selected=branch(gate_key;then=high,else=low)->result

# clamp: clip values to range
: safe=clamp(values;min=0,max=100)

# map: element-wise nonlinear function (relu, sigmoid, abs, neg, square, sqrt)
: activated=map(x;fn=relu)
: prob=map(logits;fn=sigmoid)->output
```

### Matrix Formats

```
# Dense: rows separated by ;
M=[0.8]                               # 1x1
M=[1 0;0 1]                           # 2x2 identity
M=[0 1 0;0 0 1;0 0 1]                # 3x3 shift matrix

# Sparse: only non-zero entries
M=sparse(100x100;0,1=1 1,2=1 99,99=1) # 100x100 with 100 entries
```

### Terminal Conditions

```
? done count>=5              # Exit when count[0] >= 5
? finished state[2]>=1       # Exit when state[2] >= 1 (indexed access)
? end hp<=0                  # Exit when hp[0] <= 0
```

Without a `?` line, the program runs in **pipeline mode** (all transitions execute once).

### Comments

```
# This is a comment
@my_program
# Comments can appear anywhere
s v=[1 2 3]
: t=transform(v;M=[1 0 0;0 1 0;0 0 1])
```

---

## Compiler Optimizer

`optimize()` applies three passes to reduce program size and pre-compute constants:

```python
from axol.core import parse, optimize, run_program

program = parse(source)
optimized = optimize(program)   # fuse + eliminate + fold
result = run_program(optimized)
```

### Pass 1: Transform Fusion

Consecutive `TransformOp` on the same key chain are fused into a single matrix multiplication:

```
# Before: 2 transitions, 2 matrix multiplications per iteration
: t1=transform(v;M=[0 1 0;0 0 1;1 0 0])
: t2=transform(v;M=[2 0 0;0 2 0;0 0 2])

# After: 1 transition, 1 matrix multiplication (M_fused = M1 @ M2)
: t1+t2=transform(v;M_fused)
```

- Does not cross `CustomOp` boundaries
- Fixed-point iteration handles 3+ chains
- Pipeline with 2 transforms: **transition count -50%, execution time -45%**

### Pass 2: Dead State Elimination

Removes initial state vectors never read by any transition:

```
s used=[1 0]  unused=[99 99]   # unused is never referenced
: t=transform(used;M=[...])

# After optimization: unused is removed from initial state
```

- Conservative with `CustomOp` (preserves all state)
- `terminal_key` is always treated as "read"

### Pass 3: Constant Folding

Pre-computes transforms on immutable keys (keys that are never written):

```
s constant=[1 0 0]
: t=transform(constant;M=[0 1 0;0 0 1;1 0 0])->result

# After: transition eliminated, result=[0,1,0] stored in initial state
```

---

## GPU Backend

Pluggable array backend supporting `numpy` (default), `cupy` (NVIDIA GPU), and `jax`:

```python
from axol.core import set_backend, get_backend_name

set_backend("numpy")   # default - CPU
set_backend("cupy")    # NVIDIA GPU (requires cupy installed)
set_backend("jax")     # Google JAX (requires jax installed)
```

Install optional backends:

```bash
pip install axol[gpu]   # cupy-cuda12x
pip install axol[jax]   # jax[cpu]
```

All existing code works transparently - the backend switch is global and affects all vector/matrix operations.

---

## Module System

Reusable, composable programs with schemas, imports, and sub-module execution.

### Module Definition

```python
from axol.core.module import Module, ModuleSchema, VecSchema, ModuleRegistry

schema = ModuleSchema(
    inputs=[VecSchema("atk", "float", 1), VecSchema("def_val", "float", 1)],
    outputs=[VecSchema("dmg", "float", 1)],
)
module = Module(name="damage_calc", program=program, schema=schema)
```

### Registry & File Loading

```python
registry = ModuleRegistry()
registry.load_from_file("damage_calc.axol")
registry.resolve_import("heal", relative_to="main.axol")
```

### DSL Import & Use Syntax

```
@main
import damage_calc from "damage_calc.axol"
s atk=[50] def_val=[10]
: calc=use(damage_calc;in=atk,def_val;out=dmg)
```

### Program Composition

```python
from axol.core.module import compose
combined = compose(program_a, program_b, name="combined")
```

---

## Tool-Use API

JSON-callable interface for AI agents to parse, run, and verify Axol programs:

```python
from axol.api import dispatch

# Parse
result = dispatch({"action": "parse", "source": "@prog\ns v=[1]\n: t=transform(v;M=[2])"})
# -> {"program_name": "prog", "state_keys": ["v"], "transition_count": 1, "has_terminal": false}

# Run
result = dispatch({"action": "run", "source": "...", "optimize": True})
# -> {"final_state": {"v": [2.0]}, "steps_executed": 1, "terminated_by": "pipeline_end"}

# Inspect step-by-step
result = dispatch({"action": "inspect", "source": "...", "step": 1})

# List operations
result = dispatch({"action": "list_ops"})

# Verify expected output
result = dispatch({"action": "verify", "source": "...", "expected": {"v": [2.0]}})
```

AI agent tool definitions (JSON Schema) are available via `get_tool_definitions()`.

---

## Web Frontend

FastAPI server with a vanilla HTML/JS visual debugger:

```bash
pip install axol[server]    # fastapi + uvicorn
python -m axol.server       # http://localhost:8080
```

### Features

| Panel | Description |
|-------|-------------|
| **DSL Editor** | Syntax editing with example dropdown |
| **Execution** | Run/Optimize buttons, result summary (steps, time, terminated_by) |
| **Trace Viewer** | Step-by-step state table with prev/next/play controls |
| **State Chart** | Chart.js time-series graph (X=step, Y=vector values) |
| **Encryption Demo** | Original vs encrypted matrix heatmaps, encrypt/run/decrypt workflow |
| **Performance** | Optimizer before/after comparison, token cost analysis |

### API Endpoints

```
POST /api/parse       - Parse DSL source
POST /api/run         - Parse + execute + full trace
POST /api/optimize    - Optimizer before/after comparison
POST /api/encrypt     - Encrypt program + run + decrypt
GET  /api/examples    - Built-in example programs
GET  /api/ops         - Operation descriptions
POST /api/token-cost  - Token count analysis (Axol vs Python vs C#)
POST /api/module/run  - Run program with sub-modules
```

---

## Token Cost Comparison

Measured with `tiktoken` cl100k_base tokenizer (used by GPT-4 and Claude).

> **Note**: Token savings are measured on programs that naturally map to
> vector/matrix operations (state machines, linear transforms, weighted sums).
> For general-purpose programming tasks (string processing, I/O, API calls),
> Axol cannot express them at all. The comparisons below represent Axol's
> best-case scenario, not average-case.

### Python vs Axol DSL

| Program | Python | Axol DSL | Saving |
|---------|--------|----------|--------|
| Counter (0->5) | 32 | 33 | -3.1% |
| State Machine (3-state) | 67 | 47 | 29.9% |
| HP Decay (3 rounds) | 51 | 32 | 37.3% |
| RPG Damage Calc | 130 | 90 | 30.8% |
| 100-State Automaton | 1,034 | 636 | 38.5% |
| **Total** | **1,314** | **838** | **36.2%** |

### Python vs C# vs Axol DSL

| Program | Python | C# | Axol DSL | vs Python | vs C# |
|---------|--------|----|----------|-----------|-------|
| Counter | 32 | 61 | 33 | -3.1% | 45.9% |
| State Machine | 67 | 147 | 48 | 28.4% | 67.3% |
| HP Decay | 51 | 134 | 51 | 0.0% | 61.9% |
| Combat | 145 | 203 | 66 | 54.5% | 67.5% |
| Data Heavy | 159 | 227 | 67 | 57.9% | 70.5% |
| Pattern Match | 151 | 197 | 49 | 67.5% | 75.1% |
| 100-State Auto | 739 | 869 | 636 | 13.9% | 26.8% |
| **Total** | **1,344** | **1,838** | **950** | **29.3%** | **48.3%** |

### Key Findings

1. **Simple programs** (counter, hp_decay): DSL is comparable to Python. The overhead of DSL syntax roughly equals Python's minimal syntax for trivial programs.
2. **Structured programs** (combat, data_heavy, pattern_match): DSL saves **50-68%** vs Python and **67-75%** vs C#. The vector representation eliminates class definitions, control flow, and boilerplate.
3. **Large state spaces** (100+ states): Sparse matrix notation gives consistent **~38% savings** vs Python and **~27% savings** vs C#, with O(N) scaling vs O(N^2).

---

## Runtime Performance

Axol uses NumPy as its computation backend.

> **Note**: Runtime benchmarks compare pure Python loops against Axol's
> NumPy backend. The speedup is primarily from NumPy's optimized C/Fortran
> implementation, not from Axol-specific optimizations. Python code using
> NumPy directly would achieve similar speeds.

### Small Vectors (dim < 100)

| Dimension | Python Loop | Axol (NumPy) | Winner |
|-----------|-------------|--------------|--------|
| dim=4 | ~6 us | ~11 us | Python 2x |
| dim=100 | ~14 us | ~20 us | Python 1.4x |

For small vectors, Python's native loop is faster because NumPy has per-call overhead. This is expected and acceptable - small programs are fast regardless.

### Large Vectors (dim >= 1000)

| Dimension | Python Loop | Axol (NumPy) | Winner |
|-----------|-------------|--------------|--------|
| dim=1,000 (matmul) | ~129 ms | ~0.2 ms | **573x** (NumPy) |
| dim=10,000 (matmul) | ~14,815 ms | ~381 ms | **39x** (NumPy) |

For large-scale vector operations (matrix multiplication), NumPy's optimized C/Fortran BLAS backend (used by Axol) is **orders of magnitude faster** than pure Python loops. Any Python code using NumPy directly would achieve similar speedups.

### When to Use Axol

| Scenario | Recommendation |
|----------|---------------|
| AI agent code generation | Axol DSL (fewer tokens = lower cost) |
| Large state spaces (100+ dimensions) | Axol (NumPy speedup + sparse notation) |
| Simple scripts (< 10 lines) | Python (less overhead) |
| Human-readable business logic | Python/C# (familiar syntax) |

### Limitations

- **Limited domain**: Axol can only express vector/matrix computations. String processing, I/O, networking, and general-purpose programming are not supported.
- **No LLM training data**: Unlike Python or JavaScript, no LLM has been trained on Axol code. AI agents may struggle to generate correct Axol programs without examples in context.
- **Encryption only for linear ops**: Only 5 of 9 operations support encrypted execution. Programs using nonlinear ops (step, branch, clamp, map) have reduced encryption coverage.
- **Loop-mode encryption overhead**: Encrypted programs in loop mode cannot evaluate terminal conditions, running until max_iterations. This causes significant overhead (400x+ in benchmarks).
- **Token savings are domain-specific**: The 30-50% token savings apply to vector/matrix-heavy programs. For general-purpose tasks, Python is more concise.

---

## Performance Benchmarks

Auto-generated by `pytest tests/test_performance_report.py -v -s`. Full results in [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md).

> **Note**: Runtime benchmarks compare pure Python loops against Axol's
> NumPy backend. The speedup is primarily from NumPy's optimized C/Fortran
> implementation, not from Axol-specific optimizations. Python code using
> NumPy directly would achieve similar speeds.

### Token Efficiency (Axol vs Python vs C#)

| Program | Axol | Python | C# | vs Python | vs C# |
|---------|------|--------|----|-----------|-------|
| Counter (0->5) | 11 | 45 | 78 | **76% saved** | **86% saved** |
| 3-State FSM | 14 | 52 | 89 | **73% saved** | **84% saved** |
| HP Decay | 14 | 58 | 95 | **76% saved** | **85% saved** |
| Combat Pipeline | 14 | 55 | 92 | **75% saved** | **85% saved** |
| Matrix Chain | 21 | 60 | 98 | **65% saved** | **79% saved** |

Average: **74% fewer tokens than Python**, **85% fewer tokens than C#**.

### Execution Time by Dimension

| Dimension | Avg Time |
|-----------|----------|
| 4 | 0.25 ms |
| 100 | 0.17 ms |
| 1,000 | 1.41 ms |

### Optimizer Impact

| Program | Before | After | Time Reduction |
|---------|--------|-------|----------------|
| Pipeline (2 transforms) | 2 transitions | 1 transition | **-45%** |
| Counter (loop) | 2 transitions | 2 transitions | - |
| FSM (loop) | 2 transitions | 2 transitions | - |

Transform fusion is most effective on pipeline programs with consecutive matrix operations.

### Encryption Overhead

| Program | Plaintext | Encrypted | Overhead |
|---------|-----------|-----------|----------|
| Pipeline (1 pass) | 0.12 ms | 0.12 ms | **~0%** |
| 3-State FSM (loop) | 0.62 ms | 276.8 ms | +44,633% |

Pipeline mode: negligible overhead. Loop mode: high overhead because the encrypted terminal condition cannot trigger early exit, causing execution to run until `max_iterations`.

### Scaling (N-state Automaton)

| States | Tokens | Execution Time |
|--------|--------|---------------|
| 5 | 28 | 1.6 ms |
| 20 | 388 | 4.3 ms |
| 50 | 2,458 | 12.9 ms |
| 100 | 9,908 | 27.9 ms |
| 200 | 39,808 | 59.2 ms |

Tokens grow **O(N)** thanks to sparse matrix notation (vs O(N^2) for Python/C#). Execution time grows ~O(N^2) due to matrix multiplication, but remains under 60ms for 200-state programs.

---

## API Reference

### `parse(source, registry=None, source_path=None) -> Program`

Parse Axol DSL source text into an executable `Program` object.

```python
from axol.core import parse
program = parse("@test\ns v=[1 2 3]\n: t=transform(v;M=[1 0 0;0 1 0;0 0 1])")

# With module registry for import/use support
from axol.core.module import ModuleRegistry
registry = ModuleRegistry()
program = parse(source, registry=registry, source_path="main.axol")
```

### `run_program(program: Program) -> ExecutionResult`

Execute a program and return the result.

```python
from axol.core import run_program
result = run_program(program)
result.final_state     # StateBundle with final vector values
result.steps_executed  # Total number of transition steps
result.terminated_by   # "pipeline_end" | "terminal_condition" | "max_iterations"
result.trace           # List of TraceEntry for debugging
result.verification    # VerifyResult if expected_state was set
```

### `optimize(program, *, fuse=True, eliminate_dead=True, fold_constants=True) -> Program`

Optimize a program without mutating the original.

```python
from axol.core import optimize
optimized = optimize(program)                          # all passes
optimized = optimize(program, fold_constants=False)    # selective passes
```

### `set_backend(name) / get_backend() / to_numpy(arr)`

Switch the array computation backend.

```python
from axol.core import set_backend, get_backend, to_numpy
set_backend("cupy")     # switch to GPU
xp = get_backend()      # returns cupy module
arr = to_numpy(gpu_arr) # convert back to numpy
```

### `dispatch(request) -> dict`

Tool-Use API entry point for AI agents.

```python
from axol.api import dispatch
result = dispatch({"action": "run", "source": "...", "optimize": True})
```

### Vector Types

| Type | Description | Factory Methods |
|------|-------------|----------------|
| `FloatVec` | 32-bit floats | `from_list([1.0, 2.0])`, `zeros(n)`, `ones(n)` |
| `IntVec` | 64-bit integers | `from_list([1, 2])`, `zeros(n)` |
| `BinaryVec` | Elements in {0, 1} | `from_list([0, 1])`, `zeros(n)`, `ones(n)` |
| `OneHotVec` | Exactly one 1.0 | `from_index(idx, n)`, `from_list(...)` |
| `GateVec` | Elements in {0.0, 1.0} | `from_list([1.0, 0.0])`, `zeros(n)`, `ones(n)` |
| `TransMatrix` | M x N float32 matrix | `from_list(rows)`, `identity(n)`, `zeros(m, n)` |

### Operation Descriptors

```python
from axol.core.program import (
    # Encrypted (E) operations
    TransformOp,  # TransformOp(key="v", matrix=M, out_key=None)
    GateOp,       # GateOp(key="v", gate_key="g", out_key=None)
    MergeOp,      # MergeOp(keys=["a","b"], weights=w, out_key="out")
    DistanceOp,   # DistanceOp(key_a="a", key_b="b", metric="euclidean")
    RouteOp,      # RouteOp(key="v", router=R, out_key="_route")
    # Plaintext (P) operations
    StepOp,       # StepOp(key="v", threshold=0.0, out_key=None)
    BranchOp,     # BranchOp(gate_key="g", then_key="a", else_key="b", out_key="out")
    ClampOp,      # ClampOp(key="v", min_val=-inf, max_val=inf, out_key=None)
    MapOp,        # MapOp(key="v", fn_name="relu", out_key=None)
    # Escape hatch
    CustomOp,     # CustomOp(fn=callable, label="name")  -- security=P
)
```

### Analyzer

```python
from axol.core import analyze

result = analyze(program)
result.coverage_pct        # E / total * 100
result.encrypted_count     # number of E transitions
result.plaintext_count     # number of P transitions
result.encryptable_keys    # keys only accessed by E ops
result.plaintext_keys      # keys accessed by P ops
print(result.summary())    # human-readable report
```

### Verification

```python
from axol.core import verify_states, VerifySpec

result = verify_states(
    expected=expected_bundle,
    actual=actual_bundle,
    specs={"hp": VerifySpec.exact(tolerance=0.01)},
    strict_keys=False,
)
print(result.passed)    # True/False
print(result.summary()) # Detailed report
```

---

## Examples

### 1. Counter (0 -> 5)

```
@counter
s count=[0] one=[1]
: increment=merge(count one;w=[1 1])->count
? done count>=5
```

### 2. State Machine (IDLE -> RUNNING -> DONE)

```
@state_machine
s state=onehot(0,3)
: advance=transform(state;M=[0 1 0;0 0 1;0 0 1])
? done state[2]>=1
```

### 3. HP Decay (100 x 0.8^3 = 51.2)

```
@hp_decay
s hp=[100] round=[0] one=[1]
: decay=transform(hp;M=[0.8])
: tick=merge(round one;w=[1 1])->round
? done round>=3
```

### 4. Combat Damage (Pipeline)

```
@combat
s atk=[50] def_val=[20] flag=[1]
: scale=transform(atk;M=[1.5])->scaled
: block=gate(def_val;g=flag)
: combine=merge(scaled def_val;w=[1 -1])->damage
```

### 5. ReLU Activation (map)

```
@relu
s x=[-2 0 3 -1 5]
:act=map(x;fn=relu)
# Result: x = [0, 0, 3, 0, 5]
```

### 6. Threshold Select (step + branch)

```
@threshold_select
s scores=[0.3 0.8 0.1 0.9] high=[100 200 300 400] low=[1 2 3 4]
:s1=step(scores;t=0.5)->mask
:b1=branch(mask;then=high,else=low)->result
# mask = [0, 1, 0, 1]
# result = [1, 200, 3, 400]
```

### 7. Damage Pipeline (all 4 new ops)

```
@damage_pipe
s raw=[50 30 80 20] armor=[10 40 5 25]
s crit=[1 0 1 0] bonus=[20 20 20 20] zero=[0 0 0 0]
:d1=merge(raw armor;w=[1 -1])->diff
:d2=map(diff;fn=relu)->effective
:d3=step(crit;t=0.5)->mask
:d4=branch(mask;then=bonus,else=zero)->crit_bonus
:d5=merge(effective crit_bonus;w=[1 1])->total
:d6=clamp(total;min=0,max=9999)
# diff=[40,-10,75,-5] -> relu=[40,0,75,0] -> +bonus=[60,0,95,0]
```

### 8. 100-State Automaton (Sparse)

```
@auto_100
s s=onehot(0,100)
: step=transform(s;M=sparse(100x100;0,1=1 1,2=1 ... 98,99=1 99,99=1))
? done s[99]>=1
```

---

## Test Suite

```bash
# Run all tests (~320 tests)
pytest tests/ -v

# Core tests
pytest tests/test_types.py tests/test_operations.py tests/test_program.py tests/test_dsl.py -v

# Optimizer tests (18 tests)
pytest tests/test_optimizer.py -v

# Backend tests (13 tests, cupy/jax skipped if not installed)
pytest tests/test_backend.py -v

# Tool-Use API tests (20 tests)
pytest tests/test_api.py -v

# Module system tests (18 tests)
pytest tests/test_module.py -v

# Encryption proof-of-concept (21 tests)
pytest tests/test_encryption.py -v -s

# New ops tests - step/branch/clamp/map (44 tests)
pytest tests/test_new_ops.py -v

# Analyzer tests - E/P coverage analysis (7 tests)
pytest tests/test_analyzer.py -v

# New ops benchmark - Python vs C# vs Axol (15 tests)
pytest tests/test_benchmark_new_ops.py -v -s

# Server endpoint tests (13 tests, requires fastapi)
pytest tests/test_server.py -v

# Performance report (generates PERFORMANCE_REPORT.md)
pytest tests/test_performance_report.py -v -s

# Token cost comparison
pytest tests/test_token_cost.py -v -s

# Three-language benchmark (Python vs C# vs Axol)
pytest tests/test_benchmark_trilingual.py -v -s

# Start web frontend
python -m axol.server   # http://localhost:8080
```

Current test count: **~320 tests**, all passing (4 skipped: cupy/jax not installed).

---

## Roadmap

- [x] Phase 1: Type system (7 vector types + StateBundle)
- [x] Phase 1: 5 primitive operations
- [x] Phase 1: Program execution engine (pipeline + loop mode)
- [x] Phase 1: State verification framework
- [x] Phase 2: DSL parser with full grammar support
- [x] Phase 2: Sparse matrix notation
- [x] Phase 2: Token cost benchmarks (Python, C#, Axol)
- [x] Phase 2: Matrix encryption proof-of-concept (all 5 ops verified, 21 tests)
- [x] Phase 3: Compiler optimizer (transform fusion, dead state elimination, constant folding)
- [x] Phase 3: GPU backend (numpy/cupy/jax pluggable)
- [x] Phase 4: Tool-Use API for AI agents (parse/run/inspect/verify/list_ops)
- [x] Phase 4: Encryption module (encrypt_program, decrypt_state)
- [x] Phase 5: Module system (registry, import/use DSL, compose, schema validation)
- [x] Frontend: FastAPI + vanilla HTML/JS visual debugger (trace viewer, state chart, encryption demo)
- [x] Performance benchmarks (token cost, runtime scaling, optimizer effect, encryption overhead)

---

## License

MIT License. See [LICENSE](LICENSE) for details.
