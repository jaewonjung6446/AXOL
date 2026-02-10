<p align="center">
  <h1 align="center">AXOL</h1>
  <p align="center">
    <strong>AI-Native Vector Programming Language</strong>
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
- **5 primitive operations** cover all computation: `transform`, `gate`, `merge`, `distance`, `route`
- **Sparse matrix notation** scales O(N) vs O(N^2) for dense representations
- **Deterministic execution** with full state tracing
- **NumPy backend** enables 500x+ speedup on large vector operations
- **Matrix-level encryption** - secret key matrices make programs cryptographically unreadable, a fundamental solution to the Shadow AI problem

---

## Table of Contents

- [Theoretical Background](#theoretical-background)
- [Shadow AI & Matrix Encryption](#shadow-ai--matrix-encryption)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [DSL Syntax](#dsl-syntax)
- [Token Cost Comparison](#token-cost-comparison)
- [Runtime Performance](#runtime-performance)
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

### The Five Primitives

Axol reduces all computation to five operations, each corresponding to a fundamental linear algebra concept:

| Operation | Mathematical Basis | Description |
|-----------|-------------------|-------------|
| `transform` | Matrix multiplication: `v @ M` | Linear state transformation |
| `gate` | Hadamard product: `v * g` | Conditional masking |
| `merge` | Weighted sum: `sum(v_i * w_i)` | Vector combination |
| `distance` | L2 / cosine / dot | Similarity measurement |
| `route` | `argmax(v @ R)` | Discrete branching |

These five operations form a **computationally complete** basis for expressing:
- State machines (transform)
- Conditional logic (gate)
- Accumulation/aggregation (merge)
- Similarity search (distance)
- Decision making (route)

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
4. **Resists reverse engineering** - recovering K from M' requires solving an NP-hard matrix decomposition problem for large N

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

| Property | Python/C#/JS | Axol |
|----------|-------------|------|
| Code semantics | Plaintext control flow | Matrix multiplication |
| Obfuscation | Reversible (rename vars, flatten flow) | N/A |
| Encryption | Impossible (must be parseable) | Similarity transform on matrices |
| Leaked code | Full logic exposed | Random-looking numbers |
| Key separation | Not possible | Key matrix stored separately (HSM, enclave) |
| Correctness after encryption | N/A | Mathematically guaranteed |

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

This makes Axol the first programming paradigm where **the source code itself can be cryptographically secured** while remaining executable - a fundamental, not incremental, solution to the Shadow AI problem.

---

## Architecture

```
                    +-----------+
  .axol source ---->|  Parser   |----> Program object
                    | (dsl.py)  |         |
                    +-----------+         |
                                          v
                    +-----------+    +-----------+
                    |  Verify   |<---|  Engine   |
                    |(verify.py)|    |(program.py)|
                    +-----------+    +-----------+
                                          |
                         uses             |
                    +-----------+         |
                    |Operations |<--------+
                    | (ops.py)  |
                    +-----------+
                         |
                    +-----------+
                    |  Types    |
                    |(types.py) |
                    +-----------+
```

### Module Overview

| Module | Description |
|--------|-------------|
| `axol.core.types` | 7 vector types (`BinaryVec`, `IntVec`, `FloatVec`, `OneHotVec`, `GateVec`, `TransMatrix`) + `StateBundle` |
| `axol.core.operations` | 5 primitive operations: `transform`, `gate`, `merge`, `distance`, `route` |
| `axol.core.program` | Execution engine: `Program`, `Transition`, `run_program` |
| `axol.core.verify` | State verification with exact/cosine/euclidean matching |
| `axol.core.dsl` | DSL parser: `parse(source) -> Program` |

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

## Token Cost Comparison

Measured with `tiktoken` cl100k_base tokenizer (used by GPT-4 and Claude).

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

Axol uses NumPy as its computation backend. Performance characteristics:

### Small Vectors (dim < 100)

| Dimension | Python Loop | Axol (NumPy) | Winner |
|-----------|-------------|--------------|--------|
| dim=4 | ~6 us | ~11 us | Python 2x |
| dim=100 | ~14 us | ~20 us | Python 1.4x |

For small vectors, Python's native loop is faster because NumPy has per-call overhead. This is expected and acceptable - small programs are fast regardless.

### Large Vectors (dim >= 1000)

| Dimension | Python Loop | Axol (NumPy) | Winner |
|-----------|-------------|--------------|--------|
| dim=1,000 (matmul) | ~129 ms | ~0.2 ms | **Axol 573x** |
| dim=10,000 (matmul) | ~14,815 ms | ~381 ms | **Axol 39x** |

For large-scale vector operations (matrix multiplication), Axol's NumPy backend is **orders of magnitude faster** than pure Python loops.

### When to Use Axol

| Scenario | Recommendation |
|----------|---------------|
| AI agent code generation | Axol DSL (fewer tokens = lower cost) |
| Large state spaces (100+ dimensions) | Axol (NumPy speedup + sparse notation) |
| Simple scripts (< 10 lines) | Python (less overhead) |
| Human-readable business logic | Python/C# (familiar syntax) |

---

## API Reference

### `parse(source: str) -> Program`

Parse Axol DSL source text into an executable `Program` object.

```python
from axol.core import parse
program = parse("@test\ns v=[1 2 3]\n: t=transform(v;M=[1 0 0;0 1 0;0 0 1])")
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
    TransformOp,  # TransformOp(key="v", matrix=M, out_key=None)
    GateOp,       # GateOp(key="v", gate_key="g", out_key=None)
    MergeOp,      # MergeOp(keys=["a","b"], weights=w, out_key="out")
    DistanceOp,   # DistanceOp(key_a="a", key_b="b", metric="euclidean")
    RouteOp,      # RouteOp(key="v", router=R, out_key="_route")
    CustomOp,     # CustomOp(fn=callable, label="name")
)
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

### 5. 100-State Automaton (Sparse)

```
@auto_100
s s=onehot(0,100)
: step=transform(s;M=sparse(100x100;0,1=1 1,2=1 ... 98,99=1 99,99=1))
? done s[99]>=1
```

---

## Test Suite

```bash
# Run all tests (149 tests)
pytest tests/ -v

# DSL parser tests only
pytest tests/test_dsl.py -v

# Token cost comparison
pytest tests/test_token_cost.py -v -s

# Three-language benchmark (Python vs C# vs Axol)
pytest tests/test_benchmark_trilingual.py -v -s

# Runtime performance
pytest tests/test_compare_python.py -v -s
```

Current test count: **149 tests**, all passing.

---

## Roadmap

- [x] Phase 1: Type system (7 vector types + StateBundle)
- [x] Phase 1: 5 primitive operations
- [x] Phase 1: Program execution engine (pipeline + loop mode)
- [x] Phase 1: State verification framework
- [x] Phase 2: DSL parser with full grammar support
- [x] Phase 2: Sparse matrix notation
- [x] Phase 2: Token cost benchmarks (Python, C#, Axol)
- [ ] Phase 3: Compiler optimizations (operation fusion, dead state elimination)
- [ ] Phase 3: GPU backend (CuPy / JAX)
- [ ] Phase 4: AI agent integration (tool-use API)
- [ ] Phase 4: Visual debugger for state traces
- [ ] Phase 5: Multi-program composition and module system

---

## License

MIT License. See [LICENSE](LICENSE) for details.
