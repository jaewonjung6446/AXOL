# AXOL

**The Collapse-Based Programming Language**

AXOL is a programming language where computation is **observation**. You don't calculate answers — you weave a structure of possibilities and then observe it, choosing how much to know and how much to lose.

```
declare "mood" {
    input text(8)
    input context(8)
    relate sentiment <- text, context via <~>
    output sentiment
    quality omega=0.85 phi=0.8
}

weave mood quantum=true seed=42

# See all possibilities (cost: nothing)
gaze mood { text = [...] context = [...] }

# See one answer (cost: everything else)
observe mood { text = [...] context = [...] }
```

**Observation cost: O(1). Always. Regardless of model size or data.**

---

## Why AXOL Exists

Every programming language ever made follows the same paradigm:

```
input → compute → output
```

The program takes data, transforms it through a sequence of operations, and produces a result. Whether it's C, Python, Haskell, or a neural network — the cost of getting an answer is proportional to the complexity of the computation.

AXOL inverts this:

```
structure (once) → observe (O(1), as many times as you want)
```

The computation happens when you **weave** the structure. After that, every observation — no matter how many — costs the same: **~5 microseconds**.

This isn't an optimization. It's a different model of computation, based on quantum measurement theory and chaos dynamics.

---

## Three Things Only AXOL Can Do

### 1. Choose How Much to Know

```
gaze x           # C=0: see all possibilities, lose nothing
glimpse x 0.3    # C=0.3: see partially, lose partially
glimpse x 0.7    # C=0.7: almost certain, most possibilities gone
observe model {}  # C=1: one answer, everything else destroyed
```

In every other language, you either run the function or you don't. AXOL gives you a **continuous spectrum** from total uncertainty to total certainty. The parameter C (collapse level) controls the trade-off between knowledge and possibility.

### 2. Track the Cost of Knowing

```
wave planet = scanner { reading = [...] }
gaze planet                    # all possibilities alive
focus planet 0.5               # partially collapse
widen planet 0.3               # try to undo — but can't fully recover
gaze planet                    # distribution has changed permanently
```

Observation is **irreversible**. Once you collapse possibilities, you cannot fully restore them. AXOL tracks this explicitly through **negativity** — a number that measures how much openness remains in a relation.

No other language models the fact that **knowing something changes the system**.

### 3. Compute Through Interference

```
rel agreement = wave_a <-> wave_b via <~>   # constructive: reinforcement
rel conflict  = wave_a <-> wave_b via <!>   # destructive: differences amplified
rel combined  = wave_a <-> wave_b via <+>   # additive: information accumulates
```

AXOL doesn't compute with numbers. It computes with **interference patterns** between waves. When two waves meet:
- **Constructive (`<~>`)**: what they share gets stronger
- **Destructive (`<!>`)**: what differs gets amplified, what's common cancels
- **Multiplicative (`<*>`)**: only what both have survives
- **Additive (`<+>`)**: information accumulates
- **Conditional (`<?>`)**: one wave rotates the other's phase

This is not metaphor. These are complex-valued amplitude operations following Born rule quantum mechanics.

---

## Installation

```bash
# Clone
git clone https://github.com/user/axol-lang.git
cd axol-lang

# Build
cargo build --release

# Run
./target/release/axol run examples/hello.axol
```

Requirements: Rust 1.70+ (uses `faer` for linear algebra)

---

## Language Reference

### Core Pipeline: Declare → Weave → Observe

```
# 1. Declare the structure
declare "name" {
    input x(dim)              # named input vector of dimension dim
    input y(dim)              # multiple inputs supported
    relate z <- x, y via <~>  # z is produced by interfering x and y
    output z                  # z is the observable output
    quality omega=0.9 phi=0.8 # quality parameters
}

# 2. Weave — create the possibility structure (one-time cost)
weave name quantum=true seed=42

# 3. Observe — read from the structure (O(1) per call)
observe name {
    x = [0.1, 0.2, 0.3, ...]
    y = [0.4, 0.5, 0.6, ...]
}
```

### Collapse Spectrum

| Command | Collapse | Cost | What You Get | What You Lose |
|---------|----------|------|--------------|---------------|
| `gaze x` | C=0 | nothing | full probability distribution | nothing |
| `glimpse x 0.3` | C=0.3 | some possibilities | focused distribution | minor alternatives |
| `focus x 0.7` | C=0.7 | most possibilities | near-certain distribution | most alternatives |
| `observe model {}` | C=1 | everything else | single answer (index) | all other possibilities |

### Basin Structure

Basins define the **attractor landscape** — the topology of possibility space.

```
define_basins "space" {
    dim 8
    basin [0.9, 0.1, 0.2, ...] volume=0.4   # attractor 1 (40% of space)
    basin [0.1, 0.8, 0.7, ...] volume=0.35   # attractor 2 (35%)
    basin [0.5, 0.5, 0.9, ...] volume=0.25   # attractor 3 (25%)
    fractal_dim 1.6
}

weave model quantum=true seed=42 from_basins="space"
```

### Relations (v2)

Relations are first-class objects that model the **structure between waves**.

```
# Create waves
wave a = model { x = [...] }
wave b = model { x = [...] }

# Create a relation
rel r = a <-> b via <~>          # bidirectional, constructive interference

# Observe the relation
observe r {}                      # see the interference pattern

# Expectation: what you think should happen
expect prior = [0.6, 0.3, 0.1, ...] strength=0.7

# Observe with expectation — tracks alignment and negativity change
observe r {} with prior

# Widen: reopen possibilities
widen r 0.3

# Resolve conflicts between relations
resolve r1, r2 with interfere    # quantum interference
resolve r1, r2 with superpose    # superposition
```

### Interference Patterns

| Pattern | Syntax | Behavior | Information |
|---------|--------|----------|-------------|
| Constructive | `<~>` | a + b reinforcement | entropy increases |
| Additive | `<+>` | geometric mean | entropy neutral |
| Multiplicative | `<*>` | a * b, both must be strong | entropy decreases |
| Destructive | `<!>` | a - b, differences amplified | entropy decreases most |
| Conditional | `<?>` | phase coupling | entropy preserved |

### Iteration and Convergence

```
iterate model max=10 converge=prob_delta value=0.005 {
    x = [...]
}

confident model max=50 threshold=0.95 {
    x = [...]
}
```

### Learning

```
learn "xor" dim=4 quantum=1 seed=42 {
    [0.9, 0.1, 0.9, 0.1] = 0
    [0.9, 0.1, 0.1, 0.9] = 1
    [0.1, 0.9, 0.9, 0.1] = 1
    [0.1, 0.9, 0.1, 0.9] = 0
}
```

### All Commands

| Command | Description |
|---------|-------------|
| `declare` | Define a tapestry structure |
| `weave` | Create the possibility space (one-time) |
| `observe` | Full collapse (C=1) |
| `gaze` | Zero-collapse read (C=0) |
| `glimpse` | Partial collapse (C=gamma) |
| `focus` | Partial collapse, mutates wave |
| `reobserve` | Multiple observations, averaged |
| `wave` | Create a named wave variable |
| `rel` | Create a relation between waves |
| `expect` | Define an expectation landscape |
| `widen` | Reopen possibilities |
| `resolve` | Resolve conflicts between waves |
| `iterate` | Iterate until convergence |
| `confident` | Observe until confidence threshold |
| `define_basins` | Define attractor geometry directly |
| `compose` | Chain multiple tapestries |
| `gate` | Quantum logic gate (and, or, not) |
| `learn` | Learn from labeled data |
| `design` | Search for basin structures |

---

## Performance

```
weave (one-time):     ~11ms   (dim=8, chaos dynamics)
observe:              ~5us    (dim=8, O(1))
gaze:                 ~5us    (dim=8, O(1), C=0)
glimpse:              ~30us   (dim=8, includes dephasing)
rel observe:          ~5us    (O(1))
focus:                ~25us   (dim=8, includes density matrix)
```

20 NPC observations per frame: **~80us total**. At 60fps that's 0.5% of frame budget.

---

## Examples

```bash
axol run examples/hello.axol                    # basic pipeline
axol run examples/hello_v2.axol                 # relation-first grammar
axol run examples/usecase_npc_realtime.axol      # game NPC AI
axol run examples/usecase_perception.axol        # predictive coding
axol run examples/usecase_dialogue.axol          # conversation dynamics
axol run examples/usecase_observation_cost.axol   # irreversibility of knowing
axol run examples/learn_xor.axol                 # learning from data
```

---

## Architecture

```
src/
  dsl/
    lexer.rs        # tokenizer
    parser.rs       # AST construction
    compiler.rs     # runtime execution
  wave.rs           # Wave type (complex amplitudes, Born rule)
  collapse.rs       # collapse mechanics
  density.rs        # density matrix operations
  observatory.rs    # compute_wave, gaze, glimpse, observe
  relation.rs       # Relation type (v2)
  weaver.rs         # Tapestry construction
  dynamics.rs       # chaos engine (logistic map, Lorenz)
  learn.rs          # learning from labeled data
  compose/
    iterate.rs      # convergence-based iteration
    confidence.rs   # confidence voting
    logic.rs        # quantum logic gates
    tapestry_chain.rs
    basin_designer.rs
```

---

## License

MIT
