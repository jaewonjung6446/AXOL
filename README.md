<p align="center">
  <h1 align="center">AXOL</h1>
  <p align="center"><strong>AI-Optimized Language for LLM Code Generation</strong></p>
  <p align="center">
    <a href="#token-efficiency-benchmark">Benchmarks</a> |
    <a href="#quick-start">Quick Start</a> |
    <a href="#language-reference">Language Reference</a> |
    <a href="#architecture">Architecture</a>
  </p>
</p>

---

> **Status: Experimental (v0.2-alpha)**
> AXOL is an active research project. The language specification, syntax, and APIs are subject to change. Not recommended for production use. Contributions and feedback are welcome.

## What is AXOL?

AXOL is a programming language specifically designed to **minimize BPE token consumption** when used with Large Language Models (GPT-4, Claude, etc.). It uses an S-expression-based syntax with aggressive abbreviation, enabling AI models to generate, read, and reason about programs using significantly fewer tokens than conventional languages.

### The Problem

When LLMs generate code, every token costs money and latency. A simple Python function:

```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

for i in range(0, 10):
    print(fibonacci(i))
```
**46 BPE tokens, 120 characters**

The same logic in C#:
```csharp
static int Fib(int n)
{
    if (n <= 1) return n;
    return Fib(n - 1) + Fib(n - 2);
}

static void Main()
{
    for (int i = 0; i < 10; i++)
        Console.WriteLine(Fib(i));
}
```
**77 BPE tokens, 252 characters**

In AXOL:
```lisp
(f fib [i -> i] n (? (<= n 1) n (+ (fib (- n 1)) (fib (- n 2)))))
(F i (range 0 10) (print (fib i)))
```
**46 BPE tokens, 100 characters**

AXOL matches Python's token count with 17% fewer characters, and uses **40% fewer tokens than C#**.

### The Hypothesis

> **If a language's syntax is designed around BPE tokenization patterns, it can express equivalent semantics in significantly fewer tokens, reducing LLM inference cost and latency.**

AXOL tests this hypothesis with:
- **Single-character keywords** (`f` for function, `v` for variable, `?` for if-else)
- **S-expression uniformity** (no syntactic sugar that inflates token count)
- **Optional type annotations** (add safety without token overhead when omitted)
- **Built-in contracts** (preconditions/postconditions as first-class constructs)
- **Hindley-Milner type inference** (types are checked without explicit annotations)

---

## Token Efficiency Benchmark

Measured using **tiktoken `cl100k_base`** (GPT-4 / Claude tokenizer approximation).

### AXOL vs Python vs C# (BPE Tokens)

| Example | AXOL | Python | C# | vs Python | vs C# |
|---------|:---:|:---:|:---:|:---:|:---:|
| fibonacci | 46 | 46 | 77 | **+0.0%** | **-40.3%** |
| contracts (DbC) | 74 | 83 | 123 | **-10.8%** | **-39.8%** |
| combat (struct/OOP) | 215 | 229 | 282 | **-6.1%** | **-23.8%** |
| data_heavy (collections) | 216 | 243 | 335 | **-11.1%** | **-35.5%** |
| combat_sim (logic) | 182 | 194 | 250 | **-6.2%** | **-27.2%** |
| pattern+module (Phase 2) | 233 | 238 | 295 | **-2.1%** | **-21.0%** |
| **Total** | **966** | **1033** | **1362** | **-6.5%** | **-29.1%** |

### Summary

| Metric | vs Python | vs C# |
|--------|:---------:|:-----:|
| BPE Token Reduction | **-6.5%** | **-29.1%** |
| Character Reduction | **-16.5%** | **-47.2%** |

> **Note:** Python is already one of the most token-efficient mainstream languages due to minimal syntax. AXOL still achieves measurable savings, and the gap widens significantly against verbose languages like C#, Java, or TypeScript.

### Where AXOL Excels

- **Data-heavy code** (structs, collections): -11% vs Python, -35% vs C#
- **Contract/assertion code** (DbC patterns): -11% vs Python, -40% vs C#
- **Combat/game logic** (stateful computation): -6% vs Python, -24% vs C#

### How to Reproduce

```bash
cd samples
pip install tiktoken
python benchmark_full.py
```

---

## Theoretical Background

### BPE Tokenization and LLM Cost

Large Language Models process text through **Byte Pair Encoding (BPE)** tokenization. The cost of an LLM API call is directly proportional to token count:

```
Cost = input_tokens * price_per_input_token + output_tokens * price_per_output_token
```

For GPT-4 class models, this means:
- A 1000-token program costs ~$0.03 per generation
- A 700-token equivalent saves ~$0.009 per call
- At 10,000 API calls/day, that's **$90/day saved**

AXOL's design targets this economic reality.

### Design Principles

1. **Single-Character Keywords**: Most keywords are single characters that tokenize into 1 BPE token:
   - `f` (function), `v` (variable), `?` (conditional), `F` (for), `W` (while)
   - Compare: `function` (1-2 tokens), `return` (1 token), `while` (1 token)

2. **Uniform S-Expression Syntax**: Everything is `(keyword args...)`:
   - No curly braces, semicolons, colons, or indentation requirements
   - Parentheses are single BPE tokens with high frequency in training data

3. **Structural Minimalism**: No `return` keyword (last expression is return value), no `var`/`let`/`const` distinction (just `v`)

4. **Hindley-Milner Type Inference**: Types are inferred automatically, annotations are optional:
   - `(f add a b (+ a b))` — types inferred as `Int -> Int -> Int`
   - `(f add [i i -> i] a b (+ a b))` — explicit annotation when needed

5. **Design-by-Contract (DbC)**: Preconditions (`Q`), postconditions (`G`), assertions (`!`) as single-character builtins

### Algorithm W Type Inference

AXOL implements the **Hindley-Milner type system** with Algorithm W:

```
Infer(x) where x is a variable:
    if x is in the type environment:
        return Instantiate(env[x])

Infer(lambda x. e):
    alpha = fresh type variable
    env' = env + {x: alpha}
    tau = Infer(e) in env'
    return alpha -> tau

Infer(f(e)):
    tau_f = Infer(f)
    tau_e = Infer(e)
    alpha = fresh type variable
    Unify(tau_f, tau_e -> alpha)
    return alpha
```

This provides:
- **Let-polymorphism**: Generic functions work with any compatible type
- **Occurs check**: Prevents infinite type construction
- **Unification**: Robinson's algorithm for type equation solving
- **Type error reporting**: JSON-structured diagnostics with line/column info

---

## Quick Start

### Prerequisites

- [.NET 8.0 SDK](https://dotnet.microsoft.com/download/dotnet/8.0)

### Build

```bash
git clone https://github.com/YourUsername/AXOL.git
cd AXOL
dotnet build AXOL.sln
```

### Run a Program

```bash
dotnet run --project src/Axol.Cli -- run samples/fibonacci.axol
```

### Type-Check Only

```bash
dotnet run --project src/Axol.Cli -- check samples/contracts.axol
# Output: {"status":"ok"}
```

### Count Tokens

```bash
dotnet run --project src/Axol.Cli -- tokens samples/fibonacci.axol
# Output: {"file":"fibonacci.axol","tokens":25,"chars":100}
```

### Interactive REPL

```bash
dotnet run --project src/Axol.Cli -- repl
axol> (+ 1 2)
3
axol> (f square x (* x x))
axol> (square 7)
49
```

### Run Tests

```bash
dotnet test
# 236 tests passed (Lexer: 24, Parser: 32, TypeChecker: 71, Interpreter: 106, E2E: 3)
```

---

## Language Reference

### Variables and Types

```lisp
(v name "AXOL")              ; variable binding
(v x 42)                     ; integer
(v pi 3.14)                  ; float
(v flag true)                ; boolean
(v items [1 2 3])            ; array literal (Phase 2)
(v config {"key" "value"})   ; map literal (Phase 2)
```

### Functions

```lisp
; Basic function
(f add a b (+ a b))

; With type annotation
(f add [i i -> i] a b (+ a b))

; Type abbreviations: i=int, f=float, s=string, b=bool
; Arrow type: [input -> output]
; Multiple params: [i i -> i]
```

### Control Flow

```lisp
; Conditional (ternary)
(? (> x 0) "positive" "non-positive")

; For loop
(F i (range 0 10) (print i))

; While loop
(W (> n 0) (m! n (- n 1)))

; Do block (sequence)
(D (print "a") (print "b") (+ 1 2))
```

### Structs

```lisp
; Define struct type
(t Entity hp atk def)

; Create instance
(v hero (S Entity hp 100 atk 25 def 10))

; Access field (Phase 1 syntax)
(@ hero hp)

; Access field (Phase 2 dot syntax)
hero.hp
```

### Enums

```lisp
; Simple enum
(e Color Red Green Blue)
(v c Color.Red)

; Data-bearing enum (algebraic data type)
(e Shape (Circle r) (Rect w h))
(v s (Shape.Circle 5))
```

### Pattern Matching

```lisp
; Basic matching
(X value
  1 "one"
  2 "two"
  _ "other")

; Variable binding
(X value x (* x 2))

; List destructuring
(X my_list (A head rest...) head)

; Struct destructuring
(X point (S Point x y) (+ x y))

; Enum matching with data extraction
(X shape
  (Shape.Circle r) (* 3.14 (* r r))
  (Shape.Rect w h) (* w h))

; Guard clauses
(X n
  x (when (> x 0)) "positive"
  x (when (< x 0)) "negative"
  _ "zero")
```

### Lambdas and Higher-Order Functions

```lisp
; Lambda (single param)
(L x (* x 2))

; Lambda (multi param)
(L (a b) (+ a b))

; Map, filter, reduce
(map [1 2 3] (L x (* x 2)))          ; => [2 4 6]
(filter [1 2 3 4] (L x (> x 2)))     ; => [3 4]
(reduce [1 2 3 4] 0 (L (acc x) (+ acc x)))  ; => 10

; Pipe
(P (range 1 10) (L xs (map xs (L x (* x x)))))
```

### Modules

```lisp
; Define module
(M math
  (f square x (* x x))
  (f cube x (* x (* x x))))

; Use with qualified name
(math.square 5)    ; => 25

; Import into scope
(use math.square)
(square 5)         ; => 25
```

### Design by Contract

```lisp
(f safe_div [i i -> i] a b
  (Q (!= b 0))           ; Precondition: b must not be zero
  (v r (/ a b))
  (G (>= r 0))           ; Postcondition: result must be non-negative
  r)

(! (= (safe_div 10 2) 5)) ; Assertion
```

### Error Handling

```lisp
; Try-catch
(try
  (/ 10 0)
  (catch e "division error"))
```

### Built-in Functions (49 total)

**Arithmetic**: `+`, `-`, `*`, `/`, `%`, `max`, `min`
**Comparison**: `=`, `!=`, `<`, `>`, `<=`, `>=`
**Logic**: `&`, `|`, `~` (not)
**String**: `upper`, `lower`, `split`, `join`, `trim`, `replace`, `starts_with`, `ends_with`, `slice`, `str`, `len`
**Array HOF**: `map`, `filter`, `reduce`, `sort`, `reverse`, `flatten`, `zip`, `find`, `any`, `all`
**Math**: `floor`, `ceil`, `round`, `sin`, `cos`, `pow`, `log`, `random`
**IO**: `print`, `read_file`, `write_file`
**Util**: `format`, `assert_eq`, `typeof`, `range`

---

## Architecture

```
AXOL/
├── src/
│   ├── Axol.Core/           # Shared types: AST, Tokens, Diagnostics, SourceMap
│   ├── Axol.Lexer/          # Tokenizer (comments, dot-access, brackets, braces)
│   ├── Axol.Parser/         # Recursive descent parser with error recovery
│   ├── Axol.TypeChecker/    # HM type inference (Algorithm W + Robinson unification)
│   ├── Axol.Interpreter/    # Tree-walking interpreter + 49 builtins + modules
│   └── Axol.Cli/            # CLI (run, check, tokens, repl)
├── tests/
│   ├── Axol.Lexer.Tests/        # 24 tests
│   ├── Axol.Parser.Tests/       # 32 tests
│   ├── Axol.TypeChecker.Tests/  # 71 tests (unification + inference)
│   ├── Axol.Interpreter.Tests/  # 106 tests (builtins, patterns, modules)
│   └── Axol.E2E.Tests/          # 3 end-to-end pipeline tests
└── samples/
    ├── *.axol                # AXOL sample programs
    ├── python_equiv/         # Python equivalents for comparison
    ├── csharp_equiv/         # C# equivalents for comparison
    ├── phase2/               # Phase 2 enhanced syntax samples
    └── benchmark_full.py     # Token comparison benchmark script
```

### Pipeline

```
Source Code → Lexer → Tokens → Parser → AST → TypeChecker → Interpreter → Result
                                                  ↑
                                          Hindley-Milner
                                          Type Inference
                                          (Algorithm W)
```

### Technology Stack

- **Runtime**: .NET 8.0 (C#)
- **Testing**: xUnit (236 tests)
- **CLI**: System.CommandLine
- **Tokenizer Benchmark**: Python + tiktoken

---

## Keyword Quick Reference

| Keyword | Meaning | Example |
|---------|---------|---------|
| `f` | Function definition | `(f add a b (+ a b))` |
| `v` | Variable binding | `(v x 42)` |
| `?` | Conditional (if-else) | `(? cond then else)` |
| `F` | For loop | `(F i (range 0 10) body)` |
| `W` | While loop | `(W cond body)` |
| `D` | Do block (sequence) | `(D expr1 expr2)` |
| `L` | Lambda | `(L x (* x 2))` |
| `P` | Pipe | `(P data transform)` |
| `S` | Struct instance | `(S Point x 10 y 20)` |
| `t` | Type definition | `(t Point x y)` |
| `e` | Enum definition | `(e Color Red Green Blue)` |
| `X` | Pattern match | `(X val pattern result ...)` |
| `A` | Array | `(A 1 2 3)` |
| `H` | Hash map | `(H "key" value)` |
| `M` | Module | `(M name body...)` |
| `Q` | Precondition | `(Q (> x 0))` |
| `G` | Postcondition | `(G (>= result 0))` |
| `!` | Assertion | `(! (= a b))` |
| `@` | Field access | `(@ obj field)` |
| `m` | Mutable variable | `(m counter 0)` |
| `m!` | Mutate | `(m! counter (+ counter 1))` |
| `#` | Index access | `(# list 0)` |

---

## Roadmap

- [ ] File-based module imports (`(import "path/to/module.axol")`)
- [ ] Pattern matching completeness checking
- [ ] Bytecode compilation (targeting a stack VM)
- [ ] Language Server Protocol (LSP) for editor support
- [ ] Package manager
- [ ] FFI (Foreign Function Interface) for .NET interop
- [ ] WASM compilation target
- [ ] Formal BPE optimization pass (rewrite rules that minimize token count)

---

## Contributing

AXOL is in early experimental stage. We welcome:

- **Bug reports**: Open an issue with a minimal reproducing example
- **Language design feedback**: Propose syntax changes that improve token efficiency
- **Benchmark contributions**: Add equivalent programs in other languages
- **Test cases**: Edge cases for the type checker, pattern matching, or interpreter

```bash
# Run tests before submitting
dotnet test

# Run benchmarks
cd samples && python benchmark_full.py
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>AXOL</strong> — Every token counts.
</p>
