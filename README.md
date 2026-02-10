<p align="center">
  <h1 align="center">AXOL</h1>
  <p align="center"><strong>AI-Optimized Language for LLM Code Generation</strong></p>
  <p align="center">
    <a href="#token-efficiency-benchmark">Benchmarks</a> |
    <a href="#multi-dimensional-language-evaluation">Evaluation</a> |
    <a href="#token-optimization">Optimization</a> |
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
python benchmark_optimizations.py
```

---

## Multi-Dimensional Language Evaluation

AXOL을 토큰 효율성뿐 아니라 **정확성, 실행 속도, 가독성, 표현력, LLM 생성 적합성** 등 다방면에서 평가한 결과입니다.

### Evaluation Summary

| Dimension | Score | Grade | Notes |
|-----------|:-----:|:-----:|-------|
| Token Efficiency | vs Py **-6.5%**, vs C# **-29.1%** | **A** | 데이터 정의 코드에서 최대 -57% |
| Correctness | **262 tests, 100% pass** | **A** | Lexer→Parser→TypeChecker→Interpreter→E2E 전 파이프라인 검증 |
| Execution Speed | **262 tests in <2s** | **B** | Tree-walking 인터프리터, 런타임 성능은 프로덕션 목적이 아님 |
| Readability | S-expr 균일 문법 + indent mode | **B+** | 22개 키워드로 전체 언어 학습 가능, 깊은 중첩 시 가독성 저하 |
| Expressiveness | Structs, ADTs, Pattern Match, HM Type, DbC, Modules | **A-** | 현대 언어 기능 대부분 지원, file import/LSP/WASM 미구현 |
| LLM Friendliness | 단일 문법 형태, 1-token 키워드 | **A** | 구조적 정규성으로 생성 오류 최소화 |
| Complexity Scaling | 코드 유형에 따라 상이 | **B+** | 데이터 정의에서 유리, 순수 알고리즘에서 불리 |

### 1. Token Efficiency (토큰 효율성) — Grade: A

토큰 절감 효과는 코드 **유형**에 따라 크게 달라집니다:

| Code Type | vs Python | vs C# | Verdict |
|-----------|:---------:|:-----:|---------|
| Data Definition (단순 struct) | **-57.1%** | — | AXOL 압도적 우위 |
| Data Definition (복잡 nested) | **-9.9%** | — | AXOL 우위 유지 |
| Contract/Validation (DbC) | **-12.9%** | **-39.8%** | AXOL 항상 우위 |
| Pure Algorithm (단순) | **-10.0%** | **-40.3%** | AXOL 우위 |
| Pure Algorithm (복잡) | **+22.7%** | — | **Python이 우위** |
| HOF/Functional chain | **-1.6%** | — | 거의 동등 |

최적화 적용 시 (indent mode + short enum + positional struct):
- **표준 AXOL vs Python**: -39.4%
- **최적화 AXOL vs Python**: **-47.8%** (목표 35-45% 초과 달성)

### 2. Correctness (정확성) — Grade: A

| Test Suite | Tests | Pass Rate |
|-----------|:-----:|:---------:|
| Lexer | 24 | 100% |
| Parser | 32 | 100% |
| TypeChecker | 71 | 100% |
| Interpreter | 131 | 100% |
| E2E Pipeline | 4 | 100% |
| **Total** | **262** | **100%** |

- Hindley-Milner 타입 추론 (Algorithm W + Robinson 통합)
- Design-by-Contract (Q/G/!) 런타임 검증
- JSON 구조화 에러 리포팅 (line/column 포함)
- Occurs check로 무한 타입 방지

### 3. Execution Speed (실행 속도) — Grade: B

| Metric | Value |
|--------|-------|
| Full test suite (262 tests) | **<2 seconds** |
| Individual test | **<1ms** (대부분) |
| E2E pipeline (Lex→Parse→Type→Interpret) | **~40ms** |
| REPL response | Instant |

> AXOL은 tree-walking 인터프리터로, 런타임 성능보다 **LLM 토큰 최적화**가 설계 목표입니다. 바이트코드 컴파일은 로드맵에 포함되어 있습니다.

### 4. Readability (가독성) — Grade: B+

**장점:**
- **22개 키워드**만으로 전체 언어 구성 (학습 곡선 최소화)
- **단일 문법 형태**: 모든 것이 `(keyword args...)` — 특수 구문 없음
- **Indent mode (.axoli)**: 괄호 닫기 제거로 Python 수준 가독성
- 타입 어노테이션 선택적 — 필요할 때만 `[i i -> i]` 추가

**한계:**
- 깊은 중첩 시 괄호 과다 (indent mode로 완화)
- 단일 문자 키워드 (`X`, `Q`, `G`)는 사전 지식 없이 의미 파악 어려움
- 기존 Lisp 경험이 없으면 S-expression 자체가 낯설 수 있음

### 5. Expressiveness (표현력) — Grade: A-

**구현 완료:**
- Structs (named + positional), Enums (ADTs), Pattern matching (destructuring + guards)
- Lambdas, HOF (map/filter/reduce/sort/...), Pipe operator
- HM Type Inference (Algorithm W), Modules, Design-by-Contract
- Mutable/Immutable 구분, Try-catch, REPL, 49개 내장 함수

**미구현 (로드맵):**
- File-based module imports, Pattern completeness check
- Bytecode VM, LSP, Package manager, FFI (.NET), WASM target

### 6. LLM Generation Friendliness (LLM 생성 적합성) — Grade: A

| Factor | AXOL | Python | C# |
|--------|:----:|:------:|:--:|
| 구문 정규성 | **S-expr 단일 형태** | 다중 구문 혼재 | 다중 구문 혼재 |
| 키워드 토큰 크기 | **1 BPE token** | 1-2 tokens | 1-2 tokens |
| 괄호 매칭 오류 가능성 | 높음 | 낮음 | 중간 |
| LLM 학습 데이터 존재 | **거의 없음** | 풍부 | 풍부 |
| 생성 코드 구조 오류율 | 낮음 (정규 문법) | 중간 (들여쓰기 오류) | 높음 (누락된 세미콜론/브레이스) |

> AXOL의 S-expression 문법은 **구조적으로 정규적**이므로 LLM이 올바른 형태를 생성할 확률이 높지만, **학습 데이터 부족**이 현실적 제약입니다. System prompt에 문법 규칙을 포함하는 것이 권장됩니다.

### 7. Complexity Scaling (복잡도 확장성) — Grade: B+

```
Token Savings vs Python (by code type & complexity)

Data Definition:  ████████████████████████████████████████████████████████ -57% (simple)
                  ██████████████████████████████████ -34% (medium)
                  ██████████ -10% (complex)

Contracts/DbC:    █████████████ -13% (consistent)

HOF/Functional:   ██ -2% (nearly equal)

Pure Algorithm:   ██████████ -10% (simple)
                  ▓▓▓▓▓▓▓▓▓▓▓▓ +12% (medium, Python wins)
                  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ +23% (complex, Python wins)
```

**Key insight**: "복잡할수록 유리하다"가 아니라 **"데이터 정의 비중이 높을수록 유리하다"**가 정확합니다.

---

## Token Optimization

AXOL v0.2 introduces three syntax optimizations that further reduce token count:

### 1. Indentation Mode (`.axoli`)

Eliminates closing parentheses via indentation — the preprocessor auto-wraps lines and nests children:

```lisp
;; Standard AXOL (78 tokens)
(f apply_dmg ent amt
  (Q (>= amt 0))
  (v oh (@ ent hp))
  (v raw (- amt (@ ent def)))
  (v dmg (? (< raw 0) 0 raw))
  (m! (@ ent hp) (max 0 (- oh dmg)))
  (G (<= (@ ent hp) oh))
  ent)
```

```lisp
;; Indent mode — .axoli file (71 tokens, +9.0%)
f apply_dmg ent amt
  Q (>= amt 0)
  v oh (@ ent hp)
  v raw (- amt (@ ent def))
  v dmg (? (< raw 0) 0 raw)
  m! (@ ent hp) (max 0 (- oh dmg))
  G (<= (@ ent hp) oh)
  ent
```

### 2. Short Enum Aliases (`.Variant`)

Use `.Variant` instead of `EnumName.Variant` in pattern matching — the enum name is inferred from the subject:

```lisp
;; Before (130 tokens)
(X shape (Shape.Circle r) (* 3.14 (* r r)) (Shape.Rect w h) (* w h))

;; After (122 tokens, +6.2%)
(X shape (.Circle r) (* 3.14 (* r r)) (.Rect w h) (* w h))
```

### 3. Positional Struct Fields

When a type definition exists, field names can be omitted — values are matched by position:

```lisp
;; Before (139 tokens)
(t Stats hp mp atk def spd)
(v warrior (S Stats hp 150 mp 20 atk 30 def 25 spd 10))
(v mage    (S Stats hp 80  mp 100 atk 10 def 8  spd 12))

;; After (112 tokens, +19.4%)
(t Stats hp mp atk def spd)
(v warrior (S Stats 150 20 30 25 10))
(v mage    (S Stats 80 100 10 8 12))
```

### Optimization Benchmark

| Optimization | Avg Savings | Best Case | Implementation |
|---|:---:|:---:|---|
| Indentation mode (`.axoli`) | +1.9% | +9.0% | `IndentPreprocessor.cs` + CLI `expand` |
| Short enum aliases (`.Variant`) | +4.9% | +6.2% | Interpreter `MatchPattern` |
| **Positional struct fields** | **+17.5%** | **+19.4%** | Interpreter `EvalStructLiteral` |
| **All combined** | **+13.8%** | | |

### Combined Result: AXOL vs Python

| Comparison | Tokens | Savings |
|---|:---:|:---:|
| Python equivalent | 203 | (baseline) |
| AXOL (standard syntax) | 123 | **-39.4%** |
| **AXOL (fully optimized)** | **106** | **-47.8%** |

> **Before optimization**: AXOL saved ~39% tokens vs Python. **After**: nearly **48%** — exceeding the original 35-45% target.

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

### Run Indent Mode (`.axoli`)

```bash
dotnet run --project src/Axol.Cli -- run samples/demo.axoli
```

### Expand `.axoli` to Standard AXOL

```bash
dotnet run --project src/Axol.Cli -- expand samples/demo.axoli
# Outputs the equivalent S-expression form
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
# 262 tests passed (Lexer: 24, Parser: 32, TypeChecker: 71, Interpreter: 131, E2E: 4)
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

; Create instance (named fields)
(v hero (S Entity hp 100 atk 25 def 10))

; Create instance (positional fields — requires type def)
(v hero (S Entity 100 25 10))

; Access field
(@ hero hp)
hero.hp          ; dot syntax shorthand
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

; Short enum aliases (infer enum from subject)
(X shape
  (.Circle r) (* 3.14 (* r r))
  (.Rect w h) (* w h))

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
│   ├── Axol.Core/           # Shared types: AST, Tokens, Diagnostics, SourceMap, IndentPreprocessor
│   ├── Axol.Lexer/          # Tokenizer (comments, dot-access, brackets, braces)
│   ├── Axol.Parser/         # Recursive descent parser with error recovery
│   ├── Axol.TypeChecker/    # HM type inference (Algorithm W + Robinson unification)
│   ├── Axol.Interpreter/    # Tree-walking interpreter + 49 builtins + modules
│   └── Axol.Cli/            # CLI (run, check, tokens, expand, repl)
├── tests/
│   ├── Axol.Lexer.Tests/        # 24 tests
│   ├── Axol.Parser.Tests/       # 32 tests
│   ├── Axol.TypeChecker.Tests/  # 71 tests (unification + inference)
│   ├── Axol.Interpreter.Tests/  # 131 tests (builtins, patterns, modules, preprocessor)
│   └── Axol.E2E.Tests/          # 4 end-to-end pipeline tests
└── samples/
    ├── *.axol                # AXOL sample programs
    ├── python_equiv/         # Python equivalents for comparison
    ├── csharp_equiv/         # C# equivalents for comparison
    ├── phase2/               # Phase 2 enhanced syntax samples
    ├── benchmark_full.py     # Token comparison benchmark script
    └── benchmark_optimizations.py  # Optimization savings benchmark
```

### Pipeline

```
Source (.axol/.axoli) → [IndentPreprocessor] → Lexer → Tokens → Parser → AST → TypeChecker → Interpreter → Result
                                                  ↑
                                          Hindley-Milner
                                          Type Inference
                                          (Algorithm W)
```

### Technology Stack

- **Runtime**: .NET 8.0 (C#)
- **Testing**: xUnit (262 tests)
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
| `S` | Struct instance | `(S Point x 10 y 20)` or `(S Point 10 20)` |
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
- [x] Indentation mode (`.axoli` files — eliminate closing parens)
- [x] Short enum aliases (`.Variant` pattern matching)
- [x] Positional struct fields (omit field names with type def)
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
