<p align="center">
  <h1 align="center">AXOL</h1>
  <p align="center">
    <strong>基于混沌理论的空间-概率计算语言</strong>
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

> **警告：本项目处于早期实验阶段。**
> API、DSL语法和内部架构可能在没有通知的情况下发生重大变更。不建议在生产环境中使用。欢迎贡献和反馈。

---

## 什么是AXOL？

**AXOL**是一种**否定时间轴**（顺序执行）作为计算基础的编程语言，并用两个替代轴取而代之：

- **空间轴** — 节点之间的关系决定计算
- **概率轴** — 结果的可能性决定结果

AXOL不问"先做这个，再做那个"，而是问："什么与什么相关，关联有多强？"其结果是一种基于**混沌理论**的根本不同的执行模型，计算就是构建和观测**奇异吸引子**的行为。

```
传统模式：  指令1 → 指令2 → 指令3  （时间顺序）

AXOL：
  [空间]     NodeA ──relation── NodeB     "在哪里"决定计算
  [概率] state = { alpha|possibility1> + beta|possibility2> }  "有多大可能"决定结果
```

### 核心特性

- **三阶段执行**：Declare → Weave → Observe（不是编译→运行）
- **混沌理论基础**：Tapestry = 奇异吸引子，质量通过Lyapunov指数和分形维度衡量
- **双重质量指标**：Omega（结合度）+ Phi（清晰度）— 严谨、可测量、可组合
- 相比等效Python代码**平均节省63% token**（量子DSL）
- **不可达检测** — 在计算之前警告目标是否在数学上不可达
- **Lyapunov估计精度**：平均误差0.0002
- 基础层中的**9个原语操作**：`transform`、`gate`、`merge`、`distance`、`route`（加密）+ `step`、`branch`、`clamp`、`map`（明文）
- **矩阵级加密** — 相似变换使程序在密码学层面不可读
- **NumPy后端**，可选GPU加速（CuPy/JAX）

---

## 目录

- [范式转换](#范式转换)
- [三阶段执行模型](#三阶段执行模型)
- [质量指标](#质量指标)
- [混沌理论基础](#混沌理论基础)
- [组合规则](#组合规则)
- [量子DSL](#量子dsl)
- [性能](#性能)
- [基础层](#基础层)
  - [9个原语操作](#9个原语操作)
  - [矩阵加密（Shadow AI）](#矩阵加密shadow-ai)
  - [明文操作与安全分类](#明文操作与安全分类)
  - [编译器优化器](#编译器优化器)
  - [GPU后端](#gpu后端)
  - [模块系统](#模块系统)
  - [量子干涉（Phase 6）](#量子干涉phase-6)
  - [客户端-服务器架构](#客户端-服务器架构)
- [架构](#架构)
- [快速开始](#快速开始)
- [API参考](#api参考)
- [示例](#示例)
- [测试套件](#测试套件)
- [路线图](#路线图)

---

## 范式转换

### 我们否定什么

每种现代编程语言都建立在**时间轴**（顺序执行）之上：

| 范式 | 时间轴依赖 |
|------|-----------|
| 命令式（C、Python） | "先做这个，再做那个"——显式时间顺序 |
| 函数式（Haskell、Lisp） | 声明式，但存在求值顺序 |
| 并行（Go、Rust async） | 多条时间轴同时进行——仍然受时间约束 |
| 声明式（SQL、HTML） | 描述"做什么"，但引擎在时间轴上处理 |

这是因为Von Neumann架构基于时钟周期运行——即时间轴。

### 我们提出什么

AXOL用两个替代轴取代时间轴：

| 轴 | 决定什么 | 类比 |
|----|---------|------|
| **空间轴**（关系） | 由关系连接的节点决定计算 | "在哪里"重要，而非"何时" |
| **概率轴**（可能性） | 叠加态坍缩到最可能的结果 | "有多大可能"重要，而非"精确" |

权衡：**我们牺牲精确性来消除时间瓶颈。**

```
精确性 ↑  →  纠缠成本 ↑  →  构建时间 ↑
精确性 ↓  →  纠缠成本 ↓  →  构建时间 ↓
               但观测始终是即时的
```

### 为什么这很重要

| 属性 | 传统编译 | AXOL纠缠 |
|------|---------|----------|
| 准备 | 代码→机器翻译 | 在逻辑之间构建概率关联 |
| 执行 | 顺序机器指令 | 观测（输入）→即时坍缩 |
| 瓶颈 | 与执行路径长度成正比 | 仅取决于纠缠深度 |
| 类比 | "修建一条快速公路" | "已经在目的地" |

---

## 三阶段执行模型

### 第一阶段：Declare（声明）

定义**什么与什么相关**并设定质量目标。此时不进行任何计算。

```python
from axol.quantum import DeclarationBuilder, RelationKind

decl = (
    DeclarationBuilder("search")
    .input("query", 64)
    .input("db", 64)
    .relate("relevance", ["query", "db"], RelationKind.PROPORTIONAL)
    .output("relevance")
    .quality(omega=0.9, phi=0.7)   # 质量目标
    .build()
)
```

或使用DSL：

```
entangle search(query: float[64], db: float[64]) @ Omega(0.9) Phi(0.7) {
    relevance <~> similarity(query, db)
    ranking <~> relevance
}
```

### 第二阶段：Weave（织造）

构建**奇异吸引子**（Tapestry）。这是计算成本发生的阶段。织造器（weaver）：

1. 估计纠缠成本
2. 检测不可达性（当目标在数学上不可达时发出警告）
3. 为每个节点构建吸引子结构（轨迹矩阵、Hadamard干涉）
4. 估计Lyapunov指数和分形维度
5. 组装内部`Program`以供执行

```python
from axol.quantum import weave

tapestry = weave(decl, seed=42)
print(tapestry.weaver_report)
# target:   Omega(0.90) Phi(0.70)
# achieved: Omega(0.95) Phi(0.82)
# feasible: True
```

不可达检测示例：

```
> weave predict_weather: WARNING
>   target:   Omega(0.99) Phi(0.99)
>   maximum:  Omega(0.71) Phi(0.68)
>   reason:   chaotic dependency (lambda=2.16 on path: input->atmosphere->prediction)
>   attractor_dim: D=2.06 (Lorenz-class)
```

### 第三阶段：Observe（观测）

输入值→**即时坍缩**到吸引子上的一点。时间复杂度：O(D)，其中D是吸引子的嵌入维度。

```python
from axol.quantum import observe, reobserve
from axol.core.types import FloatVec

# 单次观测
result = observe(tapestry, {
    "query": FloatVec.from_list([1.0] * 64),
    "db": FloatVec.from_list([0.5] * 64),
})
print(f"Omega: {result.omega:.2f}, Phi: {result.phi:.2f}")

# 重复观测以提升质量
result = reobserve(tapestry, inputs, count=10)
# 对概率分布取平均，重新计算经验Omega
```

### 完整管道（DSL）

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

## 质量指标

AXOL在两个独立的轴上衡量计算质量：

```
        Phi（清晰度）
        ^
   1.0  |  锐利但不稳定         理想（强纠缠）
        |
   0.0  |  噪声                 稳定但模糊
        +-----------------------------> Omega（结合度）
       0.0                             1.0
```

### Omega — 结合度（有多稳定？）

由**最大Lyapunov指数**（lambda）推导：

```
Omega = 1 / (1 + max(lambda, 0))
```

| lambda | 含义 | Omega |
|--------|------|-------|
| lambda < 0 | 收敛系统（稳定） | 1.0 |
| lambda = 0 | 中性稳定 | 1.0 |
| lambda = 0.91 | Lorenz级混沌 | 0.52 |
| lambda = 2.0 | 强混沌 | 0.33 |

**解读**：Omega = 1.0意味着重复观测总是给出相同结果。Omega < 1.0意味着混沌敏感性——微小的输入变化会导致不同的输出。

### Phi — 清晰度（有多锐利？）

由吸引子的**分形维度**（D）推导：

```
Phi = 1 / (1 + D / D_max)
```

| D | D_max | 含义 | Phi |
|---|-------|------|-----|
| 0 | 任意 | 点（delta分布） | 1.0 |
| 1 | 4 | 线性吸引子 | 0.80 |
| 2.06 | 3 | Lorenz吸引子 | 0.59 |
| D_max | D_max | 填满整个相空间 | 0.50 |

**解读**：Phi = 1.0意味着输出是一个锐利、确定的值。Phi → 0.0意味着输出分散在多种可能性中（噪声）。

### 两个指标都可组合

质量指标通过组合进行传播——参见[组合规则](#组合规则)。

---

## 混沌理论基础

AXOL的理论基础将其概念映射到成熟的混沌理论：

| AXOL概念 | 混沌理论 | 数学对象 |
|----------|---------|---------|
| Tapestry | 奇异吸引子 | 相空间中的紧致不变集 |
| Omega（结合度） | Lyapunov稳定性 | `1/(1+max(lambda,0))` |
| Phi（清晰度） | 分形维度倒数 | `1/(1+D/D_max)` |
| Weave | 吸引子构建 | 迭代映射的轨迹矩阵 |
| Observe | 吸引子上的点坍缩 | 时间复杂度O(D) |
| 纠缠范围 | 吸引域 | 收敛区域的边界 |
| 纠缠成本 | 收敛迭代次数 | `E = sum_path(iterations * complexity)` |
| 观测后复用 | 吸引子稳定性 | lambda < 0: 可复用, lambda > 0: 需重新织造 |

### Lyapunov指数估计

使用**Benettin QR分解方法**从轨迹矩阵估计最大Lyapunov指数。

- **收缩系统**（lambda < 0）：可预测，Omega趋近1.0
- **中性系统**（lambda = 0）：混沌边缘
- **混沌系统**（lambda > 0）：对初始条件敏感，Omega < 1.0

估计精度已通过已知系统验证（平均误差：0.0002）。

### 分形维度估计

两种可用方法：

- **盒计数法**：基于网格，对ln(N) vs ln(1/epsilon)进行回归
- **相关维度**（Grassberger-Procaccia）：成对距离分析

已通过已知几何体验证：线段（D=1）、Cantor集（D~0.63）、Sierpinski三角形（D~1.58）。

### 完整理论文档

- [THEORY.md](THEORY.md) — 基础理论（时间轴否定、基于纠缠的计算）
- [THEORY_MATH.md](THEORY_MATH.md) — 混沌理论形式化（Lyapunov、分形、组合证明）

---

## 组合规则

当组合多个Tapestry时，质量指标按照严格的数学规则传播：

### 串行组合（A → B）

```
lambda_total = lambda_A + lambda_B          （指数累加）
Omega_total  = 1/(1+max(lambda_total, 0))   （Omega退化）
D_total      = D_A + D_B                    （维度求和）
Phi_total    = Phi_A * Phi_B                （Phi相乘——总是退化）
```

### 并行组合（A || B）

```
lambda_total = max(lambda_A, lambda_B)      （最弱环节）
Omega_total  = min(Omega_A, Omega_B)        （最弱环节）
D_total      = max(D_A, D_B)               （最复杂）
Phi_total    = min(Phi_A, Phi_B)            （最不清晰）
```

### 复用规则

```
lambda < 0  →  观测后可复用（吸引子稳定）
lambda > 0  →  观测后需重新织造（混沌——吸引子被扰乱）
```

### 汇总表

| 模式 | lambda | Omega | D | Phi |
|------|--------|-------|---|-----|
| 串行 | sum | 1/(1+max(sum,0)) | sum | product |
| 并行 | max | min | max | min |

---

## 量子DSL

### 语法概览

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

### 关系运算符

| 运算符 | 名称 | 含义 |
|--------|------|------|
| `<~>` | 正比 | 线性相关 |
| `<+>` | 加法 | 加权和 |
| `<*>` | 乘法 | 乘积关系 |
| `<!>` | 反比 | 反向相关 |
| `<?>` | 条件 | 上下文相关 |

### 示例

#### 简单搜索

```
entangle search(query: float[64], db: float[64]) @ Omega(0.9) Phi(0.7) {
    relevance <~> similarity(query, db)
    ranking <~> relevance
}
result = observe search(query_vec, db_vec)
```

#### 多阶段管道

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

#### 分类

```
entangle classify(input: float[32]) @ Omega(0.95) Phi(0.9) {
    category <~> classify(input)
}
result = observe classify(input_vec)
```

---

## 性能

### Token效率 — 量子DSL vs Python

使用`tiktoken` cl100k_base分词器测量。

| 程序 | Python Token数 | DSL Token数 | 节省 |
|------|:------------:|:----------:|:------:|
| search | 173 | 57 | **67%** |
| classify | 129 | 39 | **70%** |
| pipeline | 210 | 73 | **65%** |
| multi_input | 191 | 74 | **61%** |
| reobserve_pattern | 131 | 62 | **53%** |
| **合计** | **834** | **305** | **63%** |

### Token效率 — 基础DSL vs Python vs C#

| 程序 | Python | C# | Axol DSL | vs Python | vs C# |
|------|:------:|:--:|:--------:|:---------:|:-----:|
| Counter | 32 | 61 | 33 | -3% | 46% |
| State Machine | 67 | 147 | 48 | 28% | 67% |
| Combat Pipeline | 145 | 203 | 66 | 55% | 68% |
| 100-State Automaton | 739 | 869 | 636 | 14% | 27% |

### 精度

| 指标 | 值 |
|------|-----|
| Lyapunov估计平均误差 | **0.0002** |
| Omega公式误差 | **0**（精确） |
| Phi公式误差 | **0**（精确） |
| 组合规则 | **全部通过** |
| 观测一致性（50次重复） | **1.0000** |

### 速度

| 操作 | 时间 |
|------|------|
| DSL解析（简单） | ~25 us |
| DSL解析（完整程序） | ~62 us |
| 成本估算 | ~40 us |
| 单次观测 | ~300 us |
| Weave（2节点，dim=8） | ~14 ms |
| Reobserve x10 | ~14 ms |
| **完整管道**（解析→织造→观测，dim=16） | **~17 ms** |

### 扩展性

| 节点数 | 维度 | 织造时间 |
|:-----:|:----:|:------:|
| 1 | 8 | 9 ms |
| 4 | 8 | 25 ms |
| 16 | 8 | 108 ms |
| 2 | 4 | 12 ms |
| 2 | 64 | 39 ms |

完整基准数据：[QUANTUM_PERFORMANCE_REPORT.md](QUANTUM_PERFORMANCE_REPORT.md) | [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md)

---

## 基础层

量子模块（`axol/quantum/`）构建在基础层（`axol/core/`）之上，不对其进行修改。基础层提供数学引擎：向量类型、矩阵操作、程序执行、加密和优化。

### 9个原语操作

| 操作 | 安全等级 | 数学基础 | 描述 |
|------|:--------:|---------|------|
| `transform` | **E** | 矩阵乘法：`v @ M` | 线性状态变换 |
| `gate` | **E** | Hadamard积：`v * g` | 条件遮蔽 |
| `merge` | **E** | 加权和：`sum(v_i * w_i)` | 向量组合 |
| `distance` | **E** | L2 / cosine / dot | 相似度度量 |
| `route` | **E** | `argmax(v @ R)` | 离散分支 |
| `step` | **P** | `where(v >= t, 1, 0)` | 阈值转二值门 |
| `branch` | **P** | `where(g, then, else)` | 条件向量选择 |
| `clamp` | **P** | `clip(v, min, max)` | 值域限制 |
| `map` | **P** | `f(v)` 逐元素 | 非线性激活 |

5个**E**（加密）操作可通过相似变换在加密数据上运行。4个**P**（明文）操作增加非线性表达能力。

### 矩阵加密（Shadow AI）

Axol中的所有计算都归约为矩阵乘法（`v @ M`）。这使得**相似变换加密**成为可能：

```
M' = K^(-1) @ M @ K     （加密后的运算矩阵）
state' = state @ K       （加密后的初始状态）
result = result' @ K^(-1)（解密后的输出）
```

- 加密程序在加密域中正确运行
- 所有业务逻辑被隐藏——矩阵表现为随机噪声
- 这不是混淆——而是密码学变换
- 全部5个E操作已验证（`tests/test_encryption.py`中21项测试）

### 明文操作与安全分类

每个操作都携带一个`SecurityLevel`（E或P）。内置分析器可报告加密覆盖率：

```python
from axol.core import parse, analyze

result = analyze(program)
print(result.summary())
# Program: damage_calc
# Transitions: 3 total, 1 encrypted (E), 2 plaintext (P)
# Coverage: 33.3%
```

### 编译器优化器

三遍优化：变换融合、死状态消除、常量折叠。

```python
from axol.core import parse, optimize, run_program

program = parse(source)
optimized = optimize(program)
result = run_program(optimized)
```

### GPU后端

可插拔数组后端：`numpy`（默认）、`cupy`（NVIDIA GPU）、`jax`。

```python
from axol.core import set_backend
set_backend("cupy")   # NVIDIA GPU
set_backend("jax")    # Google JAX
```

### 模块系统

可复用、可组合的程序，支持schema、导入和子模块执行。

```
@main
import damage_calc from "damage_calc.axol"
s atk=[50] def_val=[10]
: calc=use(damage_calc;in=atk,def_val;out=dmg)
```

### 量子干涉（Phase 6）

Phase 6引入了量子干涉——Grover搜索、量子行走——实现量子程序的**100%加密覆盖率**。Hadamard、Oracle和Diffusion生成`TransMatrix`对象，通过`TransformOp`（E-class）使用，因此现有的优化器和加密模块自动兼容。

```
@grover_4
s state=[0.5 0.5 0.5 0.5]
: oracle=oracle(state;marked=[3];n=4)
: diffuse=diffuse(state;n=4)
? found state[3]>=0.9
```

### 客户端-服务器架构

在客户端加密，在不受信任的服务器上计算：

```
Client (key)              Server (no key)
  Program ─── encrypt ──► Encrypted Program
  pad_and_encrypt()       run_program() on noise
                    ◄──── Encrypted Result
  decrypt_result()
  ──► Result
```

关键组件：`KeyFamily(seed)`、`fn_to_matrix()`、`pad_and_encrypt()`、`AxolClient` SDK。

---

## 架构

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
                                   │ 复用
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
                    │  Tool-Use API    FastAPI + HTML/JS调试器     │
                    └─────────────────────────────────────────────┘
```

### 内部引擎复用

量子模块在不修改`axol/core`的情况下复用它：

| 量子概念 | 核心实现 |
|----------|---------|
| 吸引子振幅/轨迹 | `FloatVec` |
| 吸引子相关矩阵 | `TransMatrix` |
| Tapestry内部执行 | `Program` + `run_program()` |
| Born规则概率 | `operations.measure()` |
| Weave变换构建 | `TransformOp`、`MergeOp` |
| 吸引子探索扩散 | `hadamard_matrix()`、`diffusion_matrix()` |

---

## 快速开始

### 安装

```bash
git clone https://github.com/your-username/AXOL.git
cd AXOL
pip install -e ".[dev]"
```

### 环境要求

- Python 3.11+
- NumPy >= 1.24.0
- pytest >= 7.4.0（开发用）
- tiktoken >= 0.5.0（开发用，token分析）
- fastapi >= 0.100.0、uvicorn >= 0.23.0（可选，Web前端）
- cupy-cuda12x >= 12.0.0（可选，GPU加速）
- jax[cpu] >= 0.4.0（可选，JAX后端）

### Hello World — 量子DSL（Declare → Weave → Observe）

```python
from axol.quantum import (
    DeclarationBuilder, RelationKind,
    weave, observe, parse_quantum,
)
from axol.core.types import FloatVec

# 方式1：Python API
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

# 方式2：DSL
program = parse_quantum("""
entangle hello(x: float[4]) @ Omega(0.9) Phi(0.8) {
    y <~> transform(x)
}
""")
```

### Hello World — 基础DSL（向量操作）

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

## API参考

### 量子模块（`axol.quantum`）

```python
# 声明
DeclarationBuilder(name)           # 用于构建声明的Fluent API
  .input(name, dim, labels?)       # 添加输入
  .output(name)                    # 标记输出
  .relate(target, sources, kind)   # 添加关系
  .quality(omega, phi)             # 设定质量目标
  .build() -> EntangleDeclaration

# 织造
weave(declaration, encrypt?, seed?, optimize?) -> Tapestry

# 观测
observe(tapestry, inputs, seed?) -> Observation
reobserve(tapestry, inputs, count, seed?) -> Observation

# DSL
parse_quantum(source) -> QuantumProgram

# Lyapunov
estimate_lyapunov(trajectory_matrix, steps?) -> float
lyapunov_spectrum(trajectory_matrix, dim, steps?) -> list[float]
omega_from_lyapunov(lyapunov) -> float

# 分形
estimate_fractal_dim(attractor_points, method?, phase_space_dim?) -> float
phi_from_fractal(fractal_dim, phase_space_dim) -> float
phi_from_entropy(probs) -> float

# 组合
compose_serial(omega_a, phi_a, lambda_a, d_a, ...) -> tuple
compose_parallel(omega_a, phi_a, lambda_a, d_a, ...) -> tuple
can_reuse_after_observe(lyapunov) -> bool

# 成本
estimate_cost(declaration) -> CostEstimate
```

### 核心类型

| 类型 | 描述 |
|------|------|
| `SuperposedState` | 具有振幅、标签和Born规则概率的命名状态 |
| `Attractor` | 具有Lyapunov谱、分形维度和轨迹矩阵的奇异吸引子 |
| `Tapestry` | 由`TapestryNode`组成的图，包含全局吸引子和织造报告 |
| `Observation` | 坍缩结果，包含值、Omega、Phi和概率 |
| `WeaverReport` | 目标vs达成的质量、可行性、成本明细 |
| `CostEstimate` | 每节点成本、关键路径、最大可达Omega/Phi |
| `FloatVec` | 32位浮点向量 |
| `TransMatrix` | M x N float32矩阵 |
| `StateBundle` | 命名向量集合 |
| `Program` | 可执行转移序列 |

### 基础模块（`axol.core`）

```python
parse(source) -> Program
run_program(program) -> ExecutionResult
optimize(program) -> Program
set_backend(name)    # "numpy" | "cupy" | "jax"
analyze(program) -> AnalysisResult
dispatch(request) -> dict    # Tool-Use API
```

---

## 示例

### 1. Declare → Weave → Observe（完整管道）

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

### 2. 量子DSL往返

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

### 3. 状态机（基础DSL）

```
@state_machine
s state=onehot(0,3)
: advance=transform(state;M=[0 1 0;0 0 1;0 0 1])
? done state[2]>=1
```

### 4. Grover搜索（量子干涉）

```
@grover_4
s state=[0.5 0.5 0.5 0.5]
: oracle=oracle(state;marked=[3];n=4)
: diffuse=diffuse(state;n=4)
? found state[3]>=0.9
```

### 5. 加密执行

```python
from axol.core import parse, run_program
from axol.core.encryption import encrypt_program, decrypt_state

program = parse("@test\ns v=[1 0 0]\n: t=transform(v;M=[0 1 0;0 0 1;1 0 0])")
encrypted, key = encrypt_program(program)
result = run_program(encrypted)
decrypted = decrypt_state(result.final_state, key)
```

---

## 测试套件

```bash
# 完整测试套件（545项测试）
pytest tests/ -v

# 量子模块测试（101项）
pytest tests/test_quantum_*.py tests/test_lyapunov.py tests/test_fractal.py tests/test_compose.py -v

# 性能基准测试（生成报告）
pytest tests/test_quantum_performance.py -v -s
pytest tests/test_performance_report.py -v -s

# 核心测试
pytest tests/test_types.py tests/test_operations.py tests/test_program.py tests/test_dsl.py -v

# 加密测试（21项）
pytest tests/test_encryption.py -v -s

# 量子干涉测试（37项）
pytest tests/test_quantum.py -v -s

# API + 服务器测试
pytest tests/test_api.py tests/test_server.py -v

# 启动Web前端
python -m axol.server   # http://localhost:8080
```

当前：**545项测试通过**，0项失败，4项跳过（cupy/jax未安装）。

---

## 路线图

- [x] Phase 1：类型系统（7种向量类型 + StateBundle）+ 5个原语操作 + 执行引擎
- [x] Phase 2：DSL解析器 + 稀疏矩阵表示法 + Token基准测试 + 加密概念验证
- [x] Phase 3：编译器优化器（融合、消除、折叠）+ GPU后端
- [x] Phase 4：Tool-Use API + 加密模块
- [x] Phase 5：模块系统（注册表、import/use、compose、schema）
- [x] 前端：FastAPI + HTML/JS可视化调试器
- [x] Phase 6：量子干涉（Hadamard/Oracle/Diffusion，100% E-class覆盖率）
- [x] Phase 7：KeyFamily、矩形加密、fn_to_matrix、填充、分支编译、AxolClient SDK
- [x] Phase 8：混沌理论量子模块 — Declare → Weave → Observe管道
- [x] Phase 8：Lyapunov指数估计（Benettin QR）+ Omega = 1/(1+max(lambda,0))
- [x] Phase 8：分形维度估计（盒计数/相关维度）+ Phi = 1/(1+D/D_max)
- [x] Phase 8：织造器、观测所、组合规则、成本估算、DSL解析器
- [x] Phase 8：101项新测试（总计545项，0项失败）
- [ ] Phase 9：复数振幅（a+bi）支持Shor、QPE、QFT — 完整相位干涉
- [ ] Phase 10：跨多节点的分布式Tapestry织造
- [ ] Phase 11：自适应质量 — 观测期间动态Omega/Phi调整

---

## 许可证

MIT License。详情请参阅[LICENSE](LICENSE)。
