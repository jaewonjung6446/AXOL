<p align="center">
  <h1 align="center">AXOL</h1>
  <p align="center">
    <strong>AI原生向量编程语言</strong>
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

> **警告：本项目处于早期实验阶段。**
> API、DSL语法和内部架构可能会在没有通知的情况下发生重大变更。不建议在生产环境中使用。欢迎贡献和反馈。

---

## 什么是Axol？

**Axol**是一种从零开始为**AI智能体**设计的领域特定语言（DSL），使其能够以比传统编程语言**更少的token**来读取、编写和推理程序。

Axol不使用传统的控制流（if/else、for循环、函数调用），而是将所有计算表示为不可变向量束上的**向量变换**和**状态转移**。这一设计源于一个简单的观察：**LLM按token计费**，而现有的编程语言是为人类可读性而非token效率设计的。

### 核心特性

- 相比Python节省**30~50%的token**
- 相比C#节省**48~75%的token**
- **5个原语操作**覆盖所有计算：`transform`、`gate`、`merge`、`distance`、`route`
- **稀疏矩阵表示法**：从密集表示的O(N^2)降至O(N)
- 支持完整状态追踪的**确定性执行**
- **NumPy后端**实现大规模向量运算500倍以上加速

---

## 目录

- [理论背景](#理论背景)
- [架构](#架构)
- [快速开始](#快速开始)
- [DSL语法](#dsl语法)
- [Token成本对比](#token成本对比)
- [运行时性能](#运行时性能)
- [API参考](#api参考)
- [示例](#示例)
- [测试](#测试)
- [路线图](#路线图)

---

## 理论背景

### Token经济问题

现代AI系统（GPT-4、Claude等）在**token经济**下运行。输入输出的每个字符都消耗token，直接影响成本和延迟。当AI智能体编写或读取代码时，编程语言的冗余度直接影响：

1. **成本** - 更多token = 更高的API费用
2. **延迟** - 更多token = 更慢的响应速度
3. **上下文窗口** - 更多token = 更少的其他信息空间
4. **推理准确度** - 压缩的表示减少噪声

### 为什么选择向量计算？

传统编程语言通过**控制流**（分支、循环、递归）表达逻辑。这对人类来说很直观，但对AI来说效率低下：

```python
# Python: 67个token
TRANSITIONS = {"IDLE": "RUNNING", "RUNNING": "DONE", "DONE": "DONE"}
def state_machine():
    state = "IDLE"
    steps = 0
    while state != "DONE":
        state = TRANSITIONS[state]
        steps += 1
    return state, steps
```

用向量变换表达相同的逻辑：

```
# Axol DSL: 48个token（节省28%）
@state_machine
s state=onehot(0,3)
: advance=transform(state;M=[0 1 0;0 0 1;0 0 1])
? done state[2]>=1
```

状态机的转移表变成了**矩阵**，状态推进变成了**矩阵乘法**。AI不需要推理字符串比较、字典查找或循环条件，只需处理单个矩阵运算。

### 五个原语操作

Axol将所有计算归约为五个操作，每个操作对应一个基本的线性代数概念：

| 操作 | 数学基础 | 描述 |
|------|---------|------|
| `transform` | 矩阵乘法：`v @ M` | 线性状态变换 |
| `gate` | Hadamard积：`v * g` | 条件遮蔽 |
| `merge` | 加权和：`sum(v_i * w_i)` | 向量组合 |
| `distance` | L2 / 余弦 / 点积 | 相似度度量 |
| `route` | `argmax(v @ R)` | 离散分支 |

这五个操作可以表达：
- 状态机 (transform)
- 条件逻辑 (gate)
- 累积/聚合 (merge)
- 相似度搜索 (distance)
- 决策制定 (route)

### 稀疏矩阵表示法

对于大型状态空间，密集矩阵表示在token上是O(N^2)的。Axol的稀疏表示法将其降至O(N)：

```
# 密集：O(N^2)个token - N=100时不实用
M=[0 1 0 0 ... 0; 0 0 1 0 ... 0; ...]

# 稀疏：O(N)个token - 线性扩展
M=sparse(100x100;0,1=1 1,2=1 ... 98,99=1 99,99=1)
```

| N | Python | C# | Axol DSL | DSL/Python | DSL/C# |
|---|--------|-----|----------|------------|--------|
| 5 | 74 | 109 | 66 | 0.89x | 0.61x |
| 25 | 214 | 269 | 186 | 0.87x | 0.69x |
| 100 | 739 | 869 | 636 | 0.86x | 0.73x |
| 200 | 1,439 | 1,669 | 1,236 | 0.86x | 0.74x |

---

## 架构

```
                    +-----------+
  .axol源码 ------->| 解析器    |----> Program对象
                    | (dsl.py)  |         |
                    +-----------+         |
                                          v
                    +-----------+    +-----------+
                    | 验证器    |<---| 执行引擎   |
                    |(verify.py)|    |(program.py)|
                    +-----------+    +-----------+
                                          |
                         使用             |
                    +-----------+         |
                    | 运算模块   |<--------+
                    | (ops.py)  |
                    +-----------+
                         |
                    +-----------+
                    | 类型系统   |
                    |(types.py) |
                    +-----------+
```

### 模块概览

| 模块 | 描述 |
|------|------|
| `axol.core.types` | 7种向量类型 + `StateBundle` |
| `axol.core.operations` | 5个原语操作 |
| `axol.core.program` | 执行引擎：`Program`、`Transition`、`run_program` |
| `axol.core.verify` | 状态验证（exact/cosine/euclidean匹配） |
| `axol.core.dsl` | DSL解析器：`parse(source) -> Program` |

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

print(f"最终计数: {result.final_state['count'].to_list()}")  # [5.0]
print(f"步数: {result.steps_executed}")
print(f"终止条件: {result.terminated_by}")  # terminal_condition
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
print(f"衰减后HP: {result.final_state['hp'].to_list()}")  # [80.0]
```

---

## DSL语法

### 程序结构

```
@program_name              # 头部：程序名称
s key1=[values] key2=...   # 状态：初始向量声明
: name=op(args)->out       # 转移：操作定义
? terminal condition       # 终端：循环退出条件（可选）
```

### 状态声明

```
s hp=[100]                          # 单浮点向量
s pos=[1.5 2.0 -3.0]               # 多元素向量
s state=onehot(0,5)                 # 独热向量：索引0，大小5
s buffer=zeros(10)                  # 大小10的零向量
s mask=ones(3)                      # 大小3的全1向量
s hp=[100] mp=[50] stamina=[75]     # 一行声明多个向量
```

### 操作

```
# transform：矩阵乘法
: decay=transform(hp;M=[0.8])
: advance=transform(state;M=[0 1 0;0 0 1;0 0 1])

# gate：逐元素遮蔽
: masked=gate(values;g=mask)

# merge：向量加权和
: total=merge(a b c;w=[1 1 1])->result

# distance：相似度度量
: dist=distance(pos1 pos2)
: sim=distance(vec1 vec2;metric=cosine)

# route：argmax路由
: choice=route(scores;R=[1 0 0;0 1 0;0 0 1])
```

### 矩阵格式

```
# 密集：行用;分隔
M=[0.8]                               # 1x1
M=[1 0;0 1]                           # 2x2 单位矩阵
M=[0 1 0;0 0 1;0 0 1]                # 3x3 移位矩阵

# 稀疏：只标记非零元素
M=sparse(100x100;0,1=1 1,2=1 99,99=1)
```

### 终端条件

```
? done count>=5              # count[0] >= 5时退出
? finished state[2]>=1       # state[2] >= 1时退出（索引访问）
? end hp<=0                  # hp[0] <= 0时退出
```

没有`?`行时，程序以**管道模式**运行（所有转移执行一次）。

---

## Token成本对比

使用`tiktoken` cl100k_base分词器测量（GPT-4 / Claude使用）。

### Python vs C# vs Axol DSL

| 程序 | Python | C# | Axol DSL | vs Python | vs C# |
|------|--------|----|----------|-----------|-------|
| Counter | 32 | 61 | 33 | -3.1% | 45.9% |
| State Machine | 67 | 147 | 48 | 28.4% | 67.3% |
| HP Decay | 51 | 134 | 51 | 0.0% | 61.9% |
| Combat | 145 | 203 | 66 | 54.5% | 67.5% |
| Data Heavy | 159 | 227 | 67 | 57.9% | 70.5% |
| Pattern Match | 151 | 197 | 49 | 67.5% | 75.1% |
| 100-State Auto | 739 | 869 | 636 | 13.9% | 26.8% |
| **总计** | **1,344** | **1,838** | **950** | **29.3%** | **48.3%** |

### 关键发现

1. **简单程序**（counter、hp_decay）：DSL与Python相当。
2. **结构化程序**（combat、data_heavy、pattern_match）：DSL相比Python节省**50~68%**，相比C#节省**67~75%**。向量表示消除了类定义、控制流和样板代码。
3. **大型状态空间**（100+状态）：稀疏矩阵表示法相比Python稳定节省**约38%**，相比C#节省**约27%**，实现O(N)扩展。

---

## 运行时性能

Axol使用NumPy作为计算后端。

### 小型向量（dim < 100）

| 维度 | Python循环 | Axol（NumPy） | 优势 |
|------|-----------|--------------|------|
| dim=4 | ~6 us | ~11 us | Python 2x |
| dim=100 | ~14 us | ~20 us | Python 1.4x |

### 大型向量（dim >= 1000）

| 维度 | Python循环 | Axol（NumPy） | 优势 |
|------|-----------|--------------|------|
| dim=1,000（矩阵乘） | ~129 ms | ~0.2 ms | **Axol 573x** |
| dim=10,000（矩阵乘） | ~14,815 ms | ~381 ms | **Axol 39x** |

在大规模向量运算（矩阵乘法）中，Axol的NumPy后端比纯Python循环快**数百倍**。

---

## 示例

### 1. 计数器（0 -> 5）

```
@counter
s count=[0] one=[1]
: increment=merge(count one;w=[1 1])->count
? done count>=5
```

### 2. 状态机（IDLE -> RUNNING -> DONE）

```
@state_machine
s state=onehot(0,3)
: advance=transform(state;M=[0 1 0;0 0 1;0 0 1])
? done state[2]>=1
```

### 3. HP衰减（100 x 0.8^3 = 51.2）

```
@hp_decay
s hp=[100] round=[0] one=[1]
: decay=transform(hp;M=[0.8])
: tick=merge(round one;w=[1 1])->round
? done round>=3
```

### 4. 战斗伤害（管道模式）

```
@combat
s atk=[50] def_val=[20] flag=[1]
: scale=transform(atk;M=[1.5])->scaled
: block=gate(def_val;g=flag)
: combine=merge(scaled def_val;w=[1 -1])->damage
```

---

## 测试

```bash
# 运行全部测试（149项）
pytest tests/ -v

# DSL解析器测试
pytest tests/test_dsl.py -v

# Token成本对比
pytest tests/test_token_cost.py -v -s

# 三语言基准测试（Python vs C# vs Axol）
pytest tests/test_benchmark_trilingual.py -v -s
```

当前测试数量：**149项**，全部通过。

---

## 路线图

- [x] Phase 1：类型系统（7种向量类型 + StateBundle）
- [x] Phase 1：5个原语操作
- [x] Phase 1：程序执行引擎（管道 + 循环模式）
- [x] Phase 1：状态验证框架
- [x] Phase 2：DSL解析器（完整语法支持）
- [x] Phase 2：稀疏矩阵表示法
- [x] Phase 2：Token成本基准测试（Python、C#、Axol）
- [ ] Phase 3：编译器优化（运算融合、死状态消除）
- [ ] Phase 3：GPU后端（CuPy / JAX）
- [ ] Phase 4：AI智能体集成（tool-use API）
- [ ] Phase 4：状态追踪可视化调试器
- [ ] Phase 5：多程序组合与模块系统

---

## 许可证

MIT License。详情请参阅[LICENSE](LICENSE)。
