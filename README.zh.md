<p align="center">
  <h1 align="center">AXOL</h1>
  <p align="center">
    <strong>Token高效向量编程语言</strong>
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
- **9个原语操作**覆盖所有计算：`transform`、`gate`、`merge`、`distance`、`route`（加密）+ `step`、`branch`、`clamp`、`map`（明文）
- **稀疏矩阵表示法**：从密集表示的O(N^2)降至O(N)
- 支持完整状态追踪的**确定性执行**
- **NumPy后端**支持大规模向量运算（大维度下比纯Python循环更快）
- **E/P安全分类** - 每个操作被分类为加密（E）或明文（P），通过内置分析器可视化加密覆盖率与表达能力的权衡
- **矩阵级加密** - 秘密密钥矩阵使程序在密码学层面不可读，从根本上解决影子AI问题

---

## 目录

- [理论背景](#理论背景)
- [影子AI与矩阵加密](#影子ai与矩阵加密)
  - [加密证明：全部5个操作验证完成](#加密证明全部5个操作验证完成)
- [明文操作与安全分类](#明文操作与安全分类)
- [架构](#架构)
- [快速开始](#快速开始)
- [DSL语法](#dsl语法)
- [编译器优化](#编译器优化)
- [GPU后端](#gpu后端)
- [模块系统](#模块系统)
- [Tool-Use API](#tool-use-api)
- [Web前端](#web前端)
- [Token成本对比](#token成本对比)
- [运行时性能](#运行时性能)
- [性能基准测试](#性能基准测试)
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

### 九个原语操作

Axol提供九个原语操作。前五个为**加密（E）**操作——可以在加密数据上运行。后四个为**明文（P）**操作——需要明文数据，但增加了非线性表达能力：

| 操作 | 安全等级 | 数学基础 | 描述 |
|------|:--------:|---------|------|
| `transform` | **E** | 矩阵乘法：`v @ M` | 线性状态变换 |
| `gate` | **E** | Hadamard积：`v * g` | 条件遮蔽（0/1） |
| `merge` | **E** | 加权和：`sum(v_i * w_i)` | 向量组合 |
| `distance` | **E** | L2 / 余弦 / 点积 | 相似度度量 |
| `route` | **E** | `argmax(v @ R)` | 离散分支 |
| `step` | **P** | `where(v >= t, 1, 0)` | 阈值转二值门 |
| `branch` | **P** | `where(g, then, else)` | 条件向量选择 |
| `clamp` | **P** | `clip(v, min, max)` | 值域限制 |
| `map` | **P** | `f(v)` 逐元素 | 非线性激活（relu、sigmoid、abs、neg、square、sqrt） |

五个E操作构成加密计算的**线性代数基础**：
- 状态机 (transform)
- 条件逻辑 (gate)
- 累积/聚合 (merge)
- 相似度搜索 (distance)
- 决策制定 (route)

四个P操作为AI/ML工作负载增加**非线性表达能力**：
- 激活函数（map: relu、sigmoid）
- 阈值决策（step + branch）
- 值归一化（clamp）

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

## 影子AI与矩阵加密

### 影子AI问题

**影子AI（Shadow AI）**指未经授权的AI智能体泄露、复制或逆向工程专有业务逻辑的风险。随着AI智能体越来越多地自主编写和执行代码，传统源代码成为关键的攻击面：

- AI智能体的提示词和生成代码可通过**提示注入攻击提取**
- Python/C#/JavaScript代码**设计上就是人类可读的** - 混淆是可逆的
- 嵌入代码中的专有算法、决策规则和商业秘密**以明文形式暴露**
- 传统混淆技术（变量重命名、控制流扁平化）只是略微提高门槛 - 逻辑结构完好，可以恢复

### Axol的解决方案：矩阵级加密

由于**Axol中的所有计算都归约为矩阵乘法**（`v @ M`），一种在传统编程语言中不可能实现的数学性质变得可用：**相似变换（similarity transformation）加密**。

给定一个秘密可逆密钥矩阵**K**，任何Axol程序都可以被加密：

```
原始程序：     state  -->  M  -->  new_state
加密程序：     state' -->  M' -->  new_state'

其中：
  M' = K^(-1) @ M @ K          （加密后的运算矩阵）
  state' = state @ K            （加密后的初始状态）
  result  = result' @ K^(-1)    （解密后的最终输出）
```

这不是混淆，而是**密码学变换**。加密后的程序：

1. **在加密域中正常执行**（矩阵代数保持共轭变换）
2. **生成加密输出** - 没有K^(-1)无法解码
3. **隐藏所有业务逻辑** - 矩阵M'在没有K的情况下与M在数学上无关
4. **抵抗逆向工程** - 从M'恢复K是一个随N增大而难度增加的矩阵分解问题。虽然一般情况下不存在已知的多项式时间算法，但正式的密码学困难性证明仍是一个正在研究的领域

### 具体示例

```
# 原始：状态机转移矩阵（业务逻辑可见）
M = [0 1 0]    # IDLE -> RUNNING
    [0 0 1]    # RUNNING -> DONE
    [0 0 1]    # DONE -> DONE（吸收状态）

# 用秘密密钥K加密后：
M' = [0.73  -0.21   0.48]    # 没有K毫无意义
     [0.15   0.89  -0.04]    # 无法推断状态机结构
     [0.52   0.33   0.15]    # 看起来像随机噪声
```

加密程序仍然正常执行（矩阵代数保证`K^(-1)(KvM)K = vM`），但DSL文本中**只包含加密后的矩阵**。即使整个`.axol`文件泄露：

- 状态名不可见（向量已加密）
- 转移逻辑不可见（矩阵已加密）
- 终止条件无意义（阈值在加密值上操作）

### 为什么传统语言无法实现

| 属性 | Python/C#/JS | FHE | Axol |
|------|-------------|-----|------|
| 代码语义 | 明文控制流 | 加密（任意计算） | 矩阵乘法 |
| 混淆 | 可逆（变量重命名、流程扁平化） | 不适用 | 不适用 |
| 加密 | 不可能（必须可解析） | 完全（任意计算） | 仅线性操作（9个中5个） |
| 性能开销 | 不适用 | 1000-10000倍 | 约0%（管道模式） |
| 复杂度 | 不适用 | 非常高 | 低（仅密钥矩阵） |
| 代码泄露时 | 全部逻辑暴露 | 已加密 | 看起来像随机数字 |
| 密钥分离 | 不可能 | 必需 | 密钥矩阵单独存储（HSM、安全飞地） |
| 加密后正确性 | 不适用 | 数学保证 | 数学保证 |

### 安全架构

```
  [开发者]                       [部署环境]
     |                              |
  原始.axol                    加密后的.axol
  （可读逻辑）                  （加密矩阵）
     |                              |
     +--- K（秘密密钥）---------->|
     |    存储在HSM/安全飞地      |
     v                              v
  encrypt(M, K) = K^(-1)MK     run_program(加密程序)
                                     |
                                加密输出
                                     |
                                decrypt(output, K^(-1))
                                     |
                                实际结果
```

秘密密钥矩阵K可以：
- 存储在**硬件安全模块（HSM）**中
- 通过**密钥管理服务（KMS）**管理
- 在不改变程序结构的情况下定期轮换
- 按部署环境（dev/staging/prod）使用不同密钥

Axol为基于矩阵的计算提供了全同态加密（FHE）的轻量级替代方案。与FHE（支持任意计算但开销很高）不同，Axol的相似变换高效但仅限于线性操作。这种权衡使其在5个加密操作足够的特定用例中具有实用性。

### 加密证明：全部5个操作验证完成

Axol全部5个操作的加密兼容性已**经过数学证明和测试**（`tests/test_encryption.py`，21项测试）：

| 操作 | 加密方法 | 密钥约束 | 状态 |
|------|---------|---------|------|
| `transform` | `M' = K^(-1) M K`（相似变换） | 任意可逆矩阵K | **已证明** |
| `gate` | 转换为`diag(g)`矩阵后同transform | 任意可逆矩阵K | **已证明** |
| `merge` | 线性性：`w*(v@K) = (wv)@K`（自动兼容） | 任意可逆矩阵K | **已证明** |
| `distance` | `\|\|v@K\|\| = \|\|v\|\|`（正交矩阵保范） | 正交矩阵K | **已证明** |
| `route` | `R' = K^(-1) R`（仅左乘） | 任意可逆矩阵K | **已证明** |

**复杂多操作程序同样验证通过：**

- HP衰减（transform + merge循环）- 加密/解密结果一致
- 3状态FSM（链式transform）- 加密域中状态转移正确
- 战斗流水线（transform + gate + merge）- 3个操作链式执行，误差 < 0.001
- 20状态自动机（稀疏矩阵，19步）- 加密执行结果与原始一致
- 50x50大规模矩阵 - float32精度保持

**测试证明的安全属性：**

- 加密后的矩阵呈现为随机噪声（稀疏 -> 稠密，无可见结构）
- 不同密钥产生完全不同的加密结果
- 100个随机密钥暴力破解无法恢复原始矩阵
- OneHot向量结构在加密后完全隐藏

---

## 明文操作与安全分类

### 为什么需要明文操作？

原始的5个加密操作是**线性的**——它们只能表达线性变换。许多实际的AI/ML工作负载需要**非线性**操作（激活函数、条件分支、值截断）。新增的4个明文操作填补了这一空白。

### SecurityLevel枚举

每个操作都携带一个`SecurityLevel`：

```python
from axol.core import SecurityLevel

SecurityLevel.ENCRYPTED  # "E" - 可在加密数据上运行
SecurityLevel.PLAINTEXT  # "P" - 需要明文数据
```

### 加密覆盖率分析器

内置分析器可报告程序中能以加密方式运行的操作比例：

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
# Encryptable keys: (仅E操作访问的键)
# Plaintext keys: (任何P操作访问的键)
```

### 安全性与表达能力的权衡

添加P操作会增加表达能力，但会降低加密覆盖率：

| 程序类型 | 加密覆盖率 | 表达能力 |
|---------|-----------|---------|
| 仅E操作 | 100% | 仅线性 |
| E+P混合 | 30-70%（典型） | 完全（非线性） |
| 仅P操作 | 0% | 完全（非线性） |

需要非线性操作（激活函数、条件分支）的程序必须接受部分加密覆盖率。使用内置分析器来衡量程序的覆盖率，并识别哪些键需要明文访问。

### 新操作的Token成本（Python vs C# vs Axol DSL）

| 程序 | Python | C# | Axol DSL | vs Python | vs C# |
|------|-------:|---:|--------:|---------:|------:|
| ReLU激活 | 48 | 82 | 28 | 42% | 66% |
| 阈值选择 | 140 | 184 | 80 | 43% | 57% |
| 值截断 | 66 | 95 | 31 | 53% | 67% |
| Sigmoid激活 | 57 | 88 | 28 | 51% | 68% |
| 伤害流水线 | 306 | 326 | 155 | 49% | 53% |
| **总计** | **617** | **775** | **322** | **48%** | **59%** |

### 新操作运行时性能（dim=10,000）

| 操作 | Python循环 | Axol（NumPy） | 加速比 |
|------|----------:|----------:|--------:|
| ReLU | 575 us | 21 us | **27x** |
| Sigmoid | 1.7 ms | 42 us | **40x** |
| Step+Branch | 889 us | 96 us | **9x** |
| Clamp | 937 us | 16 us | **58x** |
| 伤害流水线 | 3.8 ms | 191 us | **20x** |

---

## 架构

```
                                          +-------------+
  .axol源码 -----> 解析器 (dsl.py) -----> | Program     |
                         |                | + optimize()|
                         v                +------+------+
                    模块系统                      |
                    (module.py)                  v
                      - import             +-----------+    +-----------+
                      - use()              |  执行引擎  |--->|  验证器   |
                      - compose()          |(program.py)|    |(verify.py)|
                                           +-----------+    +-----------+
                                                |
                    +-----------+    +----------+----------+
                    |  后端     |<---|    运算模块          |
                    |(backend.py)|    | (operations.py)     |
                    | numpy/cupy|    +---------------------+
                    | /jax      |               |
                    +-----------+    +-----------+----------+
                                    |      类型系统        |
                                    |   (types.py)         |
                    +-----------+   +----------------------+
                    |  加密     |   +-----------+
                    |(encryption|   | 分析器    |
                    |       .py)|   |(analyzer  |
                    +-----------+   |       .py)|
                                    +-----------+
                    +-----------+    +-----------+
                    | Tool API  |    |  服务器   |
                    |(api/)     |    |(server/)  |
                    | dispatch  |    | FastAPI   |
                    | tools     |    | HTML/JS   |
                    +-----------+    +-----------+
```

### 模块概览

| 模块 | 描述 |
|------|------|
| `axol.core.types` | 7种向量类型（`BinaryVec`、`IntVec`、`FloatVec`、`OneHotVec`、`GateVec`、`TransMatrix`）+ `StateBundle` |
| `axol.core.operations` | 9个原语操作：`transform`、`gate`、`merge`、`distance`、`route`、`step`、`branch`、`clamp`、`map_fn` |
| `axol.core.program` | 执行引擎：`Program`、`Transition`、`run_program`、`SecurityLevel`、`StepOp`/`BranchOp`/`ClampOp`/`MapOp` |
| `axol.core.verify` | 状态验证（exact/cosine/euclidean匹配） |
| `axol.core.dsl` | DSL解析器：`parse(source) -> Program`，支持`import`/`use()` |
| `axol.core.optimizer` | 3遍编译器优化：变换融合、死状态消除、常量折叠 |
| `axol.core.backend` | 可插拔数组后端：`numpy`（默认）、`cupy`、`jax` |
| `axol.core.encryption` | 相似变换加密：`encrypt_program`、`decrypt_state`（E/P感知） |
| `axol.core.analyzer` | 加密覆盖率分析器：`analyze(program) -> AnalysisResult`，E/P分类 |
| `axol.core.module` | 模块系统：`Module`、`ModuleRegistry`、`compose()`、schema验证 |
| `axol.api` | AI智能体Tool-Use API：`dispatch(request)`、`get_tool_definitions()` |
| `axol.server` | FastAPI Web服务器 + 原生HTML/JS可视化调试前端 |

---

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/your-username/AXOL.git
cd AXOL

# 安装依赖
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
# --- 加密（E）操作 ---

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

# --- 明文（P）操作 ---

# step：阈值转二值门
: mask=step(scores;t=0.5)->gate_out

# branch：条件向量选择（需要->out_key）
: selected=branch(gate_key;then=high,else=low)->result

# clamp：将值截断到指定范围
: safe=clamp(values;min=0,max=100)

# map：逐元素非线性函数（relu、sigmoid、abs、neg、square、sqrt）
: activated=map(x;fn=relu)
: prob=map(logits;fn=sigmoid)->output
```

### 矩阵格式

```
# 密集：行用;分隔
M=[0.8]                               # 1x1
M=[1 0;0 1]                           # 2x2 单位矩阵
M=[0 1 0;0 0 1;0 0 1]                # 3x3 移位矩阵

# 稀疏：只标记非零元素
M=sparse(100x100;0,1=1 1,2=1 99,99=1) # 100x100，100个非零项
```

### 终端条件

```
? done count>=5              # count[0] >= 5时退出
? finished state[2]>=1       # state[2] >= 1时退出（索引访问）
? end hp<=0                  # hp[0] <= 0时退出
```

没有`?`行时，程序以**管道模式**运行（所有转移执行一次）。

### 注释

```
# 这是注释
@my_program
# 注释可以出现在任何位置
s v=[1 2 3]
: t=transform(v;M=[1 0 0;0 1 0;0 0 1])
```

---

## 编译器优化

`optimize()`应用三个优化遍，减少程序体积并预计算常量：

```python
from axol.core import parse, optimize, run_program

program = parse(source)
optimized = optimize(program)   # 融合 + 消除 + 折叠
result = run_program(optimized)
```

### 第1遍：变换融合

对同一键的连续`TransformOp`进行融合，合并为单次矩阵乘法：

```
# 优化前：2个转移，每次迭代进行2次矩阵乘法
: t1=transform(v;M=[0 1 0;0 0 1;1 0 0])
: t2=transform(v;M=[2 0 0;0 2 0;0 0 2])

# 优化后：1个转移，1次矩阵乘法（M_fused = M1 @ M2）
: t1+t2=transform(v;M_fused)
```

- 不跨越`CustomOp`边界
- 不动点迭代处理3个以上的链
- 含2个transform的管道：**转移数减少50%，执行时间减少45%**

### 第2遍：死状态消除

移除初始状态中从未被任何转移读取的向量：

```
s used=[1 0]  unused=[99 99]   # unused从未被引用
: t=transform(used;M=[...])

# 优化后：unused从初始状态中移除
```

- 对`CustomOp`采取保守策略（保留所有状态）
- `terminal_key`始终被视为"已读取"

### 第3遍：常量折叠

预计算对不可变键（从未被写入的键）的变换：

```
s constant=[1 0 0]
: t=transform(constant;M=[0 1 0;0 0 1;1 0 0])->result

# 优化后：转移被消除，result=[0,1,0]直接存储在初始状态中
```

---

## GPU后端

可插拔数组后端，支持`numpy`（默认）、`cupy`（NVIDIA GPU）和`jax`：

```python
from axol.core import set_backend, get_backend_name

set_backend("numpy")   # 默认 - CPU
set_backend("cupy")    # NVIDIA GPU（需安装cupy）
set_backend("jax")     # Google JAX（需安装jax）
```

安装可选后端：

```bash
pip install axol[gpu]   # cupy-cuda12x
pip install axol[jax]   # jax[cpu]
```

所有现有代码透明兼容——后端切换是全局的，影响所有向量/矩阵运算。

---

## 模块系统

可复用、可组合的程序，支持schema、导入和子模块执行。

### 模块定义

```python
from axol.core.module import Module, ModuleSchema, VecSchema, ModuleRegistry

schema = ModuleSchema(
    inputs=[VecSchema("atk", "float", 1), VecSchema("def_val", "float", 1)],
    outputs=[VecSchema("dmg", "float", 1)],
)
module = Module(name="damage_calc", program=program, schema=schema)
```

### 注册表与文件加载

```python
registry = ModuleRegistry()
registry.load_from_file("damage_calc.axol")
registry.resolve_import("heal", relative_to="main.axol")
```

### DSL导入与使用语法

```
@main
import damage_calc from "damage_calc.axol"
s atk=[50] def_val=[10]
: calc=use(damage_calc;in=atk,def_val;out=dmg)
```

### 程序组合

```python
from axol.core.module import compose
combined = compose(program_a, program_b, name="combined")
```

---

## Tool-Use API

面向AI智能体的JSON可调用接口，用于解析、运行和验证Axol程序：

```python
from axol.api import dispatch

# 解析
result = dispatch({"action": "parse", "source": "@prog\ns v=[1]\n: t=transform(v;M=[2])"})
# -> {"program_name": "prog", "state_keys": ["v"], "transition_count": 1, "has_terminal": false}

# 运行
result = dispatch({"action": "run", "source": "...", "optimize": True})
# -> {"final_state": {"v": [2.0]}, "steps_executed": 1, "terminated_by": "pipeline_end"}

# 逐步检查
result = dispatch({"action": "inspect", "source": "...", "step": 1})

# 列出操作
result = dispatch({"action": "list_ops"})

# 验证预期输出
result = dispatch({"action": "verify", "source": "...", "expected": {"v": [2.0]}})
```

AI智能体的工具定义（JSON Schema）可通过`get_tool_definitions()`获取。

---

## Web前端

FastAPI服务器配合原生HTML/JS可视化调试器：

```bash
pip install axol[server]    # fastapi + uvicorn
python -m axol.server       # http://localhost:8080
```

### 功能

| 面板 | 描述 |
|------|------|
| **DSL编辑器** | 语法编辑，内置示例下拉菜单 |
| **执行** | 运行/优化按钮，结果摘要（步数、时间、终止原因） |
| **追踪查看器** | 逐步状态表，支持上一步/下一步/播放控制 |
| **状态图表** | Chart.js时序图（X=步数，Y=向量值） |
| **加密演示** | 原始vs加密矩阵热力图，加密/运行/解密工作流 |
| **性能** | 优化器前后对比，token成本分析 |

### API端点

```
POST /api/parse       - 解析DSL源码
POST /api/run         - 解析 + 执行 + 完整追踪
POST /api/optimize    - 优化器前后对比
POST /api/encrypt     - 加密程序 + 运行 + 解密
GET  /api/examples    - 内置示例程序
GET  /api/ops         - 操作描述
POST /api/token-cost  - Token计数分析（Axol vs Python vs C#）
POST /api/module/run  - 运行带子模块的程序
```

---

## Token成本对比

使用`tiktoken` cl100k_base分词器测量（GPT-4 / Claude使用）。

> **注意**: Token节省是在自然映射到向量/矩阵操作的程序（状态机、线性变换、加权和）上测量的。对于通用编程任务（字符串处理、I/O、API调用），Axol无法表达。以下比较代表Axol的最佳情况，而非平均情况。

### Python vs Axol DSL

| 程序 | Python | Axol DSL | 节省 |
|------|--------|----------|------|
| 计数器（0->5） | 32 | 33 | -3.1% |
| 状态机（3状态） | 67 | 47 | 29.9% |
| HP衰减（3轮） | 51 | 32 | 37.3% |
| RPG伤害计算 | 130 | 90 | 30.8% |
| 100状态自动机 | 1,034 | 636 | 38.5% |
| **总计** | **1,314** | **838** | **36.2%** |

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

1. **简单程序**（counter、hp_decay）：DSL与Python相当。DSL语法的开销大致等于Python对简单程序的最少语法。
2. **结构化程序**（combat、data_heavy、pattern_match）：DSL相比Python节省**50~68%**，相比C#节省**67~75%**。向量表示消除了类定义、控制流和样板代码。
3. **大型状态空间**（100+状态）：稀疏矩阵表示法相比Python稳定节省**约38%**，相比C#节省**约27%**，实现O(N)扩展（对比O(N^2)）。

---

## 运行时性能

Axol使用NumPy作为计算后端。

> **注意**: 运行时基准测试将纯Python循环与Axol的NumPy后端进行比较。加速主要来自NumPy优化的C/Fortran实现，而非Axol特有的优化。直接使用NumPy的Python代码也能达到类似的速度。

### 小型向量（dim < 100）

| 维度 | Python循环 | Axol（NumPy） | 优势 |
|------|-----------|--------------|------|
| dim=4 | ~6 us | ~11 us | Python 2x |
| dim=100 | ~14 us | ~20 us | Python 1.4x |

对于小型向量，Python原生循环更快，因为NumPy有单次调用开销。这在预期之内且可接受——小程序无论如何都很快。

### 大型向量（dim >= 1000）

| 维度 | Python循环 | Axol（NumPy） | 优势 |
|------|-----------|--------------|------|
| dim=1,000（矩阵乘） | ~129 ms | ~0.2 ms | **573x**（NumPy） |
| dim=10,000（矩阵乘） | ~14,815 ms | ~381 ms | **39x**（NumPy） |

在大规模向量运算（矩阵乘法）中，NumPy优化的C/Fortran BLAS后端（Axol所使用的）比纯Python循环快**数个数量级**。直接使用NumPy的Python代码也能达到类似的加速。

### 使用场景建议

| 场景 | 建议 |
|------|------|
| AI智能体代码生成 | Axol DSL（更少token = 更低成本） |
| 大型状态空间（100+维） | Axol（NumPy加速 + 稀疏表示法） |
| 简单脚本（< 10行） | Python（更少开销） |
| 人类可读的业务逻辑 | Python/C#（熟悉的语法） |

### 局限性

- **有限的领域**: Axol只能表达向量/矩阵计算。不支持字符串处理、I/O、网络和通用编程。
- **无LLM训练数据**: 与Python或JavaScript不同，没有LLM在Axol代码上进行过训练。AI智能体在上下文中没有示例的情况下可能难以生成正确的Axol程序。
- **仅线性操作支持加密**: 9个操作中只有5个支持加密执行。使用非线性操作（step、branch、clamp、map）的程序加密覆盖率会降低。
- **循环模式加密开销**: 循环模式下的加密程序无法评估终端条件，会运行到max_iterations。这在基准测试中导致400倍以上的开销。
- **Token节省是领域特定的**: 30-50%的Token节省适用于以向量/矩阵为主的程序。对于通用任务，Python更简洁。

---

## 性能基准测试

由`pytest tests/test_performance_report.py -v -s`自动生成。完整结果见[PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md)。

> **注意**: 运行时基准测试将纯Python循环与Axol的NumPy后端进行比较。加速主要来自NumPy优化的C/Fortran实现，而非Axol特有的优化。直接使用NumPy的Python代码也能达到类似的速度。

### Token效率（Axol vs Python vs C#）

| 程序 | Axol | Python | C# | vs Python | vs C# |
|------|------|--------|----|-----------|-------|
| 计数器（0->5） | 11 | 45 | 78 | **节省76%** | **节省86%** |
| 3状态FSM | 14 | 52 | 89 | **节省73%** | **节省84%** |
| HP衰减 | 14 | 58 | 95 | **节省76%** | **节省85%** |
| 战斗流水线 | 14 | 55 | 92 | **节省75%** | **节省85%** |
| 矩阵链 | 21 | 60 | 98 | **节省65%** | **节省79%** |

平均：比Python**少74%的token**，比C#**少85%的token**。

### 各维度执行时间

| 维度 | 平均时间 |
|------|---------|
| 4 | 0.25 ms |
| 100 | 0.17 ms |
| 1,000 | 1.41 ms |

### 优化器效果

| 程序 | 优化前 | 优化后 | 时间减少 |
|------|--------|-------|---------|
| 管道（2个transform） | 2个转移 | 1个转移 | **-45%** |
| 计数器（循环） | 2个转移 | 2个转移 | - |
| FSM（循环） | 2个转移 | 2个转移 | - |

变换融合对含有连续矩阵运算的管道程序最为有效。

### 加密开销

| 程序 | 明文 | 加密 | 开销 |
|------|------|------|------|
| 管道（1次） | 0.12 ms | 0.12 ms | **约0%** |
| 3状态FSM（循环） | 0.62 ms | 276.8 ms | +44,633% |

管道模式：开销可忽略。循环模式：开销较高，因为加密的终端条件无法触发提前退出，导致执行运行至`max_iterations`。

### 扩展性（N状态自动机）

| 状态数 | Token数 | 执行时间 |
|--------|---------|---------|
| 5 | 28 | 1.6 ms |
| 20 | 388 | 4.3 ms |
| 50 | 2,458 | 12.9 ms |
| 100 | 9,908 | 27.9 ms |
| 200 | 39,808 | 59.2 ms |

得益于稀疏矩阵表示法，Token数呈**O(N)**增长（对比Python/C#的O(N^2)）。执行时间因矩阵乘法呈约O(N^2)增长，但200状态的程序仍在60ms以内。

---

## API参考

### `parse(source, registry=None, source_path=None) -> Program`

将Axol DSL源文本解析为可执行的`Program`对象。

```python
from axol.core import parse
program = parse("@test\ns v=[1 2 3]\n: t=transform(v;M=[1 0 0;0 1 0;0 0 1])")

# 带模块注册表，支持import/use
from axol.core.module import ModuleRegistry
registry = ModuleRegistry()
program = parse(source, registry=registry, source_path="main.axol")
```

### `run_program(program: Program) -> ExecutionResult`

执行程序并返回结果。

```python
from axol.core import run_program
result = run_program(program)
result.final_state     # StateBundle，包含最终向量值
result.steps_executed  # 总转移步数
result.terminated_by   # "pipeline_end" | "terminal_condition" | "max_iterations"
result.trace           # TraceEntry列表，用于调试
result.verification    # VerifyResult（如果设置了expected_state）
```

### `optimize(program, *, fuse=True, eliminate_dead=True, fold_constants=True) -> Program`

优化程序，不修改原始对象。

```python
from axol.core import optimize
optimized = optimize(program)                          # 全部遍
optimized = optimize(program, fold_constants=False)    # 选择性遍
```

### `set_backend(name) / get_backend() / to_numpy(arr)`

切换数组计算后端。

```python
from axol.core import set_backend, get_backend, to_numpy
set_backend("cupy")     # 切换到GPU
xp = get_backend()      # 返回cupy模块
arr = to_numpy(gpu_arr) # 转回numpy
```

### `dispatch(request) -> dict`

AI智能体的Tool-Use API入口。

```python
from axol.api import dispatch
result = dispatch({"action": "run", "source": "...", "optimize": True})
```

### 向量类型

| 类型 | 描述 | 工厂方法 |
|------|------|---------|
| `FloatVec` | 32位浮点 | `from_list([1.0, 2.0])`、`zeros(n)`、`ones(n)` |
| `IntVec` | 64位整数 | `from_list([1, 2])`、`zeros(n)` |
| `BinaryVec` | 元素为{0, 1} | `from_list([0, 1])`、`zeros(n)`、`ones(n)` |
| `OneHotVec` | 恰好一个1.0 | `from_index(idx, n)`、`from_list(...)` |
| `GateVec` | 元素为{0.0, 1.0} | `from_list([1.0, 0.0])`、`zeros(n)`、`ones(n)` |
| `TransMatrix` | M x N float32矩阵 | `from_list(rows)`、`identity(n)`、`zeros(m, n)` |

### 操作描述符

```python
from axol.core.program import (
    # 加密（E）操作
    TransformOp,  # TransformOp(key="v", matrix=M, out_key=None)
    GateOp,       # GateOp(key="v", gate_key="g", out_key=None)
    MergeOp,      # MergeOp(keys=["a","b"], weights=w, out_key="out")
    DistanceOp,   # DistanceOp(key_a="a", key_b="b", metric="euclidean")
    RouteOp,      # RouteOp(key="v", router=R, out_key="_route")
    # 明文（P）操作
    StepOp,       # StepOp(key="v", threshold=0.0, out_key=None)
    BranchOp,     # BranchOp(gate_key="g", then_key="a", else_key="b", out_key="out")
    ClampOp,      # ClampOp(key="v", min_val=-inf, max_val=inf, out_key=None)
    MapOp,        # MapOp(key="v", fn_name="relu", out_key=None)
    # 扩展接口
    CustomOp,     # CustomOp(fn=callable, label="name")  -- security=P
)
```

### 分析器

```python
from axol.core import analyze

result = analyze(program)
result.coverage_pct        # E / total * 100
result.encrypted_count     # E转移数量
result.plaintext_count     # P转移数量
result.encryptable_keys    # 仅被E操作访问的键
result.plaintext_keys      # 被P操作访问的键
print(result.summary())    # 人类可读报告
```

### 验证

```python
from axol.core import verify_states, VerifySpec

result = verify_states(
    expected=expected_bundle,
    actual=actual_bundle,
    specs={"hp": VerifySpec.exact(tolerance=0.01)},
    strict_keys=False,
)
print(result.passed)    # True/False
print(result.summary()) # 详细报告
```

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

### 5. ReLU激活（map）

```
@relu
s x=[-2 0 3 -1 5]
:act=map(x;fn=relu)
# 结果：x = [0, 0, 3, 0, 5]
```

### 6. 阈值选择（step + branch）

```
@threshold_select
s scores=[0.3 0.8 0.1 0.9] high=[100 200 300 400] low=[1 2 3 4]
:s1=step(scores;t=0.5)->mask
:b1=branch(mask;then=high,else=low)->result
# mask = [0, 1, 0, 1]
# result = [1, 200, 3, 400]
```

### 7. 伤害流水线（全部4个新操作）

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

### 8. 100状态自动机（稀疏）

```
@auto_100
s s=onehot(0,100)
: step=transform(s;M=sparse(100x100;0,1=1 1,2=1 ... 98,99=1 99,99=1))
? done s[99]>=1
```

---

## 测试

```bash
# 运行全部测试（约320项）
pytest tests/ -v

# 核心测试
pytest tests/test_types.py tests/test_operations.py tests/test_program.py tests/test_dsl.py -v

# 优化器测试（18项）
pytest tests/test_optimizer.py -v

# 后端测试（13项，未安装cupy/jax时跳过）
pytest tests/test_backend.py -v

# Tool-Use API测试（20项）
pytest tests/test_api.py -v

# 模块系统测试（18项）
pytest tests/test_module.py -v

# 加密概念验证测试（21项）
pytest tests/test_encryption.py -v -s

# 新操作测试 - step/branch/clamp/map（44项）
pytest tests/test_new_ops.py -v

# 分析器测试 - E/P覆盖率分析（7项）
pytest tests/test_analyzer.py -v

# 新操作基准测试 - Python vs C# vs Axol（15项）
pytest tests/test_benchmark_new_ops.py -v -s

# 服务器端点测试（13项，需安装fastapi）
pytest tests/test_server.py -v

# 性能报告（生成PERFORMANCE_REPORT.md）
pytest tests/test_performance_report.py -v -s

# Token成本对比
pytest tests/test_token_cost.py -v -s

# 三语言基准测试（Python vs C# vs Axol）
pytest tests/test_benchmark_trilingual.py -v -s

# 启动Web前端
python -m axol.server   # http://localhost:8080
```

当前测试数量：**约320项**，全部通过（4项跳过：未安装cupy/jax）。

---

## 路线图

- [x] Phase 1：类型系统（7种向量类型 + StateBundle）
- [x] Phase 1：5个原语操作
- [x] Phase 1：程序执行引擎（管道 + 循环模式）
- [x] Phase 1：状态验证框架
- [x] Phase 2：DSL解析器（完整语法支持）
- [x] Phase 2：稀疏矩阵表示法
- [x] Phase 2：Token成本基准测试（Python、C#、Axol）
- [x] Phase 2：矩阵加密概念验证（全部5个操作已验证，21项测试）
- [x] Phase 3：编译器优化（变换融合、死状态消除、常量折叠）
- [x] Phase 3：GPU后端（numpy/cupy/jax可插拔）
- [x] Phase 4：AI智能体Tool-Use API（parse/run/inspect/verify/list_ops）
- [x] Phase 4：加密模块（encrypt_program、decrypt_state）
- [x] Phase 5：模块系统（注册表、import/use DSL、compose、schema验证）
- [x] 前端：FastAPI + 原生HTML/JS可视化调试器（追踪查看器、状态图表、加密演示）
- [x] 性能基准测试（token成本、运行时扩展性、优化器效果、加密开销）

---

## 许可证

MIT License。详情请参阅[LICENSE](LICENSE)。
