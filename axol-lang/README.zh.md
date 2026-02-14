# AXOL

**基于坍缩的编程语言**

AXOL是一种**观测即计算**的编程语言。不计算答案——编织(weave)可能性的结构，然后观测(observe)它。选择要知道多少，愿意失去多少。

```
declare "mood" {
    input text(8)
    input context(8)
    relate sentiment <- text, context via <~>
    output sentiment
    quality omega=0.85 phi=0.8
}

weave mood quantum=true seed=42

# 看到所有可能性（代价：无）
gaze mood { text = [...] context = [...] }

# 看到一个答案（代价：其他一切）
observe mood { text = [...] context = [...] }
```

**观测代价：O(1)。始终如此。与模型大小或数据无关。**

---

## AXOL为何存在

迄今为止创造的所有编程语言都遵循同一范式：

```
输入 → 计算 → 输出
```

程序接收数据，通过一系列运算产生结果。无论是C、Python、Haskell还是神经网络——获得答案的代价与计算复杂度成正比。

AXOL将其反转：

```
构建结构（一次）→ 观测（O(1)，任意多次）
```

计算发生在**编织结构(weave)**时。此后的每次观测——无论多少次——代价相同：**约5微秒**。

这不是优化。这是基于量子测量理论和混沌动力学的**不同计算模型**。

---

## 只有AXOL能做的三件事

### 1. 选择知道多少

```
gaze x           # C=0: 看到所有可能性，不失去任何东西
glimpse x 0.3    # C=0.3: 部分看到，部分失去
glimpse x 0.7    # C=0.7: 几乎确定，大部分可能性消亡
observe model {}  # C=1: 一个答案，其余全部摧毁
```

在其他所有语言中，函数要么执行要么不执行。AXOL提供从完全不确定到完全确定的**连续光谱**。参数C（坍缩级别）控制知识与可能性的交换比。

### 2. 追踪知道的代价

```
wave planet = scanner { reading = [...] }
gaze planet                    # 所有可能性都活着
focus planet 0.5               # 部分坍缩
widen planet 0.3               # 尝试恢复——无法完全还原
gaze planet                    # 分布已永久改变
```

观测是**不可逆的**。一旦坍缩可能性，就无法完全恢复。AXOL通过**negativity**——衡量关系中剩余开放程度的数值——显式追踪这一点。

没有其他语言能建模**知道某事会改变系统**这一事实。

### 3. 通过干涉计算

```
rel agreement = wave_a <-> wave_b via <~>   # 建设性干涉：共同点增强
rel conflict  = wave_a <-> wave_b via <!>   # 破坏性干涉：差异放大
rel combined  = wave_a <-> wave_b via <+>   # 加性干涉：信息积累
```

AXOL不用数字计算。用**波之间的干涉模式**计算。当两个波相遇时：
- **建设性 (`<~>`)**: 共有的变强
- **破坏性 (`<!>`)**: 不同的被放大，相同的被抵消
- **乘法性 (`<*>`)**: 只有两者都有的才能存活
- **加性 (`<+>`)**: 信息积累
- **条件性 (`<?>`)**: 一个波旋转另一个波的相位

这不是比喻。这是遵循Born规则量子力学的复振幅运算。

---

## 为什么是新范式

### 与现有范式的比较

| | 命令式 (C, Python) | 函数式 (Haskell) | 神经网络 (PyTorch) | **AXOL** |
|---|---|---|---|---|
| **基本单位** | 变量与指令 | 函数与值 | 张量与运算 | **波(Wave)与观测** |
| **计算方式** | 顺序执行 | 函数组合 | 反向传播 | **干涉 + 坍缩** |
| **获得答案的代价** | O(运算数) | O(运算数) | O(参数数) | **O(1)** |
| **不确定性** | 无（确定性） | 无（确定性） | 概率分布（解释性） | **基本类型 (Wave)** |
| **观测的效果** | 无 | 无 | 无 | **改变系统** |
| **部分知识** | 不可能 | 不可能 | 不可能 | **C=0~1连续光谱** |
| **关系** | 引用/指针 | 函数 | 权重矩阵 | **干涉模式（一等对象）** |
| **不可逆性** | 无 | 无（纯） | 无 | **观测摧毁可能性** |

### 现有语言的隐含假设

1. **计算是确定的** — 相同输入产生相同输出。(AXOL: 取决于观测级别)
2. **读取是免费的** — 读变量不改变变量。(AXOL: 观测改变状态)
3. **答案只有一个** — 函数返回一个值。(AXOL: 返回可能性的分布)
4. **关系是派生的** — 先有A和B，A-B关系在后。(AXOL: 关系在先)

AXOL抛弃了这全部四个假设。

---

## 安装

```bash
git clone https://github.com/user/axol-lang.git
cd axol-lang
cargo build --release
./target/release/axol run examples/hello.axol
```

要求：Rust 1.70+（使用`faer`线性代数库）

---

## 性能

```
weave（一次）:      ~11ms   (dim=8, 混沌动力学)
observe:            ~5μs    (dim=8, O(1))
gaze:               ~5μs    (dim=8, O(1), C=0)
glimpse:            ~30μs   (dim=8, 含退相干)
rel observe:        ~5μs    (O(1))
focus:              ~25μs   (dim=8, 含密度矩阵)
```

每帧NPC 20次观测：**总计~80μs**。60fps下仅占帧预算的0.5%。

---

## 应用领域

| 领域 | C的含义 | AXOL的作用 |
|------|---------|------------|
| 游戏NPC AI | C=0 策略探索、C=1 行动执行 | 每帧O(1)决策 |
| 机器人学 | C=0 路径空间、C=1 移动 | kHz控制环实时处理 |
| 感知建模 | expect=预测、wave=感觉、C=观测 | 预测误差追踪（negativity_delta） |
| 对话系统 | C=0 可能的回应、C=1 发言 | 同意/反驳的干涉结构 |
| 程序化叙事 | C=0 所有分支、C=1 选择 | 选择的不可逆代价 |
| 自动驾驶 | C=0 所有路径、C=1 转向 | 实时 + 观测改变状态 |
| 金融HFT | C=0 市场状态、C=1 下单 | μs决策，下单改变市场 |
| 生成音乐 | C=0 和声可能性、C=1 演奏 | `<~>` 协和、`<!>` 不协和 |
| 社会仿真 | C=0 意见分布、C=1 行动 | 关系干涉建模社会动力学 |
| 医疗诊断 | C=0 可能的疾病、C=1 确诊 | 检查（观测）改变诊断 |

---

## 示例

```bash
axol run examples/hello.axol                     # 基本管线
axol run examples/hello_v2.axol                  # 关系优先语法
axol run examples/usecase_npc_realtime.axol       # 游戏NPC AI
axol run examples/usecase_perception.axol         # 预测编码
axol run examples/usecase_dialogue.axol           # 对话动力学
axol run examples/usecase_observation_cost.axol    # 知的不可逆性
axol run examples/learn_xor.axol                  # 数据学习
```

---

## 许可证

MIT
