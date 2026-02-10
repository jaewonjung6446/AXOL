<p align="center">
  <h1 align="center">AXOL</h1>
  <p align="center">
    <strong>AIネイティブ ベクトルプログラミング言語</strong>
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

> **警告：本プロジェクトは初期実験段階です。**
> API、DSL構文、内部アーキテクチャは予告なく変更される可能性があります。本番環境での使用は推奨しません。コントリビューションとフィードバックを歓迎します。

---

## Axolとは？

**Axol**は、**AIエージェント**が従来のプログラミング言語よりも**少ないトークン**でプログラムを読み書きし、推論できるようにゼロから設計されたドメイン固有言語（DSL）です。

従来の制御フロー（if/else、forループ、関数呼び出し）の代わりに、Axolはすべての計算を不変ベクトルバンドル上の**ベクトル変換**と**状態遷移**として表現します。この設計は単純な観察に基づいています：**LLMはトークンごとに課金され**、既存のプログラミング言語はトークン効率ではなく人間の可読性のために設計されています。

### 主な特徴

- Pythonと比較して**30〜50%少ないトークン**消費
- C#と比較して**48〜75%少ないトークン**消費
- **5つのプリミティブ演算**ですべての計算を表現：`transform`、`gate`、`merge`、`distance`、`route`
- **疎行列表記法**：密表現のO(N^2)に対しO(N)でスケーリング
- 完全な状態トレーシングによる**決定論的実行**
- **NumPyバックエンド**による大規模ベクトル演算の500倍以上の高速化

---

## 目次

- [理論的背景](#理論的背景)
- [アーキテクチャ](#アーキテクチャ)
- [クイックスタート](#クイックスタート)
- [DSL構文](#dsl構文)
- [トークンコスト比較](#トークンコスト比較)
- [ランタイム性能](#ランタイム性能)
- [APIリファレンス](#apiリファレンス)
- [使用例](#使用例)
- [テスト](#テスト)
- [ロードマップ](#ロードマップ)

---

## 理論的背景

### トークン経済問題

現代のAIシステム（GPT-4、Claudeなど）は**トークン経済**の下で動作します。入出力のすべての文字がトークンを消費し、コストとレイテンシに直接影響します。AIエージェントがコードを書いたり読んだりする際、プログラミング言語の冗長性は以下に直接影響します：

1. **コスト** - トークンが多い = API料金が高い
2. **レイテンシ** - トークンが多い = 応答が遅い
3. **コンテキストウィンドウ** - トークンが多い = 他の情報のためのスペースが減る
4. **推論精度** - 圧縮された表現がノイズを削減

### なぜベクトル計算なのか？

従来のプログラミング言語は**制御フロー**（分岐、ループ、再帰）でロジックを表現します。これは人間には直感的ですが、AIには非効率的です：

```python
# Python: 67トークン
TRANSITIONS = {"IDLE": "RUNNING", "RUNNING": "DONE", "DONE": "DONE"}
def state_machine():
    state = "IDLE"
    steps = 0
    while state != "DONE":
        state = TRANSITIONS[state]
        steps += 1
    return state, steps
```

同じロジックをベクトル変換で：

```
# Axol DSL: 48トークン（28%削減）
@state_machine
s state=onehot(0,3)
: advance=transform(state;M=[0 1 0;0 0 1;0 0 1])
? done state[2]>=1
```

状態機械の遷移テーブルが**行列**となり、状態遷移が**行列乗算**になります。AIは文字列比較、辞書検索、ループ条件を推論する必要がなく、単一の行列演算だけで済みます。

### 5つのプリミティブ演算

Axolはすべての計算を5つの演算に帰着させます。それぞれが基本的な線形代数の概念に対応します：

| 演算 | 数学的基盤 | 説明 |
|------|----------|------|
| `transform` | 行列積：`v @ M` | 線形状態変換 |
| `gate` | アダマール積：`v * g` | 条件付きマスキング |
| `merge` | 加重和：`sum(v_i * w_i)` | ベクトル結合 |
| `distance` | L2 / コサイン / 内積 | 類似度測定 |
| `route` | `argmax(v @ R)` | 離散分岐 |

これら5つの演算で以下を表現できます：
- 状態機械 (transform)
- 条件ロジック (gate)
- 蓄積・集約 (merge)
- 類似度検索 (distance)
- 意思決定 (route)

### 疎行列表記法

大規模な状態空間では、密行列表現はトークン基準でO(N^2)です。Axolの疎表記法はこれをO(N)に削減します：

```
# 密：O(N^2)トークン - N=100では非実用的
M=[0 1 0 0 ... 0; 0 0 1 0 ... 0; ...]

# 疎：O(N)トークン - 線形スケーリング
M=sparse(100x100;0,1=1 1,2=1 ... 98,99=1 99,99=1)
```

| N | Python | C# | Axol DSL | DSL/Python | DSL/C# |
|---|--------|-----|----------|------------|--------|
| 5 | 74 | 109 | 66 | 0.89x | 0.61x |
| 25 | 214 | 269 | 186 | 0.87x | 0.69x |
| 100 | 739 | 869 | 636 | 0.86x | 0.73x |
| 200 | 1,439 | 1,669 | 1,236 | 0.86x | 0.74x |

---

## アーキテクチャ

```
                    +-----------+
  .axolソース ----->| パーサー  |----> Programオブジェクト
                    | (dsl.py)  |         |
                    +-----------+         |
                                          v
                    +-----------+    +-----------+
                    | 検証器    |<---| 実行エンジン|
                    |(verify.py)|    |(program.py)|
                    +-----------+    +-----------+
                                          |
                         使用             |
                    +-----------+         |
                    | 演算モジュール|<------+
                    | (ops.py)  |
                    +-----------+
                         |
                    +-----------+
                    | 型システム |
                    |(types.py) |
                    +-----------+
```

### モジュール概要

| モジュール | 説明 |
|-----------|------|
| `axol.core.types` | 7つのベクトル型 + `StateBundle` |
| `axol.core.operations` | 5つのプリミティブ演算 |
| `axol.core.program` | 実行エンジン：`Program`、`Transition`、`run_program` |
| `axol.core.verify` | 状態検証（exact/cosine/euclidean マッチング） |
| `axol.core.dsl` | DSLパーサー：`parse(source) -> Program` |

---

## クイックスタート

### インストール

```bash
git clone https://github.com/your-username/AXOL.git
cd AXOL
pip install -e ".[dev]"
```

### 必要要件

- Python 3.11+
- NumPy >= 1.24.0
- pytest >= 7.4.0（開発用）
- tiktoken >= 0.5.0（開発用、トークン分析）

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

print(f"最終カウント: {result.final_state['count'].to_list()}")  # [5.0]
print(f"ステップ数: {result.steps_executed}")
print(f"終了条件: {result.terminated_by}")  # terminal_condition
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
print(f"減衰後HP: {result.final_state['hp'].to_list()}")  # [80.0]
```

---

## DSL構文

### プログラム構造

```
@program_name              # ヘッダー：プログラム名
s key1=[values] key2=...   # 状態：初期ベクトル宣言
: name=op(args)->out       # 遷移：演算定義
? terminal condition       # ターミナル：ループ終了条件（任意）
```

### 状態宣言

```
s hp=[100]                          # 単一浮動小数点ベクトル
s pos=[1.5 2.0 -3.0]               # 多要素ベクトル
s state=onehot(0,5)                 # ワンホットベクトル：インデックス0、サイズ5
s buffer=zeros(10)                  # サイズ10のゼロベクトル
s mask=ones(3)                      # サイズ3の全1ベクトル
s hp=[100] mp=[50] stamina=[75]     # 1行に複数ベクトル宣言
```

### 演算

```
# transform：行列乗算
: decay=transform(hp;M=[0.8])
: advance=transform(state;M=[0 1 0;0 0 1;0 0 1])

# gate：要素ごとのマスキング
: masked=gate(values;g=mask)

# merge：ベクトルの加重和
: total=merge(a b c;w=[1 1 1])->result

# distance：類似度測定
: dist=distance(pos1 pos2)
: sim=distance(vec1 vec2;metric=cosine)

# route：argmaxルーティング
: choice=route(scores;R=[1 0 0;0 1 0;0 0 1])
```

### 行列形式

```
# 密：行は;で区切り
M=[0.8]                               # 1x1
M=[1 0;0 1]                           # 2x2 単位行列
M=[0 1 0;0 0 1;0 0 1]                # 3x3 シフト行列

# 疎：非ゼロ要素のみ表記
M=sparse(100x100;0,1=1 1,2=1 99,99=1)
```

### ターミナル条件

```
? done count>=5              # count[0] >= 5で終了
? finished state[2]>=1       # state[2] >= 1で終了（インデックスアクセス）
? end hp<=0                  # hp[0] <= 0で終了
```

`?`行がない場合は**パイプラインモード**で実行されます（すべての遷移が1回実行）。

---

## トークンコスト比較

`tiktoken` cl100k_baseトークナイザーで測定（GPT-4 / Claude使用）。

### Python vs C# vs Axol DSL

| プログラム | Python | C# | Axol DSL | vs Python | vs C# |
|-----------|--------|----|----------|-----------|-------|
| Counter | 32 | 61 | 33 | -3.1% | 45.9% |
| State Machine | 67 | 147 | 48 | 28.4% | 67.3% |
| HP Decay | 51 | 134 | 51 | 0.0% | 61.9% |
| Combat | 145 | 203 | 66 | 54.5% | 67.5% |
| Data Heavy | 159 | 227 | 67 | 57.9% | 70.5% |
| Pattern Match | 151 | 197 | 49 | 67.5% | 75.1% |
| 100-State Auto | 739 | 869 | 636 | 13.9% | 26.8% |
| **合計** | **1,344** | **1,838** | **950** | **29.3%** | **48.3%** |

### 主な知見

1. **単純なプログラム**（counter、hp_decay）：DSLはPythonと同等です。
2. **構造化プログラム**（combat、data_heavy、pattern_match）：DSLはPython比**50〜68%**、C#比**67〜75%**のトークン削減を達成。ベクトル表現がクラス定義、制御フロー、ボイラープレートを排除します。
3. **大規模状態空間**（100+状態）：疎行列表記法がPython比**約38%**、C#比**約27%**の一貫した削減を提供し、O(N)スケーリングを実現します。

---

## ランタイム性能

AxolはNumPyを計算バックエンドとして使用します。

### 小規模ベクトル（dim < 100）

| 次元 | Pythonループ | Axol（NumPy） | 優位 |
|------|------------|--------------|------|
| dim=4 | ~6 us | ~11 us | Python 2x |
| dim=100 | ~14 us | ~20 us | Python 1.4x |

### 大規模ベクトル（dim >= 1000）

| 次元 | Pythonループ | Axol（NumPy） | 優位 |
|------|------------|--------------|------|
| dim=1,000（行列積） | ~129 ms | ~0.2 ms | **Axol 573x** |
| dim=10,000（行列積） | ~14,815 ms | ~381 ms | **Axol 39x** |

大規模ベクトル演算（行列乗算）において、AxolのNumPyバックエンドは純粋なPythonループよりも**数百倍高速**です。

---

## 使用例

### 1. カウンター（0 -> 5）

```
@counter
s count=[0] one=[1]
: increment=merge(count one;w=[1 1])->count
? done count>=5
```

### 2. 状態機械（IDLE -> RUNNING -> DONE）

```
@state_machine
s state=onehot(0,3)
: advance=transform(state;M=[0 1 0;0 0 1;0 0 1])
? done state[2]>=1
```

### 3. HP減衰（100 x 0.8^3 = 51.2）

```
@hp_decay
s hp=[100] round=[0] one=[1]
: decay=transform(hp;M=[0.8])
: tick=merge(round one;w=[1 1])->round
? done round>=3
```

### 4. 戦闘ダメージ（パイプライン）

```
@combat
s atk=[50] def_val=[20] flag=[1]
: scale=transform(atk;M=[1.5])->scaled
: block=gate(def_val;g=flag)
: combine=merge(scaled def_val;w=[1 -1])->damage
```

---

## テスト

```bash
# 全テスト（149件）
pytest tests/ -v

# DSLパーサーテスト
pytest tests/test_dsl.py -v

# トークンコスト比較
pytest tests/test_token_cost.py -v -s

# 3言語ベンチマーク（Python vs C# vs Axol）
pytest tests/test_benchmark_trilingual.py -v -s
```

現在のテスト数：**149件**、すべて合格。

---

## ロードマップ

- [x] Phase 1：型システム（7ベクトル型 + StateBundle）
- [x] Phase 1：5つのプリミティブ演算
- [x] Phase 1：プログラム実行エンジン（パイプライン + ループモード）
- [x] Phase 1：状態検証フレームワーク
- [x] Phase 2：DSLパーサー（完全な文法サポート）
- [x] Phase 2：疎行列表記法
- [x] Phase 2：トークンコストベンチマーク（Python、C#、Axol）
- [ ] Phase 3：コンパイラ最適化（演算融合、デッドステート除去）
- [ ] Phase 3：GPUバックエンド（CuPy / JAX）
- [ ] Phase 4：AIエージェント統合（tool-use API）
- [ ] Phase 4：状態トレースビジュアルデバッガー
- [ ] Phase 5：マルチプログラム合成とモジュールシステム

---

## ライセンス

MIT License。詳細は[LICENSE](LICENSE)を参照してください。
