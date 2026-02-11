<p align="center">
  <h1 align="center">AXOL</h1>
  <p align="center">
    <strong>カオス理論に基づく空間・確率的計算言語</strong>
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

> **警告：本プロジェクトは初期実験段階です。**
> API、DSL構文、内部アーキテクチャは予告なく破壊的変更が行われる可能性があります。本番環境での使用は推奨しません。コントリビューションやフィードバックは歓迎します。

---

## AXOLとは？

**AXOL**は、計算の基盤としての**時間軸**（逐次実行）を否定し、2つの代替軸に置き換えるプログラミング言語です。

- **空間軸** — ノード間の関係が計算を決定する
- **確率軸** — 結果の尤度が結果を決定する

「これをしてから、次にあれをする」ではなく、AXOLは「何が何に、どの程度強く関係しているか？」を問います。その結果、**カオス理論**に基づいた根本的に異なる実行モデルが生まれ、計算は**ストレンジアトラクタ**を構築・観測する行為となります。

```
従来型:  命令1 → 命令2 → 命令3  (時間逐次型)

AXOL:
  [空間]     NodeA ──関係── NodeB     「どこ」が計算を決定する
  [確率]     state = { alpha|可能性1> + beta|可能性2> }  「どの程度ありえるか」が結果を決定する
```

### 主要な特性

- **三段階実行**: 宣言 → 織り上げ → 観測 (コンパイル → 実行ではない)
- **カオス理論基盤**: Tapestry = ストレンジアトラクタ、品質はLyapunov指数とフラクタル次元で測定
- **二軸品質指標**: Omega（凝集性）+ Phi（鮮明度）— 厳密で測定可能、合成可能
- **平均63%のトークン削減** — 等価なPythonと比較（量子DSL）
- **実現不可能性検出** — 目標が数学的に達成不可能な場合、計算前に警告
- **Lyapunov推定精度**: 平均誤差 0.0002
- **基盤層の9つのプリミティブ演算**: `transform`, `gate`, `merge`, `distance`, `route`（暗号化対応）+ `step`, `branch`, `clamp`, `map`（平文）
- **行列レベル暗号化** — 相似変換によりプログラムを暗号学的に読み取り不可能にする
- **NumPyバックエンド** — オプションでGPUアクセラレーション対応（CuPy/JAX）

---

## 目次

- [パラダイムシフト](#パラダイムシフト)
- [三段階実行モデル](#三段階実行モデル)
- [品質指標](#品質指標)
- [カオス理論基盤](#カオス理論基盤)
- [合成規則](#合成規則)
- [量子DSL](#量子dsl)
- [性能](#性能)
- [基盤層](#基盤層)
  - [9つのプリミティブ演算](#9つのプリミティブ演算)
  - [行列暗号化 (Shadow AI)](#行列暗号化-shadow-ai)
  - [平文演算とセキュリティ分類](#平文演算とセキュリティ分類)
  - [コンパイラ最適化](#コンパイラ最適化)
  - [GPUバックエンド](#gpuバックエンド)
  - [モジュールシステム](#モジュールシステム)
  - [量子干渉 (Phase 6)](#量子干渉-phase-6)
  - [クライアント・サーバーアーキテクチャ](#クライアントサーバーアーキテクチャ)
- [アーキテクチャ](#アーキテクチャ)
- [クイックスタート](#クイックスタート)
- [APIリファレンス](#apiリファレンス)
- [使用例](#使用例)
- [テストスイート](#テストスイート)
- [ロードマップ](#ロードマップ)

---

## パラダイムシフト

### 否定するもの

すべての現代プログラミング言語は**時間軸**（逐次実行）の上に構築されています。

| パラダイム | 時間軸への依存 |
|----------|--------------|
| 命令型 (C, Python) | 「まずこれをし、次にあれをする」— 明示的な時間順序 |
| 関数型 (Haskell, Lisp) | 宣言的だが、評価順序は存在する |
| 並列型 (Go, Rust async) | 複数の時間軸を同時に — それでも時間に束縛される |
| 宣言型 (SQL, HTML) | 「何を」かを記述するが、エンジンは時間軸で処理する |

これはVon Neumannアーキテクチャがクロックサイクル — すなわち時間軸 — で動作するためです。

### 提案するもの

AXOLは時間軸を2つの代替軸に置き換えます。

| 軸 | 決定するもの | 類推 |
|----|------------|------|
| **空間軸**（関係性） | 関係で結ばれたノードが計算を決定する | 「いつ」ではなく「どこにあるか」が重要 |
| **確率軸**（可能性） | 重ね合わせ状態が最も確からしい結果に崩壊する | 「正確」よりも「どの程度ありえるか」が重要 |

トレードオフ：**時間のボトルネックを排除する代わりに、正確性を犠牲にします。**

```
正確性 ↑  →  エンタングルメントコスト ↑  →  構築時間 ↑
正確性 ↓  →  エンタングルメントコスト ↓  →  構築時間 ↓
               ただし観測は常に即時
```

### なぜ重要なのか

| 特性 | 従来のコンパイル | AXOLのエンタングルメント |
|------|----------------|----------------------|
| 準備 | コード → 機械語翻訳 | ロジック間の確率的相関を構築 |
| 実行 | 逐次的な機械命令 | 観測（入力）→ 即時崩壊 |
| ボトルネック | 実行パスの長さに比例 | エンタングルメントの深さにのみ依存 |
| 類推 | 「速い道路を建設する」 | 「すでに目的地にいる」 |

---

## 三段階実行モデル

### 第1段階: 宣言

**何が何に関係するか**を定義し、品質目標を設定します。この時点では計算は行われません。

```python
from axol.quantum import DeclarationBuilder, RelationKind

decl = (
    DeclarationBuilder("search")
    .input("query", 64)
    .input("db", 64)
    .relate("relevance", ["query", "db"], RelationKind.PROPORTIONAL)
    .output("relevance")
    .quality(omega=0.9, phi=0.7)   # quality targets
    .build()
)
```

またはDSLで：

```
entangle search(query: float[64], db: float[64]) @ Omega(0.9) Phi(0.7) {
    relevance <~> similarity(query, db)
    ranking <~> relevance
}
```

### 第2段階: 織り上げ (Weave)

**ストレンジアトラクタ**（Tapestry）を構築します。ここで計算コストが発生します。Weaverは以下を実行します：

1. エンタングルメントコストを推定
2. 実現不可能性を検出（目標が数学的に達成不可能な場合に警告）
3. ノードごとにアトラクタ構造を構築（軌道行列、Hadamard干渉）
4. Lyapunov指数とフラクタル次元を推定
5. 実行用の内部`Program`を組み立て

```python
from axol.quantum import weave

tapestry = weave(decl, seed=42)
print(tapestry.weaver_report)
# target:   Omega(0.90) Phi(0.70)
# achieved: Omega(0.95) Phi(0.82)
# feasible: True
```

実現不可能性検出の例：

```
> weave predict_weather: WARNING
>   target:   Omega(0.99) Phi(0.99)
>   maximum:  Omega(0.71) Phi(0.68)
>   reason:   chaotic dependency (lambda=2.16 on path: input->atmosphere->prediction)
>   attractor_dim: D=2.06 (Lorenz-class)
```

### 第3段階: 観測 (Observe)

入力値を与えると → アトラクタ上の一点に**即時崩壊**します。時間計算量：O(D)、ここでDはアトラクタの埋め込み次元です。

```python
from axol.quantum import observe, reobserve
from axol.core.types import FloatVec

# 単一観測
result = observe(tapestry, {
    "query": FloatVec.from_list([1.0] * 64),
    "db": FloatVec.from_list([0.5] * 64),
})
print(f"Omega: {result.omega:.2f}, Phi: {result.phi:.2f}")

# 品質向上のための繰り返し観測
result = reobserve(tapestry, inputs, count=10)
# 確率分布を平均化し、経験的Omegaを再計算
```

### 完全パイプライン（DSL）

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

## 品質指標

AXOLは2つの独立した軸で計算品質を測定します。

```
        Phi (鮮明度)
        ^
   1.0  |  鮮明だが不安定        理想的（強いエンタングルメント）
        |
   0.0  |  ノイズ               安定だがぼやけている
        +-----------------------------> Omega (凝集性)
       0.0                             1.0
```

### Omega — 凝集性（どの程度安定か？）

**最大Lyapunov指数**（lambda）から導出：

```
Omega = 1 / (1 + max(lambda, 0))
```

| lambda | 意味 | Omega |
|--------|------|-------|
| lambda < 0 | 収束系（安定） | 1.0 |
| lambda = 0 | 中立安定 | 1.0 |
| lambda = 0.91 | Lorenzクラスのカオス | 0.52 |
| lambda = 2.0 | 強いカオス | 0.33 |

**解釈**: Omega = 1.0 は繰り返し観測が常に同じ結果を与えることを意味します。Omega < 1.0 はカオス的感度を意味し、入力の小さな変化が異なる出力を引き起こします。

### Phi — 鮮明度（どの程度鮮明か？）

アトラクタの**フラクタル次元**（D）から導出：

```
Phi = 1 / (1 + D / D_max)
```

| D | D_max | 意味 | Phi |
|---|-------|------|-----|
| 0 | 任意 | 点（デルタ分布） | 1.0 |
| 1 | 4 | 線アトラクタ | 0.80 |
| 2.06 | 3 | Lorenzアトラクタ | 0.59 |
| D_max | D_max | 位相空間全体を埋める | 0.50 |

**解釈**: Phi = 1.0 は出力が鮮明で確定的な値であることを意味します。Phi → 0.0 は出力が多くの可能性に分散していることを意味します（ノイズ）。

### 両指標は合成可能

品質指標は合成を通じて伝播します — [合成規則](#合成規則)を参照。

---

## カオス理論基盤

AXOLの理論的基盤は、その概念を確立されたカオス理論にマッピングします。

| AXOLの概念 | カオス理論 | 数学的対象 |
|---|---|---|
| Tapestry | ストレンジアトラクタ | 位相空間のコンパクト不変集合 |
| Omega（凝集性） | Lyapunov安定性 | `1/(1+max(lambda,0))` |
| Phi（鮮明度） | フラクタル次元の逆数 | `1/(1+D/D_max)` |
| Weave | アトラクタ構築 | 反復写像の軌道行列 |
| Observe | アトラクタ上の点崩壊 | 時間計算量 O(D) |
| エンタングルメント範囲 | 引き込み領域 | 収束領域の境界 |
| エンタングルメントコスト | 収束反復回数 | `E = sum_path(iterations * complexity)` |
| 観測後の再利用 | アトラクタの安定性 | lambda < 0: 再利用可能、lambda > 0: 再織り上げ |

### Lyapunov指数推定

**Benettin QR分解法**を使用して、軌道行列から最大Lyapunov指数を推定します。

- **収束系** (lambda < 0): 予測可能、Omegaは1.0に近づく
- **中立系** (lambda = 0): カオスの縁
- **カオス系** (lambda > 0): 初期条件に敏感、Omega < 1.0

推定精度は既知のシステムに対して検証済み（平均誤差: 0.0002）。

### フラクタル次元推定

2つの手法が利用可能：

- **ボックスカウンティング**: グリッドベース、ln(N) vs ln(1/epsilon) の回帰
- **相関次元** (Grassberger-Procaccia): ペアワイズ距離分析

既知の幾何学形状に対して検証済み：線分 (D=1)、Cantor集合 (D~0.63)、Sierpinski三角形 (D~1.58)。

### 理論文書

- [THEORY.md](THEORY.md) — 基礎理論（時間軸の否定、エンタングルメントベースの計算）
- [THEORY_MATH.md](THEORY_MATH.md) — カオス理論の形式化（Lyapunov、フラクタル、合成の証明）

---

## 合成規則

複数のTapestryを結合する場合、品質指標は厳密な数学的規則に従って伝播します。

### 直列合成 (A → B)

```
lambda_total = lambda_A + lambda_B          (指数が蓄積)
Omega_total  = 1/(1+max(lambda_total, 0))   (Omegaは劣化)
D_total      = D_A + D_B                    (次元が加算)
Phi_total    = Phi_A * Phi_B                (Phiは乗算 — 常に劣化)
```

### 並列合成 (A || B)

```
lambda_total = max(lambda_A, lambda_B)      (最弱リンク)
Omega_total  = min(Omega_A, Omega_B)        (最弱リンク)
D_total      = max(D_A, D_B)               (最も複雑)
Phi_total    = min(Phi_A, Phi_B)            (最も不鮮明)
```

### 再利用規則

```
lambda < 0  →  観測後に再利用可能（アトラクタが安定）
lambda > 0  →  観測後に再織り上げが必要（カオス — アトラクタが崩壊）
```

### まとめ表

| モード | lambda | Omega | D | Phi |
|--------|--------|-------|---|-----|
| 直列 | 和 | 1/(1+max(和,0)) | 和 | 積 |
| 並列 | 最大 | 最小 | 最大 | 最小 |

---

## 量子DSL

### 構文概要

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

### 関係演算子

| 演算子 | 名称 | 意味 |
|--------|------|------|
| `<~>` | 比例 | 線形相関 |
| `<+>` | 加算 | 重み付き和 |
| `<*>` | 乗算 | 積の関係 |
| `<!>` | 逆 | 逆相関 |
| `<?>` | 条件付き | 文脈依存 |

### 使用例

#### 単純検索

```
entangle search(query: float[64], db: float[64]) @ Omega(0.9) Phi(0.7) {
    relevance <~> similarity(query, db)
    ranking <~> relevance
}
result = observe search(query_vec, db_vec)
```

#### 多段パイプライン

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

#### 分類

```
entangle classify(input: float[32]) @ Omega(0.95) Phi(0.9) {
    category <~> classify(input)
}
result = observe classify(input_vec)
```

---

## 性能

### トークン効率 — 量子DSL vs Python

`tiktoken` cl100k_base トークナイザーで測定。

| プログラム | Python トークン | DSL トークン | 削減率 |
|-----------|:--------------:|:----------:|:------:|
| search | 173 | 57 | **67%** |
| classify | 129 | 39 | **70%** |
| pipeline | 210 | 73 | **65%** |
| multi_input | 191 | 74 | **61%** |
| reobserve_pattern | 131 | 62 | **53%** |
| **合計** | **834** | **305** | **63%** |

### トークン効率 — 基盤DSL vs Python vs C#

| プログラム | Python | C# | Axol DSL | vs Python | vs C# |
|-----------|:------:|:--:|:--------:|:---------:|:-----:|
| Counter | 32 | 61 | 33 | -3% | 46% |
| State Machine | 67 | 147 | 48 | 28% | 67% |
| Combat Pipeline | 145 | 203 | 66 | 55% | 68% |
| 100-State Automaton | 739 | 869 | 636 | 14% | 27% |

### 精度

| 指標 | 値 |
|------|-----|
| Lyapunov推定平均誤差 | **0.0002** |
| Omega計算式誤差 | **0**（厳密） |
| Phi計算式誤差 | **0**（厳密） |
| 合成規則 | **全てPASS** |
| 観測一貫性（50回繰り返し） | **1.0000** |

### 速度

| 操作 | 時間 |
|------|------|
| DSLパース（単純） | ~25 us |
| DSLパース（完全プログラム） | ~62 us |
| コスト推定 | ~40 us |
| 単一観測 | ~300 us |
| Weave（2ノード、dim=8） | ~14 ms |
| Reobserve x10 | ~14 ms |
| **完全パイプライン**（パース → 織り上げ → 観測、dim=16） | **~17 ms** |

### スケーリング

| ノード数 | 次元 | Weave時間 |
|:-------:|:----:|:---------:|
| 1 | 8 | 9 ms |
| 4 | 8 | 25 ms |
| 16 | 8 | 108 ms |
| 2 | 4 | 12 ms |
| 2 | 64 | 39 ms |

完全なベンチマークデータ: [QUANTUM_PERFORMANCE_REPORT.md](QUANTUM_PERFORMANCE_REPORT.md) | [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md)

---

## 基盤層

量子モジュール (`axol/quantum/`) は基盤層 (`axol/core/`) の上に構築されており、基盤層を変更しません。基盤層は数学的エンジンを提供します：ベクトル型、行列演算、プログラム実行、暗号化、最適化。

### 9つのプリミティブ演算

| 演算 | セキュリティ | 数学的基礎 | 説明 |
|------|:----------:|-----------|------|
| `transform` | **E** | 行列乗算: `v @ M` | 線形状態変換 |
| `gate` | **E** | Hadamard積: `v * g` | 条件付きマスキング |
| `merge` | **E** | 重み付き和: `sum(v_i * w_i)` | ベクトル結合 |
| `distance` | **E** | L2 / cosine / dot | 類似度測定 |
| `route` | **E** | `argmax(v @ R)` | 離散分岐 |
| `step` | **P** | `where(v >= t, 1, 0)` | 閾値からバイナリゲートへ |
| `branch` | **P** | `where(g, then, else)` | 条件付きベクトル選択 |
| `clamp` | **P** | `clip(v, min, max)` | 値範囲制限 |
| `map` | **P** | `f(v)` 要素ごと | 非線形活性化 |

5つの**E**（暗号化）演算は相似変換により暗号化データ上で実行可能です。4つの**P**（平文）演算は非線形の表現力を追加します。

### 行列暗号化 (Shadow AI)

Axolのすべての計算は行列乗算（`v @ M`）に帰着します。これにより**相似変換暗号化**が可能になります：

```
M' = K^(-1) @ M @ K     (暗号化された演算行列)
state' = state @ K       (暗号化された初期状態)
result = result' @ K^(-1)(復号された出力)
```

- 暗号化されたプログラムは暗号化ドメインで正しく動作する
- すべてのビジネスロジックが隠蔽される — 行列はランダムノイズに見える
- 難読化とは異なる — これは暗号学的変換である
- 全5つのE演算が検証済み（`tests/test_encryption.py` の21テスト）

### 平文演算とセキュリティ分類

すべての演算は`SecurityLevel`（EまたはP）を持ちます。組み込みアナライザが暗号化カバレッジを報告します：

```python
from axol.core import parse, analyze

result = analyze(program)
print(result.summary())
# Program: damage_calc
# Transitions: 3 total, 1 encrypted (E), 2 plaintext (P)
# Coverage: 33.3%
```

### コンパイラ最適化

3パス最適化：transform融合、デッドステート除去、定数畳み込み。

```python
from axol.core import parse, optimize, run_program

program = parse(source)
optimized = optimize(program)
result = run_program(optimized)
```

### GPUバックエンド

プラグイン可能な配列バックエンド: `numpy`（デフォルト）、`cupy`（NVIDIA GPU）、`jax`。

```python
from axol.core import set_backend
set_backend("cupy")   # NVIDIA GPU
set_backend("jax")    # Google JAX
```

### モジュールシステム

再利用可能で合成可能なプログラム。スキーマ、インポート、サブモジュール実行に対応。

```
@main
import damage_calc from "damage_calc.axol"
s atk=[50] def_val=[10]
: calc=use(damage_calc;in=atk,def_val;out=dmg)
```

### 量子干渉 (Phase 6)

Phase 6では量子干渉 — Grover探索、量子ウォーク — を導入し、量子プログラムの**100%暗号化カバレッジ**を達成しました。Hadamard、Oracle、Diffusionは`TransformOp`（Eクラス）経由で使用される`TransMatrix`オブジェクトを生成するため、既存のオプティマイザと暗号化モジュールが自動的に動作します。

```
@grover_4
s state=[0.5 0.5 0.5 0.5]
: oracle=oracle(state;marked=[3];n=4)
: diffuse=diffuse(state;n=4)
? found state[3]>=0.9
```

### クライアント・サーバーアーキテクチャ

クライアントで暗号化し、信頼できないサーバーで計算：

```
Client (鍵あり)            Server (鍵なし)
  Program ─── encrypt ──► Encrypted Program
  pad_and_encrypt()       run_program() on noise
                    ◄──── Encrypted Result
  decrypt_result()
  ──► Result
```

主要コンポーネント: `KeyFamily(seed)`, `fn_to_matrix()`, `pad_and_encrypt()`, `AxolClient` SDK。

---

## アーキテクチャ

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
                                   │ reuses
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
                    │  Tool-Use API    FastAPI + HTML/JS debugger  │
                    └─────────────────────────────────────────────┘
```

### 内部エンジンの再利用

量子モジュールは`axol/core`を変更せずに再利用します：

| 量子の概念 | Core実装 |
|-----------|---------|
| アトラクタの振幅/軌道 | `FloatVec` |
| アトラクタの相関行列 | `TransMatrix` |
| Tapestryの内部実行 | `Program` + `run_program()` |
| Born ruleの確率 | `operations.measure()` |
| Weaveのtransform構築 | `TransformOp`, `MergeOp` |
| アトラクタ探索の拡散 | `hadamard_matrix()`, `diffusion_matrix()` |

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
- pytest >= 7.4.0 (dev)
- tiktoken >= 0.5.0 (dev、トークン分析用)
- fastapi >= 0.100.0, uvicorn >= 0.23.0 (オプション、Webフロントエンド用)
- cupy-cuda12x >= 12.0.0 (オプション、GPU用)
- jax[cpu] >= 0.4.0 (オプション、JAXバックエンド用)

### Hello World — 量子DSL（宣言 → 織り上げ → 観測）

```python
from axol.quantum import (
    DeclarationBuilder, RelationKind,
    weave, observe, parse_quantum,
)
from axol.core.types import FloatVec

# オプション1: Python API
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

# オプション2: DSL
program = parse_quantum("""
entangle hello(x: float[4]) @ Omega(0.9) Phi(0.8) {
    y <~> transform(x)
}
""")
```

### Hello World — 基盤DSL（ベクトル演算）

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

## APIリファレンス

### 量子モジュール (`axol.quantum`)

```python
# 宣言
DeclarationBuilder(name)           # 宣言を構築するためのFluent API
  .input(name, dim, labels?)       # 入力を追加
  .output(name)                    # 出力をマーク
  .relate(target, sources, kind)   # 関係を追加
  .quality(omega, phi)             # 品質目標を設定
  .build() -> EntangleDeclaration

# 織り上げ
weave(declaration, encrypt?, seed?, optimize?) -> Tapestry

# 観測
observe(tapestry, inputs, seed?) -> Observation
reobserve(tapestry, inputs, count, seed?) -> Observation

# DSL
parse_quantum(source) -> QuantumProgram

# Lyapunov
estimate_lyapunov(trajectory_matrix, steps?) -> float
lyapunov_spectrum(trajectory_matrix, dim, steps?) -> list[float]
omega_from_lyapunov(lyapunov) -> float

# フラクタル
estimate_fractal_dim(attractor_points, method?, phase_space_dim?) -> float
phi_from_fractal(fractal_dim, phase_space_dim) -> float
phi_from_entropy(probs) -> float

# 合成
compose_serial(omega_a, phi_a, lambda_a, d_a, ...) -> tuple
compose_parallel(omega_a, phi_a, lambda_a, d_a, ...) -> tuple
can_reuse_after_observe(lyapunov) -> bool

# コスト
estimate_cost(declaration) -> CostEstimate
```

### コア型

| 型 | 説明 |
|----|------|
| `SuperposedState` | 振幅、ラベル、Born rule確率を持つ名前付き状態 |
| `Attractor` | Lyapunovスペクトル、フラクタル次元、軌道行列を持つストレンジアトラクタ |
| `Tapestry` | グローバルアトラクタとWeaverレポートを持つ`TapestryNode`のグラフ |
| `Observation` | 値、Omega、Phi、確率を持つ崩壊結果 |
| `WeaverReport` | 目標 vs 達成品質、実現可能性、コスト内訳 |
| `CostEstimate` | ノードごとのコスト、クリティカルパス、達成可能な最大Omega/Phi |
| `FloatVec` | 32ビット浮動小数点ベクトル |
| `TransMatrix` | M x N float32行列 |
| `StateBundle` | 名前付きベクトルのコレクション |
| `Program` | 実行可能な遷移のシーケンス |

### 基盤モジュール (`axol.core`)

```python
parse(source) -> Program
run_program(program) -> ExecutionResult
optimize(program) -> Program
set_backend(name)    # "numpy" | "cupy" | "jax"
analyze(program) -> AnalysisResult
dispatch(request) -> dict    # Tool-Use API
```

---

## 使用例

### 1. 宣言 → 織り上げ → 観測（完全パイプライン）

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

### 2. 量子DSLラウンドトリップ

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

### 3. 状態機械（基盤DSL）

```
@state_machine
s state=onehot(0,3)
: advance=transform(state;M=[0 1 0;0 0 1;0 0 1])
? done state[2]>=1
```

### 4. Grover探索（量子干渉）

```
@grover_4
s state=[0.5 0.5 0.5 0.5]
: oracle=oracle(state;marked=[3];n=4)
: diffuse=diffuse(state;n=4)
? found state[3]>=0.9
```

### 5. 暗号化実行

```python
from axol.core import parse, run_program
from axol.core.encryption import encrypt_program, decrypt_state

program = parse("@test\ns v=[1 0 0]\n: t=transform(v;M=[0 1 0;0 0 1;1 0 0])")
encrypted, key = encrypt_program(program)
result = run_program(encrypted)
decrypted = decrypt_state(result.final_state, key)
```

---

## テストスイート

```bash
# 完全テストスイート（545テスト）
pytest tests/ -v

# 量子モジュールテスト（101テスト）
pytest tests/test_quantum_*.py tests/test_lyapunov.py tests/test_fractal.py tests/test_compose.py -v

# 性能ベンチマーク（レポート生成）
pytest tests/test_quantum_performance.py -v -s
pytest tests/test_performance_report.py -v -s

# コアテスト
pytest tests/test_types.py tests/test_operations.py tests/test_program.py tests/test_dsl.py -v

# 暗号化テスト（21テスト）
pytest tests/test_encryption.py -v -s

# 量子干渉テスト（37テスト）
pytest tests/test_quantum.py -v -s

# API + サーバーテスト
pytest tests/test_api.py tests/test_server.py -v

# Webフロントエンド起動
python -m axol.server   # http://localhost:8080
```

現在: **545テスト合格**、0失敗、4スキップ（cupy/jax未インストール）。

---

## ロードマップ

- [x] Phase 1: 型システム（7つのベクトル型 + StateBundle）+ 5つのプリミティブ演算 + 実行エンジン
- [x] Phase 2: DSLパーサー + スパース行列記法 + トークンベンチマーク + 暗号化PoC
- [x] Phase 3: コンパイラ最適化（融合、除去、畳み込み）+ GPUバックエンド
- [x] Phase 4: Tool-Use API + 暗号化モジュール
- [x] Phase 5: モジュールシステム（レジストリ、インポート/使用、合成、スキーマ）
- [x] フロントエンド: FastAPI + HTML/JSビジュアルデバッガ
- [x] Phase 6: 量子干渉（Hadamard/Oracle/Diffusion、100% Eクラスカバレッジ）
- [x] Phase 7: KeyFamily、矩形暗号化、fn_to_matrix、パディング、分岐コンパイル、AxolClient SDK
- [x] Phase 8: カオス理論量子モジュール — 宣言 → 織り上げ → 観測パイプライン
- [x] Phase 8: Lyapunov指数推定（Benettin QR）+ Omega = 1/(1+max(lambda,0))
- [x] Phase 8: フラクタル次元推定（ボックスカウンティング/相関）+ Phi = 1/(1+D/D_max)
- [x] Phase 8: Weaver、Observatory、合成規則、コスト推定、DSLパーサー
- [x] Phase 8: 101の新テスト（合計545、0失敗）
- [ ] Phase 9: 複素振幅 (a+bi) — Shor、QPE、QFT — 完全位相干渉
- [ ] Phase 10: 複数ノードにわたる分散Tapestry織り上げ
- [ ] Phase 11: 適応品質 — 観測中の動的Omega/Phi調整

---

## ライセンス

MITライセンス。詳細は [LICENSE](LICENSE) を参照。
