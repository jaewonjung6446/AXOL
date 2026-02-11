<p align="center">
  <h1 align="center">AXOL</h1>
  <p align="center">
    <strong>トークン効率型ベクトルプログラミング言語</strong>
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
- **9つのプリミティブ演算**ですべての計算を表現：`transform`、`gate`、`merge`、`distance`、`route`（暗号化対応） + `step`、`branch`、`clamp`、`map`（平文演算）
- **疎行列表記法**：密表現のO(N^2)に対しO(N)でスケーリング
- 完全な状態トレーシングによる**決定論的実行**
- **NumPyバックエンド**による大規模ベクトル演算（大規模次元でPureなPythonループより高速）
- **E/Pセキュリティ分類** - 各演算が暗号化（E）または平文（P）に分類され、暗号化カバレッジと表現力のトレードオフを組み込みアナライザーで可視化
- **行列レベル暗号化** - 秘密鍵行列によりプログラムを暗号学的に解読不能にし、シャドーAI問題を根本的に解決

---

## 目次

- [理論的背景](#理論的背景)
- [シャドーAIと行列暗号化](#シャドーaiと行列暗号化)
  - [暗号化証明：全5演算の検証完了](#暗号化証明全5演算の検証完了)
- [平文演算とセキュリティ分類](#平文演算とセキュリティ分類)
- [アーキテクチャ](#アーキテクチャ)
- [クイックスタート](#クイックスタート)
- [DSL構文](#dsl構文)
- [コンパイラ最適化](#コンパイラ最適化)
- [GPUバックエンド](#gpuバックエンド)
- [モジュールシステム](#モジュールシステム)
- [Tool-Use API](#tool-use-api)
- [Webフロントエンド](#webフロントエンド)
- [トークンコスト比較](#トークンコスト比較)
- [ランタイム性能](#ランタイム性能)
- [性能ベンチマーク](#性能ベンチマーク)
- [APIリファレンス](#apiリファレンス)
- [使用例](#使用例)
- [テスト](#テスト)
- [Phase 6: Quantum Axol](#phase-6-quantum-axol)
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

### 9つのプリミティブ演算

Axolは9つのプリミティブ演算を提供します。最初の5つは**暗号化対応（E）** - 暗号化されたデータ上で実行可能です。残りの4つは**平文（P）** - 平文が必要ですが、非線形な表現力を追加します：

| 演算 | セキュリティ | 数学的基盤 | 説明 |
|------|:----------:|----------|------|
| `transform` | **E** | 行列積：`v @ M` | 線形状態変換 |
| `gate` | **E** | アダマール積：`v * g` | 条件付きマスキング（0/1） |
| `merge` | **E** | 加重和：`sum(v_i * w_i)` | ベクトル結合 |
| `distance` | **E** | L2 / コサイン / 内積 | 類似度測定 |
| `route` | **E** | `argmax(v @ R)` | 離散分岐 |
| `step` | **P** | `where(v >= t, 1, 0)` | 閾値によるバイナリゲート変換 |
| `branch` | **P** | `where(g, then, else)` | 条件付きベクトル選択 |
| `clamp` | **P** | `clip(v, min, max)` | 値範囲の制限 |
| `map` | **P** | `f(v)` 要素ごと | 非線形活性化関数（relu、sigmoid、abs、neg、square、sqrt） |

5つのE演算は暗号化計算のための**線形代数基盤**を形成します：
- 状態機械 (transform)
- 条件ロジック (gate)
- 蓄積・集約 (merge)
- 類似度検索 (distance)
- 意思決定 (route)

4つのP演算はAI/MLワークロード向けの**非線形表現力**を追加します：
- 活性化関数 (map: relu, sigmoid)
- 閾値判定 (step + branch)
- 値の正規化 (clamp)

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

## シャドーAIと行列暗号化

### シャドーAI問題

**シャドーAI（Shadow AI）**とは、不正なAIエージェントが独自のビジネスロジックを漏洩、コピー、リバースエンジニアリングするリスクを指します。AIエージェントが自律的にコードを生成・実行する時代において、従来のソースコードは致命的な攻撃対象となります：

- AIエージェントのプロンプトや生成コードが**プロンプトインジェクションで抽出**可能
- Python/C#/JavaScriptのコードは**設計上人間が読める** - 難読化は元に戻せる
- コードに埋め込まれた独自アルゴリズム、意思決定ルール、企業秘密が**平文で露出**
- 従来の難読化（変数名変更、制御フロー平坦化）は障壁をわずかに上げるだけ - ロジックは構造的に無傷で復元可能

### Axolの解決策：行列レベル暗号化

**Axolのすべての計算が行列乗算**（`v @ M`）に帰着されるため、従来のプログラミング言語では不可能な数学的性質が利用可能になります：**相似変換（similarity transformation）暗号化**。

秘密の可逆鍵行列**K**が与えられると、あらゆるAxolプログラムを暗号化できます：

```
元のプログラム：     state  -->  M  -->  new_state
暗号化プログラム：   state' -->  M' -->  new_state'

ここで：
  M' = K^(-1) @ M @ K          （暗号化された演算行列）
  state' = state @ K            （暗号化された初期状態）
  result  = result' @ K^(-1)    （復号された最終出力）
```

これは難読化ではなく**暗号学的変換**です。暗号化されたプログラムは：

1. **暗号化ドメインで正常に実行**（行列代数が共役変換を保存）
2. **暗号化された出力を生成** - K^(-1)なしでは解読不能
3. **すべてのビジネスロジックを隠蔽** - 行列M'はKなしでMと数学的に無関係
4. **リバースエンジニアリングに耐性** - M'からKを復元することは、Nが大きくなるにつれて困難になる行列分解問題です。一般的な場合に対する既知の多項式時間アルゴリズムは存在しませんが、正式な暗号学的困難性の証明は進行中の研究分野です

### 具体例

```
# 元：状態機械の遷移行列（ビジネスロジックが見える）
M = [0 1 0]    # IDLE -> RUNNING
    [0 0 1]    # RUNNING -> DONE
    [0 0 1]    # DONE -> DONE（吸収状態）

# 秘密鍵Kで暗号化後：
M' = [0.73  -0.21   0.48]    # Kなしでは無意味
     [0.15   0.89  -0.04]    # 状態機械構造の推測不能
     [0.52   0.33   0.15]    # ランダムノイズに見える
```

暗号化されたプログラムは正常に実行されますが（行列代数が`K^(-1)(KvM)K = vM`を保証）、DSLテキストには**暗号化された行列のみ**が含まれます。`.axol`ファイル全体が漏洩しても：

- 状態名が見えない（ベクトルが暗号化）
- 遷移ロジックが見えない（行列が暗号化）
- ターミナル条件が無意味（閾値が暗号化された値で動作）

### 従来の言語では不可能な理由

| 属性 | Python/C#/JS | FHE | Axol |
|------|-------------|-----|------|
| コードセマンティクス | 平文の制御フロー | 暗号化（任意の計算） | 行列乗算 |
| 難読化 | 元に戻せる | 該当なし | 該当なし |
| 暗号化 | 不可能（パース可能でなければならない） | 完全（任意の計算） | 線形演算のみ（9つ中5つ） |
| 性能オーバーヘッド | 該当なし | 1000〜10000倍 | 約0%（パイプラインモード） |
| 複雑性 | 該当なし | 非常に高い | 低い（鍵行列のみ） |
| コード漏洩時 | 全ロジック露出 | 暗号化済み | ランダムに見える数値 |
| 鍵分離 | 不可能 | 必須 | 鍵行列を別保管（HSM、セキュアエンクレーブ） |
| 暗号化後の正確性 | 該当なし | 数学的に保証 | 数学的に保証 |

### セキュリティアーキテクチャ

```
  [開発者]                       [デプロイ環境]
     |                              |
  元の.axol                    暗号化された.axol
  （読めるロジック）             （暗号化された行列）
     |                              |
     +--- K（秘密鍵）------------>|
     |    HSM/セキュアエンクレーブ  |
     v                              v
  encrypt(M, K) = K^(-1)MK     run_program(暗号化プログラム)
                                     |
                                暗号化された出力
                                     |
                                decrypt(output, K^(-1))
                                     |
                                実際の結果
```

秘密鍵行列Kは以下のように管理できます：
- **ハードウェアセキュリティモジュール（HSM）**に保管
- **鍵管理サービス（KMS）**で管理
- プログラム構造を変更せずに定期的にローテーション
- デプロイ環境ごと（dev/staging/prod）に異なる鍵を使用

Axolは行列ベースの計算に対する完全準同型暗号（FHE）の軽量な代替手段を提供します。FHE（任意の計算をサポートするが高オーバーヘッド）とは異なり、Axolの相似変換は効率的ですが線形演算に限定されます。このトレードオフにより、5つの暗号化演算で十分な特定のユースケースにおいて実用的です。

### 暗号化証明：全5演算の検証完了

Axolの全5演算の暗号化互換性が**数学的に証明・テスト済み**です（`tests/test_encryption.py`、21テスト）：

| 演算 | 暗号化方法 | 鍵制約 | 状態 |
|------|----------|--------|------|
| `transform` | `M' = K^(-1) M K`（相似変換） | 任意の可逆行列K | **証明済み** |
| `gate` | `diag(g)`行列に変換後、transformと同様 | 任意の可逆行列K | **証明済み** |
| `merge` | 線形性：`w*(v@K) = (wv)@K`（自動互換） | 任意の可逆行列K | **証明済み** |
| `distance` | `\|\|v@K\|\| = \|\|v\|\|`（直交行列がノルム保存） | 直交行列K | **証明済み** |
| `route` | `R' = K^(-1) R`（左乗算のみ） | 任意の可逆行列K | **証明済み** |

**複雑なマルチ演算プログラムも検証済み：**

- HP減衰（transform + mergeループ）- 暗号化/復号結果が一致
- 3状態FSM（連鎖transform）- 暗号化ドメインで正確な状態遷移
- 戦闘パイプライン（transform + gate + merge）- 3演算連鎖、誤差 < 0.001
- 20状態オートマトン（疎行列、19ステップ）- 暗号化実行結果が元と一致
- 50x50大規模行列 - float32精度を維持

**テストで証明されたセキュリティ特性：**

- 暗号化された行列はランダムノイズに見える（疎 -> 密、構造識別不能）
- 異なる鍵は完全に異なる暗号化結果を生成
- 100個のランダム鍵によるブルートフォースで元の行列を復元不能
- OneHotベクトル構造が暗号化後に完全に隠蔽

---

## 平文演算とセキュリティ分類

### なぜ平文演算が必要か？

元の5つの暗号化対応演算は**線形**であり、線形変換のみ表現可能です。現実のAI/MLワークロードの多くは**非線形**演算（活性化関数、条件分岐、値のクランプ）を必要とします。4つの新しい平文演算がこのギャップを埋めます。

### SecurityLevel列挙型

すべての演算は`SecurityLevel`を持ちます：

```python
from axol.core import SecurityLevel

SecurityLevel.ENCRYPTED  # "E" - 暗号化データ上で実行可能
SecurityLevel.PLAINTEXT  # "P" - 平文が必要
```

### 暗号化カバレッジアナライザー

組み込みのアナライザーが、プログラムのどの程度が暗号化状態で実行可能かを報告します：

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
# Encryptable keys: (E演算のみがアクセスするキー)
# Plaintext keys: (P演算がアクセスするキー)
```

### セキュリティと表現力のトレードオフ

P演算を追加すると表現力が向上しますが、暗号化カバレッジが低下します：

| プログラムタイプ | 暗号化カバレッジ | 表現力 |
|----------------|----------------|--------|
| E演算のみ | 100% | 線形のみ |
| E+P混合 | 30〜70%（典型） | 完全（非線形） |
| P演算のみ | 0% | 完全（非線形） |

非線形演算（活性化関数、条件分岐）を必要とするプログラムは、部分的な暗号化カバレッジを受け入れる必要があります。組み込みのアナライザーを使用してプログラムのカバレッジを測定し、平文アクセスが必要なキーを特定してください。

### 新演算のトークンコスト（Python vs C# vs Axol DSL）

| プログラム | Python | C# | Axol DSL | vs Python | vs C# |
|-----------|-------:|---:|--------:|---------:|------:|
| ReLU活性化 | 48 | 82 | 28 | 42% | 66% |
| 閾値選択 | 140 | 184 | 80 | 43% | 57% |
| 値クランプ | 66 | 95 | 31 | 53% | 67% |
| Sigmoid活性化 | 57 | 88 | 28 | 51% | 68% |
| ダメージパイプライン | 306 | 326 | 155 | 49% | 53% |
| **合計** | **617** | **775** | **322** | **48%** | **59%** |

### 新演算のランタイム（dim=10,000）

| 演算 | Pythonループ | Axol（NumPy） | 高速化 |
|------|----------:|----------:|--------:|
| ReLU | 575 us | 21 us | **27x** |
| Sigmoid | 1.7 ms | 42 us | **40x** |
| Step+Branch | 889 us | 96 us | **9x** |
| Clamp | 937 us | 16 us | **58x** |
| ダメージパイプライン | 3.8 ms | 191 us | **20x** |

---

## アーキテクチャ

```
                                          +-------------+
  .axolソース -----> パーサー (dsl.py) --> | Program     |
                         |                | + optimize()|
                         v                +------+------+
                    モジュールシステム            |
                    (module.py)                  v
                      - import             +-----------+    +-----------+
                      - use()              |  エンジン  |--->|  検証器   |
                      - compose()          |(program.py)|    |(verify.py)|
                                           +-----------+    +-----------+
                                                |
                    +-----------+    +----------+----------+
                    | バックエンド|<---|    演算モジュール    |
                    |(backend.py)|    | (operations.py)     |
                    | numpy/cupy|    +---------------------+
                    | /jax      |               |
                    +-----------+    +-----------+----------+
                                    |      型システム       |
                                    |   (types.py)         |
                    +-----------+   +----------------------+
                    |  暗号化   |   +-----------+
                    |(encryption|   |アナライザー|
                    |       .py)|   |(analyzer  |
                    +-----------+   |       .py)|
                                    +-----------+
                    +-----------+    +-----------+
                    | Tool API  |    |  Server   |
                    |(api/)     |    |(server/)  |
                    | dispatch  |    | FastAPI   |
                    | tools     |    | HTML/JS   |
                    +-----------+    +-----------+
```

### モジュール概要

| モジュール | 説明 |
|-----------|------|
| `axol.core.types` | 7つのベクトル型（`BinaryVec`、`IntVec`、`FloatVec`、`OneHotVec`、`GateVec`、`TransMatrix`）+ `StateBundle` |
| `axol.core.operations` | 9つのプリミティブ演算：`transform`、`gate`、`merge`、`distance`、`route`、`step`、`branch`、`clamp`、`map_fn` |
| `axol.core.program` | 実行エンジン：`Program`、`Transition`、`run_program`、`SecurityLevel`、`StepOp`/`BranchOp`/`ClampOp`/`MapOp` |
| `axol.core.verify` | 状態検証（exact/cosine/euclidean マッチング） |
| `axol.core.dsl` | DSLパーサー：`parse(source) -> Program`（`import`/`use()`サポート） |
| `axol.core.optimizer` | 3パスコンパイラ最適化：transform融合、デッドステート除去、定数畳み込み |
| `axol.core.backend` | プラガブル配列バックエンド：`numpy`（デフォルト）、`cupy`、`jax` |
| `axol.core.encryption` | 相似変換暗号化：`encrypt_program`、`decrypt_state`（E/P対応） |
| `axol.core.analyzer` | 暗号化カバレッジアナライザー：`analyze(program) -> AnalysisResult`（E/P分類） |
| `axol.core.module` | モジュールシステム：`Module`、`ModuleRegistry`、`compose()`、スキーマ検証 |
| `axol.api` | AIエージェント向けTool-Use API：`dispatch(request)`、`get_tool_definitions()` |
| `axol.server` | FastAPI Webサーバー + バニラHTML/JSビジュアルデバッガーフロントエンド |

---

## クイックスタート

### インストール

```bash
# リポジトリをクローン
git clone https://github.com/your-username/AXOL.git
cd AXOL

# 依存関係をインストール
pip install -e ".[dev]"
```

### 必要要件

- Python 3.11+
- NumPy >= 1.24.0
- pytest >= 7.4.0（開発用）
- tiktoken >= 0.5.0（開発用、トークン分析）
- fastapi >= 0.100.0、uvicorn >= 0.23.0（オプション、Webフロントエンド用）
- cupy-cuda12x >= 12.0.0（オプション、GPU用）
- jax[cpu] >= 0.4.0（オプション、JAXバックエンド用）

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
# --- 暗号化対応（E）演算 ---

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

# --- 平文（P）演算 ---

# step：閾値によるバイナリゲート変換
: mask=step(scores;t=0.5)->gate_out

# branch：条件付きベクトル選択（->out_keyが必要）
: selected=branch(gate_key;then=high,else=low)->result

# clamp：値を範囲内にクリップ
: safe=clamp(values;min=0,max=100)

# map：要素ごとの非線形関数（relu、sigmoid、abs、neg、square、sqrt）
: activated=map(x;fn=relu)
: prob=map(logits;fn=sigmoid)->output
```

### 行列形式

```
# 密：行は;で区切り
M=[0.8]                               # 1x1
M=[1 0;0 1]                           # 2x2 単位行列
M=[0 1 0;0 0 1;0 0 1]                # 3x3 シフト行列

# 疎：非ゼロ要素のみ表記
M=sparse(100x100;0,1=1 1,2=1 99,99=1) # 100x100、100エントリ
```

### ターミナル条件

```
? done count>=5              # count[0] >= 5で終了
? finished state[2]>=1       # state[2] >= 1で終了（インデックスアクセス）
? end hp<=0                  # hp[0] <= 0で終了
```

`?`行がない場合は**パイプラインモード**で実行されます（すべての遷移が1回実行）。

### コメント

```
# これはコメントです
@my_program
# コメントはどこにでも記述可能
s v=[1 2 3]
: t=transform(v;M=[1 0 0;0 1 0;0 0 1])
```

---

## コンパイラ最適化

`optimize()`は3つのパスを適用してプログラムサイズを削減し、定数を事前計算します：

```python
from axol.core import parse, optimize, run_program

program = parse(source)
optimized = optimize(program)   # 融合 + 除去 + 畳み込み
result = run_program(optimized)
```

### パス1：Transform融合

同じキーに対する連続した`TransformOp`を単一の行列乗算に融合します：

```
# 最適化前：2つの遷移、反復ごとに2回の行列乗算
: t1=transform(v;M=[0 1 0;0 0 1;1 0 0])
: t2=transform(v;M=[2 0 0;0 2 0;0 0 2])

# 最適化後：1つの遷移、1回の行列乗算（M_fused = M1 @ M2）
: t1+t2=transform(v;M_fused)
```

- `CustomOp`の境界を越えません
- 固定点反復により3つ以上のチェーンを処理
- 2つのtransformを持つパイプライン：**遷移数-50%、実行時間-45%**

### パス2：デッドステート除去

遷移で一度も読み取られない初期状態ベクトルを削除します：

```
s used=[1 0]  unused=[99 99]   # unusedは参照されない
: t=transform(used;M=[...])

# 最適化後：unusedが初期状態から削除
```

- `CustomOp`に対して保守的（すべての状態を保持）
- `terminal_key`は常に「読み取り済み」として扱われる

### パス3：定数畳み込み

不変キー（書き込まれることのないキー）に対するtransformを事前計算します：

```
s constant=[1 0 0]
: t=transform(constant;M=[0 1 0;0 0 1;1 0 0])->result

# 最適化後：遷移が除去され、result=[0,1,0]が初期状態に格納
```

---

## GPUバックエンド

`numpy`（デフォルト）、`cupy`（NVIDIA GPU）、`jax`をサポートするプラガブル配列バックエンド：

```python
from axol.core import set_backend, get_backend_name

set_backend("numpy")   # デフォルト - CPU
set_backend("cupy")    # NVIDIA GPU（cupyのインストールが必要）
set_backend("jax")     # Google JAX（jaxのインストールが必要）
```

オプションバックエンドのインストール：

```bash
pip install axol[gpu]   # cupy-cuda12x
pip install axol[jax]   # jax[cpu]
```

既存のコードはすべて透過的に動作します - バックエンド切り替えはグローバルで、すべてのベクトル/行列演算に影響します。

---

## モジュールシステム

スキーマ、インポート、サブモジュール実行を備えた、再利用・合成可能なプログラム。

### モジュール定義

```python
from axol.core.module import Module, ModuleSchema, VecSchema, ModuleRegistry

schema = ModuleSchema(
    inputs=[VecSchema("atk", "float", 1), VecSchema("def_val", "float", 1)],
    outputs=[VecSchema("dmg", "float", 1)],
)
module = Module(name="damage_calc", program=program, schema=schema)
```

### レジストリとファイル読み込み

```python
registry = ModuleRegistry()
registry.load_from_file("damage_calc.axol")
registry.resolve_import("heal", relative_to="main.axol")
```

### DSLのインポートとuse構文

```
@main
import damage_calc from "damage_calc.axol"
s atk=[50] def_val=[10]
: calc=use(damage_calc;in=atk,def_val;out=dmg)
```

### プログラム合成

```python
from axol.core.module import compose
combined = compose(program_a, program_b, name="combined")
```

---

## Tool-Use API

AIエージェントがAxolプログラムをパース、実行、検証するためのJSON呼び出しインターフェース：

```python
from axol.api import dispatch

# パース
result = dispatch({"action": "parse", "source": "@prog\ns v=[1]\n: t=transform(v;M=[2])"})
# -> {"program_name": "prog", "state_keys": ["v"], "transition_count": 1, "has_terminal": false}

# 実行
result = dispatch({"action": "run", "source": "...", "optimize": True})
# -> {"final_state": {"v": [2.0]}, "steps_executed": 1, "terminated_by": "pipeline_end"}

# ステップごとの検査
result = dispatch({"action": "inspect", "source": "...", "step": 1})

# 演算一覧
result = dispatch({"action": "list_ops"})

# 期待出力の検証
result = dispatch({"action": "verify", "source": "...", "expected": {"v": [2.0]}})
```

AIエージェント向けツール定義（JSON Schema）は`get_tool_definitions()`で取得できます。

---

## Webフロントエンド

バニラHTML/JSビジュアルデバッガーを搭載したFastAPIサーバー：

```bash
pip install axol[server]    # fastapi + uvicorn
python -m axol.server       # http://localhost:8080
```

### 機能

| パネル | 説明 |
|--------|------|
| **DSLエディタ** | サンプルドロップダウン付きの構文編集 |
| **実行** | 実行/最適化ボタン、結果サマリー（ステップ数、時間、終了条件） |
| **トレースビューア** | ステップごとの状態テーブル（前へ/次へ/再生コントロール付き） |
| **状態チャート** | Chart.js時系列グラフ（X=ステップ、Y=ベクトル値） |
| **暗号化デモ** | 元の行列 vs 暗号化行列のヒートマップ、暗号化/実行/復号ワークフロー |
| **パフォーマンス** | オプティマイザーの前後比較、トークンコスト分析 |

### APIエンドポイント

```
POST /api/parse       - DSLソースをパース
POST /api/run         - パース + 実行 + 完全なトレース
POST /api/optimize    - オプティマイザーの前後比較
POST /api/encrypt     - プログラムの暗号化 + 実行 + 復号
GET  /api/examples    - 組み込みサンプルプログラム
GET  /api/ops         - 演算の説明
POST /api/token-cost  - トークン数分析（Axol vs Python vs C#）
POST /api/module/run  - サブモジュール付きプログラムの実行
```

---

## トークンコスト比較

`tiktoken` cl100k_baseトークナイザーで測定（GPT-4 / Claude使用）。

> **注意**: トークン削減は、ベクトル/行列演算に自然にマッピングされるプログラム（状態機械、線形変換、加重和）で測定されています。汎用プログラミングタスク（文字列処理、I/O、API呼び出し）はAxolで表現できません。以下の比較はAxolの最良ケースであり、平均的なケースではありません。

### Python vs Axol DSL

| プログラム | Python | Axol DSL | 削減 |
|-----------|--------|----------|------|
| Counter (0->5) | 32 | 33 | -3.1% |
| State Machine (3-state) | 67 | 47 | 29.9% |
| HP Decay (3 rounds) | 51 | 32 | 37.3% |
| RPG Damage Calc | 130 | 90 | 30.8% |
| 100-State Automaton | 1,034 | 636 | 38.5% |
| **合計** | **1,314** | **838** | **36.2%** |

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

1. **単純なプログラム**（counter、hp_decay）：DSLはPythonと同等です。DSL構文のオーバーヘッドは、単純なプログラムにおけるPythonの最小限の構文とほぼ同じです。
2. **構造化プログラム**（combat、data_heavy、pattern_match）：DSLはPython比**50〜68%**、C#比**67〜75%**のトークン削減を達成。ベクトル表現がクラス定義、制御フロー、ボイラープレートを排除します。
3. **大規模状態空間**（100+状態）：疎行列表記法がPython比**約38%**、C#比**約27%**の一貫した削減を提供し、O(N)スケーリング vs O(N^2)を実現します。

### Tool-Use API vs Python + FHE

**完全な暗号化ワークフロー**を比較した場合（DSL構文だけではなく）：

| タスク | Python + FHE | Axol Tool-Use API | 削減率 |
|--------|-------------|-------------------|--------|
| 暗号化分岐 | 約150トークン | 約30トークン | 80% |
| 暗号化ステートマシン | 約200トークン | 約35トークン | 82% |
| 暗号化Grover検索 | 約250トークン | 約25トークン | 90% |

削減は**構文ではなく抽象化**によるものです：LLMは暗号化コード（鍵生成、暗号化、復号化）を一切見ることなく、Tool-Use APIが内部的に処理します。

---

## ランタイム性能

AxolはNumPyを計算バックエンドとして使用します。

> **注意**: ランタイムベンチマークはPureなPythonループとAxolのNumPyバックエンドを比較しています。高速化は主にNumPyの最適化されたC/Fortran実装によるものであり、Axol固有の最適化ではありません。NumPyを直接使用するPythonコードも同様の速度を達成します。

### 小規模ベクトル（dim < 100）

| 次元 | Pythonループ | Axol（NumPy） | 優位 |
|------|------------|--------------|------|
| dim=4 | ~6 us | ~11 us | Python 2x |
| dim=100 | ~14 us | ~20 us | Python 1.4x |

小規模ベクトルでは、NumPyの呼び出しオーバーヘッドのためPythonのネイティブループが高速です。これは想定通りであり許容範囲です - 小規模プログラムはいずれにしても高速です。

### 大規模ベクトル（dim >= 1000）

| 次元 | Pythonループ | Axol（NumPy） | 優位 |
|------|------------|--------------|------|
| dim=1,000（行列積） | ~129 ms | ~0.2 ms | **573x**（NumPy） |
| dim=10,000（行列積） | ~14,815 ms | ~381 ms | **39x**（NumPy） |

大規模ベクトル演算（行列乗算）において、NumPyの最適化されたC/Fortran BLASバックエンド（Axolが使用）は純粋なPythonループよりも**桁違いに高速**です。NumPyを直接使用するPythonコードも同様の高速化を達成します。

### 使い分けガイド

| シナリオ | 推奨 |
|---------|------|
| AIエージェントの暗号化計算 | Axol Tool-Use API（LLMは暗号化知識不要） |
| 大規模状態空間（100+次元） | Axol（NumPy高速化 + 疎表記法） |
| シンプルなスクリプト（10行以下） | Python（オーバーヘッドが少ない） |
| 人間が読めるビジネスロジック | Python/C#（慣れた構文） |

### 制限事項

- **限定されたドメイン**: Axolはベクトル/行列計算のみ表現可能です。文字列処理、I/O、ネットワーキング、汎用プログラミングはサポートされていません。
- **LLM学習データなし**: PythonやJavaScriptとは異なり、AxolコードでトレーニングされたLLMはありません。AIエージェントはコンテキストに例がなければ正しいAxolプログラムの生成に苦労する可能性があります。
- **線形演算のみ暗号化**: 9つの演算のうち5つのみが暗号化実行をサポートします。非線形演算（step、branch、clamp、map）を使用するプログラムは暗号化カバレッジが低下します。
- **ループモード暗号化オーバーヘッド**: ループモードの暗号化プログラムはターミナル条件を評価できず、max_iterationsまで実行されます。ベンチマークで400倍以上のオーバーヘッドが発生します。
- **トークン削減はドメイン固有**: DSLトークン削減はドメイン固有（ベクトル/行列プログラムで30〜50%）です。ただし、Tool-Use APIは暗号化を完全に抽象化し、Python+FHE比で80〜85%の削減を提供します。

---

## 性能ベンチマーク

`pytest tests/test_performance_report.py -v -s`で自動生成されます。詳細は[PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md)を参照してください。

> **注意**: ランタイムベンチマークはPureなPythonループとAxolのNumPyバックエンドを比較しています。高速化は主にNumPyの最適化されたC/Fortran実装によるものであり、Axol固有の最適化ではありません。NumPyを直接使用するPythonコードも同様の速度を達成します。

### トークン効率（Axol vs Python vs C#）

| プログラム | Axol | Python | C# | vs Python | vs C# |
|-----------|------|--------|----|-----------|-------|
| Counter (0->5) | 11 | 45 | 78 | **76%削減** | **86%削減** |
| 3-State FSM | 14 | 52 | 89 | **73%削減** | **84%削減** |
| HP Decay | 14 | 58 | 95 | **76%削減** | **85%削減** |
| Combat Pipeline | 14 | 55 | 92 | **75%削減** | **85%削減** |
| Matrix Chain | 21 | 60 | 98 | **65%削減** | **79%削減** |

平均：Pythonと比較して**74%少ないトークン**、C#と比較して**85%少ないトークン**。

### 次元別実行時間

| 次元 | 平均時間 |
|------|---------|
| 4 | 0.25 ms |
| 100 | 0.17 ms |
| 1,000 | 1.41 ms |

### オプティマイザー効果

| プログラム | 最適化前 | 最適化後 | 時間削減 |
|-----------|---------|---------|---------|
| Pipeline (2 transforms) | 2 transitions | 1 transition | **-45%** |
| Counter (loop) | 2 transitions | 2 transitions | - |
| FSM (loop) | 2 transitions | 2 transitions | - |

Transform融合は、連続した行列演算を持つパイプラインプログラムで最も効果的です。

### 暗号化オーバーヘッド

| プログラム | 平文 | 暗号化 | オーバーヘッド |
|-----------|------|--------|-------------|
| Pipeline (1 pass) | 0.12 ms | 0.12 ms | **約0%** |
| 3-State FSM (loop) | 0.62 ms | 276.8 ms | +44,633% |

パイプラインモード：オーバーヘッドは無視できます。ループモード：暗号化されたターミナル条件が早期終了をトリガーできないため、`max_iterations`まで実行が継続し、高いオーバーヘッドが発生します。

### スケーリング（N状態オートマトン）

| 状態数 | トークン | 実行時間 |
|--------|---------|---------|
| 5 | 28 | 1.6 ms |
| 20 | 388 | 4.3 ms |
| 50 | 2,458 | 12.9 ms |
| 100 | 9,908 | 27.9 ms |
| 200 | 39,808 | 59.2 ms |

疎行列表記法によりトークンは**O(N)**で増加します（Python/C#のO(N^2)と比較）。実行時間は行列乗算によりおよそO(N^2)で増加しますが、200状態のプログラムでも60ms以下です。

---

## APIリファレンス

### `parse(source, registry=None, source_path=None) -> Program`

Axol DSLソーステキストを実行可能な`Program`オブジェクトにパースします。

```python
from axol.core import parse
program = parse("@test\ns v=[1 2 3]\n: t=transform(v;M=[1 0 0;0 1 0;0 0 1])")

# import/useサポート付きのモジュールレジストリ
from axol.core.module import ModuleRegistry
registry = ModuleRegistry()
program = parse(source, registry=registry, source_path="main.axol")
```

### `run_program(program: Program) -> ExecutionResult`

プログラムを実行し、結果を返します。

```python
from axol.core import run_program
result = run_program(program)
result.final_state     # 最終ベクトル値を含むStateBundle
result.steps_executed  # 遷移ステップの総数
result.terminated_by   # "pipeline_end" | "terminal_condition" | "max_iterations"
result.trace           # デバッグ用TraceEntryのリスト
result.verification    # expected_stateが設定されている場合のVerifyResult
```

### `optimize(program, *, fuse=True, eliminate_dead=True, fold_constants=True) -> Program`

元のプログラムを変更せずに最適化したプログラムを返します。

```python
from axol.core import optimize
optimized = optimize(program)                          # 全パス
optimized = optimize(program, fold_constants=False)    # 選択的パス
```

### `set_backend(name) / get_backend() / to_numpy(arr)`

配列計算バックエンドを切り替えます。

```python
from axol.core import set_backend, get_backend, to_numpy
set_backend("cupy")     # GPUに切り替え
xp = get_backend()      # cupyモジュールを返す
arr = to_numpy(gpu_arr) # numpyに変換
```

### `dispatch(request) -> dict`

AIエージェント向けTool-Use APIのエントリポイント。

```python
from axol.api import dispatch
result = dispatch({"action": "run", "source": "...", "optimize": True})
```

### ベクトル型

| 型 | 説明 | ファクトリメソッド |
|----|------|-----------------|
| `FloatVec` | 32ビット浮動小数点 | `from_list([1.0, 2.0])`、`zeros(n)`、`ones(n)` |
| `IntVec` | 64ビット整数 | `from_list([1, 2])`、`zeros(n)` |
| `BinaryVec` | 要素が{0, 1} | `from_list([0, 1])`、`zeros(n)`、`ones(n)` |
| `OneHotVec` | 正確に1つの1.0 | `from_index(idx, n)`、`from_list(...)` |
| `GateVec` | 要素が{0.0, 1.0} | `from_list([1.0, 0.0])`、`zeros(n)`、`ones(n)` |
| `TransMatrix` | M x N float32行列 | `from_list(rows)`、`identity(n)`、`zeros(m, n)` |

### 演算ディスクリプター

```python
from axol.core.program import (
    # 暗号化対応（E）演算
    TransformOp,  # TransformOp(key="v", matrix=M, out_key=None)
    GateOp,       # GateOp(key="v", gate_key="g", out_key=None)
    MergeOp,      # MergeOp(keys=["a","b"], weights=w, out_key="out")
    DistanceOp,   # DistanceOp(key_a="a", key_b="b", metric="euclidean")
    RouteOp,      # RouteOp(key="v", router=R, out_key="_route")
    # 平文（P）演算
    StepOp,       # StepOp(key="v", threshold=0.0, out_key=None)
    BranchOp,     # BranchOp(gate_key="g", then_key="a", else_key="b", out_key="out")
    ClampOp,      # ClampOp(key="v", min_val=-inf, max_val=inf, out_key=None)
    MapOp,        # MapOp(key="v", fn_name="relu", out_key=None)
    # エスケープハッチ
    CustomOp,     # CustomOp(fn=callable, label="name")  -- security=P
)
```

### アナライザー

```python
from axol.core import analyze

result = analyze(program)
result.coverage_pct        # E / total * 100
result.encrypted_count     # E遷移の数
result.plaintext_count     # P遷移の数
result.encryptable_keys    # E演算のみがアクセスするキー
result.plaintext_keys      # P演算がアクセスするキー
print(result.summary())    # 人間が読めるレポート
```

### 検証

```python
from axol.core import verify_states, VerifySpec

result = verify_states(
    expected=expected_bundle,
    actual=actual_bundle,
    specs={"hp": VerifySpec.exact(tolerance=0.01)},
    strict_keys=False,
)
print(result.passed)    # True/False
print(result.summary()) # 詳細レポート
```

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

### 5. ReLU活性化（map）

```
@relu
s x=[-2 0 3 -1 5]
:act=map(x;fn=relu)
# Result: x = [0, 0, 3, 0, 5]
```

### 6. 閾値選択（step + branch）

```
@threshold_select
s scores=[0.3 0.8 0.1 0.9] high=[100 200 300 400] low=[1 2 3 4]
:s1=step(scores;t=0.5)->mask
:b1=branch(mask;then=high,else=low)->result
# mask = [0, 1, 0, 1]
# result = [1, 200, 3, 400]
```

### 7. ダメージパイプライン（新4演算すべて使用）

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

### 8. 100状態オートマトン（疎行列）

```
@auto_100
s s=onehot(0,100)
: step=transform(s;M=sparse(100x100;0,1=1 1,2=1 ... 98,99=1 99,99=1))
? done s[99]>=1
```

---

## テスト

```bash
# 全テスト（約320件）
pytest tests/ -v

# コアテスト
pytest tests/test_types.py tests/test_operations.py tests/test_program.py tests/test_dsl.py -v

# オプティマイザーテスト（18件）
pytest tests/test_optimizer.py -v

# バックエンドテスト（13件、cupy/jaxが未インストールの場合スキップ）
pytest tests/test_backend.py -v

# Tool-Use APIテスト（20件）
pytest tests/test_api.py -v

# モジュールシステムテスト（18件）
pytest tests/test_module.py -v

# 暗号化証明テスト（21件）
pytest tests/test_encryption.py -v -s

# 新演算テスト - step/branch/clamp/map（44件）
pytest tests/test_new_ops.py -v

# アナライザーテスト - E/Pカバレッジ分析（7件）
pytest tests/test_analyzer.py -v

# 新演算ベンチマーク - Python vs C# vs Axol（15件）
pytest tests/test_benchmark_new_ops.py -v -s

# サーバーエンドポイントテスト（13件、fastapi必要）
pytest tests/test_server.py -v

# 性能レポート（PERFORMANCE_REPORT.mdを生成）
pytest tests/test_performance_report.py -v -s

# トークンコスト比較
pytest tests/test_token_cost.py -v -s

# 3言語ベンチマーク（Python vs C# vs Axol）
pytest tests/test_benchmark_trilingual.py -v -s

# Webフロントエンドの起動
python -m axol.server   # http://localhost:8080
```

現在のテスト数：**約320件**、すべて合格（4件スキップ：cupy/jax未インストール）。

---

## Phase 6: Quantum Axol

Phase 6はAxolに**量子干渉**を導入します — 非線形ロジックを線形行列演算として再表現し、量子プログラムの**100%暗号化カバレッジ**を実現します。また、LLMが暗号化知識ゼロで使える**暗号化透過Tool-Use API**を導入します。

### 背景理論

#### 核心的問題

Axolの暗号化は**相似変換（similarity transformation）**に基づいています：`M' = K⁻¹MK`。これは線形演算（`transform`、`gate`、`merge`、`distance`、`route`）に完全に動作しますが、非線形演算（`step`、`branch`、`clamp`、`map`）では失敗します。非線形関数は線形鍵変換と可換ではないためです。

これが根本的なトレードオフを生みます：

| プログラム種別 | 暗号化カバレッジ | 表現力 |
|-------------|---------------|-------|
| 線形演算のみ（E） | 100% | 線形代数のみ |
| 混合 E+P | 30-70% | 完全（非線形含む） |
| **量子演算（Phase 6）** | **100%** | **Groverレベル検索、量子ウォーク** |

#### 解決策：量子干渉

核心的洞察：**量子アルゴリズムは純粋な線形演算のみで非線形的な振る舞いを実現します**。例えばGrover検索は、条件分岐なしにO(√N)時間でマーク付き項目を見つけます — 行列乗算のみを使って：

1. **Hadamard**（H）：負の振幅を含む均等重ね合わせを生成
2. **Oracle**（O）：マーク付き項目の符号を反転する対角行列（-1エントリ）
3. **Diffusion**（D）：平均の周りに状態を反射（2|s⟩⟨s| - I）

3つとも**実直交行列**です → `state @ O @ D`で合成すると単純な行列乗算チェーンになります — Axolの`TransformOp`（E-class）と完全互換です。

#### 符号付き振幅で十分な理由

量子コンピューティングは通常**複素数**振幅（a + bi）を使用します。しかし、Grover検索や量子ウォークを含む多くの有用な量子アルゴリズムは**符号付き実数**振幅のみを必要とします。`FloatVec`はすでに負のfloat32をサポートしているため、量子干渉の有効化コストは実質ゼロです：

| ティア | 振幅型 | 干渉レベル | 実装コスト | アルゴリズム |
|-------|--------|----------|----------|-----------|
| 0（Phase 6以前） | 非負実数 | なし | — | 古典的FSM |
| **1（Phase 6）** | **符号付き実数** | **Groverレベル** | **約0** | **Grover検索、量子ウォーク** |
| 2（将来） | 複素数（a+bi） | 完全位相 | メモリ2倍、計算4倍 | Shor、QPE、QFT |

#### 数学的検証：N=4のGrover

均等重ね合わせ `|s⟩ = [0.5, 0.5, 0.5, 0.5]` から開始、ターゲットインデックス3：

```
ステップ1 — Oracle（インデックス3をマーク）：
  O = diag(1, 1, 1, -1)
  state = [0.5, 0.5, 0.5, -0.5]    ← 符号反転で干渉を生成

ステップ2 — Diffusion（平均の周りで反射）：
  D = 2|s⟩⟨s| - I
  state = [0, 0, 0, 1.0]           ← ターゲットで建設的干渉

結果：正確に1回の反復でターゲット発見、確率 |1.0|² = 100%。
```

N=4では、単一のOracle+Diffusion反復で**完全な**判別を達成します。より大きなNでの最適反復回数は ⌊π/4 · √N⌋ です。

### アーキテクチャ

#### 新コンポーネント

```
operations.py        program.py          dsl.py
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ measure()    │    │ MeasureOp    │    │ measure()    │
│ hadamard_m() │    │ (P-class)    │    │ hadamard()   │
│ oracle_m()   │    │              │    │ oracle()     │
│ diffusion_m()│    │ OpKind.      │    │ diffuse()    │
│              │    │   MEASURE    │    │              │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                   │
       │    ┌──────────────┴──────────────┐    │
       └───>│     既存のTransformOp       │<───┘
            │     (E-class、変更なし)      │
            │     ↓ オプティマイザー融合   │
            │     ↓ 暗号化互換            │
            └─────────────────────────────┘
```

**核心的設計決定**：Hadamard、Oracle、Diffusionは新しい演算タイプでは**ありません**。`TransMatrix`オブジェクトを生成する便利関数であり、既存の`TransformOp`（E-class）で使用されます。これにより：

- 既存の**オプティマイザー**が連続量子演算を自動融合します（例：Oracle @ Diffusion → 単一行列）
- 既存の**暗号化モジュール**が相似変換で量子演算を自動暗号化します
- 既存の**アナライザー**が純粋な量子プログラムに対して100%カバレッジを正確に報告します

`measure`のみが真に新しい演算（P-class）です。Born規則 `p_i = |α_i|²` は非線形だからです。ただし、`measure`は**最後に1回だけ**適用されます — すべての中間計算は完全暗号化状態で実行されます。

#### 量子プログラムの暗号化パイプライン

```
クライアント側：                  サーバー側（暗号化）：

  [0.5, 0.5, 0.5, 0.5]        [暗号化された状態]
         │                            │
    encrypt(state, K)          state @ O' @ D' @ O' @ D' ...
         │                            │
         └──────────────>       [暗号化された結果]
                                      │
    decrypt(result, K)  <─────────────┘
         │
  [0, 0, 0, 1.0]              すべてのO', D'はE-class！
         │
    measure() ← クライアント側（P-class、サーバーに送信なし）
         │
  [0, 0, 0, 1.0] → 答え = インデックス3
```

### 予想性能

#### 暗号化カバレッジ

| プログラム | Phase 6以前 | Phase 6以後 | 変化 |
|----------|-----------|-----------|------|
| 純粋線形（transformのみ） | 100% | 100% | — |
| 混合（transform + branch + map） | 30-70% | 30-70% | — |
| **量子検索（oracle + diffuse）** | **N/A** | **100%** | **新規** |
| **量子 + 測定** | **N/A** | **67-100%** | **新規** |

#### Grover検索の計算量

| 検索空間（N） | 古典的（線形走査） | Grover（Axol） | 高速化 |
|-------------|-----------------|---------------|--------|
| 4 | 4回比較 | 1回反復（行列積2回） | 2倍 |
| 16 | 16回比較 | 3回反復（行列積6回） | 2.7倍 |
| 64 | 64回比較 | 6回反復（行列積12回） | 5.3倍 |
| 256 | 256回比較 | 12回反復（行列積24回） | 10.7倍 |
| 1024 | 1024回比較 | 25回反復（行列積50回） | 20.5倍 |
| N | O(N) | O(√N) | O(√N) |

各「反復」は2回の行列積（Oracle + Diffusion）であり、両方ともE-classです。

#### Tool-Use APIのトークン効率

暗号化透過APIは、LLMの観点からすべての暗号化ボイラープレートを排除します：

| タスク | Python + FHE | Axol Tool-Use API | トークン削減 |
|--------|-------------|-------------------|-------------|
| 暗号化分岐 | 約150トークン | 約30トークン | **80%** |
| 暗号化ステートマシン | 約200トークン | 約35トークン | **82%** |
| 暗号化Grover検索 | 約250トークン | 約25トークン | **90%** |
| 暗号化量子ウォーク | 約300トークン | 約30トークン | **90%** |

**なぜこれほど大きな削減なのか**：Python+FHEでは、LLMは鍵生成、暗号化、回路コンパイル、暗号化実行、復号化コードをすべて生成する必要があります。AxolのTool-Use APIでは以下を送信するだけです：

```json
{"action": "quantum_search", "n": 256, "marked": [42], "encrypt": true}
```

APIが鍵生成、プログラム構築、暗号化、実行、復号化、測定を内部的に処理します。

### 差別化ポイント

#### Axol vs. 既存アプローチ

| 属性 | Python（通常） | Python + FHE | Python + TEE | Axol + Quantum |
|------|-------------|-------------|-------------|---------------|
| **暗号化範囲** | なし | 100%（任意の計算） | 100%（ハードウェア） | 100%（量子演算）/ 30-70%（混合） |
| **性能オーバーヘッド** | — | 1,000-10,000倍 | 約0% | 約0%（パイプラインモード） |
| **ハードウェア必要** | なし | なし | SGX/TrustZoneエンクレーブ | なし |
| **LLMに暗号化知識必要** | — | はい（compile, keygen, encrypt, decrypt） | いいえ（インフラレベル） | **いいえ（APIが処理）** |
| **LLMトークンコスト** | 約70トークン | 約200トークン | 約70トークン + インフラ | **約25-30トークン** |
| **ソフトウェアのみ** | はい | はい | いいえ | **はい** |

**Axolの独自ポジション**：FHEのソフトウェアレベル暗号化 + TEEの透明性（LLMは暗号化の存在を知らない）+ FHEもTEEも提供しないTool-Use API効率を組み合わせます。

#### なぜFHEを使わないのか？

完全準同型暗号（FHE）は暗号化データに対して**任意の**計算をサポートします — Axolより厳密に強力なモデルです。しかし：

1. **性能**：FHEは1,000-10,000倍のオーバーヘッドが発生します。Axolの相似変換は線形演算で約0%のオーバーヘッドです。
2. **LLMの複雑さ**：FHEはLLMがコンパイル、鍵生成、暗号化コードを生成する必要があります（約200トークン）。AxolのAPIは約25トークンです。
3. **実用的範囲**：多くのAIエージェントタスク（ステートマシン、検索、ルーティング、スコアリング）は本質的に線形です。量子干渉がこれを検索問題まで拡張します。残りの非線形ケース（活性化関数、クランプ）はクライアント側の後処理に分離できます。

#### なぜTEEを使わないのか？

信頼実行環境（Intel SGX、ARM TrustZone）はゼロ性能オーバーヘッドでハードウェアレベルの暗号化を提供します。しかし：

1. **ハードウェア依存**：TEEは特定のCPU機能を要求します。AxolはNumPyがあればどのマシンでも動作します。
2. **サプライチェーン信頼**：TEEのセキュリティはハードウェアベンダーへの信頼に依存します。Axolのセキュリティは純粋に数学的です。
3. **粒度**：TEEは全か無か（エンクレーブ全体が保護）です。Axolのアナライザーはどの演算が暗号化されるかを正確に示し、情報に基づいたトレードオフ決定を可能にします。

### DSL例

#### Grover検索（平文）

```
@grover_4
s state=[0.5 0.5 0.5 0.5]
: oracle=oracle(state;marked=[3];n=4)
: diffuse=diffuse(state;n=4)
? found state[3]>=0.9
```

結果：ターゲットインデックス3を1回の反復で100%の確率で発見。

#### Grover検索（暗号化パイプライン）

```
@grover_encrypted
s state=[0.5 0.5 0.5 0.5]
: oracle=oracle(state;marked=[3];n=4)
: diffuse=diffuse(state;n=4)
```

終了条件なし — パイプラインモードですべての演算がE-classを保証。
クライアントが復号化後ローカルで`measure()`を適用。

#### Tool-Use API（暗号化知識不要）

```json
{"action": "quantum_search", "n": 256, "marked": [42], "encrypt": true}
→ {"found_index": 42, "probability": 0.996, "iterations": 12, "encrypted": true}

{"action": "encrypted_run",
 "source": "@prog\ns state=[0.5 0.5 0.5 0.5]\n: o=oracle(state;marked=[3];n=4)\n: d=diffuse(state;n=4)",
 "dim": 4}
→ {"final_state": {"state": [0.0, 0.0, 0.0, 1.0]}, "encrypted": true}
```

### テストカバレッジ

`tests/test_quantum.py`に37テスト、カテゴリ別：

| カテゴリ | テスト数 | 検証内容 |
|---------|---------|---------|
| 単体：Hadamard | 4 | 直交性（H@H^T=I）、負の要素、2の累乗検証 |
| 単体：Oracle | 3 | 正しい符号反転、複数マークインデックス、空の場合の単位行列 |
| 単体：Diffusion | 2 | 直交性（D@D^T=I）、負の要素 |
| 単体：Measure | 5 | Born規則、負の振幅不変性、正規化、ゼロベクトル |
| 統合：Grover | 5 | N=4（1回）、N=8（2回）、暗号化パイプライン、終了警告、量子ウォーク |
| アナライザー | 2 | 純粋量子100%カバレッジ、measureのP-class |
| DSL解析 | 8 | measure、hadamard、oracle、diffuse解析 + エラーケース |
| オプティマイザー | 2 | Oracle+Diffuse融合、融合正確性 |
| API | 6 | encrypted_run、quantum_search（平文/暗号化/N=8）、エラー処理 |

```bash
# すべての量子テストを実行
pytest tests/test_quantum.py -v -s

# 特定カテゴリを実行
pytest tests/test_quantum.py::TestGrover -v -s
pytest tests/test_quantum.py::TestQuantumAnalyzer -v -s
pytest tests/test_quantum.py::TestAPI -v -s
```

### ティアロードマップ

| ティア | 状態 | 振幅 | アルゴリズム | 暗号化 |
|-------|------|------|-----------|-------|
| 0 | Phase 1-5 | 非負実数 | 古典的FSM、ルーティング | 30-100%（混合E/P） |
| **1** | **Phase 6（現在）** | **符号付き実数** | **Grover検索、量子ウォーク** | **100%（E-class）** |
| 2 | 将来 | 複素数（a+bi） | Shor、QPE、QFT | 100%（複素ユニタリ） |

---

## ロードマップ

- [x] Phase 1：型システム（7ベクトル型 + StateBundle）
- [x] Phase 1：5つのプリミティブ演算
- [x] Phase 1：プログラム実行エンジン（パイプライン + ループモード）
- [x] Phase 1：状態検証フレームワーク
- [x] Phase 2：DSLパーサー（完全な文法サポート）
- [x] Phase 2：疎行列表記法
- [x] Phase 2：トークンコストベンチマーク（Python、C#、Axol）
- [x] Phase 2：行列暗号化の概念実証（全5演算検証済み、21テスト）
- [x] Phase 3：コンパイラ最適化（transform融合、デッドステート除去、定数畳み込み）
- [x] Phase 3：GPUバックエンド（numpy/cupy/jaxプラガブル）
- [x] Phase 4：AIエージェント向けTool-Use API（parse/run/inspect/verify/list_ops）
- [x] Phase 4：暗号化モジュール（encrypt_program、decrypt_state）
- [x] Phase 5：モジュールシステム（レジストリ、import/use DSL、compose、スキーマ検証）
- [x] フロントエンド：FastAPI + バニラHTML/JSビジュアルデバッガー（トレースビューア、状態チャート、暗号化デモ）
- [x] 性能ベンチマーク（トークンコスト、ランタイムスケーリング、オプティマイザー効果、暗号化オーバーヘッド）
- [x] Phase 6：量子干渉（符号付き振幅、Hadamard/Oracle/Diffusion行列、測定演算）
- [x] Phase 6：量子プログラムの100%暗号化カバレッジ（最終測定を除く全演算がE-class）
- [x] Phase 6：暗号化透過Tool-Use API（encrypted_run、quantum_search — LLMは暗号化知識不要）

---

## ライセンス

MIT License。詳細は[LICENSE](LICENSE)を参照してください。
