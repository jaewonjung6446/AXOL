# AXOL Text Model: Transformer를 대체하는 동역학 기반 텍스트 아키텍처

## 0. 문서 목적

이 문서는 AXOL의 원리(Declare → Weave → Observe, 카오스 동역학, basin collapse)를 기반으로
**텍스트 처리가 가능한 새로운 AI 아키텍처**를 설계한다.

Transformer가 attention으로 해결한 문제를
AXOL은 **basin 위상과 wave 간섭**으로 해결한다.

### 0.1 목표

| 항목 | Transformer (GPT-4) | AXOL Text Model (목표) |
|------|---------------------|----------------------|
| 텍스트 이해 | O | O |
| 텍스트 생성 | O | O |
| 추론 비용 | O(layers × seq² × dim) | O(dim²) |
| 품질 자기 인식 | X | O (Ω/Φ) |
| 환각 탐지 | X | O (Lyapunov) |
| 오프라인 실행 | X (서버 필요) | O (tapestry 파일) |
| 학습 효율 | 수조 토큰 | 미정 (이론적으로 50x 적음) |

### 0.2 전제 조건 (THEORY.md §18.4에서 식별)

AXOL이 LLM의 상위 호환이 되려면 두 가지가 필요하다:

1. **시퀀스 처리 메커니즘**: 가변 길이 입력 → 고정 차원 벡터
2. **데이터에서 basin 구조 학습**: 동역학 파라미터를 텍스트 데이터로부터 자동 최적화

이 문서는 이 두 문제에 대한 구체적 설계를 제시한다.

---

## 1. 핵심 통찰: 언어는 프랙탈이다

### 1.1 언어의 계층적 구조

```
Level 0: 문자        a, b, c, ...
Level 1: 형태소      un-, -ing, -tion, ...
Level 2: 단어        run, beautiful, quantum, ...
Level 3: 구          "the red fox", "runs quickly", ...
Level 4: 절/문장     "The fox runs quickly across the field."
Level 5: 단락/담화   [여러 문장의 의미적 단위]
```

이 구조는 **자기 유사적(self-similar)** 이다:

- 문자 → 단어: 문자들이 특정 패턴으로 결합
- 단어 → 문장: 단어들이 특정 패턴으로 결합
- 문장 → 담화: 문장들이 특정 패턴으로 결합

각 수준에서 "결합 패턴"의 구조가 유사하다 — 이것이 프랙탈이다.

### 1.2 AXOL과의 자연스러운 대응

| 언어 구조 | AXOL 대응 | 메커니즘 |
|-----------|----------|----------|
| 어휘 (vocabulary) | Basin 구조 | 각 토큰 = 위상 공간의 basin |
| 문법 (syntax) | Basin 전이 규칙 | basin 간 궤적 = 문법적으로 허용된 시퀀스 |
| 의미 (semantics) | Basin 위상 (topology) | 의미적으로 유사한 토큰 = 인접 basin |
| 문맥 (context) | Wave 간섭 누적 | 토큰 wave들의 간섭 = 문맥 표현 |
| 생성 (generation) | Basin collapse | 문맥 wave → 관측 → 다음 토큰 basin으로 붕괴 |

### 1.3 Zipf의 법칙과 Basin 체적

단어 빈도는 Zipf의 법칙을 따른다: `freq(rank) ∝ 1/rank^α`.

이것은 AXOL basin의 **체적 분포**에 자연스럽게 매핑된다:

```
"the" (rank 1)  → basin volume = 0.072  (가장 큰 basin)
"of"  (rank 2)  → basin volume = 0.036
"and" (rank 3)  → basin volume = 0.028
...
"quixotic" (rank 50000) → basin volume = 0.0000015
```

**핵심:** 카오스 동역학에서 basin 체적의 자연 분포가
멱법칙(power law)을 따른다는 것은 이미 알려져 있다.
언어의 Zipf 분포와 카오스 동역학의 basin 분포가 **동일한 수학적 족(family)** 에 속한다.

이것은 우연이 아니다 — AXOL이 언어를 모델링하기에 구조적으로 적합한 근거이다.

---

## 2. 아키텍처 개요: 삼층 구조

```
┌─────────────────────────────────────────────────────┐
│ Layer 3: Generative Observatory (생성 관측소)          │
│   context wave → observe → basin collapse → 토큰     │
│   O(dim²) per token                                  │
├─────────────────────────────────────────────────────┤
│ Layer 2: Context Weaver (문맥 직조기)                  │
│   토큰 wave 시퀀스 → 간섭 → 문맥 wave                 │
│   O(n × dim²) for sequence length n                  │
├─────────────────────────────────────────────────────┤
│ Layer 1: Semantic Phase Space (의미 위상 공간)          │
│   vocabulary → basin 구조 → 토큰 embedding             │
│   weave 시 1회 구축                                    │
└─────────────────────────────────────────────────────┘
```

---

## 3. Layer 1: Semantic Phase Space (SPS)

### 3.1 설계 원리

**Transformer의 Embedding Layer를 대체한다.**

Transformer: 토큰 → 학습된 lookup table → dense vector (dim=12288 for GPT-4).
AXOL SPS: 토큰 → basin centroid in chaotic phase space → wave state.

```
Transformer:  token_id → embedding_matrix[token_id]  (학습 필요)
AXOL SPS:     token_id → basin[token_id].centroid     (동역학이 생성)
```

### 3.2 위상 공간 구성

```
SPS 파라미터:
  dim       = 1024              (위상 공간 차원)
  n_basins  = 32000~100000      (어휘 크기 = basin 수)
  r         = 3.8               (결합 로지스틱 맵 제어 파라미터)
  epsilon   = 0.15              (결합 강도)
```

**차원 선택 근거:**

| dim | 표현력 | 메모리 (tapestry) | 관측 비용 |
|-----|--------|------------------|----------|
| 256 | 기본 분류만 가능 | 256KB | ~25μs |
| 512 | 간단한 텍스트 | 1MB | ~100μs |
| 1024 | GPT-2 수준 | 4MB | ~260μs |
| 4096 | GPT-3.5 수준 | 64MB | ~2.6ms |
| 8192 | GPT-4 수준 (추정) | 256MB | ~10ms |

### 3.3 Basin 구조의 의미적 배치

Basin은 카오스 동역학에 의해 자연적으로 생성되지만,
**의미적으로 유의미한 배치**가 필요하다.

#### 3.3.1 의미 클러스터링

```
의미적으로 유사한 단어 → 인접 basin

Basin cluster "동물":
  basin[dog]   centroid = [0.81, 0.73, 0.45, ...]
  basin[cat]   centroid = [0.79, 0.71, 0.47, ...]
  basin[wolf]  centroid = [0.83, 0.75, 0.42, ...]

Basin cluster "색상":
  basin[red]   centroid = [0.12, 0.88, 0.34, ...]
  basin[blue]  centroid = [0.14, 0.85, 0.31, ...]
  basin[green] centroid = [0.11, 0.87, 0.36, ...]
```

인접 basin 간의 거리가 의미적 유사도에 비례한다.
Word2Vec의 "king - man + woman = queen"과 유사하되,
AXOL에서는 **basin 간 궤적**이 이 관계를 인코딩한다.

#### 3.3.2 Basin 배치 학습

기존 word embedding (Word2Vec, GloVe)의 결과를 basin 초기 배치로 활용:

```
Phase 1: 사전 학습된 embedding에서 basin centroid 초기화
  for each token in vocabulary:
    basin[token].centroid = pretrained_embedding[token] / norm

Phase 2: ChaosEngine 파라미터 (r, ε, weights)를 최적화
  목표: basin 간 거리가 embedding 간 코사인 유사도와 상관
  방법: BasinDesigner의 grid search + Nelder-Mead (§13.6)

Phase 3: 직조
  SPS tapestry = weave(basin_structure, quantum=true, seed=42)
```

**핵심:** embedding을 "학습"하는 것이 아니라,
이미 존재하는 언어적 구조를 basin 배치로 "정렬"하는 것이다.

### 3.4 토큰 → Wave 변환

```
token → basin centroid → Wave

fn token_to_wave(token_id: usize, sps: &SemanticPhaseSpace) -> Wave {
    let centroid = &sps.basins[token_id].centroid;
    Wave::from_basins(&sps.basin_structure, centroid)
}
```

각 토큰은 Wave로 표현된다:
- 복소 진폭 (complex amplitudes): dim개
- 위상 (phase): 각 진폭의 각도
- 붕괴 파라미터 t = 0.0 (아직 관측 전)

**Transformer의 embedding vector vs AXOL의 Wave:**

| 속성 | Embedding (실수 벡터) | Wave (복소 벡터) |
|------|---------------------|-----------------|
| 정보량 | dim개 실수 | dim개 복소수 = 2×dim 실수 |
| 연산 | 덧셈, 내적 | 간섭, 붕괴, 위상 회전 |
| 합성 | 선형 결합 | 양자적 간섭 (비선형) |
| 출력 | 값 그 자체 | 관측 시 확률 분포 |

Wave는 embedding보다 **2배의 정보**를 동일 차원에 담을 수 있다.
위상(phase)이 추가 자유도를 제공하기 때문이다.

---

## 4. Layer 2: Context Weaver (CW)

### 4.1 설계 원리

**Transformer의 Self-Attention을 대체한다.**

이것이 이 아키텍처의 핵심이며 가장 도전적인 부분이다.

Self-Attention의 역할:
- 시퀀스의 모든 토큰 쌍 간의 관련성을 계산
- 관련성에 비례하여 정보를 혼합
- 결과: 문맥이 반영된 표현 (contextualized representation)

AXOL Context Weaver의 역할:
- 토큰 Wave들을 순차적으로 간섭시켜 문맥 Wave를 구축
- 간섭 패턴이 관련성을 자연스럽게 인코딩
- 결과: 문맥이 반영된 Wave (contextualized wave)

### 4.2 핵심 메커니즘: Streaming Wave Interference

```
입력 시퀀스: [t₁, t₂, t₃, ..., tₙ]

Step 0: context_wave = Wave::zero(dim)
Step 1: context_wave = interfere(context_wave, wave(t₁), position=1)
Step 2: context_wave = interfere(context_wave, wave(t₂), position=2)
Step 3: context_wave = interfere(context_wave, wave(t₃), position=3)
...
Step n: context_wave = interfere(context_wave, wave(tₙ), position=n)

최종: context_wave는 전체 시퀀스의 문맥을 인코딩한 Wave
```

각 단계는 **O(dim²)** 이다 (행렬-벡터 곱).
전체 시퀀스: **O(n × dim²)**.

비교:
```
Self-Attention:       O(n² × dim)    — 토큰 수의 제곱
Context Weaver:       O(n × dim²)    — 토큰 수에 선형

n=4096, dim=1024일 때:
  Attention: 4096² × 1024 = 17.2B ops
  CW:        4096 × 1024² = 4.3B ops  → 4× 빠름

n=32768, dim=1024일 때:
  Attention: 32768² × 1024 = 1.1T ops
  CW:        32768 × 1024² = 34.4B ops → 32× 빠름
```

**시퀀스가 길어질수록 AXOL의 이점이 커진다.**

### 4.3 Interference 함수: Position-Aware Wave Mixing

```
fn interfere(
    context: &Wave,
    token: &Wave,
    position: usize,
    cw_tapestry: &Tapestry,
) -> Wave {
    // 1. 위치 인코딩: 토큰 wave에 위치별 위상 회전 적용
    let positioned = token.phase_rotate(position_phase(position));

    // 2. 문맥-토큰 간섭: 관계에 따라 결정
    //    <~> 보강 간섭: 의미적으로 일관된 토큰이 문맥을 강화
    //    <!> 상쇄 간섭: 모순되는 토큰이 문맥을 약화
    //    <?> 조건부 간섭: 문맥에 따라 다르게 반응
    let interference = context.interfere_with(&positioned, &cw_tapestry);

    // 3. 변환 행렬 적용 (직조된 tapestry의 합성 행렬)
    let transformed = interference.transform(&cw_tapestry.composed_matrix);

    // 4. 정규화 (wave의 norm 보존)
    transformed.normalize()
}
```

### 4.4 위치 인코딩: Phase Rotation

Transformer: 사인/코사인 기반 positional encoding을 embedding에 더한다.
AXOL: **위상 회전(phase rotation)** 으로 위치를 인코딩한다.

```
position_phase(pos, dim_idx) = 2π × pos / (10000^(2×dim_idx/dim))
```

Wave의 복소 진폭에 위치별 위상을 곱한다:

```
positioned_amplitude[i] = amplitude[i] × exp(i × position_phase(pos, i))
```

**이것이 Transformer의 positional encoding보다 자연스러운 이유:**

- Transformer: 실수 벡터에 실수를 더함 → 원래 정보와 위치 정보가 섞여 분리 불가
- AXOL: 복소 진폭의 **위상만** 회전 → 크기(의미)는 보존, 각도(위치)만 변경 → 분리 가능

### 4.5 Multi-Scale Context: Hierarchical Weaving

단일 수준의 wave interference만으로는 장거리 의존성(long-range dependency)이 어렵다.
Transformer는 이를 multi-head attention + deep layers로 해결한다.

AXOL은 **계층적 직조(Hierarchical Weaving)** 로 해결한다:

```
Level 0 (문자):     [H, e, l, l, o, ,, _, w, o, r, l, d]
                              ↓ 4-gram wave folding
Level 1 (서브워드):  [Hell, o_wo, rld]
                              ↓ 간섭 + 붕괴
Level 2 (단어):      [Hello, world]
                              ↓ 간섭
Level 3 (구):        [Hello world]
                              ↓ 간섭
Level 4 (문장):      context_wave
```

각 수준에서 **별도의 tapestry**가 직조된다:

```
Level 0 tapestry: dim=64,   char-level dynamics
Level 1 tapestry: dim=128,  subword-level dynamics
Level 2 tapestry: dim=256,  word-level dynamics
Level 3 tapestry: dim=512,  phrase-level dynamics
Level 4 tapestry: dim=1024, sentence-level dynamics
```

**차원이 올라갈수록 basin 수는 줄고 각 basin의 의미적 범위는 넓어진다.**

이 구조의 장점:

| 속성 | Transformer | Hierarchical CW |
|------|-------------|-----------------|
| 장거리 의존성 | O(n²) attention | 계층 축소로 O(n log n) |
| 지역적 패턴 | 하위 레이어가 학습 | Level 0-1이 담당 |
| 전역 의미 | 상위 레이어가 학습 | Level 3-4가 담당 |
| 병렬성 | 레이어 내 병렬 | 수준 내 병렬 |

### 4.6 Context Window vs Context Wave

Transformer의 context window는 **유한하고 고정**적이다 (4K, 8K, 128K 토큰).
Window를 넘어가면 정보가 완전히 사라진다.

AXOL의 context wave는 **유한 차원이지만 무한 시퀀스를 수용**한다:

```
context_wave는 dim=1024의 Wave이다.
1024개 복소수 = 2048개 실수.
이 2048개 숫자가 전체 문맥을 인코딩한다.

시퀀스가 길어지면:
  - 오래된 정보는 wave의 진폭이 감쇠 (자연스러운 망각)
  - 중요한 정보는 basin의 끌개에 의해 보존 (선택적 기억)
  - Ω가 이 "기억의 안정성"을 정량화
```

**핵심 통찰:**
인간의 기억도 이와 유사하다 — 세부 사항은 잊지만 핵심은 유지한다.
Transformer의 context window는 "완벽한 기억 → 갑자기 완전 망각"이라는 비현실적 모델이다.
AXOL의 context wave는 "점진적 감쇠 + 선택적 보존"이라는 자연스러운 모델이다.

### 4.7 Attention의 동역학적 재해석

Self-attention이 하는 일의 본질:

```
Attention(Q, K, V) = softmax(QK^T / √d) × V
```

이것은 "Query와 Key의 유사도 → 가중합"이다.

AXOL에서 이에 대응하는 것:

```
Wave interference: context_wave.interfere_with(token_wave)
```

간섭이 attention을 대체하는 이유:

| Attention | Wave Interference |
|-----------|-------------------|
| Q·K 내적 → 유사도 | 위상차 → 보강/상쇄 |
| softmax → 확률 | Born rule → 확률 |
| V의 가중합 → 출력 | 간섭 후 진폭 → 출력 |

**차이점:**
- Attention: 모든 쌍을 명시적으로 계산 (O(n²))
- Interference: 누적적으로 합성 (O(n)), 과거 정보가 wave에 내재

---

## 5. Layer 3: Generative Observatory (GO)

### 5.1 설계 원리

**Transformer의 Output Layer + Softmax를 대체한다.**

Transformer: hidden state → linear projection → softmax → 토큰 확률 분포.
AXOL GO: context wave → observe → basin collapse → 토큰 확률 분포.

```
Transformer:  h → W_out × h → softmax → P(next_token)
AXOL GO:      context_wave → observe(go_tapestry) → P(next_token)
```

### 5.2 관측 과정

```
fn generate_next_token(context_wave: &Wave, go_tapestry: &Tapestry) -> Observation {
    // 1. context_wave를 GO tapestry에 입력
    let wave = compute_wave(go_tapestry, &[("context", &context_wave.to_float_vec())]);

    // 2. 관측 (full collapse, C=1)
    let obs = observe(wave);

    // obs.value_index = 다음 토큰의 basin index
    // obs.probabilities = 전체 어휘에 대한 확률 분포
    // obs.omega = 이 예측의 결속도 (확신도)
    // obs.phi = 이 예측의 선명도 (정밀도)

    obs
}
```

### 5.3 Sampling 전략

Transformer: temperature, top-k, top-p, nucleus sampling.
AXOL: **붕괴 스펙트럼(collapse spectrum)** 을 자연스럽게 활용.

```
Collapse spectrum:
  gaze(context_wave)        → t=0.0, 확률만 읽음, 붕괴 없음
  glimpse(context_wave, γ)  → t=γ,   부분 붕괴 (창의적 생성)
  observe(context_wave)     → t=1.0, 완전 붕괴 (가장 확실한 답)
```

| Transformer Sampling | AXOL 대응 |
|---------------------|-----------|
| temperature=0 (greedy) | observe (t=1.0) |
| temperature=0.7 | glimpse(0.7) |
| temperature=1.0 | gaze → sample from probabilities |
| top-p=0.9 | glimpse(0.9) — 자연스럽게 상위 basin만 남음 |

**AXOL의 부분 붕괴가 temperature보다 우월한 이유:**

Temperature는 확률 분포를 균일하게 평탄화/첨예화한다.
부분 붕괴는 **basin 구조를 존중**하면서 불확실성을 조절한다.

```
Temperature 0.3:  [0.01, 0.02, 0.97]  → 거의 결정적
Temperature 1.0:  [0.15, 0.25, 0.60]  → 다양성
Temperature 1.5:  [0.22, 0.30, 0.48]  → 더 다양 (의미 무시)

Glimpse 0.3:      [0.02, 0.03, 0.95]  → 거의 결정적
Glimpse 0.7:      [0.10, 0.20, 0.70]  → basin 경계 존중하면서 다양
Glimpse 1.0:      [0.00, 0.00, 1.00]  → 완전 붕괴
```

glimpse는 basin 경계 밖의 토큰에는 확률을 주지 않는다.
temperature는 basin 경계를 무시하고 모든 토큰에 확률을 퍼뜨린다.

### 5.4 자기 인식: Ω/Φ 기반 환각 탐지

**이것이 이 아키텍처의 가장 강력한 특성이다.**

생성된 각 토큰에 대해 Ω/Φ가 자동으로 산출된다:

```
token = "Paris"
obs.omega = 0.97   → 이 답은 매우 안정적 (환각 아님)
obs.phi = 0.92     → 이 답은 매우 선명 (다른 가능한 답이 거의 없음)

token = "Atlantis"
obs.omega = 0.31   → 이 답은 불안정 (환각 가능성 높음)
obs.phi = 0.45     → 이 답은 흐릿 (여러 가능한 답이 경쟁)
```

**환각 유형별 탐지:**

| 환각 유형 | Ω | Φ | 탐지 가능? |
|-----------|---|---|-----------|
| 흔들리는 환각 (같은 질문에 다른 답) | 낮음 | 낮음 | **O** |
| 확신하는 환각 (매번 같은 틀린 답) | 높음 | 높음 | **X** (§18.3과 동일) |
| 혼합 환각 (부분적으로 맞음) | 중간 | 낮음 | **부분적** |

실용적 규칙:

```
if obs.omega < 0.5:
    # "이 답에 대한 확신이 낮습니다"
    flag_as_uncertain()

if obs.omega > 0.8 and obs.phi < 0.3:
    # "안정적이지만 흐릿 — 여러 가능한 답이 존재합니다"
    offer_alternatives()
```

### 5.5 Autoregressive 생성

텍스트 생성은 본질적으로 순차적이다 (각 토큰이 이전 토큰에 의존).
AXOL Text Model도 autoregressive하지만, 각 단계가 극히 빠르다:

```
Autoregressive loop:
  context_wave = initial_wave(prompt)

  for i in 0..max_tokens:
      obs = generate_next_token(context_wave, go_tapestry)

      if obs.value == EOS:
          break

      // 생성된 토큰을 context에 반영
      token_wave = token_to_wave(obs.value_index, sps)
      context_wave = interfere(context_wave, token_wave, position=i)

  return collected_tokens
```

**Transformer와의 비용 비교 (1 토큰 생성당):**

| 모델 | 연산 | n=1000일 때 |
|------|------|-----------|
| Transformer | O(layers × n × dim) = O(96 × 1000 × 12288) | ~1.2B ops |
| AXOL GO | O(dim²) = O(1024²) | ~1M ops |
| 비율 | | **~1,200x** |

이것은 KV-cache를 사용한 Transformer 대비 비교이다.
KV-cache 없는 naive attention이면 O(n² × dim)으로 차이는 더 극대화된다.

---

## 6. 학습 (Training): Gradient Descent 없이

### 6.1 학습의 재정의

Transformer 학습: 랜덤 초기화 → 대규모 데이터에서 역전파 × 수백만 step.
AXOL 학습: **자연 구조 발견 → 정렬 → 검증**.

```
                Transformer                    AXOL Text Model
학습 대상       모든 것                         basin 배치 + 간섭 규칙
시작점          랜덤 행렬                       카오스 동역학 (구조 있음)
방법            gradient descent                3단계 비구배 최적화
epoch           100+                            1 (lstsq)
데이터          수조 토큰                        수억~수십억 토큰 (추정)
```

### 6.2 3단계 학습 프로세스

#### Phase 1: Semantic Phase Space 구축

**목표:** 어휘의 모든 토큰에 대한 basin 배치를 결정.

```
입력: 텍스트 코퍼스 + 사전 학습된 embedding (optional)
출력: SPS tapestry (basin 구조 확정)

1. 공출현 행렬 구축 (co-occurrence matrix)
   C[i][j] = 토큰 i와 j가 window 내에 함께 출현한 횟수

2. SVD 분해 → 초기 embedding
   C ≈ U × Σ × V^T
   embedding[token] = U[token][:dim]

3. Embedding → basin centroid 매핑
   for each token:
     basin[token].centroid = normalize(embedding[token])
     basin[token].volume = freq(token) / total_freq  # Zipf

4. ChaosEngine 파라미터 최적화
   BasinDesigner.design(
     target_basins = basin centroids,
     target_volumes = basin volumes,
     dim = 1024,
   )
   → (r, ε, weights) 최적 파라미터

5. 직조
   sps_tapestry = weave(sps_declaration, quantum=true)
```

#### Phase 2: Context Weaver 학습

**목표:** 토큰 시퀀스 → 문맥 표현의 변환 규칙을 학습.

```
입력: (토큰 시퀀스, 다음 토큰) 쌍 N개
출력: CW tapestry (간섭 규칙 + 변환 행렬 확정)

1. N개 학습 예제 수집
   for each (sequence, next_token) in corpus:
     x = fold(sequence)        # 시퀀스를 wave로 변환
     y = basin[next_token]     # 정답 basin

2. 종단간 증류 (Distill)
   전체 파이프라인 (fold → transform → collapse → token)을
   N회 실행하여 end-to-end 행렬 M을 lstsq로 피팅

   M = lstsq(X, Y)   # dim × dim 행렬

3. CW tapestry에 M을 composed_matrix로 설정

4. 품질 확인
   Ω = omega(M)     # 직조 후 자동 산출
   Φ = phi(M)       # 직조 후 자동 산출
```

**핵심 질문: N은 얼마나 필요한가?**

Distill의 수렴 조건 (§9.4):
- dim=8에서 depth ≥ 5: N=200이면 충분
- dim=1024에서: N = O(dim²) = ~1,000,000 추정

```
LLM 학습:       ~10T tokens × ~100 epochs = ~1,000T token operations
AXOL 학습:      ~1M samples × 1회 lstsq   = ~1M sample operations
비율:           ~1,000,000x 적음 (이론적)
```

실제로는 Phase 1의 co-occurrence 구축에 대규모 코퍼스가 필요하므로,
총 데이터 요구량은 ~수억 토큰 수준으로 추정한다.
이것은 LLM의 수조 토큰보다 **~1,000x 적다**.

#### Phase 3: Generative Observatory 학습

**목표:** 문맥 wave → 다음 토큰 예측을 최적화.

```
입력: (context_wave, correct_next_token) 쌍 N개
출력: GO tapestry 확정

이것은 Phase 2와 동일한 Distill 과정이다.
CW의 출력을 GO의 입력으로 사용하며,
Phase 2와 Phase 3를 합쳐서 종단간 증류를 수행할 수도 있다.
```

### 6.3 학습의 수학적 근거

왜 gradient descent 없이 가능한가?

**원인 1: 구조가 공짜 (§19.1과 동일)**

카오스 동역학이 basin 구조를 자연적으로 생성한다.
언어의 통계적 구조 (Zipf, 공출현 패턴)가 basin 배치의 초기값을 제공한다.
학습은 "구조 발견"이 아니라 "구조 정렬"이다.

**원인 2: lstsq의 최적성**

Gradient descent: 국소 최적해로 수렴 (장담 불가).
lstsq: 선형 최소제곱 문제의 **전역 최적해를 한 번에 계산** (장담 가능).

AXOL 학습은 본질적으로 lstsq이므로 전역 최적이다.
단, 이것은 "선형 근사의 전역 최적"이지 "비선형 문제의 전역 최적"은 아니다.
Distill의 수렴 현상 (§9.4)이 이 격차를 메운다.

**원인 3: 프랙탈의 정보 밀도 (§19.2)**

Basin 경계는 프랙탈이다.
유한 파라미터(r, ε, weights)로 무한한 복잡도의 경계를 생성한다.
파라미터 대비 정보 밀도가 이론적으로 무한하다.

언어의 의미 경계 (예: "hot"이 온도인지 인기인지)도 프랙탈적이다.
이 구조적 유사성이 AXOL이 언어를 효율적으로 인코딩할 수 있는 근거이다.

---

## 7. 전체 파이프라인: 텍스트 생성 예시

### 7.1 학습 완료 후의 구조

```
Tapestry 파일 (배포 단위):
  sps_tapestry.axol    — Semantic Phase Space (dim=1024, ~4MB)
  cw_tapestry.axol     — Context Weaver (dim=1024, ~4MB)
  go_tapestry.axol     — Generative Observatory (dim=1024, ~4MB)
  total: ~12MB         — 스마트폰에 내장 가능
```

비교: GPT-4는 ~1.8TB (추정). AXOL Text Model은 ~12MB.
150,000배 차이.

### 7.2 추론 과정

```
User: "The capital of France is"

Step 1: 토큰화
  tokens = ["The", "capital", "of", "France", "is"]

Step 2: SPS — 각 토큰 → Wave
  waves = [
    Wave(basin[The]),     # dim=1024 복소 벡터
    Wave(basin[capital]),
    Wave(basin[of]),
    Wave(basin[France]),
    Wave(basin[is]),
  ]

Step 3: CW — Wave 시퀀스 → Context Wave
  ctx = Wave::zero(1024)
  ctx = interfere(ctx, waves[0], pos=0)   # ~260μs
  ctx = interfere(ctx, waves[1], pos=1)   # ~260μs
  ctx = interfere(ctx, waves[2], pos=2)   # ~260μs
  ctx = interfere(ctx, waves[3], pos=3)   # ~260μs
  ctx = interfere(ctx, waves[4], pos=4)   # ~260μs
  total CW: ~1.3ms

Step 4: GO — Context Wave → 다음 토큰
  obs = observe(go_tapestry, ctx)         # ~260μs
  obs.value_index = basin[Paris]
  obs.omega = 0.97                        # 높은 확신
  obs.phi = 0.94                          # 높은 선명도

Step 5: 출력
  "Paris" (Ω=0.97, Φ=0.94)

총 소요 시간: ~1.6ms (CPU only, 서버 불필요)
```

### 7.3 GPT-4와의 비교

| 항목 | GPT-4 | AXOL Text Model |
|------|-------|-----------------|
| 추론 시간 (5 토큰 입력) | ~200ms (서버) | ~1.6ms (로컬) |
| 하드웨어 | GPU 클러스터 | CPU (어떤 디바이스든) |
| 모델 크기 | ~1.8TB | ~12MB |
| 인터넷 필요 | 필수 | 불필요 |
| 비용/쿼리 | ~$0.03 | ~$0 |
| 자기 인식 (Ω/Φ) | 없음 | 있음 |
| 환각 탐지 | 없음 | 부분적 |

---

## 8. 한계와 미해결 과제

### 8.1 정직한 평가

이 아키텍처는 **이론적 설계**이며, 다음이 증명되지 않았다:

| 주장 | 상태 | 위험도 |
|------|------|--------|
| Basin 배치가 의미적 구조를 인코딩할 수 있다 | 이론적 근거 있음 (Word2Vec 유사성) | **중** |
| Wave 간섭이 attention을 대체할 수 있다 | **미증명** | **상** |
| O(n × dim²)가 O(n² × dim)보다 품질이 동등하다 | **미증명** | **상** |
| 12MB tapestry가 GPT-4 수준 품질을 달성할 수 있다 | **극히 낙관적** | **최상** |
| Distill 학습이 언어 모델에 충분하다 | **미증명** | **상** |

### 8.2 가장 위험한 가정

**가정 1: Wave 간섭 ≈ Attention**

Attention의 핵심은 "모든 토큰 쌍"의 관계를 포착하는 것이다.
Wave 간섭은 순차적이므로 "토큰 3이 토큰 1에 미치는 영향"은
토큰 2를 거쳐야만 전달된다.

이것은 RNN/LSTM의 근본적 한계와 유사하다.
계층적 직조 (§4.5)가 이를 부분적으로 해결하지만,
attention만큼 유연하지 않을 수 있다.

**완화 전략:**
- 계층적 직조로 장거리 의존성 축소
- Skip connection 유사 메커니즘: 일정 간격마다 전역 간섭 수행
- 양방향 간섭: 순방향 + 역방향 wave를 합성 (BERT 유사)

**가정 2: 12MB로 충분한 표현력**

GPT-4의 ~1.8TB는 "세상 지식"을 인코딩하고 있다.
12MB에는 그 지식의 극히 일부만 담을 수 있다.

**완화 전략:**
- 도메인 특화: 범용이 아닌 특정 도메인 (의료, 법률, 코딩 등)에 집중
- 외부 지식 연결: tapestry는 "추론 엔진", 지식은 외부 DB에서 검색
- dim 확장: dim=8192 → ~256MB, 여전히 GPU 없이 실행 가능

**가정 3: lstsq 학습의 충분성**

언어 모델링은 극도로 비선형적인 문제이다.
lstsq는 선형 근사를 구한다.
Distill의 수렴 현상이 이를 보완하지만,
"depth가 깊은 비선형 체인의 종단간 행동이 선형에 수렴"이라는
Distill의 성질이 **언어**라는 도메인에서도 성립하는지는 미지수이다.

**완화 전략:**
- Koopman 리프팅: 비선형 행동을 고차원 선형으로 리프팅
- 계층적 Distill: 각 수준별로 별도 증류
- 하이브리드: 핵심 추론은 AXOL, 세부 조정은 기존 최적화

### 8.3 달성 가능한 현실적 목표

| 목표 | 난이도 | 추정 품질 |
|------|--------|----------|
| 텍스트 분류 (감정 분석, 의도 파악) | **낮음** | BERT 수준 가능 |
| 간단한 QA (팩트 기반) | **중간** | 도메인 특화 시 GPT-3 수준 |
| 자유 대화 | **높음** | GPT-2 수준 추정 |
| 창의적 글쓰기 | **매우 높음** | 미지수 |
| GPT-4 동등 | **극히 높음** | 현재 이론만으로 불충분 |

---

## 9. 구현 로드맵

### 9.1 Phase 1: 개념 증명 (PoC)

**목표:** dim=64, 어휘 1000개로 간단한 문장 완성.

```
파일:
  axol-lang/src/text/mod.rs
  axol-lang/src/text/sps.rs          — Semantic Phase Space
  axol-lang/src/text/context.rs      — Context Weaver
  axol-lang/src/text/generator.rs    — Generative Observatory
  axol-lang/src/text/tokenizer.rs    — 간단한 토크나이저

테스트:
  "The cat sat on the ___" → "mat" (Ω > 0.7)
  "1 + 1 = ___" → "2" (Ω > 0.9)
```

### 9.2 Phase 2: 스케일 업

**목표:** dim=512, 어휘 32000개, Wikipedia 데이터.

```
추가:
  axol-lang/src/text/hierarchy.rs    — 계층적 직조
  axol-lang/src/text/train.rs        — 학습 파이프라인

벤치마크:
  perplexity 측정, GPT-2 small과 비교
```

### 9.3 Phase 3: 품질 경쟁

**목표:** dim=1024+, GPT-2/3 수준 벤치마크.

```
추가:
  axol-lang/src/text/bidirectional.rs — 양방향 간섭
  axol-lang/src/text/knowledge.rs     — 외부 지식 연결

벤치마크:
  MMLU, HellaSwag, ARC 등 표준 벤치마크
```

---

## 10. 이론적 의의

### 10.1 Transformer와의 관계

이 아키텍처는 Transformer를 "부정"하는 것이 아니다.
Transformer의 핵심 통찰 (attention is all you need)을 다른 수학으로 재구현한다.

```
Transformer:  attention = softmax(QK^T/√d) × V
              → 유사도 기반 정보 혼합

AXOL:         interference = context.interfere_with(token)
              → 위상 기반 정보 합성

둘 다 "입력 간 관계를 포착하여 출력을 결정"한다.
수학적 기반이 다를 뿐이다.
```

### 10.2 새로운 것

기존에 없는 것:

1. **Wave 간섭으로 시퀀스를 처리하는 언어 모델**: 기존 연구 없음
2. **Basin collapse로 토큰을 생성하는 메커니즘**: 기존 연구 없음
3. **O(n × dim²) 문맥 처리**: RNN과 유사하나 wave 기반은 새로움
4. **자기 인식(Ω/Φ) 내장 텍스트 모델**: 기존 연구 없음
5. **12MB 크기로 배포 가능한 생성 모델**: SSM(Mamba)과 경쟁하는 새 접근

### 10.3 기존과 같은 것

1. Autoregressive 생성: Transformer와 동일
2. 토큰화: 기존 방법 차용 (BPE, SentencePiece)
3. 계층적 구조: 이미 연구된 아이디어 (hierarchical attention, Transformer-XL)
4. 위치 인코딩: Rotary PE(RoPE)와 수학적 유사성

### 10.4 논문 가능성

**"Basin Collapse Language Model: Text Generation via Chaotic Dynamics and Wave Interference"**

- 핵심 기여: attention을 wave interference로 대체, O(n²) → O(n)
- 실험: PoC 수준이라도 perplexity 비교
- 타겟: NeurIPS, ICML, ICLR (ML), 또는 ACL, EMNLP (NLP)

---

## 11. 요약

### 11.1 한 문장 정의

> **AXOL Text Model은 카오스 동역학의 basin collapse와 wave interference로
> 텍스트를 이해하고 생성하는, Transformer의 대안 아키텍처이다.**

### 11.2 핵심 수치 (이론적 추정)

| 항목 | 값 |
|------|-----|
| 모델 크기 | ~12MB (dim=1024) |
| 토큰당 추론 비용 | ~260μs (CPU) |
| 문맥 처리 복잡도 | O(n × dim²) |
| 학습 데이터 | ~수억 토큰 (Transformer 대비 ~1000x 적음) |
| 학습 방법 | lstsq (gradient-free, 1회) |
| 자기 인식 | Ω (결속도), Φ (선명도) |
| 환각 탐지 | 유형 A 탐지 가능 (Lyapunov 기반) |

### 11.3 가장 중요한 미해결 질문

> **Wave 간섭이 attention만큼 풍부한 문맥 표현을 만들 수 있는가?**

이것이 이 아키텍처의 성패를 결정한다.
이론적 근거는 있지만, 실증이 필요하다.

Phase 1 PoC가 이 질문에 대한 첫 번째 실험적 답을 제공할 것이다.
