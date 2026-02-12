<p align="center">
  <h1 align="center">AXOL</h1>
  <p align="center">
    <strong>카오스 이론 기반 공간-확률적 연산 언어</strong>
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

> **경고: 이 프로젝트는 초기 실험 단계입니다.**
> API, DSL 문법, 내부 아키텍처는 예고 없이 변경될 수 있습니다. 프로덕션 사용은 권장하지 않습니다. 기여와 피드백을 환영합니다.

---

## AXOL이란?

**AXOL**은 연산의 기반으로서 **시간축**(순차 실행)을 부정하고, 이를 두 가지 대안적 축으로 대체하는 프로그래밍 언어입니다:

- **공간축** -- 노드 간의 관계가 연산을 결정
- **확률축** -- 결과의 가능성이 결과를 결정

"이것을 하고, 그다음 저것을 해라" 대신, AXOL은 "무엇이 무엇과 관계되며, 얼마나 강하게?" 라고 묻습니다. 그 결과는 **카오스 이론**에 기반한 근본적으로 다른 실행 모델이며, 연산이란 **이상 끌개(strange attractor)**를 구축하고 관측하는 행위입니다.

```
전통적 방식:  명령어 1 → 명령어 2 → 명령어 3  (시간 순차)

AXOL:
  [공간]     NodeA ──관계── NodeB     "어디에"가 연산을 결정
  [확률] state = { alpha|가능성1> + beta|가능성2> }  "얼마나 가능한지"가 결과를 결정
```

### 핵심 특성

- **3단계 실행**: 선언(Declare) → 직조(Weave) → 관측(Observe) (컴파일 → 실행이 아님)
- **카오스 이론 기반**: Tapestry = Strange Attractor, Lyapunov 지수와 프랙탈 차원으로 품질 측정
- **이중 품질 지표**: Omega (응집도) + Phi (명확도) -- 엄밀하고, 측정 가능하며, 합성 가능
- **63% 평균 토큰 절약** (동등한 Python 대비, Quantum DSL)
- **비실현성 감지** -- 목표가 수학적으로 달성 불가능할 때 연산 전에 경고
- **Lyapunov 추정 정확도**: 평균 오차 0.0002
- **3가지 비선형 합성 방식**: 순수 유니터리 (방향만), 하이브리드 (방향 + 크기), Koopman (EDMD를 통한 고차원 리프팅)
- **9개 원시 연산** (기반 계층): `transform`, `gate`, `merge`, `distance`, `route` (암호화) + `step`, `branch`, `clamp`, `map` (평문)
- **행렬 수준 암호화** -- 유사 변환이 프로그램을 암호학적으로 해독 불가능하게 만듦
- **NumPy 백엔드** + 선택적 GPU 가속 (CuPy/JAX)

---

## 목차

- [패러다임 전환](#패러다임-전환)
- [3단계 실행 모델](#3단계-실행-모델)
- [품질 지표](#품질-지표)
- [카오스 이론 기반](#카오스-이론-기반)
- [합성 규칙](#합성-규칙)
- [Quantum DSL](#quantum-dsl)
- [성능](#성능)
- [기반 계층](#기반-계층)
  - [9개 원시 연산](#9개-원시-연산)
  - [행렬 암호화 (Shadow AI)](#행렬-암호화-shadow-ai)
  - [평문 연산 & 보안 등급 분류](#평문-연산--보안-등급-분류)
  - [컴파일러 최적화기](#컴파일러-최적화기)
  - [GPU 백엔드](#gpu-백엔드)
  - [모듈 시스템](#모듈-시스템)
  - [양자 간섭 (Phase 6)](#양자-간섭-phase-6)
  - [비선형 합성 (Phase 9)](#비선형-합성-phase-9)
  - [클라이언트-서버 아키텍처](#클라이언트-서버-아키텍처)
- [아키텍처](#아키텍처)
- [빠른 시작](#빠른-시작)
- [API 레퍼런스](#api-레퍼런스)
- [예제](#예제)
- [테스트 스위트](#테스트-스위트)
- [로드맵](#로드맵)

---

## 패러다임 전환

### 우리가 부정하는 것

모든 현대 프로그래밍 언어는 **시간축**(순차 실행) 위에 구축되어 있습니다:

| 패러다임 | 시간축 의존성 |
|----------|-------------|
| 명령형 (C, Python) | "먼저 이것을 하고, 그다음 저것을" -- 명시적 시간 순서 |
| 함수형 (Haskell, Lisp) | 선언적이지만 평가 순서가 존재 |
| 병렬 (Go, Rust async) | 동시에 여러 시간축 -- 여전히 시간에 종속 |
| 선언형 (SQL, HTML) | "무엇"을 기술하지만 엔진은 시간축 위에서 처리 |

이는 Von Neumann 아키텍처가 클럭 사이클, 즉 시간축 위에서 동작하기 때문입니다.

### 우리가 제안하는 것

AXOL은 시간축을 두 가지 대안적 축으로 대체합니다:

| 축 | 결정하는 것 | 비유 |
|---|-----------|------|
| **공간축** (관계) | 관계로 연결된 노드가 연산을 결정 | "언제"가 아니라 "어디에 있는가"가 중요 |
| **확률축** (가능성) | 중첩된 상태가 가장 가능성 높은 결과로 붕괴 | "정확한 값"이 아니라 "얼마나 가능한가"가 중요 |

트레이드오프: **시간 병목을 제거하는 대신 정확성을 희생합니다.**

```
정확성 ↑  →  얽힘 비용 ↑  →  구축 시간 ↑
정확성 ↓  →  얽힘 비용 ↓  →  구축 시간 ↓
             단, 관측은 항상 즉각적
```

### 왜 이것이 중요한가

| 속성 | 전통적 컴파일 | AXOL 얽힘 |
|------|-------------|-----------|
| 준비 | 코드 → 기계어 번역 | 로직 간 확률적 상관관계 구축 |
| 실행 | 순차적 기계 명령어 | 관측(입력) → 즉각적 붕괴 |
| 병목 | 실행 경로 길이에 비례 | 얽힘 깊이에만 의존 |
| 비유 | "빠른 도로를 건설" | "이미 목적지에 있는 것" |

---

## 3단계 실행 모델

### 1단계: 선언 (Declare)

**무엇이 무엇과 관련되는지** 정의하고 품질 목표를 설정합니다. 아직 연산은 일어나지 않습니다.

```python
from axol.quantum import DeclarationBuilder, RelationKind

decl = (
    DeclarationBuilder("search")
    .input("query", 64)
    .input("db", 64)
    .relate("relevance", ["query", "db"], RelationKind.PROPORTIONAL)
    .output("relevance")
    .quality(omega=0.9, phi=0.7)   # 품질 목표
    .build()
)
```

또는 DSL로:

```
entangle search(query: float[64], db: float[64]) @ Omega(0.9) Phi(0.7) {
    relevance <~> similarity(query, db)
    ranking <~> relevance
}
```

### 2단계: 직조 (Weave)

**이상 끌개**(Tapestry)를 구축합니다. 여기서 연산 비용이 지불됩니다. 직조기는:

1. 얽힘 비용 추정
2. 비실현성 감지 (목표가 수학적으로 달성 불가능하면 경고)
3. 노드별 끌개 구조 구축 (궤적 행렬, Hadamard 간섭)
4. Lyapunov 지수 및 프랙탈 차원 추정
5. 실행을 위한 내부 `Program` 조립

```python
from axol.quantum import weave

tapestry = weave(decl, seed=42)
print(tapestry.weaver_report)
# target:   Omega(0.90) Phi(0.70)
# achieved: Omega(0.95) Phi(0.82)
# feasible: True
```

비실현성 감지 예시:

```
> weave predict_weather: WARNING
>   target:   Omega(0.99) Phi(0.99)
>   maximum:  Omega(0.71) Phi(0.68)
>   reason:   chaotic dependency (lambda=2.16 on path: input->atmosphere->prediction)
>   attractor_dim: D=2.06 (Lorenz-class)
```

### 3단계: 관측 (Observe)

입력값 → 끌개 위의 한 점으로 **즉각적 붕괴**. 시간 복잡도: O(D), 여기서 D는 끌개의 임베딩 차원입니다.

```python
from axol.quantum import observe, reobserve
from axol.core.types import FloatVec

# 단일 관측
result = observe(tapestry, {
    "query": FloatVec.from_list([1.0] * 64),
    "db": FloatVec.from_list([0.5] * 64),
})
print(f"Omega: {result.omega:.2f}, Phi: {result.phi:.2f}")

# 품질 향상을 위한 반복 관측
result = reobserve(tapestry, inputs, count=10)
# 확률 분포 평균화, 경험적 Omega 재계산
```

### 전체 파이프라인 (DSL)

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

## 품질 지표

AXOL은 두 개의 독립적인 축으로 연산 품질을 측정합니다:

```
        Phi (명확도)
        ^
   1.0  |  날카롭지만 불안정     이상적 (강한 얽힘)
        |
   0.0  |  노이즈               안정적이지만 흐릿함
        +-----------------------------> Omega (응집도)
       0.0                             1.0
```

### Omega -- 응집도 (얼마나 안정적인가?)

**최대 Lyapunov 지수** (lambda)로부터 유도:

```
Omega = 1 / (1 + max(lambda, 0))
```

| lambda | 의미 | Omega |
|--------|------|-------|
| lambda < 0 | 수렴하는 시스템 (안정) | 1.0 |
| lambda = 0 | 중립 안정성 | 1.0 |
| lambda = 0.91 | Lorenz급 카오스 | 0.52 |
| lambda = 2.0 | 강한 카오스 | 0.33 |

**해석**: Omega = 1.0은 반복 관측 시 항상 같은 결과를 의미합니다. Omega < 1.0은 카오스 민감도를 의미합니다 -- 작은 입력 변화가 다른 출력을 유발합니다.

### Phi -- 명확도 (얼마나 선명한가?)

끌개의 **프랙탈 차원** (D)으로부터 유도:

```
Phi = 1 / (1 + D / D_max)
```

| D | D_max | 의미 | Phi |
|---|-------|------|-----|
| 0 | any | 점 (델타 분포) | 1.0 |
| 1 | 4 | 선 끌개 | 0.80 |
| 2.06 | 3 | Lorenz 끌개 | 0.59 |
| D_max | D_max | 전체 위상 공간을 채움 | 0.50 |

**해석**: Phi = 1.0은 출력이 날카롭고 확정적인 값임을 의미합니다. Phi → 0.0은 출력이 여러 가능성에 걸쳐 분산되어 있음(노이즈)을 의미합니다.

### 두 지표 모두 합성 가능

품질 지표는 합성을 통해 전파됩니다 -- [합성 규칙](#합성-규칙)을 참조하세요.

---

## 카오스 이론 기반

AXOL의 이론적 기반은 그 개념들을 잘 확립된 카오스 이론에 대응시킵니다:

| AXOL 개념 | 카오스 이론 | 수학적 대상 |
|-----------|-----------|------------|
| Tapestry | Strange Attractor | 위상 공간의 컴팩트 불변 집합 |
| Omega (응집도) | Lyapunov 안정성 | `1/(1+max(lambda,0))` |
| Phi (명확도) | 프랙탈 차원의 역수 | `1/(1+D/D_max)` |
| Weave | 끌개 구성 | 반복 사상의 궤적 행렬 |
| Observe | 끌개 위의 점 붕괴 | 시간 복잡도 O(D) |
| 얽힘 범위 | 끌어당김 영역(Basin of Attraction) | 수렴 영역의 경계 |
| 얽힘 비용 | 수렴 반복 횟수 | `E = sum_path(iterations * complexity)` |
| 관측 후 재사용 | 끌개 안정성 | lambda < 0: 재사용 가능, lambda > 0: 재직조 필요 |

### Lyapunov 지수 추정

궤적 행렬에서 최대 Lyapunov 지수를 추정하기 위해 **Benettin QR 분해 방법**을 사용합니다.

- **수축 시스템** (lambda < 0): 예측 가능, Omega가 1.0에 근접
- **중립 시스템** (lambda = 0): 카오스의 경계
- **카오스 시스템** (lambda > 0): 초기 조건에 민감, Omega < 1.0

알려진 시스템에 대한 추정 정확도 검증 완료 (평균 오차: 0.0002).

### 프랙탈 차원 추정

두 가지 방법 사용 가능:

- **박스 카운팅**: 격자 기반, ln(N) vs ln(1/epsilon)의 회귀
- **상관 차원** (Grassberger-Procaccia): 쌍별 거리 분석

알려진 기하학에 대한 검증 완료: 선분 (D=1), Cantor 집합 (D~0.63), Sierpinski 삼각형 (D~1.58).

### 전체 이론 문서

- [THEORY.md](THEORY.md) -- 기초 이론 (시간축 부정, 얽힘 기반 연산)
- [THEORY_MATH.md](THEORY_MATH.md) -- 카오스 이론 형식화 (Lyapunov, 프랙탈, 합성 증명)

---

## 합성 규칙

여러 tapestry를 결합할 때, 품질 지표는 엄밀한 수학적 규칙에 따라 전파됩니다:

### 직렬 합성 (A → B)

```
lambda_total = lambda_A + lambda_B          (지수 누적)
Omega_total  = 1/(1+max(lambda_total, 0))   (Omega 저하)
D_total      = D_A + D_B                    (차원 합산)
Phi_total    = Phi_A * Phi_B                (Phi 곱셈 -- 항상 저하)
```

### 병렬 합성 (A || B)

```
lambda_total = max(lambda_A, lambda_B)      (최약 연결)
Omega_total  = min(Omega_A, Omega_B)        (최약 연결)
D_total      = max(D_A, D_B)               (가장 복잡한 것)
Phi_total    = min(Phi_A, Phi_B)            (가장 불명확한 것)
```

### 재사용 규칙

```
lambda < 0  →  관측 후 재사용 가능 (끌개가 안정적)
lambda > 0  →  관측 후 재직조 필요 (카오스 -- 끌개 교란)
```

### 요약 표

| 모드 | lambda | Omega | D | Phi |
|------|--------|-------|---|-----|
| 직렬 | sum | 1/(1+max(sum,0)) | sum | product |
| 병렬 | max | min | max | min |

---

## Quantum DSL

### 문법 개요

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

### 관계 연산자

| 연산자 | 이름 | 의미 |
|--------|------|------|
| `<~>` | 비례 | 선형 상관관계 |
| `<+>` | 가산 | 가중합 |
| `<*>` | 승산 | 곱 관계 |
| `<!>` | 역비례 | 역상관관계 |
| `<?>` | 조건부 | 문맥 의존적 |

### 예제

#### 단순 검색

```
entangle search(query: float[64], db: float[64]) @ Omega(0.9) Phi(0.7) {
    relevance <~> similarity(query, db)
    ranking <~> relevance
}
result = observe search(query_vec, db_vec)
```

#### 다단계 파이프라인

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

#### 분류

```
entangle classify(input: float[32]) @ Omega(0.95) Phi(0.9) {
    category <~> classify(input)
}
result = observe classify(input_vec)
```

---

## 성능

### 토큰 효율 -- Quantum DSL vs Python

`tiktoken` cl100k_base 토크나이저로 측정.

| 프로그램 | Python 토큰 | DSL 토큰 | 절약 |
|---------|:------------:|:----------:|:------:|
| search | 173 | 57 | **67%** |
| classify | 129 | 39 | **70%** |
| pipeline | 210 | 73 | **65%** |
| multi_input | 191 | 74 | **61%** |
| reobserve_pattern | 131 | 62 | **53%** |
| **합계** | **834** | **305** | **63%** |

### 토큰 효율 -- Foundation DSL vs Python vs C#

| 프로그램 | Python | C# | Axol DSL | vs Python | vs C# |
|---------|:------:|:--:|:--------:|:---------:|:-----:|
| Counter | 32 | 61 | 33 | -3% | 46% |
| State Machine | 67 | 147 | 48 | 28% | 67% |
| Combat Pipeline | 145 | 203 | 66 | 55% | 68% |
| 100-State Automaton | 739 | 869 | 636 | 14% | 27% |

### 정확도

| 지표 | 값 |
|------|-----|
| Lyapunov 추정 평균 오차 | **0.0002** |
| Omega 공식 오차 | **0** (정확) |
| Phi 공식 오차 | **0** (정확) |
| 합성 규칙 | **전체 PASS** |
| 관측 일관성 (50회 반복) | **1.0000** |

### 속도

| 연산 | 시간 |
|------|------|
| DSL 파싱 (단순) | ~25 us |
| DSL 파싱 (전체 프로그램) | ~62 us |
| 비용 추정 | ~40 us |
| 단일 관측 | ~300 us |
| 직조 (2노드, dim=8) | ~14 ms |
| 재관측 x10 | ~14 ms |
| **전체 파이프라인** (파싱 → 직조 → 관측, dim=16) | **~17 ms** |

### 스케일링

| 노드 | 차원 | 직조 시간 |
|:-----:|:---------:|:----------:|
| 1 | 8 | 9 ms |
| 4 | 8 | 25 ms |
| 16 | 8 | 108 ms |
| 2 | 4 | 12 ms |
| 2 | 64 | 39 ms |

전체 벤치마크 데이터: [QUANTUM_PERFORMANCE_REPORT.md](QUANTUM_PERFORMANCE_REPORT.md) | [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md)

---

## 기반 계층

양자 모듈(`axol/quantum/`)은 기반 계층(`axol/core/`)을 수정하지 않고 그 위에 구축됩니다. 기반 계층은 수학적 엔진을 제공합니다: 벡터 타입, 행렬 연산, 프로그램 실행, 암호화, 최적화.

### 9개 원시 연산

| 연산 | 보안 | 수학적 기반 | 설명 |
|------|:----:|-----------|------|
| `transform` | **E** | 행렬 곱: `v @ M` | 선형 상태 변환 |
| `gate` | **E** | Hadamard 곱: `v * g` | 조건부 마스킹 |
| `merge` | **E** | 가중합: `sum(v_i * w_i)` | 벡터 결합 |
| `distance` | **E** | L2 / cosine / dot | 유사도 측정 |
| `route` | **E** | `argmax(v @ R)` | 이산 분기 |
| `step` | **P** | `where(v >= t, 1, 0)` | 임계값 이진 게이트 |
| `branch` | **P** | `where(g, then, else)` | 조건부 벡터 선택 |
| `clamp` | **P** | `clip(v, min, max)` | 값 범위 제한 |
| `map` | **P** | `f(v)` element-wise | 비선형 활성화 |

5개 **E** (암호화) 연산은 유사 변환을 통해 암호화된 데이터에서 실행 가능합니다. 4개 **P** (평문) 연산은 비선형 표현력을 추가합니다.

### 행렬 암호화 (Shadow AI)

Axol의 모든 연산은 행렬 곱셈(`v @ M`)으로 환원됩니다. 이것이 **유사 변환 암호화**를 가능하게 합니다:

```
M' = K^(-1) @ M @ K     (암호화된 연산 행렬)
state' = state @ K       (암호화된 초기 상태)
result = result' @ K^(-1)(복호화된 출력)
```

- 암호화된 프로그램이 암호화된 도메인에서 정상 실행됨
- 모든 비즈니스 로직이 은닉됨 -- 행렬이 랜덤 노이즈처럼 보임
- 난독화와 다름 -- 암호학적 변환
- 5개 E 연산 모두 검증 완료 (`tests/test_encryption.py`의 21개 테스트)

### 평문 연산 & 보안 등급 분류

모든 연산은 `SecurityLevel` (E 또는 P)을 가집니다. 내장 분석기가 암호화 커버리지를 보고합니다:

```python
from axol.core import parse, analyze

result = analyze(program)
print(result.summary())
# Program: damage_calc
# Transitions: 3 total, 1 encrypted (E), 2 plaintext (P)
# Coverage: 33.3%
```

### 컴파일러 최적화기

3-패스 최적화: transform 융합, 데드 상태 제거, 상수 폴딩.

```python
from axol.core import parse, optimize, run_program

program = parse(source)
optimized = optimize(program)
result = run_program(optimized)
```

### GPU 백엔드

교체 가능한 배열 백엔드: `numpy` (기본값), `cupy` (NVIDIA GPU), `jax`.

```python
from axol.core import set_backend
set_backend("cupy")   # NVIDIA GPU
set_backend("jax")    # Google JAX
```

### 모듈 시스템

스키마, 임포트, 서브 모듈 실행을 지원하는 재사용 가능하고 합성 가능한 프로그램.

```
@main
import damage_calc from "damage_calc.axol"
s atk=[50] def_val=[10]
: calc=use(damage_calc;in=atk,def_val;out=dmg)
```

### 양자 간섭 (Phase 6)

Phase 6에서 양자 간섭 -- Grover 검색, 양자 걸음(quantum walk) -- 을 도입하여 양자 프로그램에 대해 **100% 암호화 커버리지**를 달성했습니다. Hadamard, Oracle, Diffusion은 `TransformOp` (E-class)를 통해 사용되는 `TransMatrix` 객체를 생성하므로, 기존 최적화기와 암호화 모듈이 자동으로 작동합니다.

```
@grover_4
s state=[0.5 0.5 0.5 0.5]
: oracle=oracle(state;marked=[3];n=4)
: diffuse=diffuse(state;n=4)
? found state[3]>=0.9
```

### 비선형 합성 (Phase 9)

비선형 파이프라인을 합성하는 3가지 접근법으로, 정확도와 속도를 트레이드오프합니다:

| 접근법 | 방법 | 차원 | 정확도 | 속도 |
|--------|------|:----:|:------:|:----:|
| **순수 유니터리** | SVD 극분해: `A = U @ S @ Vh → U @ Vh` | dim x dim | 방향만 | 가장 빠름 |
| **하이브리드** | SVD 전체: 회전 + 특이값 | dim x dim | 방향 + 크기 | 빠름 |
| **Koopman** | EDMD를 통한 다항식 공간 리프팅 | lifted_dim x lifted_dim | 완전 비선형 | 가장 느림 |

```python
from axol.quantum import (
    estimate_unitary_step, compose_unitary_chain,       # 순수 유니터리
    estimate_hybrid_step, compose_hybrid_chain,          # 하이브리드
    estimate_koopman_matrix, compose_koopman_chain,      # Koopman
    lift, unlift, lifted_dim,                            # Koopman 유틸리티
)

# 순수 유니터리: 방향만 합성
U1 = estimate_unitary_step(step_fn, dim=4)
U2 = estimate_unitary_step(step_fn2, dim=4)
U_chain = compose_unitary_chain([U1, U2])  # 재직교화 적용

# 하이브리드: 방향 + 크기 (개방 양자 시스템)
A1 = estimate_hybrid_step(step_fn, dim=4)
A2 = estimate_hybrid_step(step_fn2, dim=4)
composed, rotation, scales = compose_hybrid_chain([A1, A2])
# rotation = 양자 게이트 (유니터리), scales = 결맞음 깨짐 가중치

# Koopman: EDMD를 통한 완전 비선형
K = estimate_koopman_matrix(step_fn, dim=4, degree=2)  # lifted_dim=15
K_chain = compose_koopman_chain([K1, K2])
y = unlift(lift(x) @ K_chain.data, dim=4)
```

하이브리드는 개방 양자 시스템에 자연스럽게 대응됩니다: 유니터리 부분 = 고립 양자 진화, 스케일 부분 = 환경 상호작용 (결맞음 깨짐).

### 클라이언트-서버 아키텍처

클라이언트에서 암호화하고, 신뢰할 수 없는 서버에서 연산:

```
클라이언트 (키 보유)         서버 (키 없음)
  프로그램 ─── 암호화 ──► 암호화된 프로그램
  pad_and_encrypt()       run_program() (노이즈 위에서)
                    ◄──── 암호화된 결과
  decrypt_result()
  ──► 결과
```

주요 구성요소: `KeyFamily(seed)`, `fn_to_matrix()`, `pad_and_encrypt()`, `AxolClient` SDK.

---

## 아키텍처

```
                    ┌─────────────────────────────────────────────┐
                    │            axol/quantum/                     │
                    │                                             │
  Quantum DSL ──►  │  dsl.py ──► declare.py ──► weaver.py ──►   │
  (entangle,       │               │              │   │          │
   observe,        │  types.py   cost.py    lyapunov.py          │
   reobserve)      │               │        fractal.py           │
                    │           compose.py                        │
                    │           koopman.py  (EDMD 리프팅)          │
                    │           unitary.py  (SVD 합성)            │
                    │               │                             │
                    │           observatory.py ──► Observation    │
                    └──────────────┬──────────────────────────────┘
                                   │ 재사용
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
                    │  Tool-Use API    FastAPI + HTML/JS 디버거    │
                    └─────────────────────────────────────────────┘
```

### 내부 엔진 재사용

양자 모듈은 `axol/core`를 수정 없이 재사용합니다:

| 양자 개념 | 코어 구현체 |
|----------|-----------|
| 끌개 진폭/궤적 | `FloatVec` |
| 끌개 상관 행렬 | `TransMatrix` |
| Tapestry 내부 실행 | `Program` + `run_program()` |
| Born rule 확률 | `operations.measure()` |
| 직조 변환 구성 | `TransformOp`, `MergeOp` |
| 끌개 탐색 확산 | `hadamard_matrix()`, `diffusion_matrix()` |
| Koopman 리프팅 / EDMD | `TransMatrix`, `np.linalg.lstsq` |
| 유니터리 / 하이브리드 SVD | `TransMatrix`, `np.linalg.svd` |

---

## 빠른 시작

### 설치

```bash
git clone https://github.com/your-username/AXOL.git
cd AXOL
pip install -e ".[dev]"
```

### 요구사항

- Python 3.11+
- NumPy >= 1.24.0
- pytest >= 7.4.0 (개발용)
- tiktoken >= 0.5.0 (개발용, 토큰 분석)
- fastapi >= 0.100.0, uvicorn >= 0.23.0 (선택, 웹 프론트엔드)
- cupy-cuda12x >= 12.0.0 (선택, GPU)
- jax[cpu] >= 0.4.0 (선택, JAX 백엔드)

### Hello World -- Quantum DSL (선언 → 직조 → 관측)

```python
from axol.quantum import (
    DeclarationBuilder, RelationKind,
    weave, observe, parse_quantum,
)
from axol.core.types import FloatVec

# 방법 1: Python API
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

# 방법 2: DSL
program = parse_quantum("""
entangle hello(x: float[4]) @ Omega(0.9) Phi(0.8) {
    y <~> transform(x)
}
""")
```

### Hello World -- Foundation DSL (벡터 연산)

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
print(f"최종 카운트: {result.final_state['count'].to_list()}")  # [5.0]
```

---

## API 레퍼런스

### 양자 모듈 (`axol.quantum`)

```python
# 선언
DeclarationBuilder(name)           # 선언 구축을 위한 Fluent API
  .input(name, dim, labels?)       # 입력 추가
  .output(name)                    # 출력 지정
  .relate(target, sources, kind)   # 관계 추가
  .quality(omega, phi)             # 품질 목표 설정
  .build() -> EntangleDeclaration

# 직조
weave(declaration, encrypt?, seed?, optimize?) -> Tapestry

# 관측
observe(tapestry, inputs, seed?) -> Observation
reobserve(tapestry, inputs, count, seed?) -> Observation

# DSL
parse_quantum(source) -> QuantumProgram

# Lyapunov
estimate_lyapunov(trajectory_matrix, steps?) -> float
lyapunov_spectrum(trajectory_matrix, dim, steps?) -> list[float]
omega_from_lyapunov(lyapunov) -> float

# 프랙탈
estimate_fractal_dim(attractor_points, method?, phase_space_dim?) -> float
phi_from_fractal(fractal_dim, phase_space_dim) -> float
phi_from_entropy(probs) -> float

# 합성
compose_serial(omega_a, phi_a, lambda_a, d_a, ...) -> tuple
compose_parallel(omega_a, phi_a, lambda_a, d_a, ...) -> tuple
can_reuse_after_observe(lyapunov) -> bool

# 비용
estimate_cost(declaration) -> CostEstimate

# Koopman
lifted_dim(dim, degree?, basis?) -> int
lift(x, degree?, basis?) -> ndarray
unlift(y_lifted, dim, degree?, basis?) -> ndarray
estimate_koopman_matrix(step_fn, dim, degree?, n_samples?, seed?, basis?) -> TransMatrix
compose_koopman_chain(matrices) -> TransMatrix

# 유니터리 / 하이브리드
nearest_unitary(A) -> ndarray
reorthogonalize(U) -> ndarray
estimate_unitary_step(step_fn, dim, n_samples?, seed?) -> TransMatrix
compose_unitary_chain(matrices) -> TransMatrix
estimate_hybrid_step(step_fn, dim, n_samples?, seed?) -> TransMatrix
compose_hybrid_chain(matrices) -> tuple[TransMatrix, TransMatrix, ndarray]
```

### 코어 타입

| 타입 | 설명 |
|------|------|
| `SuperposedState` | 진폭, 레이블, Born rule 확률을 가진 명명된 상태 |
| `Attractor` | Lyapunov 스펙트럼, 프랙탈 차원, 궤적 행렬을 가진 이상 끌개 |
| `Tapestry` | 전역 끌개와 직조기 보고서를 가진 `TapestryNode` 그래프 |
| `Observation` | 값, Omega, Phi, 확률을 가진 붕괴된 결과 |
| `WeaverReport` | 목표 vs 달성 품질, 실현 가능성, 비용 분석 |
| `CostEstimate` | 노드별 비용, 임계 경로, 최대 달성 가능 Omega/Phi |
| `FloatVec` | 32비트 부동소수점 벡터 |
| `TransMatrix` | M x N float32 행렬 |
| `StateBundle` | 벡터의 명명된 컬렉션 |
| `Program` | 전이의 실행 가능한 시퀀스 |

### 기반 모듈 (`axol.core`)

```python
parse(source) -> Program
run_program(program) -> ExecutionResult
optimize(program) -> Program
set_backend(name)    # "numpy" | "cupy" | "jax"
analyze(program) -> AnalysisResult
dispatch(request) -> dict    # Tool-Use API
```

---

## 예제

### 1. 선언 → 직조 → 관측 (전체 파이프라인)

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

### 2. Quantum DSL 왕복

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

### 3. 상태 머신 (Foundation DSL)

```
@state_machine
s state=onehot(0,3)
: advance=transform(state;M=[0 1 0;0 0 1;0 0 1])
? done state[2]>=1
```

### 4. Grover 검색 (양자 간섭)

```
@grover_4
s state=[0.5 0.5 0.5 0.5]
: oracle=oracle(state;marked=[3];n=4)
: diffuse=diffuse(state;n=4)
? found state[3]>=0.9
```

### 5. 암호화 실행

```python
from axol.core import parse, run_program
from axol.core.encryption import encrypt_program, decrypt_state

program = parse("@test\ns v=[1 0 0]\n: t=transform(v;M=[0 1 0;0 0 1;1 0 0])")
encrypted, key = encrypt_program(program)
result = run_program(encrypted)
decrypted = decrypt_state(result.final_state, key)
```

---

## 테스트 스위트

```bash
# 전체 테스트 스위트 (767개 테스트)
pytest tests/ -v

# 양자 모듈 테스트 (101개 테스트)
pytest tests/test_quantum_*.py tests/test_lyapunov.py tests/test_fractal.py tests/test_compose.py -v

# Koopman / 유니터리 / 하이브리드 테스트
pytest tests/test_koopman.py tests/test_unitary.py tests/test_hybrid.py tests/test_distill.py -v

# 성능 벤치마크 (보고서 생성)
pytest tests/test_quantum_performance.py -v -s
pytest tests/test_performance_report.py -v -s

# 코어 테스트
pytest tests/test_types.py tests/test_operations.py tests/test_program.py tests/test_dsl.py -v

# 암호화 테스트 (21개 테스트)
pytest tests/test_encryption.py -v -s

# 양자 간섭 테스트 (37개 테스트)
pytest tests/test_quantum.py -v -s

# API + 서버 테스트
pytest tests/test_api.py tests/test_server.py -v

# 웹 프론트엔드 시작
python -m axol.server   # http://localhost:8080
```

현재: **767개 테스트 통과**, 0 실패, 4 건너뜀 (cupy/jax 미설치).

---

## 로드맵

- [x] Phase 1: 타입 시스템 (7개 벡터 타입 + StateBundle) + 5개 원시 연산 + 실행 엔진
- [x] Phase 2: DSL 파서 + 희소 행렬 표기법 + 토큰 벤치마크 + 암호화 PoC
- [x] Phase 3: 컴파일러 최적화기 (융합, 제거, 폴딩) + GPU 백엔드
- [x] Phase 4: Tool-Use API + 암호화 모듈
- [x] Phase 5: 모듈 시스템 (레지스트리, import/use, compose, 스키마)
- [x] 프론트엔드: FastAPI + HTML/JS 비주얼 디버거
- [x] Phase 6: 양자 간섭 (Hadamard/Oracle/Diffusion, 100% E-class 커버리지)
- [x] Phase 7: KeyFamily, 직사각형 암호화, fn_to_matrix, 패딩, branch 컴파일, AxolClient SDK
- [x] Phase 8: 카오스 이론 양자 모듈 -- 선언 → 직조 → 관측 파이프라인
- [x] Phase 8: Lyapunov 지수 추정 (Benettin QR) + Omega = 1/(1+max(lambda,0))
- [x] Phase 8: 프랙탈 차원 추정 (박스 카운팅/상관) + Phi = 1/(1+D/D_max)
- [x] Phase 8: 직조기, 관측소, 합성 규칙, 비용 추정, DSL 파서
- [x] Phase 8: 101개 신규 테스트 (총 545개, 0 실패)
- [x] Phase 9: 비선형 합성 -- 순수 유니터리 (SVD 극분해), 하이브리드 (방향 + 크기), Koopman (EDMD 리프팅)
- [x] Phase 9: 다항식 & 확장 기저를 갖춘 Koopman 연산자 (PWA 포착)
- [x] Phase 9: 222개 신규 테스트 (총 767개, 0 실패)
- [ ] Phase 10: 복소 진폭 (a+bi) -- Shor, QPE, QFT를 위한 완전 위상 간섭
- [ ] Phase 11: 분산 tapestry 직조 (다중 노드)
- [ ] Phase 12: 적응형 품질 -- 관측 중 동적 Omega/Phi 조정

---

## 라이선스

MIT 라이선스. 자세한 내용은 [LICENSE](LICENSE)를 참조하세요.
