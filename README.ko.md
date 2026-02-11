<p align="center">
  <h1 align="center">AXOL</h1>
  <p align="center">
    <strong>토큰 효율적 벡터 프로그래밍 언어</strong>
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

> **경고: 이 프로젝트는 초기 실험 단계입니다.**
> API, DSL 문법, 내부 아키텍처는 예고 없이 변경될 수 있습니다. 프로덕션 사용은 권장하지 않습니다. 기여와 피드백을 환영합니다.

---

## Axol이란?

**Axol**은 **AI 에이전트**가 기존 프로그래밍 언어보다 **적은 토큰**으로 프로그램을 읽고, 쓰고, 추론할 수 있도록 처음부터 설계된 도메인 특화 언어(DSL)입니다.

전통적인 제어 흐름(if/else, for 루프, 함수 호출) 대신, Axol은 모든 연산을 불변 벡터 번들에 대한 **벡터 변환**과 **상태 전이**로 표현합니다. 이 설계 선택은 단순한 관찰에서 시작합니다: **LLM은 토큰당 비용을 지불**하며, 기존 프로그래밍 언어는 토큰 효율성이 아닌 인간 가독성을 위해 설계되었습니다.

### 핵심 특성

- Python 대비 **30~50% 적은 토큰** 소모
- C# 대비 **48~75% 적은 토큰** 소모
- **9개 원시 연산**으로 모든 계산 표현: `transform`, `gate`, `merge`, `distance`, `route` (암호화) + `step`, `branch`, `clamp`, `map` (평문)
- **희소 행렬 표기법**: 밀집 표현의 O(N^2) 대비 O(N)으로 스케일링
- 완전한 상태 추적이 가능한 **결정론적 실행**
- **NumPy 백엔드**로 대규모 벡터 연산 지원 (대규모 차원에서 Python 순수 루프 대비 빠름)
- **E/P 보안 등급 분류** - 각 연산은 암호화(E) 또는 평문(P)으로 분류되며, 암호화 커버리지와 표현력 간 트레이드오프를 내장 분석기로 가시화
- **행렬 수준 암호화** - 비밀 키 행렬로 프로그램을 암호학적으로 해독 불가능하게 만들어, 셰도우 AI 문제를 근본적으로 해결

---

## 목차

- [이론적 배경](#이론적-배경)
- [셰도우 AI와 행렬 암호화](#셰도우-ai와-행렬-암호화)
  - [암호화 증명: 5개 연산 모두 검증 완료](#암호화-증명-5개-연산-모두-검증-완료)
- [평문 연산 & 보안 등급](#평문-연산--보안-등급)
- [아키텍처](#아키텍처)
- [빠른 시작](#빠른-시작)
- [DSL 문법](#dsl-문법)
- [컴파일러 최적화](#컴파일러-최적화)
- [GPU 백엔드](#gpu-백엔드)
- [모듈 시스템](#모듈-시스템)
- [Tool-Use API](#tool-use-api)
- [웹 프론트엔드](#웹-프론트엔드)
- [토큰 비용 비교](#토큰-비용-비교)
- [런타임 성능](#런타임-성능)
- [성능 벤치마크](#성능-벤치마크)
- [API 레퍼런스](#api-레퍼런스)
- [예제](#예제)
- [테스트](#테스트)
- [Phase 6: Quantum Axol](#phase-6-quantum-axol)
- [Phase 8: 카오스 이론 양자 모듈](#phase-8-카오스-이론-양자-모듈)
- [로드맵](#로드맵)

---

## 이론적 배경

### 토큰 경제 문제

현대 AI 시스템(GPT-4, Claude 등)은 **토큰 경제** 하에서 동작합니다. 입출력의 모든 문자가 토큰을 소비하며, 이는 비용과 지연 시간에 직접 영향을 미칩니다. AI 에이전트가 코드를 작성하거나 읽을 때, 프로그래밍 언어의 장황함은 다음에 직접적으로 영향을 줍니다:

1. **비용** - 더 많은 토큰 = 더 높은 API 비용
2. **지연 시간** - 더 많은 토큰 = 더 느린 응답
3. **컨텍스트 윈도우** - 더 많은 토큰 = 다른 정보를 위한 공간 감소
4. **추론 정확도** - 압축된 표현이 노이즈를 줄임

### 왜 벡터 연산인가?

전통적 프로그래밍 언어는 **제어 흐름**(분기, 루프, 재귀)으로 로직을 표현합니다. 이는 인간에게 직관적이지만 AI에게는 비효율적입니다:

```python
# Python: 67 토큰
TRANSITIONS = {"IDLE": "RUNNING", "RUNNING": "DONE", "DONE": "DONE"}
def state_machine():
    state = "IDLE"
    steps = 0
    while state != "DONE":
        state = TRANSITIONS[state]
        steps += 1
    return state, steps
```

동일한 로직을 벡터 변환으로:

```
# Axol DSL: 48 토큰 (28% 절약)
@state_machine
s state=onehot(0,3)
: advance=transform(state;M=[0 1 0;0 0 1;0 0 1])
? done state[2]>=1
```

상태 머신의 전이 테이블이 **행렬**이 되고, 상태 전진이 **행렬 곱셈**이 됩니다. AI는 문자열 비교, 딕셔너리 조회, 루프 조건을 추론할 필요 없이 단일 행렬 연산만 처리하면 됩니다.

### 아홉 가지 원시 연산

Axol은 9개의 원시 연산을 제공합니다. 처음 5개는 **암호화(E)** 가능 - 암호화된 데이터에서 실행할 수 있습니다. 나머지 4개는 **평문(P)** 전용 - 평문이 필요하지만 비선형 표현력을 추가합니다:

| 연산 | 보안 등급 | 수학적 기반 | 설명 |
|------|:--------:|-----------|------|
| `transform` | **E** | 행렬 곱: `v @ M` | 선형 상태 변환 |
| `gate` | **E** | 아다마르 곱: `v * g` | 조건부 마스킹 (0/1) |
| `merge` | **E** | 가중합: `sum(v_i * w_i)` | 벡터 결합 |
| `distance` | **E** | L2 / 코사인 / 내적 | 유사도 측정 |
| `route` | **E** | `argmax(v @ R)` | 이산 분기 |
| `step` | **P** | `where(v >= t, 1, 0)` | 임계값 이진 게이트 |
| `branch` | **P** | `where(g, then, else)` | 조건부 벡터 선택 |
| `clamp` | **P** | `clip(v, min, max)` | 값 범위 제한 |
| `map` | **P** | `f(v)` 요소별 적용 | 비선형 활성화 (relu, sigmoid, abs, neg, square, sqrt) |

5개 E 연산은 암호화 연산을 위한 **선형대수 기반**을 형성합니다:
- 상태 머신 (transform)
- 조건 로직 (gate)
- 누적/집계 (merge)
- 유사도 검색 (distance)
- 의사 결정 (route)

4개 P 연산은 AI/ML 워크로드를 위한 **비선형 표현력**을 추가합니다:
- 활성화 함수 (map: relu, sigmoid)
- 임계값 결정 (step + branch)
- 값 정규화 (clamp)

### 희소 행렬 표기법

대규모 상태 공간에서 밀집 행렬 표현은 토큰 기준 O(N^2)입니다. Axol의 희소 표기법은 이를 O(N)으로 줄입니다:

```
# 밀집: O(N^2) 토큰 - N=100이면 비실용적
M=[0 1 0 0 ... 0; 0 0 1 0 ... 0; ...]

# 희소: O(N) 토큰 - 선형 스케일링
M=sparse(100x100;0,1=1 1,2=1 ... 98,99=1 99,99=1)
```

| N | Python | C# | Axol DSL | DSL/Python | DSL/C# |
|---|--------|-----|----------|------------|--------|
| 5 | 74 | 109 | 66 | 0.89x | 0.61x |
| 25 | 214 | 269 | 186 | 0.87x | 0.69x |
| 100 | 739 | 869 | 636 | 0.86x | 0.73x |
| 200 | 1,439 | 1,669 | 1,236 | 0.86x | 0.74x |

---

## 셰도우 AI와 행렬 암호화

### 셰도우 AI 문제

**셰도우 AI(Shadow AI)**란 비인가 AI 에이전트가 독점 비즈니스 로직을 유출, 복제, 역공학하는 위험을 말합니다. AI 에이전트가 자율적으로 코드를 작성하고 실행하는 시대에, 전통적 소스 코드는 치명적인 공격 표면이 됩니다:

- AI 에이전트의 프롬프트와 생성된 코드가 **프롬프트 인젝션으로 추출** 가능
- Python/C#/JavaScript 코드는 **설계상 인간이 읽을 수 있음** - 난독화는 되돌릴 수 있음
- 코드에 내장된 독점 알고리즘, 의사결정 규칙, 영업 비밀이 **평문으로 노출**
- 전통적 난독화(변수 이름 변경, 제어 흐름 평탄화)는 장벽을 약간 높일 뿐 - 로직은 구조적으로 온전하며 복구 가능

### Axol의 해법: 행렬 수준 암호화

**Axol의 모든 연산이 행렬 곱셈**(`v @ M`)으로 환원되기 때문에, 전통적 프로그래밍 언어에서는 불가능한 수학적 속성이 가능해집니다: **유사 변환(similarity transformation) 암호화**.

비밀 가역 키 행렬 **K**가 주어지면, 모든 Axol 프로그램을 암호화할 수 있습니다:

```
원본 프로그램:     state  -->  M  -->  new_state
암호화 프로그램:   state' -->  M' -->  new_state'

여기서:
  M' = K^(-1) @ M @ K          (암호화된 연산 행렬)
  state' = state @ K            (암호화된 초기 상태)
  result  = result' @ K^(-1)    (복호화된 최종 출력)
```

이것은 난독화가 아니라 **암호학적 변환**입니다. 암호화된 프로그램은:

1. **암호화된 도메인에서 정상 실행됨** (행렬 대수가 켤레 변환을 보존)
2. **암호화된 출력을 생성** - K^(-1) 없이는 해독 불가
3. **모든 비즈니스 로직이 은닉됨** - 행렬 M'은 K 없이 M과 수학적으로 무관
4. **역공학에 저항** - M'에서 K를 복원하는 것은 N이 커질수록 난이도가 증가하는 행렬 분해 문제임. 일반적 경우에 대한 다항 시간 알고리즘이 알려져 있지 않으나, 형식적 암호학적 난이도 증명은 진행 중인 연구 분야임

### 구체적 예시

```
# 원본: 상태 머신 전이 행렬 (비즈니스 로직이 보임)
M = [0 1 0]    # IDLE -> RUNNING
    [0 0 1]    # RUNNING -> DONE
    [0 0 1]    # DONE -> DONE (흡수 상태)

# 비밀 키 K로 암호화 후:
M' = [0.73  -0.21   0.48]    # K 없이는 무의미
     [0.15   0.89  -0.04]    # 상태 머신 구조 추론 불가
     [0.52   0.33   0.15]    # 랜덤 노이즈처럼 보임
```

암호화된 프로그램은 여전히 정상 실행되지만 (행렬 대수가 `K^(-1)(KvM)K = vM`을 보장), DSL 텍스트에는 **암호화된 행렬만** 포함됩니다. `.axol` 파일 전체가 유출되더라도:

- 상태 이름이 보이지 않음 (벡터가 암호화됨)
- 전이 로직이 보이지 않음 (행렬이 암호화됨)
- 터미널 조건이 무의미함 (임계값이 암호화된 값에서 동작)

### 전통적 언어에서는 왜 불가능한가

| 속성 | Python/C#/JS | FHE | Axol |
|------|-------------|-----|------|
| 코드 의미론 | 평문 제어 흐름 | 암호화 (모든 연산) | 행렬 곱셈 |
| 난독화 | 되돌릴 수 있음 (변수 이름 변경, 흐름 평탄화) | 해당 없음 | 해당 없음 |
| 암호화 | 불가능 (파싱 가능해야 함) | 완전 (모든 연산) | 선형 연산만 (9개 중 5개) |
| 성능 오버헤드 | 해당 없음 | 1000-10000배 | ~0% (파이프라인 모드) |
| 복잡도 | 해당 없음 | 매우 높음 | 낮음 (키 행렬만) |
| 코드 유출 시 | 전체 로직 노출 | 암호화됨 | 랜덤처럼 보이는 숫자들 |
| 키 분리 | 불가능 | 필수 | 키 행렬을 별도 저장 (HSM, 보안 영역) |
| 암호화 후 정확성 | 해당 없음 | 수학적으로 보장됨 | 수학적으로 보장됨 |

### 보안 아키텍처

```
  [개발자]                        [배포 환경]
     |                               |
  원본 .axol                    암호화된 .axol
  (읽을 수 있는 로직)            (암호화된 행렬)
     |                               |
     +--- K (비밀 키) -------------->|
     |    HSM/보안 영역에 저장       |
     v                               v
  encrypt(M, K) = K^(-1)MK      run_program(암호화된 프로그램)
                                     |
                                암호화된 출력
                                     |
                                decrypt(output, K^(-1))
                                     |
                                실제 결과
```

비밀 키 행렬 K는 다음과 같이 관리할 수 있습니다:
- **하드웨어 보안 모듈(HSM)**에 저장
- **키 관리 서비스(KMS)**로 관리
- 프로그램 구조 변경 없이 주기적으로 교체
- 배포 환경별(dev/staging/prod) 다른 키 사용

Axol은 행렬 기반 연산에 대한 완전 동형 암호(FHE)의 경량 대안을 제공합니다. FHE(임의 연산을 지원하지만 높은 오버헤드)와 달리, Axol의 유사 변환은 효율적이지만 선형 연산에 제한됩니다. 이 트레이드오프는 5개 암호화 연산으로 충분한 특정 사용 사례에서 실용적입니다.

### 암호화 증명: 5개 연산 모두 검증 완료

Axol의 5개 연산 모두에 대한 암호화 호환성이 **수학적으로 증명되고 테스트**되었습니다 (`tests/test_encryption.py`, 21개 테스트):

| 연산 | 암호화 방법 | Key 제약 | 상태 |
|------|-----------|---------|------|
| `transform` | `M' = K^(-1) M K` (유사 변환) | 임의 가역 행렬 K | **증명됨** |
| `gate` | `diag(g)` 행렬로 변환 후 transform과 동일 | 임의 가역 행렬 K | **증명됨** |
| `merge` | 선형성: `w*(v@K) = (wv)@K` (자동 호환) | 임의 가역 행렬 K | **증명됨** |
| `distance` | `\|\|v@K\|\| = \|\|v\|\|` (직교 행렬이 노름 보존) | 직교 행렬 K | **증명됨** |
| `route` | `R' = K^(-1) R` (좌측 곱만) | 임의 가역 행렬 K | **증명됨** |

**복잡한 다중 연산 프로그램도 검증 완료:**

- HP 감소 (transform + merge 루프) - 암호화/복호화 결과 일치
- 3-상태 FSM (연쇄 transform) - 암호화 도메인에서 정확한 상태 전이
- 전투 파이프라인 (transform + gate + merge) - 3개 연산 연쇄, 오차 < 0.001
- 20-상태 오토마톤 (희소 행렬, 19단계) - 암호화 실행 결과 원본과 일치
- 50x50 대규모 행렬 - float32 정밀도 유지

**테스트로 증명된 보안 속성:**

- 암호화된 행렬은 랜덤 노이즈처럼 보임 (희소 -> 밀집, 구조 식별 불가)
- 다른 키로 암호화하면 완전히 다른 결과
- 100개 랜덤 키로 brute-force 시도해도 원본 복구 불가
- OneHot 벡터 구조가 암호화 후 완전히 은닉됨

---

## 평문 연산 & 보안 등급

### 평문 연산이 필요한 이유

기존 5개 암호화 연산은 **선형** 연산만 표현할 수 있습니다. 실제 AI/ML 워크로드에는 **비선형** 연산(활성화 함수, 조건 분기, 값 클램핑)이 필요합니다. 4개의 새 평문 연산이 이 격차를 메웁니다.

### SecurityLevel 열거형

모든 연산은 `SecurityLevel`을 가집니다:

```python
from axol.core import SecurityLevel

SecurityLevel.ENCRYPTED  # "E" - 암호화된 데이터에서 실행 가능
SecurityLevel.PLAINTEXT  # "P" - 평문 필요
```

### 암호화 커버리지 분석기

내장 분석기가 프로그램의 암호화 가능 비율을 보고합니다:

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
# Encryptable keys: (E 연산만 접근하는 키)
# Plaintext keys: (P 연산이 접근하는 키)
```

### 보안-표현력 트레이드오프

P 연산을 추가하면 표현력이 증가하지만 암호화 커버리지가 감소합니다:

| 프로그램 유형 | 암호화 커버리지 | 표현력 |
|-------------|---------------|--------|
| E 연산만 | 100% | 선형만 |
| E+P 혼합 | 30-70% (일반적) | 완전 (비선형) |
| P 연산만 | 0% | 완전 (비선형) |

비선형 연산(활성화 함수, 조건 분기)이 필요한 프로그램은 부분적 암호화 커버리지를 수용해야 합니다. 내장 분석기를 사용하여 프로그램의 커버리지를 측정하고 평문 접근이 필요한 키를 확인하세요.

### 새 연산 토큰 비용 (Python vs C# vs Axol DSL)

| 프로그램 | Python | C# | Axol DSL | vs Python | vs C# |
|---------|-------:|---:|--------:|---------:|------:|
| ReLU 활성화 | 48 | 82 | 28 | 42% | 66% |
| 임계값 선택 | 140 | 184 | 80 | 43% | 57% |
| 값 클램프 | 66 | 95 | 31 | 53% | 67% |
| Sigmoid 활성화 | 57 | 88 | 28 | 51% | 68% |
| 데미지 파이프라인 | 306 | 326 | 155 | 49% | 53% |
| **합계** | **617** | **775** | **322** | **48%** | **59%** |

### 새 연산 런타임 (dim=10,000)

| 연산 | Python 루프 | Axol (NumPy) | 속도 향상 |
|------|----------:|----------:|--------:|
| ReLU | 575 us | 21 us | **27x** |
| Sigmoid | 1.7 ms | 42 us | **40x** |
| Step+Branch | 889 us | 96 us | **9x** |
| Clamp | 937 us | 16 us | **58x** |
| 데미지 파이프라인 | 3.8 ms | 191 us | **20x** |

---

## 아키텍처

```
                                          +-------------+
  .axol 소스 -----> 파서 (dsl.py) ------> | Program     |
                         |                | + optimize()|
                         v                +------+------+
                    모듈 시스템                   |
                    (module.py)                  v
                      - import             +-----------+    +-----------+
                      - use()              |  실행엔진  |--->|  검증기   |
                      - compose()          |(program.py)|    |(verify.py)|
                                           +-----------+    +-----------+
                                                |
                    +-----------+    +----------+----------+
                    |  백엔드   |<---|    연산 모듈         |
                    |(backend.py)|    | (operations.py)     |
                    | numpy/cupy|    +---------------------+
                    | /jax      |               |
                    +-----------+    +-----------+----------+
                                    |      타입 시스템      |
                                    |   (types.py)         |
                    +-----------+   +----------------------+
                    |  암호화   |   +-----------+
                    |(encryption|   |  분석기   |
                    |       .py)|   |(analyzer  |
                    +-----------+   |       .py)|
                                    +-----------+
                    +-----------+    +-----------+
                    | Tool API  |    |  서버     |
                    |(api/)     |    |(server/)  |
                    | dispatch  |    | FastAPI   |
                    | tools     |    | HTML/JS   |
                    +-----------+    +-----------+
```

### 모듈 개요

| 모듈 | 설명 |
|------|------|
| `axol.core.types` | 7개 벡터 타입 (`BinaryVec`, `IntVec`, `FloatVec`, `OneHotVec`, `GateVec`, `TransMatrix`) + `StateBundle` |
| `axol.core.operations` | 9개 원시 연산: `transform`, `gate`, `merge`, `distance`, `route`, `step`, `branch`, `clamp`, `map_fn` |
| `axol.core.program` | 실행 엔진: `Program`, `Transition`, `run_program`, `SecurityLevel`, `StepOp`/`BranchOp`/`ClampOp`/`MapOp` |
| `axol.core.verify` | 상태 검증 (exact/cosine/euclidean 매칭) |
| `axol.core.dsl` | DSL 파서: `parse(source) -> Program`, `import`/`use()` 지원 |
| `axol.core.optimizer` | 3-패스 컴파일러 최적화: transform 융합, 데드 상태 제거, 상수 폴딩 |
| `axol.core.backend` | 교체 가능한 배열 백엔드: `numpy` (기본값), `cupy`, `jax` |
| `axol.core.encryption` | 유사 변환 암호화: `encrypt_program`, `decrypt_state` (E/P 인식) |
| `axol.core.analyzer` | 암호화 커버리지 분석기: `analyze(program) -> AnalysisResult`, E/P 분류 |
| `axol.core.module` | 모듈 시스템: `Module`, `ModuleRegistry`, `compose()`, 스키마 검증 |
| `axol.api` | AI 에이전트용 Tool-Use API: `dispatch(request)`, `get_tool_definitions()` |
| `axol.server` | FastAPI 웹 서버 + 바닐라 HTML/JS 비주얼 디버거 프론트엔드 |

---

## 빠른 시작

### 설치

```bash
# 리포지토리 클론
git clone https://github.com/your-username/AXOL.git
cd AXOL

# 의존성 설치
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

print(f"최종 카운트: {result.final_state['count'].to_list()}")  # [5.0]
print(f"스텝 수: {result.steps_executed}")
print(f"종료 조건: {result.terminated_by}")  # terminal_condition
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
print(f"감소 후 HP: {result.final_state['hp'].to_list()}")  # [80.0]
```

---

## DSL 문법

### 프로그램 구조

```
@program_name              # 헤더: 프로그램 이름
s key1=[values] key2=...   # 상태: 초기 벡터 선언
: name=op(args)->out       # 전이: 연산 정의
? terminal condition       # 터미널: 루프 종료 조건 (선택)
```

### 상태 선언

```
s hp=[100]                          # 단일 실수 벡터
s pos=[1.5 2.0 -3.0]               # 다중 요소 벡터
s state=onehot(0,5)                 # 원핫 벡터: 인덱스 0, 크기 5
s buffer=zeros(10)                  # 크기 10 영벡터
s mask=ones(3)                      # 크기 3 전체 1 벡터
s hp=[100] mp=[50] stamina=[75]     # 한 줄에 여러 벡터 선언
```

### 연산

```
# --- 암호화(E) 연산 ---

# transform: 행렬 곱셈
: decay=transform(hp;M=[0.8])
: advance=transform(state;M=[0 1 0;0 0 1;0 0 1])

# gate: 요소별 마스킹
: masked=gate(values;g=mask)

# merge: 벡터 가중합
: total=merge(a b c;w=[1 1 1])->result

# distance: 유사도 측정
: dist=distance(pos1 pos2)
: sim=distance(vec1 vec2;metric=cosine)

# route: argmax 라우팅
: choice=route(scores;R=[1 0 0;0 1 0;0 0 1])

# --- 평문(P) 연산 ---

# step: 임계값 이진 게이트
: mask=step(scores;t=0.5)->gate_out

# branch: 조건부 벡터 선택 (->out_key 필수)
: selected=branch(gate_key;then=high,else=low)->result

# clamp: 값 범위 제한
: safe=clamp(values;min=0,max=100)

# map: 요소별 비선형 함수 (relu, sigmoid, abs, neg, square, sqrt)
: activated=map(x;fn=relu)
: prob=map(logits;fn=sigmoid)->output
```

### 행렬 형식

```
# 밀집: 행은 ;로 구분
M=[0.8]                               # 1x1
M=[1 0;0 1]                           # 2x2 항등행렬
M=[0 1 0;0 0 1;0 0 1]                # 3x3 시프트 행렬

# 희소: 0이 아닌 항목만 표기
M=sparse(100x100;0,1=1 1,2=1 99,99=1) # 100개 항목을 가진 100x100
```

### 터미널 조건

```
? done count>=5              # count[0] >= 5이면 종료
? finished state[2]>=1       # state[2] >= 1이면 종료 (인덱스 접근)
? end hp<=0                  # hp[0] <= 0이면 종료
```

`?` 줄이 없으면 **파이프라인 모드**로 실행됩니다 (모든 전이가 1회 실행).

### 주석

```
# 이것은 주석입니다
@my_program
# 주석은 어디에나 올 수 있습니다
s v=[1 2 3]
: t=transform(v;M=[1 0 0;0 1 0;0 0 1])
```

---

## 컴파일러 최적화

`optimize()`는 3개 패스를 적용하여 프로그램 크기를 줄이고 상수를 사전 계산합니다:

```python
from axol.core import parse, optimize, run_program

program = parse(source)
optimized = optimize(program)   # 융합 + 제거 + 폴딩
result = run_program(optimized)
```

### 패스 1: Transform 융합

동일한 키에 대한 연속 `TransformOp`을 단일 행렬 곱셈으로 융합합니다:

```
# 이전: 2개 전이, 반복당 2회 행렬 곱셈
: t1=transform(v;M=[0 1 0;0 0 1;1 0 0])
: t2=transform(v;M=[2 0 0;0 2 0;0 0 2])

# 이후: 1개 전이, 1회 행렬 곱셈 (M_fused = M1 @ M2)
: t1+t2=transform(v;M_fused)
```

- `CustomOp` 경계를 넘지 않음
- 고정점 반복으로 3개 이상 연쇄 처리
- 2개 transform 파이프라인: **전이 수 -50%, 실행 시간 -45%**

### 패스 2: 데드 상태 제거

어떤 전이에서도 참조되지 않는 초기 상태 벡터를 제거합니다:

```
s used=[1 0]  unused=[99 99]   # unused는 참조되지 않음
: t=transform(used;M=[...])

# 최적화 후: unused가 초기 상태에서 제거됨
```

- `CustomOp`에 대해 보수적 (모든 상태 보존)
- `terminal_key`는 항상 "읽음"으로 처리

### 패스 3: 상수 폴딩

불변 키(쓰기가 없는 키)에 대한 transform을 사전 계산합니다:

```
s constant=[1 0 0]
: t=transform(constant;M=[0 1 0;0 0 1;1 0 0])->result

# 이후: 전이 제거, result=[0,1,0]이 초기 상태에 저장됨
```

---

## GPU 백엔드

`numpy` (기본값), `cupy` (NVIDIA GPU), `jax`를 지원하는 교체 가능한 배열 백엔드:

```python
from axol.core import set_backend, get_backend_name

set_backend("numpy")   # 기본값 - CPU
set_backend("cupy")    # NVIDIA GPU (cupy 설치 필요)
set_backend("jax")     # Google JAX (jax 설치 필요)
```

선택적 백엔드 설치:

```bash
pip install axol[gpu]   # cupy-cuda12x
pip install axol[jax]   # jax[cpu]
```

기존 코드가 투명하게 동작합니다 - 백엔드 전환은 전역적이며 모든 벡터/행렬 연산에 적용됩니다.

---

## 모듈 시스템

스키마, 임포트, 서브모듈 실행을 갖춘 재사용 가능하고 합성 가능한 프로그램.

### 모듈 정의

```python
from axol.core.module import Module, ModuleSchema, VecSchema, ModuleRegistry

schema = ModuleSchema(
    inputs=[VecSchema("atk", "float", 1), VecSchema("def_val", "float", 1)],
    outputs=[VecSchema("dmg", "float", 1)],
)
module = Module(name="damage_calc", program=program, schema=schema)
```

### 레지스트리 & 파일 로딩

```python
registry = ModuleRegistry()
registry.load_from_file("damage_calc.axol")
registry.resolve_import("heal", relative_to="main.axol")
```

### DSL Import & Use 문법

```
@main
import damage_calc from "damage_calc.axol"
s atk=[50] def_val=[10]
: calc=use(damage_calc;in=atk,def_val;out=dmg)
```

### 프로그램 합성

```python
from axol.core.module import compose
combined = compose(program_a, program_b, name="combined")
```

---

## Tool-Use API

AI 에이전트가 Axol 프로그램을 파싱, 실행, 검증할 수 있는 JSON 호출 인터페이스:

```python
from axol.api import dispatch

# 파싱
result = dispatch({"action": "parse", "source": "@prog\ns v=[1]\n: t=transform(v;M=[2])"})
# -> {"program_name": "prog", "state_keys": ["v"], "transition_count": 1, "has_terminal": false}

# 실행
result = dispatch({"action": "run", "source": "...", "optimize": True})
# -> {"final_state": {"v": [2.0]}, "steps_executed": 1, "terminated_by": "pipeline_end"}

# 단계별 검사
result = dispatch({"action": "inspect", "source": "...", "step": 1})

# 연산 목록
result = dispatch({"action": "list_ops"})

# 기대 출력 검증
result = dispatch({"action": "verify", "source": "...", "expected": {"v": [2.0]}})
```

AI 에이전트 도구 정의(JSON Schema)는 `get_tool_definitions()`로 제공됩니다.

---

## 웹 프론트엔드

바닐라 HTML/JS 비주얼 디버거를 포함한 FastAPI 서버:

```bash
pip install axol[server]    # fastapi + uvicorn
python -m axol.server       # http://localhost:8080
```

### 기능

| 패널 | 설명 |
|------|------|
| **DSL 에디터** | 예제 드롭다운이 있는 구문 편집기 |
| **실행** | 실행/최적화 버튼, 결과 요약 (스텝, 시간, terminated_by) |
| **트레이스 뷰어** | prev/next/play 컨트롤이 있는 단계별 상태 테이블 |
| **상태 차트** | Chart.js 시계열 그래프 (X=스텝, Y=벡터 값) |
| **암호화 데모** | 원본 vs 암호화 행렬 히트맵, 암호화/실행/복호화 워크플로우 |
| **성능** | 최적화 전/후 비교, 토큰 비용 분석 |

### API 엔드포인트

```
POST /api/parse       - DSL 소스 파싱
POST /api/run         - 파싱 + 실행 + 전체 트레이스
POST /api/optimize    - 최적화 전/후 비교
POST /api/encrypt     - 프로그램 암호화 + 실행 + 복호화
GET  /api/examples    - 내장 예제 프로그램
GET  /api/ops         - 연산 설명
POST /api/token-cost  - 토큰 수 분석 (Axol vs Python vs C#)
POST /api/module/run  - 서브모듈 포함 프로그램 실행
```

---

## 토큰 비용 비교

`tiktoken` cl100k_base 토크나이저로 측정 (GPT-4 / Claude 사용).

> **참고**: 토큰 절약률은 벡터/행렬 연산에 자연스럽게 매핑되는 프로그램(상태 머신, 선형 변환, 가중합)에서 측정되었습니다. 범용 프로그래밍 작업(문자열 처리, I/O, API 호출)에서는 Axol로 표현할 수 없습니다. 아래 비교는 Axol의 최선의 경우를 나타내며, 평균적인 경우가 아닙니다.

### Python vs Axol DSL

| 프로그램 | Python | Axol DSL | 절약률 |
|---------|--------|----------|--------|
| Counter (0->5) | 32 | 33 | -3.1% |
| State Machine (3-state) | 67 | 47 | 29.9% |
| HP Decay (3 rounds) | 51 | 32 | 37.3% |
| RPG Damage Calc | 130 | 90 | 30.8% |
| 100-State Automaton | 1,034 | 636 | 38.5% |
| **합계** | **1,314** | **838** | **36.2%** |

### Python vs C# vs Axol DSL

| 프로그램 | Python | C# | Axol DSL | vs Python | vs C# |
|---------|--------|----|----------|-----------|-------|
| Counter | 32 | 61 | 33 | -3.1% | 45.9% |
| State Machine | 67 | 147 | 48 | 28.4% | 67.3% |
| HP Decay | 51 | 134 | 51 | 0.0% | 61.9% |
| Combat | 145 | 203 | 66 | 54.5% | 67.5% |
| Data Heavy | 159 | 227 | 67 | 57.9% | 70.5% |
| Pattern Match | 151 | 197 | 49 | 67.5% | 75.1% |
| 100-State Auto | 739 | 869 | 636 | 13.9% | 26.8% |
| **합계** | **1,344** | **1,838** | **950** | **29.3%** | **48.3%** |

### 핵심 발견

1. **단순 프로그램** (counter, hp_decay): DSL이 Python과 비슷합니다. 간단한 프로그램에서는 DSL 구문 오버헤드가 Python의 최소 구문과 대등합니다.
2. **구조화된 프로그램** (combat, data_heavy, pattern_match): DSL이 Python 대비 **50~68%**, C# 대비 **67~75%** 절약합니다. 벡터 표현이 클래스 정의, 제어 흐름, 보일러플레이트를 제거합니다.
3. **대규모 상태 공간** (100+ 상태): 희소 행렬 표기법이 Python 대비 **~38%**, C# 대비 **~27%** 일관된 절약을 제공하며 O(N) 스케일링을 보입니다.

### Tool-Use API vs Python + FHE

**전체 암호화 워크플로우**를 비교할 때 (DSL 문법만이 아닌):

| 작업 | Python + FHE | Axol Tool-Use API | 절약 |
|------|-------------|-------------------|------|
| 암호화 분기 | ~150 토큰 | ~30 토큰 | 80% |
| 암호화 상태 머신 | ~200 토큰 | ~35 토큰 | 82% |
| 암호화 Grover 검색 | ~250 토큰 | ~25 토큰 | 90% |

절약은 **문법이 아닌 추상화**에서 비롯됩니다: LLM은 암호화 코드(키 생성, 암호화, 복호화)를 전혀 보지 않으며, Tool-Use API가 내부적으로 처리합니다.

---

## 런타임 성능

Axol은 NumPy를 연산 백엔드로 사용합니다.

> **참고**: 런타임 벤치마크는 순수 Python 루프와 Axol의 NumPy 백엔드를 비교합니다. 속도 향상은 주로 NumPy의 최적화된 C/Fortran 구현에서 비롯되며, Axol 고유의 최적화가 아닙니다. NumPy를 직접 사용하는 Python 코드도 유사한 속도를 달성합니다.

### 소규모 벡터 (dim < 100)

| 차원 | Python 루프 | Axol (NumPy) | 우위 |
|------|-----------|-------------|------|
| dim=4 | ~6 us | ~11 us | Python 2x |
| dim=100 | ~14 us | ~20 us | Python 1.4x |

소규모 벡터에서는 Python 네이티브 루프가 더 빠릅니다 (NumPy 호출 오버헤드). 예상된 결과이며 수용 가능합니다 - 소규모 프로그램은 어떤 방식이든 빠릅니다.

### 대규모 벡터 (dim >= 1000)

| 차원 | Python 루프 | Axol (NumPy) | 우위 |
|------|-----------|-------------|------|
| dim=1,000 (행렬곱) | ~129 ms | ~0.2 ms | **573x** (NumPy) |
| dim=10,000 (행렬곱) | ~14,815 ms | ~381 ms | **39x** (NumPy) |

대규모 벡터 연산(행렬 곱셈)에서는 NumPy의 최적화된 C/Fortran BLAS 백엔드(Axol이 사용)가 순수 Python 루프보다 **수백 배 빠릅니다**. NumPy를 직접 사용하는 Python 코드도 유사한 속도 향상을 달성합니다.

### 사용 가이드

| 시나리오 | 권장 |
|---------|------|
| AI 에이전트 암호화 연산 | Axol Tool-Use API (LLM이 암호화를 몰라도 됨) |
| 대규모 상태 공간 (100+ 차원) | Axol (NumPy 가속 + 희소 표기법) |
| 클라이언트-서버 암호화 위임 | AxolClient SDK (로컬 암호화, 원격 연산) |
| 가변 차원 상태 전이 | KeyFamily + 직사각 암호화 (N→M) |
| 차원 은닉 프라이버시 | 패딩 암호화 (균일 max_dim) |
| 함수→행렬 컴파일 | `fn_to_matrix` / `truth_table_to_matrix` 컴파일러 |
| 단순 스크립트 (10줄 미만) | Python (오버헤드 적음) |
| 사람이 읽을 비즈니스 로직 | Python/C# (익숙한 문법) |

### 제한 사항

- **제한된 도메인**: Axol은 벡터/행렬 연산만 표현할 수 있습니다. 문자열 처리, I/O, 네트워킹, 범용 프로그래밍은 지원되지 않습니다.
- **LLM 학습 데이터 없음**: Python이나 JavaScript와 달리, 어떤 LLM도 Axol 코드로 학습되지 않았습니다. AI 에이전트가 컨텍스트에 예제 없이 올바른 Axol 프로그램을 생성하는 데 어려움을 겪을 수 있습니다.
- **선형 연산만 암호화**: 9개 연산 중 5개만 암호화 실행을 지원합니다. 비선형 연산(step, branch, clamp, map)을 사용하는 프로그램은 암호화 커버리지가 감소합니다. 단, BranchOp은 컴파일 타임 게이트가 알려진 경우 암호화된 TransformOp으로 컴파일 가능합니다.
- **루프 모드 암호화 오버헤드**: 루프 모드의 암호화된 프로그램은 터미널 조건을 평가할 수 없어 max_iterations까지 실행됩니다. 이로 인해 벤치마크에서 400배 이상의 오버헤드가 발생합니다.
- **토큰 절약은 도메인 특화**: DSL 토큰 절약은 도메인 특화(벡터/행렬 프로그램에서 30-50%)입니다. 하지만 Tool-Use API는 암호화를 완전히 추상화하여 Python+FHE 대비 80-85% 절약을 제공합니다.
- **패딩 오버헤드**: 패딩 암호화는 모든 차원을 max_dim으로 확장하여 연산량이 O(max_dim²/dim²) 증가합니다. 차원 은닉이 필요한 경우에만 사용하세요.

---

## 성능 벤치마크

`pytest tests/test_performance_report.py -v -s`로 자동 생성됩니다. 전체 결과는 [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md)를 참조하세요.

> **참고**: 런타임 벤치마크는 순수 Python 루프와 Axol의 NumPy 백엔드를 비교합니다. 속도 향상은 주로 NumPy의 최적화된 C/Fortran 구현에서 비롯되며, Axol 고유의 최적화가 아닙니다. NumPy를 직접 사용하는 Python 코드도 유사한 속도를 달성합니다.

### 토큰 효율성 (Axol vs Python vs C#)

| 프로그램 | Axol | Python | C# | vs Python | vs C# |
|---------|------|--------|----|-----------|-------|
| Counter (0->5) | 11 | 45 | 78 | **76% 절약** | **86% 절약** |
| 3-State FSM | 14 | 52 | 89 | **73% 절약** | **84% 절약** |
| HP Decay | 14 | 58 | 95 | **76% 절약** | **85% 절약** |
| Combat Pipeline | 14 | 55 | 92 | **75% 절약** | **85% 절약** |
| Matrix Chain | 21 | 60 | 98 | **65% 절약** | **79% 절약** |

평균: Python 대비 **74% 토큰 절약**, C# 대비 **85% 토큰 절약**.

### 차원별 실행 시간

| 차원 | 평균 시간 |
|------|----------|
| 4 | 0.25 ms |
| 100 | 0.17 ms |
| 1,000 | 1.41 ms |

### 최적화 효과

| 프로그램 | 이전 | 이후 | 시간 절감 |
|---------|------|------|----------|
| Pipeline (2 transforms) | 2 transitions | 1 transition | **-45%** |
| Counter (loop) | 2 transitions | 2 transitions | - |
| FSM (loop) | 2 transitions | 2 transitions | - |

Transform 융합은 연속 행렬 연산이 있는 파이프라인 프로그램에서 가장 효과적입니다.

### 암호화 오버헤드

| 프로그램 | 평문 | 암호화 | 오버헤드 |
|---------|------|--------|---------|
| Pipeline (1 pass) | 0.12 ms | 0.12 ms | **~0%** |
| 3-State FSM (loop) | 0.62 ms | 276.8 ms | +44,633% |

파이프라인 모드: 오버헤드 무시 가능. 루프 모드: 암호화된 터미널 조건이 조기 종료를 트리거할 수 없어 `max_iterations`까지 실행되므로 높은 오버헤드.

### 스케일링 (N-상태 오토마톤)

| 상태 수 | 토큰 | 실행 시간 |
|--------|------|----------|
| 5 | 28 | 1.6 ms |
| 20 | 388 | 4.3 ms |
| 50 | 2,458 | 12.9 ms |
| 100 | 9,908 | 27.9 ms |
| 200 | 39,808 | 59.2 ms |

희소 행렬 표기법 덕분에 토큰은 **O(N)**으로 증가합니다 (Python/C#의 O(N^2) 대비). 실행 시간은 행렬 곱셈으로 인해 ~O(N^2)이지만, 200-상태 프로그램에서도 60ms 미만을 유지합니다.

---

## API 레퍼런스

### `parse(source, registry=None, source_path=None) -> Program`

Axol DSL 소스 텍스트를 실행 가능한 `Program` 객체로 파싱합니다.

```python
from axol.core import parse
program = parse("@test\ns v=[1 2 3]\n: t=transform(v;M=[1 0 0;0 1 0;0 0 1])")

# 모듈 레지스트리를 사용한 import/use 지원
from axol.core.module import ModuleRegistry
registry = ModuleRegistry()
program = parse(source, registry=registry, source_path="main.axol")
```

### `run_program(program: Program) -> ExecutionResult`

프로그램을 실행하고 결과를 반환합니다.

```python
from axol.core import run_program
result = run_program(program)
result.final_state     # 최종 벡터 값을 가진 StateBundle
result.steps_executed  # 총 전이 스텝 수
result.terminated_by   # "pipeline_end" | "terminal_condition" | "max_iterations"
result.trace           # 디버깅용 TraceEntry 리스트
result.verification    # expected_state가 설정된 경우 VerifyResult
```

### `optimize(program, *, fuse=True, eliminate_dead=True, fold_constants=True) -> Program`

원본을 변경하지 않고 프로그램을 최적화합니다.

```python
from axol.core import optimize
optimized = optimize(program)                          # 모든 패스
optimized = optimize(program, fold_constants=False)    # 선택적 패스
```

### `set_backend(name) / get_backend() / to_numpy(arr)`

배열 연산 백엔드를 전환합니다.

```python
from axol.core import set_backend, get_backend, to_numpy
set_backend("cupy")     # GPU로 전환
xp = get_backend()      # cupy 모듈 반환
arr = to_numpy(gpu_arr) # numpy로 변환
```

### `dispatch(request) -> dict`

AI 에이전트용 Tool-Use API 진입점.

```python
from axol.api import dispatch
result = dispatch({"action": "run", "source": "...", "optimize": True})
```

### 벡터 타입

| 타입 | 설명 | 팩토리 메서드 |
|------|------|-------------|
| `FloatVec` | 32비트 실수 | `from_list([1.0, 2.0])`, `zeros(n)`, `ones(n)` |
| `IntVec` | 64비트 정수 | `from_list([1, 2])`, `zeros(n)` |
| `BinaryVec` | {0, 1} 원소 | `from_list([0, 1])`, `zeros(n)`, `ones(n)` |
| `OneHotVec` | 정확히 1개의 1.0 | `from_index(idx, n)`, `from_list(...)` |
| `GateVec` | {0.0, 1.0} 원소 | `from_list([1.0, 0.0])`, `zeros(n)`, `ones(n)` |
| `TransMatrix` | M x N float32 행렬 | `from_list(rows)`, `identity(n)`, `zeros(m, n)` |

### 연산 디스크립터

```python
from axol.core.program import (
    # 암호화(E) 연산
    TransformOp,  # TransformOp(key="v", matrix=M, out_key=None)
    GateOp,       # GateOp(key="v", gate_key="g", out_key=None)
    MergeOp,      # MergeOp(keys=["a","b"], weights=w, out_key="out")
    DistanceOp,   # DistanceOp(key_a="a", key_b="b", metric="euclidean")
    RouteOp,      # RouteOp(key="v", router=R, out_key="_route")
    # 평문(P) 연산
    StepOp,       # StepOp(key="v", threshold=0.0, out_key=None)
    BranchOp,     # BranchOp(gate_key="g", then_key="a", else_key="b", out_key="out")
    ClampOp,      # ClampOp(key="v", min_val=-inf, max_val=inf, out_key=None)
    MapOp,        # MapOp(key="v", fn_name="relu", out_key=None)
    # 이스케이프 해치
    CustomOp,     # CustomOp(fn=callable, label="name")  -- security=P
)
```

### 분석기

```python
from axol.core import analyze

result = analyze(program)
result.coverage_pct        # E / total * 100
result.encrypted_count     # E 전이 수
result.plaintext_count     # P 전이 수
result.encryptable_keys    # E 연산만 접근하는 키
result.plaintext_keys      # P 연산이 접근하는 키
print(result.summary())    # 사람이 읽을 수 있는 보고서
```

### 검증

```python
from axol.core import verify_states, VerifySpec

result = verify_states(
    expected=expected_bundle,
    actual=actual_bundle,
    specs={"hp": VerifySpec.exact(tolerance=0.01)},
    strict_keys=False,
)
print(result.passed)    # True/False
print(result.summary()) # 상세 보고서
```

---

## 예제

### 1. 카운터 (0 -> 5)

```
@counter
s count=[0] one=[1]
: increment=merge(count one;w=[1 1])->count
? done count>=5
```

### 2. 상태 머신 (IDLE -> RUNNING -> DONE)

```
@state_machine
s state=onehot(0,3)
: advance=transform(state;M=[0 1 0;0 0 1;0 0 1])
? done state[2]>=1
```

### 3. HP 감소 (100 x 0.8^3 = 51.2)

```
@hp_decay
s hp=[100] round=[0] one=[1]
: decay=transform(hp;M=[0.8])
: tick=merge(round one;w=[1 1])->round
? done round>=3
```

### 4. 전투 데미지 (파이프라인)

```
@combat
s atk=[50] def_val=[20] flag=[1]
: scale=transform(atk;M=[1.5])->scaled
: block=gate(def_val;g=flag)
: combine=merge(scaled def_val;w=[1 -1])->damage
```

### 5. ReLU 활성화 (map)

```
@relu
s x=[-2 0 3 -1 5]
:act=map(x;fn=relu)
# Result: x = [0, 0, 3, 0, 5]
```

### 6. 임계값 선택 (step + branch)

```
@threshold_select
s scores=[0.3 0.8 0.1 0.9] high=[100 200 300 400] low=[1 2 3 4]
:s1=step(scores;t=0.5)->mask
:b1=branch(mask;then=high,else=low)->result
# mask = [0, 1, 0, 1]
# result = [1, 200, 3, 400]
```

### 7. 데미지 파이프라인 (4개 새 연산 전부 사용)

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

### 8. 100-상태 오토마톤 (희소)

```
@auto_100
s s=onehot(0,100)
: step=transform(s;M=sparse(100x100;0,1=1 1,2=1 ... 98,99=1 99,99=1))
? done s[99]>=1
```

---

## 테스트

```bash
# 전체 테스트 (~320개)
pytest tests/ -v

# 코어 테스트
pytest tests/test_types.py tests/test_operations.py tests/test_program.py tests/test_dsl.py -v

# 최적화 테스트 (18개)
pytest tests/test_optimizer.py -v

# 백엔드 테스트 (13개, cupy/jax 미설치 시 스킵)
pytest tests/test_backend.py -v

# Tool-Use API 테스트 (20개)
pytest tests/test_api.py -v

# 모듈 시스템 테스트 (18개)
pytest tests/test_module.py -v

# 암호화 증명 테스트 (21개)
pytest tests/test_encryption.py -v -s

# 새 연산 테스트 - step/branch/clamp/map (44개)
pytest tests/test_new_ops.py -v

# 분석기 테스트 - E/P 커버리지 분석 (7개)
pytest tests/test_analyzer.py -v

# 새 연산 벤치마크 - Python vs C# vs Axol (15개)
pytest tests/test_benchmark_new_ops.py -v -s

# 서버 엔드포인트 테스트 (13개, fastapi 필요)
pytest tests/test_server.py -v

# 성능 보고서 생성 (PERFORMANCE_REPORT.md)
pytest tests/test_performance_report.py -v -s

# 토큰 비용 비교
pytest tests/test_token_cost.py -v -s

# 3개 언어 벤치마크 (Python vs C# vs Axol)
pytest tests/test_benchmark_trilingual.py -v -s

# 웹 프론트엔드 실행
python -m axol.server   # http://localhost:8080
```

현재 테스트 수: **~320개**, 전부 통과 (4개 스킵: cupy/jax 미설치).

---

## Phase 6: Quantum Axol

Phase 6은 Axol에 **양자 간섭**을 도입합니다 — 비선형 로직을 선형 행렬 연산으로 재표현하여 양자 프로그램의 **100% 암호화 커버리지**를 달성합니다. 또한 LLM이 암호화 지식 없이 사용할 수 있는 **암호화 투명 Tool-Use API**를 도입합니다.

### 배경 이론

#### 핵심 문제

Axol의 암호화는 **유사변환(similarity transformation)** 기반입니다: `M' = K⁻¹MK`. 이것은 선형 연산(`transform`, `gate`, `merge`, `distance`, `route`)에는 완벽하게 작동하지만, 비선형 연산(`step`, `branch`, `clamp`, `map`)에는 실패합니다. 비선형 함수는 선형 키 변환과 교환 불가능하기 때문입니다.

이것이 근본적 트레이드오프를 만듭니다:

| 프로그램 유형 | 암호화 커버리지 | 표현력 |
|-------------|---------------|-------|
| 선형 연산만 (E) | 100% | 선형 대수학만 |
| 혼합 E+P | 30-70% | 완전 (비선형 포함) |
| **양자 연산 (Phase 6)** | **100%** | **Grover 수준 검색, 양자 워크** |

#### 해결책: 양자 간섭

핵심 통찰: **양자 알고리즘은 순수 선형 연산만으로 비선형적 행동을 구현합니다**. 예를 들어 Grover 검색은 조건 분기 없이 O(√N) 시간에 표시된 항목을 찾습니다 — 행렬 곱셈만 사용하여:

1. **Hadamard** (H): 음수 진폭을 포함한 균등 중첩 생성
2. **Oracle** (O): 표시된 항목의 부호를 뒤집는 대각 행렬 (-1 항목)
3. **Diffusion** (D): 평균 주위로 상태를 반사 (2|s⟩⟨s| - I)

세 행렬 모두 **실수 직교 행렬**입니다 → `state @ O @ D`로 합성하면 단순 행렬 곱셈 체인이 됩니다 — Axol의 `TransformOp` (E-class)와 완벽 호환됩니다.

#### 부호 있는 진폭이면 충분한 이유

양자 컴퓨팅은 일반적으로 **복소수** 진폭(a + bi)을 사용합니다. 그러나 Grover 검색과 양자 워크를 포함한 다수의 유용한 양자 알고리즘은 **부호 있는 실수** 진폭만 필요합니다. `FloatVec`은 이미 음수 float32를 지원하므로, 양자 간섭 활성화 비용은 사실상 0입니다:

| 티어 | 진폭 유형 | 간섭 수준 | 구현 비용 | 알고리즘 |
|------|----------|----------|----------|---------|
| 0 (Phase 6 이전) | 비음수 실수 | 없음 | — | 고전적 FSM |
| **1 (Phase 6)** | **부호 있는 실수** | **Grover급** | **~0** | **Grover 검색, 양자 워크** |
| 2 (미래) | 복소수 (a+bi) | 완전 위상 | 메모리 2배, 연산 4배 | Shor, QPE, QFT |

#### 수학적 검증: N=4 Grover

균등 중첩 `|s⟩ = [0.5, 0.5, 0.5, 0.5]`에서 시작, 타겟 인덱스 3:

```
단계 1 — Oracle (인덱스 3 표시):
  O = diag(1, 1, 1, -1)
  state = [0.5, 0.5, 0.5, -0.5]    ← 부호 뒤집기로 간섭 생성

단계 2 — Diffusion (평균 주위 반사):
  D = 2|s⟩⟨s| - I
  state = [0, 0, 0, 1.0]           ← 타겟에 보강 간섭

결과: 정확히 1회 반복으로 타겟 발견, 확률 |1.0|² = 100%.
```

N=4에서는 단일 Oracle+Diffusion 반복으로 **완벽한** 판별을 달성합니다. 더 큰 N에서의 최적 반복 횟수는 ⌊π/4 · √N⌋입니다.

### 아키텍처

#### 새 컴포넌트

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
       └───>│     기존 TransformOp        │<───┘
            │     (E-class, 변경 없음)     │
            │     ↓ 옵티마이저 융합        │
            │     ↓ 암호화 호환           │
            └─────────────────────────────┘
```

**핵심 설계 결정**: Hadamard, Oracle, Diffusion은 새로운 연산 유형이 **아닙니다**. 이들은 `TransMatrix` 객체를 생성하는 편의 함수이며, 기존 `TransformOp` (E-class)으로 사용됩니다. 이것은:

- 기존 **옵티마이저**가 연속 양자 연산을 자동 융합합니다 (예: Oracle @ Diffusion → 단일 행렬)
- 기존 **암호화 모듈**이 유사변환으로 양자 연산을 자동 암호화합니다
- 기존 **분석기**가 순수 양자 프로그램에 대해 100% 커버리지를 정확히 보고합니다

`measure`만이 진정한 새 연산(P-class)입니다. Born 규칙 `p_i = |α_i|²`은 비선형이기 때문입니다. 그러나 `measure`는 **맨 마지막에 한 번만** 적용됩니다 — 모든 중간 연산은 완전 암호화 상태로 실행됩니다.

#### 양자 프로그램의 암호화 파이프라인

```
클라이언트 측:                    서버 측 (암호화):

  [0.5, 0.5, 0.5, 0.5]        [암호화된 상태]
         │                            │
    encrypt(state, K)          state @ O' @ D' @ O' @ D' ...
         │                            │
         └──────────────>       [암호화된 결과]
                                      │
    decrypt(result, K)  <─────────────┘
         │
  [0, 0, 0, 1.0]              모든 O', D'는 E-class!
         │
    measure() ← 클라이언트 측 (P-class, 서버에 전송 안 됨)
         │
  [0, 0, 0, 1.0] → 정답 = 인덱스 3
```

### 예상 성능

#### 암호화 커버리지

| 프로그램 | Phase 6 이전 | Phase 6 이후 | 변화 |
|---------|------------|------------|------|
| 순수 선형 (transform만) | 100% | 100% | — |
| 혼합 (transform + branch + map) | 30-70% | 30-70% | — |
| **양자 검색 (oracle + diffuse)** | **N/A** | **100%** | **신규** |
| **양자 + 측정** | **N/A** | **67-100%** | **신규** |

#### Grover 검색 복잡도

| 검색 공간 (N) | 고전적 (선형 탐색) | Grover (Axol) | 속도 향상 |
|-------------|-----------------|--------------|----------|
| 4 | 4회 비교 | 1회 반복 (행렬 곱 2회) | 2배 |
| 16 | 16회 비교 | 3회 반복 (행렬 곱 6회) | 2.7배 |
| 64 | 64회 비교 | 6회 반복 (행렬 곱 12회) | 5.3배 |
| 256 | 256회 비교 | 12회 반복 (행렬 곱 24회) | 10.7배 |
| 1024 | 1024회 비교 | 25회 반복 (행렬 곱 50회) | 20.5배 |
| N | O(N) | O(√N) | O(√N) |

각 "반복"은 2번의 행렬 곱(Oracle + Diffusion)이며, 둘 다 E-class입니다.

#### Tool-Use API 토큰 효율

암호화 투명 API는 LLM 관점에서 모든 암호화 보일러플레이트를 제거합니다:

| 작업 | Python + FHE | Axol Tool-Use API | 토큰 절약 |
|------|-------------|-------------------|----------|
| 암호화 분기 | ~150 토큰 | ~30 토큰 | **80%** |
| 암호화 상태 머신 | ~200 토큰 | ~35 토큰 | **82%** |
| 암호화 Grover 검색 | ~250 토큰 | ~25 토큰 | **90%** |
| 암호화 양자 워크 | ~300 토큰 | ~30 토큰 | **90%** |

**왜 이렇게 큰 절약인가**: Python+FHE에서 LLM은 키 생성, 암호화, 회로 컴파일, 암호화 실행, 복호화 코드를 모두 생성해야 합니다. Axol의 Tool-Use API에서는 다음만 보내면 됩니다:

```json
{"action": "quantum_search", "n": 256, "marked": [42], "encrypt": true}
```

API가 키 생성, 프로그램 구성, 암호화, 실행, 복호화, 측정을 내부적으로 처리합니다.

### 차별점

#### Axol vs. 기존 접근법

| 속성 | Python (일반) | Python + FHE | Python + TEE | Axol + Quantum |
|------|-------------|-------------|-------------|---------------|
| **암호화 범위** | 없음 | 100% (모든 연산) | 100% (하드웨어) | 100% (양자 연산) / 30-70% (혼합) |
| **성능 오버헤드** | — | 1,000-10,000배 | ~0% | ~0% (파이프라인 모드) |
| **하드웨어 필요** | 없음 | 없음 | SGX/TrustZone 엔클레이브 | 없음 |
| **LLM 암호화 지식 필요** | — | 예 (compile, keygen, encrypt, decrypt) | 아니오 (인프라 레벨) | **아니오 (API가 처리)** |
| **LLM 토큰 비용** | ~70 토큰 | ~200 토큰 | ~70 토큰 + 인프라 | **~25-30 토큰** |
| **소프트웨어 전용** | 예 | 예 | 아니오 | **예** |

**Axol의 고유 포지션**: FHE의 소프트웨어 수준 암호화 + TEE의 투명성(LLM이 암호화 존재를 모름) + FHE와 TEE 모두 제공하지 않는 Tool-Use API 효율성을 결합합니다.

#### 왜 그냥 FHE를 쓰지 않는가?

완전 동형 암호화(FHE)는 암호화된 데이터에 대해 **모든** 연산을 지원합니다 — Axol보다 엄밀하게 더 강력한 모델입니다. 그러나:

1. **성능**: FHE는 1,000-10,000배 오버헤드를 발생시킵니다. Axol의 유사변환은 선형 연산에서 ~0% 오버헤드입니다.
2. **LLM 복잡도**: FHE는 LLM이 컴파일, 키 생성, 암호화 코드를 생성해야 합니다 (~200 토큰). Axol의 API는 ~25 토큰입니다.
3. **실용 범위**: 많은 AI 에이전트 작업(상태 머신, 검색, 라우팅, 점수화)은 본질적으로 선형입니다. 양자 간섭이 이를 검색 문제까지 확장합니다. 나머지 비선형 케이스(활성화 함수, 클램핑)는 클라이언트 측 후처리로 격리할 수 있습니다.

#### 왜 그냥 TEE를 쓰지 않는가?

신뢰 실행 환경(Intel SGX, ARM TrustZone)은 제로 성능 오버헤드로 하드웨어 수준 암호화를 제공합니다. 그러나:

1. **하드웨어 의존성**: TEE는 특정 CPU 기능을 요구합니다. Axol은 NumPy가 있는 모든 머신에서 실행됩니다.
2. **공급망 신뢰**: TEE 보안은 하드웨어 벤더에 대한 신뢰에 의존합니다. Axol의 보안은 순수 수학적입니다.
3. **세분성**: TEE는 전부 아니면 전무(전체 엔클레이브가 보호됨)입니다. Axol의 분석기는 어떤 연산이 암호화되는지 정확히 보여주어, 정보에 기반한 트레이드오프 결정을 가능하게 합니다.

### DSL 예시

#### Grover 검색 (평문)

```
@grover_4
s state=[0.5 0.5 0.5 0.5]
: oracle=oracle(state;marked=[3];n=4)
: diffuse=diffuse(state;n=4)
? found state[3]>=0.9
```

결과: 타겟 인덱스 3을 1회 반복에서 100% 확률로 발견.

#### Grover 검색 (암호화 파이프라인)

```
@grover_encrypted
s state=[0.5 0.5 0.5 0.5]
: oracle=oracle(state;marked=[3];n=4)
: diffuse=diffuse(state;n=4)
```

종료 조건 없음 — 파이프라인 모드로 모든 연산이 E-class 보장.
클라이언트가 복호화 후 로컬에서 `measure()` 적용.

#### Tool-Use API (암호화 지식 불필요)

```json
{"action": "quantum_search", "n": 256, "marked": [42], "encrypt": true}
→ {"found_index": 42, "probability": 0.996, "iterations": 12, "encrypted": true}

{"action": "encrypted_run",
 "source": "@prog\ns state=[0.5 0.5 0.5 0.5]\n: o=oracle(state;marked=[3];n=4)\n: d=diffuse(state;n=4)",
 "dim": 4}
→ {"final_state": {"state": [0.0, 0.0, 0.0, 1.0]}, "encrypted": true}
```

### 테스트 커버리지

`tests/test_quantum.py`에 37개 테스트, 카테고리별 정리:

| 카테고리 | 테스트 수 | 검증 내용 |
|---------|---------|----------|
| 단위: Hadamard | 4 | 직교성 (H@H^T=I), 음수 원소, 2의 거듭제곱 검증 |
| 단위: Oracle | 3 | 올바른 부호 뒤집기, 다중 표시 인덱스, 빈 경우 단위행렬 |
| 단위: Diffusion | 2 | 직교성 (D@D^T=I), 음수 원소 |
| 단위: Measure | 5 | Born 규칙, 음수 진폭 불변성, 정규화, 영벡터 |
| 통합: Grover | 5 | N=4 (1회), N=8 (2회), 암호화 파이프라인, 종료 경고, 양자 워크 |
| 분석기 | 2 | 순수 양자 100% 커버리지, measure의 P-class |
| DSL 파싱 | 8 | measure, hadamard, oracle, diffuse 파싱 + 오류 케이스 |
| 옵티마이저 | 2 | Oracle+Diffuse 융합, 융합 정확성 |
| API | 6 | encrypted_run, quantum_search (평문/암호화/N=8), 오류 처리 |

```bash
# 모든 양자 테스트 실행
pytest tests/test_quantum.py -v -s

# 특정 카테고리 실행
pytest tests/test_quantum.py::TestGrover -v -s
pytest tests/test_quantum.py::TestQuantumAnalyzer -v -s
pytest tests/test_quantum.py::TestAPI -v -s
```

### 티어 로드맵

| 티어 | 상태 | 진폭 | 알고리즘 | 암호화 |
|------|------|------|---------|-------|
| 0 | Phase 1-5 | 비음수 실수 | 고전적 FSM, 라우팅 | 30-100% (혼합 E/P) |
| 1 | Phase 6 | 부호 있는 실수 | Grover 검색, 양자 워크 | 100% (E-class) |
| **2** | **Phase 8 (현재)** | **카오스 이론 기반** | **Declare->Weave->Observe, 리아푸노프/프랙탈** | **Omega/Phi 품질 척도** |
| 3 | 미래 | 복소수 (a+bi) | Shor, QPE, QFT | 100% (복소 유니터리) |

---

## 로드맵

- [x] Phase 1: 타입 시스템 (7개 벡터 타입 + StateBundle)
- [x] Phase 1: 5개 원시 연산
- [x] Phase 1: 프로그램 실행 엔진 (파이프라인 + 루프 모드)
- [x] Phase 1: 상태 검증 프레임워크
- [x] Phase 2: DSL 파서 (완전한 문법 지원)
- [x] Phase 2: 희소 행렬 표기법
- [x] Phase 2: 토큰 비용 벤치마크 (Python, C#, Axol)
- [x] Phase 2: 행렬 암호화 증명 (5개 연산 모두 검증, 21개 테스트)
- [x] Phase 3: 컴파일러 최적화 (transform 융합, 데드 상태 제거, 상수 폴딩)
- [x] Phase 3: GPU 백엔드 (numpy/cupy/jax 교체 가능)
- [x] Phase 4: AI 에이전트용 Tool-Use API (parse/run/inspect/verify/list_ops)
- [x] Phase 4: 암호화 모듈 (encrypt_program, decrypt_state)
- [x] Phase 5: 모듈 시스템 (레지스트리, import/use DSL, compose, 스키마 검증)
- [x] 프론트엔드: FastAPI + 바닐라 HTML/JS 비주얼 디버거 (트레이스 뷰어, 상태 차트, 암호화 데모)
- [x] 성능 벤치마크 (토큰 비용, 런타임 스케일링, 최적화 효과, 암호화 오버헤드)
- [x] Phase 6: 양자 간섭 (부호 있는 진폭, Hadamard/Oracle/Diffusion 행렬, 측정 연산)
- [x] Phase 6: 양자 프로그램 100% 암호화 커버리지 (최종 측정을 제외한 모든 연산이 E-class)
- [x] Phase 6: 암호화 투명 Tool-Use API (encrypted_run, quantum_search — LLM이 암호화 지식 불필요)
- [x] Phase 7: KeyFamily — 단일 시드에서 다차원 키 결정적 파생
- [x] Phase 7: 직사각 행렬 암호화 (KeyFamily를 통한 N→M 차원 변환)
- [x] Phase 7: 함수→행렬 컴파일러 (fn_to_matrix, truth_table_to_matrix)
- [x] Phase 7: 패딩 레이어 — 차원 은닉 이중 암호화 (균일 max_dim)
- [x] Phase 7: 분기→변환 컴파일 (BranchOp → 암호화된 대각 TransformOp)
- [x] Phase 7: AxolClient SDK — 클라이언트 암호화, 서버 연산 아키텍처
- [x] Phase 8: 카오스 이론 양자 모듈 (`axol/quantum/`) — Declare -> Weave -> Observe 파이프라인
- [x] Phase 8: 리아푸노프 지수 추정 (Benettin QR법) + Omega = 1/(1+max(lambda,0))
- [x] Phase 8: 프랙탈 차원 추정 (box-counting/correlation) + Phi = 1/(1+D/D_max)
- [x] Phase 8: 직조기(weaver) — 선언으로부터 끌개(attractor) 기반 Tapestry 구축
- [x] Phase 8: 관측소(observatory) — 단일/반복 관측으로 결과 붕괴 + 품질 향상
- [x] Phase 8: 합성 규칙 (직렬: lambda 합산, 병렬: min/max 규칙)
- [x] Phase 8: 얽힘 비용 산출 + 달성 불가 감지
- [x] Phase 8: 양자 DSL 파서 (entangle/observe/reobserve/if 블록)
- [x] Phase 8: 101개 신규 테스트 (전체 545 passed, 0 failed)

---

## Phase 8: 카오스 이론 양자 모듈

Phase 8은 AXOL의 이론적 토대(THEORY.md)를 **카오스 이론**으로 형식화하고, **Declare -> Weave -> Observe** 파이프라인을 실행 가능한 코드로 구현합니다. 기존 `axol/core` 엔진을 변경 없이 재활용하며, `axol/quantum/` 패키지로 독립 구현됩니다.

### 핵심 매핑

| AXOL 개념 | 카오스 이론 대응 | 수식 |
|-----------|-----------------|------|
| Tapestry (직물) | 이상한 끌개 (Strange Attractor) | 위상 공간의 컴팩트 불변 집합 |
| Omega (결속도) | 리아푸노프 안정성 | `1/(1+max(lambda,0))` |
| Phi (선명도) | 프랙탈 차원 역수 | `1/(1+D/D_max)` |
| Weave (직조) | 끌개 구조 구축 | 반복 사상의 궤적 행렬 |
| Observe (관측) | 끌개 위 점 붕괴 | 시간 복잡도 O(D) |
| 얽힘 범위 | 끌개 분지 (Basin of Attraction) | 수렴 영역의 경계 |

### 파이프라인

```
[Declare]                    [Weave]                       [Observe]
관계 선언 + 품질 목표    ->   끌개 구조 구축 + 비용 산출  ->   입력 -> 즉각 붕괴
entangle search(q, db)        weave(declaration)              observe(tapestry, inputs)
  @ Omega(0.9) Phi(0.7)        -> Tapestry                     -> Observation
  { relevance <~> ... }        + WeaverReport                   + Omega, Phi
```

### 품질 척도

```
        Phi (선명도)
        ^
   1.0  |  날카롭지만 불안정    이상적 (강한 얽힘)
        |
   0.0  |  노이즈              안정적이지만 흐릿함
        +-------------------------> Omega (결속도)
       0.0                        1.0
```

### 합성 규칙

| 합성 | lambda | Omega | D | Phi |
|------|--------|-------|---|-----|
| 직렬 | lambda_A + lambda_B | 1/(1+max(sum,0)) | D_A + D_B | Phi_A * Phi_B |
| 병렬 | max(lambda_A, lambda_B) | min(Omega_A, Omega_B) | max(D_A, D_B) | min(Phi_A, Phi_B) |

### DSL 문법

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

### 사용 예시

```python
from axol.quantum import DeclarationBuilder, RelationKind, weave, observe
from axol.core.types import FloatVec

# 선언
decl = (
    DeclarationBuilder("search")
    .input("query", 64)
    .input("db", 64)
    .relate("relevance", ["query", "db"], RelationKind.PROPORTIONAL)
    .output("relevance")
    .quality(0.9, 0.7)
    .build()
)

# 직조
tapestry = weave(decl, seed=42)
print(f"Omega: {tapestry.weaver_report.estimated_omega:.2f}")
print(f"Phi: {tapestry.weaver_report.estimated_phi:.2f}")

# 관측
result = observe(tapestry, {"query": FloatVec.zeros(64), "db": FloatVec.zeros(64)})
print(f"Result Omega: {result.omega:.2f}, Phi: {result.phi:.2f}")
```

### 테스트

```bash
# 신규 양자 모듈 테스트만 실행
pytest tests/test_quantum_*.py tests/test_lyapunov.py tests/test_fractal.py tests/test_compose.py -v

# 전체 테스트 (기존 + 신규)
pytest tests/ -v
# 545 passed, 0 failed, 4 skipped
```

---

## 클라이언트-서버 아키텍처

Phase 7에서는 클라이언트에서 암호화하고 신뢰할 수 없는 서버에서 연산하는 분리 아키텍처를 도입합니다:

```
┌─────────────────┐         ┌─────────────────────┐
│  클라이언트 (키) │         │  서버 (키 없음)      │
│                 │         │                      │
│  프로그램 ──────►│ 암호화  │  암호화된 프로그램    │
│  fn_to_matrix() │────────►│  run_program()       │
│  pad_and_encrypt│         │  (노이즈에 대해 연산) │
│                 │◄────────│  암호화된 결과        │
│  decrypt_result │ 복호화  │                      │
│  ──────► 결과   │         │                      │
└─────────────────┘         └─────────────────────┘
```

### 핵심 구성 요소

| 구성 요소 | 설명 |
|----------|------|
| `KeyFamily(seed)` | 단일 시드에서 모든 차원의 직교 키 파생 |
| `fn_to_matrix(fn, N, M)` | Python 함수를 변환 행렬로 컴파일 |
| `encrypt_matrix_rect(M, kf)` | N×M 직사각 행렬 암호화 |
| `pad_and_encrypt(prog, kf, max_dim)` | 모든 차원을 max_dim으로 패딩 후 암호화 |
| `AxolClient(seed, max_dim)` | 상위 SDK: prepare → 전송 → decrypt |

### 사용법

```python
from axol.api.client import AxolClient
from axol.core.compiler import fn_to_matrix

# 함수를 행렬로 컴파일
M = fn_to_matrix(lambda x: (x + 1) % 4, 4, 4)

# 빌드 및 암호화
client = AxolClient(seed=42, max_dim=8, use_padding=True)
result = client.run_local(program)  # 암호화 → 실행 → 복호화
```

### 보안 속성

- **차원 은닉**: 패딩을 사용하면 서버가 원본 벡터 차원을 파악할 수 없습니다.
- **키 격리**: 각 차원은 고유한 파생 키를 가져 하나가 유출되어도 다른 키는 안전합니다.
- **분기 컴파일**: 컴파일 타임 게이트를 가진 BranchOp은 암호화된 변환으로 변환되어 E-class 커버리지가 증가합니다.
- **투명한 I/O**: 클라이언트가 모든 암호화/복호화를 처리하며, 서버는 노이즈에 대한 선형대수만 실행합니다.

---

## 라이선스

MIT License. 자세한 내용은 [LICENSE](LICENSE)를 참조하세요.
