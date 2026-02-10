<p align="center">
  <h1 align="center">AXOL</h1>
  <p align="center">
    <strong>AI 네이티브 벡터 프로그래밍 언어</strong>
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
- **5개 원시 연산**으로 모든 계산 표현: `transform`, `gate`, `merge`, `distance`, `route`
- **희소 행렬 표기법**: 밀집 표현의 O(N^2) 대비 O(N)으로 스케일링
- 완전한 상태 추적이 가능한 **결정론적 실행**
- **NumPy 백엔드**로 대규모 벡터 연산 500배 이상 가속

---

## 목차

- [이론적 배경](#이론적-배경)
- [아키텍처](#아키텍처)
- [빠른 시작](#빠른-시작)
- [DSL 문법](#dsl-문법)
- [토큰 비용 비교](#토큰-비용-비교)
- [런타임 성능](#런타임-성능)
- [API 레퍼런스](#api-레퍼런스)
- [예제](#예제)
- [테스트](#테스트)
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

### 다섯 가지 원시 연산

Axol은 모든 연산을 다섯 가지 연산으로 환원합니다. 각각은 기본적인 선형대수 개념에 대응합니다:

| 연산 | 수학적 기반 | 설명 |
|------|-----------|------|
| `transform` | 행렬 곱: `v @ M` | 선형 상태 변환 |
| `gate` | 아다마르 곱: `v * g` | 조건부 마스킹 |
| `merge` | 가중합: `sum(v_i * w_i)` | 벡터 결합 |
| `distance` | L2 / 코사인 / 내적 | 유사도 측정 |
| `route` | `argmax(v @ R)` | 이산 분기 |

이 다섯 연산으로 다음을 표현할 수 있습니다:
- 상태 머신 (transform)
- 조건 로직 (gate)
- 누적/집계 (merge)
- 유사도 검색 (distance)
- 의사 결정 (route)

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

## 아키텍처

```
                    +-----------+
  .axol 소스 ------>|  파서     |----> Program 객체
                    | (dsl.py)  |         |
                    +-----------+         |
                                          v
                    +-----------+    +-----------+
                    |  검증기   |<---|  실행엔진  |
                    |(verify.py)|    |(program.py)|
                    +-----------+    +-----------+
                                          |
                         사용             |
                    +-----------+         |
                    | 연산 모듈  |<--------+
                    | (ops.py)  |
                    +-----------+
                         |
                    +-----------+
                    | 타입 시스템 |
                    |(types.py) |
                    +-----------+
```

### 모듈 개요

| 모듈 | 설명 |
|------|------|
| `axol.core.types` | 7개 벡터 타입 + `StateBundle` |
| `axol.core.operations` | 5개 원시 연산: `transform`, `gate`, `merge`, `distance`, `route` |
| `axol.core.program` | 실행 엔진: `Program`, `Transition`, `run_program` |
| `axol.core.verify` | 상태 검증 (exact/cosine/euclidean 매칭) |
| `axol.core.dsl` | DSL 파서: `parse(source) -> Program` |

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
```

### 행렬 형식

```
# 밀집: 행은 ;로 구분
M=[0.8]                               # 1x1
M=[1 0;0 1]                           # 2x2 항등행렬
M=[0 1 0;0 0 1;0 0 1]                # 3x3 시프트 행렬

# 희소: 0이 아닌 항목만 표기
M=sparse(100x100;0,1=1 1,2=1 99,99=1)
```

### 터미널 조건

```
? done count>=5              # count[0] >= 5이면 종료
? finished state[2]>=1       # state[2] >= 1이면 종료 (인덱스 접근)
? end hp<=0                  # hp[0] <= 0이면 종료
```

`?` 줄이 없으면 **파이프라인 모드**로 실행됩니다 (모든 전이가 1회 실행).

---

## 토큰 비용 비교

`tiktoken` cl100k_base 토크나이저로 측정 (GPT-4 / Claude 사용).

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

---

## 런타임 성능

Axol은 NumPy를 연산 백엔드로 사용합니다.

### 소규모 벡터 (dim < 100)

| 차원 | Python 루프 | Axol (NumPy) | 우위 |
|------|-----------|-------------|------|
| dim=4 | ~6 us | ~11 us | Python 2x |
| dim=100 | ~14 us | ~20 us | Python 1.4x |

소규모 벡터에서는 Python 네이티브 루프가 더 빠릅니다 (NumPy 호출 오버헤드).

### 대규모 벡터 (dim >= 1000)

| 차원 | Python 루프 | Axol (NumPy) | 우위 |
|------|-----------|-------------|------|
| dim=1,000 (행렬곱) | ~129 ms | ~0.2 ms | **Axol 573x** |
| dim=10,000 (행렬곱) | ~14,815 ms | ~381 ms | **Axol 39x** |

대규모 벡터 연산(행렬 곱셈)에서는 Axol의 NumPy 백엔드가 순수 Python 루프보다 **수백 배 빠릅니다**.

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

---

## 테스트

```bash
# 전체 테스트 (149개)
pytest tests/ -v

# DSL 파서 테스트
pytest tests/test_dsl.py -v

# 토큰 비용 비교
pytest tests/test_token_cost.py -v -s

# 3개 언어 벤치마크 (Python vs C# vs Axol)
pytest tests/test_benchmark_trilingual.py -v -s
```

현재 테스트 수: **149개**, 전부 통과.

---

## 로드맵

- [x] Phase 1: 타입 시스템 (7개 벡터 타입 + StateBundle)
- [x] Phase 1: 5개 원시 연산
- [x] Phase 1: 프로그램 실행 엔진 (파이프라인 + 루프 모드)
- [x] Phase 1: 상태 검증 프레임워크
- [x] Phase 2: DSL 파서 (완전한 문법 지원)
- [x] Phase 2: 희소 행렬 표기법
- [x] Phase 2: 토큰 비용 벤치마크 (Python, C#, Axol)
- [ ] Phase 3: 컴파일러 최적화 (연산 융합, 데드 상태 제거)
- [ ] Phase 3: GPU 백엔드 (CuPy / JAX)
- [ ] Phase 4: AI 에이전트 통합 (tool-use API)
- [ ] Phase 4: 상태 추적 시각 디버거
- [ ] Phase 5: 멀티 프로그램 합성 및 모듈 시스템

---

## 라이선스

MIT License. 자세한 내용은 [LICENSE](LICENSE)를 참조하세요.
