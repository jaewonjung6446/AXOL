# AXOL

**붕괴 기반 프로그래밍 언어**

AXOL은 **관측이 곧 계산**인 프로그래밍 언어다. 답을 계산하지 않는다 — 가능성의 구조를 짜고(weave), 그것을 관측한다(observe). 얼마나 알 것인지, 얼마나 잃을 것인지를 선택한다.

```
declare "mood" {
    input text(8)
    input context(8)
    relate sentiment <- text, context via <~>
    output sentiment
    quality omega=0.85 phi=0.8
}

weave mood quantum=true seed=42

# 모든 가능성을 본다 (비용: 없음)
gaze mood { text = [...] context = [...] }

# 하나의 답을 본다 (비용: 나머지 전부)
observe mood { text = [...] context = [...] }
```

**관측 비용: O(1). 항상. 모델 크기나 데이터와 무관.**

---

## AXOL이 존재하는 이유

지금까지 만들어진 모든 프로그래밍 언어는 같은 패러다임을 따른다:

```
입력 → 계산 → 출력
```

프로그램은 데이터를 받아서 일련의 연산을 거쳐 결과를 낸다. C든, Python이든, Haskell이든, 신경망이든 — 답을 얻는 비용은 계산의 복잡도에 비례한다.

AXOL은 이것을 뒤집는다:

```
구조 짜기 (1회) → 관측 (O(1), 원하는 만큼)
```

계산은 구조를 **짤 때(weave)** 일어난다. 그 이후의 모든 관측은 — 몇 번이든 — 같은 비용: **~5 마이크로초**.

이것은 최적화가 아니다. 양자 측정 이론과 카오스 역학에 기반한 **다른 계산 모델**이다.

---

## 기존 언어가 할 수 없고 AXOL만 할 수 있는 세 가지

### 1. 얼마나 알지 선택할 수 있다

```
gaze x           # C=0: 모든 가능성을 봄, 아무것도 잃지 않음
glimpse x 0.3    # C=0.3: 부분적으로 봄, 부분적으로 잃음
glimpse x 0.7    # C=0.7: 거의 확실, 대부분의 가능성 소멸
observe model {}  # C=1: 하나의 답, 나머지 전부 파괴
```

다른 모든 언어에서 함수는 실행하거나 실행하지 않거나 둘 중 하나다. AXOL은 완전한 불확실성에서 완전한 확실성까지 **연속적인 스펙트럼**을 제공한다. 매개변수 C(붕괴 수준)가 지식과 가능성의 교환비를 제어한다.

### 2. 아는 것의 대가를 추적한다

```
wave planet = scanner { reading = [...] }
gaze planet                    # 모든 가능성이 살아있음
focus planet 0.5               # 부분 붕괴
widen planet 0.3               # 되돌리기 시도 — 완전히 복구 불가
gaze planet                    # 분포가 영구적으로 변했음
```

관측은 **비가역적**이다. 가능성을 한번 붕괴시키면 완전히 복원할 수 없다. AXOL은 이것을 **negativity** — 관계에 남은 열림의 양을 측정하는 수치 — 로 명시적으로 추적한다.

**아는 것이 시스템을 바꾼다는 사실**을 모델링하는 언어는 AXOL 외에 없다.

### 3. 간섭으로 계산한다

```
rel agreement = wave_a <-> wave_b via <~>   # 보강 간섭: 공통점 강화
rel conflict  = wave_a <-> wave_b via <!>   # 상쇄 간섭: 차이점 부각
rel combined  = wave_a <-> wave_b via <+>   # 가산 간섭: 정보 축적
```

AXOL은 숫자로 계산하지 않는다. **파동 사이의 간섭 패턴**으로 계산한다. 두 파동이 만날 때:
- **보강 (`<~>`)**: 공유하는 것이 강해진다
- **상쇄 (`<!>`)**: 다른 것이 증폭되고, 같은 것이 소멸한다
- **곱셈 (`<*>`)**: 둘 다 가진 것만 살아남는다
- **가산 (`<+>`)**: 정보가 축적된다
- **조건 (`<?>`)**: 한 파동이 다른 파동의 위상을 회전시킨다

이것은 비유가 아니다. Born rule 양자역학을 따르는 복소 진폭 연산이다.

---

## 왜 새로운 패러다임인가

### 기존 패러다임과의 비교

| | 명령형 (C, Python) | 함수형 (Haskell) | 신경망 (PyTorch) | **AXOL** |
|---|---|---|---|---|
| **기본 단위** | 변수와 명령 | 함수와 값 | 텐서와 연산 | **파동(Wave)과 관측** |
| **계산 방식** | 순차 실행 | 함수 합성 | 역전파 | **간섭 + 붕괴** |
| **답을 얻는 비용** | O(연산 수) | O(연산 수) | O(파라미터 수) | **O(1)** |
| **불확실성** | 없음 (확정적) | 없음 (확정적) | 확률 분포 (해석) | **기본 타입 (Wave)** |
| **관측의 효과** | 없음 | 없음 | 없음 | **시스템을 변화시킴** |
| **부분적 앎** | 불가 | 불가 | 불가 | **C=0~1 연속 스펙트럼** |
| **관계** | 참조/포인터 | 함수 | 가중치 행렬 | **간섭 패턴 (1급 객체)** |
| **비가역성** | 없음 | 없음 (순수) | 없음 | **관측이 가능성을 파괴** |

### 기존 언어가 가정하는 것

1. **계산은 결정적이다** — 같은 입력에 같은 출력. (AXOL: 관측 수준에 따라 다르다)
2. **읽기는 무료다** — 변수를 읽어도 변수는 바뀌지 않는다. (AXOL: 관측이 상태를 바꾼다)
3. **답은 하나다** — 함수는 하나의 값을 반환한다. (AXOL: 가능성의 분포를 반환한다)
4. **관계는 파생적이다** — A와 B가 먼저, A-B 관계는 나중. (AXOL: 관계가 먼저)

AXOL은 이 네 가지 가정을 모두 버린다.

### 이것이 가능한 이유: 붕괴 기반 계산

```
전통적 계산:
  f(x) = y
  비용: f의 복잡도에 비례
  매 호출마다 같은 비용

AXOL:
  weave(구조)              # 1회: O(chaos dynamics)
  observe(구조, 입력, C)    # N회: O(dim) = O(1)
```

구조(Tapestry)가 만들어진 순간, 답은 이미 그 안에 있다. 관측은 그것을 읽는 행위일 뿐이다. Born rule에 따라 복소 진폭의 제곱을 구하고(`|ψ_i|²`), argmax를 취한다. dim이 고정되어 있으므로 상수 시간이다.

이것은 양자 컴퓨팅과 같은 구조다:
- **상태 준비**(weave)는 비싸다
- **측정**(observe)은 즉시다
- **측정은 상태를 변화시킨다** (비가역)

---

## 설치

```bash
# 클론
git clone https://github.com/user/axol-lang.git
cd axol-lang

# 빌드
cargo build --release

# 실행
./target/release/axol run examples/hello.axol
```

요구 사항: Rust 1.70+ (`faer` 선형대수 라이브러리 사용)

---

## 언어 레퍼런스

### 핵심 파이프라인: Declare → Weave → Observe

```
# 1. 구조를 선언한다
declare "name" {
    input x(dim)              # dim 차원의 입력 벡터
    input y(dim)              # 복수 입력 지원
    relate z <- x, y via <~>  # x와 y의 간섭으로 z 생성
    output z                  # z가 관측 대상
    quality omega=0.9 phi=0.8 # 품질 매개변수
}

# 2. 짜기 — 가능성의 구조 생성 (1회 비용)
weave name quantum=true seed=42

# 3. 관측 — 구조에서 읽기 (O(1)/회)
observe name {
    x = [0.1, 0.2, 0.3, ...]
    y = [0.4, 0.5, 0.6, ...]
}
```

### 붕괴 스펙트럼

| 명령 | 붕괴 수준 | 비용 | 얻는 것 | 잃는 것 |
|------|-----------|------|---------|---------|
| `gaze x` | C=0 | 없음 | 전체 확률 분포 | 없음 |
| `glimpse x 0.3` | C=0.3 | 일부 가능성 | 집중된 분포 | 약한 대안들 |
| `focus x 0.7` | C=0.7 | 대부분의 가능성 | 거의 확정된 분포 | 대부분의 대안 |
| `observe model {}` | C=1 | 나머지 전부 | 하나의 답 (인덱스) | 다른 모든 가능성 |

### Basin 구조

Basin은 **끌개 지형(attractor landscape)** — 가능성 공간의 위상 구조를 정의한다.

```
define_basins "space" {
    dim 8
    basin [0.9, 0.1, 0.2, ...] volume=0.4   # 끌개 1 (공간의 40%)
    basin [0.1, 0.8, 0.7, ...] volume=0.35   # 끌개 2 (35%)
    basin [0.5, 0.5, 0.9, ...] volume=0.25   # 끌개 3 (25%)
    fractal_dim 1.6
}

weave model quantum=true seed=42 from_basins="space"
```

### 관계 (v2)

관계는 **파동 사이의 구조**를 모델링하는 1급 객체다.

```
# 파동 생성
wave a = model { x = [...] }
wave b = model { x = [...] }

# 관계 생성
rel r = a <-> b via <~>          # 양방향, 보강 간섭

# 관계 관측
observe r {}                      # 간섭 패턴 확인

# 기대: 예상되는 결과
expect prior = [0.6, 0.3, 0.1, ...] strength=0.7

# 기대와 함께 관측 — 정렬도와 negativity 변화 추적
observe r {} with prior

# 확장: 가능성 재개방
widen r 0.3

# 충돌 해결
resolve r1, r2 with interfere    # 양자 간섭
resolve r1, r2 with superpose    # 중첩
```

### 간섭 패턴

| 패턴 | 구문 | 행동 | 정보 |
|------|------|------|------|
| 보강 | `<~>` | a + b 강화 | 엔트로피 증가 |
| 가산 | `<+>` | 기하 평균 | 엔트로피 중립 |
| 곱셈 | `<*>` | a × b, 양쪽 모두 강해야 생존 | 엔트로피 감소 |
| 상쇄 | `<!>` | a - b, 차이 증폭 | 엔트로피 가장 큰 감소 |
| 조건 | `<?>` | 위상 결합 | 엔트로피 보존 |

### 반복과 수렴

```
iterate model max=10 converge=prob_delta value=0.005 {
    x = [...]
}

confident model max=50 threshold=0.95 {
    x = [...]
}
```

### 학습

```
learn "xor" dim=4 quantum=1 seed=42 {
    [0.9, 0.1, 0.9, 0.1] = 0
    [0.9, 0.1, 0.1, 0.9] = 1
    [0.1, 0.9, 0.9, 0.1] = 1
    [0.1, 0.9, 0.1, 0.9] = 0
}
```

### 전체 명령어

| 명령 | 설명 |
|------|------|
| `declare` | Tapestry 구조 선언 |
| `weave` | 가능성 공간 생성 (1회) |
| `observe` | 완전 붕괴 (C=1) |
| `gaze` | 무붕괴 읽기 (C=0) |
| `glimpse` | 부분 붕괴 (C=gamma) |
| `focus` | 부분 붕괴, 파동 변이 |
| `reobserve` | 다중 관측, 평균 |
| `wave` | 이름 있는 파동 변수 생성 |
| `rel` | 파동 간 관계 생성 |
| `expect` | 기대 landscape 정의 |
| `widen` | 가능성 재개방 |
| `resolve` | 파동 간 충돌 해결 |
| `iterate` | 수렴까지 반복 |
| `confident` | 신뢰 임계값까지 관측 |
| `define_basins` | 끌개 기하 직접 정의 |
| `compose` | 다중 Tapestry 체이닝 |
| `gate` | 양자 논리 게이트 (and, or, not) |
| `learn` | 레이블 데이터로 학습 |
| `design` | Basin 구조 탐색 |

---

## 성능

```
weave (1회):        ~11ms   (dim=8, 카오스 역학)
observe:            ~5μs    (dim=8, O(1))
gaze:               ~5μs    (dim=8, O(1), C=0)
glimpse:            ~30μs   (dim=8, dephasing 포함)
rel observe:        ~5μs    (O(1))
focus:              ~25μs   (dim=8, density matrix 포함)
```

프레임당 NPC 20회 관측: **총 ~80μs**. 60fps에서 프레임 예산의 0.5%.

---

## 적용 분야

| 도메인 | C의 의미 | AXOL의 역할 |
|--------|----------|-------------|
| 게임 NPC AI | C=0 전략 탐색, C=1 행동 실행 | 매 프레임 O(1) 의사결정 |
| 로보틱스 | C=0 경로 공간, C=1 이동 | kHz 제어 루프 실시간 처리 |
| 지각 모델링 | expect=예측, wave=감각, C=관측 | 예측 오류 추적 (negativity_delta) |
| 대화 시스템 | C=0 가능한 응답, C=1 발화 | 동의/반박의 간섭 구조 |
| 절차적 내러티브 | C=0 모든 분기, C=1 선택 | 선택의 비가역적 비용 |
| 자율주행 | C=0 모든 경로, C=1 조향 | 실시간 + 관측이 상태를 바꿈 |
| 금융 HFT | C=0 시장 상태, C=1 주문 | μs 결정, 주문이 시장을 바꿈 |
| 생성 음악 | C=0 화성 가능성, C=1 연주 | `<~>` 협화, `<!>` 불협화 |
| 사회 시뮬레이션 | C=0 의견 분포, C=1 행동 | 관계 간섭으로 사회 역학 |
| 의료 진단 | C=0 가능한 질환, C=1 확정 | 검사(관측)가 진단을 바꿈 |

---

## 예제

```bash
axol run examples/hello.axol                     # 기본 파이프라인
axol run examples/hello_v2.axol                  # 관계 중심 문법
axol run examples/usecase_npc_realtime.axol       # 게임 NPC AI
axol run examples/usecase_perception.axol         # 예측 부호화
axol run examples/usecase_dialogue.axol           # 대화 역학
axol run examples/usecase_observation_cost.axol    # 앎의 비가역성
axol run examples/learn_xor.axol                  # 데이터 학습
```

---

## 아키텍처

```
src/
  dsl/
    lexer.rs        # 토크나이저
    parser.rs       # AST 구축
    compiler.rs     # 런타임 실행
  wave.rs           # Wave 타입 (복소 진폭, Born rule)
  collapse.rs       # 붕괴 역학
  density.rs        # 밀도 행렬 연산
  observatory.rs    # compute_wave, gaze, glimpse, observe
  relation.rs       # Relation 타입 (v2)
  weaver.rs         # Tapestry 구축
  dynamics.rs       # 카오스 엔진 (로지스틱 맵, 로렌츠)
  learn.rs          # 레이블 데이터 학습
  compose/
    iterate.rs      # 수렴 기반 반복
    confidence.rs   # 신뢰 투표
    logic.rs        # 양자 논리 게이트
    tapestry_chain.rs
    basin_designer.rs
```

---

## 라이선스

MIT
