"""한글 자모(jamo) 분해·조합 — 어미 변형 일반화용.

한글 음절은 초성·중성·종성의 합으로 생성되므로, 음절 단위 토큰화는
"안녕" / "안녕하세요" / "안녕히" 같은 어미 변형이 완전히 별개 토큰처럼
보입니다.  자모 단위로 내려가면 앞 음절을 공유하게 되어 AXOL의
Verbalizer 공간에서 cosine 유사도가 자연스럽게 높아집니다.

기본 사용법
----------
    from axol.quantum.jamo import decompose, compose

    seq = decompose("안녕하세요")        # "ㅇㅏㄴ ㄴㅕㅇ ㅎㅏ ㅅㅔ ㅇㅛ"
                                         # (실제로는 공백 없이 연결)
    text = compose("ㅇㅏㄴ ㄴㅕㅇ")       # "안녕"

주의
----
한국어 자모는 여러 방식이 있지만 여기서는 유니코드 범위 U+AC00~U+D7A3
(완성형 한글)만 다루며, 호환 자모(U+3131 등)를 사용합니다.
조합은 greedy parsing입니다 — 문맥상 모호할 때는 "최대 매칭"으로 결정.
"""

from __future__ import annotations


HANGUL_BASE = 0xAC00
HANGUL_END = 0xD7A3

_CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"                    # 19 초성
_JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"              # 21 중성
_JONG = ["",  # 0 = 종성 없음
         "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ",
         "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ",
         "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ",
         "ㅋ", "ㅌ", "ㅍ", "ㅎ"]                                # 28 종성

_CHO_IDX = {c: i for i, c in enumerate(_CHO)}
_JUNG_IDX = {c: i for i, c in enumerate(_JUNG)}
_JONG_IDX = {c: i for i, c in enumerate(_JONG)}

# 유효한 자모 집합 (공백 제외)
_JAMO_SET = set(_CHO) | set(_JUNG) | set(c for c in _JONG if c)


# ---------------------------------------------------------------------------
# 분해
# ---------------------------------------------------------------------------

def decompose_syllable(ch: str) -> str:
    """음절 하나를 자모 시퀀스로 분해. 한글이 아니면 그대로."""
    if len(ch) != 1:
        raise ValueError("단일 문자를 넘겨야 합니다")
    code = ord(ch)
    if HANGUL_BASE <= code <= HANGUL_END:
        offset = code - HANGUL_BASE
        cho_i = offset // (21 * 28)
        jung_i = (offset // 28) % 21
        jong_i = offset % 28
        out = _CHO[cho_i] + _JUNG[jung_i]
        if jong_i > 0:
            out += _JONG[jong_i]
        return out
    return ch


def decompose(text: str) -> str:
    """문자열 전체를 자모 시퀀스로 분해."""
    return "".join(decompose_syllable(c) for c in text)


# ---------------------------------------------------------------------------
# 조합
# ---------------------------------------------------------------------------

def _compose_one(cho: str, jung: str, jong: str = "") -> str:
    """자모 → 음절 하나."""
    if cho not in _CHO_IDX or jung not in _JUNG_IDX:
        return cho + jung + jong
    jong_i = _JONG_IDX[jong] if jong else 0
    code = (
        HANGUL_BASE
        + _CHO_IDX[cho] * 21 * 28
        + _JUNG_IDX[jung] * 28
        + jong_i
    )
    return chr(code)


def compose(jamo_text: str) -> str:
    """자모 시퀀스 → 음절 복원 (greedy).

    규칙:
      - cho + jung + jong  (3자모)  → 한 음절
      - cho + jung         (2자모)  → 한 음절
      - 애매한 경우:  cho + jung + jong_candidate + cho_next + jung_next
        에서 jong_candidate를 종성으로 해석 (greedy max match).
        그러나 종성 뒤에 cho+jung이 없으면 단독 자모로 보고 패스.
    """
    out: list[str] = []
    i = 0
    n = len(jamo_text)
    while i < n:
        c = jamo_text[i]
        if c in _CHO_IDX and i + 1 < n and jamo_text[i + 1] in _JUNG_IDX:
            cho = c
            jung = jamo_text[i + 1]
            # 종성 후보: 다음 글자가 종성 가능 && 그 다음이 (cho+jung) 또는 끝
            jong = ""
            if i + 2 < n and jamo_text[i + 2] in _JONG_IDX and _JONG_IDX[jamo_text[i + 2]] > 0:
                # 뒤에 새 음절이 오면 종성으로, 그렇지 않으면 단독 자모
                if i + 3 >= n:
                    jong = jamo_text[i + 2]
                elif jamo_text[i + 3] in _CHO_IDX and i + 4 < n and jamo_text[i + 4] in _JUNG_IDX:
                    jong = jamo_text[i + 2]
                elif jamo_text[i + 3] not in _JAMO_SET:
                    jong = jamo_text[i + 2]
            out.append(_compose_one(cho, jung, jong))
            i += 3 if jong else 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


# ---------------------------------------------------------------------------
# 편의: corpus 전체 분해
# ---------------------------------------------------------------------------

def decompose_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """(prompt, response) 쌍 리스트 전체를 자모 시퀀스로 분해."""
    return [(decompose(q), decompose(a)) for q, a in pairs]


def compose_safely(text: str) -> str:
    """자모가 섞여 있을 수 있는 문자열을 안전하게 음절화.

    모르는 자모 조합이 있으면 그 부분은 원본 그대로 남겨둡니다.
    """
    try:
        return compose(text)
    except Exception:
        return text
