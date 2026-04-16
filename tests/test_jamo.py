"""Tests for axol.quantum.jamo — Hangul decompose/compose."""

from __future__ import annotations

import pytest

from axol.quantum.jamo import (
    compose,
    compose_safely,
    decompose,
    decompose_pairs,
    decompose_syllable,
)


class TestDecomposeSyllable:
    def test_simple_cho_jung(self):
        # 가 = ㄱ + ㅏ
        assert decompose_syllable("가") == "ㄱㅏ"

    def test_cho_jung_jong(self):
        # 한 = ㅎ + ㅏ + ㄴ
        assert decompose_syllable("한") == "ㅎㅏㄴ"

    def test_non_hangul_passthrough(self):
        assert decompose_syllable("a") == "a"
        assert decompose_syllable("1") == "1"
        assert decompose_syllable(" ") == " "

    def test_rejects_multi_char(self):
        with pytest.raises(ValueError):
            decompose_syllable("ab")


class TestDecomposeString:
    def test_basic_word(self):
        assert decompose("안녕") == "ㅇㅏㄴㄴㅕㅇ"

    def test_mixed_with_space(self):
        assert decompose("가 나") == "ㄱㅏ ㄴㅏ"

    def test_mixed_with_latin(self):
        assert decompose("hello 한") == "hello ㅎㅏㄴ"

    def test_empty(self):
        assert decompose("") == ""


class TestComposeRoundtrip:
    @pytest.mark.parametrize("text", [
        "안녕",
        "안녕하세요",
        "반가워요",
        "저는 악솔 입니다",
        "세 살 버릇 여든까지 간다",
        "가",
        "한",
    ])
    def test_decompose_then_compose_equals_original(self, text):
        assert compose(decompose(text)) == text

    def test_compose_non_hangul_passthrough(self):
        assert compose("hello") == "hello"
        assert compose("") == ""

    def test_compose_mixed_with_punctuation(self):
        original = "안녕, 세상!"
        assert compose(decompose(original)) == original


class TestDecomposePairs:
    def test_pairs(self):
        pairs = [("안녕", "반가워요"), ("잘 가", "안녕히 가세요")]
        out = decompose_pairs(pairs)
        assert len(out) == 2
        assert out[0][0] == decompose("안녕")
        assert out[0][1] == decompose("반가워요")


class TestComposeSafely:
    def test_valid_input(self):
        assert compose_safely(decompose("안녕")) == "안녕"

    def test_garbage_passthrough(self):
        # If weird jamo sequence can't compose, shouldn't crash
        result = compose_safely("ㅇㅏㄴ")
        assert isinstance(result, str)


class TestSuffixSimilarity:
    """자모 분해의 핵심 목적: 어미 변형이 공통 prefix를 공유하게 됨."""

    def test_greeting_family_shares_prefix(self):
        for base in ["안녕하세요", "안녕하십니까", "안녕히 가세요"]:
            decomposed = decompose(base)
            assert decomposed.startswith(decompose("안녕")), (
                f"{base} -> {decomposed} should start with decomposed '안녕'"
            )

    def test_emotion_family_shares_prefix(self):
        # 기뻐요, 기뻤어요, 기쁘다 -- 공통 어간 prefix
        shared = decompose("기")
        for word in ["기뻐요", "기뻤어요", "기쁘다"]:
            assert shared in decompose(word)
