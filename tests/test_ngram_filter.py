"""Tests for axol.quantum.ngram_filter."""

from __future__ import annotations

import pytest

from axol.quantum.ngram_filter import NgramFilter


class TestConstruction:
    def test_rejects_non_positive_n(self):
        with pytest.raises(ValueError):
            NgramFilter(["hi"], n=0)

    def test_counts_known_ngrams(self):
        f = NgramFilter(["the quick brown fox", "the lazy dog"], n=2)
        # bigrams from text 1: (the, quick)(quick, brown)(brown, fox)
        # bigrams from text 2: (the, lazy)(lazy, dog)
        assert f.known_ngrams() == 5
        assert f.total_observations() == 5


class TestScore:
    def test_short_text_returns_one(self):
        f = NgramFilter(["a b c"], n=2)
        assert f.score("a") == 1.0
        assert f.score("") == 1.0

    def test_full_match(self):
        f = NgramFilter(["the quick brown fox"], n=2)
        # "the quick brown" has bigrams (the,quick) and (quick,brown) — both known
        assert f.score("the quick brown") == 1.0

    def test_partial_match(self):
        f = NgramFilter(["the quick brown fox"], n=2)
        # "the quick zzz" -> bigrams (the,quick) known, (quick,zzz) unknown
        score = f.score("the quick zzz")
        assert score == pytest.approx(0.5)

    def test_no_match(self):
        f = NgramFilter(["the quick brown fox"], n=2)
        assert f.score("foo bar baz") == 0.0

    def test_is_grammatical(self):
        f = NgramFilter(["the quick brown fox"], n=2)
        assert f.is_grammatical("the quick brown", threshold=0.9)
        assert not f.is_grammatical("foo bar", threshold=0.5)


class TestPickBest:
    def test_picks_highest_score(self):
        f = NgramFilter(["the quick brown fox"], n=2)
        candidates = [
            "foo bar baz",              # 0.0
            "the quick zzz",            # 0.5
            "the quick brown",          # 1.0
        ]
        best, score = f.pick_best(candidates)
        assert best == "the quick brown"
        assert score == 1.0

    def test_empty_candidates(self):
        f = NgramFilter(["a b"], n=2)
        text, score = f.pick_best([])
        assert text == ""
        assert score == 0.0

    def test_tie_breaker(self):
        f = NgramFilter(["a b"], n=2)
        # Both score 1.0 (single word -> auto 1.0)
        best, _ = f.pick_best(["x", "y"], tie_breaker="y")
        assert best == "y"


class TestKoreanNgrams:
    def test_korean_bigrams(self):
        texts = [
            "안녕하세요 반갑습니다",
            "저는 악솔 입니다",
        ]
        f = NgramFilter(texts, n=2)
        # Known: (안녕하세요, 반갑습니다), (저는, 악솔), (악솔, 입니다)
        assert f.known_ngrams() == 3
        assert f.is_grammatical("안녕하세요 반갑습니다", threshold=0.9)
        assert not f.is_grammatical("고양이 개구리", threshold=0.5)
