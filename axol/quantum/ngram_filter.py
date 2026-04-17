"""N-gram 기반 문법 필터 — 프랙탈 블렌드 출력의 비문 방지.

프랙탈 노이즈 합성은 단어 단위로 여러 문장 조각을 섞습니다.  조합이
그럴듯하면 창조적이지만, 학습 corpus에 본 적 없는 조합(비문)도
나올 수 있습니다.  이 필터는:

    - 학습 corpus에서 단어 n-gram 집합을 미리 수집
    - 생성된 문장의 n-gram 중 얼마가 학습집합에 포함되는지 계산
    - 임계값 미만이면 "grammatical=False"로 보고

사용:

    filt = NgramFilter(training_texts, n=2)
    score = filt.score("새로 생성된 문장")   # 0.0..1.0
    if filt.is_grammatical(text, threshold=0.5):
        ...

프랙탈 generator와 통합하려면 ``FractalTextGenerator.variations()``
결과 중 점수가 가장 높은 것을 고르면 됩니다.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable


class NgramFilter:
    """Grammar filter based on known n-gram occurrences in training texts."""

    def __init__(self, texts: Iterable[str], n: int = 2) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = int(n)
        self._ngrams: set[tuple[str, ...]] = set()
        self._counts: Counter[tuple[str, ...]] = Counter()
        total = 0
        for text in texts:
            words = text.split()
            if len(words) < self.n:
                continue
            for i in range(len(words) - self.n + 1):
                gram = tuple(words[i:i + self.n])
                self._ngrams.add(gram)
                self._counts[gram] += 1
                total += 1
        self._total = total

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, text: str) -> float:
        """학습된 n-gram에 속하는 비율. 짧은 문장은 자동 1.0."""
        words = text.split()
        if len(words) < self.n:
            return 1.0
        total = 0
        seen = 0
        for i in range(len(words) - self.n + 1):
            total += 1
            if tuple(words[i:i + self.n]) in self._ngrams:
                seen += 1
        return seen / total if total > 0 else 1.0

    def is_grammatical(self, text: str, threshold: float = 0.5) -> bool:
        return self.score(text) >= threshold

    def known_ngrams(self) -> int:
        return len(self._ngrams)

    def total_observations(self) -> int:
        return self._total

    # ------------------------------------------------------------------
    # Pick best of variations
    # ------------------------------------------------------------------

    def pick_best(
        self,
        candidates: list[str],
        tie_breaker: str | None = None,
    ) -> tuple[str, float]:
        """여러 후보 중 n-gram 점수가 가장 높은 것을 고름.

        동점이면 ``tie_breaker``(예: macro 문장)에 가까운 것을 우선,
        그래도 동점이면 첫 번째 후보.
        """
        if not candidates:
            return "", 0.0
        scored = [(c, self.score(c)) for c in candidates]
        best_score = max(s for _, s in scored)
        top = [c for c, s in scored if s == best_score]
        if len(top) == 1 or tie_breaker is None:
            return top[0], best_score
        # Tie-breaker: prefer candidate matching tie_breaker exactly
        for c in top:
            if c == tie_breaker:
                return c, best_score
        return top[0], best_score
