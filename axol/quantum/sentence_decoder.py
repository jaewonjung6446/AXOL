"""Sentence-snap language model — Intent core + non-autoregressive decoder.

Solves the autoregressive drift problem observed in the char-level demos:
instead of producing one token at a time (and accumulating error), the
decoder *snaps* the predicted intent vector to the closest registered
sentence and emits it whole.

Metaphor (user's own):
    Intent Core   — "calculus toward a destination"
                    (continuous dynamics in phase space, closed-form)
    Sentence dict — "prunes away the unnecessary"
                    (discrete collapse to the nearest valid sentence)

Properties
----------
* Exact recall for every taught pair (whichever sentence is closest is
  the one that was taught in that intent region).
* Lookup cost: O(N * dim) for N registered sentences.
* No autoregressive error accumulation — a wrong first character can't
  snowball because generation is whole-sentence lookup.
* Cannot emit a novel sentence that was not registered — by design.

Axiom alignment
---------------
* Axiom 1: IntentCore update is still closed-form rank-1.  Dictionary
  registration is a write, not an iterative optimisation.
* Axiom 2: Within the registered set, recall is 100 % — the 99 %-accuracy
  target is met trivially for any prompt whose intent vector is closer
  to the correct entry than to any other.
* Axiom 3: The cosine confidence of the nearest match IS the confidence.
  ``min_confidence`` can gate outputs that the decoder is unsure about.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from axol.quantum.conversation import (
    CharTokenizer,
    Verbalizer,
    WorkingMemory,
)
from axol.quantum.online import OnlineLearner


# ---------------------------------------------------------------------------
# SentenceDictionary
# ---------------------------------------------------------------------------

class SentenceDictionary:
    """Maps intent vectors to whole sentences via cosine nearest-neighbour.

    Multiple registrations of the same ``text`` merge via an exponential
    moving average on the intent vector (default ``ema=0.5``: equal
    weight on old and new).  This gives the same "additive accumulation"
    flavour as the rest of AXOL — repeated exposures strengthen the
    stored representation.
    """

    def __init__(self, dim: int, ema: float = 0.5) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if not (0.0 < ema <= 1.0):
            raise ValueError("ema must be in (0, 1]")
        self.dim = int(dim)
        self.ema = float(ema)
        self._texts: list[str] = []
        self._vecs: list[np.ndarray] = []   # each L2-normalised
        self._index_of_text: dict[str, int] = {}

    @property
    def size(self) -> int:
        return len(self._texts)

    def register(self, vec: np.ndarray, text: str) -> None:
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        if v.size != self.dim:
            raise ValueError(f"vec has length {v.size}, expected {self.dim}")
        # L2-normalise so cosine similarity == dot product.
        n = float(np.linalg.norm(v))
        v = v / n if n > 1e-12 else v.copy()

        if text in self._index_of_text:
            # EMA-blend with the existing representation and renormalise.
            i = self._index_of_text[text]
            merged = self.ema * v + (1.0 - self.ema) * self._vecs[i]
            mn = float(np.linalg.norm(merged))
            if mn > 1e-12:
                merged = merged / mn
            self._vecs[i] = merged.astype(np.float32)
        else:
            self._texts.append(text)
            self._vecs.append(v.astype(np.float32))
            self._index_of_text[text] = len(self._texts) - 1

    def lookup(self, query_vec: np.ndarray, top_k: int = 1) -> list[tuple[str, float]]:
        """Return ``top_k`` (text, cosine similarity) tuples, highest first."""
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not self._texts:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        if q.size != self.dim:
            raise ValueError(f"query has length {q.size}, expected {self.dim}")
        qn = float(np.linalg.norm(q))
        if qn < 1e-12:
            return [(self._texts[0], 0.0)]
        q = q / qn
        V = np.stack(self._vecs, axis=0)      # (N, dim)
        sims = V @ q                           # (N,)
        k = min(top_k, sims.size)
        # Partial sort for efficiency, then full sort within the top-k block.
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self._texts[i], float(sims[i])) for i in idx]

    def clear(self) -> None:
        self._texts.clear()
        self._vecs.clear()
        self._index_of_text.clear()


# ---------------------------------------------------------------------------
# Snap generation result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SnapResult:
    """Output of ``SentenceDecoderLanguageModel.generate``."""

    text: str
    confidence: float                       # cosine to the chosen match
    alternatives: list[tuple[str, float]]   # top-k alternatives
    stopped_reason: str                     # "snap" | "low_confidence" | "empty"


# ---------------------------------------------------------------------------
# Sentence-decoder language model
# ---------------------------------------------------------------------------

class SentenceDecoderLanguageModel:
    """Intent core + sentence dictionary. Whole-sentence snap decoder.

    Drop-in replacement for LanguageModel in the "FAQ / taught-pairs"
    regime where autoregressive character-level generation breaks down.
    """

    DEFAULT_VOCAB = (
        "\n !\"#$%&'()*+,-./0123456789:;<=>?@"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
        "abcdefghijklmnopqrstuvwxyz{|}~"
    )

    def __init__(
        self,
        vocab: str | None = None,
        intent_dim: int = 16,
        intent_window: int = 8,
        decay: float = 0.7,
        forgetting_factor: float = 1.0,
        regularization: float = 1e-4,
        seed: int = 0,
        degree: int = 2,
        dict_ema: float = 0.5,
    ) -> None:
        self.vocab = vocab if vocab is not None else self.DEFAULT_VOCAB
        self.tokenizer = CharTokenizer(self.vocab)
        self.intent_verb = Verbalizer(
            vocab_size=self.tokenizer.vocab_size,
            dim=intent_dim,
            seed=seed,
        )
        self.intent_wm = WorkingMemory(
            dim=intent_dim, window=intent_window, decay=decay,
        )
        self.intent_core = OnlineLearner(
            dim=intent_dim,
            degree=degree,
            forgetting_factor=forgetting_factor,
            regularization=regularization,
        )
        self.dictionary = SentenceDictionary(dim=intent_dim, ema=dict_ema)
        self._pairs_taught = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _summarise_intent(self, text: str) -> np.ndarray:
        """Run ``text`` through the intent WorkingMemory and return its EMA."""
        self.intent_wm.reset()
        for tid in self.tokenizer.encode(text):
            self.intent_wm.add(self.intent_verb.encode_id(tid))
        return self.intent_wm.context()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def teach(self, text_in: str, text_out: str) -> None:
        """Absorb one (stimulus, response) pair.

        Two writes happen:

        1. IntentCore.observe_sample: learns ``summary(text_in) -> summary(text_out)``.
        2. Dictionary.register: pairs ``summary(text_out)`` with the literal
           ``text_out`` so the snap decoder can return it verbatim.
        """
        input_ctx = self._summarise_intent(text_in)
        target_intent = self._summarise_intent(text_out)
        self.intent_core.observe_sample(input_ctx, target_intent)
        self.dictionary.register(target_intent, text_out)
        self._pairs_taught += 1

    def train_pairs(
        self,
        pairs: list[tuple[str, str]],
        epochs: int = 1,
    ) -> None:
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        for _ in range(epochs):
            for tin, tout in pairs:
                self.teach(tin, tout)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        top_k: int = 1,
        min_confidence: float = 0.0,
    ) -> SnapResult:
        """Predict the intent vector and snap to the nearest registered sentence."""
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        if self.dictionary.size == 0:
            return SnapResult(
                text="", confidence=0.0, alternatives=[],
                stopped_reason="empty",
            )

        input_ctx = self._summarise_intent(prompt)
        predicted_intent = self.intent_core.predict(input_ctx)
        matches = self.dictionary.lookup(predicted_intent, top_k=top_k)

        if not matches:
            return SnapResult(
                text="", confidence=0.0, alternatives=[],
                stopped_reason="empty",
            )

        best_text, best_sim = matches[0]
        if best_sim < min_confidence:
            return SnapResult(
                text="", confidence=best_sim, alternatives=matches,
                stopped_reason="low_confidence",
            )
        return SnapResult(
            text=best_text, confidence=best_sim, alternatives=matches,
            stopped_reason="snap",
        )

    # ------------------------------------------------------------------
    # Diagnostics & admin
    # ------------------------------------------------------------------

    @property
    def omega(self) -> float:
        return self.intent_core.omega

    @property
    def phi(self) -> float:
        return self.intent_core.phi

    @property
    def pairs_taught(self) -> int:
        return self._pairs_taught

    @property
    def dictionary_size(self) -> int:
        return self.dictionary.size

    def forget(self, factor: float) -> None:
        """Decay the intent core; dictionary contents are preserved.

        This asymmetry is intentional: forgetting here weakens the
        *mapping* from prompts to intent regions, but the sentences
        themselves remain available if a future prompt lands near them.
        """
        self.intent_core.decay(factor)

    def forget_by_time(self, elapsed: float, half_life: float) -> None:
        self.intent_core.decay_by_time(elapsed, half_life)

    def forget_dictionary(self) -> None:
        """Clear the sentence dictionary — full amnesia for what to say."""
        self.dictionary.clear()

    def reset(self) -> None:
        """Wipe everything (intent mapping and dictionary)."""
        self.intent_core.reset()
        self.intent_wm.reset()
        self.dictionary.clear()
        self._pairs_taught = 0
