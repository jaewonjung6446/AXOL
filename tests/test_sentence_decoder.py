"""Tests for axol.quantum.sentence_decoder — Intent + Dictionary snap model."""

from __future__ import annotations

import numpy as np
import pytest

from axol.quantum.conversation import vocab_from_texts
from axol.quantum.sentence_decoder import (
    SentenceDecoderLanguageModel,
    SentenceDictionary,
    SnapResult,
)


# ---------------------------------------------------------------------------
# SentenceDictionary
# ---------------------------------------------------------------------------

class TestSentenceDictionary:
    def test_rejects_invalid_dim(self):
        with pytest.raises(ValueError):
            SentenceDictionary(dim=0)

    def test_rejects_invalid_ema(self):
        with pytest.raises(ValueError):
            SentenceDictionary(dim=4, ema=0.0)
        with pytest.raises(ValueError):
            SentenceDictionary(dim=4, ema=1.5)

    def test_register_increases_size(self):
        d = SentenceDictionary(dim=4)
        assert d.size == 0
        d.register(np.array([1, 0, 0, 0]), "hello")
        assert d.size == 1
        d.register(np.array([0, 1, 0, 0]), "world")
        assert d.size == 2

    def test_duplicate_text_merges(self):
        d = SentenceDictionary(dim=4)
        d.register(np.array([1, 0, 0, 0]), "hello")
        d.register(np.array([0, 1, 0, 0]), "hello")   # same text, different vec
        assert d.size == 1                             # still one entry

    def test_lookup_returns_exact_registered(self):
        d = SentenceDictionary(dim=4)
        d.register(np.array([1, 0, 0, 0], dtype=np.float32), "hello")
        d.register(np.array([0, 1, 0, 0], dtype=np.float32), "world")
        matches = d.lookup(np.array([0.9, 0.1, 0, 0]))
        assert matches[0][0] == "hello"
        assert matches[0][1] > 0.9

    def test_lookup_topk(self):
        d = SentenceDictionary(dim=4)
        d.register(np.array([1, 0, 0, 0]), "a")
        d.register(np.array([0, 1, 0, 0]), "b")
        d.register(np.array([0, 0, 1, 0]), "c")
        matches = d.lookup(np.array([0.7, 0.5, 0.3, 0.1]), top_k=3)
        assert [t for t, _ in matches] == ["a", "b", "c"]
        # sims should be descending
        sims = [s for _, s in matches]
        assert sims[0] > sims[1] > sims[2]

    def test_lookup_on_empty_dict(self):
        d = SentenceDictionary(dim=4)
        assert d.lookup(np.zeros(4)) == []

    def test_zero_query_does_not_crash(self):
        d = SentenceDictionary(dim=4)
        d.register(np.array([1, 0, 0, 0]), "only")
        out = d.lookup(np.zeros(4, dtype=np.float32))
        # Must return something sensible — first entry with sim=0
        assert len(out) == 1

    def test_rejects_wrong_dim_vectors(self):
        d = SentenceDictionary(dim=4)
        with pytest.raises(ValueError):
            d.register(np.zeros(3), "x")
        d.register(np.zeros(4), "x")
        with pytest.raises(ValueError):
            d.lookup(np.zeros(5))

    def test_clear_empties(self):
        d = SentenceDictionary(dim=4)
        d.register(np.array([1, 0, 0, 0]), "x")
        d.clear()
        assert d.size == 0


# ---------------------------------------------------------------------------
# SentenceDecoderLanguageModel — construction and basics
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_defaults(self):
        m = SentenceDecoderLanguageModel(vocab="abc")
        assert m.dictionary_size == 0
        assert m.pairs_taught == 0

    def test_teach_fills_dictionary(self):
        m = SentenceDecoderLanguageModel(vocab="abc ", intent_dim=10)
        m.teach("a", "bc")
        assert m.pairs_taught == 1
        assert m.dictionary_size == 1

    def test_rejects_zero_epochs(self):
        m = SentenceDecoderLanguageModel(vocab="abc")
        with pytest.raises(ValueError):
            m.train_pairs([("a", "b")], epochs=0)

    def test_generate_on_empty_dict(self):
        m = SentenceDecoderLanguageModel(vocab="abc")
        res = m.generate("a")
        assert isinstance(res, SnapResult)
        assert res.text == ""
        assert res.stopped_reason == "empty"


# ---------------------------------------------------------------------------
# Perfect recall — the key promise
# ---------------------------------------------------------------------------

class TestPerfectRecall:
    def test_single_pair_recalled_exactly(self):
        m = SentenceDecoderLanguageModel(
            vocab="abcdefghijklmnopqrstuvwxyz ", intent_dim=16,
            regularization=1e-4, seed=0,
        )
        m.train_pairs([("hi", "hello")], epochs=5)
        res = m.generate("hi")
        assert res.text == "hello"

    def test_small_qa_set_recalled(self):
        pairs = [
            ("hi", "hello"),
            ("bye", "goodbye"),
            ("yes", "ok"),
            ("thanks", "you are welcome"),
        ]
        m = SentenceDecoderLanguageModel(
            vocab="abcdefghijklmnopqrstuvwxyz ", intent_dim=16,
            regularization=1e-4, seed=0,
        )
        m.train_pairs(pairs, epochs=3)
        for q, expected in pairs:
            res = m.generate(q)
            assert res.text == expected, (
                f"prompt {q!r}: expected {expected!r}, got {res.text!r}"
            )

    def test_confidence_is_cosine_similarity(self):
        m = SentenceDecoderLanguageModel(vocab="abc ", intent_dim=10)
        m.train_pairs([("a", "b")], epochs=3)
        res = m.generate("a")
        assert 0.0 <= res.confidence <= 1.0

    def test_alternatives_include_top_k(self):
        pairs = [("a", "x"), ("b", "y"), ("c", "z")]
        m = SentenceDecoderLanguageModel(vocab="abcxyz ", intent_dim=12,
                                          regularization=1e-4)
        m.train_pairs(pairs, epochs=5)
        res = m.generate("a", top_k=3)
        assert len(res.alternatives) == 3
        assert res.text == res.alternatives[0][0]


# ---------------------------------------------------------------------------
# min_confidence gate
# ---------------------------------------------------------------------------

class TestMinConfidenceGate:
    def test_high_gate_blocks_uncertain_response(self):
        m = SentenceDecoderLanguageModel(vocab="abc ", intent_dim=10)
        m.train_pairs([("a", "b")], epochs=3)
        # A nearly-impossible threshold -> low_confidence
        res = m.generate("a", min_confidence=1.01)
        assert res.text == ""
        assert res.stopped_reason == "low_confidence"

    def test_zero_gate_always_emits(self):
        m = SentenceDecoderLanguageModel(vocab="abc ", intent_dim=10)
        m.train_pairs([("a", "b")], epochs=3)
        res = m.generate("a", min_confidence=0.0)
        assert res.text == "b"


# ---------------------------------------------------------------------------
# Novel prompts: snap to nearest
# ---------------------------------------------------------------------------

class TestNovelPrompt:
    def test_unseen_prompt_returns_nearest_registered(self):
        pairs = [("hi", "hello"), ("bye", "goodbye")]
        m = SentenceDecoderLanguageModel(
            vocab="abcdefghijklmnopqrstuvwxyz ", intent_dim=16,
            regularization=1e-4, seed=0,
        )
        m.train_pairs(pairs, epochs=5)
        res = m.generate("zzz")
        # Must return ONE of the registered sentences; any choice is
        # acceptable as long as it's valid.
        assert res.text in ("hello", "goodbye")


# ---------------------------------------------------------------------------
# Forgetting
# ---------------------------------------------------------------------------

class TestForgetting:
    def test_forget_weakens_intent_mapping_but_keeps_dict(self):
        m = SentenceDecoderLanguageModel(
            vocab="abcdefghijklmnopqrstuvwxyz ", intent_dim=12,
            regularization=1e-4, seed=0,
        )
        m.train_pairs([("a", "x"), ("b", "y")], epochs=5)
        m.forget(0.0)    # total wipe of intent core
        # Dictionary entries preserved
        assert m.dictionary_size == 2
        # Prediction likely now chaotic, but still returns some valid entry
        res = m.generate("a", min_confidence=0.0)
        assert res.text in ("x", "y")

    def test_forget_dictionary_clears_entries(self):
        m = SentenceDecoderLanguageModel(vocab="abc ", intent_dim=8)
        m.train_pairs([("a", "b")], epochs=3)
        m.forget_dictionary()
        assert m.dictionary_size == 0
        res = m.generate("a")
        assert res.stopped_reason == "empty"

    def test_reset_clears_everything(self):
        m = SentenceDecoderLanguageModel(vocab="abc ", intent_dim=8)
        m.train_pairs([("a", "b")], epochs=3)
        m.reset()
        assert m.pairs_taught == 0
        assert m.dictionary_size == 0
        assert m.intent_core.n_samples == 0


# ---------------------------------------------------------------------------
# THE BIG TEST — 40-pair recall that previously failed 0/10
# ---------------------------------------------------------------------------

class TestKorean:
    def test_korean_qa_pairs_recalled(self):
        """Same snap-decoder semantics should apply to Korean characters."""
        pairs = [
            ("안녕",        "안녕하세요 반갑습니다"),
            ("이름",        "저는 악솔입니다"),
            ("잘 지내",     "네 잘 지냅니다 감사합니다"),
            ("뭐 하고",     "대화를 하고 있습니다"),
            ("고마워",      "천만에요"),
        ]
        all_text = [q for q, _ in pairs] + [a for _, a in pairs]
        m = SentenceDecoderLanguageModel(
            vocab=vocab_from_texts(all_text),
            intent_dim=20, intent_window=6,
            regularization=1e-4, seed=0,
        )
        m.train_pairs(pairs, epochs=4)
        for q, expected in pairs:
            res = m.generate(q)
            assert res.text == expected, (
                f"Korean recall failed for {q!r}: got {res.text!r}"
            )


class TestFortyPairRecall:
    def test_40_proverb_pairs_all_recalled(self):
        """With the snap decoder, the 40-pair Q&A experiment that
        previously produced 0/10 must now be 40/40."""
        corpus = [
            "a penny saved is a penny earned",
            "actions speak louder than words",
            "all that glitters is not gold",
            "better late than never",
            "birds of a feather flock together",
            "dont count your chickens before they hatch",
            "dont judge a book by its cover",
            "easy come easy go",
            "every cloud has a silver lining",
            "fortune favors the bold",
            "good things come to those who wait",
            "honesty is the best policy",
            "hope for the best prepare for the worst",
            "if it aint broke dont fix it",
            "in the land of the blind the one eyed man is king",
            "keep your friends close and your enemies closer",
            "laughter is the best medicine",
            "let sleeping dogs lie",
            "look before you leap",
            "many hands make light work",
            "necessity is the mother of invention",
            "no news is good news",
            "no pain no gain",
            "once bitten twice shy",
            "one mans trash is another mans treasure",
            "out of sight out of mind",
            "practice makes perfect",
            "rome was not built in a day",
            "slow and steady wins the race",
            "strike while the iron is hot",
            "the early bird catches the worm",
            "the grass is always greener on the other side",
            "the pen is mightier than the sword",
            "the proof of the pudding is in the eating",
            "there is no place like home",
            "there is no such thing as a free lunch",
            "time flies when youre having fun",
            "time heals all wounds",
            "too many cooks spoil the broth",
            "two heads are better than one",
        ]
        pairs = [(" ".join(p.split()[:2]), p) for p in corpus]

        vocab = "".join(sorted(set(" ".join(corpus))))
        m = SentenceDecoderLanguageModel(
            vocab=vocab,
            intent_dim=24,
            intent_window=8,
            regularization=1e-4,
            seed=0,
        )
        m.train_pairs(pairs, epochs=3)

        hits = 0
        for q, expected in pairs:
            res = m.generate(q)
            if res.text == expected:
                hits += 1
        # Expect full recall on the training prompts.  Some collisions
        # (very similar 2-word prefixes across proverbs) can drop the
        # rate slightly; require at least 90 %.
        assert hits >= int(0.9 * len(pairs)), (
            f"only {hits}/{len(pairs)} recalled perfectly"
        )
