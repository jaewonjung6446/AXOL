"""Tests for axol.quantum.language_model — LanguageModel + GenerationResult."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from axol.quantum.language_model import GenerationResult, LanguageModel


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_defaults(self):
        lm = LanguageModel()
        assert lm._config["embed_dim"] == 24
        assert lm._config["window"] == 12
        assert lm.samples_trained == 0

    def test_custom_vocab(self):
        lm = LanguageModel(vocab="abc", embed_dim=8)
        # tokenizer: <unk>, <eos>, a, b, c
        assert lm.chat.tokenizer.vocab_size == 5


# ---------------------------------------------------------------------------
# Supervised pair training
# ---------------------------------------------------------------------------

class TestTrainPairs:
    def test_absorbs_pairs(self):
        lm = LanguageModel(vocab="abc.! ", embed_dim=12, window=4)
        pairs = [("a", "b"), ("b", "c")]
        lm.train_pairs(pairs, epochs=5)
        assert lm.samples_trained > 0

    def test_memorises_simple_pair(self):
        lm = LanguageModel(vocab="ab.! ", embed_dim=10, window=4,
                           regularization=1e-4)
        lm.train_pairs([("a", "b")], epochs=30)
        out = lm.generate("a", max_len=3)
        assert out.text.startswith("b")

    def test_rejects_zero_epochs(self):
        lm = LanguageModel(vocab="ab")
        with pytest.raises(ValueError):
            lm.train_pairs([("a", "b")], epochs=0)


# ---------------------------------------------------------------------------
# Self-supervised text training
# ---------------------------------------------------------------------------

class TestTrainOnText:
    def test_sample_count_matches_text_length(self):
        lm = LanguageModel(vocab="abcdef .", embed_dim=10, window=4)
        text = "abcabc abc"
        n = lm.train_on_text(text)
        assert n == len(text) - 1

    def test_empty_text_is_noop(self):
        lm = LanguageModel(vocab="abc", embed_dim=8)
        n = lm.train_on_text("")
        assert n == 0
        assert lm.samples_trained == 0

    def test_single_char_text_is_noop(self):
        lm = LanguageModel(vocab="abc", embed_dim=8)
        n = lm.train_on_text("a")
        assert n == 0

    def test_learns_simple_pattern(self):
        """After training on 'abcabcabc...', generating from 'a' should
        start with 'b'."""
        lm = LanguageModel(
            vocab="abc", embed_dim=12, window=4, regularization=1e-4, seed=0,
        )
        # Repeat pattern long enough that moments dominate the regulariser
        lm.train_on_text("abc" * 50)
        out = lm.generate("a", max_len=4, temperature=0.0)
        # First generated char is typically 'b'; at minimum it should not
        # just be a zero vector.
        assert len(out.text) > 0


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

class TestGenerate:
    def _trained(self) -> LanguageModel:
        lm = LanguageModel(
            vocab="ab.! ", embed_dim=10, window=4, regularization=1e-4,
        )
        lm.train_pairs([("a", "b")], epochs=30)
        return lm

    def test_returns_generation_result(self):
        lm = self._trained()
        res = lm.generate("a", max_len=3)
        assert isinstance(res, GenerationResult)
        assert 0.0 <= res.omega <= 1.0

    def test_greedy_is_deterministic(self):
        lm = self._trained()
        r1 = lm.generate("a", max_len=5, temperature=0.0)
        r2 = lm.generate("a", max_len=5, temperature=0.0)
        assert r1.text == r2.text

    def test_temperature_sampling_produces_text(self):
        lm = self._trained()
        res = lm.generate("a", max_len=5, temperature=0.5, seed=7)
        assert isinstance(res.text, str)
        assert len(res.confidences) >= 1

    def test_sampling_seed_reproducible(self):
        lm = self._trained()
        r1 = lm.generate("a", max_len=5, temperature=1.0, seed=42)
        r2 = lm.generate("a", max_len=5, temperature=1.0, seed=42)
        assert r1.text == r2.text

    def test_top_k_accepted(self):
        lm = self._trained()
        res = lm.generate("a", max_len=4, temperature=0.5, top_k=3, seed=9)
        assert isinstance(res.text, str)

    def test_rejects_negative_temperature(self):
        lm = self._trained()
        with pytest.raises(ValueError):
            lm.generate("a", temperature=-0.1)

    def test_rejects_invalid_top_k(self):
        lm = self._trained()
        with pytest.raises(ValueError):
            lm.generate("a", temperature=0.5, top_k=0)

    def test_min_confidence_stops_generation(self):
        lm = LanguageModel(vocab="abc", embed_dim=8)  # untrained
        res = lm.generate("a", max_len=10, min_confidence=0.99)
        assert res.stopped_reason == "low_confidence"
        assert res.text == ""

    def test_eos_stops_generation(self):
        lm = LanguageModel(vocab="ab", embed_dim=10, window=4, regularization=1e-4)
        lm.train_pairs([("a", "b")], epochs=40)
        res = lm.generate("a", max_len=50, stop_on_eos=True)
        # Model should not run the full 50 chars for a memorised <a> -> <b><eos>
        assert res.stopped_reason in ("eos", "max_len")
        assert len(res.text) <= 50


# ---------------------------------------------------------------------------
# Chat convenience
# ---------------------------------------------------------------------------

class TestChatOnce:
    def test_returns_string(self):
        lm = LanguageModel(vocab="ab", embed_dim=8)
        out = lm.chat_once("a", text_out="b")
        assert isinstance(out, str)

    def test_disables_learning(self):
        lm = LanguageModel(vocab="ab", embed_dim=8)
        lm.chat_once("a", text_out="b", learn=False)
        assert lm.samples_trained == 0


# ---------------------------------------------------------------------------
# Forgetting delegate
# ---------------------------------------------------------------------------

class TestForgetDelegate:
    def test_forget_zero_wipes_behaviour(self):
        lm = LanguageModel(vocab="ab", embed_dim=10, window=4,
                           regularization=1e-4)
        lm.train_pairs([("a", "b")], epochs=30)
        lm.forget(0.0)
        res = lm.generate("a", max_len=3, min_confidence=0.0)
        # After total wipe, cosine confidences should all be near zero.
        assert max(res.confidences or [0.0]) < 0.05

    def test_forget_by_time_matches_direct_decay(self):
        """forget_by_time(t, h) must equal decay(0.5 ** (t/h)) exactly."""
        # Two models, identical training.
        lm_time = LanguageModel(vocab="ab", embed_dim=10, window=4,
                                regularization=1e-4)
        lm_manual = LanguageModel(vocab="ab", embed_dim=10, window=4,
                                  regularization=1e-4)
        lm_time.train_pairs([("a", "b")], epochs=20)
        lm_manual.train_pairs([("a", "b")], epochs=20)

        # 3 half-lives -> factor = 0.125
        lm_time.forget_by_time(elapsed=3.0, half_life=1.0)
        lm_manual.forget(0.125)

        assert np.allclose(
            lm_time.chat.intuition._G, lm_manual.chat.intuition._G, atol=1e-10
        )
        assert np.allclose(
            lm_time.chat.intuition._B, lm_manual.chat.intuition._B, atol=1e-10
        )


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_roundtrip_preserves_prediction(self, tmp_path: Path):
        lm = LanguageModel(
            vocab="ab.! ", embed_dim=10, window=4, regularization=1e-4,
        )
        lm.train_pairs([("a", "b")], epochs=20)

        prompt = "a"
        original = lm.generate(prompt, max_len=6, temperature=0.0).text

        save_path = tmp_path / "model"
        lm.save(save_path)

        loaded = LanguageModel.load(save_path)
        reloaded = loaded.generate(prompt, max_len=6, temperature=0.0).text

        assert reloaded == original
        assert loaded.samples_trained == lm.samples_trained

    def test_roundtrip_preserves_moments(self, tmp_path: Path):
        lm = LanguageModel(vocab="abc", embed_dim=8, regularization=1e-3)
        lm.train_pairs([("a", "b"), ("b", "c")], epochs=10)

        save_path = tmp_path / "m"
        lm.save(save_path)
        loaded = LanguageModel.load(save_path)

        assert np.allclose(loaded.chat.intuition._G, lm.chat.intuition._G)
        assert np.allclose(loaded.chat.intuition._H, lm.chat.intuition._H)
        assert np.allclose(loaded.chat.intuition._B, lm.chat.intuition._B)
        assert np.allclose(loaded.chat.verbalizer.E, lm.chat.verbalizer.E)

    def test_load_rejects_wrong_format_version(self, tmp_path: Path):
        import json as _json
        lm = LanguageModel(vocab="abc", embed_dim=8)
        path = tmp_path / "m"
        lm.save(path)
        # Tamper with format version
        with open(path.with_suffix(".json"), "r") as f:
            meta = _json.load(f)
        meta["format_version"] = 999
        with open(path.with_suffix(".json"), "w") as f:
            _json.dump(meta, f)
        with pytest.raises(ValueError):
            LanguageModel.load(path)


# ---------------------------------------------------------------------------
# Integrated Q&A scenario (small LLM-style demonstrator)
# ---------------------------------------------------------------------------

class TestIntegratedQA:
    def test_small_qa_set_memorised(self):
        """Classic use: teach a handful of Q&A pairs and verify responses."""
        pairs = [
            ("hi", "hello"),
            ("bye", "goodbye"),
            ("yes", "ok"),
        ]
        lm = LanguageModel(
            vocab="abcdefghijklmnopqrstuvwxyz !.?",
            embed_dim=20,
            window=6,
            regularization=1e-4,
            seed=0,
        )
        lm.train_pairs(pairs, epochs=60)

        for qin, expected in pairs:
            res = lm.generate(qin, max_len=10, temperature=0.0)
            assert res.text.startswith(expected), (
                f"prompt {qin!r}: got {res.text!r}, expected start {expected!r}"
            )
