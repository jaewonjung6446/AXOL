"""Tests for axol.quantum.two_stage — intent/surface split language model."""

from __future__ import annotations

import numpy as np
import pytest

from axol.quantum.koopman import lifted_dim
from axol.quantum.two_stage import (
    TwoStageGeneration,
    TwoStageLanguageModel,
    TwoStageReport,
)


# ---------------------------------------------------------------------------
# Construction / config
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_defaults(self):
        m = TwoStageLanguageModel(vocab="abc")
        assert m.intent_dim == 8
        assert m.surface_dim == 16

    def test_rejects_non_positive_dims(self):
        with pytest.raises(ValueError):
            TwoStageLanguageModel(vocab="abc", intent_dim=0)
        with pytest.raises(ValueError):
            TwoStageLanguageModel(vocab="abc", surface_dim=-1)

    def test_projection_shape(self):
        m = TwoStageLanguageModel(vocab="abc", intent_dim=6, surface_dim=12)
        assert m._projection.shape == (6, 12)


# ---------------------------------------------------------------------------
# Memory advantage (the whole point of the split)
# ---------------------------------------------------------------------------

class TestMemoryAdvantage:
    def test_total_matrix_elements_beats_single_core(self):
        """dim=8 + dim=16 must beat a single dim=24 in stored elements."""
        m = TwoStageLanguageModel(vocab="abc", intent_dim=8, surface_dim=16)
        single_core_ld = lifted_dim(24, 2)
        single_elements = single_core_ld ** 2
        assert m.total_matrix_elements < single_elements
        # ~4x savings expected
        assert m.total_matrix_elements * 3 < single_elements

    def test_explicit_lifted_dims(self):
        m = TwoStageLanguageModel(vocab="abc", intent_dim=8, surface_dim=16)
        assert m.intent_core.ld == lifted_dim(8, 2)    # 45
        assert m.surface_core.ld == lifted_dim(16, 2)  # 153


# ---------------------------------------------------------------------------
# Both cores receive updates during teaching
# ---------------------------------------------------------------------------

class TestTeachingUpdatesBothCores:
    def test_sample_counts_rise(self):
        m = TwoStageLanguageModel(vocab="abc.! ", intent_dim=6, surface_dim=10)
        assert m.intent_core.n_samples == 0
        assert m.surface_core.n_samples == 0
        m.teach("a", "bc")
        assert m.intent_core.n_samples == 1          # one intent mapping
        assert m.surface_core.n_samples >= len("bc") # surface tokens + eos

    def test_report_tracks_pairs(self):
        m = TwoStageLanguageModel(vocab="abc.! ", intent_dim=6, surface_dim=10)
        m.teach("a", "b")
        m.teach("b", "c")
        r = m.report()
        assert isinstance(r, TwoStageReport)
        assert r.pairs_taught == 2
        assert r.tokens_seen >= 2


# ---------------------------------------------------------------------------
# Generation behaviour
# ---------------------------------------------------------------------------

class TestGeneration:
    def test_generate_returns_struct(self):
        m = TwoStageLanguageModel(vocab="abc.! ", intent_dim=6, surface_dim=10)
        res = m.generate("a", max_len=3)
        assert isinstance(res, TwoStageGeneration)
        assert isinstance(res.text, str)
        assert 0.0 <= res.omega <= 1.0

    def test_memorises_single_pair(self):
        m = TwoStageLanguageModel(
            vocab="abc.! ", intent_dim=8, surface_dim=14,
            regularization=1e-4,
        )
        m.train_pairs([("a", "bc")], epochs=40)
        res = m.generate("a", max_len=5)
        assert res.text.startswith("b"), f"got {res.text!r}"

    def test_distinguishes_two_inputs(self):
        pairs = [("x", "1"), ("y", "2")]
        m = TwoStageLanguageModel(
            vocab="xy12.! ", intent_dim=8, surface_dim=14,
            regularization=1e-4, seed=0,
        )
        m.train_pairs(pairs, epochs=50)
        ox = m.generate("x", max_len=2).text
        oy = m.generate("y", max_len=2).text
        assert ox != oy

    def test_min_confidence_can_stop(self):
        m = TwoStageLanguageModel(vocab="abc", intent_dim=6, surface_dim=10)
        res = m.generate("a", max_len=10, min_confidence=0.99)
        assert res.stopped_reason == "low_confidence"
        assert res.text == ""

    def test_greedy_is_deterministic(self):
        m = TwoStageLanguageModel(
            vocab="abc.! ", intent_dim=8, surface_dim=14,
            regularization=1e-4,
        )
        m.train_pairs([("a", "bc")], epochs=20)
        r1 = m.generate("a", max_len=5, temperature=0.0)
        r2 = m.generate("a", max_len=5, temperature=0.0)
        assert r1.text == r2.text

    def test_sampling_seed_reproducible(self):
        m = TwoStageLanguageModel(
            vocab="abc.! ", intent_dim=8, surface_dim=14,
            regularization=1e-4,
        )
        m.train_pairs([("a", "bc")], epochs=20)
        r1 = m.generate("a", max_len=5, temperature=1.0, seed=42)
        r2 = m.generate("a", max_len=5, temperature=1.0, seed=42)
        assert r1.text == r2.text


# ---------------------------------------------------------------------------
# Forgetting delegates to both cores
# ---------------------------------------------------------------------------

class TestForgetDelegates:
    def test_forget_wipes_both(self):
        m = TwoStageLanguageModel(
            vocab="abc", intent_dim=6, surface_dim=10,
            regularization=1e-4,
        )
        m.train_pairs([("a", "b")], epochs=20)
        G_intent_before = m.intent_core._G.copy()
        G_surface_before = m.surface_core._G.copy()
        m.forget(0.0)
        # Both cores should now have G = reg*I
        reg = m.intent_core.regularization
        assert np.allclose(m.intent_core._G, np.eye(m.intent_core.ld) * reg)
        assert np.allclose(
            m.surface_core._G,
            np.eye(m.surface_core.ld) * m.surface_core.regularization,
        )

    def test_forget_by_time_matches_direct(self):
        m_time = TwoStageLanguageModel(vocab="abc", intent_dim=6, surface_dim=10)
        m_manual = TwoStageLanguageModel(vocab="abc", intent_dim=6, surface_dim=10)
        m_time.train_pairs([("a", "b")], epochs=15)
        m_manual.train_pairs([("a", "b")], epochs=15)

        m_time.forget_by_time(elapsed=3.0, half_life=1.0)  # factor = 0.125
        m_manual.forget(0.125)

        assert np.allclose(
            m_time.intent_core._G, m_manual.intent_core._G, atol=1e-10
        )
        assert np.allclose(
            m_time.surface_core._G, m_manual.surface_core._G, atol=1e-10
        )


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_counters(self):
        m = TwoStageLanguageModel(vocab="abc", intent_dim=6, surface_dim=10)
        m.train_pairs([("a", "b")], epochs=5)
        assert m.report().pairs_taught > 0
        m.reset()
        assert m.report().pairs_taught == 0
        assert m.intent_core.n_samples == 0
        assert m.surface_core.n_samples == 0


# ---------------------------------------------------------------------------
# Omega-as-weakest-link semantics
# ---------------------------------------------------------------------------

class TestWeakestLinkOmega:
    def test_omega_equals_min_of_stages(self):
        m = TwoStageLanguageModel(vocab="abc", intent_dim=6, surface_dim=10)
        m.train_pairs([("a", "b")], epochs=10)
        r = m.report()
        assert r.omega == pytest.approx(min(r.intent_omega, r.surface_omega))
        assert r.phi == pytest.approx(min(r.intent_phi, r.surface_phi))


# ---------------------------------------------------------------------------
# Integrated Q&A with the same data the single-core test used
# ---------------------------------------------------------------------------

class TestIntegratedQA:
    def test_small_qa_set(self):
        pairs = [
            ("hi", "hello"),
            ("bye", "goodbye"),
            ("yes", "ok"),
        ]
        m = TwoStageLanguageModel(
            vocab="abcdefghijklmnopqrstuvwxyz !.?",
            intent_dim=10,
            surface_dim=20,
            intent_window=6,
            surface_window=8,
            regularization=1e-4,
            seed=0,
        )
        m.train_pairs(pairs, epochs=80)

        hits = 0
        for qin, expected in pairs:
            res = m.generate(qin, max_len=10, temperature=0.0)
            if res.text.startswith(expected):
                hits += 1
        # Two-stage is slightly lossier than single-core; require >= 2/3.
        assert hits >= 2, (
            f"only {hits}/{len(pairs)} pairs memorised — "
            f"two-stage may need more epochs or bigger dims"
        )
