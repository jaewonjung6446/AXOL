"""Tests for axol.quantum.hierarchical — N-level language model."""

from __future__ import annotations

import numpy as np
import pytest

from axol.quantum.hierarchical import (
    HierarchicalLanguageModel,
    HierarchicalReport,
    HierarchicalResult,
    LevelConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_3level(**overrides) -> HierarchicalLanguageModel:
    """Default small 3-level model for tests."""
    cfg = dict(
        vocab="abcdef .!",
        levels=[
            LevelConfig(dim=10, window=8, chunk_size=4),
            LevelConfig(dim=8,  window=6, chunk_size=3),
            LevelConfig(dim=6,  window=4, chunk_size=0),  # top
        ],
        regularization=1e-4,
        seed=0,
    )
    cfg.update(overrides)
    return HierarchicalLanguageModel(**cfg)


# ---------------------------------------------------------------------------
# LevelConfig validation
# ---------------------------------------------------------------------------

class TestLevelConfig:
    def test_defaults(self):
        c = LevelConfig(dim=8, window=4)
        assert c.chunk_size == 0
        assert c.decay == 0.7

    def test_rejects_invalid(self):
        with pytest.raises(ValueError):
            LevelConfig(dim=0, window=4)
        with pytest.raises(ValueError):
            LevelConfig(dim=8, window=0)
        with pytest.raises(ValueError):
            LevelConfig(dim=8, window=4, chunk_size=-1)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_rejects_empty_levels(self):
        with pytest.raises(ValueError):
            HierarchicalLanguageModel(vocab="abc", levels=[])

    def test_builds_all_levels(self):
        m = _tiny_3level()
        assert len(m.levels) == 3
        assert [L.dim for L in m.levels] == [10, 8, 6]

    def test_top_level_is_top(self):
        m = _tiny_3level()
        assert m.levels[-1].is_top
        assert not m.levels[0].is_top
        assert not m.levels[1].is_top

    def test_projection_shapes(self):
        m = _tiny_3level()
        # up[k]: level k -> level k+1
        assert m._up_projs[0].shape == (10, 8)
        assert m._up_projs[1].shape == (8, 6)
        # down[k]: level k+1 -> level k
        assert m._down_projs[0].shape == (8, 10)
        assert m._down_projs[1].shape == (6, 8)


# ---------------------------------------------------------------------------
# Chunk propagation — the structural heart
# ---------------------------------------------------------------------------

class TestChunkPropagation:
    def test_sub_chunk_does_not_propagate(self):
        m = _tiny_3level()  # L0 chunk=4
        for _ in range(3):
            m.ingest_token(2, learn=False)
        # L1 should still be empty
        assert m.levels[1].wm.filled == 0

    def test_exact_chunk_triggers_push(self):
        m = _tiny_3level()  # L0 chunk=4
        for _ in range(4):
            m.ingest_token(2, learn=False)
        assert m.levels[1].wm.filled == 1
        # L0's per-chunk counter resets
        assert m.levels[0].tokens_since_chunk == 0

    def test_nested_chunk_boundary_fires_top(self):
        m = _tiny_3level()  # L0 chunk=4, L1 chunk=3 → top updates every 12 tokens
        for _ in range(11):
            m.ingest_token(2, learn=False)
        assert m.levels[2].wm.filled == 0
        for _ in range(1):
            m.ingest_token(2, learn=False)
        # 12 tokens: L1 has 3 summaries → push to L2
        assert m.levels[2].wm.filled == 1

    def test_top_level_never_propagates(self):
        m = _tiny_3level()
        # Ingest enough tokens to hypothetically push through a fourth level
        for _ in range(50):
            m.ingest_token(2, learn=False)
        # No exception; top level is intact
        assert m.levels[2].wm.filled <= m.levels[2].window

    def test_learn_flag_affects_sample_counts(self):
        """learn=False must not add to core sample counts at any level."""
        m = _tiny_3level()
        for _ in range(12):
            m.ingest_token(2, learn=False)
        assert all(L.core.n_samples == 0 for L in m.levels)

        for _ in range(12):
            m.ingest_token(2, learn=True)
        assert m.levels[0].core.n_samples == 12
        assert m.levels[1].core.n_samples >= 1  # at least one chunk boundary
        assert m.levels[2].core.n_samples >= 0


# ---------------------------------------------------------------------------
# Training: teach_pair primes input and learns on output
# ---------------------------------------------------------------------------

class TestTeachPair:
    def test_surface_samples_only_from_output(self):
        m = _tiny_3level()
        # "abc" (3 chars input) + "de" (2 chars output) + eos = 3 surface samples
        m.teach_pair("abc", "de")
        # Only output-side ingestions add surface samples.
        # "de" is 2 tokens + 1 eos = 3.
        assert m.levels[0].core.n_samples == 3

    def test_rejects_zero_epochs(self):
        m = _tiny_3level()
        with pytest.raises(ValueError):
            m.teach_pairs([("a", "b")], epochs=0)

    def test_multiple_epochs_accumulate(self):
        m = _tiny_3level()
        m.teach_pairs([("a", "b")], epochs=5)
        # "b" + eos = 2 surface tokens per pair; 5 epochs = 10
        assert m.levels[0].core.n_samples == 10


# ---------------------------------------------------------------------------
# Teach text / corpus absorption
# ---------------------------------------------------------------------------

class TestTeachText:
    def test_returns_token_count(self):
        m = _tiny_3level()
        n = m.teach_text("abc abc")
        assert n == len("abc abc")

    def test_accumulates_samples(self):
        m = _tiny_3level()
        m.teach_text("a" * 20)
        assert m.levels[0].core.n_samples == 20
        # L0 chunk=4 → 5 pushes to L1
        assert m.levels[1].core.n_samples == 5


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

class TestGeneration:
    def test_returns_hierarchical_result(self):
        m = _tiny_3level()
        res = m.generate("a", max_len=3)
        assert isinstance(res, HierarchicalResult)
        assert len(res.omegas) == 3
        assert len(res.phis) == 3

    def test_memorises_simple_pair(self):
        m = _tiny_3level()
        m.teach_pairs([("a", "bc")], epochs=40)
        res = m.generate("a", max_len=5, temperature=0.0)
        assert res.text.startswith("b")

    def test_greedy_is_deterministic(self):
        m = _tiny_3level()
        m.teach_pairs([("a", "bc")], epochs=20)
        r1 = m.generate("a", max_len=5, temperature=0.0)
        r2 = m.generate("a", max_len=5, temperature=0.0)
        assert r1.text == r2.text

    def test_sampling_seed_reproducible(self):
        m = _tiny_3level()
        m.teach_pairs([("a", "bc")], epochs=20)
        r1 = m.generate("a", max_len=5, temperature=1.0, seed=7)
        r2 = m.generate("a", max_len=5, temperature=1.0, seed=7)
        assert r1.text == r2.text

    def test_min_confidence_can_stop(self):
        m = _tiny_3level()
        res = m.generate("a", max_len=10, min_confidence=0.99)
        assert res.stopped_reason == "low_confidence"
        assert res.text == ""

    def test_rejects_invalid_params(self):
        m = _tiny_3level()
        with pytest.raises(ValueError):
            m.generate("a", temperature=-0.1)
        with pytest.raises(ValueError):
            m.generate("a", temperature=0.5, top_k=0)


# ---------------------------------------------------------------------------
# Forget + reset
# ---------------------------------------------------------------------------

class TestForgetReset:
    def test_forget_wipes_all_levels(self):
        m = _tiny_3level()
        m.teach_pairs([("a", "bc")], epochs=20)
        m.forget(0.0)
        for L in m.levels:
            expected = np.eye(L.core.ld) * L.core.regularization
            assert np.allclose(L.core._G, expected)

    def test_reset_clears_counters(self):
        m = _tiny_3level()
        m.teach_pairs([("a", "b")], epochs=5)
        assert m.report().pairs_taught == 5
        m.reset()
        r = m.report()
        assert r.pairs_taught == 0
        assert r.tokens_seen == 0
        assert all(s == 0 for s in r.n_samples)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

class TestReport:
    def test_report_structure(self):
        m = _tiny_3level()
        m.teach_pairs([("a", "b")], epochs=3)
        r = m.report()
        assert isinstance(r, HierarchicalReport)
        assert r.n_levels == 3
        assert r.dims == [10, 8, 6]
        assert len(r.lifted_dims) == 3
        assert len(r.omegas) == 3

    def test_total_matrix_elements_matches(self):
        m = _tiny_3level()
        elems = sum(3 * L.core.ld ** 2 for L in m.levels)
        assert m.total_matrix_elements == elems
        assert m.report().total_matrix_elements == elems


# ---------------------------------------------------------------------------
# Long-context signal: top-level WM accumulates across many tokens
# ---------------------------------------------------------------------------

class TestLongContextReach:
    def test_top_level_accumulates_over_long_stream(self):
        """Ingesting many tokens must propagate into every level."""
        m = _tiny_3level()  # L0 chunk=4, L1 chunk=3 → L2 push per 12 tokens
        # 60 tokens → 60/4 = 15 pushes to L1 → 15/3 = 5 pushes to L2
        for _ in range(60):
            m.ingest_token(2, learn=True)
        assert m.levels[2].wm.filled == min(
            5, m.levels[2].window
        )

    def test_pyramid_geometry(self):
        """With chunk factors c1 c2, L2 sees one summary per (c1*c2) tokens."""
        m = HierarchicalLanguageModel(
            vocab="abc",
            levels=[
                LevelConfig(dim=6, window=10, chunk_size=5),
                LevelConfig(dim=4, window=10, chunk_size=4),
                LevelConfig(dim=4, window=10, chunk_size=0),
            ],
            regularization=1e-4,
            seed=0,
        )
        # Every 20 tokens -> 1 L2 sample
        for _ in range(20):
            m.ingest_token(2, learn=True)
        assert m.levels[2].wm.filled == 1
        for _ in range(20):
            m.ingest_token(2, learn=True)
        assert m.levels[2].wm.filled == 2
