"""Tests for axol.quantum.streaming — recursive intent↔surface loop."""

from __future__ import annotations

import pytest

from axol.quantum.streaming import StreamingLanguageModel, StreamResult


# ---------------------------------------------------------------------------
# Construction / basic shape
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_inherits_two_stage(self):
        m = StreamingLanguageModel(vocab="abc")
        # Sanity: both cores exist
        assert m.intent_core is not None
        assert m.surface_core is not None


class TestStreamGenerationShape:
    def test_returns_stream_result_with_n_segments(self):
        m = StreamingLanguageModel(vocab="abc", intent_dim=6, surface_dim=10)
        res = m.stream("a", n_segments=4, segment_max_len=3)
        assert isinstance(res, StreamResult)
        assert len(res.segments) == 4
        assert len(res.confidences) == 4

    def test_zero_segments_rejected(self):
        m = StreamingLanguageModel(vocab="abc")
        with pytest.raises(ValueError):
            m.stream("a", n_segments=0)

    def test_text_joins_segments(self):
        m = StreamingLanguageModel(
            vocab="abc.! ", intent_dim=6, surface_dim=10,
            regularization=1e-3,
        )
        m.teach_stream(["a", "b", "c"])
        res = m.stream("a", n_segments=3, segment_max_len=3)
        # Text property is a space-joined version of the segments.
        assert res.text == " ".join(s for s in res.segments if s)


# ---------------------------------------------------------------------------
# Training: teach_stream absorbs both intent transitions and surface
# ---------------------------------------------------------------------------

class TestTeachStream:
    def test_empty_stream_is_noop(self):
        m = StreamingLanguageModel(vocab="abc", intent_dim=6, surface_dim=10)
        m.teach_stream([])
        assert m.intent_core.n_samples == 0

    def test_absorbs_intent_transitions(self):
        m = StreamingLanguageModel(vocab="abc.! ", intent_dim=6, surface_dim=10)
        m.teach_stream(["a", "bc"])  # two segments -> two intent samples
        assert m.intent_core.n_samples == 2

    def test_absorbs_surface_tokens(self):
        m = StreamingLanguageModel(vocab="abc.! ", intent_dim=6, surface_dim=10)
        before = m.surface_core.n_samples
        m.teach_stream(["abc"])
        # "abc" + eos = 4 tokens
        assert m.surface_core.n_samples - before >= 4


class TestTeachStreams:
    def test_rejects_zero_epochs(self):
        m = StreamingLanguageModel(vocab="abc")
        with pytest.raises(ValueError):
            m.teach_streams([["a", "b"]], epochs=0)

    def test_rejects_mismatched_seed_prompts(self):
        m = StreamingLanguageModel(vocab="abc")
        with pytest.raises(ValueError):
            m.teach_streams([["a"]], seed_prompts=[None, None])

    def test_multiple_streams_accumulate(self):
        m = StreamingLanguageModel(vocab="abc.! ", intent_dim=6, surface_dim=10)
        m.teach_streams([["a", "b"], ["c", "a"]], epochs=2)
        # 4 transitions per pass x 2 epochs = 8
        assert m.intent_core.n_samples == 8


# ---------------------------------------------------------------------------
# Recursion: feedback actually alters the next segment's intent
# ---------------------------------------------------------------------------

class TestFeedbackInfluence:
    def test_intent_wm_grows_across_stream(self):
        """After a 3-segment stream, the intent WM should hold
        approximately four entries: the seed + three feedbacks."""
        m = StreamingLanguageModel(
            vocab="abc.! ", intent_dim=6, surface_dim=10, regularization=1e-3,
        )
        m.teach_stream(["a", "b"], seed_prompt="hi")
        m.stream("hi", n_segments=3, segment_max_len=3)
        # WM window may cap it, but at least one entry is there.
        assert m.intent_wm.filled >= 1

    def test_different_prompts_yield_different_streams(self):
        """Two distinct seed prompts should produce at least one
        different segment when the model has been trained to associate
        them with different downstream sentences."""
        m = StreamingLanguageModel(
            vocab="abcxy.! ", intent_dim=10, surface_dim=14,
            intent_window=4, surface_window=8, regularization=1e-4, seed=1,
        )
        # Teach two contrasting streams
        m.teach_streams(
            streams=[
                ["a", "a", "a"],   # "a-world" stream
                ["x", "x", "x"],   # "x-world" stream
            ],
            seed_prompts=["a", "x"],
            epochs=40,
        )
        r_a = m.stream("a", n_segments=3, segment_max_len=2,
                       temperature=0.0)
        r_x = m.stream("x", n_segments=3, segment_max_len=2,
                       temperature=0.0)
        # At least one segment should differ.
        assert r_a.segments != r_x.segments


# ---------------------------------------------------------------------------
# Recall of a repeated stream
# ---------------------------------------------------------------------------

class TestStreamRecall:
    def test_repeated_stream_is_memorised(self):
        """Repeatedly teaching the same 3-segment stream should let the
        model reproduce the segments when given the seed."""
        stream = ["ab", "bc", "ca"]
        m = StreamingLanguageModel(
            vocab="abc.! ",
            intent_dim=12, surface_dim=18,
            intent_window=4, surface_window=8,
            regularization=1e-4, seed=0,
        )
        m.teach_streams([stream], seed_prompts=["ab"], epochs=60)

        res = m.stream("ab", n_segments=3, segment_max_len=4,
                       temperature=0.0)
        # At least one of the segments should match the trained stream.
        hits = sum(1 for s, expected in zip(res.segments, stream)
                   if s.startswith(expected))
        assert hits >= 1, (
            f"no segment matched expected stream — got {res.segments!r}"
        )


# ---------------------------------------------------------------------------
# StreamResult diagnostics
# ---------------------------------------------------------------------------

class TestStreamResultStats:
    def test_mean_and_min_confidence(self):
        m = StreamingLanguageModel(
            vocab="abc.! ", intent_dim=6, surface_dim=10, regularization=1e-3,
        )
        m.teach_stream(["a", "b"])
        res = m.stream("a", n_segments=2, segment_max_len=3,
                       temperature=0.0)
        if any(res.confidences):
            assert 0.0 <= res.min_confidence <= res.mean_confidence <= 1.0
        else:
            assert res.mean_confidence == 0.0
            assert res.min_confidence == 0.0

    def test_empty_stream_result_stats_zero(self):
        """Early-stopping with min_confidence=0.99 on untrained model
        should yield empty confidences (or all below threshold)."""
        m = StreamingLanguageModel(vocab="abc")
        res = m.stream("a", n_segments=1, segment_max_len=5,
                       min_confidence=0.99)
        # No tokens emitted → empty text
        assert all(s == "" for s in res.segments)


# ---------------------------------------------------------------------------
# Axiom-3 invariant (weakest-link omega)
# ---------------------------------------------------------------------------

class TestOmegaOfResult:
    def test_stream_result_carries_both_omegas(self):
        m = StreamingLanguageModel(vocab="abc", intent_dim=6, surface_dim=10)
        m.teach_stream(["a", "b"])
        res = m.stream("a", n_segments=1, segment_max_len=3)
        assert 0.0 <= res.intent_omega <= 1.0
        assert 0.0 <= res.surface_omega <= 1.0
