"""Tests for axol.quantum.conversation — dual-layer conversational agent.

Covers each subsystem in isolation plus end-to-end teach/respond behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from axol.quantum.conversation import (
    CharTokenizer,
    ConversationalAxol,
    ConversationReport,
    Verbalizer,
    WorkingMemory,
)


# ---------------------------------------------------------------------------
# CharTokenizer
# ---------------------------------------------------------------------------

class TestCharTokenizer:
    def test_special_tokens_reserved(self):
        tok = CharTokenizer("abc")
        assert tok.tok_to_id[CharTokenizer.UNK] == 0
        assert tok.tok_to_id[CharTokenizer.EOS] == 1

    def test_roundtrip(self):
        tok = CharTokenizer("hello, world!")
        ids = tok.encode("hello")
        assert tok.decode(ids) == "hello"

    def test_unknown_char_mapped_to_unk(self):
        tok = CharTokenizer("abc")
        ids = tok.encode("abz")
        # 'z' not in vocab -> unk_id
        assert ids[2] == tok.unk_id

    def test_decode_strips_specials(self):
        tok = CharTokenizer("hi")
        # Inject an explicit eos id into the id stream
        ids = tok.encode("hi") + [tok.eos_id]
        assert tok.decode(ids) == "hi"

    def test_deduplicates_vocab(self):
        tok = CharTokenizer("aabbcc")
        # 2 specials + 3 unique chars
        assert tok.vocab_size == 5


# ---------------------------------------------------------------------------
# Verbalizer
# ---------------------------------------------------------------------------

class TestVerbalizer:
    def test_embedding_rows_normalised(self):
        v = Verbalizer(vocab_size=10, dim=8, seed=1)
        norms = np.linalg.norm(v.E, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_encode_decode_identity(self):
        v = Verbalizer(vocab_size=12, dim=16, seed=2)
        for tid in range(v.vocab_size):
            emb = v.encode_id(tid)
            recovered, sim = v.decode_vec(emb)
            assert recovered == tid
            assert sim > 0.99  # cosine against itself ~1.0

    def test_decode_zero_vec_returns_zero_sim(self):
        v = Verbalizer(vocab_size=5, dim=4, seed=3)
        tid, sim = v.decode_vec(np.zeros(4, dtype=np.float32))
        assert sim == 0.0

    def test_decode_distribution_sums_to_one(self):
        v = Verbalizer(vocab_size=6, dim=4, seed=4)
        probs = v.decode_distribution(v.encode_id(2), temperature=1.0)
        assert probs.shape == (6,)
        assert np.isclose(probs.sum(), 1.0, atol=1e-5)

    def test_temperature_zero_rejected(self):
        v = Verbalizer(vocab_size=5, dim=4, seed=5)
        with pytest.raises(ValueError):
            v.decode_distribution(v.encode_id(0), temperature=0.0)

    def test_rejects_wrong_dim(self):
        v = Verbalizer(vocab_size=4, dim=8, seed=6)
        with pytest.raises(ValueError):
            v.decode_vec(np.zeros(7))


# ---------------------------------------------------------------------------
# WorkingMemory
# ---------------------------------------------------------------------------

class TestWorkingMemory:
    def test_empty_context_is_zero(self):
        wm = WorkingMemory(dim=4, window=3)
        ctx = wm.context()
        assert np.allclose(ctx, 0.0)

    def test_window_bounds(self):
        wm = WorkingMemory(dim=2, window=3)
        for i in range(5):
            wm.add(np.array([float(i), 0.0], dtype=np.float32))
        assert wm.filled == 3  # only last 3 retained

    def test_most_recent_weighted_most(self):
        """decay=0.5 means the last token should dominate the context."""
        wm = WorkingMemory(dim=2, window=4, decay=0.5)
        wm.add(np.array([1.0, 0.0], dtype=np.float32))
        wm.add(np.array([0.0, 0.0], dtype=np.float32))
        wm.add(np.array([0.0, 0.0], dtype=np.float32))
        wm.add(np.array([10.0, 0.0], dtype=np.float32))  # most recent, big
        ctx = wm.context()
        # weight on last item = 1 / (0.125+0.25+0.5+1) = 0.533 -> 10*0.533=5.33
        # first item weight = 0.125 / 1.875 = 0.0667 -> 1*0.0667=0.067
        assert ctx[0] > 5.0

    def test_reset_clears(self):
        wm = WorkingMemory(dim=2, window=3)
        wm.add(np.ones(2, dtype=np.float32))
        wm.reset()
        assert wm.filled == 0
        assert np.allclose(wm.context(), 0.0)

    def test_rejects_invalid_params(self):
        with pytest.raises(ValueError):
            WorkingMemory(dim=0, window=3)
        with pytest.raises(ValueError):
            WorkingMemory(dim=4, window=0)
        with pytest.raises(ValueError):
            WorkingMemory(dim=4, window=3, decay=1.5)


# ---------------------------------------------------------------------------
# ConversationalAxol — end to end
# ---------------------------------------------------------------------------

class TestConversationalAxol:
    def test_construction_defaults(self):
        chat = ConversationalAxol(vocab="hello")
        assert chat.verbalizer.dim == 8
        assert chat.tokenizer.vocab_size >= 2 + 4  # specials + 'h','e','l','o'

    def test_teach_updates_intuition(self):
        chat = ConversationalAxol(vocab="abc", embed_dim=6)
        assert chat.intuition.n_samples == 0
        chat.teach("a", "b")
        assert chat.intuition.n_samples > 0
        assert chat.report().pairs_taught == 1

    def test_respond_empty_before_teaching(self):
        """Untrained model should produce something short (no crash)."""
        chat = ConversationalAxol(vocab="abcdef", embed_dim=6)
        out = chat.respond("a", max_len=3)
        assert isinstance(out, str)
        assert len(out) <= 3

    def test_memorises_single_pair(self):
        """After enough repetitions of a single (in, out), generation matches."""
        chat = ConversationalAxol(
            vocab="abc.,! ?",
            embed_dim=12,
            window=6,
            regularization=1e-4,
        )
        pair = ("a", "b")
        # Re-present many times; additive moments strengthen the mapping.
        for _ in range(30):
            chat.teach(*pair)
        out = chat.respond("a", max_len=4)
        # First produced char should be 'b'
        assert out.startswith("b"), f"got {out!r}"

    def test_distinguishes_two_inputs(self):
        """Given two distinct pairs, model should route inputs correctly."""
        chat = ConversationalAxol(
            vocab="xyzXYZ12 ",
            embed_dim=16,
            window=6,
            regularization=1e-4,
        )
        pairs = [("x", "1"), ("y", "2")]
        for _ in range(30):
            for tin, tout in pairs:
                chat.teach(tin, tout)
        out_x = chat.respond("x", max_len=2)
        out_y = chat.respond("y", max_len=2)
        assert out_x != out_y
        assert out_x.startswith("1") or out_y.startswith("2")

    def test_confidence_reported(self):
        chat = ConversationalAxol(vocab="abc", embed_dim=8)
        for _ in range(10):
            chat.teach("a", "b")
        out, confidences = chat.respond_with_confidence("a", max_len=3)
        assert len(confidences) >= 1
        # Post-training cosine confidence for a memorised pair should be non-trivial.
        assert confidences[0] > 0.3

    def test_report_structure(self):
        chat = ConversationalAxol(vocab="abc", embed_dim=6)
        chat.teach("a", "b")
        r = chat.report()
        assert isinstance(r, ConversationReport)
        assert r.pairs_taught == 1
        assert r.tokens_seen >= 1
        assert r.embed_dim == 6
        assert 0.0 <= r.intuition.omega <= 1.0

    def test_reset_clears_intuition(self):
        chat = ConversationalAxol(vocab="abc", embed_dim=6)
        for _ in range(5):
            chat.teach("a", "b")
        assert chat.intuition.n_samples > 0
        chat.reset()
        assert chat.intuition.n_samples == 0
        assert chat.report().pairs_taught == 0

    def test_min_confidence_gate_stops_generation(self):
        """High min_confidence with an untrained model should yield empty output."""
        chat = ConversationalAxol(vocab="abcdef", embed_dim=6)
        out = chat.respond("a", max_len=10, min_confidence=0.99)
        # Untrained predictions are unlikely to clear cos>0.99 against any token.
        assert out == ""

    def test_eos_terminates_response(self):
        """If the model learns to emit <eos> after 'b', respond should stop."""
        chat = ConversationalAxol(
            vocab="ab", embed_dim=10, window=4, regularization=1e-4
        )
        # Teach with append_eos=True (default): context for 'a' -> 'b' -> <eos>
        for _ in range(40):
            chat.teach("a", "b", append_eos=True)
        out = chat.respond("a", max_len=20, stop_on_eos=True)
        # Should not run to full 20 characters; 'b' then stop is typical.
        assert len(out) < 20


# ---------------------------------------------------------------------------
# Axiom-3 invariant: ambiguous mappings surface as low Omega
# ---------------------------------------------------------------------------

class TestAmbiguityLowersOmega:
    def test_contradictory_pairs_reduce_cohesion(self):
        """Teaching (a -> b) then (a -> c) repeatedly is internally
        contradictory.  The intuition operator should end up with lower
        Omega than when trained on a single consistent mapping."""
        consistent = ConversationalAxol(
            vocab="abc", embed_dim=10, window=4, regularization=1e-3
        )
        ambiguous = ConversationalAxol(
            vocab="abc", embed_dim=10, window=4, regularization=1e-3
        )
        for _ in range(40):
            consistent.teach("a", "b")
            ambiguous.teach("a", "b")
            ambiguous.teach("a", "c")
        # The ambiguous agent sees twice as many samples per iteration, but
        # with contradictory signals.  Its intuition max-Lyapunov should be
        # no lower (typically higher), yielding Omega <= consistent's.
        assert ambiguous.omega <= consistent.omega + 1e-6
