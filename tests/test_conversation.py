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
    vocab_from_texts,
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

    def test_from_texts_builds_union_vocab(self):
        tok = CharTokenizer.from_texts(["hello", "world!", "hello world"])
        # 2 specials + unique chars from 'hello world!'
        assert tok.vocab_size == 2 + len(set("hello world!"))

    def test_from_texts_handles_korean(self):
        tok = CharTokenizer.from_texts(["안녕하세요", "반갑습니다"])
        for ch in "안녕하세요반갑습니다":
            assert ch in tok.tok_to_id

    def test_korean_roundtrip(self):
        tok = CharTokenizer.from_texts(["안녕 반가워요"])
        ids = tok.encode("안녕 반가워요")
        assert tok.decode(ids) == "안녕 반가워요"


class TestVocabFromTexts:
    def test_preserves_first_seen_order(self):
        v = vocab_from_texts(["cba", "ab"])
        # 'c' seen first, then 'b', then 'a'; 'ab' adds nothing new
        assert v == "cba"

    def test_handles_mixed_scripts(self):
        v = vocab_from_texts(["hi안녕"])
        assert set(v) == set("hi안녕")

    def test_empty(self):
        assert vocab_from_texts([]) == ""
        assert vocab_from_texts([""]) == ""


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

# ---------------------------------------------------------------------------
# Animal-style continual learning: converse() + forgetting
# ---------------------------------------------------------------------------

def _mean_confidence(chat: ConversationalAxol, text_in: str, max_len: int = 4) -> float:
    """Helper: average per-token cosine confidence for ``text_in``'s response."""
    _, confs = chat.respond_with_confidence(text_in, max_len=max_len)
    if not confs:
        return 0.0
    return float(sum(confs) / len(confs))


class TestConverseRealtime:
    def test_converse_returns_reflex_before_learning(self):
        """Response is produced from the *current* intuition (not post-teach)."""
        chat = ConversationalAxol(
            vocab="ab", embed_dim=10, window=4,
            forgetting_factor=1.0, regularization=1e-4,
        )
        # Prime it with 'a' -> 'b'
        for _ in range(30):
            chat.teach("a", "b")
        # Now converse with a teaching signal that would *change* the mapping.
        # The returned response must reflect the BEFORE state.
        r = chat.converse("a", text_out="a", learn=True)
        assert r.startswith("b"), f"expected reflex 'b', got {r!r}"

    def test_converse_learns_when_text_out_given(self):
        """With text_out, converse should accumulate samples."""
        chat = ConversationalAxol(
            vocab="ab", embed_dim=8, window=3,
            forgetting_factor=1.0, regularization=1e-3,
        )
        assert chat.intuition.n_samples == 0
        chat.converse("a", text_out="b", learn=True)
        assert chat.intuition.n_samples > 0

    def test_converse_without_text_out_does_not_learn(self):
        chat = ConversationalAxol(vocab="ab", embed_dim=8)
        chat.converse("a", text_out=None)
        assert chat.intuition.n_samples == 0

    def test_converse_learn_false_disables_learning(self):
        chat = ConversationalAxol(vocab="ab", embed_dim=8)
        chat.converse("a", text_out="b", learn=False)
        assert chat.intuition.n_samples == 0


class TestAcquisitionForgettingRelearn:
    """Pavlovian acquisition/extinction/re-acquisition with ``converse``."""

    def _make(self) -> ConversationalAxol:
        return ConversationalAxol(
            vocab="abcde.! ",
            embed_dim=16,
            window=6,
            forgetting_factor=1.0,      # decay only via explicit forget
            regularization=1e-4,
            seed=0,
        )

    def test_acquisition_raises_confidence(self):
        chat = self._make()
        c_before = _mean_confidence(chat, "a")
        for _ in range(40):
            chat.converse("a", text_out="b")
        c_after = _mean_confidence(chat, "a")
        assert c_after > c_before + 0.2, (
            f"confidence did not rise enough: {c_before:.3f} -> {c_after:.3f}"
        )

    def test_time_decay_lowers_confidence(self):
        chat = self._make()
        for _ in range(40):
            chat.converse("a", text_out="b")
        c_learned = _mean_confidence(chat, "a")

        # Long time passes with no new experience
        chat.forget_by_time(elapsed=100.0, half_life=10.0)  # 10 half-lives
        c_decayed = _mean_confidence(chat, "a")

        assert c_decayed < c_learned, (
            f"decay did not weaken confidence: {c_learned:.3f} -> {c_decayed:.3f}"
        )

    def test_full_forget_collapses_behaviour(self):
        chat = self._make()
        for _ in range(40):
            chat.converse("a", text_out="b")
        chat.forget(0.0)
        # After total forgetting the learned operator is ~ 0 -> predict ~ 0
        out, confs = chat.respond_with_confidence("a", max_len=3)
        # Confidence should be ~0 because predicted vector is near zero
        assert max(confs) < 0.05

    def test_relearn_after_forget_recovers(self):
        """After total wipe, re-teaching should restore the association."""
        chat = self._make()
        for _ in range(40):
            chat.converse("a", text_out="b")
        chat.forget(0.0)
        for _ in range(40):
            chat.converse("a", text_out="b")
        out = chat.respond("a", max_len=2)
        assert out.startswith("b")

    def test_integrated_scenario_acquire_decay_reacquire(self):
        """End-to-end story:
            1) acquire  -> confidence high
            2) time decays it to near zero
            3) reacquire -> confidence high again
        """
        chat = self._make()

        for _ in range(40):
            chat.converse("a", text_out="b")
        c1 = _mean_confidence(chat, "a")

        chat.forget_by_time(elapsed=200.0, half_life=5.0)
        c2 = _mean_confidence(chat, "a")

        for _ in range(40):
            chat.converse("a", text_out="b")
        c3 = _mean_confidence(chat, "a")

        assert c1 > 0.2
        assert c2 < c1
        assert c3 > c2


class TestConverseWithAutoDecay:
    """``converse`` can apply time-based decay before each turn."""

    def test_elapsed_time_decays_between_turns(self):
        chat = ConversationalAxol(
            vocab="ab", embed_dim=12, window=4,
            forgetting_factor=1.0, regularization=1e-3,
        )
        for _ in range(30):
            chat.converse("a", text_out="b")
        c_before = _mean_confidence(chat, "a")

        # Simulate "the animal slept for 50 time units"
        chat.converse("a", text_out=None, elapsed_time=50.0, half_life=5.0)
        c_after = _mean_confidence(chat, "a")
        assert c_after < c_before


# ---------------------------------------------------------------------------
# Ambiguity still lowers Omega (unchanged from previous suite)
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
