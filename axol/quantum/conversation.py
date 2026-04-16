"""Conversational extension for AXOL — dual-layer architecture.

Human-learning analogy
----------------------
System 1  (intuition / instinct)      : ``IntuitionCore`` = OnlineLearner
System 2  (verbalisation / organisation): ``Verbalizer`` (token <-> vector)
Working memory (hippocampus-like)      : ``WorkingMemory`` (recent context buffer)
Discrete surface (phonology-like)      : ``CharTokenizer`` (text <-> ids)

Data flow
---------
Teaching (System 1 absorbs; System 2 supervises):
    (text_in, text_out)
        -> tokenize both
        -> verbalize each token into the semantic phase space
        -> walk through output, at each step accumulate context and push
           (context, next_embedding) into IntuitionCore

Responding (System 1 drives; System 2 names the result):
    text_in
        -> tokenize + verbalize
        -> WorkingMemory absorbs the sequence
        -> loop:
             context -> IntuitionCore.predict -> semantic vector
             Verbalizer.decode -> next token
             WorkingMemory absorbs it
        -> tokens decoded back into text

Axiom alignment
---------------
* Axiom 1 (no time axis): IntuitionCore updates are still closed-form per
  sample.  Autoregressive generation is *external* time, not internal
  iteration toward convergence.
* Axiom 2 (accuracy): Omega/Phi from the IntuitionCore surface how confident
  the system is in its next-token prediction.
* Axiom 3 (probability is the price): ambiguous contexts produce low Omega
  directly.
* Axiom 4 (production): char-level embedding + small dim keeps the model
  runnable on CPU; this module is a demonstrator, not a production LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from axol.core.types import FloatVec
from axol.quantum.online import LearningReport, OnlineLearner


# ---------------------------------------------------------------------------
# Vocab helper — works for any language / script, Korean included.
# ---------------------------------------------------------------------------

def vocab_from_texts(texts: Iterable[str]) -> str:
    """Return a string containing every unique character seen in ``texts``.

    Preserves first-seen order (useful for deterministic tokeniser IDs).
    Applicable to any script — Hangul, CJK, accented Latin, emoji — since
    the tokeniser operates on Unicode code points.
    """
    seen: dict[str, None] = {}
    for text in texts:
        for ch in text:
            if ch not in seen:
                seen[ch] = None
    return "".join(seen.keys())


# ---------------------------------------------------------------------------
# CharTokenizer — minimal character-level surface
# ---------------------------------------------------------------------------

class CharTokenizer:
    """Character-level tokeniser with a fixed vocabulary.

    Unknown characters are mapped to a reserved ``<unk>`` id; the optional
    ``<eos>`` marker lets generators signal "end of response".
    """

    UNK = "<unk>"
    EOS = "<eos>"

    def __init__(self, vocab: str) -> None:
        # Start with special tokens, then user-provided characters,
        # deduplicated while preserving order.
        seen: dict[str, None] = {}
        tokens: list[str] = [self.UNK, self.EOS]
        for ch in vocab:
            if ch not in seen and ch not in (self.UNK, self.EOS):
                seen[ch] = None
                tokens.append(ch)
        self.tokens = tokens
        self.tok_to_id = {t: i for i, t in enumerate(tokens)}
        self.id_to_tok = {i: t for i, t in enumerate(tokens)}

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def unk_id(self) -> int:
        return self.tok_to_id[self.UNK]

    @property
    def eos_id(self) -> int:
        return self.tok_to_id[self.EOS]

    def encode(self, text: str) -> list[int]:
        return [self.tok_to_id.get(ch, self.unk_id) for ch in text]

    def decode(self, ids: list[int]) -> str:
        parts: list[str] = []
        for i in ids:
            t = self.id_to_tok.get(int(i), self.UNK)
            if t in (self.UNK, self.EOS):
                continue
            parts.append(t)
        return "".join(parts)

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> "CharTokenizer":
        """Build a tokeniser whose vocab is the union of characters seen
        in ``texts`` — convenient for ad-hoc Korean / multilingual corpora.
        """
        return cls(vocab_from_texts(texts))


# ---------------------------------------------------------------------------
# Verbalizer — token <-> semantic vector (System 2)
# ---------------------------------------------------------------------------

class Verbalizer:
    """Bidirectional bridge between discrete tokens and the phase space.

    Uses a fixed random-orthogonal-ish embedding matrix (rows L2-normalised).
    For vocab_size << dim the rows are nearly orthogonal by the
    Johnson-Lindenstrauss lemma; for vocab_size > dim decoding becomes a
    cosine-nearest-neighbour search in a tightly packed sphere.

    Decoding is the System-2 act of naming what System 1 produced.
    """

    def __init__(self, vocab_size: int, dim: int, seed: int = 0) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.vocab_size = vocab_size
        self.dim = dim

        rng = np.random.default_rng(seed)
        E = rng.standard_normal((vocab_size, dim)).astype(np.float64)
        # L2-normalise rows so cosine similarity == dot product
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        E = E / norms
        self.E = E.astype(np.float32)

    def encode_id(self, tid: int) -> np.ndarray:
        """Token id -> semantic vector (dim,)."""
        return self.E[int(tid)].copy()

    def decode_vec(self, vec: np.ndarray) -> tuple[int, float]:
        """Semantic vector -> (best_token_id, cosine_similarity).

        Returns the nearest token in cosine similarity, together with that
        similarity so callers can gate on confidence.
        """
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        if v.size != self.dim:
            raise ValueError(f"vec must have length {self.dim}, got {v.size}")
        n = float(np.linalg.norm(v))
        if n < 1e-12:
            return 0, 0.0
        sims = self.E @ (v / n)
        tid = int(np.argmax(sims))
        return tid, float(sims[tid])

    def decode_distribution(self, vec: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Softmax distribution over tokens given a semantic vector.

        ``temperature=0`` degenerates to a one-hot argmax (not computed here;
        callers should use ``decode_vec`` for that path).
        """
        if temperature <= 0:
            raise ValueError("temperature must be > 0; use decode_vec for argmax")
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        if n < 1e-12:
            # Uniform distribution over vocab
            return np.full(self.vocab_size, 1.0 / self.vocab_size, dtype=np.float32)
        sims = self.E @ (v / n)
        logits = sims / float(temperature)
        logits = logits - float(logits.max())  # numerical stability
        probs = np.exp(logits)
        probs /= probs.sum()
        return probs.astype(np.float32)


# ---------------------------------------------------------------------------
# WorkingMemory — recent-token aggregator (hippocampus-like)
# ---------------------------------------------------------------------------

class WorkingMemory:
    """Exponentially-weighted moving average over the last ``window`` tokens.

    Produces a fixed-dimension ``context`` vector from a variable-length
    history — the conversion ``variable sequence -> fixed vector`` that the
    IntuitionCore needs.

    ``decay`` controls how quickly older tokens fade.  ``decay=1.0`` gives a
    uniform average; ``decay=0.5`` emphasises the most recent token strongly.
    """

    def __init__(self, dim: int, window: int = 8, decay: float = 0.7) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if window <= 0:
            raise ValueError("window must be positive")
        if not (0.0 < decay <= 1.0):
            raise ValueError("decay must be in (0, 1]")
        self.dim = dim
        self.window = window
        self.decay = decay
        self._buffer: list[np.ndarray] = []

    def add(self, emb: np.ndarray) -> None:
        v = np.asarray(emb, dtype=np.float32).reshape(-1)
        if v.size != self.dim:
            raise ValueError(f"emb must have length {self.dim}, got {v.size}")
        self._buffer.append(v.copy())
        if len(self._buffer) > self.window:
            self._buffer.pop(0)

    def context(self) -> np.ndarray:
        """Fixed-dimension context vector summarising recent history."""
        if not self._buffer:
            return np.zeros(self.dim, dtype=np.float32)
        n = len(self._buffer)
        weights = np.array(
            [self.decay ** (n - 1 - i) for i in range(n)], dtype=np.float32
        )
        weights /= weights.sum()
        ctx = np.zeros(self.dim, dtype=np.float32)
        for w, e in zip(weights, self._buffer):
            ctx = ctx + w * e
        return ctx

    def reset(self) -> None:
        self._buffer.clear()

    @property
    def filled(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
# ConversationalAxol — orchestrator
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConversationReport:
    """Diagnostic snapshot of the whole conversational system."""

    vocab_size: int
    embed_dim: int
    window: int
    intuition: LearningReport
    tokens_seen: int = 0
    pairs_taught: int = 0


class ConversationalAxol:
    """End-to-end dual-layer conversational agent.

    The four subsystems play distinct roles:

    * ``tokenizer``  : discrete surface  — what to read/write.
    * ``verbalizer`` : System 2          — what each token *means*
                                           in the phase space.
    * ``working_memory``: hippocampus    — what the agent just saw.
    * ``intuition``  : System 1          — reflex: context -> next.

    Two primary verbs:
    ``teach(text_in, text_out)`` and ``respond(text_in)``.
    """

    DEFAULT_VOCAB = (
        # ASCII printable (space through ~), excluding controls
        " !\"#$%&'()*+,-./0123456789:;<=>?@"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
        "abcdefghijklmnopqrstuvwxyz{|}~"
        # Korean Hangul syllables: leave empty by default, user can pass custom
    )

    def __init__(
        self,
        vocab: str | None = None,
        embed_dim: int = 8,
        window: int = 8,
        decay: float = 0.7,
        forgetting_factor: float = 0.998,
        regularization: float = 1e-3,
        seed: int = 0,
        degree: int = 2,
    ) -> None:
        vocab_str = vocab if vocab is not None else self.DEFAULT_VOCAB
        self.tokenizer = CharTokenizer(vocab_str)
        self.verbalizer = Verbalizer(
            vocab_size=self.tokenizer.vocab_size, dim=embed_dim, seed=seed
        )
        self.working_memory = WorkingMemory(dim=embed_dim, window=window, decay=decay)
        self.intuition = OnlineLearner(
            dim=embed_dim,
            degree=degree,
            forgetting_factor=forgetting_factor,
            regularization=regularization,
        )
        self._tokens_seen = 0
        self._pairs_taught = 0

    # ------------------------------------------------------------------
    # Teaching: System 1 absorbs (context, next_embedding) pairs
    # ------------------------------------------------------------------

    def teach(self, text_in: str, text_out: str, append_eos: bool = True) -> None:
        """Learn from one (stimulus, response) pair.

        Uses *teacher forcing*: for each target token, compute the context
        from the current history (input + preceding output tokens) and push
        (context, target_embedding) into the IntuitionCore.
        """
        self.working_memory.reset()

        # Absorb the input, building up context -- no teaching signal here.
        for tid in self.tokenizer.encode(text_in):
            self.working_memory.add(self.verbalizer.encode_id(tid))

        # Target sequence, optionally terminated by <eos>.
        out_ids = self.tokenizer.encode(text_out)
        if append_eos:
            out_ids = out_ids + [self.tokenizer.eos_id]

        for tid in out_ids:
            ctx = self.working_memory.context()
            target_emb = self.verbalizer.encode_id(tid)
            self.intuition.observe_sample(ctx, target_emb)
            self.working_memory.add(target_emb)
            self._tokens_seen += 1

        self._pairs_taught += 1

    def teach_many(self, pairs: list[tuple[str, str]], epochs: int = 1) -> None:
        """Convenience: teach a list of (in, out) pairs.

        ``epochs`` here means *re-presentation passes*, not gradient epochs.
        Each pass is still a stream of closed-form rank-1 updates; repeating
        strengthens the moments via additive accumulation.
        """
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        for _ in range(epochs):
            for text_in, text_out in pairs:
                self.teach(text_in, text_out)

    # ------------------------------------------------------------------
    # Responding: System 1 drives, System 2 names each step
    # ------------------------------------------------------------------

    def respond(
        self,
        text_in: str,
        max_len: int = 64,
        stop_on_eos: bool = True,
        min_confidence: float = 0.0,
    ) -> str:
        """Generate a response to ``text_in``.

        Stops when: ``<eos>`` is produced (if enabled), ``max_len`` is reached,
        or the cosine confidence drops below ``min_confidence``.  The last
        gate is the verbalisation-layer analogue of "I'm not sure what word
        comes next, so I'll stop rather than guess" — Axiom 3 in action.
        """
        self.working_memory.reset()
        for tid in self.tokenizer.encode(text_in):
            self.working_memory.add(self.verbalizer.encode_id(tid))

        output_ids: list[int] = []
        for _ in range(max_len):
            ctx = self.working_memory.context()
            next_vec = self.intuition.predict(ctx)
            tid, sim = self.verbalizer.decode_vec(next_vec)
            if sim < min_confidence:
                break
            if stop_on_eos and tid == self.tokenizer.eos_id:
                break
            output_ids.append(tid)
            self.working_memory.add(self.verbalizer.encode_id(tid))

        return self.tokenizer.decode(output_ids)

    def respond_with_confidence(
        self,
        text_in: str,
        max_len: int = 64,
        stop_on_eos: bool = True,
    ) -> tuple[str, list[float]]:
        """Like ``respond`` but returns per-token cosine similarities.

        Useful for inspecting how confident the verbaliser was at each step.
        """
        self.working_memory.reset()
        for tid in self.tokenizer.encode(text_in):
            self.working_memory.add(self.verbalizer.encode_id(tid))

        output_ids: list[int] = []
        confidences: list[float] = []
        for _ in range(max_len):
            ctx = self.working_memory.context()
            next_vec = self.intuition.predict(ctx)
            tid, sim = self.verbalizer.decode_vec(next_vec)
            confidences.append(sim)
            if stop_on_eos and tid == self.tokenizer.eos_id:
                break
            output_ids.append(tid)
            self.working_memory.add(self.verbalizer.encode_id(tid))

        return self.tokenizer.decode(output_ids), confidences

    # ------------------------------------------------------------------
    # Real-time interaction (animal-style): respond AND learn in one turn
    # ------------------------------------------------------------------

    def converse(
        self,
        text_in: str,
        text_out: str | None = None,
        max_len: int = 64,
        stop_on_eos: bool = True,
        min_confidence: float = 0.0,
        learn: bool = True,
        elapsed_time: float = 0.0,
        half_life: float | None = None,
    ) -> str:
        """One interaction turn: reflex + (optional) reinforcement.

        Mirrors animal operant conditioning.  On each turn the agent:

        1. Decays prior memory by ``elapsed_time`` if ``half_life`` is given
           (time between turns passively weakens traces).
        2. Reacts to ``text_in`` using its current intuition -> response.
        3. If ``text_out`` is provided and ``learn`` is True, the pair
           ``(text_in, text_out)`` is pushed into the IntuitionCore as a
           single stream of rank-1 updates.  This is the reinforcement:
           the teacher signal strengthens the (stimulus, response) link.

        Returns the *reflexive* response (produced before learning), which
        is what the animal actually did before the teacher corrected it.
        """
        if half_life is not None and elapsed_time > 0.0:
            self.intuition.decay_by_time(elapsed_time, half_life)

        response = self.respond(
            text_in,
            max_len=max_len,
            stop_on_eos=stop_on_eos,
            min_confidence=min_confidence,
        )

        if learn and text_out is not None:
            self.teach(text_in, text_out)

        return response

    # ------------------------------------------------------------------
    # Forgetting (Ebbinghaus-style)
    # ------------------------------------------------------------------

    def forget(self, factor: float) -> None:
        """Uniformly decay the intuition's accumulated memory by ``factor``.

        ``factor=1.0`` keeps everything; ``factor=0.0`` wipes the learned
        operator while preserving the embedding and tokenizer.
        """
        self.intuition.decay(factor)

    def forget_by_time(self, elapsed: float, half_life: float) -> None:
        """Exponential memory decay proportional to elapsed time.

        ``factor = 0.5 ** (elapsed / half_life)`` — the Ebbinghaus curve.
        ``elapsed`` and ``half_life`` share arbitrary time units.
        """
        self.intuition.decay_by_time(elapsed, half_life)

    # ------------------------------------------------------------------
    # Diagnostics (Axiom 3)
    # ------------------------------------------------------------------

    @property
    def omega(self) -> float:
        """Cohesion of the learned intuition operator."""
        return self.intuition.omega

    @property
    def phi(self) -> float:
        """Clarity of the learned intuition operator."""
        return self.intuition.phi

    def report(self) -> ConversationReport:
        return ConversationReport(
            vocab_size=self.tokenizer.vocab_size,
            embed_dim=self.verbalizer.dim,
            window=self.working_memory.window,
            intuition=self.intuition.report(),
            tokens_seen=self._tokens_seen,
            pairs_taught=self._pairs_taught,
        )

    def reset(self) -> None:
        """Forget everything the intuition has learned; vocab/embeddings remain."""
        self.intuition.reset()
        self.working_memory.reset()
        self._tokens_seen = 0
        self._pairs_taught = 0
