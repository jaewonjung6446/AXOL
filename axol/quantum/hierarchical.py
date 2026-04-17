"""Hierarchical (multi-level) language model.

Generalises ``TwoStageLanguageModel``/``StreamingLanguageModel`` to N levels.
Each level has its own OnlineLearner (core), WorkingMemory, and Verbalizer
at its own dimension.  Chunk boundaries automatically propagate summaries
from a level to the one above it, forming a recursive hierarchy that
trades exact self-attention for log-depth context coverage.

Motivation
----------
Transformers access N tokens at O(N^2) cost.  If the N tokens are split
into K chunks, each level sees only O(K) items and a pyramid of log_c(N)
levels (each with chunk factor c) covers all tokens at O(N) total cost.
This module realises exactly that pyramid under the AXOL axioms.

Level conventions
-----------------
Level 0 is the *surface* (tokens).  Level ``len(levels)-1`` is the *top*
(most abstract, slowest-moving).  Every level has:

* ``dim``           semantic phase-space dimension
* ``window``        WorkingMemory window
* ``chunk_size``    how many same-level samples constitute one upward push
                    (ignored for the top level)

Propagation
-----------
On every token ingested at level 0:

1. Surface core learns ``context -> target_embedding`` (if ``learn=True``).
2. Token embedding is added to L0's WM.
3. If L0 has accumulated ``chunk_size`` tokens, its WM context is
   summarised, projected into L1's space, and appended to L1's WM.
4. At L1 the same rule fires: if L1 collected ``chunk_size`` summaries,
   push up to L2, and so on.

Learning at upper levels happens *before* the new summary is added, so
each level learns ``its_own_context -> next_summary_from_below``.

Conditioning
------------
During generation the current parent-level intent is projected down and
added to the child level's context.  Only the *immediate* parent is used
(chain-of-intent through the whole stack is a possible extension).

Axiom alignment
---------------
* Axiom 1: every per-level update stays closed-form rank-1.  Chunk
  propagation is an external branch, not an internal iteration.
* Axiom 3: each level exposes its own Omega/Phi so the stack can be
  diagnosed layer-by-layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from axol.quantum.conversation import CharTokenizer, Verbalizer, WorkingMemory
from axol.quantum.online import OnlineLearner


# ---------------------------------------------------------------------------
# Config & result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LevelConfig:
    """Per-level hyperparameters."""

    dim: int
    window: int
    chunk_size: int = 0          # 0 = top level (never propagates up)
    decay: float = 0.7

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.window <= 0:
            raise ValueError("window must be positive")
        if self.chunk_size < 0:
            raise ValueError("chunk_size must be >= 0 (0 means top level)")


@dataclass(frozen=True)
class HierarchicalResult:
    """Output of ``generate``."""

    text: str
    confidences: list[float]
    omegas: list[float]          # per level (index 0 = surface)
    phis: list[float]
    stopped_reason: str          # "eos" | "max_len" | "low_confidence"

    @property
    def surface_omega(self) -> float:
        return self.omegas[0] if self.omegas else 0.0

    @property
    def top_omega(self) -> float:
        return self.omegas[-1] if self.omegas else 0.0

    @property
    def mean_confidence(self) -> float:
        return (float(sum(self.confidences) / len(self.confidences))
                if self.confidences else 0.0)


@dataclass(frozen=True)
class HierarchicalReport:
    """Diagnostic snapshot of the whole stack."""

    n_levels: int
    dims: list[int]
    lifted_dims: list[int]
    windows: list[int]
    chunk_sizes: list[int]
    n_samples: list[int]         # per level
    omegas: list[float]
    phis: list[float]
    tokens_seen: int
    pairs_taught: int

    @property
    def total_matrix_elements(self) -> int:
        """G + H + B stored per level."""
        return sum(3 * ld * ld for ld in self.lifted_dims)


# ---------------------------------------------------------------------------
# Internal per-level state
# ---------------------------------------------------------------------------

@dataclass
class _Level:
    """Mutable container for one level's runtime state."""

    dim: int
    window: int
    chunk_size: int              # 0 if top
    core: OnlineLearner
    wm: WorkingMemory
    verb: Verbalizer
    tokens_since_chunk: int = 0  # counts ingestions since last upward push

    @property
    def is_top(self) -> bool:
        return self.chunk_size <= 0


# ---------------------------------------------------------------------------
# HierarchicalLanguageModel
# ---------------------------------------------------------------------------

class HierarchicalLanguageModel:
    """N-level hierarchical language model with automatic chunk propagation."""

    DEFAULT_VOCAB = (
        "\n !\"#$%&'()*+,-./0123456789:;<=>?@"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
        "abcdefghijklmnopqrstuvwxyz{|}~"
    )

    def __init__(
        self,
        vocab: str | None = None,
        levels: list[LevelConfig] | None = None,
        forgetting_factor: float = 1.0,
        regularization: float = 1e-4,
        degree: int = 2,
        seed: int = 0,
    ) -> None:
        if levels is None or len(levels) < 1:
            raise ValueError("levels must be a non-empty list of LevelConfig")

        self.vocab = vocab if vocab is not None else self.DEFAULT_VOCAB
        self.tokenizer = CharTokenizer(self.vocab)

        # Build every level.
        self.levels: list[_Level] = []
        for i, cfg in enumerate(levels):
            core = OnlineLearner(
                dim=cfg.dim,
                degree=degree,
                forgetting_factor=forgetting_factor,
                regularization=regularization,
            )
            wm = WorkingMemory(dim=cfg.dim, window=cfg.window, decay=cfg.decay)
            # Surface (level 0) has a vocab-indexed verbaliser; upper levels
            # also own a verbaliser at their own dim but it's only used via
            # the downward/upward projections (no direct tokenisation).
            verb = Verbalizer(
                vocab_size=self.tokenizer.vocab_size,
                dim=cfg.dim,
                seed=seed + 101 * (i + 1),
            )
            # Ensure the top level is marked top.
            chunk = cfg.chunk_size if i < len(levels) - 1 else 0
            self.levels.append(_Level(
                dim=cfg.dim,
                window=cfg.window,
                chunk_size=chunk,
                core=core,
                wm=wm,
                verb=verb,
            ))

        # Random projections between adjacent levels.  ``up[k]`` maps level k
        # -> level k+1 (for upward summaries); ``down[k]`` maps level k+1 ->
        # level k (for downward conditioning).  Scaled Johnson-Lindenstrauss.
        rng = np.random.default_rng(seed + 7919)
        self._up_projs: list[np.ndarray] = []
        self._down_projs: list[np.ndarray] = []
        for k in range(len(self.levels) - 1):
            d_lo = self.levels[k].dim
            d_hi = self.levels[k + 1].dim
            up = rng.standard_normal((d_lo, d_hi)) / np.sqrt(d_lo)
            down = rng.standard_normal((d_hi, d_lo)) / np.sqrt(d_hi)
            self._up_projs.append(up.astype(np.float32))
            self._down_projs.append(down.astype(np.float32))

        self._tokens_seen = 0
        self._pairs_taught = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _conditioning_for_level(self, level: int) -> np.ndarray:
        """Parent intent projected down to this level's space.

        Returns a zero vector at the top level.  Uses the immediate parent
        only (chain-down-all-levels is a possible extension).
        """
        L = self.levels[level]
        if level >= len(self.levels) - 1:
            return np.zeros(L.dim, dtype=np.float32)

        parent = self.levels[level + 1]
        parent_intent = parent.core.predict(parent.wm.context())
        projected = (
            np.asarray(parent_intent, dtype=np.float32)
            @ self._down_projs[level]
        )
        return projected.astype(np.float32)

    def _propagate_up_from(self, level: int, learn: bool) -> None:
        """After ingesting into ``level``, push summary up if chunk complete."""
        L = self.levels[level]
        if L.is_top:
            return
        if L.tokens_since_chunk < L.chunk_size:
            return

        # Summarise this level's current WM and project into the upper space.
        summary_here = L.wm.context()
        summary_up = (summary_here @ self._up_projs[level]).astype(np.float32)

        upper = self.levels[level + 1]

        if learn:
            # Upper core learns: its current context -> the new summary.
            upper_ctx = upper.wm.context()
            upper.core.observe_sample(upper_ctx, summary_up)

        upper.wm.add(summary_up)
        upper.tokens_since_chunk += 1
        L.tokens_since_chunk = 0

        # Recurse up (the upper level might also hit its chunk boundary).
        self._propagate_up_from(level + 1, learn=learn)

    def _reset_all_wms(self) -> None:
        for L in self.levels:
            L.wm.reset()
            L.tokens_since_chunk = 0

    # ------------------------------------------------------------------
    # Core operation: ingest one token
    # ------------------------------------------------------------------

    def ingest_token(self, tid: int, learn: bool = True) -> None:
        """Absorb a single token at the surface; propagate summaries up.

        * Surface core learns ``surface_input -> target_embedding`` when
          ``learn`` is True.  ``surface_input = surface_wm.context() +
          parent_intent_projected``.
        * Token embedding is appended to the surface WM.
        * Chunk propagation runs for level 0 (and recursively for higher
          levels as their chunks complete).
        """
        L0 = self.levels[0]
        target_emb = L0.verb.encode_id(tid)

        if learn:
            surface_input = L0.wm.context() + self._conditioning_for_level(0)
            L0.core.observe_sample(surface_input, target_emb)
            self._tokens_seen += 1

        L0.wm.add(target_emb)
        L0.tokens_since_chunk += 1

        self._propagate_up_from(0, learn=learn)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def teach_text(self, text: str, reset_wm: bool = True) -> int:
        """Stream a text into the model with learning.

        Useful for self-supervised corpus absorption.  Returns the number
        of tokens ingested.
        """
        if reset_wm:
            self._reset_all_wms()
        ids = self.tokenizer.encode(text)
        for tid in ids:
            self.ingest_token(tid, learn=True)
        return len(ids)

    def teach_pair(
        self,
        text_in: str,
        text_out: str,
        append_eos: bool = True,
    ) -> None:
        """Q&A-style: prime with input (no learning), then absorb output."""
        self._reset_all_wms()
        # Prime the WMs with the input, but do not add surface learning
        # samples for input tokens — the target of interest is the output.
        for tid in self.tokenizer.encode(text_in):
            self.ingest_token(tid, learn=False)
        out_ids = self.tokenizer.encode(text_out)
        if append_eos:
            out_ids = out_ids + [self.tokenizer.eos_id]
        for tid in out_ids:
            self.ingest_token(tid, learn=True)
        self._pairs_taught += 1

    def teach_pairs(
        self,
        pairs: list[tuple[str, str]],
        epochs: int = 1,
        append_eos: bool = True,
    ) -> None:
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        for _ in range(epochs):
            for tin, tout in pairs:
                self.teach_pair(tin, tout, append_eos=append_eos)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_len: int = 80,
        temperature: float = 0.0,
        top_k: int | None = None,
        stop_on_eos: bool = True,
        min_confidence: float = 0.0,
        seed: int | None = None,
    ) -> HierarchicalResult:
        """Autoregressive generation conditioned on an ingested prompt."""
        if temperature < 0:
            raise ValueError("temperature must be >= 0")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be >= 1")

        rng = (np.random.default_rng(seed) if seed is not None
               else np.random.default_rng())

        # Prime WMs from the prompt (no learning).
        self._reset_all_wms()
        for tid in self.tokenizer.encode(prompt):
            self.ingest_token(tid, learn=False)

        L0 = self.levels[0]
        output_ids: list[int] = []
        confidences: list[float] = []
        stopped_reason = "max_len"

        for _ in range(max_len):
            surface_input = L0.wm.context() + self._conditioning_for_level(0)
            next_vec = L0.core.predict(surface_input)

            if temperature == 0.0:
                tid, sim = L0.verb.decode_vec(next_vec)
            else:
                probs = L0.verb.decode_distribution(next_vec, temperature)
                if top_k is not None and top_k < probs.size:
                    kept = np.argpartition(probs, -top_k)[-top_k:]
                    mask = np.zeros_like(probs)
                    mask[kept] = probs[kept]
                    s = mask.sum()
                    if s <= 0:
                        tid, sim = L0.verb.decode_vec(next_vec)
                    else:
                        probs = mask / s
                        tid = int(rng.choice(probs.size, p=probs))
                        sim = float(probs[tid])
                else:
                    tid = int(rng.choice(probs.size, p=probs))
                    sim = float(probs[tid])

            confidences.append(sim)
            if sim < min_confidence:
                stopped_reason = "low_confidence"
                break
            if stop_on_eos and tid == self.tokenizer.eos_id:
                stopped_reason = "eos"
                break

            output_ids.append(tid)
            # Ingest without learning so WMs and chunks evolve naturally.
            self.ingest_token(tid, learn=False)

        return HierarchicalResult(
            text=self.tokenizer.decode(output_ids),
            confidences=confidences,
            omegas=[L.core.omega for L in self.levels],
            phis=[L.core.phi for L in self.levels],
            stopped_reason=stopped_reason,
        )

    # ------------------------------------------------------------------
    # Diagnostics + forgetting
    # ------------------------------------------------------------------

    def report(self) -> HierarchicalReport:
        return HierarchicalReport(
            n_levels=len(self.levels),
            dims=[L.dim for L in self.levels],
            lifted_dims=[L.core.ld for L in self.levels],
            windows=[L.window for L in self.levels],
            chunk_sizes=[L.chunk_size for L in self.levels],
            n_samples=[L.core.n_samples for L in self.levels],
            omegas=[L.core.omega for L in self.levels],
            phis=[L.core.phi for L in self.levels],
            tokens_seen=self._tokens_seen,
            pairs_taught=self._pairs_taught,
        )

    @property
    def total_matrix_elements(self) -> int:
        return sum(3 * L.core.ld * L.core.ld for L in self.levels)

    def forget(self, factor: float) -> None:
        for L in self.levels:
            L.core.decay(factor)

    def forget_by_time(self, elapsed: float, half_life: float) -> None:
        for L in self.levels:
            L.core.decay_by_time(elapsed, half_life)

    def reset(self) -> None:
        """Wipe all learned state; embeddings and projections preserved."""
        for L in self.levels:
            L.core.reset()
            L.wm.reset()
            L.tokens_since_chunk = 0
        self._tokens_seen = 0
        self._pairs_taught = 0
