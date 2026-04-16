"""Two-stage language model: Intent -> Surface.

Splits what the single-core ``LanguageModel`` does into two cooperating
cores, mirroring Chomsky's *deep structure / surface structure* distinction
(Chomsky, 1965):

    Stage 1 — Intent Core (small dim):
        input context  ─▶  intent vector
        Learns what the response *means*, compactly.

    Stage 2 — Surface Core (medium dim):
        (intent, current output so far)  ─▶  next token
        Learns how to *spell out* the intent character by character,
        conditioned on the intent summary.

Why this saves memory
---------------------
``lifted_dim`` grows as O(dim^2) (degree=2), so two small cores beat one
medium core quadratically.  Concrete:

    single dim=24  →  lifted_dim=325  →  105,625 matrix elements
    8 + 16          →  45 + 153        →   25,434   (4.2× less)

The split also matches human-like cognition: meaning and form are
separately learned and separately forgotten.

Cost
----
* Two models to keep in sync.
* Stage-1 error propagates into Stage-2 conditioning — so very small
  intent_dim loses information that the surface core cannot recover.
* Intent vector is a dense summary of the response, built by running the
  output through a small WorkingMemory; this is a heuristic, not a
  learned encoder.

Axiom alignment
---------------
* Axiom 1: both cores update via closed-form rank-1 moment accumulation
  (``OnlineLearner``).  No inner iteration.
* Axiom 3: Stage-1 and Stage-2 expose their own Omega/Phi.  The overall
  confidence is their minimum — the weaker stage gates the system.
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
# TwoStageReport
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TwoStageReport:
    """Per-stage diagnostic snapshot."""

    intent_dim: int
    surface_dim: int
    intent_omega: float
    intent_phi: float
    surface_omega: float
    surface_phi: float
    pairs_taught: int
    tokens_seen: int

    @property
    def omega(self) -> float:
        """System-wide cohesion = min of the two stages (weakest link)."""
        return min(self.intent_omega, self.surface_omega)

    @property
    def phi(self) -> float:
        return min(self.intent_phi, self.surface_phi)


# ---------------------------------------------------------------------------
# Two-stage model
# ---------------------------------------------------------------------------

class TwoStageLanguageModel:
    """Intent-then-surface character-level language model.

    The intent core operates in a small phase space and maps input context
    directly to an intent summary.  The surface core consumes (intent
    projected into the surface space) + (current surface context) and
    predicts the next character.
    """

    DEFAULT_VOCAB = (
        "\n !\"#$%&'()*+,-./0123456789:;<=>?@"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
        "abcdefghijklmnopqrstuvwxyz{|}~"
    )

    def __init__(
        self,
        vocab: str | None = None,
        intent_dim: int = 8,
        surface_dim: int = 16,
        intent_window: int = 8,
        surface_window: int = 10,
        decay: float = 0.7,
        forgetting_factor: float = 1.0,
        regularization: float = 1e-4,
        seed: int = 0,
        degree: int = 2,
    ) -> None:
        if intent_dim <= 0 or surface_dim <= 0:
            raise ValueError("intent_dim and surface_dim must be positive")

        self.vocab = vocab if vocab is not None else self.DEFAULT_VOCAB
        self.intent_dim = int(intent_dim)
        self.surface_dim = int(surface_dim)

        self.tokenizer = CharTokenizer(self.vocab)
        # Separate embeddings per stage — different dims, distinct seeds
        # so intent and surface spaces are independent.
        self.intent_verb = Verbalizer(
            vocab_size=self.tokenizer.vocab_size,
            dim=self.intent_dim,
            seed=seed,
        )
        self.surface_verb = Verbalizer(
            vocab_size=self.tokenizer.vocab_size,
            dim=self.surface_dim,
            seed=seed + 101,
        )

        self.intent_wm = WorkingMemory(
            dim=self.intent_dim, window=intent_window, decay=decay
        )
        self.surface_wm = WorkingMemory(
            dim=self.surface_dim, window=surface_window, decay=decay
        )

        self.intent_core = OnlineLearner(
            dim=self.intent_dim,
            degree=degree,
            forgetting_factor=forgetting_factor,
            regularization=regularization,
        )
        self.surface_core = OnlineLearner(
            dim=self.surface_dim,
            degree=degree,
            forgetting_factor=forgetting_factor,
            regularization=regularization,
        )

        # Fixed random projection intent_dim -> surface_dim (Johnson-Lindenstrauss).
        # Not learned — a deterministic link between the two phase spaces.
        rng = np.random.default_rng(seed + 7919)
        P = rng.standard_normal((self.intent_dim, self.surface_dim))
        # Scale so that ||P @ v|| ~ ||v|| for unit v (JL-style)
        P /= np.sqrt(self.intent_dim)
        self._projection = P.astype(np.float32)

        self._pairs_taught = 0
        self._tokens_seen = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _summarise_intent(self, text: str) -> np.ndarray:
        """Walk ``text`` through the intent WorkingMemory and return the EMA."""
        self.intent_wm.reset()
        for tid in self.tokenizer.encode(text):
            self.intent_wm.add(self.intent_verb.encode_id(tid))
        return self.intent_wm.context()

    def _project_intent(self, intent_vec: np.ndarray) -> np.ndarray:
        """Lift an intent-space vector into surface space."""
        v = np.asarray(intent_vec, dtype=np.float32).reshape(-1)
        return (v @ self._projection).astype(np.float32)

    def _surface_input(self, intent_projected: np.ndarray) -> np.ndarray:
        """Build the Surface Core input = surface WM context + intent.

        Additive conditioning keeps the input dimension at ``surface_dim``
        (so the lifted space stays at ``lifted_dim(surface_dim)``).
        """
        return self.surface_wm.context() + intent_projected

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def teach(self, text_in: str, text_out: str, append_eos: bool = True) -> None:
        """Absorb one (stimulus, response) pair into both cores."""
        # ---- Stage 1: intent mapping ----
        intent_input = self._summarise_intent(text_in)
        intent_target = self._summarise_intent(text_out)
        self.intent_core.observe_sample(intent_input, intent_target)

        # ---- Stage 2: surface autoregression, conditioned on intent ----
        intent_projected = self._project_intent(intent_target)  # teacher-forced
        self.surface_wm.reset()

        out_ids = self.tokenizer.encode(text_out)
        if append_eos:
            out_ids = out_ids + [self.tokenizer.eos_id]

        for tid in out_ids:
            ctx = self._surface_input(intent_projected)
            target_emb = self.surface_verb.encode_id(tid)
            self.surface_core.observe_sample(ctx, target_emb)
            self.surface_wm.add(target_emb)
            self._tokens_seen += 1

        self._pairs_taught += 1

    def train_pairs(
        self,
        pairs: list[tuple[str, str]],
        epochs: int = 1,
        append_eos: bool = True,
    ) -> None:
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        for _ in range(epochs):
            for tin, tout in pairs:
                self.teach(tin, tout, append_eos=append_eos)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_len: int = 200,
        temperature: float = 0.0,
        top_k: int | None = None,
        stop_on_eos: bool = True,
        min_confidence: float = 0.0,
        seed: int | None = None,
    ) -> "TwoStageGeneration":
        """Run Intent inference then autoregressive Surface generation."""
        if temperature < 0:
            raise ValueError("temperature must be >= 0")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be >= 1")

        rng = (np.random.default_rng(seed) if seed is not None
               else np.random.default_rng())

        # ---- Stage 1: infer intent ----
        intent_input = self._summarise_intent(prompt)
        intent_pred = self.intent_core.predict(intent_input)
        intent_projected = self._project_intent(intent_pred)

        # ---- Stage 2: surface autoregression ----
        self.surface_wm.reset()

        output_ids: list[int] = []
        confidences: list[float] = []
        stopped_reason = "max_len"

        for _ in range(max_len):
            ctx = self._surface_input(intent_projected)
            next_vec = self.surface_core.predict(ctx)

            if temperature == 0.0:
                tid, sim = self.surface_verb.decode_vec(next_vec)
            else:
                probs = self.surface_verb.decode_distribution(
                    next_vec, temperature=temperature
                )
                if top_k is not None and top_k < probs.size:
                    kept = np.argpartition(probs, -top_k)[-top_k:]
                    mask = np.zeros_like(probs)
                    mask[kept] = probs[kept]
                    s = mask.sum()
                    if s <= 0:
                        tid, sim = self.surface_verb.decode_vec(next_vec)
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
            self.surface_wm.add(self.surface_verb.encode_id(tid))

        return TwoStageGeneration(
            text=self.tokenizer.decode(output_ids),
            confidences=confidences,
            stopped_reason=stopped_reason,
            intent_omega=self.intent_core.omega,
            surface_omega=self.surface_core.omega,
        )

    # ------------------------------------------------------------------
    # Diagnostics + forgetting
    # ------------------------------------------------------------------

    def report(self) -> TwoStageReport:
        return TwoStageReport(
            intent_dim=self.intent_dim,
            surface_dim=self.surface_dim,
            intent_omega=self.intent_core.omega,
            intent_phi=self.intent_core.phi,
            surface_omega=self.surface_core.omega,
            surface_phi=self.surface_core.phi,
            pairs_taught=self._pairs_taught,
            tokens_seen=self._tokens_seen,
        )

    @property
    def omega(self) -> float:
        return min(self.intent_core.omega, self.surface_core.omega)

    @property
    def phi(self) -> float:
        return min(self.intent_core.phi, self.surface_core.phi)

    @property
    def total_matrix_elements(self) -> int:
        """Combined size of the two moment matrices (dim² measure)."""
        return self.intent_core.ld ** 2 + self.surface_core.ld ** 2

    def forget(self, factor: float) -> None:
        """Decay both cores uniformly."""
        self.intent_core.decay(factor)
        self.surface_core.decay(factor)

    def forget_by_time(self, elapsed: float, half_life: float) -> None:
        self.intent_core.decay_by_time(elapsed, half_life)
        self.surface_core.decay_by_time(elapsed, half_life)

    def reset(self) -> None:
        """Forget all learned weights (embeddings and projection preserved)."""
        self.intent_core.reset()
        self.surface_core.reset()
        self.intent_wm.reset()
        self.surface_wm.reset()
        self._pairs_taught = 0
        self._tokens_seen = 0


# ---------------------------------------------------------------------------
# Result of a two-stage generation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TwoStageGeneration:
    text: str
    confidences: list[float]
    stopped_reason: str
    intent_omega: float
    surface_omega: float

    @property
    def omega(self) -> float:
        return min(self.intent_omega, self.surface_omega)
