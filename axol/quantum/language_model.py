"""AXOL language model — conversational + generative, built on dual-layer.

This module composes the previously-implemented pieces into a usable
character-level language model:

    text corpus  --(train_on_text)-->  sliding-window (context, next_char) stream
    Q&A pairs    --(train_pairs)---->  (stimulus, response) stream
                                              |
                                              v
                                    ConversationalAxol
                                    (Tokenizer + Verbalizer +
                                     WorkingMemory + IntuitionCore)
                                              |
                                              v
                                generate(prompt, temperature, max_len)

Limits (honest)
---------------
Character-level, small embed_dim.  This is a demonstrator — not competitive
with transformer LLMs.  What it demonstrates is that the AXOL axioms
(closed-form updates, dual-layer cognition, forgetting, confidence as
first-class output) compose into a working generative system.

File format for save/load
-------------------------
Two files written side-by-side at ``path`` (no extension) and ``path.json``:

    <path>.npz   — numpy arrays: verbalizer E, intuition G, H, B
    <path>.json  — plain JSON: vocab, dims, hyperparams, counters
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from axol.quantum.conversation import ConversationalAxol


# ---------------------------------------------------------------------------
# Saved-model format version
# ---------------------------------------------------------------------------

_FORMAT_VERSION = 1


@dataclass(frozen=True)
class GenerationResult:
    """Output of ``generate`` including per-token diagnostics."""

    text: str
    confidences: list[float]
    stopped_reason: str       # "eos" | "max_len" | "low_confidence"
    omega: float
    phi: float


# ---------------------------------------------------------------------------
# LanguageModel
# ---------------------------------------------------------------------------

class LanguageModel:
    """Character-level AXOL language model with two training modes.

    Parameters (all forwarded to the underlying ConversationalAxol):
        vocab:             character set.  Defaults to ASCII printable.
        embed_dim:         semantic phase-space dimension.
        window:            WorkingMemory window (context length).
        decay:             EMA decay inside WorkingMemory.
        forgetting_factor: per-sample memory decay for IntuitionCore.
        regularization:    RLS regulariser for the moment matrix G.
        seed:              RNG seed for the Verbalizer's random basis.
        degree:            polynomial degree for Koopman lifting.
    """

    DEFAULT_VOCAB = (
        # ASCII printable + newline
        "\n !\"#$%&'()*+,-./0123456789:;<=>?@"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
        "abcdefghijklmnopqrstuvwxyz{|}~"
    )

    def __init__(
        self,
        vocab: str | None = None,
        embed_dim: int = 24,
        window: int = 12,
        decay: float = 0.7,
        forgetting_factor: float = 1.0,   # LMs default to full retention
        regularization: float = 1e-4,
        seed: int = 0,
        degree: int = 2,
    ) -> None:
        self._config = {
            "vocab": vocab if vocab is not None else self.DEFAULT_VOCAB,
            "embed_dim": int(embed_dim),
            "window": int(window),
            "decay": float(decay),
            "forgetting_factor": float(forgetting_factor),
            "regularization": float(regularization),
            "seed": int(seed),
            "degree": int(degree),
        }
        self.chat = ConversationalAxol(
            vocab=self._config["vocab"],
            embed_dim=self._config["embed_dim"],
            window=self._config["window"],
            decay=self._config["decay"],
            forgetting_factor=self._config["forgetting_factor"],
            regularization=self._config["regularization"],
            seed=self._config["seed"],
            degree=self._config["degree"],
        )

    # ------------------------------------------------------------------
    # Training — supervised Q&A
    # ------------------------------------------------------------------

    def train_pairs(
        self,
        pairs: list[tuple[str, str]],
        epochs: int = 1,
        append_eos: bool = True,
    ) -> None:
        """Teacher-forced training from (stimulus, response) pairs.

        ``epochs`` here means re-presentations of the same list (each
        presentation is still a stream of closed-form rank-1 updates).
        """
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        for _ in range(epochs):
            for tin, tout in pairs:
                self.chat.teach(tin, tout, append_eos=append_eos)

    # ------------------------------------------------------------------
    # Training — self-supervised sliding-window
    # ------------------------------------------------------------------

    def train_on_text(self, text: str) -> int:
        """Predict each next character from its rolling context.

        Streams through ``text``: at each position the WorkingMemory holds
        the previous characters, and the IntuitionCore learns to map the
        current context to the *next* character's embedding.

        Returns the number of (context, next_char) pairs absorbed.
        """
        ids = self.chat.tokenizer.encode(text)
        if len(ids) < 2:
            return 0

        self.chat.working_memory.reset()
        # Prime working memory with the first token
        self.chat.working_memory.add(
            self.chat.verbalizer.encode_id(ids[0])
        )

        n_samples = 0
        for i in range(1, len(ids)):
            ctx = self.chat.working_memory.context()
            target_emb = self.chat.verbalizer.encode_id(ids[i])
            self.chat.intuition.observe_sample(ctx, target_emb)
            self.chat.working_memory.add(target_emb)
            n_samples += 1

        self.chat._tokens_seen += n_samples
        return n_samples

    # ------------------------------------------------------------------
    # Generation — argmax or temperature sampling
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
    ) -> GenerationResult:
        """Autoregressive generation from ``prompt``.

        * ``temperature == 0``  : greedy argmax (cheapest, deterministic).
        * ``temperature  > 0``  : softmax-sample over the Verbalizer's
                                  cosine scores.  ``top_k`` truncates
                                  the distribution before sampling.
        * ``min_confidence``    : if the greedy max cosine falls below
                                  this threshold the generator stops
                                  (refuses to guess).
        """
        if temperature < 0:
            raise ValueError("temperature must be >= 0")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be >= 1 if given")

        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        tok = self.chat.tokenizer
        verb = self.chat.verbalizer
        wm = self.chat.working_memory
        intuition = self.chat.intuition

        # Load the prompt into working memory (no learning).
        wm.reset()
        for tid in tok.encode(prompt):
            wm.add(verb.encode_id(tid))

        output_ids: list[int] = []
        confidences: list[float] = []
        stopped_reason = "max_len"

        for _ in range(max_len):
            ctx = wm.context()
            next_vec = intuition.predict(ctx)

            if temperature == 0.0:
                tid, sim = verb.decode_vec(next_vec)
            else:
                probs = verb.decode_distribution(next_vec, temperature=temperature)
                if top_k is not None and top_k < probs.size:
                    # Zero out all but the top_k entries, renormalise.
                    kept = np.argpartition(probs, -top_k)[-top_k:]
                    mask = np.zeros_like(probs)
                    mask[kept] = probs[kept]
                    s = mask.sum()
                    if s <= 0:
                        tid, sim = verb.decode_vec(next_vec)
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
            if stop_on_eos and tid == tok.eos_id:
                stopped_reason = "eos"
                break

            output_ids.append(tid)
            wm.add(verb.encode_id(tid))

        return GenerationResult(
            text=tok.decode(output_ids),
            confidences=confidences,
            stopped_reason=stopped_reason,
            omega=self.chat.omega,
            phi=self.chat.phi,
        )

    # ------------------------------------------------------------------
    # Single-turn chat convenience
    # ------------------------------------------------------------------

    def chat_once(
        self,
        text_in: str,
        text_out: str | None = None,
        max_len: int = 120,
        learn: bool = True,
        elapsed_time: float = 0.0,
        half_life: float | None = None,
    ) -> str:
        """One turn: reflex response + optional reinforcement."""
        return self.chat.converse(
            text_in,
            text_out=text_out,
            max_len=max_len,
            learn=learn,
            elapsed_time=elapsed_time,
            half_life=half_life,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def omega(self) -> float:
        return self.chat.omega

    @property
    def phi(self) -> float:
        return self.chat.phi

    @property
    def samples_trained(self) -> int:
        return self.chat.intuition.n_samples

    # ------------------------------------------------------------------
    # Forgetting (delegates)
    # ------------------------------------------------------------------

    def forget(self, factor: float) -> None:
        self.chat.forget(factor)

    def forget_by_time(self, elapsed: float, half_life: float) -> None:
        self.chat.forget_by_time(elapsed, half_life)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the full learned state next to ``path``.

        Writes ``<path>.npz`` (arrays) and ``<path>.json`` (metadata).
        """
        path = Path(path)
        npz_path = path.with_suffix(".npz")
        json_path = path.with_suffix(".json")

        intuition = self.chat.intuition
        np.savez(
            npz_path,
            verbalizer_E=self.chat.verbalizer.E,
            intuition_G=intuition._G,
            intuition_H=intuition._H,
            intuition_B=intuition._B,
        )
        meta = {
            "format_version": _FORMAT_VERSION,
            "config": self._config,
            "counters": {
                "intuition_n_samples": intuition._n_samples,
                "intuition_residual_sq": intuition._last_residual_sq,
                "intuition_residual_count": intuition._residual_count,
                "tokens_seen": self.chat._tokens_seen,
                "pairs_taught": self.chat._pairs_taught,
            },
        }
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "LanguageModel":
        """Load a model previously saved with ``save``."""
        path = Path(path)
        npz_path = path.with_suffix(".npz")
        json_path = path.with_suffix(".json")

        with json_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("format_version") != _FORMAT_VERSION:
            raise ValueError(
                f"unsupported format version {meta.get('format_version')!r}; "
                f"expected {_FORMAT_VERSION}"
            )

        cfg = meta["config"]
        lm = cls(
            vocab=cfg["vocab"],
            embed_dim=cfg["embed_dim"],
            window=cfg["window"],
            decay=cfg["decay"],
            forgetting_factor=cfg["forgetting_factor"],
            regularization=cfg["regularization"],
            seed=cfg["seed"],
            degree=cfg["degree"],
        )

        arrs = np.load(npz_path)
        # Restore Verbalizer embedding matrix
        lm.chat.verbalizer.E = arrs["verbalizer_E"].astype(np.float32)
        # Restore IntuitionCore moments
        intuition = lm.chat.intuition
        intuition._G = arrs["intuition_G"].astype(np.float64)
        intuition._H = arrs["intuition_H"].astype(np.float64)
        intuition._B = arrs["intuition_B"].astype(np.float64)
        intuition._K_cache = None
        intuition._n_samples = int(meta["counters"]["intuition_n_samples"])
        intuition._last_residual_sq = float(meta["counters"]["intuition_residual_sq"])
        intuition._residual_count = int(meta["counters"]["intuition_residual_count"])

        lm.chat._tokens_seen = int(meta["counters"]["tokens_seen"])
        lm.chat._pairs_taught = int(meta["counters"]["pairs_taught"])
        return lm
