"""Fractal text generator — Perlin/fBm-style noise over a sentence dictionary.

User's analogy: in procedural map generation, fractal Brownian motion
(fBm) noise is what makes terrain *look natural* rather than random or
grid-like.  The same mechanism applied to a phrase dictionary yields
text that lies between "pure memorisation" and "pure randomness" —
a deterministic yet coherent *variation* of learned material.

How it works
------------
    prompt
      │
      ▼
    intent vector ──▶ SentenceDictionary.lookup(top_k)
                              │
                              ▼
                   anchors:   [macro, alt_1, alt_2, ...]
                              │
                              │   for each word slot `pos`:
                              │     n = fBm(pos * chunk_scale, seed)   ∈ [-1, 1]
                              │     if |n| * noise_strength < threshold:
                              │         take macro[pos]              ← stay faithful
                              │     else:
                              │         take alt_k[pos_mapped]       ← borrow
                              ▼
                         composed sentence

Why this "feels natural"
------------------------
Pure per-slot random substitution would produce word salad.  fBm noise
varies *smoothly* across positions, so adjacent slots tend to agree
about which anchor to borrow from — producing coherent chunks
("actions speak" together, "the bold" together) rather than jumbled
words.  This is exactly how noise-based terrain generators produce
continents and regions rather than single-pixel static.

Parameters (game-developer intuitions carry over):
  * ``noise_strength`` (0..1)    master blend — 0 = pure snap, 1 = pure mix
  * ``chunk_scale``    (>0)      slot frequency — small = long chunks
  * ``octaves``        (1..5)    detail layers — higher = more variation
  * ``persistence``    (0..1)    amplitude decay per octave
  * ``lacunarity``     (>1)      frequency growth per octave
  * ``seed``           (int)     deterministic reproduction

Axiom alignment
---------------
* Axiom 1:  NoiseField is a deterministic function of ``(seed, position)``
  with no iteration — just a hash + smoothstep + octave sum.
* Axiom 3:  The generator reports a substitution_rate and the raw noise
  trace; confidence thus lives at both the intent level (cosine to
  chosen anchors) and the surface level (how much noise deflected).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from axol.quantum.sentence_decoder import SentenceDecoderLanguageModel


# ---------------------------------------------------------------------------
# NoiseField
# ---------------------------------------------------------------------------

class NoiseField:
    """Deterministic 1-D fractal Brownian motion noise in [-1, 1].

    Uses hash-based *value noise* (grid lookup + smoothstep interpolation)
    summed over ``octaves`` layers.  Output is normalised by the total
    amplitude so the result stays in [-1, 1] regardless of octave count.
    """

    def __init__(
        self,
        seed: int = 0,
        octaves: int = 3,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
    ) -> None:
        if octaves < 1:
            raise ValueError("octaves must be >= 1")
        if not (0.0 < persistence <= 1.0):
            raise ValueError("persistence must be in (0, 1]")
        if lacunarity <= 1.0:
            raise ValueError("lacunarity must be > 1")
        self.seed = int(seed)
        self.octaves = int(octaves)
        self.persistence = float(persistence)
        self.lacunarity = float(lacunarity)

    # -- deterministic hash, 32-bit mix ------------------------------------

    def _hash01(self, x: int, octave: int) -> float:
        """Integer -> uniform float in [-1, 1]."""
        h = (int(x) * 1103515245 + int(octave) * 12345 + self.seed) & 0xFFFFFFFF
        h ^= (h >> 16)
        h = (h * 0x45D9F3B) & 0xFFFFFFFF
        h ^= (h >> 16)
        h = (h * 0x45D9F3B) & 0xFFFFFFFF
        h ^= (h >> 16)
        return (h / 0xFFFFFFFF) * 2.0 - 1.0

    def _value_noise(self, x: float, octave: int) -> float:
        """Smoothstep-interpolated value noise at a single octave."""
        xi = int(np.floor(x))
        xf = x - xi
        a = self._hash01(xi, octave)
        b = self._hash01(xi + 1, octave)
        t = xf * xf * (3.0 - 2.0 * xf)        # smoothstep
        return a * (1.0 - t) + b * t

    def sample(self, x: float) -> float:
        """Octave-summed fBm at position ``x``.  Output in [-1, 1]."""
        total = 0.0
        amp = 1.0
        freq = 1.0
        norm = 0.0
        for k in range(self.octaves):
            total += amp * self._value_noise(x * freq, k)
            norm += amp
            amp *= self.persistence
            freq *= self.lacunarity
        return total / norm if norm > 0 else 0.0


# ---------------------------------------------------------------------------
# FractalResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FractalResult:
    """Output of ``FractalTextGenerator.compose``."""

    text: str
    macro_source: str                 # top-1 anchor that supplied the skeleton
    anchors: list[str]                # all anchors considered
    noise_trace: list[float]          # noise value at each slot
    substitutions: list[int]          # per-slot anchor index chosen (0 = macro)
    substitution_rate: float          # fraction of slots taken from non-macro


# ---------------------------------------------------------------------------
# FractalTextGenerator
# ---------------------------------------------------------------------------

class FractalTextGenerator:
    """Wraps a ``SentenceDecoderLanguageModel`` with a noise-driven composer."""

    def __init__(self, decoder: SentenceDecoderLanguageModel) -> None:
        self.decoder = decoder

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _map_position(rel_pos: float, target_len: int) -> int:
        """Map a relative position in [0, 1] to an index in ``target_len``."""
        if target_len <= 0:
            return 0
        idx = int(round(rel_pos * (target_len - 1)))
        return max(0, min(idx, target_len - 1))

    def _candidate_pool(
        self,
        anchors: Sequence[str],
    ) -> list[list[str]]:
        """Build per-slot candidate word lists aligned to the macro anchor's length."""
        if not anchors:
            return []
        macro_words = anchors[0].split()
        n_slots = len(macro_words)
        if n_slots == 0:
            return []
        pool: list[list[str]] = [[word] for word in macro_words]
        for alt in anchors[1:]:
            alt_words = alt.split()
            if not alt_words:
                continue
            for pos in range(n_slots):
                rel = pos / max(n_slots - 1, 1)
                alt_idx = self._map_position(rel, len(alt_words))
                pool[pos].append(alt_words[alt_idx])
        return pool

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(
        self,
        prompt: str,
        noise_strength: float = 0.5,
        k_anchors: int = 5,
        chunk_scale: float = 0.7,
        octaves: int = 3,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        threshold: float = 0.3,
        seed: int | None = None,
    ) -> FractalResult:
        """Generate text by fBm-noise-driven phrase composition.

        Notes
        -----
        * ``noise_strength=0`` reproduces the plain snap decoder output.
        * ``noise_strength=1`` lets noise fully decide every slot.
        * ``chunk_scale`` controls how quickly noise changes across slots;
          smaller values = longer coherent chunks.
        """
        if not (0.0 <= noise_strength <= 1.0):
            raise ValueError("noise_strength must be in [0, 1]")
        if k_anchors < 1:
            raise ValueError("k_anchors must be >= 1")
        if chunk_scale <= 0:
            raise ValueError("chunk_scale must be > 0")
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("threshold must be in [0, 1]")

        # 1. Intent inference
        intent_in = self.decoder._summarise_intent(prompt)
        predicted = self.decoder.intent_core.predict(intent_in)
        matches = self.decoder.dictionary.lookup(predicted, top_k=k_anchors)

        if not matches:
            return FractalResult(
                text="", macro_source="", anchors=[],
                noise_trace=[], substitutions=[], substitution_rate=0.0,
            )

        anchor_texts = [t for t, _ in matches]
        macro = anchor_texts[0]

        # Fast path: no noise → plain snap
        macro_words = macro.split()
        if noise_strength == 0.0 or len(anchor_texts) == 1 or not macro_words:
            return FractalResult(
                text=macro, macro_source=macro, anchors=anchor_texts,
                noise_trace=[], substitutions=[0] * len(macro_words),
                substitution_rate=0.0,
            )

        # 2. Build per-slot candidate pool
        pool = self._candidate_pool(anchor_texts)

        # 3. Noise field drives selection
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 1 << 30))
        noise = NoiseField(
            seed=seed, octaves=octaves,
            persistence=persistence, lacunarity=lacunarity,
        )

        chosen: list[str] = []
        noise_trace: list[float] = []
        substitutions: list[int] = []

        for pos, slot_pool in enumerate(pool):
            n = noise.sample(pos * chunk_scale)
            noise_trace.append(n)
            magnitude = abs(n)
            # Decision: stay with macro if noise*strength is small
            if magnitude * noise_strength < threshold or len(slot_pool) <= 1:
                chosen.append(slot_pool[0])
                substitutions.append(0)
                continue
            # Otherwise pick an alternate anchor: map n to a pool index
            # (skip index 0 which is macro), keeping the choice deterministic
            # and *spatially coherent* because n itself is smooth.
            n_alts = len(slot_pool) - 1
            # rescale magnitude to [0, 1]
            u = min(max((magnitude - threshold) / (1.0 - threshold), 0.0), 1.0)
            alt_idx = 1 + int(u * n_alts) % n_alts
            alt_idx = min(alt_idx, len(slot_pool) - 1)
            chosen.append(slot_pool[alt_idx])
            substitutions.append(alt_idx)

        text = " ".join(chosen)
        n_subs = sum(1 for s in substitutions if s != 0)
        rate = n_subs / max(len(substitutions), 1)

        return FractalResult(
            text=text, macro_source=macro, anchors=anchor_texts,
            noise_trace=noise_trace, substitutions=substitutions,
            substitution_rate=rate,
        )

    # ------------------------------------------------------------------
    # Batch variations (useful for picking the best of N samples)
    # ------------------------------------------------------------------

    def variations(
        self,
        prompt: str,
        n: int = 5,
        base_seed: int | None = None,
        **compose_kwargs,
    ) -> list[FractalResult]:
        """Return ``n`` distinct compositions with different seeds."""
        if n < 1:
            raise ValueError("n must be >= 1")
        rng = (np.random.default_rng(base_seed) if base_seed is not None
               else np.random.default_rng())
        out: list[FractalResult] = []
        used_seeds: set[int] = set()
        for _ in range(n):
            s = int(rng.integers(0, 1 << 30))
            while s in used_seeds:
                s = int(rng.integers(0, 1 << 30))
            used_seeds.add(s)
            out.append(self.compose(prompt, seed=s, **compose_kwargs))
        return out
