"""Stream-of-consciousness language model.

Extends ``TwoStageLanguageModel`` with a *recursive* intent-surface cycle:

    intent_vec_k  ─▶  surface segment_k  ─▶  summary of segment_k
          ▲                                            │
          └────────────────── feedback ────────────────┘

Each Intent WorkingMemory state is a spatial-axis snapshot of the
"current thought" (공간축).  Between snapshots the Surface WorkingMemory
drives production of one output segment; the summary of that segment is
then folded back into the Intent WM, which predicts the next intent.

This is the 의식의 흐름 (stream of consciousness) explicitly realised:
thought → utterance → changed thought → next utterance.

Relation to prior pieces
------------------------
* ``TwoStageLanguageModel`` already splits intent and surface cores.
* ``StreamingLanguageModel`` keeps those pieces and adds:
    - ``teach_stream(segments)``  — learn across a sentence sequence,
                                    both the intent-to-intent transitions
                                    and each segment's surface.
    - ``stream(prompt, n)``       — generate ``n`` segments where each
                                    segment influences the next via the
                                    summary feedback.

Axiom alignment
---------------
* Axiom 1:  each core update remains closed-form rank-1.  The
  multi-segment loop is *external* recursion — conceptually equivalent
  to an animal having a second thought after hearing itself speak —
  not an internal convergence iteration.
* Axiom 3:  segments accumulate ``confidence`` per step; the whole stream
  reports min/avg/final confidence so low-confidence drift is visible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from axol.quantum.two_stage import TwoStageLanguageModel


# ---------------------------------------------------------------------------
# Result of a streaming generation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StreamResult:
    """All segments produced plus per-step diagnostics."""

    segments: list[str]
    confidences: list[list[float]]    # per-segment per-token cosine
    intent_omega: float
    surface_omega: float

    @property
    def text(self) -> str:
        """Join all segments with a separator for convenience."""
        return " ".join(s for s in self.segments if s)

    @property
    def mean_confidence(self) -> float:
        flat = [c for seg in self.confidences for c in seg]
        return float(sum(flat) / len(flat)) if flat else 0.0

    @property
    def min_confidence(self) -> float:
        flat = [c for seg in self.confidences for c in seg]
        return float(min(flat)) if flat else 0.0


# ---------------------------------------------------------------------------
# StreamingLanguageModel
# ---------------------------------------------------------------------------

class StreamingLanguageModel(TwoStageLanguageModel):
    """Recursive intent↔surface loop on top of TwoStageLanguageModel."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_segment(
        self,
        intent_vec: np.ndarray,
        max_len: int,
        temperature: float,
        top_k: int | None,
        stop_on_eos: bool,
        min_confidence: float,
        rng: np.random.Generator,
    ) -> tuple[list[int], list[float]]:
        """Run the Surface core autoregressively from a single intent."""
        intent_projected = self._project_intent(intent_vec)
        self.surface_wm.reset()

        tokens: list[int] = []
        confs: list[float] = []
        for _ in range(max_len):
            ctx = self.surface_wm.context() + intent_projected
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

            confs.append(sim)
            if sim < min_confidence:
                break
            if stop_on_eos and tid == self.tokenizer.eos_id:
                break

            tokens.append(tid)
            self.surface_wm.add(self.surface_verb.encode_id(tid))

        return tokens, confs

    # ------------------------------------------------------------------
    # Training on a sequence of segments
    # ------------------------------------------------------------------

    def teach_stream(
        self,
        segments: list[str],
        seed_prompt: str | None = None,
        append_eos: bool = True,
    ) -> None:
        """Learn across a sentence sequence.

        Two signals are absorbed per segment:

        * Intent transition: from the intent WM's context *before* this
          segment, predict this segment's intent summary.
        * Surface generation: conditioned on this segment's intent,
          teacher-force the characters of the segment.

        After each segment the summary is pushed into the intent WM so
        subsequent segments see it (this is the same feedback the
        generator will use at inference time).
        """
        if not segments:
            return

        # Pre-compute all summaries up front.  ``_summarise_intent``
        # resets the intent WM as a side-effect, so we must take its
        # outputs first and then rebuild the WM from scratch.
        target_intents = [self._summarise_intent(s) for s in segments]
        seed_summary = (
            self._summarise_intent(seed_prompt) if seed_prompt else None
        )

        self.intent_wm.reset()
        if seed_summary is not None:
            self.intent_wm.add(seed_summary)

        for segment, target_intent in zip(segments, target_intents):
            # (a) Intent transition: predict target from current WM context.
            ctx = self.intent_wm.context()
            self.intent_core.observe_sample(ctx, target_intent)

            # (b) Surface training conditioned on this segment's intent.
            intent_projected = self._project_intent(target_intent)
            self.surface_wm.reset()
            out_ids = self.tokenizer.encode(segment)
            if append_eos:
                out_ids = out_ids + [self.tokenizer.eos_id]
            for tid in out_ids:
                sctx = self.surface_wm.context() + intent_projected
                target_emb = self.surface_verb.encode_id(tid)
                self.surface_core.observe_sample(sctx, target_emb)
                self.surface_wm.add(target_emb)
                self._tokens_seen += 1

            # (c) Feedback: make this segment's summary available to the
            # next iteration, matching what the generator will do.
            self.intent_wm.add(target_intent)

        self._pairs_taught += 1

    def teach_streams(
        self,
        streams: list[list[str]],
        seed_prompts: list[str | None] | None = None,
        epochs: int = 1,
    ) -> None:
        """Learn from multiple streams, optionally re-presented ``epochs`` times."""
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        if seed_prompts is None:
            seed_prompts = [None] * len(streams)
        if len(seed_prompts) != len(streams):
            raise ValueError("seed_prompts must have the same length as streams")
        for _ in range(epochs):
            for segs, seed in zip(streams, seed_prompts):
                self.teach_stream(segs, seed_prompt=seed)

    # ------------------------------------------------------------------
    # Generation: the recursive stream
    # ------------------------------------------------------------------

    def stream(
        self,
        prompt: str,
        n_segments: int = 3,
        segment_max_len: int = 40,
        temperature: float = 0.0,
        top_k: int | None = None,
        stop_on_eos: bool = True,
        min_confidence: float = 0.0,
        seed: int | None = None,
    ) -> StreamResult:
        """Generate a stream of ``n_segments`` segments.

        The intent WM is seeded from ``prompt``.  Each cycle:

        1. Predict the next intent from the current intent WM context.
        2. Generate a surface segment conditioned on that intent.
        3. Summarise the generated segment and add to the intent WM.

        Step 3 is the feedback: what was just said shapes what will be
        thought next.
        """
        if temperature < 0:
            raise ValueError("temperature must be >= 0")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be >= 1")
        if n_segments < 1:
            raise ValueError("n_segments must be >= 1")

        rng = (np.random.default_rng(seed) if seed is not None
               else np.random.default_rng())

        # Seed the intent WM from the prompt
        self.intent_wm.reset()
        self.intent_wm.add(self._summarise_intent(prompt))

        segments: list[str] = []
        confidences: list[list[float]] = []

        for _ in range(n_segments):
            # 1. Predict next intent
            ctx = self.intent_wm.context()
            intent_vec = self.intent_core.predict(ctx)

            # 2. Generate a surface segment
            tokens, confs = self._generate_segment(
                intent_vec=intent_vec,
                max_len=segment_max_len,
                temperature=temperature,
                top_k=top_k,
                stop_on_eos=stop_on_eos,
                min_confidence=min_confidence,
                rng=rng,
            )
            text = self.tokenizer.decode(tokens)
            segments.append(text)
            confidences.append(confs)

            # 3. Feedback: summary of the *actually generated* text
            #    (not the predicted intent) flows back into the intent WM.
            #    Using the generated text is important because surface
            #    noise / misreading should perturb the next thought.
            if text:
                feedback_summary = self._summarise_intent(text)
            else:
                feedback_summary = intent_vec  # no surface output → use intent
            self.intent_wm.add(feedback_summary)

        return StreamResult(
            segments=segments,
            confidences=confidences,
            intent_omega=self.intent_core.omega,
            surface_omega=self.surface_core.omega,
        )
