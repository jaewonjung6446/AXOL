"""Tests for axol.quantum.fractal_text — fBm-driven phrase composition."""

from __future__ import annotations

import numpy as np
import pytest

from axol.quantum.fractal_text import (
    FractalResult,
    FractalTextGenerator,
    NoiseField,
)
from axol.quantum.sentence_decoder import SentenceDecoderLanguageModel


# ---------------------------------------------------------------------------
# NoiseField
# ---------------------------------------------------------------------------

class TestNoiseField:
    def test_determinism(self):
        a = NoiseField(seed=42)
        b = NoiseField(seed=42)
        for x in [0.0, 0.3, 1.7, -2.4, 10.5]:
            assert a.sample(x) == b.sample(x)

    def test_different_seeds_differ(self):
        a = NoiseField(seed=1)
        b = NoiseField(seed=2)
        # They will differ at *most* positions
        diffs = sum(1 for x in np.linspace(0, 10, 50)
                    if abs(a.sample(x) - b.sample(x)) > 1e-6)
        assert diffs > 40

    def test_range_is_minus_one_to_one(self):
        f = NoiseField(seed=7, octaves=4)
        xs = np.linspace(-20, 20, 500)
        vals = np.array([f.sample(x) for x in xs])
        assert vals.min() >= -1.0 - 1e-9
        assert vals.max() <= 1.0 + 1e-9

    def test_local_smoothness(self):
        """Adjacent positions should give similar values (small derivative)."""
        f = NoiseField(seed=3, octaves=2)
        diffs = []
        for x in np.linspace(0, 5, 100):
            d = f.sample(x + 0.01) - f.sample(x)
            diffs.append(abs(d))
        # 99th percentile step should still be small
        assert float(np.percentile(diffs, 99)) < 0.1

    def test_rejects_invalid_params(self):
        with pytest.raises(ValueError):
            NoiseField(octaves=0)
        with pytest.raises(ValueError):
            NoiseField(persistence=0.0)
        with pytest.raises(ValueError):
            NoiseField(persistence=1.5)
        with pytest.raises(ValueError):
            NoiseField(lacunarity=1.0)


# ---------------------------------------------------------------------------
# Helpers: a trained decoder fixture for the composer tests
# ---------------------------------------------------------------------------

def _trained_decoder() -> SentenceDecoderLanguageModel:
    pairs = [
        ("actions", "actions speak louder than words"),
        ("fortune", "fortune favors the bold"),
        ("time",    "time heals all wounds"),
        ("birds",   "birds of a feather flock together"),
        ("practice","practice makes perfect"),
    ]
    m = SentenceDecoderLanguageModel(
        vocab="abcdefghijklmnopqrstuvwxyz ", intent_dim=16,
        regularization=1e-4, seed=0,
    )
    m.train_pairs(pairs, epochs=4)
    return m


# ---------------------------------------------------------------------------
# FractalTextGenerator
# ---------------------------------------------------------------------------

class TestFractalGenerator:
    def test_zero_noise_reproduces_snap(self):
        dec = _trained_decoder()
        fg = FractalTextGenerator(dec)
        snap = dec.generate("actions").text
        res = fg.compose("actions", noise_strength=0.0)
        assert res.text == snap
        assert res.substitution_rate == 0.0

    def test_seed_determinism(self):
        dec = _trained_decoder()
        fg = FractalTextGenerator(dec)
        r1 = fg.compose("actions", noise_strength=0.5, seed=123)
        r2 = fg.compose("actions", noise_strength=0.5, seed=123)
        assert r1.text == r2.text
        assert r1.substitutions == r2.substitutions
        assert r1.noise_trace == r2.noise_trace

    def test_high_noise_introduces_substitutions(self):
        dec = _trained_decoder()
        fg = FractalTextGenerator(dec)
        # Try a range of seeds; at least one must trigger substitution.
        found_sub = False
        for s in range(20):
            r = fg.compose("actions", noise_strength=1.0,
                           k_anchors=5, seed=s)
            if r.substitution_rate > 0.0:
                found_sub = True
                break
        assert found_sub

    def test_substitution_monotonic_with_noise(self):
        """Averaged over many seeds, higher noise_strength -> more substitution."""
        dec = _trained_decoder()
        fg = FractalTextGenerator(dec)

        def avg_rate(ns: float) -> float:
            rates = [fg.compose("actions", noise_strength=ns, seed=s).substitution_rate
                     for s in range(25)]
            return float(np.mean(rates))

        low = avg_rate(0.2)
        high = avg_rate(1.0)
        assert high >= low

    def test_result_structure(self):
        dec = _trained_decoder()
        fg = FractalTextGenerator(dec)
        r = fg.compose("actions", noise_strength=0.5, seed=1)
        assert isinstance(r, FractalResult)
        assert r.macro_source != ""
        assert len(r.anchors) >= 1
        # substitutions length should match the macro token count
        assert len(r.substitutions) == len(r.macro_source.split())
        assert 0.0 <= r.substitution_rate <= 1.0

    def test_empty_dictionary_returns_empty(self):
        m = SentenceDecoderLanguageModel(vocab="abc", intent_dim=8)
        fg = FractalTextGenerator(m)
        r = fg.compose("x", noise_strength=0.5)
        assert r.text == ""
        assert r.anchors == []


class TestComposeParamValidation:
    def test_rejects_invalid_strengths(self):
        dec = _trained_decoder()
        fg = FractalTextGenerator(dec)
        with pytest.raises(ValueError):
            fg.compose("actions", noise_strength=-0.1)
        with pytest.raises(ValueError):
            fg.compose("actions", noise_strength=1.1)

    def test_rejects_invalid_k_anchors(self):
        dec = _trained_decoder()
        fg = FractalTextGenerator(dec)
        with pytest.raises(ValueError):
            fg.compose("actions", k_anchors=0)

    def test_rejects_invalid_chunk_scale(self):
        dec = _trained_decoder()
        fg = FractalTextGenerator(dec)
        with pytest.raises(ValueError):
            fg.compose("actions", chunk_scale=0.0)


# ---------------------------------------------------------------------------
# Variations
# ---------------------------------------------------------------------------

class TestVariations:
    def test_produces_n_results(self):
        dec = _trained_decoder()
        fg = FractalTextGenerator(dec)
        variants = fg.variations("actions", n=5, base_seed=99,
                                  noise_strength=0.7)
        assert len(variants) == 5
        assert all(isinstance(v, FractalResult) for v in variants)

    def test_base_seed_reproducible(self):
        dec = _trained_decoder()
        fg = FractalTextGenerator(dec)
        a = fg.variations("actions", n=4, base_seed=55, noise_strength=0.5)
        b = fg.variations("actions", n=4, base_seed=55, noise_strength=0.5)
        assert [v.text for v in a] == [v.text for v in b]

    def test_rejects_zero_n(self):
        dec = _trained_decoder()
        fg = FractalTextGenerator(dec)
        with pytest.raises(ValueError):
            fg.variations("actions", n=0)


# ---------------------------------------------------------------------------
# Spatial coherence: adjacent slots tend to choose the same anchor
# ---------------------------------------------------------------------------

class TestSpatialCoherence:
    def test_adjacent_slots_often_share_anchor(self):
        """Because fBm is smooth, long stretches should choose the same
        anchor index more often than uniform random would predict."""
        dec = _trained_decoder()
        fg = FractalTextGenerator(dec)

        # Use a small chunk_scale so noise changes slowly.
        r = fg.compose(
            "actions", noise_strength=1.0, chunk_scale=0.2,
            k_anchors=5, seed=2024,
        )
        subs = r.substitutions
        if len(subs) < 3:
            pytest.skip("macro too short")

        # Count adjacent equal pairs
        same = sum(1 for i in range(len(subs) - 1) if subs[i] == subs[i + 1])
        total_pairs = len(subs) - 1
        # Random baseline among {0..4}: ~0.2; we expect notably higher.
        assert same / total_pairs > 0.3
