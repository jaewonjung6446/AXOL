"""Tests for fractal dimension estimation and Phi calculation (Phase 2)."""

import pytest
import numpy as np

from axol.core.types import FloatVec
from axol.quantum.fractal import (
    estimate_fractal_dim,
    phi_from_fractal,
    phi_from_entropy,
)


class TestEstimateFractalDim:
    def test_single_point(self):
        """A single repeated point should have D ≈ 0."""
        points = FloatVec.from_list([1.0, 2.0] * 50)
        d = estimate_fractal_dim(points, phase_space_dim=2)
        assert d < 0.5

    def test_line_segment(self):
        """Points along a line should have D ≈ 1."""
        t = np.linspace(0, 1, 200)
        points_data = np.column_stack([t, t * 0.5]).flatten()
        points = FloatVec.from_list(points_data.tolist())
        d = estimate_fractal_dim(points, phase_space_dim=2)
        assert 0.5 < d < 1.5

    def test_filled_square(self):
        """Uniformly distributed 2D points should have D > 1."""
        rng = np.random.default_rng(42)
        pts = rng.uniform(0, 1, (500, 2)).flatten()
        points = FloatVec.from_list(pts.tolist())
        d = estimate_fractal_dim(points, phase_space_dim=2)
        assert 1.0 < d < 2.5

    def test_empty(self):
        points = FloatVec.from_list([])
        d = estimate_fractal_dim(points)
        assert d == 0.0

    def test_correlation_method(self):
        """Correlation dimension method should also work."""
        rng = np.random.default_rng(42)
        pts = rng.uniform(0, 1, (200, 2)).flatten()
        points = FloatVec.from_list(pts.tolist())
        d = estimate_fractal_dim(points, method="correlation", phase_space_dim=2)
        assert d > 0.5

    def test_invalid_method(self):
        points = FloatVec.from_list([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError):
            estimate_fractal_dim(points, method="invalid")


class TestPhiFromFractal:
    def test_fixed_point(self):
        """D = 0 → Phi = 1.0"""
        assert phi_from_fractal(0.0, 4) == 1.0

    def test_space_filling(self):
        """D = D_max → Phi = 0.5"""
        assert abs(phi_from_fractal(4.0, 4) - 0.5) < 0.001

    def test_half_filling(self):
        """D = D_max/2 → Phi = 2/3"""
        assert abs(phi_from_fractal(2.0, 4) - 2.0 / 3.0) < 0.001

    def test_zero_phase_space(self):
        """Zero phase space dim → Phi = 1.0"""
        assert phi_from_fractal(1.0, 0) == 1.0

    def test_monotonically_decreasing(self):
        """Phi decreases as D increases."""
        prev = phi_from_fractal(0.0, 10)
        for d in [0.5, 1.0, 2.0, 5.0, 10.0]:
            curr = phi_from_fractal(d, 10)
            assert curr <= prev + 0.001
            prev = curr


class TestPhiFromEntropy:
    def test_delta_distribution(self):
        """One-hot distribution (zero entropy) → Phi = 1.0"""
        probs = FloatVec.from_list([0.0, 0.0, 1.0, 0.0])
        phi = phi_from_entropy(probs)
        assert abs(phi - 1.0) < 0.01

    def test_uniform_distribution(self):
        """Uniform distribution (max entropy) → Phi ≈ 0.0"""
        n = 8
        probs = FloatVec.from_list([1.0 / n] * n)
        phi = phi_from_entropy(probs)
        assert abs(phi) < 0.05

    def test_peaked_distribution(self):
        """Nearly delta → Phi near 1.0"""
        probs = FloatVec.from_list([0.01, 0.01, 0.97, 0.01])
        phi = phi_from_entropy(probs)
        assert phi > 0.5

    def test_two_peaked(self):
        """Two equal peaks → intermediate Phi"""
        probs = FloatVec.from_list([0.5, 0.5, 0.0, 0.0])
        phi = phi_from_entropy(probs)
        assert 0.3 < phi < 0.8

    def test_single_element(self):
        """Single element → Phi = 1.0"""
        probs = FloatVec.from_list([1.0])
        phi = phi_from_entropy(probs)
        assert phi == 1.0

    def test_all_zero(self):
        """All zeros → Phi = 1.0 (degenerate)"""
        probs = FloatVec.from_list([0.0, 0.0, 0.0])
        phi = phi_from_entropy(probs)
        assert phi == 1.0
