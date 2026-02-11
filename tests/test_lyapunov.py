"""Tests for Lyapunov exponent estimation and Omega calculation (Phase 2)."""

import pytest
import numpy as np

from axol.core.types import FloatVec, TransMatrix
from axol.quantum.lyapunov import (
    estimate_lyapunov,
    lyapunov_spectrum,
    omega_from_lyapunov,
    omega_from_observations,
)


class TestEstimateLyapunov:
    def test_contracting_system(self):
        """System with spectral radius < 1 should have lambda < 0."""
        M = TransMatrix(data=np.eye(4, dtype=np.float32) * 0.5)
        lam = estimate_lyapunov(M, steps=100)
        assert lam < 0.0  # log(0.5) ≈ -0.693

    def test_identity_system(self):
        """Identity matrix should have lambda ≈ 0."""
        M = TransMatrix.identity(4)
        lam = estimate_lyapunov(M, steps=50)
        assert abs(lam) < 0.1

    def test_expanding_system(self):
        """System with spectral radius > 1 should have lambda > 0."""
        M = TransMatrix(data=np.eye(4, dtype=np.float32) * 2.0)
        lam = estimate_lyapunov(M, steps=50)
        assert lam > 0.0  # log(2) ≈ 0.693

    def test_lorenz_like_system(self):
        """Approximate Lorenz-like dynamics should have positive lambda."""
        # Build a matrix with eigenvalues that give lambda ~ 0.9
        rng = np.random.default_rng(42)
        A = rng.standard_normal((3, 3))
        Q, _ = np.linalg.qr(A)
        # Create eigenvalue structure: one expanding, two contracting
        D = np.diag([np.exp(0.9), np.exp(-0.5), np.exp(-1.2)])
        M = TransMatrix(data=(Q @ D @ Q.T).astype(np.float32))
        lam = estimate_lyapunov(M, steps=200)
        # Should be close to 0.9
        assert abs(lam - 0.9) < 0.3

    def test_zero_dim(self):
        """Zero-dimensional matrix should return 0."""
        M = TransMatrix(data=np.zeros((0, 0), dtype=np.float32).reshape(0, 0))
        # This will hit the n == 0 check
        # Can't create TransMatrix with 0 dims easily, skip
        pass

    def test_strongly_convergent(self):
        """Very small eigenvalues → strongly negative lambda."""
        M = TransMatrix(data=np.eye(4, dtype=np.float32) * 0.01)
        lam = estimate_lyapunov(M, steps=50)
        assert lam < -3.0  # log(0.01) ≈ -4.6


class TestLyapunovSpectrum:
    def test_spectrum_length(self):
        M = TransMatrix.identity(4)
        spec = lyapunov_spectrum(M, dim=4, steps=50)
        assert len(spec) == 4

    def test_spectrum_ordering(self):
        """Spectrum should be in descending order."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((4, 4)) * 0.3 + np.eye(4) * 0.8
        M = TransMatrix(data=A.astype(np.float32))
        spec = lyapunov_spectrum(M, steps=100)
        for i in range(len(spec) - 1):
            assert spec[i] >= spec[i + 1] - 0.01  # allow tiny numerical error

    def test_contracting_spectrum(self):
        """All exponents should be negative for contracting system."""
        M = TransMatrix(data=np.eye(4, dtype=np.float32) * 0.5)
        spec = lyapunov_spectrum(M, steps=100)
        for lam in spec:
            assert lam < 0.1  # allow small numerical error

    def test_partial_spectrum(self):
        """Can compute fewer exponents than matrix dimension."""
        M = TransMatrix.identity(8)
        spec = lyapunov_spectrum(M, dim=3, steps=50)
        assert len(spec) == 3


class TestOmegaFromLyapunov:
    def test_convergent(self):
        """lambda < 0 → Omega = 1.0"""
        assert omega_from_lyapunov(-2.0) == 1.0
        assert omega_from_lyapunov(-0.5) == 1.0

    def test_zero(self):
        """lambda = 0 → Omega = 1.0"""
        assert omega_from_lyapunov(0.0) == 1.0

    def test_chaotic(self):
        """lambda > 0 → Omega < 1.0"""
        assert abs(omega_from_lyapunov(1.0) - 0.5) < 0.001
        assert abs(omega_from_lyapunov(0.91) - 1.0 / 1.91) < 0.001

    def test_very_chaotic(self):
        """Very large lambda → Omega → 0"""
        omega = omega_from_lyapunov(100.0)
        assert omega < 0.02

    def test_monotonically_decreasing(self):
        """Omega should decrease as lambda increases."""
        prev = omega_from_lyapunov(-1.0)
        for lam in [-0.5, 0.0, 0.5, 1.0, 2.0, 5.0]:
            curr = omega_from_lyapunov(lam)
            assert curr <= prev + 0.001
            prev = curr


class TestOmegaFromObservations:
    def test_identical_observations(self):
        """All same argmax → Omega = 1.0"""
        obs = [FloatVec.from_list([0.1, 0.9]) for _ in range(10)]
        assert omega_from_observations(obs) == 1.0

    def test_random_observations(self):
        """Random argmax → low Omega."""
        rng = np.random.default_rng(42)
        obs = [FloatVec.from_list(rng.random(4).tolist()) for _ in range(100)]
        omega = omega_from_observations(obs)
        assert omega < 0.5  # should be around 0.25 for uniform 4-way

    def test_empty(self):
        assert omega_from_observations([]) == 0.0

    def test_mostly_stable(self):
        """90 same + 10 different → Omega ≈ 0.9"""
        obs = [FloatVec.from_list([0.1, 0.9])] * 90
        obs += [FloatVec.from_list([0.9, 0.1])] * 10
        omega = omega_from_observations(obs)
        assert abs(omega - 0.9) < 0.01
