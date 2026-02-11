"""Tests for composition rules (Phase 3)."""

import pytest

from axol.quantum.compose import (
    compose_serial,
    compose_parallel,
    can_reuse_after_observe,
)
from axol.quantum.lyapunov import omega_from_lyapunov


class TestComposeSerial:
    def test_convergent_stays_convergent(self):
        """Two convergent stages → convergent."""
        omega, phi, lam, d = compose_serial(
            1.0, 0.9, -1.0, 0.5,
            1.0, 0.8, -0.5, 0.3,
        )
        assert lam < 0  # -1.0 + -0.5 = -1.5
        assert omega == 1.0
        assert abs(phi - 0.9 * 0.8) < 0.001
        assert abs(d - 0.8) < 0.001

    def test_chaos_accumulates(self):
        """Serial composition: lambda_total = lambda_A + lambda_B"""
        omega, phi, lam, d = compose_serial(
            0.8, 0.7, 0.5, 1.0,
            0.7, 0.6, 0.3, 0.5,
        )
        assert abs(lam - 0.8) < 0.001  # 0.5 + 0.3
        expected_omega = omega_from_lyapunov(0.8)
        assert abs(omega - expected_omega) < 0.001

    def test_omega_degrades_in_serial(self):
        """Serial composition should degrade Omega."""
        omega_a = omega_from_lyapunov(0.3)
        omega_b = omega_from_lyapunov(0.4)
        omega_total, _, _, _ = compose_serial(
            omega_a, 0.8, 0.3, 0.5,
            omega_b, 0.7, 0.4, 0.3,
        )
        # Total lambda = 0.7, omega = 1/(1+0.7) ≈ 0.588
        assert omega_total < min(omega_a, omega_b)

    def test_phi_degrades_in_serial(self):
        """Phi_total = Phi_A * Phi_B"""
        _, phi, _, _ = compose_serial(
            1.0, 0.9, -1.0, 0.0,
            1.0, 0.8, -1.0, 0.0,
        )
        assert abs(phi - 0.72) < 0.001  # 0.9 * 0.8

    def test_d_accumulates(self):
        """D_total = D_A + D_B"""
        _, _, _, d = compose_serial(
            1.0, 0.9, -1.0, 1.5,
            1.0, 0.8, -1.0, 2.0,
        )
        assert abs(d - 3.5) < 0.001


class TestComposeParallel:
    def test_parallel_min_omega(self):
        """Parallel: Omega_total = min(Omega_A, Omega_B)"""
        omega, phi, lam, d = compose_parallel(
            0.8, 0.7, 0.3, 1.0,
            0.6, 0.9, 0.5, 0.5,
        )
        assert omega == 0.6

    def test_parallel_min_phi(self):
        """Parallel: Phi_total = min(Phi_A, Phi_B)"""
        _, phi, _, _ = compose_parallel(
            0.8, 0.7, 0.3, 1.0,
            0.6, 0.9, 0.5, 0.5,
        )
        assert phi == 0.7

    def test_parallel_max_lambda(self):
        """Parallel: lambda_total = max(lambda_A, lambda_B)"""
        _, _, lam, _ = compose_parallel(
            0.8, 0.7, 0.3, 1.0,
            0.6, 0.9, 0.8, 0.5,
        )
        assert lam == 0.8

    def test_parallel_max_d(self):
        """Parallel: D_total = max(D_A, D_B)"""
        _, _, _, d = compose_parallel(
            0.8, 0.7, 0.3, 1.0,
            0.6, 0.9, 0.5, 2.0,
        )
        assert d == 2.0

    def test_parallel_weaker_link(self):
        """Parallel is limited by the weakest component."""
        omega, phi, _, _ = compose_parallel(
            0.99, 0.99, -1.0, 0.0,
            0.3, 0.2, 2.0, 3.0,
        )
        assert omega == 0.3
        assert phi == 0.2


class TestCanReuseAfterObserve:
    def test_convergent_reusable(self):
        """lambda < 0 → can reuse."""
        assert can_reuse_after_observe(-1.0) is True
        assert can_reuse_after_observe(-0.01) is True

    def test_chaotic_not_reusable(self):
        """lambda >= 0 → must re-weave."""
        assert can_reuse_after_observe(0.0) is False
        assert can_reuse_after_observe(0.5) is False
        assert can_reuse_after_observe(100.0) is False
