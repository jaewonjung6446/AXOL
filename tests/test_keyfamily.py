"""Tests for KeyFamily — deterministic multi-dimension key derivation.

Verifies that KeyFamily correctly derives orthogonal keys from a single seed,
producing deterministic, dimension-independent, and mathematically valid results.
"""

import pytest
import numpy as np

from axol.core.encryption import KeyFamily


class TestKeyFamily:

    def test_deterministic(self):
        """Same seed + dim always produces the identical key matrix."""
        kf1 = KeyFamily(seed=42)
        kf2 = KeyFamily(seed=42)
        for dim in (3, 5, 10):
            np.testing.assert_allclose(kf1.key(dim), kf2.key(dim), atol=1e-5)

    def test_different_seeds(self):
        """Different seeds must produce different keys for the same dim."""
        kf_a = KeyFamily(seed=1)
        kf_b = KeyFamily(seed=2)
        for dim in (3, 8):
            assert not np.allclose(kf_a.key(dim), kf_b.key(dim), atol=1e-3), (
                f"Keys from different seeds should differ (dim={dim})"
            )

    def test_different_dims(self):
        """Same seed but different dims must produce independent keys.

        Since the matrices have different shapes we cannot directly compare
        element-wise; instead we verify that the derived seeds differ, which
        guarantees different RNG streams.
        """
        kf = KeyFamily(seed=99)
        seed_3 = kf._derived_seed(3)
        seed_5 = kf._derived_seed(5)
        assert seed_3 != seed_5, (
            "Derived seeds for different dims must differ"
        )

    def test_orthogonal(self):
        """K @ K.T should approximate the identity matrix (orthogonality)."""
        kf = KeyFamily(seed=7)
        for dim in (3, 8, 16):
            K = kf.key(dim)
            product = K @ K.T
            np.testing.assert_allclose(product, np.eye(dim), atol=1e-5)

    def test_inv_is_transpose(self):
        """kf.inv(n) must equal kf.key(n).T for orthogonal keys."""
        kf = KeyFamily(seed=12)
        for dim in (4, 10):
            np.testing.assert_allclose(kf.inv(dim), kf.key(dim).T, atol=1e-5)

    def test_inverse_property(self):
        """K @ K_inv should approximate the identity matrix."""
        kf = KeyFamily(seed=55)
        for dim in (3, 7, 20):
            K = kf.key(dim)
            K_inv = kf.inv(dim)
            np.testing.assert_allclose(K @ K_inv, np.eye(dim), atol=1e-5)

    def test_frozen(self):
        """KeyFamily is a frozen dataclass; attribute assignment must raise."""
        kf = KeyFamily(seed=0)
        with pytest.raises(AttributeError):
            kf.seed = 999

    @pytest.mark.parametrize("dim", [1, 2, 3, 5, 10, 50])
    def test_various_dims(self, dim):
        """KeyFamily.key() returns a valid orthogonal matrix for various dims."""
        kf = KeyFamily(seed=42)
        K = kf.key(dim)
        assert K.shape == (dim, dim), f"Expected shape ({dim}, {dim}), got {K.shape}"
        np.testing.assert_allclose(K @ K.T, np.eye(dim), atol=1e-5)
