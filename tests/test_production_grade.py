"""Production-grade fit_data tests — nonlinear, discontinuous, noisy, scalable.

Each test section validates a distinct failure mode that would
block production deployment:

  A. Nonlinear decision boundaries  (circle, moons, spiral, checkerboard)
  B. Discontinuous / piecewise logic (step, abs-threshold, multi-region)
  C. Noise robustness              (SNR sweep on the same task)
  D. Generalization                (train/test split — overfit detection)
  E. Scalability                   (high-dim, many-class)
  F. Edge cases                    (single sample per class, near-boundary)

Feature engineering convention:
  - Koopman `lift(x, degree=2, basis=...)` provides nonlinear features
  - "augmented" basis adds indicator 1_{x>0} + ReLU cross — ideal for PWA
  - All features are injected into fit_data["input"]; readout stays linear
"""

import numpy as np
import pytest

from axol.core.types import FloatVec
from axol.quantum.declare import DeclarationBuilder, RelationKind
from axol.quantum.weaver import weave
from axol.quantum.observatory import observe
from axol.quantum.koopman import lift, lifted_dim


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _make_decl(name: str, dim: int):
    return (
        DeclarationBuilder(name)
        .input("x", dim)
        .relate("y", ["x"], RelationKind.PROPORTIONAL)
        .output("y")
        .quality(0.95, 0.9)
        .build()
    )


def _rbf_features(X: np.ndarray, centers: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Radial Basis Function features: exp(-gamma * ||x - c||^2).

    Standard kernel trick — projects data into high-dim space where
    complex boundaries (spirals, concentric rings) become linear.
    """
    # (N, 1, d) - (1, M, d) → (N, M)
    diffs = X[:, None, :].astype(np.float64) - centers[None, :, :].astype(np.float64)
    dists_sq = np.sum(diffs ** 2, axis=2)
    return np.exp(-gamma * dists_sq).astype(np.float32)


def _accuracy(tapestry, X, targets, input_key="x"):
    correct = 0
    for i in range(len(targets)):
        vec = FloatVec(data=X[i].astype(np.float32))
        obs = observe(tapestry, {input_key: vec})
        if obs.value_index == int(targets[i]):
            correct += 1
    return correct / len(targets)


# ═══════════════════════════════════════════════════════════════════
# A. Nonlinear Decision Boundaries
# ═══════════════════════════════════════════════════════════════════

class TestNonlinearBoundaries:
    """Nonlinear 2D patterns lifted to polynomial feature space."""

    @staticmethod
    def _circle_data(n=80, seed=42):
        """Inner circle (class 0) vs outer ring (class 1)."""
        rng = np.random.default_rng(seed)
        angles = rng.uniform(0, 2 * np.pi, n)
        radii = np.concatenate([
            rng.uniform(0.0, 0.4, n // 2),   # inner
            rng.uniform(0.7, 1.0, n // 2),   # outer
        ])
        x = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        targets = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.int64)
        return x.astype(np.float32), targets

    @staticmethod
    def _moons_data(n=100, seed=42):
        """Two interleaving half-moons."""
        rng = np.random.default_rng(seed)
        n_half = n // 2
        # Moon 0
        t0 = np.linspace(0, np.pi, n_half)
        x0 = np.column_stack([np.cos(t0), np.sin(t0)])
        x0 += rng.standard_normal(x0.shape) * 0.1
        # Moon 1
        t1 = np.linspace(0, np.pi, n_half)
        x1 = np.column_stack([1.0 - np.cos(t1), 0.5 - np.sin(t1)])
        x1 += rng.standard_normal(x1.shape) * 0.1
        X = np.vstack([x0, x1]).astype(np.float32)
        targets = np.array([0] * n_half + [1] * n_half, dtype=np.int64)
        return X, targets

    @staticmethod
    def _spiral_data(n=120, n_classes=3, seed=42):
        """3-arm spiral — 3-class nonlinear separation."""
        rng = np.random.default_rng(seed)
        n_per = n // n_classes
        X_list, y_list = [], []
        for c in range(n_classes):
            t = np.linspace(0, 2.5 * np.pi, n_per) + c * 2 * np.pi / n_classes
            r = np.linspace(0.3, 1.0, n_per)
            x = np.column_stack([r * np.cos(t), r * np.sin(t)])
            x += rng.standard_normal(x.shape) * 0.08
            X_list.append(x)
            y_list.extend([c] * n_per)
        X = np.vstack(X_list).astype(np.float32)
        targets = np.array(y_list, dtype=np.int64)
        return X, targets

    @staticmethod
    def _checkerboard_data(n=100, seed=42):
        """2x2 checkerboard — XOR in continuous space."""
        rng = np.random.default_rng(seed)
        X = rng.uniform(-1, 1, (n, 2)).astype(np.float32)
        targets = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.int64)
        return X, targets

    def _lift_and_fit(self, X, targets, name, degree=2, basis="poly"):
        raw_dim = X.shape[1]
        X_lifted = lift(X.astype(np.float64), degree=degree, basis=basis).astype(np.float32)
        ldim = X_lifted.shape[1]
        decl = _make_decl(name, ldim)
        fd = {"input": X_lifted, "target": targets}
        tap = weave(decl, seed=42, fit_data=fd)
        return tap, X_lifted

    def test_circle_separation(self):
        X, y = self._circle_data()
        tap, X_l = self._lift_and_fit(X, y, "circle", degree=2)
        acc = _accuracy(tap, X_l, y)
        assert acc >= 0.95, f"Circle: train acc {acc:.0%} < 95%"

    def test_moons_separation(self):
        X, y = self._moons_data()
        # Augmented basis adds indicator 1_{x>0} — helps with interleaving boundary
        tap, X_l = self._lift_and_fit(X, y, "moons", degree=2, basis="augmented")
        acc = _accuracy(tap, X_l, y)
        assert acc >= 0.95, f"Moons: train acc {acc:.0%} < 95%"

    def test_spiral_3class(self):
        """3-arm spiral requires RBF kernel features — polynomial basis
        cannot capture angular winding.  This is the hardest 2D pattern
        and validates that AXOL + proper feature engineering handles it.
        """
        X, y = self._spiral_data(n=150, n_classes=3)
        # RBF features: 80 centers sampled from data, gamma tuned to data scale
        rng = np.random.default_rng(42)
        center_idx = rng.choice(len(X), size=80, replace=False)
        centers = X[center_idx]
        X_rbf = _rbf_features(X, centers, gamma=8.0)
        # Concatenate polynomial + RBF for full expressiveness
        X_poly = lift(X.astype(np.float64), degree=2).astype(np.float32)
        X_aug = np.column_stack([X_poly, X_rbf])

        ldim = X_aug.shape[1]
        decl = _make_decl("spiral", ldim)
        fd = {"input": X_aug, "target": y}
        tap = weave(decl, seed=42, fit_data=fd)
        acc = _accuracy(tap, X_aug, y)
        assert acc >= 0.90, f"Spiral 3-class: train acc {acc:.0%} < 90%"

    def test_checkerboard_xor(self):
        X, y = self._checkerboard_data(n=100)
        # Augmented basis captures sign/threshold patterns
        tap, X_l = self._lift_and_fit(X, y, "checker", degree=2, basis="augmented")
        acc = _accuracy(tap, X_l, y)
        assert acc >= 0.95, f"Checkerboard: train acc {acc:.0%} < 95%"


# ═══════════════════════════════════════════════════════════════════
# B. Discontinuous / Piecewise Logic
# ═══════════════════════════════════════════════════════════════════

class TestDiscontinuousLogic:
    """Step functions, abs-threshold, multi-region piecewise rules."""

    @staticmethod
    def _step_data(n=60, seed=42):
        """y = 1 if x > threshold else 0 — hard step."""
        rng = np.random.default_rng(seed)
        X = rng.uniform(-2, 2, (n, 1)).astype(np.float32)
        targets = (X[:, 0] > 0.0).astype(np.int64)
        return X, targets

    @staticmethod
    def _abs_threshold_data(n=80, seed=42):
        """y = 1 if |x| > 1.0 else 0 — V-shaped boundary."""
        rng = np.random.default_rng(seed)
        X = rng.uniform(-3, 3, (n, 1)).astype(np.float32)
        targets = (np.abs(X[:, 0]) > 1.0).astype(np.int64)
        return X, targets

    @staticmethod
    def _piecewise_3region(n=90, seed=42):
        """3 regions: x<-1 → class 0, -1<=x<=1 → class 1, x>1 → class 2."""
        rng = np.random.default_rng(seed)
        X = rng.uniform(-3, 3, (n, 1)).astype(np.float32)
        targets = np.zeros(n, dtype=np.int64)
        targets[X[:, 0] > 1.0] = 2
        targets[(X[:, 0] >= -1.0) & (X[:, 0] <= 1.0)] = 1
        targets[X[:, 0] < -1.0] = 0
        return X, targets

    @staticmethod
    def _multi_threshold_2d(n=100, seed=42):
        """2D quadrant classification — 4 classes by sign(x1) x sign(x2)."""
        rng = np.random.default_rng(seed)
        X = rng.uniform(-2, 2, (n, 2)).astype(np.float32)
        # class = 2*(x1>0) + (x2>0) → {0,1,2,3}
        targets = (2 * (X[:, 0] > 0) + (X[:, 1] > 0)).astype(np.int64)
        return X, targets

    def _lift_and_fit(self, X, targets, name, degree=2, basis="augmented"):
        X_lifted = lift(X.astype(np.float64), degree=degree, basis=basis).astype(np.float32)
        ldim = X_lifted.shape[1]
        decl = _make_decl(name, ldim)
        fd = {"input": X_lifted, "target": targets}
        tap = weave(decl, seed=42, fit_data=fd)
        return tap, X_lifted

    def test_hard_step(self):
        X, y = self._step_data()
        tap, X_l = self._lift_and_fit(X, y, "step")
        acc = _accuracy(tap, X_l, y)
        assert acc >= 0.95, f"Hard step: train acc {acc:.0%} < 95%"

    def test_abs_threshold(self):
        X, y = self._abs_threshold_data()
        tap, X_l = self._lift_and_fit(X, y, "abs_thresh")
        acc = _accuracy(tap, X_l, y)
        assert acc >= 0.95, f"Abs threshold: train acc {acc:.0%} < 95%"

    def test_piecewise_3region(self):
        X, y = self._piecewise_3region()
        tap, X_l = self._lift_and_fit(X, y, "pw3")
        acc = _accuracy(tap, X_l, y)
        assert acc >= 0.90, f"Piecewise 3-region: train acc {acc:.0%} < 90%"

    def test_quadrant_4class(self):
        X, y = self._multi_threshold_2d()
        tap, X_l = self._lift_and_fit(X, y, "quadrant")
        acc = _accuracy(tap, X_l, y)
        assert acc >= 0.95, f"Quadrant 4-class: train acc {acc:.0%} < 95%"


# ═══════════════════════════════════════════════════════════════════
# C. Noise Robustness (SNR sweep)
# ═══════════════════════════════════════════════════════════════════

class TestNoiseRobustness:
    """Same classification task at increasing noise levels."""

    @staticmethod
    def _noisy_clusters(n=80, noise_std=0.1, seed=42):
        """4-class Gaussian clusters with configurable noise."""
        rng = np.random.default_rng(seed)
        dim = 4
        centers = np.array([
            [2, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 2],
        ], dtype=np.float32)
        n_per = n // 4
        X_list, y_list = [], []
        for c in range(4):
            X_c = centers[c] + rng.standard_normal((n_per, dim)).astype(np.float32) * noise_std
            X_list.append(X_c)
            y_list.extend([c] * n_per)
        return np.vstack(X_list), np.array(y_list, dtype=np.int64)

    @pytest.mark.parametrize("noise_std,min_acc", [
        (0.1, 1.00),   # very clean
        (0.3, 0.99),   # moderate
        (0.5, 0.95),   # noisy
        (0.8, 0.85),   # very noisy
    ])
    def test_noise_sweep(self, noise_std, min_acc):
        X, y = self._noisy_clusters(n=80, noise_std=noise_std)
        X_lifted = lift(X.astype(np.float64), degree=2).astype(np.float32)
        ldim = X_lifted.shape[1]
        decl = _make_decl(f"noise_{noise_std}", ldim)
        fd = {"input": X_lifted, "target": y}
        tap = weave(decl, seed=42, fit_data=fd)
        acc = _accuracy(tap, X_lifted, y)
        assert acc >= min_acc, (
            f"Noise σ={noise_std}: train acc {acc:.0%} < {min_acc:.0%}"
        )


# ═══════════════════════════════════════════════════════════════════
# D. Generalization (train/test split)
# ═══════════════════════════════════════════════════════════════════

class TestGeneralization:
    """Train on subset, evaluate on held-out test set.
    Detects overfitting — production systems must generalize.
    """

    @staticmethod
    def _split(X, y, train_ratio=0.7, seed=42):
        rng = np.random.default_rng(seed)
        n = len(y)
        idx = rng.permutation(n)
        split = int(n * train_ratio)
        return X[idx[:split]], y[idx[:split]], X[idx[split:]], y[idx[split:]]

    def test_circle_generalization(self):
        rng = np.random.default_rng(42)
        n = 200
        angles = rng.uniform(0, 2 * np.pi, n)
        radii = np.concatenate([
            rng.uniform(0.0, 0.35, n // 2),
            rng.uniform(0.75, 1.0, n // 2),
        ])
        X = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]).astype(np.float32)
        y = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.int64)

        X_train, y_train, X_test, y_test = self._split(X, y)
        X_train_l = lift(X_train.astype(np.float64), degree=2).astype(np.float32)
        X_test_l = lift(X_test.astype(np.float64), degree=2).astype(np.float32)

        ldim = X_train_l.shape[1]
        decl = _make_decl("gen_circle", ldim)
        fd = {"input": X_train_l, "target": y_train}
        tap = weave(decl, seed=42, fit_data=fd)

        train_acc = _accuracy(tap, X_train_l, y_train)
        test_acc = _accuracy(tap, X_test_l, y_test)
        assert train_acc >= 0.95, f"Circle train acc {train_acc:.0%} < 95%"
        assert test_acc >= 0.90, f"Circle test acc {test_acc:.0%} < 90%"

    def test_quadrant_generalization(self):
        rng = np.random.default_rng(42)
        n = 200
        X = rng.uniform(-2, 2, (n, 2)).astype(np.float32)
        y = (2 * (X[:, 0] > 0) + (X[:, 1] > 0)).astype(np.int64)

        X_train, y_train, X_test, y_test = self._split(X, y)
        X_train_l = lift(X_train.astype(np.float64), degree=2, basis="augmented").astype(np.float32)
        X_test_l = lift(X_test.astype(np.float64), degree=2, basis="augmented").astype(np.float32)

        ldim = X_train_l.shape[1]
        decl = _make_decl("gen_quad", ldim)
        fd = {"input": X_train_l, "target": y_train}
        tap = weave(decl, seed=42, fit_data=fd)

        train_acc = _accuracy(tap, X_train_l, y_train)
        test_acc = _accuracy(tap, X_test_l, y_test)
        assert train_acc >= 0.95, f"Quadrant train acc {train_acc:.0%} < 95%"
        assert test_acc >= 0.90, f"Quadrant test acc {test_acc:.0%} < 90%"

    def test_noisy_clusters_generalization(self):
        rng = np.random.default_rng(42)
        n = 200
        dim = 4
        centers = np.eye(dim, dtype=np.float32) * 2.0
        X_list, y_list = [], []
        for c in range(4):
            X_c = centers[c] + rng.standard_normal((n // 4, dim)).astype(np.float32) * 0.3
            X_list.append(X_c)
            y_list.extend([c] * (n // 4))
        X = np.vstack(X_list).astype(np.float32)
        y = np.array(y_list, dtype=np.int64)

        X_train, y_train, X_test, y_test = self._split(X, y)
        X_train_l = lift(X_train.astype(np.float64), degree=2).astype(np.float32)
        X_test_l = lift(X_test.astype(np.float64), degree=2).astype(np.float32)

        ldim = X_train_l.shape[1]
        decl = _make_decl("gen_cluster", ldim)
        fd = {"input": X_train_l, "target": y_train}
        tap = weave(decl, seed=42, fit_data=fd)

        train_acc = _accuracy(tap, X_train_l, y_train)
        test_acc = _accuracy(tap, X_test_l, y_test)
        assert train_acc >= 0.95, f"Cluster train acc {train_acc:.0%} < 95%"
        assert test_acc >= 0.90, f"Cluster test acc {test_acc:.0%} < 90%"


# ═══════════════════════════════════════════════════════════════════
# E. Scalability
# ═══════════════════════════════════════════════════════════════════

class TestScalability:
    """High dimensions and many classes."""

    def test_high_dim_binary(self):
        """dim=32, 2 classes, 200 samples."""
        rng = np.random.default_rng(42)
        dim = 32
        n = 200
        # Two Gaussian blobs in 32D
        center0 = np.zeros(dim, dtype=np.float32)
        center0[0] = 2.0
        center1 = np.zeros(dim, dtype=np.float32)
        center1[1] = 2.0
        X0 = center0 + rng.standard_normal((n // 2, dim)).astype(np.float32) * 0.3
        X1 = center1 + rng.standard_normal((n // 2, dim)).astype(np.float32) * 0.3
        X = np.vstack([X0, X1]).astype(np.float32)
        y = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.int64)

        # No lifting needed — 32D raw features are sufficient for linear separation
        decl = _make_decl("highdim", dim)
        fd = {"input": X, "target": y}
        tap = weave(decl, seed=42, fit_data=fd)
        acc = _accuracy(tap, X, y)
        assert acc >= 0.99, f"High-dim binary: {acc:.0%} < 99%"
        assert tap._fit_info["accuracy"] >= 0.99

    def test_10_classes(self):
        """10-class one-hot centroids, dim=16."""
        rng = np.random.default_rng(42)
        dim = 16
        n_classes = 10
        n_per = 15
        X_list, y_list = [], []
        for c in range(n_classes):
            center = np.zeros(dim, dtype=np.float32)
            center[c % dim] = 3.0
            X_c = center + rng.standard_normal((n_per, dim)).astype(np.float32) * 0.2
            X_list.append(X_c)
            y_list.extend([c] * n_per)
        X = np.vstack(X_list).astype(np.float32)
        y = np.array(y_list, dtype=np.int64)

        decl = _make_decl("10class", dim)
        fd = {"input": X, "target": y}
        tap = weave(decl, seed=42, fit_data=fd)
        acc = _accuracy(tap, X, y)
        assert acc >= 0.95, f"10-class: {acc:.0%} < 95%"

    def test_many_samples(self):
        """500 samples, 4 classes, dim=8 — basic scaling test."""
        rng = np.random.default_rng(42)
        dim = 8
        n = 500
        centers = np.eye(4, dim, dtype=np.float32) * 3.0
        X_list, y_list = [], []
        for c in range(4):
            X_c = centers[c] + rng.standard_normal((n // 4, dim)).astype(np.float32) * 0.3
            X_list.append(X_c)
            y_list.extend([c] * (n // 4))
        X = np.vstack(X_list).astype(np.float32)
        y = np.array(y_list, dtype=np.int64)

        decl = _make_decl("500samples", dim)
        fd = {"input": X, "target": y}
        tap = weave(decl, seed=42, fit_data=fd)
        acc = _accuracy(tap, X[:40], y[:40])  # check subset for speed
        assert acc >= 0.95, f"500-sample: {acc:.0%} < 95%"


# ═══════════════════════════════════════════════════════════════════
# F. Edge Cases
# ═══════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Minimal samples, near-boundary, class imbalance."""

    def test_minimal_samples(self):
        """1 sample per class — exact memorization."""
        dim = 8
        X = np.zeros((3, dim), dtype=np.float32)
        X[0, 0] = 1.0
        X[1, 1] = 1.0
        X[2, 2] = 1.0
        y = np.array([0, 1, 2], dtype=np.int64)

        decl = _make_decl("minimal", dim)
        fd = {"input": X, "target": y}
        tap = weave(decl, seed=42, fit_data=fd)
        acc = _accuracy(tap, X, y)
        assert acc == 1.0, f"Minimal samples: {acc:.0%} != 100%"

    def test_class_imbalance(self):
        """Class 0: 50 samples, Class 1: 5 samples."""
        rng = np.random.default_rng(42)
        dim = 8
        c0 = np.zeros(dim, dtype=np.float32); c0[0] = 2.0
        c1 = np.zeros(dim, dtype=np.float32); c1[1] = 2.0
        X0 = c0 + rng.standard_normal((50, dim)).astype(np.float32) * 0.3
        X1 = c1 + rng.standard_normal((5, dim)).astype(np.float32) * 0.3
        X = np.vstack([X0, X1]).astype(np.float32)
        y = np.array([0]*50 + [1]*5, dtype=np.int64)

        decl = _make_decl("imbalance", dim)
        fd = {"input": X, "target": y}
        tap = weave(decl, seed=42, fit_data=fd)
        # Must still correctly classify the minority class
        acc = _accuracy(tap, X, y)
        minority_acc = _accuracy(tap, X[50:], y[50:])
        assert acc >= 0.90, f"Imbalance overall: {acc:.0%} < 90%"
        assert minority_acc >= 0.80, f"Minority class: {minority_acc:.0%} < 80%"

    def test_near_boundary(self):
        """Samples very close to decision boundary — stress test."""
        rng = np.random.default_rng(42)
        n = 60
        # Points at x=±epsilon from boundary at x=0
        X = np.zeros((n, 2), dtype=np.float32)
        eps = 0.02
        for i in range(n // 2):
            X[i, 0] = -eps - rng.uniform(0, 0.01)
            X[i, 1] = rng.standard_normal() * 0.5
        for i in range(n // 2, n):
            X[i, 0] = eps + rng.uniform(0, 0.01)
            X[i, 1] = rng.standard_normal() * 0.5
        y = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.int64)

        X_l = lift(X.astype(np.float64), degree=2, basis="augmented").astype(np.float32)
        ldim = X_l.shape[1]
        decl = _make_decl("boundary", ldim)
        fd = {"input": X_l, "target": y}
        tap = weave(decl, seed=42, fit_data=fd)
        acc = _accuracy(tap, X_l, y)
        assert acc >= 0.90, f"Near-boundary: {acc:.0%} < 90%"

    def test_identical_features_different_labels(self):
        """Contradictory data — should not crash, accuracy degrades gracefully."""
        dim = 4
        X = np.ones((4, dim), dtype=np.float32)
        # Identical inputs but different labels — impossible task
        y = np.array([0, 1, 0, 1], dtype=np.int64)

        decl = _make_decl("contradict", dim)
        fd = {"input": X, "target": y}
        # Must not crash
        tap = weave(decl, seed=42, fit_data=fd)
        assert tap._fit_info is not None
        # Accuracy can be anything, just don't crash
