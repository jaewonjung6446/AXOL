"""Tests for fit_data parameter in weave() — Reservoir Computing readout."""

import numpy as np
import pytest

from axol.core.types import FloatVec
from axol.quantum.declare import DeclarationBuilder, RelationKind
from axol.quantum.weaver import weave
from axol.quantum.observatory import observe
from axol.quantum.types import Tapestry


def _xor_dataset(dim: int = 8):
    """XOR problem: 4 patterns, 2 classes.

    Includes polynomial cross-term (x1*x2) so that XOR is linearly separable
    from the feature vector.  This mirrors the Reservoir Computing convention:
    the reservoir (or explicit feature engineering) provides nonlinear features,
    and the readout is a single-shot linear solve.
    """
    X = np.zeros((4, dim), dtype=np.float32)
    patterns = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    for i, (x1, x2) in enumerate(patterns):
        X[i, 0] = x1
        X[i, 1] = x2
        X[i, 2] = x1 * x2  # cross-term makes XOR linearly separable
        X[i, 3] = 1.0       # bias term
    targets = np.array([0, 1, 1, 0], dtype=np.int64)
    return {"input": X, "target": targets}


def _multiclass_dataset(n_classes: int = 4, samples_per_class: int = 5, dim: int = 8):
    """Multi-class dataset: each class has a distinct centroid."""
    rng = np.random.default_rng(123)
    N = n_classes * samples_per_class
    X = np.zeros((N, dim), dtype=np.float32)
    targets = np.zeros(N, dtype=np.int64)
    for c in range(n_classes):
        centroid = np.zeros(dim, dtype=np.float32)
        centroid[c % dim] = 3.0  # strong signal on one dimension
        for i in range(samples_per_class):
            idx = c * samples_per_class + i
            X[idx] = centroid + rng.standard_normal(dim).astype(np.float32) * 0.1
            targets[idx] = c
    return {"input": X, "target": targets}


class TestFitDataXOR:
    """XOR with fit_data should achieve 100% training accuracy."""

    def _build_decl(self, dim: int = 8):
        return (
            DeclarationBuilder("xor_fit")
            .input("x", dim, labels={0: "class_0", 1: "class_1"})
            .relate("y", ["x"], RelationKind.PROPORTIONAL)
            .output("y")
            .quality(0.9, 0.8)
            .build()
        )

    def test_xor_100_accuracy(self):
        dim = 8
        decl = self._build_decl(dim)
        fd = _xor_dataset(dim)
        tapestry = weave(decl, seed=42, fit_data=fd)

        assert tapestry._fit_info is not None
        assert tapestry._fit_info["n_samples"] == 4
        assert tapestry._fit_info["n_classes"] == 2
        assert tapestry._fit_info["accuracy"] == 1.0

    def test_composed_matrix_updated(self):
        dim = 8
        decl = self._build_decl(dim)
        fd = _xor_dataset(dim)
        tapestry = weave(decl, seed=42, fit_data=fd)

        # composed_matrix must exist (fit injects it)
        assert tapestry._composed_matrix is not None

    def test_observe_fast_path(self):
        """observe() should use the composed_matrix fast path after fit."""
        dim = 8
        decl = self._build_decl(dim)
        fd = _xor_dataset(dim)
        tapestry = weave(decl, seed=42, fit_data=fd)

        # Test all 4 XOR patterns
        for i in range(4):
            vec = FloatVec(data=fd["input"][i])
            obs = observe(tapestry, {"x": vec})
            assert obs.value_index == fd["target"][i], (
                f"Pattern {i}: expected class {fd['target'][i]}, got {obs.value_index}"
            )


class TestFitDataMultiClass:
    """Multi-class (4 classes, dim=8) should get 100% training accuracy."""

    def test_multiclass_100_accuracy(self):
        dim = 8
        decl = (
            DeclarationBuilder("multi_fit")
            .input("x", dim)
            .relate("y", ["x"], RelationKind.PROPORTIONAL)
            .output("y")
            .build()
        )
        fd = _multiclass_dataset(n_classes=4, samples_per_class=5, dim=dim)
        tapestry = weave(decl, seed=42, fit_data=fd)

        assert tapestry._fit_info is not None
        assert tapestry._fit_info["n_classes"] == 4
        assert tapestry._fit_info["accuracy"] == 1.0

    def test_multiclass_observe_all_correct(self):
        dim = 8
        decl = (
            DeclarationBuilder("multi_obs")
            .input("x", dim)
            .relate("y", ["x"], RelationKind.PROPORTIONAL)
            .output("y")
            .build()
        )
        fd = _multiclass_dataset(n_classes=4, samples_per_class=5, dim=dim)
        tapestry = weave(decl, seed=42, fit_data=fd)

        correct = 0
        N = len(fd["target"])
        for i in range(N):
            vec = FloatVec(data=fd["input"][i])
            obs = observe(tapestry, {"x": vec})
            if obs.value_index == fd["target"][i]:
                correct += 1
        assert correct == N, f"Expected {N}/{N} correct, got {correct}/{N}"


class TestFitDataMetadata:
    """Verify fit_info metadata is populated correctly."""

    def test_fit_info_fields(self):
        dim = 8
        decl = (
            DeclarationBuilder("meta")
            .input("x", dim)
            .relate("y", ["x"], RelationKind.PROPORTIONAL)
            .output("y")
            .build()
        )
        fd = _xor_dataset(dim)
        tapestry = weave(decl, seed=42, fit_data=fd)

        info = tapestry._fit_info
        assert info is not None
        assert "n_samples" in info
        assert "n_classes" in info
        assert "accuracy" in info
        assert "method" in info
        assert info["method"] in ("reservoir", "end_to_end")

    def test_fit_info_none_without_fit_data(self):
        dim = 8
        decl = (
            DeclarationBuilder("no_fit")
            .input("x", dim)
            .relate("y", ["x"], RelationKind.PROPORTIONAL)
            .output("y")
            .build()
        )
        tapestry = weave(decl, seed=42)
        assert tapestry._fit_info is None


class TestFitDataBackwardCompat:
    """Existing weave (fit_data=None) must behave identically."""

    def test_no_fit_data_unchanged(self):
        decl = (
            DeclarationBuilder("compat")
            .input("x", 8)
            .relate("y", ["x"], RelationKind.PROPORTIONAL)
            .output("y")
            .quality(0.8, 0.7)
            .build()
        )
        t1 = weave(decl, seed=42)
        t2 = weave(decl, seed=42, fit_data=None)

        assert t1.weaver_report.estimated_omega == t2.weaver_report.estimated_omega
        assert t1.weaver_report.estimated_phi == t2.weaver_report.estimated_phi
        # composed matrix should be identical (both from linear chain compose)
        if t1._composed_matrix is not None and t2._composed_matrix is not None:
            np.testing.assert_array_equal(
                t1._composed_matrix.data, t2._composed_matrix.data
            )
