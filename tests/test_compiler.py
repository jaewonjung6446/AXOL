"""Tests for axol.core.compiler."""

import numpy as np
import pytest

from axol.core.compiler import fn_to_matrix, truth_table_to_matrix
from axol.core.types import OneHotVec, TransMatrix


class TestFnToMatrix:
    def test_fn_identity(self):
        """fn_to_matrix with identity function produces identity matrix."""
        M = fn_to_matrix(lambda x: x, 4, 4)
        assert M.shape == (4, 4)
        np.testing.assert_array_equal(M.data, np.eye(4, dtype=np.float32))

    def test_fn_shift(self):
        """fn_to_matrix with cyclic shift produces correct permutation matrix."""
        M = fn_to_matrix(lambda x: (x + 1) % 3, 3, 3)
        expected = np.array(
            [[0, 1, 0],
             [0, 0, 1],
             [1, 0, 0]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(M.data, expected)
        # Verify OneHot(0) @ M = OneHot(1)
        v0 = OneHotVec.from_index(0, 3)
        result = v0.data @ M.data
        np.testing.assert_array_equal(result, OneHotVec.from_index(1, 3).data)

    def test_fn_expand(self):
        """fn_to_matrix with rectangular output (input_size=3, output_size=5)."""
        M = fn_to_matrix(lambda x: x * 2, 3, 5)
        assert M.shape == (3, 5)
        expected = np.zeros((3, 5), dtype=np.float32)
        expected[0, 0] = 1.0  # fn(0) = 0
        expected[1, 2] = 1.0  # fn(1) = 2
        expected[2, 4] = 1.0  # fn(2) = 4
        np.testing.assert_array_equal(M.data, expected)

    def test_fn_out_of_range_error(self):
        """fn returning value >= output_size raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            fn_to_matrix(lambda x: x + 3, 3, 3)


class TestTruthTableToMatrix:
    def test_truth_table_basic(self):
        """truth_table_to_matrix with cyclic mapping {0:1, 1:2, 2:0}."""
        M = truth_table_to_matrix({0: 1, 1: 2, 2: 0}, 3, 3)
        expected = np.array(
            [[0, 1, 0],
             [0, 0, 1],
             [1, 0, 0]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(M.data, expected)
        # Verify OneHot(i) @ M = OneHot(table[i])
        for i, j in {0: 1, 1: 2, 2: 0}.items():
            result = OneHotVec.from_index(i, 3).data @ M.data
            np.testing.assert_array_equal(result, OneHotVec.from_index(j, 3).data)

    def test_truth_table_default(self):
        """Unmapped inputs get the default output column."""
        # Only map input 0 -> output 2; inputs 1 and 2 default to output 0
        M = truth_table_to_matrix({0: 2}, 3, 3, default=0)
        assert M.shape == (3, 3)
        # Row 0: mapped to column 2
        np.testing.assert_array_equal(M.data[0], [0, 0, 1])
        # Row 1: unmapped, defaults to column 0
        np.testing.assert_array_equal(M.data[1], [1, 0, 0])
        # Row 2: unmapped, defaults to column 0
        np.testing.assert_array_equal(M.data[2], [1, 0, 0])
