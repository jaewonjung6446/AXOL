"""Axol function-to-matrix compiler — convert Python functions to TransMatrix.

Enables compiling arbitrary discrete functions into matrix form:
  OneHot(i) @ M = OneHot(fn(i))  or  OneHot(i) @ M = result_vector(i)
"""

from __future__ import annotations

import numpy as np

from axol.core.types import TransMatrix


def fn_to_matrix(
    fn: callable,
    input_size: int,
    output_size: int,
) -> TransMatrix:
    """Compile a function into a TransMatrix.

    The function maps input indices to output indices:
      fn(i) -> j  where 0 <= i < input_size, 0 <= j < output_size

    The resulting matrix M satisfies:
      OneHot(i, input_size) @ M = OneHot(fn(i), output_size)

    Args:
        fn: Function mapping int -> int.
        input_size: Number of input states.
        output_size: Number of output states.

    Returns:
        TransMatrix of shape (input_size, output_size).
    """
    M = np.zeros((input_size, output_size), dtype=np.float32)
    for i in range(input_size):
        j = fn(i)
        if not (0 <= j < output_size):
            raise ValueError(
                f"fn({i}) = {j} is out of range [0, {output_size})"
            )
        M[i, j] = 1.0
    return TransMatrix(data=M)


def truth_table_to_matrix(
    table: dict[int, int],
    input_size: int,
    output_size: int,
    default: int | None = None,
) -> TransMatrix:
    """Compile a truth table (sparse mapping) into a TransMatrix.

    Args:
        table: Mapping {input_index: output_index}.
        input_size: Number of input states.
        output_size: Number of output states.
        default: Default output index for unmapped inputs.
                 If None, unmapped inputs produce zero vectors.

    Returns:
        TransMatrix of shape (input_size, output_size).
    """
    M = np.zeros((input_size, output_size), dtype=np.float32)
    for i in range(input_size):
        if i in table:
            j = table[i]
            if not (0 <= j < output_size):
                raise ValueError(
                    f"table[{i}] = {j} is out of range [0, {output_size})"
                )
            M[i, j] = 1.0
        elif default is not None:
            if not (0 <= default < output_size):
                raise ValueError(
                    f"default = {default} is out of range [0, {output_size})"
                )
            M[i, default] = 1.0
    return TransMatrix(data=M)
