"""Axol vector type system — 7 types + StateBundle."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Self

import numpy as np


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _VecBase:
    """Immutable wrapper around np.ndarray."""

    data: np.ndarray

    @property
    def size(self) -> int:
        return self.data.shape[0]

    def to_list(self) -> list:
        from axol.core.backend import to_numpy
        return to_numpy(self.data).tolist()

    def __repr__(self) -> str:
        name = type(self).__name__
        return f"{name}[{self.size}]({self.data.tolist()})"

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        from axol.core.backend import to_numpy
        return np.array_equal(to_numpy(self.data), to_numpy(other.data))  # type: ignore[union-attr]

    def __hash__(self) -> int:
        from axol.core.backend import to_numpy
        return hash((type(self).__name__, to_numpy(self.data).tobytes()))


# ---------------------------------------------------------------------------
# BinaryVec — elements in {0, 1}
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class BinaryVec(_VecBase):

    def __post_init__(self) -> None:
        if self.data.dtype != np.int8:
            object.__setattr__(self, "data", self.data.astype(np.int8))
        if not np.all((self.data == 0) | (self.data == 1)):
            raise ValueError("BinaryVec elements must be 0 or 1")

    @classmethod
    def from_list(cls, values: list[int]) -> Self:
        return cls(data=np.array(values, dtype=np.int8))

    @classmethod
    def zeros(cls, n: int) -> Self:
        return cls(data=np.zeros(n, dtype=np.int8))

    @classmethod
    def ones(cls, n: int) -> Self:
        return cls(data=np.ones(n, dtype=np.int8))


# ---------------------------------------------------------------------------
# IntVec — arbitrary integers
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class IntVec(_VecBase):

    def __post_init__(self) -> None:
        if self.data.dtype != np.int64:
            object.__setattr__(self, "data", self.data.astype(np.int64))

    @classmethod
    def from_list(cls, values: list[int]) -> Self:
        return cls(data=np.array(values, dtype=np.int64))

    @classmethod
    def zeros(cls, n: int) -> Self:
        return cls(data=np.zeros(n, dtype=np.int64))


# ---------------------------------------------------------------------------
# FloatVec — float32
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class FloatVec(_VecBase):

    def __post_init__(self) -> None:
        if self.data.dtype != np.float32:
            object.__setattr__(self, "data", self.data.astype(np.float32))

    @classmethod
    def from_list(cls, values: list[float]) -> Self:
        return cls(data=np.array(values, dtype=np.float32))

    @classmethod
    def zeros(cls, n: int) -> Self:
        return cls(data=np.zeros(n, dtype=np.float32))

    @classmethod
    def ones(cls, n: int) -> Self:
        return cls(data=np.ones(n, dtype=np.float32))


# ---------------------------------------------------------------------------
# OneHotVec — exactly one element is 1.0, rest 0.0
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class OneHotVec(_VecBase):

    def __post_init__(self) -> None:
        if self.data.dtype != np.float32:
            object.__setattr__(self, "data", self.data.astype(np.float32))
        if not (np.count_nonzero(self.data) == 1 and np.max(self.data) == 1.0):
            raise ValueError("OneHotVec must have exactly one 1.0 element")

    @classmethod
    def from_index(cls, index: int, n: int) -> Self:
        arr = np.zeros(n, dtype=np.float32)
        arr[index] = 1.0
        return cls(data=arr)

    @classmethod
    def from_list(cls, values: list[float]) -> Self:
        return cls(data=np.array(values, dtype=np.float32))

    @property
    def active_index(self) -> int:
        return int(np.argmax(self.data))


# ---------------------------------------------------------------------------
# GateVec — elements in {0.0, 1.0}
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class GateVec(_VecBase):

    def __post_init__(self) -> None:
        if self.data.dtype != np.float32:
            object.__setattr__(self, "data", self.data.astype(np.float32))
        if not np.all((self.data == 0.0) | (self.data == 1.0)):
            raise ValueError("GateVec elements must be 0.0 or 1.0")

    @classmethod
    def from_list(cls, values: list[float]) -> Self:
        return cls(data=np.array(values, dtype=np.float32))

    @classmethod
    def zeros(cls, n: int) -> Self:
        return cls(data=np.zeros(n, dtype=np.float32))

    @classmethod
    def ones(cls, n: int) -> Self:
        return cls(data=np.ones(n, dtype=np.float32))

    @property
    def all_open(self) -> bool:
        return bool(np.all(self.data == 1.0))


# ---------------------------------------------------------------------------
# TransMatrix — M×N float32 matrix
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class TransMatrix(_VecBase):

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError("TransMatrix must be 2-dimensional")
        if self.data.dtype != np.float32:
            object.__setattr__(self, "data", self.data.astype(np.float32))

    @property
    def size(self) -> int:  # type: ignore[override]
        return self.data.shape[0]

    @property
    def shape(self) -> tuple[int, int]:
        return (self.data.shape[0], self.data.shape[1])

    @classmethod
    def from_list(cls, rows: list[list[float]]) -> Self:
        return cls(data=np.array(rows, dtype=np.float32))

    @classmethod
    def identity(cls, n: int) -> Self:
        return cls(data=np.eye(n, dtype=np.float32))

    @classmethod
    def zeros(cls, m: int, n: int) -> Self:
        return cls(data=np.zeros((m, n), dtype=np.float32))

    def __repr__(self) -> str:
        m, n = self.shape
        return f"TransMatrix[{m},{n}]"


# ---------------------------------------------------------------------------
# StateBundle — named collection of vectors
# ---------------------------------------------------------------------------

@dataclass
class StateBundle:
    """Mutable dictionary of named vectors representing program state."""

    vectors: dict[str, _VecBase] = field(default_factory=dict)

    def __getitem__(self, key: str) -> _VecBase:
        return self.vectors[key]

    def __setitem__(self, key: str, value: _VecBase) -> None:
        self.vectors[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.vectors

    def keys(self):
        return self.vectors.keys()

    def items(self):
        return self.vectors.items()

    def copy(self) -> StateBundle:
        """Deep copy — safe for branching program state."""
        return StateBundle(vectors=copy.deepcopy(self.vectors))

    def get_flat_array(self) -> np.ndarray:
        """Concatenate all 1-D vectors (sorted by key) into a single float32 array."""
        parts: list[np.ndarray] = []
        for key in sorted(self.vectors.keys()):
            v = self.vectors[key]
            arr = v.data.astype(np.float32)
            if arr.ndim == 2:
                arr = arr.flatten()
            parts.append(arr)
        if not parts:
            return np.array([], dtype=np.float32)
        return np.concatenate(parts)

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}: {v!r}" for k, v in self.vectors.items())
        return f"StateBundle({{{inner}}})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StateBundle):
            return NotImplemented
        if self.vectors.keys() != other.vectors.keys():
            return False
        return all(self.vectors[k] == other.vectors[k] for k in self.vectors)
