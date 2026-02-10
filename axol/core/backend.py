"""Axol GPU backend — pluggable array backend (numpy/cupy/jax).

Usage:
    from axol.core.backend import get_backend, set_backend, to_numpy

    xp = get_backend()       # returns numpy-like module
    set_backend("cupy")      # switch to CuPy (requires cupy installed)
    arr = to_numpy(gpu_arr)  # convert any backend array to numpy
"""

from __future__ import annotations

import importlib
from types import ModuleType

import numpy as np

_VALID_BACKENDS = {"numpy", "cupy", "jax"}
_current_backend: str = "numpy"


def get_backend() -> ModuleType:
    """Return the current array backend module (numpy-compatible API)."""
    if _current_backend == "numpy":
        return np
    elif _current_backend == "cupy":
        import cupy
        return cupy
    elif _current_backend == "jax":
        import jax.numpy as jnp
        return jnp
    raise RuntimeError(f"Unknown backend: {_current_backend}")


def set_backend(name: str) -> None:
    """Set the array backend. One of: 'numpy', 'cupy', 'jax'."""
    global _current_backend
    if name not in _VALID_BACKENDS:
        raise ValueError(f"Unknown backend: {name!r}. Must be one of {_VALID_BACKENDS}")
    if name == "cupy":
        importlib.import_module("cupy")
    elif name == "jax":
        importlib.import_module("jax")
    _current_backend = name


def get_backend_name() -> str:
    """Return the name of the current backend."""
    return _current_backend


def to_numpy(arr) -> np.ndarray:
    """Convert any backend array to a numpy ndarray."""
    if isinstance(arr, np.ndarray):
        return arr
    # CuPy
    if hasattr(arr, "get"):
        return arr.get()
    # JAX
    if hasattr(arr, "__jax_array__") or type(arr).__module__.startswith("jax"):
        return np.asarray(arr)
    return np.asarray(arr)
