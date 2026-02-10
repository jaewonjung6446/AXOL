"""Tests for the pluggable array backend system."""

import numpy as np
import pytest

from axol.core.backend import get_backend, set_backend, get_backend_name, to_numpy


# ═══════════════════════════════════════════════════════════════════════════
# 1. Backend Basics
# ═══════════════════════════════════════════════════════════════════════════

class TestBackendBasics:
    def setup_method(self):
        """Reset to numpy before each test."""
        set_backend("numpy")

    def test_default_is_numpy(self):
        assert get_backend_name() == "numpy"
        assert get_backend() is np

    def test_set_and_get(self):
        set_backend("numpy")
        assert get_backend_name() == "numpy"

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            set_backend("tensorflow")

    def test_to_numpy_passthrough(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = to_numpy(arr)
        assert result is arr  # same object, no copy

    def test_to_numpy_from_list(self):
        result = to_numpy([1, 2, 3])
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1, 2, 3])


# ═══════════════════════════════════════════════════════════════════════════
# 2. CuPy Backend (skipped if not installed)
# ═══════════════════════════════════════════════════════════════════════════

class TestCuPyBackend:
    def setup_method(self):
        set_backend("numpy")

    def test_cupy_set(self):
        cupy = pytest.importorskip("cupy")
        set_backend("cupy")
        assert get_backend_name() == "cupy"

    def test_cupy_to_numpy(self):
        cupy = pytest.importorskip("cupy")
        arr = cupy.array([1.0, 2.0, 3.0])
        result = to_numpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def teardown_method(self):
        set_backend("numpy")


# ═══════════════════════════════════════════════════════════════════════════
# 3. JAX Backend (skipped if not installed)
# ═══════════════════════════════════════════════════════════════════════════

class TestJAXBackend:
    def setup_method(self):
        set_backend("numpy")

    def test_jax_set(self):
        jax = pytest.importorskip("jax")
        set_backend("jax")
        assert get_backend_name() == "jax"

    def test_jax_to_numpy(self):
        jax = pytest.importorskip("jax")
        import jax.numpy as jnp
        arr = jnp.array([1.0, 2.0, 3.0])
        result = to_numpy(arr)
        assert isinstance(result, np.ndarray)

    def teardown_method(self):
        set_backend("numpy")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Numpy Backend with Axol Programs
# ═══════════════════════════════════════════════════════════════════════════

class TestNumpyBackendPrograms:
    """Verify existing programs work correctly under numpy backend."""

    def setup_method(self):
        set_backend("numpy")

    def test_state_machine(self):
        from axol.core.dsl import parse
        from axol.core.program import run_program

        source = "@fsm\ns state=onehot(0,3)\n: advance=transform(state;M=[0 1 0;0 0 1;0 0 1])\n? done state[2]>=1"
        result = run_program(parse(source))
        assert result.terminated_by == "terminal_condition"
        assert result.final_state["state"].to_list() == pytest.approx([0, 0, 1])

    def test_hp_decay(self):
        from axol.core.dsl import parse
        from axol.core.program import run_program

        source = "@hp\ns hp=[100] round=[0] one=[1]\n: d=transform(hp;M=[0.8])\n: t=merge(round one;w=[1 1])->round\n? done round>=3"
        result = run_program(parse(source))
        hp = float(result.final_state["hp"].data[0])
        assert hp == pytest.approx(51.2, abs=0.1)

    def test_to_list_returns_python_list(self):
        from axol.core.types import FloatVec
        v = FloatVec.from_list([1.0, 2.0, 3.0])
        result = v.to_list()
        assert isinstance(result, list)
        assert result == [1.0, 2.0, 3.0]

    def test_vec_equality(self):
        from axol.core.types import FloatVec
        v1 = FloatVec.from_list([1.0, 2.0])
        v2 = FloatVec.from_list([1.0, 2.0])
        assert v1 == v2

    def test_vec_hash(self):
        from axol.core.types import FloatVec
        v1 = FloatVec.from_list([1.0, 2.0])
        v2 = FloatVec.from_list([1.0, 2.0])
        assert hash(v1) == hash(v2)
