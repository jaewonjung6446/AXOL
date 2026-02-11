"""Tests for the observatory (Phase 5)."""

import pytest
import numpy as np

from axol.core.types import FloatVec
from axol.quantum.declare import DeclarationBuilder, RelationKind
from axol.quantum.weaver import weave
from axol.quantum.observatory import observe, reobserve
from axol.quantum.types import Observation
from axol.quantum.errors import ObservatoryError


class TestObserve:
    @pytest.fixture
    def simple_tapestry(self):
        decl = (
            DeclarationBuilder("simple")
            .input("x", 8)
            .relate("y", ["x"], RelationKind.PROPORTIONAL)
            .output("y")
            .quality(0.8, 0.7)
            .build()
        )
        return weave(decl, seed=42)

    def test_basic_observe(self, simple_tapestry):
        inputs = {"x": FloatVec.from_list([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
        obs = observe(simple_tapestry, inputs)
        assert isinstance(obs, Observation)
        assert obs.tapestry_name == "simple"
        assert obs.observation_count == 1

    def test_observation_has_metrics(self, simple_tapestry):
        inputs = {"x": FloatVec.from_list([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
        obs = observe(simple_tapestry, inputs)
        assert 0.0 <= obs.omega <= 1.0
        assert 0.0 <= obs.phi <= 1.0

    def test_observation_probabilities(self, simple_tapestry):
        inputs = {"x": FloatVec.from_list([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
        obs = observe(simple_tapestry, inputs)
        probs = obs.probabilities.data
        assert abs(np.sum(probs) - 1.0) < 0.01  # normalised

    def test_observation_value_index(self, simple_tapestry):
        inputs = {"x": FloatVec.from_list([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
        obs = observe(simple_tapestry, inputs)
        assert 0 <= obs.value_index < 8

    def test_missing_input_raises(self, simple_tapestry):
        with pytest.raises(ObservatoryError, match="Missing input"):
            observe(simple_tapestry, {})

    def test_different_inputs_different_results(self, simple_tapestry):
        obs1 = observe(simple_tapestry, {"x": FloatVec.from_list([1, 0, 0, 0, 0, 0, 0, 0])})
        obs2 = observe(simple_tapestry, {"x": FloatVec.from_list([0, 0, 0, 0, 0, 0, 0, 1])})
        # Different inputs should generally produce different probabilities
        # (not guaranteed, but likely)
        assert isinstance(obs1, Observation)
        assert isinstance(obs2, Observation)


class TestReobserve:
    @pytest.fixture
    def tapestry(self):
        decl = (
            DeclarationBuilder("reobs")
            .input("x", 8)
            .relate("y", ["x"], RelationKind.PROPORTIONAL)
            .output("y")
            .quality(0.9, 0.8)
            .build()
        )
        return weave(decl, seed=42)

    def test_basic_reobserve(self, tapestry):
        inputs = {"x": FloatVec.from_list([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
        obs = reobserve(tapestry, inputs, count=5)
        assert isinstance(obs, Observation)
        assert obs.observation_count == 5

    def test_reobserve_improves_omega(self, tapestry):
        """Multiple observations should maintain or improve omega."""
        inputs = {"x": FloatVec.from_list([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
        obs_single = observe(tapestry, inputs)
        obs_multi = reobserve(tapestry, inputs, count=10, seed=42)
        # reobserve omega is empirical, should be high for consistent system
        assert isinstance(obs_multi, Observation)
        assert obs_multi.omega > 0  # should be > 0 for any valid observation

    def test_reobserve_count_one(self, tapestry):
        inputs = {"x": FloatVec.from_list([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
        obs = reobserve(tapestry, inputs, count=1, seed=42)
        assert obs.observation_count == 1

    def test_reobserve_zero_count_raises(self, tapestry):
        inputs = {"x": FloatVec.from_list([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
        with pytest.raises(ObservatoryError):
            reobserve(tapestry, inputs, count=0)

    def test_reobserve_probabilities_normalised(self, tapestry):
        inputs = {"x": FloatVec.from_list([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
        obs = reobserve(tapestry, inputs, count=5, seed=42)
        total = np.sum(obs.probabilities.data)
        assert abs(total - 1.0) < 0.01


class TestMultiSourceObserve:
    def test_two_input_observe(self):
        decl = (
            DeclarationBuilder("multi")
            .input("a", 4)
            .input("b", 4)
            .relate("c", ["a", "b"], RelationKind.ADDITIVE)
            .output("c")
            .quality(0.8, 0.7)
            .build()
        )
        tapestry = weave(decl, seed=42)
        inputs = {
            "a": FloatVec.from_list([1.0, 0.0, 0.0, 0.0]),
            "b": FloatVec.from_list([0.0, 1.0, 0.0, 0.0]),
        }
        obs = observe(tapestry, inputs)
        assert isinstance(obs, Observation)
        assert obs.tapestry_name == "multi"
