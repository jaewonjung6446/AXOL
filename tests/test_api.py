"""Tests for the Axol tool-use API (dispatch + tool definitions)."""

import pytest

from axol.api.dispatch import dispatch, get_tool_definitions


# ═══════════════════════════════════════════════════════════════════════════
# 1. Parse Action
# ═══════════════════════════════════════════════════════════════════════════

class TestParseAction:
    def test_parse_counter(self):
        r = dispatch({"action": "parse", "source": "@counter\ns count=[0] one=[1]\n: tick=merge(count one;w=[1 1])->count\n? done count>=5"})
        assert r["program_name"] == "counter"
        assert "count" in r["state_keys"]
        assert r["has_terminal"] is True

    def test_parse_no_terminal(self):
        r = dispatch({"action": "parse", "source": "@pipe\ns v=[1]\n: t=transform(v;M=[2])"})
        assert r["has_terminal"] is False
        assert r["transition_count"] == 1

    def test_parse_missing_source(self):
        r = dispatch({"action": "parse"})
        assert "error" in r

    def test_parse_invalid_source(self):
        r = dispatch({"action": "parse", "source": "invalid stuff"})
        assert "error" in r


# ═══════════════════════════════════════════════════════════════════════════
# 2. Run Action
# ═══════════════════════════════════════════════════════════════════════════

class TestRunAction:
    def test_run_counter(self):
        r = dispatch({"action": "run", "source": "@counter\ns count=[0] one=[1]\n: tick=merge(count one;w=[1 1])->count\n? done count>=5"})
        assert r["terminated_by"] == "terminal_condition"
        assert r["final_state"]["count"] == pytest.approx([5.0])

    def test_run_with_optimize(self):
        r = dispatch({
            "action": "run",
            "source": "@pipe\ns v=[1 0 0]\n: t1=transform(v;M=[0 1 0;0 0 1;1 0 0])\n: t2=transform(v;M=[2 0 0;0 2 0;0 0 2])",
            "optimize": True,
        })
        assert "final_state" in r

    def test_run_missing_source(self):
        r = dispatch({"action": "run"})
        assert "error" in r


# ═══════════════════════════════════════════════════════════════════════════
# 3. Inspect Action
# ═══════════════════════════════════════════════════════════════════════════

class TestInspectAction:
    def test_inspect_step_0(self):
        r = dispatch({
            "action": "inspect",
            "source": "@pipe\ns v=[1]\n: t=transform(v;M=[2])",
            "step": 0,
        })
        assert r["step"] == 0
        assert r["transition_name"] == "(initial)"
        assert r["state"]["v"] == pytest.approx([1.0])

    def test_inspect_step_1(self):
        r = dispatch({
            "action": "inspect",
            "source": "@pipe\ns v=[1]\n: t=transform(v;M=[2])",
            "step": 1,
        })
        assert r["step"] == 1
        assert r["state"]["v"] == pytest.approx([2.0])

    def test_inspect_out_of_range(self):
        r = dispatch({
            "action": "inspect",
            "source": "@pipe\ns v=[1]\n: t=transform(v;M=[2])",
            "step": 100,
        })
        assert "error" in r


# ═══════════════════════════════════════════════════════════════════════════
# 4. List Ops Action
# ═══════════════════════════════════════════════════════════════════════════

class TestListOpsAction:
    def test_list_ops(self):
        r = dispatch({"action": "list_ops"})
        assert "operations" in r
        names = [op["name"] for op in r["operations"]]
        assert "transform" in names
        assert "gate" in names
        assert "merge" in names
        assert "distance" in names
        assert "route" in names

    def test_list_ops_has_descriptions(self):
        r = dispatch({"action": "list_ops"})
        for op in r["operations"]:
            assert "description" in op
            assert "params" in op


# ═══════════════════════════════════════════════════════════════════════════
# 5. Verify Action
# ═══════════════════════════════════════════════════════════════════════════

class TestVerifyAction:
    def test_verify_pass(self):
        r = dispatch({
            "action": "verify",
            "source": "@counter\ns count=[0] one=[1]\n: tick=merge(count one;w=[1 1])->count\n? done count>=5",
            "expected": {"count": [5.0]},
            "tolerance": 0.1,
        })
        assert r["passed"] is True

    def test_verify_fail(self):
        r = dispatch({
            "action": "verify",
            "source": "@counter\ns count=[0] one=[1]\n: tick=merge(count one;w=[1 1])->count\n? done count>=5",
            "expected": {"count": [999.0]},
            "tolerance": 0.1,
        })
        assert r["passed"] is False

    def test_verify_missing_fields(self):
        r = dispatch({"action": "verify", "source": "@p\ns v=[1]\n: t=transform(v;M=[2])"})
        assert "error" in r


# ═══════════════════════════════════════════════════════════════════════════
# 6. Dispatch Errors
# ═══════════════════════════════════════════════════════════════════════════

class TestDispatchErrors:
    def test_missing_action(self):
        r = dispatch({})
        assert "error" in r

    def test_unknown_action(self):
        r = dispatch({"action": "nonexistent"})
        assert "error" in r
        assert "nonexistent" in r["error"]


# ═══════════════════════════════════════════════════════════════════════════
# 7. Tool Definitions
# ═══════════════════════════════════════════════════════════════════════════

class TestToolDefinitions:
    def test_tool_definitions_list(self):
        defs = get_tool_definitions()
        assert isinstance(defs, list)
        assert len(defs) == 5

    def test_tool_definitions_structure(self):
        defs = get_tool_definitions()
        for d in defs:
            assert "name" in d
            assert "description" in d
            assert "parameters" in d
            assert d["parameters"]["type"] == "object"

    def test_tool_names(self):
        defs = get_tool_definitions()
        names = {d["name"] for d in defs}
        assert names == {"parse", "run", "inspect", "list_ops", "verify"}
