"""Tests for the Axol FastAPI server endpoints."""

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from axol.server.app import app

client = TestClient(app)


# ═══════════════════════════════════════════════════════════════════════════
# API endpoint tests
# ═══════════════════════════════════════════════════════════════════════════

class TestServerEndpoints:
    def test_root_html(self):
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_api_parse(self):
        r = client.post("/api/parse", json={
            "source": "@test\ns v=[1]\n: t=transform(v;M=[2])"
        })
        assert r.status_code == 200
        data = r.json()
        assert data["program_name"] == "test"

    def test_api_parse_error(self):
        r = client.post("/api/parse", json={"source": "bad"})
        assert r.status_code == 400

    def test_api_run(self):
        r = client.post("/api/run", json={
            "source": "@counter\ns count=[0] one=[1]\n: tick=merge(count one;w=[1 1])->count\n? done count>=5"
        })
        assert r.status_code == 200
        data = r.json()
        assert data["terminated_by"] == "terminal_condition"
        assert data["final_state"]["count"] == pytest.approx([5.0])
        assert "trace" in data
        assert "elapsed_ms" in data

    def test_api_optimize(self):
        r = client.post("/api/optimize", json={
            "source": "@pipe\ns v=[1 0 0]\n: t1=transform(v;M=[0 1 0;0 0 1;1 0 0])\n: t2=transform(v;M=[2 0 0;0 2 0;0 0 2])"
        })
        assert r.status_code == 200
        data = r.json()
        assert "original" in data
        assert "optimized" in data

    def test_api_encrypt(self):
        r = client.post("/api/encrypt", json={
            "source": "@fsm\ns state=onehot(0,3)\n: advance=transform(state;M=[0 1 0;0 0 1;0 0 1])\n? done state[2]>=1"
        })
        assert r.status_code == 200
        data = r.json()
        assert "original_state" in data
        assert "encrypted_state" in data
        assert "decrypted_state" in data

    def test_api_examples(self):
        r = client.get("/api/examples")
        assert r.status_code == 200
        data = r.json()
        assert "examples" in data
        assert len(data["examples"]) >= 3

    def test_api_ops(self):
        r = client.get("/api/ops")
        assert r.status_code == 200
        data = r.json()
        assert "operations" in data

    def test_api_token_cost(self):
        r = client.post("/api/token-cost", json={
            "source": "@counter\ns count=[0] one=[1]\n: tick=merge(count one;w=[1 1])->count\n? done count>=5"
        })
        assert r.status_code == 200
        data = r.json()
        assert "axol_tokens" in data
        assert "python_estimate" in data

    def test_api_module_run(self):
        r = client.post("/api/module/run", json={
            "source": "@main\ns v=[1]\n: t=transform(v;M=[3])",
            "module_sources": {},
        })
        assert r.status_code == 200
        data = r.json()
        assert data["final_state"]["v"] == pytest.approx([3.0])

    def test_static_files(self):
        r = client.get("/static/style.css")
        assert r.status_code == 200
