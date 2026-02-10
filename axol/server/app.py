"""Axol FastAPI server — REST API + static frontend."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from axol.core.dsl import parse, ParseError
from axol.core.program import run_program, Program
from axol.core.optimizer import optimize
from axol.core.types import FloatVec, StateBundle
from axol.core.encryption import (
    random_key, random_orthogonal_key, encrypt_matrix, encrypt_vec, decrypt_vec,
    encrypt_program, decrypt_state,
)
from axol.api.dispatch import dispatch as api_dispatch

import numpy as np

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Axol DSL", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class SourceRequest(BaseModel):
    source: str

class RunRequest(BaseModel):
    source: str
    optimize: bool = False
    max_iterations: int = 1000

class OptimizeRequest(BaseModel):
    source: str

class EncryptRequest(BaseModel):
    source: str
    dim: int = 0
    seed: int = 42

class TokenCostRequest(BaseModel):
    source: str

class ModuleRunRequest(BaseModel):
    source: str
    module_sources: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# HTML root
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    index = STATIC_DIR / "index.html"
    return HTMLResponse(content=index.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.post("/api/parse")
async def api_parse(req: SourceRequest):
    try:
        prog = parse(req.source)
        return {
            "program_name": prog.name,
            "state_keys": list(prog.initial_state.keys()),
            "transition_count": len(prog.transitions),
            "has_terminal": prog.terminal_key is not None,
        }
    except ParseError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/api/run")
async def api_run(req: RunRequest):
    try:
        prog = parse(req.source)
        if req.max_iterations:
            prog.max_iterations = req.max_iterations
        if req.optimize:
            prog = optimize(prog)

        t0 = time.perf_counter()
        result = run_program(prog)
        elapsed = time.perf_counter() - t0

        # Build trace data
        trace = []
        for entry in result.trace:
            state_dict = {}
            for key in entry.state_after.keys():
                vec = entry.state_after[key]
                state_dict[key] = vec.to_list()
            trace.append({
                "step": entry.step,
                "transition": entry.transition_name,
                "state": state_dict,
            })

        final = {}
        for key in result.final_state.keys():
            final[key] = result.final_state[key].to_list()

        return {
            "final_state": final,
            "steps_executed": result.steps_executed,
            "terminated_by": result.terminated_by,
            "elapsed_ms": round(elapsed * 1000, 3),
            "trace": trace,
        }
    except ParseError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/optimize")
async def api_optimize(req: OptimizeRequest):
    try:
        prog = parse(req.source)
        prog_opt = optimize(prog)

        t0 = time.perf_counter()
        result_orig = run_program(prog)
        t_orig = time.perf_counter() - t0

        t0 = time.perf_counter()
        result_opt = run_program(prog_opt)
        t_opt = time.perf_counter() - t0

        return {
            "original": {
                "transition_count": len(prog.transitions),
                "steps_executed": result_orig.steps_executed,
                "elapsed_ms": round(t_orig * 1000, 3),
            },
            "optimized": {
                "transition_count": len(prog_opt.transitions),
                "steps_executed": result_opt.steps_executed,
                "elapsed_ms": round(t_opt * 1000, 3),
            },
        }
    except ParseError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/encrypt")
async def api_encrypt(req: EncryptRequest):
    try:
        prog = parse(req.source)

        # Determine dimension from first vector
        dim = req.dim
        if dim == 0:
            for key in prog.initial_state.keys():
                vec = prog.initial_state[key]
                if vec.data.ndim == 1:
                    dim = vec.data.shape[0]
                    break
        if dim == 0:
            return JSONResponse(status_code=400, content={"error": "Could not determine vector dimension"})

        K = random_key(dim, seed=req.seed)

        # Run original
        result_orig = run_program(prog)

        # Encrypt program
        prog_enc = encrypt_program(prog, K, dim)

        # Run encrypted
        result_enc = run_program(prog_enc)

        # Decrypt results
        decrypted = decrypt_state(result_enc.final_state, K)

        # Build response
        orig_state = {}
        enc_state = {}
        dec_state = {}
        for key in result_orig.final_state.keys():
            orig_state[key] = result_orig.final_state[key].to_list()
        for key in result_enc.final_state.keys():
            enc_state[key] = result_enc.final_state[key].to_list()
        for key in decrypted.keys():
            dec_state[key] = decrypted[key].to_list()

        # Matrix heatmap data (first transform matrix)
        orig_matrix = None
        enc_matrix = None
        for t in prog.transitions:
            from axol.core.program import TransformOp
            if isinstance(t.operation, TransformOp):
                orig_matrix = t.operation.matrix.data.tolist()
                break
        for t in prog_enc.transitions:
            if isinstance(t.operation, TransformOp):
                enc_matrix = t.operation.matrix.data.tolist()
                break

        return {
            "original_state": orig_state,
            "encrypted_state": enc_state,
            "decrypted_state": dec_state,
            "original_matrix": orig_matrix,
            "encrypted_matrix": enc_matrix,
            "dim": dim,
        }
    except ParseError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/examples")
async def api_examples():
    return {
        "examples": [
            {
                "name": "Counter (0→5)",
                "source": "@counter\ns count=[0] one=[1]\n: tick=merge(count one;w=[1 1])->count\n? done count>=5",
            },
            {
                "name": "3-State Machine",
                "source": "@fsm\ns state=onehot(0,3)\n: advance=transform(state;M=[0 1 0;0 0 1;0 0 1])\n? done state[2]>=1",
            },
            {
                "name": "HP Decay",
                "source": "@hp_decay\ns hp=[100] round=[0] one=[1]\n: decay=transform(hp;M=[0.8])\n: tick=merge(round one;w=[1 1])->round\n? done round>=3",
            },
            {
                "name": "Combat Pipeline",
                "source": "@combat\ns atk=[50 30] def_val=[10 5]\n: scale=transform(atk;M=[1.5 0;0 1.5])\n: dmg=merge(atk def_val;w=[1 -0.5])->dmg",
            },
            {
                "name": "Matrix Chain",
                "source": "@chain\ns v=[1 0 0]\n: step1=transform(v;M=[0 1 0;0 0 1;1 0 0])\n: step2=transform(v;M=[2 0 0;0 2 0;0 0 2])",
            },
        ],
    }


@app.get("/api/ops")
async def api_ops():
    return api_dispatch({"action": "list_ops"})


@app.post("/api/token-cost")
async def api_token_cost(req: TokenCostRequest):
    try:
        source = req.source
        token_count = len(source.split())

        prog = parse(source)
        state_keys = list(prog.initial_state.keys())
        transition_count = len(prog.transitions)

        # Estimate equivalent Python/C# tokens
        python_estimate = int(token_count * 2.2)
        csharp_estimate = int(token_count * 3.5)

        return {
            "axol_tokens": token_count,
            "python_estimate": python_estimate,
            "csharp_estimate": csharp_estimate,
            "savings_vs_python": f"{round((1 - token_count / python_estimate) * 100)}%",
            "savings_vs_csharp": f"{round((1 - token_count / csharp_estimate) * 100)}%",
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/module/run")
async def api_module_run(req: ModuleRunRequest):
    try:
        from axol.core.module import ModuleRegistry, Module

        reg = ModuleRegistry()

        # Register sub-modules
        for name, src in req.module_sources.items():
            sub_prog = parse(src)
            reg.register(Module(name=name, program=sub_prog))

        # Parse main program with registry
        prog = parse(req.source, registry=reg)
        result = run_program(prog)

        final = {}
        for key in result.final_state.keys():
            final[key] = result.final_state[key].to_list()

        return {
            "final_state": final,
            "steps_executed": result.steps_executed,
            "terminated_by": result.terminated_by,
        }
    except ParseError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
