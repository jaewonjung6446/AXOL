"""Axol Client SDK — encrypt-on-client, compute-on-server architecture.

Usage:
    client = AxolClient(seed=42, max_dim=16, use_padding=True)
    payload = client.prepare(program)     # encrypt locally
    # ... send payload to server ...
    result = client.decrypt_result(encrypted_result, payload["metadata"])

    # Or test locally:
    result = client.run_local(program)
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from axol.core.types import FloatVec, TransMatrix, StateBundle
from axol.core.program import Program, Transition, TransformOp, run_program
from axol.core.encryption import (
    KeyFamily,
    encrypt_program_full,
    decrypt_state_full,
)
from axol.core.padding import (
    PaddedProgram,
    pad_and_encrypt,
    unpad_and_decrypt,
)


@dataclass
class AxolClient:
    """Client-side encryption manager.

    Encrypts programs before sending to an untrusted server.
    Decrypts results after receiving from the server.
    The server never sees plaintext data or keys.
    """

    seed: int
    max_dim: int = 32
    use_padding: bool = True

    def __post_init__(self) -> None:
        self._kf = KeyFamily(seed=self.seed)

    @property
    def key_family(self) -> KeyFamily:
        return self._kf

    def prepare(self, program: Program) -> dict:
        """Encrypt a program for server-side execution.

        Returns a dict with:
          - "program": serialized encrypted program
          - "metadata": info needed for decryption (kept client-side)
        """
        if self.use_padding:
            padded = pad_and_encrypt(program, self._kf, self.max_dim)
            return {
                "program": serialize_program(padded.encrypted_program),
                "metadata": {
                    "mode": "padded",
                    "max_dim": self.max_dim,
                    "seed": self.seed,
                    "original_dims": padded.original_dims,
                    "dim_map": padded.dim_map,
                },
            }
        else:
            enc_prog, dim_map = encrypt_program_full(program, self._kf)
            return {
                "program": serialize_program(enc_prog),
                "metadata": {
                    "mode": "full",
                    "seed": self.seed,
                    "dim_map": dim_map,
                },
            }

    def decrypt_result(
        self, encrypted_state: StateBundle, metadata: dict
    ) -> StateBundle:
        """Decrypt server results using stored metadata."""
        mode = metadata["mode"]
        if mode == "padded":
            padded = PaddedProgram(
                encrypted_program=Program(
                    name="_dummy", initial_state=StateBundle(), transitions=[]
                ),
                max_dim=metadata["max_dim"],
                key_family=self._kf,
                original_dims=metadata["original_dims"],
                dim_map=metadata["dim_map"],
            )
            return unpad_and_decrypt(encrypted_state, padded)
        else:
            dim_map = metadata["dim_map"]
            return decrypt_state_full(encrypted_state, self._kf, dim_map)

    def run_local(self, program: Program) -> StateBundle:
        """Encrypt, run, and decrypt locally (for testing)."""
        payload = self.prepare(program)
        metadata = payload["metadata"]

        # Deserialize to get the encrypted program
        enc_prog = deserialize_program(payload["program"])
        result = run_program(enc_prog)

        return self.decrypt_result(result.final_state, metadata)


def serialize_program(program: Program) -> dict:
    """Serialize an Axol Program to a JSON-compatible dict."""
    state_data = {}
    for key, vec in program.initial_state.items():
        state_data[key] = {
            "data": vec.data.tolist(),
            "ndim": vec.data.ndim,
        }

    transitions_data = []
    for t in program.transitions:
        op = t.operation
        op_data: dict = {"type": type(op).__name__}
        if isinstance(op, TransformOp):
            op_data["key"] = op.key
            op_data["matrix"] = op.matrix.data.tolist()
            op_data["out_key"] = op.out_key
        else:
            # Serialize other op types as needed
            for attr_name in ("key", "keys", "gate_key", "then_key", "else_key",
                              "out_key", "key_a", "key_b", "metric",
                              "threshold", "fn_name", "min_val", "max_val", "label"):
                if hasattr(op, attr_name):
                    val = getattr(op, attr_name)
                    if isinstance(val, np.ndarray):
                        val = val.tolist()
                    elif isinstance(val, (FloatVec, TransMatrix)):
                        val = val.data.tolist()
                    op_data[attr_name] = val
            if hasattr(op, "weights"):
                op_data["weights"] = op.weights.data.tolist()
            if hasattr(op, "router"):
                op_data["router"] = op.router.data.tolist()
            if hasattr(op, "matrix"):
                op_data["matrix"] = op.matrix.data.tolist()

        transitions_data.append({
            "name": t.name,
            "operation": op_data,
            "metadata": t.metadata,
        })

    return {
        "name": program.name,
        "initial_state": state_data,
        "transitions": transitions_data,
        "terminal_key": program.terminal_key,
        "max_iterations": program.max_iterations,
    }


def deserialize_program(data: dict) -> Program:
    """Deserialize a dict back into an Axol Program."""
    from axol.core.program import (
        TransformOp, GateOp, MergeOp, DistanceOp, RouteOp,
        StepOp, BranchOp, ClampOp, MapOp, MeasureOp,
    )

    # Rebuild initial state
    vectors = {}
    for key, vdata in data["initial_state"].items():
        arr = np.array(vdata["data"], dtype=np.float32)
        if vdata["ndim"] == 2:
            vectors[key] = TransMatrix(data=arr)
        else:
            vectors[key] = FloatVec(data=arr)

    state = StateBundle(vectors=vectors)

    # Rebuild transitions
    transitions = []
    for tdata in data["transitions"]:
        op_data = tdata["operation"]
        op_type = op_data["type"]

        if op_type == "TransformOp":
            op = TransformOp(
                key=op_data["key"],
                matrix=TransMatrix(data=np.array(op_data["matrix"], dtype=np.float32)),
                out_key=op_data.get("out_key"),
            )
        elif op_type == "GateOp":
            op = GateOp(
                key=op_data["key"],
                gate_key=op_data["gate_key"],
                out_key=op_data.get("out_key"),
            )
        elif op_type == "MergeOp":
            op = MergeOp(
                keys=op_data["keys"],
                weights=FloatVec(data=np.array(op_data["weights"], dtype=np.float32)),
                out_key=op_data["out_key"],
            )
        elif op_type == "DistanceOp":
            op = DistanceOp(
                key_a=op_data["key_a"],
                key_b=op_data["key_b"],
                metric=op_data.get("metric", "euclidean"),
                out_key=op_data.get("out_key", "_distance"),
            )
        elif op_type == "RouteOp":
            op = RouteOp(
                key=op_data["key"],
                router=TransMatrix(data=np.array(op_data["router"], dtype=np.float32)),
                out_key=op_data.get("out_key", "_route"),
            )
        elif op_type == "StepOp":
            op = StepOp(
                key=op_data["key"],
                threshold=op_data.get("threshold", 0.0),
                out_key=op_data.get("out_key"),
            )
        elif op_type == "BranchOp":
            op = BranchOp(
                gate_key=op_data["gate_key"],
                then_key=op_data["then_key"],
                else_key=op_data["else_key"],
                out_key=op_data["out_key"],
            )
        elif op_type == "ClampOp":
            op = ClampOp(
                key=op_data["key"],
                min_val=op_data.get("min_val", float("-inf")),
                max_val=op_data.get("max_val", float("inf")),
                out_key=op_data.get("out_key"),
            )
        elif op_type == "MapOp":
            op = MapOp(
                key=op_data["key"],
                fn_name=op_data["fn_name"],
                out_key=op_data.get("out_key"),
            )
        elif op_type == "MeasureOp":
            op = MeasureOp(
                key=op_data["key"],
                out_key=op_data.get("out_key"),
            )
        else:
            raise ValueError(f"Unknown operation type: {op_type}")

        transitions.append(Transition(
            name=tdata["name"],
            operation=op,
            metadata=tdata.get("metadata", {}),
        ))

    return Program(
        name=data["name"],
        initial_state=state,
        transitions=transitions,
        terminal_key=data.get("terminal_key"),
        max_iterations=data.get("max_iterations", 1000),
    )
