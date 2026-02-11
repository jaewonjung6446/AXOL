"""Axol DSL parser — compact text format to Program objects.

Grammar:
    program     := header state_lines+ transition_lines+ terminal?
    header      := '@' NAME NEWLINE
    state_line  := 's' (NAME '=' value)+ NEWLINE
    value       := '[' NUMBER+ ']' | 'onehot(' INT ',' INT ')' | 'zeros(' INT ')' | 'ones(' INT ')'
    transition  := ':' NAME '=' op_call ('->' NAME)? NEWLINE
    op_call     := 'transform(' ... ')' | 'gate(' ... ')' | 'merge(' ... ')' | ...
    terminal    := '?' NAME condition
"""

from __future__ import annotations

import os
import re
from typing import Callable

import numpy as np

from axol.core.types import (
    FloatVec,
    GateVec,
    OneHotVec,
    TransMatrix,
    StateBundle,
    _VecBase,
)
from axol.core.program import (
    TransformOp,
    GateOp,
    MergeOp,
    DistanceOp,
    RouteOp,
    CustomOp,
    StepOp,
    BranchOp,
    ClampOp,
    MapOp,
    Transition,
    Program,
)


class ParseError(Exception):
    """Raised when DSL source cannot be parsed."""

    def __init__(self, message: str, line_num: int | None = None):
        self.line_num = line_num
        prefix = f"line {line_num}: " if line_num is not None else ""
        super().__init__(f"{prefix}{message}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse(
    source: str,
    registry: object | None = None,
    source_path: str | None = None,
) -> Program:
    """Parse Axol DSL source text into a Program object.

    Args:
        source: DSL source code.
        registry: Optional ModuleRegistry for resolving imports.
        source_path: Optional path of the source file (for relative imports).
    """
    lines = _strip_lines(source)
    if not lines:
        raise ParseError("Empty program")

    idx = 0
    num, line = lines[idx]

    # --- header ---
    if not line.startswith("@"):
        raise ParseError("Program must start with '@name'", num)
    name = _parse_header(line, num)
    idx += 1

    # --- schema lines (optional) ---
    schema_inputs = []
    schema_outputs = []
    while idx < len(lines) and lines[idx][1].startswith("schema "):
        num, line = lines[idx]
        direction, specs = _parse_schema_line(line, num)
        if direction == "in":
            schema_inputs.extend(specs)
        elif direction == "out":
            schema_outputs.extend(specs)
        idx += 1

    # --- import lines (optional) ---
    imports: dict[str, str] = {}  # alias -> module_name
    while idx < len(lines) and lines[idx][1].startswith("import "):
        num, line = lines[idx]
        alias, mod_name, mod_path = _parse_import_line(line, num)
        imports[alias] = mod_name
        # Resolve imports via registry if available
        if registry is not None:
            try:
                _resolve_import(registry, mod_name, mod_path, source_path)
            except Exception as e:
                raise ParseError(f"Failed to resolve import '{mod_name}': {e}", num)
        idx += 1

    # --- state lines ---
    state_vectors: dict[str, _VecBase] = {}
    while idx < len(lines) and lines[idx][1].startswith("s "):
        num, line = lines[idx]
        state_vectors.update(_parse_state(line, num))
        idx += 1

    if not state_vectors:
        raise ParseError("Program must have at least one state line ('s ...')")

    # --- transitions ---
    transitions: list[Transition] = []
    while idx < len(lines) and lines[idx][1].startswith(":"):
        num, line = lines[idx]
        transitions.append(_parse_transition(line, num, registry=registry, imports=imports))
        idx += 1

    if not transitions:
        raise ParseError("Program must have at least one transition (':' ...)")

    # --- optional terminal ---
    terminal_key: str | None = None
    if idx < len(lines) and lines[idx][1].startswith("?"):
        num, line = lines[idx]
        t_key, t_transition = _parse_terminal(line, state_vectors, num)
        terminal_key = t_key
        transitions.append(t_transition)
        idx += 1

    # Build initial StateBundle
    initial = StateBundle(vectors=dict(state_vectors))

    # Auto-coerce FloatVec → GateVec for keys referenced by GateOps or BranchOps
    for t in transitions:
        if isinstance(t.operation, GateOp):
            gk = t.operation.gate_key
            if gk in initial.vectors and not isinstance(initial[gk], GateVec):
                vec = initial[gk]
                vals = vec.data.astype(np.float32)
                if np.all((vals == 0.0) | (vals == 1.0)):
                    initial[gk] = GateVec.from_list(vals.tolist())
        elif isinstance(t.operation, BranchOp):
            gk = t.operation.gate_key
            if gk in initial.vectors and not isinstance(initial[gk], GateVec):
                vec = initial[gk]
                vals = vec.data.astype(np.float32)
                if np.all((vals == 0.0) | (vals == 1.0)):
                    initial[gk] = GateVec.from_list(vals.tolist())

    # Add terminal gate if needed
    if terminal_key is not None and terminal_key not in initial.vectors:
        initial[terminal_key] = GateVec.zeros(1)

    return Program(
        name=name,
        initial_state=initial,
        transitions=transitions,
        terminal_key=terminal_key,
    )


# ---------------------------------------------------------------------------
# Line preprocessing
# ---------------------------------------------------------------------------

def _strip_lines(source: str) -> list[tuple[int, str]]:
    """Return (1-based line number, stripped content) for non-empty, non-comment lines."""
    result = []
    for i, raw in enumerate(source.splitlines(), 1):
        stripped = raw.strip()
        if stripped and not stripped.startswith("#"):
            result.append((i, stripped))
    return result


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def _parse_header(line: str, line_num: int) -> str:
    name = line[1:].strip()
    if not name or not re.match(r'^[A-Za-z_]\w*$', name):
        raise ParseError(f"Invalid program name: '{name}'", line_num)
    return name


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def _parse_state(line: str, line_num: int) -> dict[str, _VecBase]:
    """Parse 's key=value key2=value2 ...'"""
    body = line[1:].strip()  # strip leading 's'
    result: dict[str, _VecBase] = {}

    # Match key=value pairs; values can contain nested parens or brackets
    for m in re.finditer(r'([A-Za-z_]\w*)=((?:\[[^\]]*\])|(?:\w+\([^)]*\)))', body):
        key = m.group(1)
        val_str = m.group(2)
        result[key] = _parse_value(val_str, line_num)

    if not result:
        raise ParseError("State line has no key=value pairs", line_num)
    return result


def _parse_value(token: str, line_num: int) -> _VecBase:
    """Parse a value token into a VecBase."""
    token = token.strip()

    # onehot(idx, n)
    m = re.match(r'^onehot\((\d+)\s*,\s*(\d+)\)$', token)
    if m:
        idx, n = int(m.group(1)), int(m.group(2))
        return OneHotVec.from_index(idx, n)

    # zeros(n)
    m = re.match(r'^zeros\((\d+)\)$', token)
    if m:
        return FloatVec.zeros(int(m.group(1)))

    # ones(n)
    m = re.match(r'^ones\((\d+)\)$', token)
    if m:
        return FloatVec.ones(int(m.group(1)))

    # [number number ...]
    m = re.match(r'^\[([^\]]*)\]$', token)
    if m:
        nums = _parse_numbers(m.group(1))
        return FloatVec.from_list(nums)

    raise ParseError(f"Invalid value: '{token}'", line_num)


def _parse_numbers(text: str) -> list[float]:
    """Parse space-separated numbers."""
    parts = text.split()
    return [float(x) for x in parts]


def _split_args(text: str, sep: str = ";") -> list[str]:
    """Split on *sep* respecting bracket nesting ([...] and (...))."""
    parts: list[str] = []
    depth_sq = 0  # [...]
    depth_rn = 0  # (...)
    start = 0
    for i, ch in enumerate(text):
        if ch == '[':
            depth_sq += 1
        elif ch == ']':
            depth_sq -= 1
        elif ch == '(':
            depth_rn += 1
        elif ch == ')':
            depth_rn -= 1
        elif ch == sep and depth_sq == 0 and depth_rn == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return parts


# ---------------------------------------------------------------------------
# Transition
# ---------------------------------------------------------------------------

def _parse_transition(
    line: str,
    line_num: int,
    registry: object | None = None,
    imports: dict[str, str] | None = None,
) -> Transition:
    """Parse ': name=op_call(...)  (->out_key)?'"""
    body = line[1:].strip()  # strip leading ':'

    # Split name=rest
    eq_idx = body.index("=") if "=" in body else -1
    if eq_idx <= 0:
        raise ParseError("Transition must have 'name=op(...)'", line_num)

    name = body[:eq_idx].strip()

    # Check for ->out_key at the end (outside parens)
    rest = body[eq_idx + 1:]
    out_key: str | None = None

    # Find the closing paren of the op call, then check for ->
    paren_depth = 0
    close_idx = -1
    for i, ch in enumerate(rest):
        if ch == '(':
            paren_depth += 1
        elif ch == ')':
            paren_depth -= 1
            if paren_depth == 0:
                close_idx = i
                break

    if close_idx < 0:
        raise ParseError(f"Unmatched parenthesis in transition '{name}'", line_num)

    after_paren = rest[close_idx + 1:].strip()
    if after_paren.startswith("->"):
        out_key = after_paren[2:].strip()

    op_str = rest[:close_idx + 1].strip()
    operation = _parse_op_call(op_str, out_key, line_num, registry=registry, imports=imports)
    return Transition(name=name, operation=operation)


def _parse_op_call(
    op_str: str,
    out_key: str | None,
    line_num: int,
    registry: object | None = None,
    imports: dict[str, str] | None = None,
):
    """Parse 'op_name(args)' into an Operation descriptor."""
    m = re.match(r'^(\w+)\((.+)\)$', op_str, re.DOTALL)
    if not m:
        raise ParseError(f"Invalid operation: '{op_str}'", line_num)

    op_name = m.group(1)
    args_str = m.group(2)

    if op_name == "transform":
        return _parse_transform_op(args_str, out_key, line_num)
    elif op_name == "gate":
        return _parse_gate_op(args_str, out_key, line_num)
    elif op_name == "merge":
        return _parse_merge_op(args_str, out_key, line_num)
    elif op_name == "distance":
        return _parse_distance_op(args_str, out_key, line_num)
    elif op_name == "route":
        return _parse_route_op(args_str, out_key, line_num)
    elif op_name == "step":
        return _parse_step_op(args_str, out_key, line_num)
    elif op_name == "branch":
        return _parse_branch_op(args_str, out_key, line_num)
    elif op_name == "clamp":
        return _parse_clamp_op(args_str, out_key, line_num)
    elif op_name == "map":
        return _parse_map_op(args_str, out_key, line_num)
    elif op_name == "use":
        return _parse_use_op(args_str, line_num, registry=registry, imports=imports)
    else:
        raise ParseError(f"Unknown operation: '{op_name}'", line_num)


def _parse_transform_op(args: str, out_key: str | None, line_num: int) -> TransformOp:
    """Parse 'key;M=matrix_expr'"""
    parts = _split_args(args)
    key = parts[0].strip()
    matrix: TransMatrix | None = None

    for part in parts[1:]:
        part = part.strip()
        if part.startswith("M="):
            matrix = _parse_matrix(part[2:], line_num)

    if matrix is None:
        raise ParseError("transform requires M= parameter", line_num)

    return TransformOp(key=key, matrix=matrix, out_key=out_key)


def _parse_gate_op(args: str, out_key: str | None, line_num: int) -> GateOp:
    """Parse 'key;g=gate_key'"""
    parts = _split_args(args)
    key = parts[0].strip()
    gate_key: str | None = None

    for part in parts[1:]:
        part = part.strip()
        if part.startswith("g="):
            gate_key = part[2:].strip()

    if gate_key is None:
        raise ParseError("gate requires g= parameter", line_num)

    return GateOp(key=key, gate_key=gate_key, out_key=out_key)


def _parse_merge_op(args: str, out_key: str | None, line_num: int) -> MergeOp:
    """Parse 'key1 key2 ...;w=[w1 w2 ...]'"""
    parts = _split_args(args)
    keys = parts[0].strip().split()
    weights: FloatVec | None = None

    for part in parts[1:]:
        part = part.strip()
        if part.startswith("w="):
            w_str = part[2:].strip()
            m = re.match(r'^\[([^\]]*)\]$', w_str)
            if not m:
                raise ParseError(f"Invalid weights: '{w_str}'", line_num)
            nums = _parse_numbers(m.group(1))
            weights = FloatVec.from_list(nums)

    if weights is None:
        raise ParseError("merge requires w= parameter", line_num)

    if out_key is None:
        raise ParseError("merge requires ->out_key", line_num)

    return MergeOp(keys=keys, weights=weights, out_key=out_key)


def _parse_distance_op(args: str, out_key: str | None, line_num: int) -> DistanceOp:
    """Parse 'key_a key_b (;metric=name)?'"""
    parts = _split_args(args)
    keys = parts[0].strip().split()
    if len(keys) != 2:
        raise ParseError("distance requires exactly 2 keys", line_num)

    metric = "euclidean"
    for part in parts[1:]:
        part = part.strip()
        if part.startswith("metric="):
            metric = part[7:].strip()

    return DistanceOp(
        key_a=keys[0], key_b=keys[1], metric=metric,
        out_key=out_key or "_distance",
    )


def _parse_route_op(args: str, out_key: str | None, line_num: int) -> RouteOp:
    """Parse 'key;R=matrix_expr'"""
    parts = _split_args(args)
    key = parts[0].strip()
    router: TransMatrix | None = None

    for part in parts[1:]:
        part = part.strip()
        if part.startswith("R="):
            router = _parse_matrix(part[2:], line_num)

    if router is None:
        raise ParseError("route requires R= parameter", line_num)

    return RouteOp(key=key, router=router, out_key=out_key or "_route")


# ---------------------------------------------------------------------------
# Plaintext op parsers (step, branch, clamp, map)
# ---------------------------------------------------------------------------

def _parse_step_op(args: str, out_key: str | None, line_num: int) -> StepOp:
    """Parse 'step(key;t=threshold)'"""
    parts = _split_args(args)
    key = parts[0].strip()
    threshold = 0.0

    for part in parts[1:]:
        part = part.strip()
        if part.startswith("t="):
            threshold = float(part[2:].strip())

    return StepOp(key=key, threshold=threshold, out_key=out_key)


def _parse_branch_op(args: str, out_key: str | None, line_num: int) -> BranchOp:
    """Parse 'branch(gate_key;then=a,else=b)->out'  (out_key required)."""
    if out_key is None:
        raise ParseError("branch requires ->out_key", line_num)

    parts = _split_args(args)
    gate_key = parts[0].strip()
    then_key: str | None = None
    else_key: str | None = None

    for part in parts[1:]:
        part = part.strip()
        # Support both "then=x,else=y" (comma-separated) and separate "then=x;else=y"
        sub_parts = [p.strip() for p in part.split(",")]
        for sp in sub_parts:
            if sp.startswith("then="):
                then_key = sp[5:].strip()
            elif sp.startswith("else="):
                else_key = sp[5:].strip()

    if then_key is None:
        raise ParseError("branch requires then= parameter", line_num)
    if else_key is None:
        raise ParseError("branch requires else= parameter", line_num)

    return BranchOp(gate_key=gate_key, then_key=then_key, else_key=else_key, out_key=out_key)


def _parse_clamp_op(args: str, out_key: str | None, line_num: int) -> ClampOp:
    """Parse 'clamp(key;min=0,max=100)' — semicolons and commas both supported."""
    # Normalize: replace commas at top-level with semicolons for uniform splitting
    parts = _split_args(args)
    key = parts[0].strip()
    min_val = float("-inf")
    max_val = float("inf")

    for part in parts[1:]:
        part = part.strip()
        # Also handle comma-separated within a part (e.g., "min=0,max=100")
        sub_parts = [p.strip() for p in part.split(",")]
        for sp in sub_parts:
            if sp.startswith("min="):
                min_val = float(sp[4:].strip())
            elif sp.startswith("max="):
                max_val = float(sp[4:].strip())

    return ClampOp(key=key, min_val=min_val, max_val=max_val, out_key=out_key)


def _parse_map_op(args: str, out_key: str | None, line_num: int) -> MapOp:
    """Parse 'map(key;fn=relu)'."""
    parts = _split_args(args)
    key = parts[0].strip()
    fn_name: str | None = None

    for part in parts[1:]:
        part = part.strip()
        if part.startswith("fn="):
            fn_name = part[3:].strip()

    if fn_name is None:
        raise ParseError("map requires fn= parameter", line_num)

    return MapOp(key=key, fn_name=fn_name, out_key=out_key)


# ---------------------------------------------------------------------------
# Schema parsing
# ---------------------------------------------------------------------------

def _parse_schema_line(line: str, line_num: int) -> tuple[str, list[tuple[str, str, int]]]:
    """Parse 'schema in|out name:type[dim] ...'

    Returns (direction, [(name, type, dim), ...]).
    """
    body = line[len("schema"):].strip()
    parts = body.split(None, 1)
    if len(parts) < 2:
        raise ParseError("Schema line needs 'in' or 'out' followed by specs", line_num)

    direction = parts[0]
    if direction not in ("in", "out"):
        raise ParseError(f"Schema direction must be 'in' or 'out', got '{direction}'", line_num)

    specs = []
    for m in re.finditer(r'(\w+):(\w+)\[(\d+)\]', parts[1]):
        specs.append((m.group(1), m.group(2), int(m.group(3))))

    return direction, specs


# ---------------------------------------------------------------------------
# Import parsing
# ---------------------------------------------------------------------------

def _parse_import_line(line: str, line_num: int) -> tuple[str, str, str | None]:
    """Parse 'import alias from \"path\"' or 'import name'.

    Returns (alias, module_name, path_or_None).
    """
    body = line[len("import"):].strip()

    # import name from "path"
    m = re.match(r'^(\w+)\s+from\s+"([^"]+)"$', body)
    if m:
        alias = m.group(1)
        path = m.group(2)
        # Extract module name from path (without .axol extension)
        mod_name = os.path.splitext(os.path.basename(path))[0] if '/' in path or '\\' in path or '.' in path else path
        return alias, mod_name, path

    # import name
    m = re.match(r'^(\w+)$', body)
    if m:
        return m.group(1), m.group(1), None

    raise ParseError(f"Invalid import: '{body}'", line_num)


def _resolve_import(registry: object, mod_name: str, mod_path: str | None, source_path: str | None) -> None:
    """Resolve and register an import in the registry."""
    from axol.core.module import ModuleRegistry
    if not isinstance(registry, ModuleRegistry):
        return
    if registry.has(mod_name):
        return
    if mod_path and source_path:
        base_dir = os.path.dirname(source_path)
        full_path = os.path.join(base_dir, mod_path)
        if not full_path.endswith(".axol"):
            full_path += ".axol"
        if os.path.isfile(full_path):
            registry.load_from_file(full_path, name=mod_name)
            return
    registry.resolve_import(mod_name, source_path)


# ---------------------------------------------------------------------------
# Use op parsing
# ---------------------------------------------------------------------------

def _parse_use_op(args_str: str, line_num: int, registry: object | None = None, imports: dict[str, str] | None = None):
    """Parse 'use(module_name;in=k1,k2;out=o1)' into a UseOp."""
    from axol.core.module import UseOp, ModuleRegistry

    parts = _split_args(args_str)
    if len(parts) < 1:
        raise ParseError("use() requires at least a module name", line_num)

    module_ref = parts[0].strip()
    # Resolve alias
    if imports and module_ref in imports:
        module_name = imports[module_ref]
    else:
        module_name = module_ref

    input_mapping: dict[str, str] = {}
    output_mapping: dict[str, str] = {}

    for part in parts[1:]:
        part = part.strip()
        if part.startswith("in="):
            pairs = part[3:].split(",")
            for p in pairs:
                p = p.strip()
                if ":" in p:
                    mod_key, parent_key = p.split(":", 1)
                    input_mapping[mod_key.strip()] = parent_key.strip()
                else:
                    input_mapping[p] = p
        elif part.startswith("out="):
            pairs = part[4:].split(",")
            for p in pairs:
                p = p.strip()
                if ":" in p:
                    mod_key, parent_key = p.split(":", 1)
                    output_mapping[mod_key.strip()] = parent_key.strip()
                else:
                    output_mapping[p] = p

    reg = registry if isinstance(registry, ModuleRegistry) else None

    return UseOp(
        module_name=module_name,
        input_mapping=input_mapping,
        output_mapping=output_mapping,
        registry=reg,
    )


# ---------------------------------------------------------------------------
# Matrix parsing
# ---------------------------------------------------------------------------

def _parse_matrix(token: str, line_num: int) -> TransMatrix:
    """Parse dense '[r00 r01;r10 r11]' or 'sparse(MxN;i,j=v ...)'."""
    token = token.strip()

    # sparse(MxN;entries...)
    m = re.match(r'^sparse\((\d+)x(\d+);(.+)\)$', token)
    if m:
        rows, cols = int(m.group(1)), int(m.group(2))
        entries_str = m.group(3).strip()
        mat = np.zeros((rows, cols), dtype=np.float32)
        for entry in entries_str.split():
            em = re.match(r'^(\d+),(\d+)=([^\s]+)$', entry)
            if not em:
                raise ParseError(f"Invalid sparse entry: '{entry}'", line_num)
            r, c, v = int(em.group(1)), int(em.group(2)), float(em.group(3))
            mat[r, c] = v
        return TransMatrix(data=mat)

    # dense [row;row;...] — rows separated by ;, values by space
    m = re.match(r'^\[([^\]]*)\]$', token)
    if m:
        content = m.group(1).strip()
        if ";" in content:
            row_strs = content.split(";")
            rows_data = [_parse_numbers(r.strip()) for r in row_strs]
        else:
            # Single row (or 1x1 matrix like [0.8])
            nums = _parse_numbers(content)
            rows_data = [nums]
        return TransMatrix.from_list(rows_data)

    raise ParseError(f"Invalid matrix: '{token}'", line_num)


# ---------------------------------------------------------------------------
# Terminal
# ---------------------------------------------------------------------------

_COMP_OPS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">":  lambda a, b: a > b,
    "<":  lambda a, b: a < b,
    "==": lambda a, b: a == b,
}


def _parse_terminal(line: str, state_keys: dict, line_num: int) -> tuple[str, Transition]:
    """Parse '? terminal_name condition' → (terminal_key, Transition with CustomOp).

    The CustomOp checks the condition each iteration and sets the terminal GateVec.
    """
    body = line[1:].strip()  # strip leading '?'
    parts = body.split(None, 1)
    if len(parts) < 2:
        raise ParseError("Terminal line needs 'name condition'", line_num)

    terminal_name = parts[0]
    condition_str = parts[1]

    key, index, comp_str, threshold = _parse_condition(condition_str, line_num)

    comp_fn = _COMP_OPS.get(comp_str)
    if comp_fn is None:
        raise ParseError(f"Unknown comparator: '{comp_str}'", line_num)

    terminal_key = f"_{terminal_name}"

    def make_check(k: str, idx: int | None, cmp: Callable, thr: float, t_key: str):
        def check(state: StateBundle) -> StateBundle:
            s = state.copy()
            vec = s[k]
            if idx is not None:
                val = float(vec.data[idx])
            else:
                val = float(vec.data[0])
            if cmp(val, thr):
                s[t_key] = GateVec.ones(1)
            return s
        return check

    fn = make_check(key, index, comp_fn, threshold, terminal_key)
    op = CustomOp(fn=fn, label=f"check_{terminal_name}")
    transition = Transition(name=f"_check_{terminal_name}", operation=op)

    return terminal_key, transition


def _parse_condition(expr: str, line_num: int) -> tuple[str, int | None, str, float]:
    """Parse 'key[idx]>=threshold' or 'key>=threshold'.

    Returns (key, index_or_None, comparator_str, threshold).
    """
    # Try key[idx]comp_threshold
    m = re.match(r'^([A-Za-z_]\w*)\[(\d+)\]\s*(>=|<=|>|<|==)\s*([^\s]+)$', expr)
    if m:
        return m.group(1), int(m.group(2)), m.group(3), float(m.group(4))

    # Try key comp threshold
    m = re.match(r'^([A-Za-z_]\w*)\s*(>=|<=|>|<|==)\s*([^\s]+)$', expr)
    if m:
        return m.group(1), None, m.group(2), float(m.group(3))

    raise ParseError(f"Invalid condition: '{expr}'", line_num)
