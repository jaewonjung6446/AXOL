"""Quantum DSL parser.

Grammar:
  entangle NAME(INPUTS) @ Omega(X) Phi(Y) {
      TARGET <OP> EXPR
      ...
  }
  result = observe NAME(ARGS)
  result = reobserve NAME(ARGS) x COUNT
  if result.FIELD < VALUE { ... }

Relation operators:
  <~>  PROPORTIONAL
  <+>  ADDITIVE
  <*>  MULTIPLICATIVE
  <!>  INVERSE
  <?>  CONDITIONAL
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from axol.quantum.errors import QuantumParseError
from axol.quantum.declare import (
    EntangleDeclaration,
    DeclarationBuilder,
    DeclaredInput,
    DeclaredRelation,
    QualityTarget,
    RelationKind,
)


# ---------------------------------------------------------------------------
# AST nodes for the quantum DSL
# ---------------------------------------------------------------------------

@dataclass
class ObserveStatement:
    """result = observe name(args...)"""
    result_var: str
    tapestry_name: str
    arguments: list[str]


@dataclass
class ReobserveStatement:
    """result = reobserve name(args...) x count"""
    result_var: str
    tapestry_name: str
    arguments: list[str]
    count: int


@dataclass
class ConditionalBlock:
    """if result.field < value { statements... }"""
    variable: str
    field_name: str
    operator: str      # "<", ">", "<=", ">="
    threshold: float
    body: list[ObserveStatement | ReobserveStatement]


@dataclass
class QuantumProgram:
    """Parsed quantum DSL program."""
    declarations: list[EntangleDeclaration]
    observations: list[ObserveStatement | ReobserveStatement]
    conditionals: list[ConditionalBlock]


# ---------------------------------------------------------------------------
# Operator mapping
# ---------------------------------------------------------------------------

_OPERATOR_MAP: dict[str, RelationKind] = {
    "<~>": RelationKind.PROPORTIONAL,
    "<+>": RelationKind.ADDITIVE,
    "<*>": RelationKind.MULTIPLICATIVE,
    "<!>": RelationKind.INVERSE,
    "<?>": RelationKind.CONDITIONAL,
}


# ---------------------------------------------------------------------------
# Parser internals
# ---------------------------------------------------------------------------

class _QuantumParser:
    """Stateful parser for quantum DSL source."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.lines = source.strip().split("\n")
        self.pos = 0
        self.declarations: list[EntangleDeclaration] = []
        self.observations: list[ObserveStatement | ReobserveStatement] = []
        self.conditionals: list[ConditionalBlock] = []

    def parse(self) -> QuantumProgram:
        while self.pos < len(self.lines):
            line = self.lines[self.pos].strip()

            # Skip empty lines and comments
            if not line or line.startswith("#") or line.startswith("//"):
                self.pos += 1
                continue

            if line.startswith("entangle "):
                self._parse_entangle()
            elif "= observe " in line:
                self._parse_observe(line)
            elif "= reobserve " in line:
                self._parse_reobserve(line)
            elif line.startswith("if "):
                self._parse_conditional()
            else:
                raise QuantumParseError(f"Unexpected line {self.pos + 1}: {line}")

        return QuantumProgram(
            declarations=self.declarations,
            observations=self.observations,
            conditionals=self.conditionals,
        )

    def _parse_entangle(self) -> None:
        """Parse an entangle block."""
        line = self.lines[self.pos].strip()
        self.pos += 1

        # Pattern: entangle NAME(PARAM: TYPE[DIM], ...) @ Omega(X) Phi(Y) {
        match = re.match(
            r"entangle\s+(\w+)\s*\(([^)]*)\)\s*@\s*Omega\(([^)]+)\)\s*Phi\(([^)]+)\)\s*\{",
            line,
        )
        if not match:
            raise QuantumParseError(f"Invalid entangle declaration: {line}")

        name = match.group(1)
        params_str = match.group(2).strip()
        omega = float(match.group(3))
        phi = float(match.group(4))

        # Parse parameters
        inputs: list[DeclaredInput] = []
        if params_str:
            for param in params_str.split(","):
                param = param.strip()
                # Patterns: "name: type[dim]" or just "name"
                type_match = re.match(r"(\w+)\s*:\s*(\w+)\[(\d+)\]", param)
                if type_match:
                    pname = type_match.group(1)
                    pdim = int(type_match.group(3))
                    inputs.append(DeclaredInput(name=pname, dim=pdim))
                else:
                    # Simple name — default dimension
                    pname = param.strip().split(":")[0].strip()
                    inputs.append(DeclaredInput(name=pname, dim=8))

        # Parse relations until closing brace
        relations: list[DeclaredRelation] = []
        while self.pos < len(self.lines):
            rline = self.lines[self.pos].strip()
            self.pos += 1

            if rline == "}" or rline.startswith("}"):
                break

            if not rline or rline.startswith("#") or rline.startswith("//"):
                continue

            rel = self._parse_relation(rline)
            if rel:
                relations.append(rel)

        # Build declaration
        builder = DeclarationBuilder(name)
        for inp in inputs:
            builder.input(inp.name, inp.dim, inp.labels)
        for rel in relations:
            builder.relate(rel.target, rel.sources, rel.kind, rel.transform_fn, rel.weight)
        builder.quality(omega, phi)

        self.declarations.append(builder.build())

    def _parse_relation(self, line: str) -> DeclaredRelation | None:
        """Parse a relation line: target <op> expr"""
        # Find the operator
        for op_str, kind in _OPERATOR_MAP.items():
            if op_str in line:
                parts = line.split(op_str, 1)
                target = parts[0].strip()
                expr = parts[1].strip()

                # Parse expression to extract source names
                sources = self._extract_names(expr)
                if not sources:
                    raise QuantumParseError(f"No source variables in relation: {line}")

                return DeclaredRelation(
                    target=target,
                    sources=sources,
                    kind=kind,
                    transform_fn=None,
                    weight=1.0,
                )

        raise QuantumParseError(f"No known operator found in relation: {line}")

    def _extract_names(self, expr: str) -> list[str]:
        """Extract variable names from an expression.

        Handles patterns like:
        - similarity(query, db)
        - relevance
        - a + b
        - function(x, y, z)
        """
        # Remove function calls but keep arguments
        # First, find function arguments
        func_match = re.findall(r"\w+\(([^)]*)\)", expr)
        names = set()

        for args_str in func_match:
            for arg in args_str.split(","):
                name = arg.strip()
                if name and re.match(r"^[a-zA-Z_]\w*$", name):
                    names.add(name)

        # Also get standalone names (not function names)
        # Remove function names first
        cleaned = re.sub(r"\w+\(", "(", expr)
        for token in re.findall(r"[a-zA-Z_]\w*", cleaned):
            names.add(token)

        return sorted(names)

    def _parse_observe(self, line: str) -> None:
        """Parse: result = observe name(args...)"""
        match = re.match(r"(\w+)\s*=\s*observe\s+(\w+)\s*\(([^)]*)\)", line)
        if not match:
            raise QuantumParseError(f"Invalid observe statement: {line}")

        result_var = match.group(1)
        tapestry_name = match.group(2)
        args_str = match.group(3).strip()
        arguments = [a.strip() for a in args_str.split(",") if a.strip()]

        self.observations.append(ObserveStatement(
            result_var=result_var,
            tapestry_name=tapestry_name,
            arguments=arguments,
        ))
        self.pos += 1

    def _parse_reobserve(self, line: str) -> None:
        """Parse: result = reobserve name(args...) x count"""
        match = re.match(
            r"(\w+)\s*=\s*reobserve\s+(\w+)\s*\(([^)]*)\)\s*x\s*(\d+)",
            line,
        )
        if not match:
            raise QuantumParseError(f"Invalid reobserve statement: {line}")

        self.observations.append(ReobserveStatement(
            result_var=match.group(1),
            tapestry_name=match.group(2),
            arguments=[a.strip() for a in match.group(3).split(",") if a.strip()],
            count=int(match.group(4)),
        ))
        self.pos += 1

    def _parse_conditional(self) -> None:
        """Parse: if result.field < value { body }"""
        line = self.lines[self.pos].strip()
        self.pos += 1

        match = re.match(
            r"if\s+(\w+)\.(\w+)\s*([<>]=?)\s*([\d.]+)\s*\{",
            line,
        )
        if not match:
            raise QuantumParseError(f"Invalid conditional: {line}")

        variable = match.group(1)
        field_name = match.group(2)
        operator = match.group(3)
        threshold = float(match.group(4))

        body: list[ObserveStatement | ReobserveStatement] = []
        while self.pos < len(self.lines):
            bline = self.lines[self.pos].strip()

            if bline == "}" or bline.startswith("}"):
                self.pos += 1
                break

            if not bline or bline.startswith("#") or bline.startswith("//"):
                self.pos += 1
                continue

            if "= observe " in bline:
                obs_match = re.match(r"(\w+)\s*=\s*observe\s+(\w+)\s*\(([^)]*)\)", bline)
                if obs_match:
                    body.append(ObserveStatement(
                        result_var=obs_match.group(1),
                        tapestry_name=obs_match.group(2),
                        arguments=[a.strip() for a in obs_match.group(3).split(",") if a.strip()],
                    ))
            elif "= reobserve " in bline:
                reobs_match = re.match(
                    r"(\w+)\s*=\s*reobserve\s+(\w+)\s*\(([^)]*)\)\s*x\s*(\d+)",
                    bline,
                )
                if reobs_match:
                    body.append(ReobserveStatement(
                        result_var=reobs_match.group(1),
                        tapestry_name=reobs_match.group(2),
                        arguments=[a.strip() for a in reobs_match.group(3).split(",") if a.strip()],
                        count=int(reobs_match.group(4)),
                    ))

            self.pos += 1

        self.conditionals.append(ConditionalBlock(
            variable=variable,
            field_name=field_name,
            operator=operator,
            threshold=threshold,
            body=body,
        ))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_quantum(source: str) -> QuantumProgram:
    """Parse quantum DSL source into a QuantumProgram.

    Args:
        source: The DSL source string.

    Returns:
        Parsed QuantumProgram with declarations, observations, and conditionals.

    Raises:
        QuantumParseError: If the source cannot be parsed.
    """
    parser = _QuantumParser(source)
    return parser.parse()
