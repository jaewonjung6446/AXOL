"""Quantum module exception hierarchy."""

from __future__ import annotations


class QuantumError(Exception):
    """Base exception for all quantum module errors."""


class WeaverError(QuantumError):
    """Error during tapestry weaving."""


class ObservatoryError(QuantumError):
    """Error during observation."""


class QuantumParseError(QuantumError):
    """Error parsing quantum DSL source."""
