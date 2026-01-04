"""Neon - Near-equality and tolerance arithmetic for floating-point numbers.

elemental-neon is a zero-dependency library for floating-point comparison and
tolerance math. It handles approximate equality, safe division, ULP comparisons,
and numerical clamping.

Modules:
    compare: Approximate equality comparisons
    clamp: Value snapping and clamping
    safe: Safe arithmetic operations
    ulp: ULP-based operations

Example:
    >>> from neon import compare
    >>> compare.near(0.1 + 0.2, 0.3)
    True
"""

from . import compare, clamp, safe, ulp
from .exceptions import (
    NeonError,
    InvalidValueError,
    EmptyInputError,
)

__version__ = "0.1.0"

__all__ = [
    # Modules
    "compare",
    "clamp",
    "safe",
    "ulp",
    # Exceptions
    "NeonError",
    "InvalidValueError",
    "EmptyInputError",
]
