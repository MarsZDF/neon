"""Internal validation utilities."""

import math
from collections.abc import Sequence

from .exceptions import EmptyInputError, InvalidValueError


def validate_finite(x: float, name: str = "value") -> None:
    """Validate that a value is finite (not NaN or inf).

    Args:
        x: Value to validate
        name: Name of the parameter for error messages

    Raises:
        InvalidValueError: If x is NaN or infinite
    """
    if math.isnan(x):
        raise InvalidValueError(f"{name} cannot be NaN", x)
    if math.isinf(x):
        raise InvalidValueError(f"{name} cannot be infinite", x)


def validate_non_empty(seq: Sequence[float], name: str = "sequence") -> None:
    """Validate that a sequence is non-empty.

    Args:
        seq: Sequence to validate
        name: Name of the parameter for error messages

    Raises:
        EmptyInputError: If sequence is empty
    """
    if len(seq) == 0:
        raise EmptyInputError(f"Cannot operate on empty {name}")


def validate_tolerance(tol: float, name: str = "tolerance") -> None:
    """Validate that a tolerance value is non-negative and finite.

    Args:
        tol: Tolerance value to validate
        name: Name of the parameter for error messages

    Raises:
        InvalidValueError: If tolerance is negative, NaN, or infinite
    """
    if math.isnan(tol):
        raise InvalidValueError(f"{name} cannot be NaN", tol)
    if math.isinf(tol):
        raise InvalidValueError(f"{name} cannot be infinite", tol)
    if tol < 0:
        raise InvalidValueError(f"{name} must be non-negative, got {tol}", tol)
