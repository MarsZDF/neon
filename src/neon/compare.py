"""Comparison functions for floating-point numbers."""

import math
from typing import Sequence


def near(a: float, b: float, *, rel_tol: float = 1e-9, abs_tol: float = 0.0) -> bool:
    """Check if two floats are approximately equal.

    Uses the same algorithm as math.isclose():
        abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    Special cases:
        - near(nan, nan) → False (NaN is not near anything)
        - near(inf, inf) → True (same infinity)
        - near(-inf, inf) → False
        - near(inf, x) → False for any finite x
        - near(0.0, -0.0) → True

    Args:
        a: First value
        b: Second value
        rel_tol: Relative tolerance (default 1e-9)
        abs_tol: Absolute tolerance (default 0.0)

    Returns:
        True if a and b are approximately equal

    Examples:
        >>> near(0.1 + 0.2, 0.3)
        True
        >>> near(1.0, 1.001, rel_tol=1e-2)
        True
        >>> near(1.0, 1.001, rel_tol=1e-4)
        False
        >>> near(float('nan'), float('nan'))
        False
        >>> near(float('inf'), float('inf'))
        True
    """
    # Handle NaN - NaN is not near anything, including itself
    if math.isnan(a) or math.isnan(b):
        return False

    # Handle infinities
    if math.isinf(a) or math.isinf(b):
        return a == b  # inf == inf, but inf != -inf and inf != finite

    # Standard relative/absolute tolerance check
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def near_zero(x: float, *, abs_tol: float = 1e-9) -> bool:
    """Check if a float is approximately zero.

    Args:
        x: Value to check
        abs_tol: Absolute tolerance (default 1e-9)

    Returns:
        True if x is within abs_tol of zero

    Examples:
        >>> near_zero(0.0)
        True
        >>> near_zero(-0.0)
        True
        >>> near_zero(1e-15)
        True
        >>> near_zero(1e-5)
        False
        >>> near_zero(float('nan'))
        False
    """
    if math.isnan(x):
        return False
    if math.isinf(x):
        return False
    return abs(x) <= abs_tol


def less_or_near(a: float, b: float, *, rel_tol: float = 1e-9, abs_tol: float = 0.0) -> bool:
    """Check if a < b or a ≈ b.

    Args:
        a: First value
        b: Second value
        rel_tol: Relative tolerance (default 1e-9)
        abs_tol: Absolute tolerance (default 0.0)

    Returns:
        True if a is less than or approximately equal to b

    Examples:
        >>> less_or_near(1.0, 2.0)
        True
        >>> less_or_near(1.0, 1.0 + 1e-15)
        True
        >>> less_or_near(2.0, 1.0)
        False
    """
    return a < b or near(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def greater_or_near(
    a: float, b: float, *, rel_tol: float = 1e-9, abs_tol: float = 0.0
) -> bool:
    """Check if a > b or a ≈ b.

    Args:
        a: First value
        b: Second value
        rel_tol: Relative tolerance (default 1e-9)
        abs_tol: Absolute tolerance (default 0.0)

    Returns:
        True if a is greater than or approximately equal to b

    Examples:
        >>> greater_or_near(2.0, 1.0)
        True
        >>> greater_or_near(1.0, 1.0 - 1e-15)
        True
        >>> greater_or_near(1.0, 2.0)
        False
    """
    return a > b or near(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def compare(a: float, b: float, *, rel_tol: float = 1e-9, abs_tol: float = 0.0) -> int:
    """Compare two floats with tolerance (spaceship operator).

    Returns:
        -1 if a < b
         0 if a ≈ b (within tolerance)
        +1 if a > b

    Args:
        a: First value
        b: Second value
        rel_tol: Relative tolerance (default 1e-9)
        abs_tol: Absolute tolerance (default 0.0)

    Examples:
        >>> compare(1.0, 2.0)
        -1
        >>> compare(2.0, 1.0)
        1
        >>> compare(1.0, 1.0 + 1e-15)
        0
    """
    if near(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
        return 0
    return -1 if a < b else 1


def all_near(
    pairs: Sequence[tuple[float, float]], *, rel_tol: float = 1e-9, abs_tol: float = 0.0
) -> bool:
    """Check if all pairs of values are approximately equal.

    Args:
        pairs: Sequence of (a, b) tuples
        rel_tol: Relative tolerance (default 1e-9)
        abs_tol: Absolute tolerance (default 0.0)

    Returns:
        True if all pairs are approximately equal

    Examples:
        >>> all_near([(0.1 + 0.2, 0.3), (1.0, 1.0)])
        True
        >>> all_near([(1.0, 1.0), (1.0, 2.0)])
        False
    """
    return all(near(a, b, rel_tol=rel_tol, abs_tol=abs_tol) for a, b in pairs)


def is_integer(x: float, *, abs_tol: float = 1e-9) -> bool:
    """Check if a float is near an integer value.

    Args:
        x: Value to check
        abs_tol: Absolute tolerance (default 1e-9)

    Returns:
        True if x is approximately an integer

    Examples:
        >>> is_integer(3.0)
        True
        >>> is_integer(3.0000000001)
        True
        >>> is_integer(3.1)
        False
        >>> is_integer(float('inf'))
        False
        >>> is_integer(float('nan'))
        False
    """
    if math.isnan(x) or math.isinf(x):
        return False
    return abs(x - round(x)) <= abs_tol


def near_many(
    pairs: Sequence[tuple[float, float]], *, rel_tol: float = 1e-9, abs_tol: float = 0.0
) -> list[bool]:
    """Batch comparison of pairs.

    Args:
        pairs: Sequence of (a, b) tuples
        rel_tol: Relative tolerance (default 1e-9)
        abs_tol: Absolute tolerance (default 0.0)

    Returns:
        List of boolean results for each pair

    Examples:
        >>> near_many([(0.1 + 0.2, 0.3), (1.0, 1.0), (1.0, 2.0)])
        [True, True, False]
    """
    return [near(a, b, rel_tol=rel_tol, abs_tol=abs_tol) for a, b in pairs]
