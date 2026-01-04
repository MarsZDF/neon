"""Safe arithmetic operations with edge case handling."""

import math
from typing import Optional, Sequence

from .compare import near_zero
from ._validation import validate_non_empty


def div(a: float, b: float, *, default: Optional[float] = None, zero_tol: float = 0.0) -> Optional[float]:
    """Safe division with configurable zero handling.

    Args:
        a: Numerator
        b: Denominator
        default: Value to return if b is zero/near-zero (default None)
        zero_tol: Tolerance for considering b as zero (default 0.0)

    Returns:
        a / b if b is not zero, otherwise default

    Examples:
        >>> div(6, 3)
        2.0
        >>> div(1, 0)
        >>> div(1, 0, default=0.0)
        0.0
        >>> div(1, 1e-15, zero_tol=1e-10)
    """
    if near_zero(b, abs_tol=zero_tol):
        return default
    return a / b


def div_or_zero(a: float, b: float, *, zero_tol: float = 0.0) -> float:
    """Safe division that returns 0.0 if denominator is zero.

    Args:
        a: Numerator
        b: Denominator
        zero_tol: Tolerance for considering b as zero (default 0.0)

    Returns:
        a / b if b is not zero, otherwise 0.0

    Examples:
        >>> div_or_zero(6, 3)
        2.0
        >>> div_or_zero(1, 0)
        0.0
    """
    result = div(a, b, default=0.0, zero_tol=zero_tol)
    return result if result is not None else 0.0


def div_or_inf(a: float, b: float, *, zero_tol: float = 0.0) -> float:
    """Safe division that returns ±inf if denominator is zero.

    Args:
        a: Numerator
        b: Denominator
        zero_tol: Tolerance for considering b as zero (default 0.0)

    Returns:
        a / b if b is not zero, otherwise ±inf (sign matches a)
        Special case: 0/0 returns NaN

    Examples:
        >>> div_or_inf(6, 3)
        2.0
        >>> div_or_inf(1, 0)
        inf
        >>> div_or_inf(-1, 0)
        -inf
        >>> import math
        >>> math.isnan(div_or_inf(0, 0))
        True
    """
    if near_zero(b, abs_tol=zero_tol):
        if near_zero(a, abs_tol=zero_tol):
            return math.nan  # 0/0 → NaN
        return math.copysign(math.inf, a)
    return a / b


def mod(
    a: float, b: float, *, default: Optional[float] = None, zero_tol: float = 0.0
) -> Optional[float]:
    """Safe modulo with zero handling.

    Args:
        a: Dividend
        b: Divisor
        default: Value to return if b is zero/near-zero (default None)
        zero_tol: Tolerance for considering b as zero (default 0.0)

    Returns:
        a % b if b is not zero, otherwise default

    Examples:
        >>> mod(7, 3)
        1.0
        >>> mod(7, 0)
        >>> mod(7, 0, default=0.0)
        0.0
    """
    if near_zero(b, abs_tol=zero_tol):
        return default
    return a % b


def sqrt(x: float, *, default: Optional[float] = None) -> Optional[float]:
    """Safe square root that handles negative inputs.

    Args:
        x: Value to take square root of
        default: Value to return if x < 0 (default None)

    Returns:
        sqrt(x) if x >= 0, otherwise default

    Examples:
        >>> sqrt(4)
        2.0
        >>> sqrt(-1)
        >>> sqrt(-1, default=0.0)
        0.0
        >>> sqrt(0)
        0.0
    """
    if x < 0:
        return default
    return math.sqrt(x)


def log(x: float, *, base: Optional[float] = None, default: Optional[float] = None) -> Optional[float]:
    """Safe logarithm that handles non-positive inputs.

    Args:
        x: Value to take logarithm of
        base: Logarithm base (default e, natural log)
        default: Value to return if x <= 0 (default None)

    Returns:
        log(x) if x > 0, otherwise default

    Examples:
        >>> import math
        >>> abs(log(math.e) - 1.0) < 1e-10
        True
        >>> log(0)
        >>> log(-1)
        >>> log(100, base=10)
        2.0
    """
    if x <= 0:
        return default

    if base is None:
        return math.log(x)
    return math.log(x, base)


def pow(base: float, exp: float, *, default: Optional[float] = None) -> Optional[float]:
    """Safe power that handles edge cases.

    Args:
        base: Base value
        exp: Exponent
        default: Value to return on error (default None)

    Returns:
        base ** exp, or default if operation would raise an error

    Examples:
        >>> pow(2, 3)
        8.0
        >>> pow(-1, 0.5)  # Would raise error
        >>> pow(-1, 0.5, default=0.0)
        0.0
        >>> pow(0, 0)
        1.0
    """
    try:
        return base**exp
    except (ValueError, ZeroDivisionError):
        return default


def sum_exact(values: Sequence[float]) -> float:
    """Sum using Kahan summation for improved precision.

    Uses compensated summation to reduce floating-point errors.

    Args:
        values: Sequence of values to sum

    Returns:
        Sum of values with improved precision

    Raises:
        EmptyInputError: If values is empty

    Examples:
        >>> sum_exact([0.1] * 10) == 1.0
        True
        >>> # More accurate than built-in sum for some cases:
        >>> values = [1e16, 1.0, -1e16]
        >>> sum(values)
        0.0
        >>> sum_exact(values)
        1.0
    """
    validate_non_empty(values, "values")

    total = 0.0
    compensation = 0.0

    for x in values:
        y = x - compensation
        t = total + y
        compensation = (t - total) - y
        total = t

    return total


def mean_exact(values: Sequence[float]) -> float:
    """Mean using Kahan summation for improved precision.

    Args:
        values: Sequence of values

    Returns:
        Mean of values with improved precision

    Raises:
        EmptyInputError: If values is empty

    Examples:
        >>> mean_exact([1.0, 2.0, 3.0])
        2.0
    """
    validate_non_empty(values, "values")
    return sum_exact(values) / len(values)
