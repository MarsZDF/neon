"""ULP (Unit in the Last Place) operations for floating-point numbers."""

import math

from .exceptions import InvalidValueError


def of(x: float) -> float:
    """Return the ULP (Unit in the Last Place) of x.

    The ULP is the gap between x and the next representable float.

    Args:
        x: Value to get ULP of

    Returns:
        ULP of x

    Raises:
        InvalidValueError: If x is NaN

    Examples:
        >>> of(1.0)
        2.220446049250313e-16
        >>> of(0.0) > 0  # Smallest positive denormal
        True
        >>> of(float('inf'))
        inf
    """
    if math.isnan(x):
        raise InvalidValueError("Cannot compute ULP of NaN", x)

    if math.isinf(x):
        return math.inf

    if x == 0:
        # Return smallest positive denormal
        return math.nextafter(0.0, 1.0)

    return abs(math.nextafter(x, math.inf) - x)


def diff(a: float, b: float) -> int:
    """Return the distance between a and b in ULPs.

    Args:
        a: First value
        b: Second value

    Returns:
        Number of ULPs between a and b

    Raises:
        InvalidValueError: If either value is NaN or inf

    Examples:
        >>> diff(1.0, 1.0)
        0
        >>> diff(1.0, next(1.0))
        1
    """
    if math.isnan(a) or math.isnan(b):
        raise InvalidValueError("Cannot compute ULP distance with NaN")

    if math.isinf(a) or math.isinf(b):
        raise InvalidValueError("Cannot compute ULP distance with infinity")

    if a == b:
        return 0

    # Count ULPs between a and b
    count = 0
    current = a
    direction = math.inf if b > a else -math.inf

    while current != b and count < 1000000:  # Safety limit
        current = math.nextafter(current, direction)
        count += 1

    return count


def near(a: float, b: float, *, max_ulps: int = 4) -> bool:
    """Check if a and b are within max_ulps of each other.

    Args:
        a: First value
        b: Second value
        max_ulps: Maximum ULP distance (default 4)

    Returns:
        True if a and b are within max_ulps

    Examples:
        >>> near(1.0, 1.0)
        True
        >>> near(1.0, add(1.0, 4))
        True
        >>> near(1.0, add(1.0, 5))
        False
    """
    # Special case: exact equality
    if a == b:
        return True

    # NaN or inf handling
    if math.isnan(a) or math.isnan(b):
        return False
    if math.isinf(a) or math.isinf(b):
        return a == b

    try:
        return diff(a, b) <= max_ulps
    except InvalidValueError:
        return False


def next(x: float) -> float:
    """Return the next representable float above x.

    Args:
        x: Value

    Returns:
        Next float above x

    Examples:
        >>> next(1.0) > 1.0
        True
        >>> next(float('inf'))
        inf
    """
    return math.nextafter(x, math.inf)


def prev(x: float) -> float:
    """Return the next representable float below x.

    Args:
        x: Value

    Returns:
        Next float below x

    Examples:
        >>> prev(1.0) < 1.0
        True
        >>> prev(float('-inf'))
        -inf
    """
    return math.nextafter(x, -math.inf)


def add(x: float, n: int) -> float:
    """Move n ULPs from x.

    Args:
        x: Starting value
        n: Number of ULPs to move (positive or negative)

    Returns:
        Value n ULPs away from x

    Examples:
        >>> add(1.0, 0) == 1.0
        True
        >>> add(1.0, 1) == next(1.0)
        True
        >>> add(1.0, -1) == prev(1.0)
        True
    """
    if n == 0:
        return x

    direction = math.inf if n > 0 else -math.inf
    current = x

    for _ in range(abs(n)):
        current = math.nextafter(current, direction)

    return current
