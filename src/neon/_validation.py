"""Internal validation utilities."""

from collections.abc import Sequence

from .exceptions import EmptyInputError


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
