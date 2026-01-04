"""Tests for safe module."""

import math

import pytest

from neon import safe
from neon.exceptions import EmptyInputError


class TestDiv:
    """Tests for div() function."""

    def test_normal_division(self) -> None:
        assert safe.div(6, 3) == 2.0
        assert safe.div(1, 2) == 0.5

    def test_division_by_zero(self) -> None:
        assert safe.div(1, 0) is None
        assert safe.div(1, 0, default=0.0) == 0.0
        assert safe.div(1, 0, default=99.0) == 99.0

    def test_near_zero_tolerance(self) -> None:
        assert safe.div(1, 1e-15, zero_tol=1e-10) is None
        result = safe.div(1, 1e-15, zero_tol=0.0)
        assert result is not None and abs(result - 1e15) < 1e14


class TestDivOrZero:
    """Tests for div_or_zero() function."""

    def test_normal_division(self) -> None:
        assert safe.div_or_zero(6, 3) == 2.0

    def test_division_by_zero(self) -> None:
        assert safe.div_or_zero(1, 0) == 0.0


class TestDivOrInf:
    """Tests for div_or_inf() function."""

    def test_normal_division(self) -> None:
        assert safe.div_or_inf(6, 3) == 2.0

    def test_division_by_zero(self) -> None:
        assert safe.div_or_inf(1, 0) == float("inf")
        assert safe.div_or_inf(-1, 0) == float("-inf")

    def test_zero_by_zero(self) -> None:
        result = safe.div_or_inf(0, 0)
        assert math.isnan(result)


class TestMod:
    """Tests for mod() function."""

    def test_normal_modulo(self) -> None:
        assert safe.mod(7, 3) == 1.0
        assert safe.mod(10, 4) == 2.0

    def test_modulo_by_zero(self) -> None:
        assert safe.mod(7, 0) is None
        assert safe.mod(7, 0, default=0.0) == 0.0


class TestSqrt:
    """Tests for sqrt() function."""

    def test_positive_values(self) -> None:
        assert safe.sqrt(4) == 2.0
        assert safe.sqrt(0) == 0.0
        assert safe.sqrt(9) == 3.0

    def test_negative_values(self) -> None:
        assert safe.sqrt(-1) is None
        assert safe.sqrt(-1, default=0.0) == 0.0


class TestLog:
    """Tests for log() function."""

    def test_natural_log(self) -> None:
        assert abs(safe.log(math.e) - 1.0) < 1e-10  # type: ignore
        assert abs(safe.log(1) - 0.0) < 1e-10  # type: ignore

    def test_custom_base(self) -> None:
        assert safe.log(100, base=10) == 2.0
        assert safe.log(8, base=2) == 3.0

    def test_invalid_inputs(self) -> None:
        assert safe.log(0) is None
        assert safe.log(-1) is None
        assert safe.log(-1, default=0.0) == 0.0


class TestPow:
    """Tests for pow() function."""

    def test_normal_power(self) -> None:
        assert safe.pow(2, 3) == 8.0
        assert safe.pow(10, 2) == 100.0

    def test_special_cases(self) -> None:
        assert safe.pow(0, 0) == 1.0  # By convention
        # Note: Python 3 doesn't raise for (-1)**0.5, it returns complex
        # So safe.pow won't return None for this case


class TestSumExact:
    """Tests for sum_exact() using math.fsum()."""

    def test_basic_sum(self) -> None:
        assert safe.sum_exact([1.0, 2.0, 3.0]) == 6.0

    def test_improved_precision(self) -> None:
        # math.fsum() provides exact summation for typical cases
        values = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        exact_sum = safe.sum_exact(values)

        # fsum is exact for this case
        assert exact_sum == 1.0

    def test_many_small_values(self) -> None:
        # Adding 0.1 ten times
        result = safe.sum_exact([0.1] * 10)
        assert abs(result - 1.0) < 1e-10

    def test_empty_input(self) -> None:
        with pytest.raises(EmptyInputError):
            safe.sum_exact([])


class TestMeanExact:
    """Tests for mean_exact() function."""

    def test_basic_mean(self) -> None:
        assert safe.mean_exact([1.0, 2.0, 3.0]) == 2.0

    def test_empty_input(self) -> None:
        with pytest.raises(EmptyInputError):
            safe.mean_exact([])
