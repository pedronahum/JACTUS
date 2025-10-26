"""Tests for day count convention implementations.

Tests the implementation of day count conventions according to
ACTUS Technical Specification v1.1, Section 4 and ISDA 2006 Definitions.
"""

from __future__ import annotations

import pytest

from jactus.core.time import ActusDateTime
from jactus.core.types import DayCountConvention
from jactus.utilities.conventions import (
    days_between_30_360_methods,
    year_fraction,
)


class TestActualActual:
    """Test Actual/Actual ISDA (A/A) convention."""

    def test_same_year(self):
        """Year fraction within same year."""
        start = ActusDateTime(2024, 1, 1, 0, 0, 0)
        end = ActusDateTime(2024, 7, 1, 0, 0, 0)
        result = year_fraction(start, end, DayCountConvention.AA)

        # 182 days in leap year (2024) / 366
        expected = 182 / 366
        assert abs(result - expected) < 0.0001

    def test_across_years(self):
        """Year fraction across multiple years."""
        start = ActusDateTime(2023, 7, 1, 0, 0, 0)
        end = ActusDateTime(2024, 7, 1, 0, 0, 0)
        result = year_fraction(start, end, DayCountConvention.AA)

        # Jul 1, 2023 to Dec 31, 2023: 184 days (inclusive) in 2023 (non-leap) / 365
        # Jan 1, 2024 to Jul 1, 2024: 182 days in 2024 (leap) / 366
        expected = 184 / 365 + 182 / 366
        assert abs(result - expected) < 0.0001

    def test_leap_year(self):
        """Full leap year."""
        start = ActusDateTime(2024, 1, 1, 0, 0, 0)
        end = ActusDateTime(2025, 1, 1, 0, 0, 0)
        result = year_fraction(start, end, DayCountConvention.AA)

        # Full leap year = 366/366 = 1.0
        assert abs(result - 1.0) < 0.0001

    def test_non_leap_year(self):
        """Full non-leap year."""
        start = ActusDateTime(2023, 1, 1, 0, 0, 0)
        end = ActusDateTime(2024, 1, 1, 0, 0, 0)
        result = year_fraction(start, end, DayCountConvention.AA)

        # Full non-leap year = 365/365 = 1.0
        assert abs(result - 1.0) < 0.0001

    def test_six_months(self):
        """Six months in a leap year."""
        start = ActusDateTime(2024, 1, 15, 0, 0, 0)
        end = ActusDateTime(2024, 7, 15, 0, 0, 0)
        result = year_fraction(start, end, DayCountConvention.AA)

        # 182 days / 366
        expected = 182 / 366
        assert abs(result - expected) < 0.0001

    def test_start_equals_end(self):
        """Zero days returns zero."""
        start = ActusDateTime(2024, 1, 15, 0, 0, 0)
        result = year_fraction(start, start, DayCountConvention.AA)
        assert result == 0.0


class TestActual360:
    """Test Actual/360 (A/360) convention."""

    def test_six_months(self):
        """Six months using A/360."""
        start = ActusDateTime(2024, 1, 15, 0, 0, 0)
        end = ActusDateTime(2024, 7, 15, 0, 0, 0)
        result = year_fraction(start, end, DayCountConvention.A360)

        # 182 actual days / 360
        expected = 182 / 360
        assert abs(result - expected) < 0.0001

    def test_one_year(self):
        """One year using A/360."""
        start = ActusDateTime(2024, 1, 1, 0, 0, 0)
        end = ActusDateTime(2025, 1, 1, 0, 0, 0)
        result = year_fraction(start, end, DayCountConvention.A360)

        # 366 days (leap year) / 360
        expected = 366 / 360
        assert abs(result - expected) < 0.0001

    def test_30_days(self):
        """30 days using A/360."""
        start = ActusDateTime(2024, 1, 1, 0, 0, 0)
        end = ActusDateTime(2024, 1, 31, 0, 0, 0)
        result = year_fraction(start, end, DayCountConvention.A360)

        # 30 actual days / 360
        expected = 30 / 360
        assert abs(result - expected) < 0.0001


class TestActual365:
    """Test Actual/365 Fixed (A/365) convention."""

    def test_six_months(self):
        """Six months using A/365."""
        start = ActusDateTime(2024, 1, 15, 0, 0, 0)
        end = ActusDateTime(2024, 7, 15, 0, 0, 0)
        result = year_fraction(start, end, DayCountConvention.A365)

        # 182 actual days / 365
        expected = 182 / 365
        assert abs(result - expected) < 0.0001

    def test_one_year(self):
        """One year using A/365."""
        start = ActusDateTime(2024, 1, 1, 0, 0, 0)
        end = ActusDateTime(2025, 1, 1, 0, 0, 0)
        result = year_fraction(start, end, DayCountConvention.A365)

        # 366 days (leap year) / 365 > 1.0
        expected = 366 / 365
        assert abs(result - expected) < 0.0001

    def test_non_leap_year(self):
        """Non-leap year using A/365."""
        start = ActusDateTime(2023, 1, 1, 0, 0, 0)
        end = ActusDateTime(2024, 1, 1, 0, 0, 0)
        result = year_fraction(start, end, DayCountConvention.A365)

        # 365 days / 365 = 1.0
        expected = 1.0
        assert abs(result - expected) < 0.0001


class Test30E360:
    """Test 30E/360 (Eurobond basis) convention."""

    def test_six_months(self):
        """Six months using 30E/360."""
        start = ActusDateTime(2024, 2, 15, 0, 0, 0)
        end = ActusDateTime(2024, 8, 15, 0, 0, 0)
        result = year_fraction(start, end, DayCountConvention.E30360)

        # (8-2)*30 + (15-15) = 180 days / 360 = 0.5
        expected = 0.5
        assert abs(result - expected) < 0.0001

    def test_one_year(self):
        """One year using 30E/360."""
        start = ActusDateTime(2024, 1, 15, 0, 0, 0)
        end = ActusDateTime(2025, 1, 15, 0, 0, 0)
        result = year_fraction(start, end, DayCountConvention.E30360)

        # 1*360 = 360 days / 360 = 1.0
        expected = 1.0
        assert abs(result - expected) < 0.0001

    def test_31st_adjustment(self):
        """Test 31st day adjustment."""
        start = ActusDateTime(2024, 1, 31, 0, 0, 0)
        end = ActusDateTime(2024, 3, 31, 0, 0, 0)
        result = year_fraction(start, end, DayCountConvention.E30360)

        # D1=31->30, D2=31->30
        # (3-1)*30 + (30-30) = 60 days / 360
        expected = 60 / 360
        assert abs(result - expected) < 0.0001

    def test_mixed_month_lengths(self):
        """Test across months of different lengths."""
        start = ActusDateTime(2024, 1, 15, 0, 0, 0)
        end = ActusDateTime(2024, 2, 29, 0, 0, 0)  # Leap day
        result = year_fraction(start, end, DayCountConvention.E30360)

        # (2-1)*30 + (29-15) = 30 + 14 = 44 days / 360
        expected = 44 / 360
        assert abs(result - expected) < 0.0001


class Test30360:
    """Test 30/360 (Bond Basis, US) convention."""

    def test_six_months(self):
        """Six months using 30/360."""
        start = ActusDateTime(2024, 2, 15, 0, 0, 0)
        end = ActusDateTime(2024, 8, 15, 0, 0, 0)
        result = year_fraction(start, end, DayCountConvention.B30360)

        # (8-2)*30 + (15-15) = 180 days / 360 = 0.5
        expected = 0.5
        assert abs(result - expected) < 0.0001

    def test_31st_to_31st(self):
        """Test 31st to 31st adjustment."""
        start = ActusDateTime(2024, 1, 31, 0, 0, 0)
        end = ActusDateTime(2024, 3, 31, 0, 0, 0)
        result = year_fraction(start, end, DayCountConvention.B30360)

        # D1=31->30, D2=31->30 (because D1>=30)
        # (3-1)*30 + (30-30) = 60 days / 360
        expected = 60 / 360
        assert abs(result - expected) < 0.0001

    def test_30th_to_31st(self):
        """Test 30th to 31st adjustment."""
        start = ActusDateTime(2024, 1, 30, 0, 0, 0)
        end = ActusDateTime(2024, 2, 31, 0, 0, 0)  # Feb doesn't have 31
        result = year_fraction(start, end, DayCountConvention.B30360)

        # D1=30, D2=31->30 (because D1>=30)
        # (2-1)*30 + (30-30) = 30 days / 360
        expected = 30 / 360
        assert abs(result - expected) < 0.0001


class Test30E360ISDA:
    """Test 30E/360 ISDA convention."""

    def test_six_months(self):
        """Six months using 30E/360 ISDA."""
        start = ActusDateTime(2024, 2, 15, 0, 0, 0)
        end = ActusDateTime(2024, 8, 15, 0, 0, 0)
        maturity = ActusDateTime(2029, 8, 15, 0, 0, 0)
        result = year_fraction(start, end, DayCountConvention.E30360ISDA, maturity)

        # (8-2)*30 + (15-15) = 180 days / 360 = 0.5
        expected = 0.5
        assert abs(result - expected) < 0.0001

    def test_february_last_day(self):
        """Test February last day adjustment."""
        start = ActusDateTime(2024, 2, 29, 0, 0, 0)  # Leap year last day of Feb
        end = ActusDateTime(2024, 3, 15, 0, 0, 0)
        maturity = ActusDateTime(2029, 3, 15, 0, 0, 0)
        result = year_fraction(start, end, DayCountConvention.E30360ISDA, maturity)

        # D1=29 (last of Feb)->30, D2=15
        # (3-2)*30 + (15-30) = 30 - 15 = 15 days / 360
        expected = 15 / 360
        assert abs(result - expected) < 0.0001

    def test_maturity_date_handling(self):
        """Test maturity date special handling."""
        start = ActusDateTime(2024, 1, 15, 0, 0, 0)
        end = ActusDateTime(2024, 2, 29, 0, 0, 0)
        maturity = ActusDateTime(2024, 2, 29, 0, 0, 0)  # End is maturity
        result = year_fraction(start, end, DayCountConvention.E30360ISDA, maturity)

        # End = maturity, so D2=29 (not adjusted to 30)
        # (2-1)*30 + (29-15) = 30 + 14 = 44 days / 360
        expected = 44 / 360
        assert abs(result - expected) < 0.0001

    def test_requires_maturity(self):
        """Test that maturity is required."""
        start = ActusDateTime(2024, 1, 15, 0, 0, 0)
        end = ActusDateTime(2024, 7, 15, 0, 0, 0)

        with pytest.raises(ValueError, match="Maturity date required"):
            year_fraction(start, end, DayCountConvention.E30360ISDA)


class TestBUS252:
    """Test BUS/252 (Brazilian business days) convention."""

    def test_one_week(self):
        """One week (5 business days)."""
        start = ActusDateTime(2024, 1, 1, 0, 0, 0)  # Monday
        end = ActusDateTime(2024, 1, 8, 0, 0, 0)  # Next Monday
        result = year_fraction(start, end, DayCountConvention.BUS252)

        # 5 business days (Mon-Fri) / 252
        expected = 5 / 252
        assert abs(result - expected) < 0.0001

    def test_includes_weekend(self):
        """Period including weekend."""
        start = ActusDateTime(2024, 1, 1, 0, 0, 0)  # Monday
        end = ActusDateTime(2024, 1, 15, 0, 0, 0)  # Two weeks later
        result = year_fraction(start, end, DayCountConvention.BUS252)

        # 10 business days (2 full weeks) / 252
        expected = 10 / 252
        assert abs(result - expected) < 0.0001

    def test_starting_on_weekend(self):
        """Starting on weekend day."""
        start = ActusDateTime(2024, 1, 6, 0, 0, 0)  # Saturday
        end = ActusDateTime(2024, 1, 13, 0, 0, 0)  # Next Saturday
        result = year_fraction(start, end, DayCountConvention.BUS252)

        # Only business days count: Mon-Fri = 5 days
        expected = 5 / 252
        assert abs(result - expected) < 0.0001


class TestDaysBetween30360Methods:
    """Test helper function for 30/360 day calculations."""

    def test_30e360_method(self):
        """Test 30E/360 day calculation."""
        start = ActusDateTime(2024, 2, 15, 0, 0, 0)
        end = ActusDateTime(2024, 8, 15, 0, 0, 0)
        result = days_between_30_360_methods(start, end, "30E/360")

        # (8-2)*30 + (15-15) = 180 days
        assert result == 180

    def test_30360_method(self):
        """Test 30/360 day calculation."""
        start = ActusDateTime(2024, 1, 31, 0, 0, 0)
        end = ActusDateTime(2024, 3, 31, 0, 0, 0)
        result = days_between_30_360_methods(start, end, "30/360")

        # D1=31->30, D2=31->30
        # (3-1)*30 + (30-30) = 60 days
        assert result == 60

    def test_30e360_isda_method(self):
        """Test 30E/360 ISDA day calculation."""
        start = ActusDateTime(2024, 2, 29, 0, 0, 0)  # Last day of Feb
        end = ActusDateTime(2024, 3, 15, 0, 0, 0)
        result = days_between_30_360_methods(start, end, "30E/360 ISDA")

        # D1=29->30 (last of Feb), D2=15
        # (3-2)*30 + (15-30) = 15 days
        assert result == 15

    def test_negative_days(self):
        """Test negative days (end before start)."""
        start = ActusDateTime(2024, 8, 15, 0, 0, 0)
        end = ActusDateTime(2024, 2, 15, 0, 0, 0)
        result = days_between_30_360_methods(start, end, "30E/360")

        # (2-8)*30 + (15-15) = -180 days
        assert result == -180

    def test_invalid_method(self):
        """Test invalid method raises error."""
        start = ActusDateTime(2024, 1, 15, 0, 0, 0)
        end = ActusDateTime(2024, 7, 15, 0, 0, 0)

        with pytest.raises(ValueError, match="Unknown method"):
            days_between_30_360_methods(start, end, "INVALID")


class TestEdgeCases:
    """Test edge cases across all conventions."""

    def test_same_date_all_conventions(self):
        """All conventions return 0 for same date."""
        date = ActusDateTime(2024, 1, 15, 0, 0, 0)

        for convention in [
            DayCountConvention.AA,
            DayCountConvention.A360,
            DayCountConvention.A365,
            DayCountConvention.E30360,
            DayCountConvention.B30360,
            DayCountConvention.BUS252,
        ]:
            result = year_fraction(date, date, convention)
            assert result == 0.0, f"Failed for {convention}"

    def test_one_day_difference(self):
        """One day difference across conventions."""
        start = ActusDateTime(2024, 1, 15, 0, 0, 0)
        end = ActusDateTime(2024, 1, 16, 0, 0, 0)

        # A/A: 1 day in leap year
        result_aa = year_fraction(start, end, DayCountConvention.AA)
        assert abs(result_aa - 1 / 366) < 0.0001

        # A/360: 1/360
        result_a360 = year_fraction(start, end, DayCountConvention.A360)
        assert abs(result_a360 - 1 / 360) < 0.0001

        # A/365: 1/365
        result_a365 = year_fraction(start, end, DayCountConvention.A365)
        assert abs(result_a365 - 1 / 365) < 0.0001

        # 30E/360: 1/360
        result_30e360 = year_fraction(start, end, DayCountConvention.E30360)
        assert abs(result_30e360 - 1 / 360) < 0.0001

    def test_leap_day(self):
        """Test including leap day."""
        start = ActusDateTime(2024, 2, 28, 0, 0, 0)
        end = ActusDateTime(2024, 3, 1, 0, 0, 0)

        # A/A: 2 days / 366
        result = year_fraction(start, end, DayCountConvention.AA)
        expected = 2 / 366
        assert abs(result - expected) < 0.0001

    def test_year_end_boundary(self):
        """Test across year-end boundary."""
        start = ActusDateTime(2024, 12, 15, 0, 0, 0)
        end = ActusDateTime(2025, 1, 15, 0, 0, 0)

        # A/A: 17 days in 2024 (leap) + 14 days in 2025
        result = year_fraction(start, end, DayCountConvention.AA)
        expected = 17 / 366 + 14 / 365
        assert abs(result - expected) < 0.0001

    def test_unsupported_convention(self):
        """Test unsupported convention raises error."""
        start = ActusDateTime(2024, 1, 15, 0, 0, 0)
        end = ActusDateTime(2024, 7, 15, 0, 0, 0)

        # Create an invalid enum value by casting
        with pytest.raises(ValueError, match="Unsupported day count convention"):
            year_fraction(start, end, "INVALID")  # type: ignore


class TestKnownValues:
    """Test against known reference values."""

    def test_standard_six_months(self):
        """Standard 6-month period across conventions."""
        start = ActusDateTime(2024, 1, 1, 0, 0, 0)
        end = ActusDateTime(2024, 7, 1, 0, 0, 0)

        # A/A: 182 days / 366 (leap year)
        assert abs(year_fraction(start, end, DayCountConvention.AA) - 182 / 366) < 0.0001

        # A/360: 182 / 360
        assert abs(year_fraction(start, end, DayCountConvention.A360) - 182 / 360) < 0.0001

        # A/365: 182 / 365
        assert abs(year_fraction(start, end, DayCountConvention.A365) - 182 / 365) < 0.0001

        # 30E/360: (7-1)*30 + (1-1) = 180 days / 360
        assert abs(year_fraction(start, end, DayCountConvention.E30360) - 180 / 360) < 0.0001

    def test_full_year_2024(self):
        """Full calendar year 2024 (leap year)."""
        start = ActusDateTime(2024, 1, 1, 0, 0, 0)
        end = ActusDateTime(2025, 1, 1, 0, 0, 0)

        # A/A: Should be exactly 1.0
        assert abs(year_fraction(start, end, DayCountConvention.AA) - 1.0) < 0.0001

        # A/360: 366 / 360 > 1
        assert abs(year_fraction(start, end, DayCountConvention.A360) - 366 / 360) < 0.0001

        # A/365: 366 / 365 > 1
        assert abs(year_fraction(start, end, DayCountConvention.A365) - 366 / 365) < 0.0001

        # 30E/360: 360 / 360 = 1.0
        assert abs(year_fraction(start, end, DayCountConvention.E30360) - 1.0) < 0.0001
