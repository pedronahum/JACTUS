"""Unit tests for date/time handling.

Test ID: T1.2
"""

from datetime import datetime

import jax

from jactus.core.time import (
    ActusDateTime,
    adjust_to_business_day,
    is_business_day,
    parse_cycle,
    parse_iso_datetime,
)
from jactus.core.types import BusinessDayConvention, Calendar, EndOfMonthConvention


class TestActusDateTime:
    """Test ActusDateTime class."""

    def test_init_valid(self):
        """Test creating valid ActusDateTime instances."""
        dt = ActusDateTime(2024, 1, 15, 12, 30, 45)
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15
        assert dt.hour == 12
        assert dt.minute == 30
        assert dt.second == 45

    def test_init_defaults(self):
        """Test default time values (00:00:00)."""
        dt = ActusDateTime(2024, 1, 15)
        assert dt.hour == 0
        assert dt.minute == 0
        assert dt.second == 0

    def test_init_end_of_day(self):
        """Test 24:00:00 (end of day) format."""
        dt = ActusDateTime(2024, 1, 15, 24, 0, 0)
        assert dt.hour == 24

    def test_init_invalid_hour_24(self):
        """Test that 24:XX:XX is invalid (only 24:00:00 allowed)."""
        try:
            ActusDateTime(2024, 1, 15, 24, 30, 0)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "24:00:00" in str(e)

    def test_init_invalid_ranges(self):
        """Test validation of component ranges."""
        # Invalid year
        try:
            ActusDateTime(0, 1, 15)
            assert False
        except ValueError:
            pass

        # Invalid month
        try:
            ActusDateTime(2024, 13, 15)
            assert False
        except ValueError:
            pass

        # Invalid day
        try:
            ActusDateTime(2024, 1, 32)
            assert False
        except ValueError:
            pass

        # Invalid hour
        try:
            ActusDateTime(2024, 1, 15, 25)
            assert False
        except ValueError:
            pass


class TestISOParsing:
    """Test ISO 8601 parsing."""

    def test_parse_date_only(self):
        """Test parsing date-only format."""
        dt = parse_iso_datetime("2024-01-15")
        assert dt == ActusDateTime(2024, 1, 15, 0, 0, 0)

    def test_parse_datetime_t_separator(self):
        """Test parsing with T separator."""
        dt = parse_iso_datetime("2024-01-15T12:30:45")
        assert dt == ActusDateTime(2024, 1, 15, 12, 30, 45)

    def test_parse_datetime_space_separator(self):
        """Test parsing with space separator."""
        dt = parse_iso_datetime("2024-01-15 12:30:45")
        assert dt == ActusDateTime(2024, 1, 15, 12, 30, 45)

    def test_parse_end_of_day(self):
        """Test parsing 24:00:00 format."""
        dt = parse_iso_datetime("2024-01-15T24:00:00")
        assert dt == ActusDateTime(2024, 1, 15, 24, 0, 0)

    def test_parse_invalid_format(self):
        """Test that invalid formats raise ValueError."""
        try:
            parse_iso_datetime("15-01-2024")
            assert False
        except ValueError as e:
            assert "Invalid ISO 8601" in str(e)

    def test_from_iso_class_method(self):
        """Test ActusDateTime.from_iso() class method."""
        dt = ActusDateTime.from_iso("2024-01-15T12:30:45")
        assert dt == ActusDateTime(2024, 1, 15, 12, 30, 45)


class TestISOOutput:
    """Test conversion to ISO 8601 strings."""

    def test_to_iso_full(self):
        """Test converting to ISO 8601 string."""
        dt = ActusDateTime(2024, 1, 15, 12, 30, 45)
        assert dt.to_iso() == "2024-01-15T12:30:45"

    def test_to_iso_midnight(self):
        """Test converting midnight to ISO."""
        dt = ActusDateTime(2024, 1, 15, 0, 0, 0)
        assert dt.to_iso() == "2024-01-15T00:00:00"

    def test_to_iso_end_of_day(self):
        """Test converting 24:00:00 to ISO."""
        dt = ActusDateTime(2024, 1, 15, 24, 0, 0)
        assert dt.to_iso() == "2024-01-15T24:00:00"


class TestDateTimeConversion:
    """Test conversion to Python datetime."""

    def test_to_datetime_normal(self):
        """Test normal datetime conversion."""
        dt = ActusDateTime(2024, 1, 15, 12, 30, 45)
        py_dt = dt.to_datetime()
        assert py_dt == datetime(2024, 1, 15, 12, 30, 45)

    def test_to_datetime_end_of_day(self):
        """Test that 24:00:00 converts to next day midnight."""
        dt = ActusDateTime(2024, 1, 15, 24, 0, 0)
        py_dt = dt.to_datetime()
        assert py_dt == datetime(2024, 1, 16, 0, 0, 0)

    def test_to_datetime_month_boundary(self):
        """Test 24:00:00 crossing month boundary."""
        dt = ActusDateTime(2024, 1, 31, 24, 0, 0)
        py_dt = dt.to_datetime()
        assert py_dt == datetime(2024, 2, 1, 0, 0, 0)


class TestComparison:
    """Test datetime comparison operators."""

    def test_equality(self):
        """Test equality comparison."""
        dt1 = ActusDateTime(2024, 1, 15, 12, 30, 0)
        dt2 = ActusDateTime(2024, 1, 15, 12, 30, 0)
        dt3 = ActusDateTime(2024, 1, 15, 12, 31, 0)
        assert dt1 == dt2
        assert dt1 != dt3

    def test_less_than(self):
        """Test less than comparison."""
        dt1 = ActusDateTime(2024, 1, 15, 12, 30, 0)
        dt2 = ActusDateTime(2024, 1, 15, 12, 31, 0)
        assert dt1 < dt2
        assert not dt2 < dt1

    def test_less_equal(self):
        """Test less than or equal comparison."""
        dt1 = ActusDateTime(2024, 1, 15, 12, 30, 0)
        dt2 = ActusDateTime(2024, 1, 15, 12, 30, 0)
        dt3 = ActusDateTime(2024, 1, 15, 12, 31, 0)
        assert dt1 <= dt2
        assert dt1 <= dt3

    def test_greater_than(self):
        """Test greater than comparison."""
        dt1 = ActusDateTime(2024, 1, 15, 12, 31, 0)
        dt2 = ActusDateTime(2024, 1, 15, 12, 30, 0)
        assert dt1 > dt2
        assert not dt2 > dt1

    def test_greater_equal(self):
        """Test greater than or equal comparison."""
        dt1 = ActusDateTime(2024, 1, 15, 12, 30, 0)
        dt2 = ActusDateTime(2024, 1, 15, 12, 30, 0)
        dt3 = ActusDateTime(2024, 1, 15, 12, 29, 0)
        assert dt1 >= dt2
        assert dt1 >= dt3


class TestHashing:
    """Test hash support for dicts/sets."""

    def test_hash_equal_objects(self):
        """Test that equal objects have equal hashes."""
        dt1 = ActusDateTime(2024, 1, 15, 12, 30, 0)
        dt2 = ActusDateTime(2024, 1, 15, 12, 30, 0)
        assert hash(dt1) == hash(dt2)

    def test_hash_in_dict(self):
        """Test using ActusDateTime as dict key."""
        dt = ActusDateTime(2024, 1, 15, 12, 30, 0)
        d = {dt: "value"}
        assert d[dt] == "value"

    def test_hash_in_set(self):
        """Test using ActusDateTime in sets."""
        dt1 = ActusDateTime(2024, 1, 15, 12, 30, 0)
        dt2 = ActusDateTime(2024, 1, 15, 12, 30, 0)
        s = {dt1, dt2}
        assert len(s) == 1  # Same datetime


class TestParseCycle:
    """Test cycle string parsing."""

    def test_parse_days(self):
        """Test parsing day cycles."""
        assert parse_cycle("1D") == (1, "D", "")
        assert parse_cycle("30D") == (30, "D", "")

    def test_parse_weeks(self):
        """Test parsing week cycles."""
        assert parse_cycle("1W") == (1, "W", "")
        assert parse_cycle("2W") == (2, "W", "")

    def test_parse_months(self):
        """Test parsing month cycles."""
        assert parse_cycle("1M") == (1, "M", "")
        assert parse_cycle("3M") == (3, "M", "")
        assert parse_cycle("12M") == (12, "M", "")

    def test_parse_quarters(self):
        """Test parsing quarter cycles."""
        assert parse_cycle("1Q") == (1, "Q", "")
        assert parse_cycle("2Q") == (2, "Q", "")

    def test_parse_half_years(self):
        """Test parsing half-year cycles."""
        assert parse_cycle("1H") == (1, "H", "")

    def test_parse_years(self):
        """Test parsing year cycles."""
        assert parse_cycle("1Y") == (1, "Y", "")
        assert parse_cycle("5Y") == (5, "Y", "")

    def test_parse_with_stub_short(self):
        """Test parsing with short stub indicator."""
        assert parse_cycle("3M-") == (3, "M", "-")

    def test_parse_with_stub_long(self):
        """Test parsing with long stub indicator."""
        assert parse_cycle("1Y+") == (1, "Y", "+")

    def test_parse_case_insensitive(self):
        """Test that parsing is case insensitive."""
        assert parse_cycle("3m") == (3, "M", "")
        assert parse_cycle("1y") == (1, "Y", "")

    def test_parse_invalid_format(self):
        """Test that invalid cycles raise ValueError."""
        try:
            parse_cycle("3X")  # Invalid period type
            assert False
        except ValueError as e:
            assert "Invalid cycle format" in str(e)

        try:
            parse_cycle("M3")  # Wrong order
            assert False
        except ValueError:
            pass


class TestAddPeriod:
    """Test adding periods to dates."""

    def test_add_days(self):
        """Test adding days."""
        dt = ActusDateTime(2024, 1, 15, 0, 0, 0)
        result = dt.add_period("5D")
        assert result == ActusDateTime(2024, 1, 20, 0, 0, 0)

    def test_add_weeks(self):
        """Test adding weeks."""
        dt = ActusDateTime(2024, 1, 15, 0, 0, 0)
        result = dt.add_period("2W")
        assert result == ActusDateTime(2024, 1, 29, 0, 0, 0)

    def test_add_months(self):
        """Test adding months."""
        dt = ActusDateTime(2024, 1, 15, 0, 0, 0)
        result = dt.add_period("3M")
        assert result == ActusDateTime(2024, 4, 15, 0, 0, 0)

    def test_add_quarters(self):
        """Test adding quarters."""
        dt = ActusDateTime(2024, 1, 15, 0, 0, 0)
        result = dt.add_period("2Q")
        assert result == ActusDateTime(2024, 7, 15, 0, 0, 0)

    def test_add_half_years(self):
        """Test adding half years."""
        dt = ActusDateTime(2024, 1, 15, 0, 0, 0)
        result = dt.add_period("1H")
        assert result == ActusDateTime(2024, 7, 15, 0, 0, 0)

    def test_add_years(self):
        """Test adding years."""
        dt = ActusDateTime(2024, 1, 15, 0, 0, 0)
        result = dt.add_period("2Y")
        assert result == ActusDateTime(2026, 1, 15, 0, 0, 0)

    def test_add_month_overflow_day(self):
        """Test adding month when day doesn't exist in target."""
        dt = ActusDateTime(2024, 1, 31, 0, 0, 0)
        result = dt.add_period("1M")
        # Jan 31 + 1M = Feb 29 (2024 is leap year)
        assert result == ActusDateTime(2024, 2, 29, 0, 0, 0)

    def test_add_month_overflow_non_leap(self):
        """Test month overflow in non-leap year."""
        dt = ActusDateTime(2023, 1, 31, 0, 0, 0)
        result = dt.add_period("1M")
        # Jan 31 + 1M = Feb 28 (2023 is not leap year)
        assert result == ActusDateTime(2023, 2, 28, 0, 0, 0)

    def test_add_year_leap_day(self):
        """Test adding year to leap day."""
        dt = ActusDateTime(2024, 2, 29, 0, 0, 0)
        result = dt.add_period("1Y")
        # Feb 29 2024 + 1Y = Feb 28 2025 (not leap year)
        assert result == ActusDateTime(2025, 2, 28, 0, 0, 0)


class TestEndOfMonthConvention:
    """Test end-of-month convention handling."""

    def test_eom_convention_applied(self):
        """Test that EOM convention moves to end of target month."""
        dt = ActusDateTime(2024, 1, 31, 0, 0, 0)  # End of Jan
        result = dt.add_period("1M", EndOfMonthConvention.EOM)
        assert result == ActusDateTime(2024, 2, 29, 0, 0, 0)  # End of Feb (leap)

    def test_eom_convention_non_leap(self):
        """Test EOM convention in non-leap year."""
        dt = ActusDateTime(2023, 1, 31, 0, 0, 0)
        result = dt.add_period("1M", EndOfMonthConvention.EOM)
        assert result == ActusDateTime(2023, 2, 28, 0, 0, 0)

    def test_eom_convention_mid_month(self):
        """Test that EOM convention doesn't affect mid-month dates."""
        dt = ActusDateTime(2024, 1, 15, 0, 0, 0)  # Mid-month
        result = dt.add_period("1M", EndOfMonthConvention.EOM)
        assert result == ActusDateTime(2024, 2, 15, 0, 0, 0)  # Still mid-month

    def test_sd_convention(self):
        """Test that SD (Same Day) convention is default."""
        dt = ActusDateTime(2024, 1, 31, 0, 0, 0)
        result = dt.add_period("1M", EndOfMonthConvention.SD)
        # SD tries to keep day 31, but Feb doesn't have it
        assert result == ActusDateTime(2024, 2, 29, 0, 0, 0)


class TestIsEndOfMonth:
    """Test end-of-month detection."""

    def test_is_eom_january(self):
        """Test end of January."""
        dt = ActusDateTime(2024, 1, 31, 0, 0, 0)
        assert dt.is_end_of_month()

    def test_is_eom_february_leap(self):
        """Test end of February in leap year."""
        dt = ActusDateTime(2024, 2, 29, 0, 0, 0)
        assert dt.is_end_of_month()

    def test_is_eom_february_non_leap(self):
        """Test end of February in non-leap year."""
        dt = ActusDateTime(2023, 2, 28, 0, 0, 0)
        assert dt.is_end_of_month()

    def test_not_eom(self):
        """Test mid-month date."""
        dt = ActusDateTime(2024, 1, 15, 0, 0, 0)
        assert not dt.is_end_of_month()

    def test_not_eom_feb_28_leap(self):
        """Test that Feb 28 is not EOM in leap year."""
        dt = ActusDateTime(2024, 2, 28, 0, 0, 0)
        assert not dt.is_end_of_month()  # Feb 29 exists


class TestDaysBetween:
    """Test days_between calculation."""

    def test_days_between_positive(self):
        """Test positive day difference."""
        dt1 = ActusDateTime(2024, 1, 15, 0, 0, 0)
        dt2 = ActusDateTime(2024, 1, 20, 0, 0, 0)
        assert dt1.days_between(dt2) == 5

    def test_days_between_negative(self):
        """Test negative day difference."""
        dt1 = ActusDateTime(2024, 1, 20, 0, 0, 0)
        dt2 = ActusDateTime(2024, 1, 15, 0, 0, 0)
        assert dt1.days_between(dt2) == -5

    def test_days_between_same(self):
        """Test same date."""
        dt = ActusDateTime(2024, 1, 15, 0, 0, 0)
        assert dt.days_between(dt) == 0

    def test_days_between_month_boundary(self):
        """Test across month boundary."""
        dt1 = ActusDateTime(2024, 1, 25, 0, 0, 0)
        dt2 = ActusDateTime(2024, 2, 5, 0, 0, 0)
        assert dt1.days_between(dt2) == 11

    def test_days_between_year_boundary(self):
        """Test across year boundary."""
        dt1 = ActusDateTime(2023, 12, 25, 0, 0, 0)
        dt2 = ActusDateTime(2024, 1, 5, 0, 0, 0)
        assert dt1.days_between(dt2) == 11


class TestYearsBetween:
    """Test years_between calculation."""

    def test_years_between_one_year(self):
        """Test one year difference."""
        dt1 = ActusDateTime(2024, 1, 15, 0, 0, 0)
        dt2 = ActusDateTime(2025, 1, 15, 0, 0, 0)
        years = dt1.years_between(dt2)
        assert abs(years - 1.0) < 0.01  # Approximately 1 year

    def test_years_between_half_year(self):
        """Test half year difference."""
        dt1 = ActusDateTime(2024, 1, 15, 0, 0, 0)
        dt2 = ActusDateTime(2024, 7, 15, 0, 0, 0)
        years = dt1.years_between(dt2)
        assert abs(years - 0.5) < 0.01

    def test_years_between_negative(self):
        """Test negative years."""
        dt1 = ActusDateTime(2025, 1, 15, 0, 0, 0)
        dt2 = ActusDateTime(2024, 1, 15, 0, 0, 0)
        years = dt1.years_between(dt2)
        assert years < 0


class TestIsBusinessDay:
    """Test business day detection."""

    def test_monday_is_business_day(self):
        """Test that Monday is a business day."""
        dt = ActusDateTime(2024, 1, 15, 0, 0, 0)  # Monday
        assert is_business_day(dt, Calendar.MONDAY_TO_FRIDAY)

    def test_friday_is_business_day(self):
        """Test that Friday is a business day."""
        dt = ActusDateTime(2024, 1, 19, 0, 0, 0)  # Friday
        assert is_business_day(dt, Calendar.MONDAY_TO_FRIDAY)

    def test_saturday_is_not_business_day(self):
        """Test that Saturday is not a business day."""
        dt = ActusDateTime(2024, 1, 13, 0, 0, 0)  # Saturday
        assert not is_business_day(dt, Calendar.MONDAY_TO_FRIDAY)

    def test_sunday_is_not_business_day(self):
        """Test that Sunday is not a business day."""
        dt = ActusDateTime(2024, 1, 14, 0, 0, 0)  # Sunday
        assert not is_business_day(dt, Calendar.MONDAY_TO_FRIDAY)

    def test_no_calendar_all_days_business(self):
        """Test that NO_CALENDAR treats all days as business days."""
        dt_saturday = ActusDateTime(2024, 1, 13, 0, 0, 0)
        dt_sunday = ActusDateTime(2024, 1, 14, 0, 0, 0)
        assert is_business_day(dt_saturday, Calendar.NO_CALENDAR)
        assert is_business_day(dt_sunday, Calendar.NO_CALENDAR)


class TestAdjustToBusinessDay:
    """Test business day adjustment."""

    def test_null_convention_no_adjustment(self):
        """Test that NULL convention makes no adjustment."""
        dt = ActusDateTime(2024, 1, 13, 0, 0, 0)  # Saturday
        result = adjust_to_business_day(dt, BusinessDayConvention.NULL)
        assert result == dt

    def test_already_business_day(self):
        """Test that business days are not adjusted."""
        dt = ActusDateTime(2024, 1, 15, 0, 0, 0)  # Monday
        result = adjust_to_business_day(dt, BusinessDayConvention.SCF)
        assert result == dt

    def test_following_from_saturday(self):
        """Test following convention from Saturday."""
        dt = ActusDateTime(2024, 1, 13, 0, 0, 0)  # Saturday
        result = adjust_to_business_day(dt, BusinessDayConvention.SCF)
        assert result == ActusDateTime(2024, 1, 15, 0, 0, 0)  # Monday

    def test_following_from_sunday(self):
        """Test following convention from Sunday."""
        dt = ActusDateTime(2024, 1, 14, 0, 0, 0)  # Sunday
        result = adjust_to_business_day(dt, BusinessDayConvention.SCF)
        assert result == ActusDateTime(2024, 1, 15, 0, 0, 0)  # Monday

    def test_preceding_from_saturday(self):
        """Test preceding convention from Saturday."""
        dt = ActusDateTime(2024, 1, 13, 0, 0, 0)  # Saturday
        result = adjust_to_business_day(dt, BusinessDayConvention.SCP)
        assert result == ActusDateTime(2024, 1, 12, 0, 0, 0)  # Friday


class TestJAXPytree:
    """Test JAX pytree registration."""

    def test_pytree_flatten_unflatten(self):
        """Test that ActusDateTime can be flattened and unflattened."""
        dt = ActusDateTime(2024, 1, 15, 12, 30, 45)
        flat, tree_def = jax.tree_util.tree_flatten(dt)
        reconstructed = jax.tree_util.tree_unflatten(tree_def, flat)
        assert reconstructed == dt

    def test_pytree_in_structure(self):
        """Test ActusDateTime in nested structure."""
        data = {
            "dates": [
                ActusDateTime(2024, 1, 15, 0, 0, 0),
                ActusDateTime(2024, 2, 15, 0, 0, 0),
            ],
            "count": 2,
        }
        flat, tree_def = jax.tree_util.tree_flatten(data)
        reconstructed = jax.tree_util.tree_unflatten(tree_def, flat)
        assert reconstructed["dates"][0] == data["dates"][0]
        assert reconstructed["dates"][1] == data["dates"][1]

    def test_pytree_with_jax_function(self):
        """Test that pytree registration works with JAX functions."""

        def process_date(dt: ActusDateTime) -> int:
            """Extract year from ActusDateTime."""
            return dt.year

        dt = ActusDateTime(2024, 1, 15, 0, 0, 0)
        result = jax.tree_util.tree_map(lambda x: x if not isinstance(x, int) else x * 2, dt)
        # Just verify it doesn't crash - JAX can process the structure
        assert isinstance(result, ActusDateTime)
