"""Tests for business day calendar implementations.

Tests the implementation of holiday calendars and business day navigation
functions.
"""

from __future__ import annotations

import pytest

from jactus.core.time import ActusDateTime
from jactus.utilities.calendars import (
    CustomCalendar,
    MondayToFridayCalendar,
    NoHolidayCalendar,
    get_calendar,
    is_weekend,
)


class TestNoHolidayCalendar:
    """Test NoHolidayCalendar - every day is a business day."""

    def test_weekday_is_business_day(self):
        """Weekdays are business days."""
        cal = NoHolidayCalendar()
        monday = ActusDateTime(2024, 1, 1, 0, 0, 0)
        assert cal.is_business_day(monday) is True

    def test_weekend_is_business_day(self):
        """Even weekends are business days in NoHolidayCalendar."""
        cal = NoHolidayCalendar()
        saturday = ActusDateTime(2024, 1, 6, 0, 0, 0)
        sunday = ActusDateTime(2024, 1, 7, 0, 0, 0)
        assert cal.is_business_day(saturday) is True
        assert cal.is_business_day(sunday) is True

    def test_is_holiday_always_false(self):
        """No holidays in NoHolidayCalendar."""
        cal = NoHolidayCalendar()
        date = ActusDateTime(2024, 1, 1, 0, 0, 0)
        assert cal.is_holiday(date) is False

    def test_next_business_day_same_day(self):
        """Next business day is always the same day."""
        cal = NoHolidayCalendar()
        date = ActusDateTime(2024, 1, 6, 0, 0, 0)  # Saturday
        assert cal.next_business_day(date) == date

    def test_previous_business_day_same_day(self):
        """Previous business day is always the same day."""
        cal = NoHolidayCalendar()
        date = ActusDateTime(2024, 1, 7, 0, 0, 0)  # Sunday
        assert cal.previous_business_day(date) == date


class TestMondayToFridayCalendar:
    """Test MondayToFridayCalendar - weekends are not business days."""

    def test_monday_is_business_day(self):
        """Monday is a business day."""
        cal = MondayToFridayCalendar()
        monday = ActusDateTime(2024, 1, 1, 0, 0, 0)
        assert cal.is_business_day(monday) is True

    def test_friday_is_business_day(self):
        """Friday is a business day."""
        cal = MondayToFridayCalendar()
        friday = ActusDateTime(2024, 1, 5, 0, 0, 0)
        assert cal.is_business_day(friday) is True

    def test_saturday_not_business_day(self):
        """Saturday is not a business day."""
        cal = MondayToFridayCalendar()
        saturday = ActusDateTime(2024, 1, 6, 0, 0, 0)
        assert cal.is_business_day(saturday) is False
        assert cal.is_holiday(saturday) is True

    def test_sunday_not_business_day(self):
        """Sunday is not a business day."""
        cal = MondayToFridayCalendar()
        sunday = ActusDateTime(2024, 1, 7, 0, 0, 0)
        assert cal.is_business_day(sunday) is False
        assert cal.is_holiday(sunday) is True

    def test_next_business_day_from_friday(self):
        """Next business day from Friday is Monday."""
        cal = MondayToFridayCalendar()
        friday = ActusDateTime(2024, 1, 5, 0, 0, 0)
        # Friday is already a business day
        assert cal.next_business_day(friday) == friday

    def test_next_business_day_from_saturday(self):
        """Next business day from Saturday is Monday."""
        cal = MondayToFridayCalendar()
        saturday = ActusDateTime(2024, 1, 6, 0, 0, 0)
        monday = ActusDateTime(2024, 1, 8, 0, 0, 0)
        assert cal.next_business_day(saturday) == monday

    def test_next_business_day_from_sunday(self):
        """Next business day from Sunday is Monday."""
        cal = MondayToFridayCalendar()
        sunday = ActusDateTime(2024, 1, 7, 0, 0, 0)
        monday = ActusDateTime(2024, 1, 8, 0, 0, 0)
        assert cal.next_business_day(sunday) == monday

    def test_previous_business_day_from_monday(self):
        """Previous business day from Monday is Friday."""
        cal = MondayToFridayCalendar()
        monday = ActusDateTime(2024, 1, 8, 0, 0, 0)
        # Monday is already a business day
        assert cal.previous_business_day(monday) == monday

    def test_previous_business_day_from_saturday(self):
        """Previous business day from Saturday is Friday."""
        cal = MondayToFridayCalendar()
        saturday = ActusDateTime(2024, 1, 6, 0, 0, 0)
        friday = ActusDateTime(2024, 1, 5, 0, 0, 0)
        assert cal.previous_business_day(saturday) == friday

    def test_previous_business_day_from_sunday(self):
        """Previous business day from Sunday is Friday."""
        cal = MondayToFridayCalendar()
        sunday = ActusDateTime(2024, 1, 7, 0, 0, 0)
        friday = ActusDateTime(2024, 1, 5, 0, 0, 0)
        assert cal.previous_business_day(sunday) == friday


class TestCustomCalendar:
    """Test CustomCalendar with specific holidays."""

    def test_no_holidays_weekdays_ok(self):
        """Weekdays are business days when no holidays."""
        cal = CustomCalendar()
        monday = ActusDateTime(2024, 1, 1, 0, 0, 0)
        assert cal.is_business_day(monday) is True

    def test_weekends_not_business_days(self):
        """Weekends are not business days by default."""
        cal = CustomCalendar()
        saturday = ActusDateTime(2024, 1, 6, 0, 0, 0)
        assert cal.is_business_day(saturday) is False

    def test_add_holiday(self):
        """Can add custom holidays."""
        cal = CustomCalendar()
        # Jan 1, 2024 is a Monday
        jan1 = ActusDateTime(2024, 1, 1, 0, 0, 0)

        # Before adding, it's a business day
        assert cal.is_business_day(jan1) is True

        # Add as holiday
        cal.add_holiday(jan1)
        assert cal.is_business_day(jan1) is False
        assert cal.is_holiday(jan1) is True

    def test_remove_holiday(self):
        """Can remove holidays."""
        jan1 = ActusDateTime(2024, 1, 1, 0, 0, 0)
        cal = CustomCalendar(holidays=[jan1])

        # Initially a holiday
        assert cal.is_business_day(jan1) is False

        # Remove holiday
        cal.remove_holiday(jan1)
        assert cal.is_business_day(jan1) is True

    def test_load_from_list(self):
        """Can load holidays from a list."""
        cal = CustomCalendar()
        holidays = [
            ActusDateTime(2024, 1, 1, 0, 0, 0),  # New Year
            ActusDateTime(2024, 12, 25, 0, 0, 0),  # Christmas
        ]

        cal.load_from_list(holidays)

        assert cal.is_business_day(holidays[0]) is False
        assert cal.is_business_day(holidays[1]) is False

    def test_without_weekends(self):
        """Can create calendar without weekend treatment."""
        cal = CustomCalendar(include_weekends=False)
        saturday = ActusDateTime(2024, 1, 6, 0, 0, 0)

        # Saturday is a business day when weekends not included
        assert cal.is_business_day(saturday) is True

    def test_custom_holiday_on_weekend(self):
        """Custom holiday on weekend is still not a business day."""
        saturday = ActusDateTime(2024, 1, 6, 0, 0, 0)
        cal = CustomCalendar(holidays=[saturday])

        # Not a business day (weekend + custom holiday)
        assert cal.is_business_day(saturday) is False


class TestBusinessDayNavigation:
    """Test business day navigation functions."""

    def test_add_business_days_positive(self):
        """Add positive business days."""
        cal = MondayToFridayCalendar()
        friday = ActusDateTime(2024, 1, 5, 0, 0, 0)

        # Add 1 business day: Friday -> Monday
        result = cal.add_business_days(friday, 1)
        assert result == ActusDateTime(2024, 1, 8, 0, 0, 0)

    def test_add_business_days_skip_weekend(self):
        """Adding business days skips weekends."""
        cal = MondayToFridayCalendar()
        thursday = ActusDateTime(2024, 1, 4, 0, 0, 0)

        # Add 3 business days: Thu -> Fri -> Mon -> Tue
        result = cal.add_business_days(thursday, 3)
        assert result == ActusDateTime(2024, 1, 9, 0, 0, 0)  # Tuesday

    def test_add_business_days_negative(self):
        """Add negative business days (go backwards)."""
        cal = MondayToFridayCalendar()
        monday = ActusDateTime(2024, 1, 8, 0, 0, 0)

        # Subtract 1 business day: Monday -> Friday
        result = cal.add_business_days(monday, -1)
        assert result == ActusDateTime(2024, 1, 5, 0, 0, 0)

    def test_add_zero_business_days(self):
        """Adding zero business days returns same date."""
        cal = MondayToFridayCalendar()
        date = ActusDateTime(2024, 1, 5, 0, 0, 0)
        assert cal.add_business_days(date, 0) == date

    def test_business_days_between_same_week(self):
        """Count business days in same week."""
        cal = MondayToFridayCalendar()
        monday = ActusDateTime(2024, 1, 1, 0, 0, 0)
        friday = ActusDateTime(2024, 1, 5, 0, 0, 0)

        # Mon to Fri (exclusive): 4 business days
        assert cal.business_days_between(monday, friday) == 4

    def test_business_days_between_include_end(self):
        """Count business days including end."""
        cal = MondayToFridayCalendar()
        monday = ActusDateTime(2024, 1, 1, 0, 0, 0)
        friday = ActusDateTime(2024, 1, 5, 0, 0, 0)

        # Mon to Fri (inclusive): 5 business days
        assert cal.business_days_between(monday, friday, include_end=True) == 5

    def test_business_days_between_across_weekend(self):
        """Count business days across weekend."""
        cal = MondayToFridayCalendar()
        friday = ActusDateTime(2024, 1, 5, 0, 0, 0)
        monday = ActusDateTime(2024, 1, 8, 0, 0, 0)

        # Friday to Monday (exclusive): 0 business days
        assert cal.business_days_between(friday, monday) == 1  # Just Friday

    def test_business_days_between_reversed(self):
        """Count business days with reversed dates."""
        cal = MondayToFridayCalendar()
        monday = ActusDateTime(2024, 1, 1, 0, 0, 0)
        friday = ActusDateTime(2024, 1, 5, 0, 0, 0)

        # Reversed should return negative
        assert cal.business_days_between(friday, monday) == -4

    def test_business_days_between_same_date(self):
        """Count business days for same date."""
        cal = MondayToFridayCalendar()
        date = ActusDateTime(2024, 1, 1, 0, 0, 0)

        assert cal.business_days_between(date, date) == 0


class TestGetCalendar:
    """Test calendar factory function."""

    def test_get_no_calendar(self):
        """Get NoHolidayCalendar by name."""
        cal = get_calendar("NO_CALENDAR")
        assert isinstance(cal, NoHolidayCalendar)

    def test_get_monday_to_friday(self):
        """Get MondayToFridayCalendar by name."""
        cal = get_calendar("MONDAY_TO_FRIDAY")
        assert isinstance(cal, MondayToFridayCalendar)

    def test_get_calendar_case_insensitive(self):
        """Calendar names are case insensitive."""
        cal = get_calendar("monday_to_friday")
        assert isinstance(cal, MondayToFridayCalendar)

    def test_get_calendar_alias(self):
        """Calendar aliases work."""
        cal_none = get_calendar("NONE")
        assert isinstance(cal_none, NoHolidayCalendar)

        cal_mtf = get_calendar("MTF")
        assert isinstance(cal_mtf, MondayToFridayCalendar)

    def test_get_calendar_unknown(self):
        """Unknown calendar raises ValueError."""
        with pytest.raises(ValueError, match="Unknown calendar"):
            get_calendar("INVALID_CALENDAR")


class TestIsWeekend:
    """Test is_weekend helper function."""

    def test_saturday_is_weekend(self):
        """Saturday is a weekend."""
        saturday = ActusDateTime(2024, 1, 6, 0, 0, 0)
        assert is_weekend(saturday) is True

    def test_sunday_is_weekend(self):
        """Sunday is a weekend."""
        sunday = ActusDateTime(2024, 1, 7, 0, 0, 0)
        assert is_weekend(sunday) is True

    def test_monday_not_weekend(self):
        """Monday is not a weekend."""
        monday = ActusDateTime(2024, 1, 1, 0, 0, 0)
        assert is_weekend(monday) is False

    def test_friday_not_weekend(self):
        """Friday is not a weekend."""
        friday = ActusDateTime(2024, 1, 5, 0, 0, 0)
        assert is_weekend(friday) is False


class TestEdgeCases:
    """Test edge cases for calendar operations."""

    def test_preserves_time_components(self):
        """Navigation preserves hour/minute/second."""
        cal = MondayToFridayCalendar()
        saturday = ActusDateTime(2024, 1, 6, 14, 30, 45)

        next_day = cal.next_business_day(saturday)
        assert next_day.hour == 14
        assert next_day.minute == 30
        assert next_day.second == 45

    def test_add_many_business_days(self):
        """Can add many business days."""
        cal = MondayToFridayCalendar()
        start = ActusDateTime(2024, 1, 1, 0, 0, 0)  # Monday

        # Add 20 business days (4 weeks)
        result = cal.add_business_days(start, 20)
        expected = ActusDateTime(2024, 1, 29, 0, 0, 0)  # Monday, 4 weeks later
        assert result == expected

    def test_year_boundary(self):
        """Handle year boundaries correctly."""
        cal = MondayToFridayCalendar()
        dec_friday = ActusDateTime(2023, 12, 29, 0, 0, 0)  # Friday

        # Add 1 business day crosses year
        result = cal.add_business_days(dec_friday, 1)
        assert result == ActusDateTime(2024, 1, 1, 0, 0, 0)  # Monday

    def test_month_boundary(self):
        """Handle month boundaries correctly."""
        cal = MondayToFridayCalendar()
        jan_friday = ActusDateTime(2024, 1, 31, 0, 0, 0)  # Wednesday actually

        # Should work across month boundary
        result = cal.next_business_day(jan_friday)
        assert result.month == 1  # Still Jan (Wed is business day)

    def test_custom_calendar_multiple_years(self):
        """Custom calendar handles holidays across years."""
        holidays = [
            ActusDateTime(2024, 12, 25, 0, 0, 0),
            ActusDateTime(2025, 1, 1, 0, 0, 0),
        ]
        cal = CustomCalendar(holidays=holidays)

        assert cal.is_business_day(holidays[0]) is False
        assert cal.is_business_day(holidays[1]) is False
