"""Tests for schedule generation utilities.

Tests the implementation of schedule generation functions according to
ACTUS Technical Specification v1.1, Section 3.
"""

from __future__ import annotations

import pytest

from jactus.core.time import ActusDateTime
from jactus.core.types import BusinessDayConvention, Calendar, EndOfMonthConvention
from jactus.utilities.schedules import (
    apply_business_day_convention,
    apply_end_of_month_convention,
    expand_period_to_months,
    generate_array_schedule,
    generate_schedule,
)


class TestGenerateSchedule:
    """Test basic schedule generation S(s, c, T)."""

    def test_empty_schedule_no_start(self):
        """Empty schedule if start is None."""
        result = generate_schedule(start=None, cycle="3M", end=ActusDateTime(2025, 1, 1, 0, 0, 0))
        assert result == []

    def test_empty_schedule_no_end(self):
        """Empty schedule if end is None."""
        result = generate_schedule(start=ActusDateTime(2024, 1, 1, 0, 0, 0), cycle="3M", end=None)
        assert result == []

    def test_single_date_no_cycle(self):
        """Single date if cycle is None."""
        start = ActusDateTime(2024, 1, 15, 0, 0, 0)
        result = generate_schedule(start=start, cycle=None, end=ActusDateTime(2025, 1, 15, 0, 0, 0))
        assert result == [start]

    def test_single_date_empty_cycle(self):
        """Single date if cycle is empty string."""
        start = ActusDateTime(2024, 1, 15, 0, 0, 0)
        result = generate_schedule(start=start, cycle="", end=ActusDateTime(2025, 1, 15, 0, 0, 0))
        assert result == [start]

    def test_monthly_schedule(self):
        """Generate monthly schedule."""
        start = ActusDateTime(2024, 1, 15, 0, 0, 0)
        end = ActusDateTime(2024, 6, 15, 0, 0, 0)
        result = generate_schedule(start=start, cycle="1M", end=end)

        expected = [
            ActusDateTime(2024, 1, 15, 0, 0, 0),
            ActusDateTime(2024, 2, 15, 0, 0, 0),
            ActusDateTime(2024, 3, 15, 0, 0, 0),
            ActusDateTime(2024, 4, 15, 0, 0, 0),
            ActusDateTime(2024, 5, 15, 0, 0, 0),
            ActusDateTime(2024, 6, 15, 0, 0, 0),
        ]
        assert result == expected

    def test_quarterly_schedule(self):
        """Generate quarterly schedule."""
        start = ActusDateTime(2024, 1, 15, 0, 0, 0)
        end = ActusDateTime(2025, 1, 15, 0, 0, 0)
        result = generate_schedule(start=start, cycle="1Q", end=end)

        expected = [
            ActusDateTime(2024, 1, 15, 0, 0, 0),
            ActusDateTime(2024, 4, 15, 0, 0, 0),
            ActusDateTime(2024, 7, 15, 0, 0, 0),
            ActusDateTime(2024, 10, 15, 0, 0, 0),
            ActusDateTime(2025, 1, 15, 0, 0, 0),
        ]
        assert result == expected

    def test_semi_annual_schedule(self):
        """Generate semi-annual schedule (half-yearly)."""
        start = ActusDateTime(2024, 1, 15, 0, 0, 0)
        end = ActusDateTime(2026, 1, 15, 0, 0, 0)
        result = generate_schedule(start=start, cycle="1H", end=end)

        expected = [
            ActusDateTime(2024, 1, 15, 0, 0, 0),
            ActusDateTime(2024, 7, 15, 0, 0, 0),
            ActusDateTime(2025, 1, 15, 0, 0, 0),
            ActusDateTime(2025, 7, 15, 0, 0, 0),
            ActusDateTime(2026, 1, 15, 0, 0, 0),
        ]
        assert result == expected

    def test_annual_schedule(self):
        """Generate annual schedule."""
        start = ActusDateTime(2024, 1, 15, 0, 0, 0)
        end = ActusDateTime(2028, 1, 15, 0, 0, 0)
        result = generate_schedule(start=start, cycle="1Y", end=end)

        expected = [
            ActusDateTime(2024, 1, 15, 0, 0, 0),
            ActusDateTime(2025, 1, 15, 0, 0, 0),
            ActusDateTime(2026, 1, 15, 0, 0, 0),
            ActusDateTime(2027, 1, 15, 0, 0, 0),
            ActusDateTime(2028, 1, 15, 0, 0, 0),
        ]
        assert result == expected

    def test_daily_schedule(self):
        """Generate daily schedule."""
        start = ActusDateTime(2024, 1, 15, 0, 0, 0)
        end = ActusDateTime(2024, 1, 20, 0, 0, 0)
        result = generate_schedule(start=start, cycle="1D", end=end)

        expected = [
            ActusDateTime(2024, 1, 15, 0, 0, 0),
            ActusDateTime(2024, 1, 16, 0, 0, 0),
            ActusDateTime(2024, 1, 17, 0, 0, 0),
            ActusDateTime(2024, 1, 18, 0, 0, 0),
            ActusDateTime(2024, 1, 19, 0, 0, 0),
            ActusDateTime(2024, 1, 20, 0, 0, 0),
        ]
        assert result == expected

    def test_weekly_schedule(self):
        """Generate weekly schedule."""
        start = ActusDateTime(2024, 1, 1, 0, 0, 0)  # Monday
        end = ActusDateTime(2024, 1, 29, 0, 0, 0)
        result = generate_schedule(start=start, cycle="1W", end=end)

        expected = [
            ActusDateTime(2024, 1, 1, 0, 0, 0),
            ActusDateTime(2024, 1, 8, 0, 0, 0),
            ActusDateTime(2024, 1, 15, 0, 0, 0),
            ActusDateTime(2024, 1, 22, 0, 0, 0),
            ActusDateTime(2024, 1, 29, 0, 0, 0),
        ]
        assert result == expected

    def test_multi_month_cycle(self):
        """Generate schedule with multi-month cycle."""
        start = ActusDateTime(2024, 1, 15, 0, 0, 0)
        end = ActusDateTime(2025, 1, 15, 0, 0, 0)
        result = generate_schedule(start=start, cycle="3M", end=end)

        expected = [
            ActusDateTime(2024, 1, 15, 0, 0, 0),
            ActusDateTime(2024, 4, 15, 0, 0, 0),
            ActusDateTime(2024, 7, 15, 0, 0, 0),
            ActusDateTime(2024, 10, 15, 0, 0, 0),
            ActusDateTime(2025, 1, 15, 0, 0, 0),
        ]
        assert result == expected

    def test_schedule_beyond_end(self):
        """Schedule stops at end date."""
        start = ActusDateTime(2024, 1, 15, 0, 0, 0)
        end = ActusDateTime(2024, 5, 20, 0, 0, 0)  # Between cycle dates
        result = generate_schedule(start=start, cycle="1M", end=end)

        # Should include dates up to and including end
        assert ActusDateTime(2024, 1, 15, 0, 0, 0) in result
        assert ActusDateTime(2024, 2, 15, 0, 0, 0) in result
        assert ActusDateTime(2024, 3, 15, 0, 0, 0) in result
        assert ActusDateTime(2024, 4, 15, 0, 0, 0) in result
        assert ActusDateTime(2024, 5, 15, 0, 0, 0) in result
        # Should not include date after end
        assert ActusDateTime(2024, 6, 15, 0, 0, 0) not in result


class TestStubHandling:
    """Test stub period handling (+ and -)."""

    def test_short_stub(self):
        """Short stub removes last date beyond end."""
        start = ActusDateTime(2024, 1, 15, 0, 0, 0)
        end = ActusDateTime(2024, 5, 20, 0, 0, 0)  # Between cycle dates
        result = generate_schedule(start=start, cycle="1M-", end=end)

        # Last date 2024-05-15 is before end, so keep it
        # Next would be 2024-06-15 which is beyond end, already excluded
        assert ActusDateTime(2024, 5, 15, 0, 0, 0) in result

    def test_long_stub(self):
        """Long stub allows dates beyond end up to one cycle."""
        start = ActusDateTime(2024, 1, 15, 0, 0, 0)
        end = ActusDateTime(2024, 5, 20, 0, 0, 0)
        result = generate_schedule(start=start, cycle="1M+", end=end)

        # Should include dates up to end
        assert ActusDateTime(2024, 5, 15, 0, 0, 0) in result


class TestEndOfMonthConvention:
    """Test end-of-month convention handling."""

    def test_eom_convention_applied(self):
        """EOM convention moves dates to end of month."""
        # Start on Jan 31 (end of month)
        start = ActusDateTime(2024, 1, 31, 0, 0, 0)
        end = ActusDateTime(2024, 6, 30, 0, 0, 0)
        result = generate_schedule(
            start=start, cycle="1M", end=end, end_of_month_convention=EndOfMonthConvention.EOM
        )

        # All dates should be at end of their respective months
        assert ActusDateTime(2024, 1, 31, 0, 0, 0) in result  # Jan 31
        assert ActusDateTime(2024, 2, 29, 0, 0, 0) in result  # Feb 29 (leap year)
        assert ActusDateTime(2024, 3, 31, 0, 0, 0) in result  # Mar 31
        assert ActusDateTime(2024, 4, 30, 0, 0, 0) in result  # Apr 30
        assert ActusDateTime(2024, 5, 31, 0, 0, 0) in result  # May 31
        assert ActusDateTime(2024, 6, 30, 0, 0, 0) in result  # Jun 30

    def test_eom_only_applies_to_monthly_cycles(self):
        """EOM convention only applies to monthly-based cycles."""
        start = ActusDateTime(2024, 1, 15, 0, 0, 0)  # Not EOM
        end = ActusDateTime(2024, 2, 15, 0, 0, 0)
        # Daily cycle - EOM should not apply even if we were at EOM
        result = generate_schedule(
            start=start, cycle="7D", end=end, end_of_month_convention=EndOfMonthConvention.EOM
        )

        # Should be 7-day intervals, not moved to month end
        assert ActusDateTime(2024, 1, 15, 0, 0, 0) in result
        assert ActusDateTime(2024, 1, 22, 0, 0, 0) in result
        assert ActusDateTime(2024, 1, 29, 0, 0, 0) in result
        assert ActusDateTime(2024, 2, 5, 0, 0, 0) in result
        assert ActusDateTime(2024, 2, 12, 0, 0, 0) in result

    def test_sd_convention_no_adjustment(self):
        """SD (same day) convention makes no adjustments."""
        start = ActusDateTime(2024, 1, 31, 0, 0, 0)
        end = ActusDateTime(2024, 4, 30, 0, 0, 0)
        result = generate_schedule(
            start=start, cycle="1M", end=end, end_of_month_convention=EndOfMonthConvention.SD
        )

        # Feb doesn't have 31 days, so it adjusts to Feb 29 naturally
        # But this is from date arithmetic, not EOM convention
        assert ActusDateTime(2024, 1, 31, 0, 0, 0) in result


class TestBusinessDayConvention:
    """Test business day convention adjustments."""

    def test_null_convention_no_adjustment(self):
        """NULL convention makes no adjustments."""
        dates = [
            ActusDateTime(2024, 1, 6, 0, 0, 0),  # Saturday
            ActusDateTime(2024, 1, 7, 0, 0, 0),  # Sunday
        ]
        result = apply_business_day_convention(
            dates, BusinessDayConvention.NULL, Calendar.NO_CALENDAR
        )
        assert result == dates

    def test_following_convention(self):
        """SCF (Following) moves to next business day."""
        # Create schedule with weekend dates
        dates = [
            ActusDateTime(2024, 1, 5, 0, 0, 0),  # Friday
            ActusDateTime(2024, 1, 6, 0, 0, 0),  # Saturday -> Monday
            ActusDateTime(2024, 1, 7, 0, 0, 0),  # Sunday -> Monday
        ]
        result = apply_business_day_convention(
            dates, BusinessDayConvention.SCF, Calendar.MONDAY_TO_FRIDAY
        )

        # Friday stays Friday (already business day)
        assert ActusDateTime(2024, 1, 5, 0, 0, 0) in result
        # Saturday moves to Monday Jan 8
        assert ActusDateTime(2024, 1, 8, 0, 0, 0) in result
        # Note: result should contain adjusted dates

    def test_preceding_convention(self):
        """SCP (Preceding) moves to previous business day."""
        dates = [
            ActusDateTime(2024, 1, 6, 0, 0, 0),  # Saturday -> Friday
            ActusDateTime(2024, 1, 7, 0, 0, 0),  # Sunday -> Friday
            ActusDateTime(2024, 1, 8, 0, 0, 0),  # Monday
        ]
        result = apply_business_day_convention(
            dates, BusinessDayConvention.SCP, Calendar.MONDAY_TO_FRIDAY
        )

        # Saturday moves to Friday Jan 5
        assert ActusDateTime(2024, 1, 5, 0, 0, 0) in result
        # Monday stays Monday (already business day)
        assert ActusDateTime(2024, 1, 8, 0, 0, 0) in result

    def test_modified_following_convention(self):
        """SCMF (Modified Following) moves forward unless crosses month."""
        # Saturday Feb 3, 2024
        dates = [
            ActusDateTime(2024, 2, 3, 0, 0, 0),  # Saturday
        ]
        result = apply_business_day_convention(
            dates, BusinessDayConvention.SCMF, Calendar.MONDAY_TO_FRIDAY
        )

        # Should move forward to Monday Feb 5 (same month, so forward is OK)
        assert ActusDateTime(2024, 2, 5, 0, 0, 0) in result


class TestArraySchedule:
    """Test array schedule generation S~(~s, ~c, T)."""

    def test_array_schedule_multiple_anchors(self):
        """Generate schedule from multiple anchors and cycles."""
        anchors = [
            ActusDateTime(2024, 1, 15, 0, 0, 0),
            ActusDateTime(2024, 7, 15, 0, 0, 0),
        ]
        cycles = ["1M", "3M"]
        end = ActusDateTime(2025, 1, 15, 0, 0, 0)

        result = generate_array_schedule(anchors, cycles, end)

        # First sub-schedule: 2024-01-15 + 1M until 2024-07-15
        assert ActusDateTime(2024, 1, 15, 0, 0, 0) in result
        assert ActusDateTime(2024, 2, 15, 0, 0, 0) in result
        assert ActusDateTime(2024, 3, 15, 0, 0, 0) in result
        assert ActusDateTime(2024, 6, 15, 0, 0, 0) in result

        # Second sub-schedule: 2024-07-15 + 3M until 2025-01-15
        assert ActusDateTime(2024, 7, 15, 0, 0, 0) in result
        assert ActusDateTime(2024, 10, 15, 0, 0, 0) in result
        assert ActusDateTime(2025, 1, 15, 0, 0, 0) in result

    def test_array_schedule_empty(self):
        """Empty array schedule returns empty list."""
        result = generate_array_schedule([], [], ActusDateTime(2025, 1, 1, 0, 0, 0))
        assert result == []

    def test_array_schedule_mismatched_lengths(self):
        """Raise error if anchors and cycles have different lengths."""
        anchors = [ActusDateTime(2024, 1, 15, 0, 0, 0)]
        cycles = ["1M", "3M"]  # Mismatch
        end = ActusDateTime(2025, 1, 15, 0, 0, 0)

        with pytest.raises(ValueError, match="same length"):
            generate_array_schedule(anchors, cycles, end)

    def test_array_schedule_no_duplicates(self):
        """Array schedule removes duplicate dates."""
        # Anchors that might create overlapping dates
        anchors = [
            ActusDateTime(2024, 1, 15, 0, 0, 0),
            ActusDateTime(2024, 4, 15, 0, 0, 0),
        ]
        cycles = ["3M", "3M"]
        end = ActusDateTime(2024, 10, 15, 0, 0, 0)

        result = generate_array_schedule(anchors, cycles, end)

        # Should not have duplicates
        assert len(result) == len(set(result))


class TestHelperFunctions:
    """Test helper utility functions."""

    def test_expand_period_to_months(self):
        """Convert period to months."""
        assert expand_period_to_months("M", 1) == 1
        assert expand_period_to_months("M", 3) == 3
        assert expand_period_to_months("Q", 1) == 3
        assert expand_period_to_months("Q", 2) == 6
        assert expand_period_to_months("H", 1) == 6
        assert expand_period_to_months("Y", 1) == 12
        assert expand_period_to_months("Y", 2) == 24

    def test_expand_period_non_monthly(self):
        """Non-monthly periods return None."""
        assert expand_period_to_months("D", 7) is None
        assert expand_period_to_months("W", 2) is None


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_leap_year_handling(self):
        """Handle leap year correctly."""
        start = ActusDateTime(2024, 2, 29, 0, 0, 0)  # Leap day
        end = ActusDateTime(2024, 8, 29, 0, 0, 0)
        result = generate_schedule(start=start, cycle="1M", end=end)

        assert ActusDateTime(2024, 2, 29, 0, 0, 0) in result
        # March through August
        assert ActusDateTime(2024, 3, 29, 0, 0, 0) in result
        assert ActusDateTime(2024, 8, 29, 0, 0, 0) in result

    def test_month_boundary_31st(self):
        """Handle 31st day of month correctly."""
        start = ActusDateTime(2024, 1, 31, 0, 0, 0)
        end = ActusDateTime(2024, 5, 31, 0, 0, 0)
        result = generate_schedule(start=start, cycle="1M", end=end)

        # Should handle months without 31 days
        assert ActusDateTime(2024, 1, 31, 0, 0, 0) in result
        # Feb only has 29 days in 2024, so Jan 31 + 1M = Feb 29
        assert ActusDateTime(2024, 2, 29, 0, 0, 0) in result
        # Anchor-based: Jan 31 + 2M = Mar 31 (computed from anchor, not chained from Feb 29)
        assert ActusDateTime(2024, 3, 31, 0, 0, 0) in result

    def test_year_boundary(self):
        """Handle year boundaries correctly."""
        start = ActusDateTime(2024, 11, 15, 0, 0, 0)
        end = ActusDateTime(2025, 2, 15, 0, 0, 0)
        result = generate_schedule(start=start, cycle="1M", end=end)

        assert ActusDateTime(2024, 11, 15, 0, 0, 0) in result
        assert ActusDateTime(2024, 12, 15, 0, 0, 0) in result
        assert ActusDateTime(2025, 1, 15, 0, 0, 0) in result
        assert ActusDateTime(2025, 2, 15, 0, 0, 0) in result

    def test_same_start_and_end(self):
        """Handle start equal to end."""
        start = ActusDateTime(2024, 1, 15, 0, 0, 0)
        result = generate_schedule(start=start, cycle="1M", end=start)

        assert result == [start]

    def test_end_before_start(self):
        """Handle end before start (should return just start)."""
        start = ActusDateTime(2024, 6, 15, 0, 0, 0)
        end = ActusDateTime(2024, 1, 15, 0, 0, 0)
        result = generate_schedule(start=start, cycle="1M", end=end)

        # While loop condition current <= end is false immediately
        assert result == []

    def test_preserves_time_components(self):
        """Schedule preserves hour/minute/second from start."""
        start = ActusDateTime(2024, 1, 15, 14, 30, 45)
        end = ActusDateTime(2024, 3, 15, 14, 30, 45)
        result = generate_schedule(start=start, cycle="1M", end=end)

        # All dates should preserve time
        for date in result:
            assert date.hour == 14
            assert date.minute == 30
            assert date.second == 45


class TestApplyEndOfMonthConvention:
    """Test apply_end_of_month_convention function directly."""

    def test_apply_eom_non_eom_start(self):
        """No adjustment if start is not end of month."""
        dates = [
            ActusDateTime(2024, 1, 15, 0, 0, 0),
            ActusDateTime(2024, 2, 15, 0, 0, 0),
        ]
        start = ActusDateTime(2024, 1, 15, 0, 0, 0)
        result = apply_end_of_month_convention(dates, start, "1M", EndOfMonthConvention.EOM)

        # No adjustment because start is not EOM
        assert result == dates

    def test_apply_eom_daily_cycle(self):
        """No adjustment for non-monthly cycles."""
        dates = [
            ActusDateTime(2024, 1, 31, 0, 0, 0),
            ActusDateTime(2024, 2, 1, 0, 0, 0),
        ]
        start = ActusDateTime(2024, 1, 31, 0, 0, 0)
        result = apply_end_of_month_convention(dates, start, "1D", EndOfMonthConvention.EOM)

        # No adjustment for daily cycle
        assert result == dates

    def test_apply_eom_sd_convention(self):
        """No adjustment for SD convention."""
        dates = [
            ActusDateTime(2024, 1, 31, 0, 0, 0),
            ActusDateTime(2024, 2, 29, 0, 0, 0),
        ]
        start = ActusDateTime(2024, 1, 31, 0, 0, 0)
        result = apply_end_of_month_convention(dates, start, "1M", EndOfMonthConvention.SD)

        # SD means same day - no adjustment
        assert result == dates
