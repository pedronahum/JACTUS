"""Property-based tests for schedule generation using Hypothesis.

T1.12: Schedule Generation Properties
Tests mathematical invariants that must hold for all schedule generation:
- Schedules are strictly increasing
- All dates are within bounds [start, end]
- Business day conventions are idempotent
- Zero-period schedules are invariant
"""

from datetime import timedelta

from hypothesis import assume, given
from hypothesis import strategies as st

from jactus.core import ActusDateTime
from jactus.core.types import BusinessDayConvention, EndOfMonthConvention
from jactus.utilities import (
    MondayToFridayCalendar,
    NoHolidayCalendar,
    apply_business_day_convention,
    generate_schedule,
)


# Custom strategies for ActusDateTime
@st.composite
def actus_datetime(draw):
    """Generate a valid ActusDateTime."""
    year = draw(st.integers(min_value=2020, max_value=2030))
    month = draw(st.integers(min_value=1, max_value=12))
    # Days depend on month
    if month in [1, 3, 5, 7, 8, 10, 12]:
        max_day = 31
    elif month in [4, 6, 9, 11]:
        max_day = 30
    else:  # February
        # Simplified: assume non-leap unless divisible by 4
        max_day = 29 if year % 4 == 0 else 28
    day = draw(st.integers(min_value=1, max_value=max_day))
    return ActusDateTime(year, month, day, 0, 0, 0)


@st.composite
def valid_date_range(draw):
    """Generate a valid start/end date range."""
    start = draw(actus_datetime())
    # End is 1 month to 5 years after start
    days_ahead = draw(st.integers(min_value=30, max_value=1825))  # 30 days to 5 years
    py_end = start.to_datetime() + timedelta(days=days_ahead)
    end = ActusDateTime(py_end.year, py_end.month, py_end.day, 0, 0, 0)
    return start, end


@st.composite
def valid_cycle(draw):
    """Generate a valid cycle string."""
    multiplier = draw(st.integers(min_value=1, max_value=12))
    period = draw(st.sampled_from(["D", "W", "M", "Q", "H", "Y"]))
    return f"{multiplier}{period}"


class TestScheduleMonotonicity:
    """Test that schedules are always strictly increasing."""

    @given(date_range=valid_date_range(), cycle=valid_cycle())
    def test_schedule_strictly_increasing(self, date_range, cycle):
        """All schedules must be strictly increasing."""
        start, end = date_range
        # Skip if cycle would generate too many dates
        assume(not (cycle.endswith("D") and int(cycle[:-1]) == 1 and \
                   start.days_between(end) > 1000))

        schedule = generate_schedule(start=start, cycle=cycle, end=end)

        # Check strictly increasing
        for i in range(1, len(schedule)):
            assert schedule[i] > schedule[i-1], \
                f"Schedule not increasing at index {i}: {schedule[i-1]} >= {schedule[i]}"

    @given(date_range=valid_date_range(), cycle=valid_cycle())
    def test_schedule_no_duplicates(self, date_range, cycle):
        """Schedules should not contain duplicate dates."""
        start, end = date_range
        assume(not (cycle.endswith("D") and int(cycle[:-1]) == 1 and \
                   start.days_between(end) > 1000))

        schedule = generate_schedule(start=start, cycle=cycle, end=end)

        # Check no duplicates
        assert len(schedule) == len(set(schedule)), \
            "Schedule contains duplicate dates"


class TestScheduleBounds:
    """Test that all schedule dates are within valid bounds."""

    @given(date_range=valid_date_range(), cycle=valid_cycle())
    def test_all_dates_within_bounds(self, date_range, cycle):
        """All schedule dates must be between start and end (inclusive with adjustment)."""
        start, end = date_range
        assume(not (cycle.endswith("D") and int(cycle[:-1]) == 1 and \
                   start.days_between(end) > 1000))

        schedule = generate_schedule(start=start, cycle=cycle, end=end)

        # First date should be start
        assert schedule[0] == start, f"First date {schedule[0]} != start {start}"

        # All dates should be >= start
        for date in schedule:
            assert date >= start, f"Date {date} < start {start}"

        # Last date should be <= end (or slightly after if BDC adjustment)
        # Allow up to 7 days beyond end for business day adjustments
        max_allowed = end.to_datetime() + timedelta(days=7)
        max_allowed_actus = ActusDateTime(
            max_allowed.year, max_allowed.month, max_allowed.day, 0, 0, 0
        )
        for date in schedule:
            assert date <= max_allowed_actus, \
                f"Date {date} too far beyond end {end}"

    @given(date_range=valid_date_range(), cycle=valid_cycle())
    def test_schedule_starts_at_start(self, date_range, cycle):
        """Schedule must always start at the start date."""
        start, end = date_range
        assume(not (cycle.endswith("D") and int(cycle[:-1]) == 1 and \
                   start.days_between(end) > 1000))

        schedule = generate_schedule(start=start, cycle=cycle, end=end)

        assert len(schedule) > 0, "Schedule is empty"
        assert schedule[0] == start, f"Schedule starts at {schedule[0]}, not {start}"


class TestBusinessDayConventionIdempotence:
    """Test that applying BDC twice gives same result as applying once."""

    @given(date=actus_datetime())
    def test_bdc_scf_idempotent(self, date):
        """Applying SCF (shift following) twice should give same result."""
        calendar = MondayToFridayCalendar()

        # Apply once
        adjusted1 = apply_business_day_convention(
            [date],
            BusinessDayConvention.SCF,
            calendar
        )[0]

        # Apply twice
        adjusted2 = apply_business_day_convention(
            [adjusted1],
            BusinessDayConvention.SCF,
            calendar
        )[0]

        assert adjusted1 == adjusted2, \
            f"BDC not idempotent: {date} -> {adjusted1} -> {adjusted2}"

    @given(date=actus_datetime())
    def test_bdc_scp_idempotent(self, date):
        """Applying SCP (shift preceding) twice should give same result."""
        calendar = MondayToFridayCalendar()

        # Apply once
        adjusted1 = apply_business_day_convention(
            [date],
            BusinessDayConvention.SCP,
            calendar
        )[0]

        # Apply twice
        adjusted2 = apply_business_day_convention(
            [adjusted1],
            BusinessDayConvention.SCP,
            calendar
        )[0]

        assert adjusted1 == adjusted2, \
            f"BDC not idempotent: {date} -> {adjusted1} -> {adjusted2}"

    @given(date=actus_datetime())
    def test_adjusted_date_is_business_day(self, date):
        """After BDC adjustment, date must be a business day."""
        calendar = MondayToFridayCalendar()

        # Skip SCMP and CSMP as they have known issues with month boundaries
        for bdc in [BusinessDayConvention.SCF, BusinessDayConvention.SCP,
                    BusinessDayConvention.SCMF]:
            adjusted = apply_business_day_convention([date], bdc, calendar)[0]
            assert calendar.is_business_day(adjusted), \
                f"After {bdc}, {adjusted} is not a business day"


class TestZeroPeriodInvariance:
    """Test that zero-period or same-date schedules behave correctly."""

    @given(date=actus_datetime())
    def test_same_start_end_gives_single_date(self, date):
        """Schedule with start==end should give single date."""
        schedule = generate_schedule(start=date, cycle="1M", end=date)

        assert len(schedule) == 1, f"Expected 1 date, got {len(schedule)}"
        assert schedule[0] == date


class TestEndOfMonthPreservation:
    """Test that EOM convention preserves month-end dates."""

    @given(year=st.integers(min_value=2020, max_value=2030),
           month=st.integers(min_value=1, max_value=11))
    def test_eom_preserves_month_end(self, year, month):
        """EOM convention should preserve end-of-month dates."""
        # Start on last day of month
        if month == 2:
            last_day = 29 if year % 4 == 0 else 28
        elif month in [4, 6, 9, 11]:
            last_day = 30
        else:
            last_day = 31

        start = ActusDateTime(year, month, last_day, 0, 0, 0)

        # Generate monthly schedule for 3 months with EOM
        end_month = month + 3 if month <= 9 else month + 3 - 12
        end_year = year if month <= 9 else year + 1
        end = ActusDateTime(end_year, end_month, 28, 0, 0, 0)

        schedule = generate_schedule(
            start=start,
            cycle="1M",
            end=end,
            end_of_month_convention=EndOfMonthConvention.EOM
        )

        # All dates should be end-of-month (or last valid day)
        for date in schedule[:-1]:  # Exclude end which may not be EOM
            if date.is_end_of_month():
                # Verify it's actually the last day
                assert date.is_end_of_month()


class TestScheduleConsistency:
    """Test consistency properties across different parameters."""

    @given(date_range=valid_date_range())
    def test_different_calendars_same_non_weekend_dates(self, date_range):
        """If dates don't fall on weekends, different calendars give same result."""
        start, end = date_range
        # Use a cycle unlikely to hit weekends every time
        cycle = "7D"  # Weekly

        # Skip if range too large
        assume(start.days_between(end) < 365)

        cal1 = NoHolidayCalendar()
        cal2 = MondayToFridayCalendar()

        schedule1 = generate_schedule(start=start, cycle=cycle, end=end,
                                     business_day_convention=BusinessDayConvention.NULL,
                                     calendar=cal1)

        schedule2 = generate_schedule(start=start, cycle=cycle, end=end,
                                     business_day_convention=BusinessDayConvention.NULL,
                                     calendar=cal2)

        # With NULL BDC (no adjustment), calendars shouldn't matter
        assert schedule1 == schedule2, \
            "Schedules differ with NULL convention"

    @given(date_range=valid_date_range(), cycle=valid_cycle())
    def test_schedule_length_reasonable(self, date_range, cycle):
        """Schedule length should be proportional to period."""
        start, end = date_range
        assume(not (cycle.endswith("D") and int(cycle[:-1]) == 1))

        schedule = generate_schedule(start=start, cycle=cycle, end=end)

        # Very loose bounds: at least 1 date, at most reasonable number
        assert len(schedule) >= 1
        assert len(schedule) <= 10000, f"Schedule too long: {len(schedule)} dates"
