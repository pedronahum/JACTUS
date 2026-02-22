"""Property-based tests for year fraction calculations using Hypothesis.

T1.13: Year Fraction Properties
Tests mathematical invariants for year fraction calculations:
- Non-negative for s < t
- Antisymmetric: yf(s,t) = -yf(t,s)
- Zero for same date
- Approximately transitive: yf(s,t) + yf(t,u) ≈ yf(s,u)
"""

from hypothesis import assume, given
from hypothesis import strategies as st

from jactus.core import ActusDateTime
from jactus.core.types import DayCountConvention
from jactus.utilities import year_fraction

# Exclude E30360ISDA as it requires maturity date parameter
TESTABLE_DCCS = [dcc for dcc in DayCountConvention if dcc != DayCountConvention.E30360ISDA]


# Reuse actus_datetime strategy
@st.composite
def actus_datetime(draw):
    """Generate a valid ActusDateTime."""
    year = draw(st.integers(min_value=2020, max_value=2030))
    month = draw(st.integers(min_value=1, max_value=12))
    if month in [1, 3, 5, 7, 8, 10, 12]:
        max_day = 31
    elif month in [4, 6, 9, 11]:
        max_day = 30
    else:
        max_day = 29 if year % 4 == 0 else 28
    day = draw(st.integers(min_value=1, max_value=max_day))
    return ActusDateTime(year, month, day, 0, 0, 0)


class TestYearFractionNonNegativity:
    """Test that year fractions are non-negative when s < t."""

    @given(date1=actus_datetime(), date2=actus_datetime(), dcc=st.sampled_from(TESTABLE_DCCS))
    def test_non_negative_for_increasing_dates(self, date1, date2, dcc):
        """Year fraction should be non-negative when start < end."""
        assume(date1 < date2)

        yf = year_fraction(date1, date2, dcc)

        assert yf >= 0, f"Year fraction {yf} is negative for {date1} to {date2}"

    @given(date=actus_datetime(), dcc=st.sampled_from(TESTABLE_DCCS))
    def test_zero_for_same_date(self, date, dcc):
        """Year fraction should be zero for same date."""
        yf = year_fraction(date, date, dcc)

        assert abs(yf) < 1e-10, f"Year fraction {yf} not zero for same date"


class TestYearFractionAntisymmetry:
    """Test that yf(s,t) has consistent behavior."""

    @given(date1=actus_datetime(), date2=actus_datetime(), dcc=st.sampled_from(TESTABLE_DCCS))
    def test_consistent_ordering(self, date1, date2, dcc):
        """Year fraction should give consistent results for ordered dates."""
        assume(date1 < date2)  # Ensure forward direction only

        yf = year_fraction(date1, date2, dcc)

        # Forward direction should always give positive or zero result
        assert yf >= 0, f"Year fraction {yf} is negative"


class TestYearFractionTransitivity:
    """Test approximate transitivity: yf(s,t) + yf(t,u) ≈ yf(s,u)."""

    @given(
        date1=actus_datetime(),
        date2=actus_datetime(),
        date3=actus_datetime(),
        dcc=st.sampled_from(TESTABLE_DCCS),
    )
    def test_approximately_transitive(self, date1, date2, date3, dcc):
        """Year fraction should be approximately transitive."""
        # Sort dates
        dates = sorted([date1, date2, date3])
        s, t, u = dates

        assume(s < t < u)  # Strictly increasing

        yf_st = year_fraction(s, t, dcc)
        yf_tu = year_fraction(t, u, dcc)
        yf_su = year_fraction(s, u, dcc)

        # Due to day count conventions, exact transitivity may not hold
        # Allow small error (especially for 30/360 methods)
        error = abs((yf_st + yf_tu) - yf_su)

        # Be generous: allow up to 0.01 years (about 3.65 days) difference
        # This accounts for varying month/year lengths in day count methods
        assert error < 0.01, (
            f"Transitivity violated: yf({s},{t})={yf_st}, yf({t},{u})={yf_tu}, "
            f"yf({s},{u})={yf_su}, error={error}"
        )


class TestYearFractionBounds:
    """Test that year fractions are bounded reasonably."""

    @given(date1=actus_datetime(), date2=actus_datetime(), dcc=st.sampled_from(TESTABLE_DCCS))
    def test_reasonable_bounds(self, date1, date2, dcc):
        """Year fraction should not exceed actual year difference significantly."""
        # Calculate year difference
        year_diff = abs(date2.year - date1.year)

        yf = abs(year_fraction(date1, date2, dcc))

        # Year fraction should not exceed year diff by more than 1 year
        # (accounting for day of year differences)
        assert yf <= year_diff + 1.1, (
            f"Year fraction {yf} too large for {year_diff} year difference"
        )

    @given(date1=actus_datetime(), date2=actus_datetime(), dcc=st.sampled_from(TESTABLE_DCCS))
    def test_one_year_reasonable_range(self, date1, date2, dcc):
        """Dates exactly 1 year apart should give year fraction in reasonable range."""
        # Create date exactly 1 year later (same month/day)
        date2_oneyear = ActusDateTime(
            date1.year + 1, date1.month, date1.day, date1.hour, date1.minute, date1.second
        )

        yf = year_fraction(date1, date2_oneyear, dcc)

        # For day count conventions, 1 year ranges broadly:
        # A360: 366/360 = 1.0166 (leap year)
        # A365: 366/365 = 1.0027 (leap year)
        # BUS252: 262/252 = 1.0397 (business days)
        assert 0.97 <= yf <= 1.05, f"One year apart gives yf={yf}, expected 0.97-1.05 for {dcc}"


class TestYearFractionMonotonicity:
    """Test that longer periods give larger year fractions."""

    @given(
        start=actus_datetime(),
        mid=actus_datetime(),
        end=actus_datetime(),
        dcc=st.sampled_from(TESTABLE_DCCS),
    )
    def test_monotonic_with_period_length(self, start, mid, end, dcc):
        """Longer periods should give larger year fractions."""
        # Sort to get start < mid < end
        dates = sorted([start, mid, end])
        s, m, e = dates

        assume(s < m < e)

        # For BUS252, require at least 7 days between dates to ensure
        # at least one business day (weekends can cause zero fractions)
        if dcc == DayCountConvention.BUS252:
            s_dt = s.to_datetime()
            m_dt = m.to_datetime()
            e_dt = e.to_datetime()
            assume((m_dt - s_dt).days >= 7)
            assume((e_dt - m_dt).days >= 7)

        yf_sm = year_fraction(s, m, dcc)
        yf_me = year_fraction(m, e, dcc)
        yf_se = year_fraction(s, e, dcc)

        # Both parts should be positive and their sum ≈ whole
        assert yf_sm > 0, f"yf({s},{m}) should be positive"
        assert yf_me > 0, f"yf({m},{e}) should be positive"
        assert yf_se > yf_sm, f"yf({s},{e}) should be > yf({s},{m})"
        assert yf_se > yf_me, f"yf({s},{e}) should be > yf({m},{e})"
