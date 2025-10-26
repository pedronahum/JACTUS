"""Unit tests for contract lifecycle management.

T2.6: Contract Lifetime Tests

Tests for:
- calculate_contract_end()
- is_contract_active()
- get_contract_phase()
- filter_events_by_lifecycle()
- Edge cases and boundary conditions
"""

import jax.numpy as jnp

from jactus.core import ActusDateTime, ContractAttributes, ContractEvent
from jactus.core.types import ContractRole, ContractType, EventType
from jactus.engine.lifecycle import (
    DEFAULT_T_MAX_YEARS,
    ContractPhase,
    calculate_contract_end,
    filter_events_by_lifecycle,
    get_contract_phase,
    is_contract_active,
)


class TestCalculateContractEnd:
    """Test calculate_contract_end function."""

    def test_termination_date_has_highest_priority(self):
        """Test that termination date takes precedence over all others."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2034, 1, 1, 0, 0, 0),
            termination_date=ActusDateTime(2025, 6, 1, 0, 0, 0),
        )

        end_date = calculate_contract_end(attrs)
        assert end_date == ActusDateTime(2025, 6, 1, 0, 0, 0)

    def test_maturity_date_when_no_termination(self):
        """Test that maturity date is used when no termination date."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2034, 1, 1, 0, 0, 0),
        )

        end_date = calculate_contract_end(attrs)
        assert end_date == ActusDateTime(2034, 1, 1, 0, 0, 0)

    def test_fallback_to_status_date_plus_tmax(self):
        """Test fallback to status_date + t_max for perpetual contracts."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        )

        end_date = calculate_contract_end(attrs, t_max_years=50)
        expected = ActusDateTime(2024, 1, 1, 0, 0, 0).add_period("50Y")
        assert end_date == expected

    def test_uses_purchase_date_if_available(self):
        """Test that purchase date is used instead of status date when available."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            purchase_date=ActusDateTime(2024, 6, 1, 0, 0, 0),
        )

        end_date = calculate_contract_end(attrs, t_max_years=10)
        expected = ActusDateTime(2024, 6, 1, 0, 0, 0).add_period("10Y")
        assert end_date == expected

    def test_default_tmax_is_100_years(self):
        """Test that default t_max is 100 years."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        )

        end_date = calculate_contract_end(attrs)
        expected = ActusDateTime(2024, 1, 1, 0, 0, 0).add_period(f"{DEFAULT_T_MAX_YEARS}Y")
        assert end_date == expected


class TestIsContractActive:
    """Test is_contract_active function."""

    def test_active_during_contract_period(self):
        """Test that contract is active between IED and maturity."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2034, 1, 1, 0, 0, 0),
        )

        # During active period
        assert is_contract_active(attrs, ActusDateTime(2025, 1, 1, 0, 0, 0)) is True

    def test_not_active_before_ied(self):
        """Test that contract is not active before IED."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2034, 1, 1, 0, 0, 0),
        )

        # Before IED
        assert is_contract_active(attrs, ActusDateTime(2024, 1, 10, 0, 0, 0)) is False

    def test_not_active_after_maturity(self):
        """Test that contract is not active after maturity."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2034, 1, 1, 0, 0, 0),
        )

        # After maturity
        assert is_contract_active(attrs, ActusDateTime(2035, 1, 1, 0, 0, 0)) is False

    def test_active_on_ied(self):
        """Test that contract is active on IED (inclusive)."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2034, 1, 1, 0, 0, 0),
        )

        # On IED
        assert is_contract_active(attrs, ActusDateTime(2024, 1, 15, 0, 0, 0)) is True

    def test_active_on_maturity_date(self):
        """Test that contract is active on maturity date (inclusive)."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2034, 1, 1, 0, 0, 0),
        )

        # On maturity date
        assert is_contract_active(attrs, ActusDateTime(2034, 1, 1, 0, 0, 0)) is True

    def test_uses_status_date_if_no_ied(self):
        """Test that status_date is used if IED is not set."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2034, 1, 1, 0, 0, 0),
        )

        # At status date (should be active)
        assert is_contract_active(attrs, ActusDateTime(2024, 1, 1, 0, 0, 0)) is True

        # Before status date (should not be active)
        assert is_contract_active(attrs, ActusDateTime(2023, 12, 1, 0, 0, 0)) is False


class TestGetContractPhase:
    """Test get_contract_phase function."""

    def test_pre_inception_phase(self):
        """Test PRE_INCEPTION phase before IED."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2034, 1, 1, 0, 0, 0),
        )

        phase = get_contract_phase(attrs, ActusDateTime(2024, 1, 10, 0, 0, 0))
        assert phase == ContractPhase.PRE_INCEPTION

    def test_active_phase(self):
        """Test ACTIVE phase during contract period."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2034, 1, 1, 0, 0, 0),
        )

        phase = get_contract_phase(attrs, ActusDateTime(2025, 1, 1, 0, 0, 0))
        assert phase == ContractPhase.ACTIVE

    def test_matured_phase(self):
        """Test MATURED phase after maturity date."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2034, 1, 1, 0, 0, 0),
        )

        phase = get_contract_phase(attrs, ActusDateTime(2035, 1, 1, 0, 0, 0))
        assert phase == ContractPhase.MATURED

    def test_terminated_phase(self):
        """Test TERMINATED phase after termination date."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2034, 1, 1, 0, 0, 0),
            termination_date=ActusDateTime(2025, 6, 1, 0, 0, 0),
        )

        phase = get_contract_phase(attrs, ActusDateTime(2025, 7, 1, 0, 0, 0))
        assert phase == ContractPhase.TERMINATED

    def test_terminated_takes_precedence_over_matured(self):
        """Test that TERMINATED phase takes precedence over MATURED."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2034, 1, 1, 0, 0, 0),
            termination_date=ActusDateTime(2025, 6, 1, 0, 0, 0),
        )

        # After both termination and maturity
        phase = get_contract_phase(attrs, ActusDateTime(2035, 1, 1, 0, 0, 0))
        assert phase == ContractPhase.TERMINATED

    def test_perpetual_contract_matured_after_tmax(self):
        """Test that perpetual contract becomes matured after t_max."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        )

        # Way past t_max
        phase = get_contract_phase(attrs, ActusDateTime(2200, 1, 1, 0, 0, 0), t_max_years=10)
        assert phase == ContractPhase.MATURED


class TestFilterEventsByLifecycle:
    """Test filter_events_by_lifecycle function."""

    def test_filters_to_active_events_only(self):
        """Test filtering to only active period events."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
        )

        events = [
            ContractEvent(
                event_type=EventType.IED,
                event_time=ActusDateTime(2024, 1, 10, 0, 0, 0),  # Before IED
                payoff=jnp.array(0.0),
                currency="USD",
            ),
            ContractEvent(
                event_type=EventType.IP,
                event_time=ActusDateTime(2024, 6, 1, 0, 0, 0),  # Active
                payoff=jnp.array(100.0),
                currency="USD",
            ),
            ContractEvent(
                event_type=EventType.MD,
                event_time=ActusDateTime(2025, 1, 15, 0, 0, 0),  # After maturity
                payoff=jnp.array(0.0),
                currency="USD",
            ),
        ]

        filtered = filter_events_by_lifecycle(events, attrs)
        assert len(filtered) == 1
        assert filtered[0].event_time == ActusDateTime(2024, 6, 1, 0, 0, 0)

    def test_includes_pre_inception_when_requested(self):
        """Test including pre-inception events."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
        )

        events = [
            ContractEvent(
                event_type=EventType.IED,
                event_time=ActusDateTime(2024, 1, 10, 0, 0, 0),  # Before IED
                payoff=jnp.array(0.0),
                currency="USD",
            ),
            ContractEvent(
                event_type=EventType.IP,
                event_time=ActusDateTime(2024, 6, 1, 0, 0, 0),  # Active
                payoff=jnp.array(100.0),
                currency="USD",
            ),
        ]

        filtered = filter_events_by_lifecycle(events, attrs, include_pre_inception=True)
        assert len(filtered) == 2

    def test_includes_post_maturity_when_requested(self):
        """Test including post-maturity events."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
        )

        events = [
            ContractEvent(
                event_type=EventType.IP,
                event_time=ActusDateTime(2024, 6, 1, 0, 0, 0),  # Active
                payoff=jnp.array(100.0),
                currency="USD",
            ),
            ContractEvent(
                event_type=EventType.MD,
                event_time=ActusDateTime(2025, 1, 15, 0, 0, 0),  # After maturity
                payoff=jnp.array(0.0),
                currency="USD",
            ),
        ]

        filtered = filter_events_by_lifecycle(events, attrs, include_post_maturity=True)
        assert len(filtered) == 2

    def test_includes_terminated_events_when_post_maturity_requested(self):
        """Test that terminated events are included with post_maturity flag."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            termination_date=ActusDateTime(2024, 6, 15, 0, 0, 0),
        )

        events = [
            ContractEvent(
                event_type=EventType.IP,
                event_time=ActusDateTime(2024, 5, 1, 0, 0, 0),  # Active
                payoff=jnp.array(100.0),
                currency="USD",
            ),
            ContractEvent(
                event_type=EventType.TD,
                event_time=ActusDateTime(2024, 7, 1, 0, 0, 0),  # After termination
                payoff=jnp.array(0.0),
                currency="USD",
            ),
        ]

        filtered = filter_events_by_lifecycle(events, attrs, include_post_maturity=True)
        assert len(filtered) == 2

    def test_empty_list_when_no_events_match(self):
        """Test that empty list is returned when no events match."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
        )

        events = [
            ContractEvent(
                event_type=EventType.IED,
                event_time=ActusDateTime(2024, 1, 10, 0, 0, 0),  # Before IED
                payoff=jnp.array(0.0),
                currency="USD",
            ),
            ContractEvent(
                event_type=EventType.MD,
                event_time=ActusDateTime(2025, 1, 15, 0, 0, 0),  # After maturity
                payoff=jnp.array(0.0),
                currency="USD",
            ),
        ]

        filtered = filter_events_by_lifecycle(events, attrs)
        assert len(filtered) == 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_contract_with_only_status_date(self):
        """Test contract with only status_date (no IED, no MD)."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        )

        # Should be active at status date
        assert is_contract_active(attrs, ActusDateTime(2024, 1, 1, 0, 0, 0)) is True

        # Should get ACTIVE phase
        phase = get_contract_phase(attrs, ActusDateTime(2024, 6, 1, 0, 0, 0))
        assert phase == ContractPhase.ACTIVE

    def test_one_day_contract(self):
        """Test contract with very short duration (one day)."""
        ied = ActusDateTime(2024, 1, 15, 0, 0, 0)
        maturity = ActusDateTime(2024, 1, 16, 0, 0, 0)
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ied,
            maturity_date=maturity,
        )

        # Should be active on IED
        assert is_contract_active(attrs, ied) is True
        phase = get_contract_phase(attrs, ied)
        assert phase == ContractPhase.ACTIVE

        # Should be matured on maturity date
        assert is_contract_active(attrs, maturity) is True
        phase_maturity = get_contract_phase(attrs, maturity)
        assert phase_maturity == ContractPhase.MATURED

        # Should be matured day after
        next_day = maturity.add_period("1D")
        phase_next = get_contract_phase(attrs, next_day)
        assert phase_next == ContractPhase.MATURED

    def test_very_large_tmax(self):
        """Test with very large t_max value."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        )

        end_date = calculate_contract_end(attrs, t_max_years=1000)
        expected = ActusDateTime(2024, 1, 1, 0, 0, 0).add_period("1000Y")
        assert end_date == expected

    def test_filter_preserves_event_order(self):
        """Test that filtering preserves original event order."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
        )

        events = [
            ContractEvent(
                event_type=EventType.IP,
                event_time=ActusDateTime(2024, 3, 1, 0, 0, 0),
                payoff=jnp.array(100.0),
                currency="USD",
            ),
            ContractEvent(
                event_type=EventType.IP,
                event_time=ActusDateTime(2024, 6, 1, 0, 0, 0),
                payoff=jnp.array(100.0),
                currency="USD",
            ),
            ContractEvent(
                event_type=EventType.IP,
                event_time=ActusDateTime(2024, 9, 1, 0, 0, 0),
                payoff=jnp.array(100.0),
                currency="USD",
            ),
        ]

        filtered = filter_events_by_lifecycle(events, attrs)
        assert len(filtered) == 3
        # Check order preserved
        assert filtered[0].event_time == ActusDateTime(2024, 3, 1, 0, 0, 0)
        assert filtered[1].event_time == ActusDateTime(2024, 6, 1, 0, 0, 0)
        assert filtered[2].event_time == ActusDateTime(2024, 9, 1, 0, 0, 0)
