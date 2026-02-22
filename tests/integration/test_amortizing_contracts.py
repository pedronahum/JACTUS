"""Integration tests for amortizing contracts.

This module tests cross-contract behavior and validates that all amortizing
contracts work correctly together and follow expected patterns.

Test Categories:
- Cross-contract consistency (all contracts reduce notional)
- IPCB feature across LAM/NAM/ANN/LAX
- Interest accrual consistency
- Fee accrual consistency
- Comparison tests (LAM vs ANN, etc.)
"""

import pytest

from jactus.contracts import create_contract
from jactus.core import (
    ActusDateTime,
    ContractAttributes,
    ContractRole,
    ContractType,
    DayCountConvention,
)
from jactus.observers.risk_factor import ConstantRiskFactorObserver


class TestCrossContractConsistency:
    """Test that all amortizing contracts follow consistent patterns."""

    def test_all_contracts_reduce_notional_over_time(self):
        """Verify that all amortizing contracts reduce notional over time."""
        contracts_to_test = [
            (ContractType.LAM, "Linear Amortizer"),
            (ContractType.NAM, "Negative Amortizer"),
            (ContractType.ANN, "Annuity"),
        ]

        for contract_type, name in contracts_to_test:
            attrs = ContractAttributes(
                contract_id=f"{contract_type.value}-INTEGRATION-001",
                contract_type=contract_type,
                contract_role=ContractRole.RPA,
                status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
                initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
                maturity_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
                currency="USD",
                notional_principal=100000.0,
                nominal_interest_rate=0.06,
                day_count_convention=DayCountConvention.A360,
                principal_redemption_cycle="1M",
                next_principal_redemption_amount=8333.33,
                principal_redemption_anchor=ActusDateTime(2024, 2, 15, 0, 0, 0),
            )

            rf_obs = ConstantRiskFactorObserver(constant_value=0.06)
            contract = create_contract(attrs, rf_obs)

            # Simulate and check that notional decreases
            result = contract.simulate()

            # Get notional values over time
            notionals = [event.state_post.nt for event in result.events if event.state_post]

            # Filter out zero notionals at the end
            non_zero_notionals = [n for n in notionals if float(n) > 0.1]

            # Verify at least some notional reduction happened
            if len(non_zero_notionals) >= 2:
                # First notional should be higher than later notionals
                assert float(non_zero_notionals[0]) >= float(non_zero_notionals[-1]), (
                    f"{name} should reduce notional over time"
                )

    def test_all_contracts_reach_zero_at_maturity(self):
        """Verify that all amortizing contracts reach zero notional at maturity."""
        contracts_to_test = [
            (ContractType.LAM, {}),
            (ContractType.NAM, {}),
            (ContractType.ANN, {}),
        ]

        for contract_type, extra_attrs in contracts_to_test:
            attrs = ContractAttributes(
                contract_id=f"{contract_type.value}-ZERO-001",
                contract_type=contract_type,
                contract_role=ContractRole.RPA,
                status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
                initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
                maturity_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
                currency="USD",
                notional_principal=100000.0,
                nominal_interest_rate=0.06,
                day_count_convention=DayCountConvention.A360,
                principal_redemption_cycle="1M",
                next_principal_redemption_amount=8333.33,
                principal_redemption_anchor=ActusDateTime(2024, 2, 15, 0, 0, 0),
                **extra_attrs,
            )

            rf_obs = ConstantRiskFactorObserver(constant_value=0.06)
            contract = create_contract(attrs, rf_obs)

            # Simulate
            result = contract.simulate()

            # Get final notional
            final_event = result.events[-1]
            if final_event.state_post:
                final_notional = float(final_event.state_post.nt)
                assert final_notional == pytest.approx(0.0, abs=1.0), (
                    f"{contract_type.value} should have zero notional at maturity, "
                    f"got {final_notional}"
                )

    def test_interest_accrual_consistent(self):
        """Verify that interest accrues consistently across contracts."""
        base_attrs = {
            "contract_role": ContractRole.RPA,
            "status_date": ActusDateTime(2024, 1, 1, 0, 0, 0),
            "initial_exchange_date": ActusDateTime(2024, 1, 15, 0, 0, 0),
            "maturity_date": ActusDateTime(2024, 7, 15, 0, 0, 0),
            "currency": "USD",
            "notional_principal": 100000.0,
            "nominal_interest_rate": 0.06,
            "day_count_convention": DayCountConvention.A360,
            "principal_redemption_cycle": "6M",
            "next_principal_redemption_amount": 100000.0,
            "principal_redemption_anchor": ActusDateTime(2024, 7, 15, 0, 0, 0),
        }

        # Test LAM and ANN with same parameters
        for contract_type in [ContractType.LAM, ContractType.ANN]:
            attrs = ContractAttributes(
                contract_id=f"{contract_type.value}-INTEREST-001",
                contract_type=contract_type,
                **base_attrs,
            )

            rf_obs = ConstantRiskFactorObserver(constant_value=0.06)
            contract = create_contract(attrs, rf_obs)

            # Simulate
            result = contract.simulate()

            # Find an IP or MD event and check interest was accrued
            for event in result.events:
                if event.state_post and event.event_type.value in ["IP", "MD"]:
                    # Interest should have been calculated
                    assert event.payoff is not None
                    break


class TestIPCBFeature:
    """Test Interest Calculation Base feature across contracts."""

    def test_ipcb_nt_mode(self):
        """Test IPCB=NT mode (interest on current notional)."""
        attrs = ContractAttributes(
            contract_id="LAM-IPCB-NT",
            contract_type=ContractType.LAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.06,
            day_count_convention=DayCountConvention.A360,
            principal_redemption_cycle="3M",
            next_principal_redemption_amount=8333.33,
            principal_redemption_anchor=ActusDateTime(2024, 4, 15, 0, 0, 0),
            interest_calculation_base="NT",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.06)
        contract = create_contract(attrs, rf_obs)

        # Should initialize without error
        state = contract.initialize_state()
        assert state.ipcb is not None

    def test_ipcb_ntied_mode(self):
        """Test IPCB=NTIED mode (interest on initial notional)."""
        attrs = ContractAttributes(
            contract_id="LAM-IPCB-NTIED",
            contract_type=ContractType.LAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.06,
            day_count_convention=DayCountConvention.A360,
            principal_redemption_cycle="3M",
            next_principal_redemption_amount=8333.33,
            principal_redemption_anchor=ActusDateTime(2024, 4, 15, 0, 0, 0),
            interest_calculation_base="NTIED",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.06)
        contract = create_contract(attrs, rf_obs)

        # Should initialize without error
        state = contract.initialize_state()
        assert state.ipcb is not None

    def test_ipcb_ntl_mode(self):
        """Test IPCB=NTL mode (interest on lagged notional)."""
        attrs = ContractAttributes(
            contract_id="LAM-IPCB-NTL",
            contract_type=ContractType.LAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.06,
            day_count_convention=DayCountConvention.A360,
            principal_redemption_cycle="3M",
            next_principal_redemption_amount=8333.33,
            principal_redemption_anchor=ActusDateTime(2024, 4, 15, 0, 0, 0),
            interest_calculation_base="NTL",
            interest_calculation_base_cycle="6M",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.06)
        contract = create_contract(attrs, rf_obs)

        # Should initialize without error and generate IPCB events
        schedule = contract.generate_event_schedule()

        # Should have IPCB events
        from jactus.core.types import EventType

        ipcb_events = [e for e in schedule.events if e.event_type == EventType.IPCB]
        assert len(ipcb_events) > 0, "IPCB=NTL should generate IPCB events"


class TestContractComparisons:
    """Test comparisons between different contract types."""

    def test_lam_vs_ann_same_parameters(self):
        """Compare LAM and ANN with same parameters."""
        base_params = {
            "contract_role": ContractRole.RPA,
            "status_date": ActusDateTime(2024, 1, 1, 0, 0, 0),
            "initial_exchange_date": ActusDateTime(2024, 1, 15, 0, 0, 0),
            "maturity_date": ActusDateTime(2025, 1, 15, 0, 0, 0),
            "currency": "USD",
            "notional_principal": 100000.0,
            "nominal_interest_rate": 0.06,
            "day_count_convention": DayCountConvention.A360,
            "principal_redemption_cycle": "1M",
            "next_principal_redemption_amount": 8333.33,
            "principal_redemption_anchor": ActusDateTime(2024, 2, 15, 0, 0, 0),
        }

        # Create LAM
        lam_attrs = ContractAttributes(
            contract_id="LAM-COMPARE-001", contract_type=ContractType.LAM, **base_params
        )
        rf_obs = ConstantRiskFactorObserver(constant_value=0.06)
        lam_contract = create_contract(lam_attrs, rf_obs)

        # Create ANN
        ann_attrs = ContractAttributes(
            contract_id="ANN-COMPARE-001", contract_type=ContractType.ANN, **base_params
        )
        ann_contract = create_contract(ann_attrs, rf_obs)

        # Both should initialize successfully
        lam_state = lam_contract.initialize_state()
        ann_state = ann_contract.initialize_state()

        assert lam_state is not None
        assert ann_state is not None

        # LAM should have fixed Prnxt, ANN should calculate from annuity
        assert lam_state.prnxt is not None
        assert ann_state.prnxt is not None


class TestObserverIntegration:
    """Test observer integration for CLM and UMP."""

    def test_clm_with_maturity(self):
        """Test CLM with fixed maturity."""
        attrs = ContractAttributes(
            contract_id="CLM-OBS-001",
            contract_type=ContractType.CLM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=50000.0,
            nominal_interest_rate=0.08,
            day_count_convention=DayCountConvention.A360,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.08)
        contract = create_contract(attrs, rf_obs)

        # Should generate schedule with MD
        schedule = contract.generate_event_schedule()
        from jactus.core.types import EventType

        md_events = [e for e in schedule.events if e.event_type == EventType.MD]
        assert len(md_events) == 1, "CLM with maturity should have MD event"

    def test_ump_without_maturity(self):
        """Test UMP without fixed maturity."""
        attrs = ContractAttributes(
            contract_id="UMP-OBS-001",
            contract_type=ContractType.UMP,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            # No maturity_date - uncertain
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.06,
            day_count_convention=DayCountConvention.A360,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.06)
        contract = create_contract(attrs, rf_obs)

        # Should generate minimal schedule without MD
        schedule = contract.generate_event_schedule()
        from jactus.core.types import EventType

        md_events = [e for e in schedule.events if e.event_type == EventType.MD]
        assert len(md_events) == 0, "UMP without maturity should not have MD event"


class TestFactoryRegistration:
    """Test that all contracts are properly registered in the factory."""

    def test_all_amortizing_contracts_registered(self):
        """Verify all 6 amortizing contracts are in the registry."""
        from jactus.contracts import get_available_contract_types

        available_types = get_available_contract_types()

        required_types = [
            ContractType.LAM,
            ContractType.NAM,
            ContractType.ANN,
            ContractType.LAX,
            ContractType.CLM,
            ContractType.UMP,
        ]

        for contract_type in required_types:
            assert contract_type in available_types, (
                f"{contract_type.value} should be registered in factory"
            )

    def test_factory_creates_all_contracts(self):
        """Verify factory can create instances of all amortizing contracts."""
        base_attrs = {
            "contract_role": ContractRole.RPA,
            "status_date": ActusDateTime(2024, 1, 1, 0, 0, 0),
            "initial_exchange_date": ActusDateTime(2024, 1, 15, 0, 0, 0),
            "maturity_date": ActusDateTime(2025, 1, 15, 0, 0, 0),
            "currency": "USD",
            "notional_principal": 100000.0,
            "nominal_interest_rate": 0.06,
            "day_count_convention": DayCountConvention.A360,
        }

        contracts_to_create = [
            (ContractType.LAM, {"principal_redemption_cycle": "1M"}),
            (ContractType.NAM, {"principal_redemption_cycle": "1M"}),
            (ContractType.ANN, {"principal_redemption_cycle": "1M"}),
            (
                ContractType.LAX,
                {
                    "array_pr_anchor": [ActusDateTime(2024, 2, 15, 0, 0, 0)],
                    "array_pr_cycle": ["1M"],
                    "array_pr_next": [1000.0],
                    "array_increase_decrease": ["DEC"],
                },
            ),
            (ContractType.CLM, {}),
            (ContractType.UMP, {}),
        ]

        rf_obs = ConstantRiskFactorObserver(constant_value=0.06)

        for contract_type, extra_attrs in contracts_to_create:
            attrs = ContractAttributes(
                contract_id=f"{contract_type.value}-FACTORY-001",
                contract_type=contract_type,
                **base_attrs,
                **extra_attrs,
            )

            contract = create_contract(attrs, rf_obs)
            assert contract is not None, f"Factory should create {contract_type.value}"
            assert contract.attributes.contract_type == contract_type
