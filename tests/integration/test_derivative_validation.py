"""Integration tests for derivative contract validation and cross-contract consistency.

These tests ensure that all derivative contracts follow consistent patterns,
handle errors properly, and integrate correctly with the factory system.
"""

import pytest

from jactus.contracts import create_contract, get_available_contract_types
from jactus.core import ContractAttributes, ContractRole, ContractType, ActusDateTime
from jactus.observers import ConstantRiskFactorObserver, MockChildContractObserver


# Test fixtures
@pytest.fixture
def status_date():
    """Standard status date."""
    return ActusDateTime(2024, 1, 1, 0, 0, 0)


@pytest.fixture
def maturity_date():
    """Standard maturity date."""
    return ActusDateTime(2029, 1, 1, 0, 0, 0)


@pytest.fixture
def rf_observer():
    """Risk factor observer."""
    return ConstantRiskFactorObserver(0.05)


# ==================== Test: Factory Registry ====================


class TestFactoryRegistry:
    """Test contract factory registry for all derivatives."""

    def test_all_derivatives_registered(self):
        """Test that all Phase 5 derivatives are registered."""
        available_types = get_available_contract_types()

        # Phase 5 derivatives
        phase5_types = [
            ContractType.FXOUT,
            ContractType.SWPPV,
            ContractType.SWAPS,
            ContractType.CAPFL,
            ContractType.OPTNS,
            ContractType.FUTUR,
            ContractType.CEG,
            ContractType.CEC,
        ]

        for contract_type in phase5_types:
            assert contract_type in available_types, f"{contract_type.value} not registered"

    def test_factory_creates_fxout(self, status_date, maturity_date, rf_observer):
        """Test factory creates FXOUT."""
        attrs = ContractAttributes(
            contract_id="FX-001",
            contract_type=ContractType.FXOUT,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            settlement_date=maturity_date,
            currency="USD",
            currency_2="EUR",
            notional_principal=100000.0,
            notional_principal_2=90000.0,
            delivery_settlement="D",  # Required for FXOUT
        )

        contract = create_contract(attrs, rf_observer)
        assert contract.attributes.contract_type == ContractType.FXOUT

    def test_factory_creates_optns(self, status_date, maturity_date, rf_observer):
        """Test factory creates OPTNS."""
        attrs = ContractAttributes(
            contract_id="OPT-001",
            contract_type=ContractType.OPTNS,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            option_type="C",
            option_strike_1=105000.0,
            option_exercise_type="E",
            currency="USD",
            contract_structure='{"Underlier": "UNDERLIER-001"}',  # Required for OPTNS
        )

        contract = create_contract(attrs, rf_observer)
        assert contract.attributes.contract_type == ContractType.OPTNS

    def test_factory_creates_futur(self, status_date, maturity_date, rf_observer):
        """Test factory creates FUTUR."""
        attrs = ContractAttributes(
            contract_id="FUT-001",
            contract_type=ContractType.FUTUR,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            future_price=105000.0,
            currency="USD",
            contract_structure='{"Underlier": "UNDERLIER-001"}',  # Required for FUTUR
        )

        contract = create_contract(attrs, rf_observer)
        assert contract.attributes.contract_type == ContractType.FUTUR

    def test_factory_creates_swppv(self, status_date, maturity_date, rf_observer):
        """Test factory creates SWPPV."""
        attrs = ContractAttributes(
            contract_id="SWAP-001",
            contract_type=ContractType.SWPPV,
            contract_role=ContractRole.RFL,
            status_date=status_date,
            maturity_date=maturity_date,
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            nominal_interest_rate_2=0.03,
            currency="USD",
            interest_payment_cycle="3M",
            rate_reset_cycle="3M",
        )

        contract = create_contract(attrs, rf_observer)
        assert contract.attributes.contract_type == ContractType.SWPPV

    def test_factory_requires_child_observer_for_composition(
        self, status_date, maturity_date, rf_observer
    ):
        """Test that composition contracts require child observer."""
        # SWAPS requires child observer
        attrs = ContractAttributes(
            contract_id="SWAP-001",
            contract_type=ContractType.SWAPS,
            contract_role=ContractRole.RFL,
            status_date=status_date,
            maturity_date=maturity_date,
            contract_structure='{"FirstLeg": "LEG1", "SecondLeg": "LEG2"}',
            currency="USD",
        )

        with pytest.raises(ValueError, match="child_contract_observer"):
            create_contract(attrs, rf_observer, None)


# ==================== Test: Contract Consistency ====================


class TestContractConsistency:
    """Test that all contracts follow consistent patterns."""

    def test_all_contracts_have_initialize_state(self, status_date, maturity_date, rf_observer):
        """Test that all contracts implement initialize_state."""
        test_contracts = [
            (
                ContractType.FXOUT,
                {
                    "currency": "USD",
                    "currency_2": "EUR",
                    "notional_principal": 100000.0,
                    "notional_principal_2": 90000.0,
                    "settlement_date": maturity_date,
                    "delivery_settlement": "D",
                },
            ),
            (
                ContractType.OPTNS,
                {
                    "option_type": "C",
                    "option_strike_1": 105000.0,
                    "option_exercise_type": "E",
                    "currency": "USD",
                    "contract_structure": '{"Underlier": "UNDERLIER-001"}',
                },
            ),
            (
                ContractType.FUTUR,
                {
                    "future_price": 105000.0,
                    "currency": "USD",
                    "contract_structure": '{"Underlier": "UNDERLIER-001"}',
                },
            ),
            (
                ContractType.SWPPV,
                {
                    "notional_principal": 100000.0,
                    "nominal_interest_rate": 0.05,
                    "nominal_interest_rate_2": 0.03,
                    "currency": "USD",
                    "interest_payment_cycle": "3M",
                    "rate_reset_cycle": "3M",
                },
            ),
        ]

        for contract_type, extra_attrs in test_contracts:
            attrs = ContractAttributes(
                contract_id=f"{contract_type.value}-001",
                contract_type=contract_type,
                contract_role=ContractRole.RPA,
                status_date=status_date,
                maturity_date=maturity_date,
                **extra_attrs,
            )

            contract = create_contract(attrs, rf_observer)
            state = contract.initialize_state()

            # All contracts should have basic state
            assert state is not None
            assert state.sd == status_date
            assert hasattr(state, "tmd")
            assert hasattr(state, "prf")

    def test_all_contracts_generate_schedule(self, status_date, maturity_date, rf_observer):
        """Test that all contracts generate event schedules."""
        test_contracts = [
            (
                ContractType.FXOUT,
                {
                    "currency": "USD",
                    "currency_2": "EUR",
                    "notional_principal": 100000.0,
                    "notional_principal_2": 90000.0,
                    "settlement_date": maturity_date,
                    "delivery_settlement": "D",
                },
            ),
            (
                ContractType.OPTNS,
                {
                    "option_type": "C",
                    "option_strike_1": 105000.0,
                    "option_exercise_type": "E",
                    "currency": "USD",
                    "contract_structure": '{"Underlier": "UNDERLIER-001"}',
                },
            ),
            (
                ContractType.FUTUR,
                {
                    "future_price": 105000.0,
                    "currency": "USD",
                    "contract_structure": '{"Underlier": "UNDERLIER-001"}',
                },
            ),
            (
                ContractType.SWPPV,
                {
                    "notional_principal": 100000.0,
                    "nominal_interest_rate": 0.05,
                    "nominal_interest_rate_2": 0.03,
                    "currency": "USD",
                    "interest_payment_cycle": "3M",
                    "rate_reset_cycle": "3M",
                },
            ),
        ]

        for contract_type, extra_attrs in test_contracts:
            attrs = ContractAttributes(
                contract_id=f"{contract_type.value}-001",
                contract_type=contract_type,
                contract_role=ContractRole.RPA,
                status_date=status_date,
                maturity_date=maturity_date,
                **extra_attrs,
            )

            contract = create_contract(attrs, rf_observer)
            schedule = contract.generate_event_schedule()

            # All contracts should generate schedules
            assert schedule is not None
            assert hasattr(schedule, "events")
            assert schedule.contract_id == attrs.contract_id


# ==================== Test: Error Handling ====================


class TestDerivativeErrorHandling:
    """Test error handling across derivative contracts."""

    def test_invalid_contract_type_raises_error(self, status_date, maturity_date, rf_observer):
        """Test that invalid contract type raises error."""
        attrs = ContractAttributes(
            contract_id="INVALID-001",
            contract_type=ContractType.FXOUT,  # Will be overridden
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            currency="USD",
        )

        # Manually set to invalid type (simulating corruption)
        # This should be caught by factory
        # Note: Can't actually create invalid enum, so test other validations

        # Test missing required attributes instead
        with pytest.raises((ValueError, TypeError)):
            attrs_invalid = ContractAttributes(
                contract_id="FXOUT-INVALID",
                contract_type=ContractType.FXOUT,
                contract_role=ContractRole.RPA,
                status_date=status_date,
                maturity_date=maturity_date,
                currency="USD",
                # Missing currency_2, notional_principal, etc.
            )
            contract = create_contract(attrs_invalid, rf_observer)
            contract.initialize_state()

    def test_composition_without_child_observer_raises(
        self, status_date, maturity_date, rf_observer
    ):
        """Test that composition contracts without child observer raise errors."""
        composition_types = [
            (ContractType.SWAPS, '{"FirstLeg": "L1", "SecondLeg": "L2"}'),
            (ContractType.CAPFL, '{"Underlying": "SWAP-001"}'),
            (ContractType.CEG, '{"CoveredContract": "LOAN-001"}'),
            (ContractType.CEC, '{"CoveredContract": "LOAN-001", "CoveringContract": "STK-001"}'),
        ]

        for contract_type, contract_structure in composition_types:
            attrs = ContractAttributes(
                contract_id=f"{contract_type.value}-001",
                contract_type=contract_type,
                contract_role=ContractRole.RPA,
                status_date=status_date,
                maturity_date=maturity_date,
                contract_structure=contract_structure,
                currency="USD",
                # Type-specific attributes
                coverage=0.8 if contract_type in [ContractType.CEG, ContractType.CEC] else None,
                credit_event_type=(
                    "DL" if contract_type in [ContractType.CEG, ContractType.CEC] else None
                ),
                credit_enhancement_guarantee_extent=(
                    "NO" if contract_type in [ContractType.CEG, ContractType.CEC] else None
                ),
                rate_reset_cap=0.06 if contract_type == ContractType.CAPFL else None,
            )

            with pytest.raises(ValueError, match="child_contract_observer"):
                create_contract(attrs, rf_observer, None)


# ==================== Test: Simulation Consistency ====================


class TestSimulationConsistency:
    """Test that simulations work consistently across derivatives."""

    def test_fxout_simulation_completes(self, status_date, maturity_date, rf_observer):
        """Test FXOUT simulation completes without errors."""
        attrs = ContractAttributes(
            contract_id="FX-001",
            contract_type=ContractType.FXOUT,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            settlement_date=maturity_date,
            currency="USD",
            currency_2="EUR",
            notional_principal=100000.0,
            notional_principal_2=90000.0,
            delivery_settlement="D",
        )

        contract = create_contract(attrs, rf_observer)

        # Simulation should complete
        result = contract.simulate(rf_observer)
        assert result is not None
        assert hasattr(result, "events")

    def test_option_simulation_completes(self, status_date, maturity_date, rf_observer):
        """Test OPTNS simulation completes without errors."""
        attrs = ContractAttributes(
            contract_id="OPT-001",
            contract_type=ContractType.OPTNS,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            option_type="C",
            option_strike_1=105000.0,
            option_exercise_type="E",
            currency="USD",
            contract_structure='{"Underlier": "UNDERLIER-001"}',
        )

        contract = create_contract(attrs, rf_observer)

        # Simulation should complete
        result = contract.simulate(rf_observer)
        assert result is not None

    def test_future_simulation_completes(self, status_date, maturity_date, rf_observer):
        """Test FUTUR simulation completes without errors."""
        attrs = ContractAttributes(
            contract_id="FUT-001",
            contract_type=ContractType.FUTUR,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            future_price=105000.0,
            currency="USD",
            contract_structure='{"Underlier": "UNDERLIER-001"}',
        )

        contract = create_contract(attrs, rf_observer)

        # Simulation should complete
        result = contract.simulate(rf_observer)
        assert result is not None

    def test_swap_simulation_completes(self, status_date, maturity_date, rf_observer):
        """Test SWPPV simulation completes without errors."""
        attrs = ContractAttributes(
            contract_id="SWAP-001",
            contract_type=ContractType.SWPPV,
            contract_role=ContractRole.RFL,
            status_date=status_date,
            maturity_date=maturity_date,
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            nominal_interest_rate_2=0.03,
            currency="USD",
            interest_payment_cycle="3M",
            rate_reset_cycle="3M",
        )

        contract = create_contract(attrs, rf_observer)

        # Simulation should complete
        result = contract.simulate(rf_observer)
        assert result is not None
