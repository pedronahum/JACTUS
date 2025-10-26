"""Integration tests for derivative contract composition patterns.

These tests validate that contracts correctly compose with child contracts,
including SWAPS with legs, CAPFL with underliers, OPTNS with underliers,
and credit enhancements with covered/covering contracts.
"""

import jax.numpy as jnp
import pytest

from jactus.contracts import create_contract
from jactus.core import (
    ActusDateTime,
    ContractAttributes,
    ContractPerformance,
    ContractRole,
    ContractState,
    ContractType,
)
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


@pytest.fixture
def child_observer():
    """Child contract observer."""
    return MockChildContractObserver()


# ==================== Test: SWAPS Composition ====================


class TestSWAPSComposition:
    """Test SWAPS contract composition with different leg types."""

    def test_swaps_with_pam_legs(self, status_date, maturity_date, rf_observer, child_observer):
        """Test SWAPS with two PAM legs."""
        # Create states for two PAM legs
        leg1_state = ContractState(
            tmd=maturity_date,
            sd=status_date,
            nt=jnp.array(100000.0, dtype=jnp.float32),
            ipnr=jnp.array(0.05, dtype=jnp.float32),
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
            prf=ContractPerformance.PF,
        )

        leg2_state = ContractState(
            tmd=maturity_date,
            sd=status_date,
            nt=jnp.array(100000.0, dtype=jnp.float32),
            ipnr=jnp.array(0.03, dtype=jnp.float32),
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
            prf=ContractPerformance.PF,
        )

        child_observer.register_child("PAM-LEG1", state=leg1_state, events=[])
        child_observer.register_child("PAM-LEG2", state=leg2_state, events=[])

        # Create SWAPS contract
        attrs = ContractAttributes(
            contract_id="SWAP-001",
            contract_type=ContractType.SWAPS,
            contract_role=ContractRole.RFL,
            status_date=status_date,
            maturity_date=maturity_date,
            delivery_settlement="D",  # Net settlement
            contract_structure='{"FirstLeg": "PAM-LEG1", "SecondLeg": "PAM-LEG2"}',
            currency="USD",
        )

        swap = create_contract(attrs, rf_observer, child_observer)

        # Verify swap initializes
        state = swap.initialize_state()
        assert state is not None
        assert state.tmd == maturity_date

        # Verify schedule generation
        # Note: Since child contracts have no events registered,
        # the SWAPS schedule will be empty (this is correct behavior)
        schedule = swap.generate_event_schedule()
        assert schedule is not None
        assert len(schedule.events) == 0  # No events because child legs have no events

    def test_swaps_gross_vs_net_settlement(
        self, status_date, maturity_date, rf_observer, child_observer
    ):
        """Test SWAPS gross vs net settlement modes."""
        leg_state = ContractState(
            tmd=maturity_date,
            sd=status_date,
            nt=jnp.array(100000.0, dtype=jnp.float32),
            ipnr=jnp.array(0.05, dtype=jnp.float32),
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
            prf=ContractPerformance.PF,
        )

        child_observer.register_child("LEG1", state=leg_state, events=[])
        child_observer.register_child("LEG2", state=leg_state, events=[])

        # Net settlement
        attrs_net = ContractAttributes(
            contract_id="SWAP-NET",
            contract_type=ContractType.SWAPS,
            contract_role=ContractRole.RFL,
            status_date=status_date,
            maturity_date=maturity_date,
            delivery_settlement="D",  # Net
            contract_structure='{"FirstLeg": "LEG1", "SecondLeg": "LEG2"}',
            currency="USD",
        )

        swap_net = create_contract(attrs_net, rf_observer, child_observer)
        schedule_net = swap_net.generate_event_schedule()

        # Gross settlement
        attrs_gross = ContractAttributes(
            contract_id="SWAP-GROSS",
            contract_type=ContractType.SWAPS,
            contract_role=ContractRole.RFL,
            status_date=status_date,
            maturity_date=maturity_date,
            delivery_settlement="S",  # Gross
            contract_structure='{"FirstLeg": "LEG1", "SecondLeg": "LEG2"}',
            currency="USD",
        )

        swap_gross = create_contract(attrs_gross, rf_observer, child_observer)
        schedule_gross = swap_gross.generate_event_schedule()

        # Gross should have more events (separate leg events)
        # Net should merge congruent events
        assert schedule_net is not None
        assert schedule_gross is not None


# ==================== Test: CAPFL Composition ====================


class TestCAPFLComposition:
    """Test CAPFL contract composition with underlier."""

    def test_capfl_with_swap_underlier(
        self, status_date, maturity_date, rf_observer, child_observer
    ):
        """Test CAPFL with swap as underlier."""
        # Create underlier swap state
        swap_state = ContractState(
            tmd=maturity_date,
            sd=status_date,
            nt=jnp.array(100000.0, dtype=jnp.float32),
            ipnr=jnp.array(0.05, dtype=jnp.float32),
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
            prf=ContractPerformance.PF,
        )

        child_observer.register_child("SWAP-001", state=swap_state, events=[])

        # Create CAPFL (interest rate cap)
        attrs = ContractAttributes(
            contract_id="CAP-001",
            contract_type=ContractType.CAPFL,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            rate_reset_cap=0.06,  # 6% cap
            contract_structure='{"Underlying": "SWAP-001"}',
            currency="USD",
        )

        cap = create_contract(attrs, rf_observer, child_observer)

        # Verify cap initializes
        state = cap.initialize_state()
        assert state is not None

        # Verify schedule generation
        schedule = cap.generate_event_schedule()
        assert schedule is not None


# ==================== Test: CEG/CEC Composition ====================


class TestCreditEnhancementComposition:
    """Test CEG and CEC composition with covered/covering contracts."""

    def test_ceg_with_loan_coverage(self, status_date, maturity_date, rf_observer, child_observer):
        """Test CEG covering a loan."""
        # Create covered loan state
        loan_state = ContractState(
            tmd=maturity_date,
            sd=status_date,
            nt=jnp.array(100000.0, dtype=jnp.float32),
            ipnr=jnp.array(0.05, dtype=jnp.float32),
            ipac=jnp.array(5000.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
            prf=ContractPerformance.PF,
        )

        child_observer.register_child("LOAN-001", state=loan_state, events=[])

        # Create CEG (credit guarantee)
        attrs = ContractAttributes(
            contract_id="CEG-001",
            contract_type=ContractType.CEG,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=0.8,  # 80% coverage
            credit_event_type=ContractPerformance.DL,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001"}',
            currency="USD",
        )

        ceg = create_contract(attrs, rf_observer, child_observer)

        # Verify CEG initializes
        state = ceg.initialize_state()
        assert state is not None
        # Coverage amount = 0.8 * 100000
        assert float(state.nt) == pytest.approx(80000.0, abs=0.01)

    def test_cec_with_collateral_and_loan(
        self, status_date, maturity_date, rf_observer, child_observer
    ):
        """Test CEC with collateral securing a loan."""
        # Covered loan
        loan_state = ContractState(
            tmd=maturity_date,
            sd=status_date,
            nt=jnp.array(100000.0, dtype=jnp.float32),
            ipnr=jnp.array(0.05, dtype=jnp.float32),
            ipac=jnp.array(5000.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
            prf=ContractPerformance.PF,
        )

        # Covering collateral (stock)
        collateral_state = ContractState(
            tmd=maturity_date,
            sd=status_date,
            nt=jnp.array(150000.0, dtype=jnp.float32),
            ipnr=jnp.array(0.0, dtype=jnp.float32),
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
            prf=ContractPerformance.PF,
        )

        child_observer.register_child("LOAN-001", state=loan_state, events=[])
        child_observer.register_child("STK-001", state=collateral_state, events=[])

        # Create CEC
        attrs = ContractAttributes(
            contract_id="CEC-001",
            contract_type=ContractType.CEC,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=1.2,  # 120% collateral requirement
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001", "CoveringContract": "STK-001"}',
            currency="USD",
        )

        cec = create_contract(attrs, rf_observer, child_observer)

        # Verify CEC initializes
        state = cec.initialize_state()
        assert state is not None

        # Nt = min(collateral, required)
        # Required = 1.2 × 100000 = 120000
        # Collateral = 150000
        # Nt = 120000
        assert float(state.nt) == pytest.approx(120000.0, abs=0.01)

    def test_ceg_multiple_covered_contracts(
        self, status_date, maturity_date, rf_observer, child_observer
    ):
        """Test CEG with multiple covered contracts."""
        loan1_state = ContractState(
            tmd=maturity_date,
            sd=status_date,
            nt=jnp.array(100000.0, dtype=jnp.float32),
            ipnr=jnp.array(0.05, dtype=jnp.float32),
            ipac=jnp.array(5000.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
            prf=ContractPerformance.PF,
        )

        loan2_state = ContractState(
            tmd=maturity_date,
            sd=status_date,
            nt=jnp.array(50000.0, dtype=jnp.float32),
            ipnr=jnp.array(0.05, dtype=jnp.float32),
            ipac=jnp.array(2500.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
            prf=ContractPerformance.PF,
        )

        child_observer.register_child("LOAN-001", state=loan1_state, events=[])
        child_observer.register_child("LOAN-002", state=loan2_state, events=[])

        attrs = ContractAttributes(
            contract_id="CEG-001",
            contract_type=ContractType.CEG,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=0.8,
            credit_event_type=ContractPerformance.DL,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContracts": ["LOAN-001", "LOAN-002"]}',
            currency="USD",
        )

        ceg = create_contract(attrs, rf_observer, child_observer)
        state = ceg.initialize_state()

        # Coverage = 0.8 × (100000 + 50000) = 120000
        assert float(state.nt) == pytest.approx(120000.0, abs=0.01)


# ==================== Test: Cross-Contract Interactions ====================


class TestCrossContractInteractions:
    """Test interactions between different derivative types."""

    def test_option_on_future(self, status_date, maturity_date, rf_observer, child_observer):
        """Test option with future as underlier."""
        # Future state
        future_state = ContractState(
            tmd=maturity_date,
            sd=status_date,
            nt=jnp.array(100000.0, dtype=jnp.float32),
            ipnr=jnp.array(0.0, dtype=jnp.float32),
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
            prf=ContractPerformance.PF,
        )

        child_observer.register_child("FUT-001", state=future_state, events=[])

        # Option on future
        attrs = ContractAttributes(
            contract_id="OPT-001",
            contract_type=ContractType.OPTNS,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            option_type="C",  # Call option
            option_strike_1=105000.0,
            option_exercise_type="E",  # European
            contract_structure='{"Underlying": "FUT-001"}',
            currency="USD",
        )

        option = create_contract(attrs, rf_observer, child_observer)
        state = option.initialize_state()

        assert state is not None

    def test_factory_creates_all_derivatives(self, status_date, maturity_date, rf_observer):
        """Test that factory can create all derivative types."""
        # Test each derivative type can be created
        derivative_types = [
            ContractType.FXOUT,
            ContractType.OPTNS,
            ContractType.FUTUR,
            ContractType.SWPPV,
        ]

        for contract_type in derivative_types:
            attrs = ContractAttributes(
                contract_id=f"{contract_type.value}-001",
                contract_type=contract_type,
                contract_role=ContractRole.RPA,
                status_date=status_date,
                maturity_date=maturity_date,
                # Type-specific required attributes
                currency="USD",
                currency_2="EUR" if contract_type == ContractType.FXOUT else None,
                notional_principal=100000.0,
                notional_principal_2=100000.0 if contract_type == ContractType.FXOUT else None,
                nominal_interest_rate=0.05 if contract_type == ContractType.SWPPV else None,
                nominal_interest_rate_2=0.03 if contract_type == ContractType.SWPPV else None,
                option_type="C" if contract_type == ContractType.OPTNS else None,
                option_strike_1=105000.0 if contract_type == ContractType.OPTNS else None,
                option_exercise_type="E" if contract_type == ContractType.OPTNS else None,
                future_price=105000.0 if contract_type == ContractType.FUTUR else None,
                # Additional required attributes
                delivery_settlement="D" if contract_type == ContractType.FXOUT else None,
                contract_structure='{"Underlier": "UNDERLIER-001"}' if contract_type in [ContractType.OPTNS, ContractType.FUTUR] else None,
                interest_payment_cycle="3M" if contract_type == ContractType.SWPPV else None,
                rate_reset_cycle="3M" if contract_type == ContractType.SWPPV else None,
            )

            contract = create_contract(attrs, rf_observer, None)
            assert contract is not None
            assert contract.attributes.contract_type == contract_type
