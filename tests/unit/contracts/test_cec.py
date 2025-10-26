"""Tests for Credit Enhancement Collateral (CEC) contract implementation."""

import jax.numpy as jnp
import pytest

from jactus.contracts.cec import CreditEnhancementCollateralContract
from jactus.core import (
    ActusDateTime,
    ContractAttributes,
    ContractPerformance,
    ContractRole,
    ContractState,
    ContractType,
    EventType,
)
from jactus.observers import ConstantRiskFactorObserver, MockChildContractObserver


# Test fixtures
@pytest.fixture
def status_date():
    """Standard status date for tests."""
    return ActusDateTime(2024, 1, 1, 0, 0, 0)


@pytest.fixture
def maturity_date():
    """Standard maturity date for tests."""
    return ActusDateTime(2029, 1, 1, 0, 0, 0)


@pytest.fixture
def rf_observer():
    """Risk factor observer with constant rate."""
    return ConstantRiskFactorObserver(0.03)


@pytest.fixture
def child_observer():
    """Child contract observer for CEC tests."""
    return MockChildContractObserver()


@pytest.fixture
def covered_state(status_date, maturity_date):
    """Standard covered contract state."""
    return ContractState(
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


@pytest.fixture
def covering_state(status_date, maturity_date):
    """Standard covering contract state (collateral)."""
    return ContractState(
        tmd=maturity_date,
        sd=status_date,
        nt=jnp.array(150000.0, dtype=jnp.float32),  # Higher than covered
        ipnr=jnp.array(0.0, dtype=jnp.float32),
        ipac=jnp.array(0.0, dtype=jnp.float32),
        feac=jnp.array(0.0, dtype=jnp.float32),
        nsc=jnp.array(1.0, dtype=jnp.float32),
        isc=jnp.array(1.0, dtype=jnp.float32),
        prf=ContractPerformance.PF,
    )


# ==================== Test: CEC Initialization ====================


class TestCECInitialization:
    """Test CEC contract initialization and validation."""

    def test_cec_contract_creation(
        self,
        status_date,
        maturity_date,
        rf_observer,
        child_observer,
        covered_state,
        covering_state,
    ):
        """Test creating CEC with covered and covering contracts."""
        # Register covered and covering contracts
        child_observer.register_child("LOAN-001", state=covered_state)
        child_observer.register_child("STK-001", state=covering_state)

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

        cec = CreditEnhancementCollateralContract(attrs, rf_observer, child_observer)

        assert cec.attributes.contract_type == ContractType.CEC
        assert cec.attributes.coverage == 1.2
        assert cec.attributes.credit_enhancement_guarantee_extent == "NO"

    def test_cec_requires_child_observer(self, status_date, maturity_date, rf_observer):
        """Test that CEC requires a child contract observer."""
        attrs = ContractAttributes(
            contract_id="CEC-001",
            contract_type=ContractType.CEC,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=1.2,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001", "CoveringContract": "STK-001"}',
            currency="USD",
        )

        with pytest.raises(ValueError, match="child_contract_observer is required"):
            CreditEnhancementCollateralContract(attrs, rf_observer, None)

    def test_cec_requires_contract_structure(
        self, status_date, maturity_date, rf_observer, child_observer
    ):
        """Test that CEC requires contract_structure."""
        attrs = ContractAttributes(
            contract_id="CEC-001",
            contract_type=ContractType.CEC,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=1.2,
            credit_enhancement_guarantee_extent="NO",
            contract_structure=None,
            currency="USD",
        )

        with pytest.raises(ValueError, match="contract_structure.*is required"):
            CreditEnhancementCollateralContract(attrs, rf_observer, child_observer)

    def test_cec_requires_covered_contract(
        self, status_date, maturity_date, rf_observer, child_observer
    ):
        """Test that contract_structure must contain CoveredContract."""
        attrs = ContractAttributes(
            contract_id="CEC-001",
            contract_type=ContractType.CEC,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=1.2,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveringContract": "STK-001"}',  # Missing covered
            currency="USD",
        )

        with pytest.raises(ValueError, match="CoveredContract"):
            CreditEnhancementCollateralContract(attrs, rf_observer, child_observer)

    def test_cec_requires_covering_contract(
        self, status_date, maturity_date, rf_observer, child_observer
    ):
        """Test that contract_structure must contain CoveringContract."""
        attrs = ContractAttributes(
            contract_id="CEC-001",
            contract_type=ContractType.CEC,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=1.2,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001"}',  # Missing covering
            currency="USD",
        )

        with pytest.raises(ValueError, match="CoveringContract"):
            CreditEnhancementCollateralContract(attrs, rf_observer, child_observer)

    def test_cec_requires_coverage(self, status_date, maturity_date, rf_observer, child_observer):
        """Test that CEC requires coverage amount (CECV)."""
        attrs = ContractAttributes(
            contract_id="CEC-001",
            contract_type=ContractType.CEC,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=None,  # Missing
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001", "CoveringContract": "STK-001"}',
            currency="USD",
        )

        with pytest.raises(ValueError, match="coverage.*is required"):
            CreditEnhancementCollateralContract(attrs, rf_observer, child_observer)

    def test_cec_requires_guarantee_extent(
        self, status_date, maturity_date, rf_observer, child_observer
    ):
        """Test that CEC requires guarantee extent (CEGE)."""
        attrs = ContractAttributes(
            contract_id="CEC-001",
            contract_type=ContractType.CEC,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=1.2,
            credit_enhancement_guarantee_extent=None,  # Missing
            contract_structure='{"CoveredContract": "LOAN-001", "CoveringContract": "STK-001"}',
            currency="USD",
        )

        with pytest.raises(ValueError, match="credit_enhancement_guarantee_extent.*is required"):
            CreditEnhancementCollateralContract(attrs, rf_observer, child_observer)

    def test_cec_validates_guarantee_extent_values(
        self, status_date, maturity_date, rf_observer, child_observer
    ):
        """Test that CEGE must be NO, NI, or MV."""
        attrs = ContractAttributes(
            contract_id="CEC-001",
            contract_type=ContractType.CEC,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=1.2,
            credit_enhancement_guarantee_extent="INVALID",
            contract_structure='{"CoveredContract": "LOAN-001", "CoveringContract": "STK-001"}',
            currency="USD",
        )

        with pytest.raises(ValueError, match="must be NO, NI, or MV"):
            CreditEnhancementCollateralContract(attrs, rf_observer, child_observer)


# ==================== Test: Collateral and Exposure Calculation ====================


class TestCECCalculations:
    """Test CEC collateral and exposure calculations."""

    def test_cec_collateral_value_calculation(
        self,
        status_date,
        maturity_date,
        rf_observer,
        child_observer,
        covered_state,
        covering_state,
    ):
        """Test collateral value calculation from covering contracts."""
        child_observer.register_child("LOAN-001", state=covered_state)
        child_observer.register_child("STK-001", state=covering_state)

        attrs = ContractAttributes(
            contract_id="CEC-001",
            contract_type=ContractType.CEC,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=1.2,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001", "CoveringContract": "STK-001"}',
            currency="USD",
        )

        cec = CreditEnhancementCollateralContract(attrs, rf_observer, child_observer)

        # Collateral value = covering contract notional
        collateral_value = cec._calculate_collateral_value(status_date)
        assert collateral_value == pytest.approx(150000.0, abs=0.01)

    def test_cec_exposure_calculation_notional_only(
        self,
        status_date,
        maturity_date,
        rf_observer,
        child_observer,
        covered_state,
        covering_state,
    ):
        """Test exposure calculation for CEGE='NO' (notional only)."""
        child_observer.register_child("LOAN-001", state=covered_state)
        child_observer.register_child("STK-001", state=covering_state)

        attrs = ContractAttributes(
            contract_id="CEC-001",
            contract_type=ContractType.CEC,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=1.2,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001", "CoveringContract": "STK-001"}',
            currency="USD",
        )

        cec = CreditEnhancementCollateralContract(attrs, rf_observer, child_observer)

        # Exposure = covered contract notional only
        exposure = cec._calculate_exposure(status_date)
        assert exposure == pytest.approx(100000.0, abs=0.01)

    def test_cec_exposure_calculation_notional_plus_interest(
        self,
        status_date,
        maturity_date,
        rf_observer,
        child_observer,
        covered_state,
        covering_state,
    ):
        """Test exposure calculation for CEGE='NI' (notional + interest)."""
        child_observer.register_child("LOAN-001", state=covered_state)
        child_observer.register_child("STK-001", state=covering_state)

        attrs = ContractAttributes(
            contract_id="CEC-001",
            contract_type=ContractType.CEC,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=1.2,
            credit_enhancement_guarantee_extent="NI",
            contract_structure='{"CoveredContract": "LOAN-001", "CoveringContract": "STK-001"}',
            currency="USD",
        )

        cec = CreditEnhancementCollateralContract(attrs, rf_observer, child_observer)

        # Exposure = notional + interest
        exposure = cec._calculate_exposure(status_date)
        assert exposure == pytest.approx(105000.0, abs=0.01)  # 100000 + 5000

    def test_cec_exposure_calculation_market_value(
        self,
        status_date,
        maturity_date,
        rf_observer,
        child_observer,
        covered_state,
        covering_state,
    ):
        """Test exposure calculation for CEGE='MV' (market value)."""
        child_observer.register_child("LOAN-001", state=covered_state)
        child_observer.register_child("STK-001", state=covering_state)

        attrs = ContractAttributes(
            contract_id="CEC-001",
            contract_type=ContractType.CEC,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=1.2,
            credit_enhancement_guarantee_extent="MV",
            contract_structure='{"CoveredContract": "LOAN-001", "CoveringContract": "STK-001"}',
            currency="USD",
        )

        cec = CreditEnhancementCollateralContract(attrs, rf_observer, child_observer)

        # Exposure = market value (approximated as notional)
        exposure = cec._calculate_exposure(status_date)
        assert exposure == pytest.approx(100000.0, abs=0.01)

    def test_cec_collateral_sufficiency_check_sufficient(
        self,
        status_date,
        maturity_date,
        rf_observer,
        child_observer,
        covered_state,
        covering_state,
    ):
        """Test collateral sufficiency check when sufficient."""
        child_observer.register_child("LOAN-001", state=covered_state)
        child_observer.register_child("STK-001", state=covering_state)

        attrs = ContractAttributes(
            contract_id="CEC-001",
            contract_type=ContractType.CEC,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=1.2,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001", "CoveringContract": "STK-001"}',
            currency="USD",
        )

        cec = CreditEnhancementCollateralContract(attrs, rf_observer, child_observer)

        # Collateral = 150000, Required = 1.2 × 100000 = 120000
        # Sufficient with excess of 30000
        is_sufficient, difference = cec._check_collateral_sufficiency(status_date)
        assert is_sufficient is True
        assert difference == pytest.approx(30000.0, abs=0.01)

    def test_cec_collateral_sufficiency_check_insufficient(
        self, status_date, maturity_date, rf_observer, child_observer, covered_state
    ):
        """Test collateral sufficiency check when insufficient."""
        # Insufficient covering contract
        insufficient_covering = ContractState(
            tmd=maturity_date,
            sd=status_date,
            nt=jnp.array(80000.0, dtype=jnp.float32),  # Lower than required
            ipnr=jnp.array(0.0, dtype=jnp.float32),
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
            prf=ContractPerformance.PF,
        )

        child_observer.register_child("LOAN-001", state=covered_state)
        child_observer.register_child("STK-001", state=insufficient_covering)

        attrs = ContractAttributes(
            contract_id="CEC-001",
            contract_type=ContractType.CEC,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=1.2,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001", "CoveringContract": "STK-001"}',
            currency="USD",
        )

        cec = CreditEnhancementCollateralContract(attrs, rf_observer, child_observer)

        # Collateral = 80000, Required = 1.2 × 100000 = 120000
        # Insufficient with shortfall of -40000
        is_sufficient, difference = cec._check_collateral_sufficiency(status_date)
        assert is_sufficient is False
        assert difference == pytest.approx(-40000.0, abs=0.01)

    def test_cec_multiple_covered_contracts(
        self, status_date, maturity_date, rf_observer, child_observer, covering_state
    ):
        """Test CEC with multiple covered contracts."""
        # Create two covered contracts
        covered1 = ContractState(
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

        covered2 = ContractState(
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

        child_observer.register_child("LOAN-001", state=covered1)
        child_observer.register_child("LOAN-002", state=covered2)
        child_observer.register_child("STK-001", state=covering_state)

        attrs = ContractAttributes(
            contract_id="CEC-001",
            contract_type=ContractType.CEC,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=1.2,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContracts": ["LOAN-001", "LOAN-002"], "CoveringContract": "STK-001"}',
            currency="USD",
        )

        cec = CreditEnhancementCollateralContract(attrs, rf_observer, child_observer)

        # Total exposure = 100000 + 50000 = 150000
        exposure = cec._calculate_exposure(status_date)
        assert exposure == pytest.approx(150000.0, abs=0.01)


# ==================== Test: Event Schedule Generation ====================


class TestCECEventSchedule:
    """Test CEC event schedule generation."""

    def test_cec_generates_maturity_and_settlement(
        self,
        status_date,
        maturity_date,
        rf_observer,
        child_observer,
        covered_state,
        covering_state,
    ):
        """Test that CEC generates maturity and settlement events."""
        child_observer.register_child("LOAN-001", state=covered_state)
        child_observer.register_child("STK-001", state=covering_state)

        attrs = ContractAttributes(
            contract_id="CEC-001",
            contract_type=ContractType.CEC,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=1.2,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001", "CoveringContract": "STK-001"}',
            currency="USD",
        )

        cec = CreditEnhancementCollateralContract(attrs, rf_observer, child_observer)
        schedule = cec.generate_event_schedule()

        # Should have STD and MD events
        std_events = [e for e in schedule.events if e.event_type == EventType.STD]
        md_events = [e for e in schedule.events if e.event_type == EventType.MD]

        assert len(std_events) == 1
        assert len(md_events) == 1
        assert std_events[0].event_time == maturity_date
        assert md_events[0].event_time == maturity_date

    def test_cec_with_analysis_dates(
        self,
        status_date,
        maturity_date,
        rf_observer,
        child_observer,
        covered_state,
        covering_state,
    ):
        """Test CEC with analysis dates."""
        child_observer.register_child("LOAN-001", state=covered_state)
        child_observer.register_child("STK-001", state=covering_state)

        ad1 = ActusDateTime(2025, 1, 1, 0, 0, 0)
        ad2 = ActusDateTime(2026, 1, 1, 0, 0, 0)

        attrs = ContractAttributes(
            contract_id="CEC-001",
            contract_type=ContractType.CEC,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=1.2,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001", "CoveringContract": "STK-001"}',
            analysis_dates=[ad1, ad2],
            currency="USD",
        )

        cec = CreditEnhancementCollateralContract(attrs, rf_observer, child_observer)
        schedule = cec.generate_event_schedule()

        # Should have AD events
        ad_events = [e for e in schedule.events if e.event_type == EventType.AD]
        assert len(ad_events) == 2


# ==================== Test: State Initialization ====================


class TestCECStateInitialization:
    """Test CEC state initialization."""

    def test_cec_initialize_state_sufficient_collateral(
        self,
        status_date,
        maturity_date,
        rf_observer,
        child_observer,
        covered_state,
        covering_state,
    ):
        """Test CEC state initialization with sufficient collateral."""
        child_observer.register_child("LOAN-001", state=covered_state)
        child_observer.register_child("STK-001", state=covering_state)

        attrs = ContractAttributes(
            contract_id="CEC-001",
            contract_type=ContractType.CEC,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=1.2,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001", "CoveringContract": "STK-001"}',
            currency="USD",
        )

        cec = CreditEnhancementCollateralContract(attrs, rf_observer, child_observer)
        state = cec.initialize_state()

        # Nt = min(collateral, required)
        # Required = 1.2 × 100000 = 120000
        # Collateral = 150000
        # Nt = min(150000, 120000) = 120000
        assert state.tmd == maturity_date
        assert state.sd == status_date
        assert float(state.nt) == pytest.approx(120000.0, abs=0.01)
        assert state.prf == ContractPerformance.PF

    def test_cec_initialize_state_insufficient_collateral(
        self, status_date, maturity_date, rf_observer, child_observer, covered_state
    ):
        """Test CEC state initialization with insufficient collateral."""
        # Insufficient covering contract
        insufficient_covering = ContractState(
            tmd=maturity_date,
            sd=status_date,
            nt=jnp.array(80000.0, dtype=jnp.float32),
            ipnr=jnp.array(0.0, dtype=jnp.float32),
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
            prf=ContractPerformance.PF,
        )

        child_observer.register_child("LOAN-001", state=covered_state)
        child_observer.register_child("STK-001", state=insufficient_covering)

        attrs = ContractAttributes(
            contract_id="CEC-001",
            contract_type=ContractType.CEC,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=1.2,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001", "CoveringContract": "STK-001"}',
            currency="USD",
        )

        cec = CreditEnhancementCollateralContract(attrs, rf_observer, child_observer)
        state = cec.initialize_state()

        # Nt = min(collateral, required)
        # Required = 1.2 × 100000 = 120000
        # Collateral = 80000
        # Nt = min(80000, 120000) = 80000
        assert float(state.nt) == pytest.approx(80000.0, abs=0.01)


# ==================== Test: Edge Cases ====================


class TestCECEdgeCases:
    """Test CEC edge cases and error handling."""

    def test_cec_invalid_json_structure(
        self, status_date, maturity_date, rf_observer, child_observer
    ):
        """Test that invalid JSON in contract_structure raises error."""
        attrs = ContractAttributes(
            contract_id="CEC-001",
            contract_type=ContractType.CEC,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=1.2,
            credit_enhancement_guarantee_extent="NO",
            contract_structure="not valid json",
            currency="USD",
        )

        with pytest.raises(ValueError, match="valid JSON"):
            CreditEnhancementCollateralContract(attrs, rf_observer, child_observer)

    def test_cec_invalid_covered_contracts_type(
        self,
        status_date,
        maturity_date,
        rf_observer,
        child_observer,
        covered_state,
        covering_state,
    ):
        """Test error handling for invalid CoveredContracts type."""
        child_observer.register_child("LOAN-001", state=covered_state)
        child_observer.register_child("STK-001", state=covering_state)

        attrs = ContractAttributes(
            contract_id="CEC-001",
            contract_type=ContractType.CEC,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=1.2,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContracts": 123, "CoveringContract": "STK-001"}',  # Invalid type
            currency="USD",
        )

        cec = CreditEnhancementCollateralContract(attrs, rf_observer, child_observer)

        # Should raise error when trying to get covered contract IDs
        with pytest.raises(ValueError, match="must be list or string"):
            cec._get_covered_contract_ids()

    def test_cec_invalid_covering_contracts_type(
        self,
        status_date,
        maturity_date,
        rf_observer,
        child_observer,
        covered_state,
        covering_state,
    ):
        """Test error handling for invalid CoveringContracts type."""
        child_observer.register_child("LOAN-001", state=covered_state)
        child_observer.register_child("STK-001", state=covering_state)

        attrs = ContractAttributes(
            contract_id="CEC-001",
            contract_type=ContractType.CEC,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=1.2,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001", "CoveringContracts": 123}',  # Invalid type
            currency="USD",
        )

        cec = CreditEnhancementCollateralContract(attrs, rf_observer, child_observer)

        # Should raise error when trying to get covering contract IDs
        with pytest.raises(ValueError, match="must be list or string"):
            cec._get_covering_contract_ids()

    def test_cec_covered_contracts_as_string(
        self,
        status_date,
        maturity_date,
        rf_observer,
        child_observer,
        covered_state,
        covering_state,
    ):
        """Test CoveredContracts as a string (should be converted to list)."""
        child_observer.register_child("LOAN-001", state=covered_state)
        child_observer.register_child("STK-001", state=covering_state)

        attrs = ContractAttributes(
            contract_id="CEC-001",
            contract_type=ContractType.CEC,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=1.2,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContracts": "LOAN-001", "CoveringContract": "STK-001"}',
            currency="USD",
        )

        cec = CreditEnhancementCollateralContract(attrs, rf_observer, child_observer)
        covered_ids = cec._get_covered_contract_ids()

        assert covered_ids == ["LOAN-001"]

    def test_cec_covering_contracts_as_string(
        self,
        status_date,
        maturity_date,
        rf_observer,
        child_observer,
        covered_state,
        covering_state,
    ):
        """Test CoveringContracts as a string (should be converted to list)."""
        child_observer.register_child("LOAN-001", state=covered_state)
        child_observer.register_child("STK-001", state=covering_state)

        attrs = ContractAttributes(
            contract_id="CEC-001",
            contract_type=ContractType.CEC,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=1.2,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001", "CoveringContracts": "STK-001"}',
            currency="USD",
        )

        cec = CreditEnhancementCollateralContract(attrs, rf_observer, child_observer)
        covering_ids = cec._get_covering_contract_ids()

        assert covering_ids == ["STK-001"]

    def test_cec_multiple_covering_contracts(
        self, status_date, maturity_date, rf_observer, child_observer, covered_state
    ):
        """Test CEC with multiple covering contracts."""
        # Create two covering contracts
        covering1 = ContractState(
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

        covering2 = ContractState(
            tmd=maturity_date,
            sd=status_date,
            nt=jnp.array(50000.0, dtype=jnp.float32),
            ipnr=jnp.array(0.0, dtype=jnp.float32),
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
            prf=ContractPerformance.PF,
        )

        child_observer.register_child("LOAN-001", state=covered_state)
        child_observer.register_child("STK-001", state=covering1)
        child_observer.register_child("STK-002", state=covering2)

        attrs = ContractAttributes(
            contract_id="CEC-001",
            contract_type=ContractType.CEC,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=1.2,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001", "CoveringContracts": ["STK-001", "STK-002"]}',
            currency="USD",
        )

        cec = CreditEnhancementCollateralContract(attrs, rf_observer, child_observer)

        # Total collateral = 100000 + 50000 = 150000
        collateral_value = cec._calculate_collateral_value(status_date)
        assert collateral_value == pytest.approx(150000.0, abs=0.01)

    def test_cec_non_dict_json_structure(
        self, status_date, maturity_date, rf_observer, child_observer
    ):
        """Test that non-dict JSON in contract_structure raises error."""
        attrs = ContractAttributes(
            contract_id="CEC-001",
            contract_type=ContractType.CEC,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=1.2,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='["array", "not", "dict"]',
            currency="USD",
        )

        with pytest.raises(ValueError, match="must be a JSON object"):
            CreditEnhancementCollateralContract(attrs, rf_observer, child_observer)

    def test_cec_wrong_contract_type(self, status_date, maturity_date, rf_observer, child_observer):
        """Test that wrong contract type raises error."""
        attrs = ContractAttributes(
            contract_id="CEC-001",
            contract_type=ContractType.PAM,  # Wrong type
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=1.2,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001", "CoveringContract": "STK-001"}',
            currency="USD",
        )

        with pytest.raises(ValueError, match="Expected contract_type=CEC"):
            CreditEnhancementCollateralContract(attrs, rf_observer, child_observer)

    def test_cec_with_termination_date(
        self,
        status_date,
        maturity_date,
        rf_observer,
        child_observer,
        covered_state,
        covering_state,
    ):
        """Test CEC with termination date."""
        child_observer.register_child("LOAN-001", state=covered_state)
        child_observer.register_child("STK-001", state=covering_state)

        termination = ActusDateTime(2025, 6, 1, 0, 0, 0)

        attrs = ContractAttributes(
            contract_id="CEC-001",
            contract_type=ContractType.CEC,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            termination_date=termination,
            coverage=1.2,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001", "CoveringContract": "STK-001"}',
            currency="USD",
        )

        cec = CreditEnhancementCollateralContract(attrs, rf_observer, child_observer)
        schedule = cec.generate_event_schedule()

        # Should have TD event
        td_events = [e for e in schedule.events if e.event_type == EventType.TD]
        assert len(td_events) == 1
        assert td_events[0].event_time == termination


# ==================== Test: Payoff and State Functions ====================


class TestCECPayoffAndState:
    """Test CEC payoff and state transition functions."""

    def test_cec_payoff_function(
        self,
        status_date,
        maturity_date,
        rf_observer,
        child_observer,
        covered_state,
        covering_state,
    ):
        """Test getting payoff function."""
        child_observer.register_child("LOAN-001", state=covered_state)
        child_observer.register_child("STK-001", state=covering_state)

        attrs = ContractAttributes(
            contract_id="CEC-001",
            contract_type=ContractType.CEC,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=1.2,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001", "CoveringContract": "STK-001"}',
            currency="USD",
        )

        cec = CreditEnhancementCollateralContract(attrs, rf_observer, child_observer)
        payoff_fn = cec.get_payoff_function(EventType.STD)

        assert payoff_fn is not None
        # Test payoff calculation
        state = cec.initialize_state()
        payoff = payoff_fn.calculate_payoff(EventType.STD, state, attrs, maturity_date, rf_observer)
        assert float(payoff) == 0.0  # Returns zero (actual payoffs in events)

    def test_cec_state_transition_function(
        self,
        status_date,
        maturity_date,
        rf_observer,
        child_observer,
        covered_state,
        covering_state,
    ):
        """Test getting state transition function."""
        child_observer.register_child("LOAN-001", state=covered_state)
        child_observer.register_child("STK-001", state=covering_state)

        attrs = ContractAttributes(
            contract_id="CEC-001",
            contract_type=ContractType.CEC,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=1.2,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001", "CoveringContract": "STK-001"}',
            currency="USD",
        )

        cec = CreditEnhancementCollateralContract(attrs, rf_observer, child_observer)
        stf_fn = cec.get_state_transition_function(EventType.STD)

        assert stf_fn is not None
        # Test state transition
        state = cec.initialize_state()
        new_state = stf_fn.transition_state(EventType.STD, state, attrs, maturity_date, rf_observer)
        assert new_state.sd == maturity_date


# ==================== Test: Factory Creation ====================


class TestCECFactory:
    """Test CEC creation through factory."""

    def test_cec_factory_creation(
        self,
        status_date,
        maturity_date,
        rf_observer,
        child_observer,
        covered_state,
        covering_state,
    ):
        """Test creating CEC through contract factory."""
        from jactus.contracts import create_contract

        child_observer.register_child("LOAN-001", state=covered_state)
        child_observer.register_child("STK-001", state=covering_state)

        attrs = ContractAttributes(
            contract_id="CEC-001",
            contract_type=ContractType.CEC,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=1.2,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001", "CoveringContract": "STK-001"}',
            currency="USD",
        )

        cec = create_contract(attrs, rf_observer, child_observer)

        assert isinstance(cec, CreditEnhancementCollateralContract)
        assert cec.attributes.contract_type == ContractType.CEC
