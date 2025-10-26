"""Tests for Credit Enhancement Guarantee (CEG) contract implementation."""

import jax.numpy as jnp
import pytest

from jactus.contracts.ceg import CreditEnhancementGuaranteeContract
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
    """Child contract observer for CEG tests."""
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


# ==================== Test: CEG Initialization ====================


class TestCEGInitialization:
    """Test CEG contract initialization and validation."""

    def test_ceg_contract_creation_with_single_covered(
        self, status_date, maturity_date, rf_observer, child_observer, covered_state
    ):
        """Test creating CEG with single covered contract."""
        # Register covered contract
        child_observer.register_child("LOAN-001", state=covered_state)

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

        ceg = CreditEnhancementGuaranteeContract(attrs, rf_observer, child_observer)

        assert ceg.attributes.contract_type == ContractType.CEG
        assert ceg.attributes.coverage == 0.8
        assert ceg.attributes.credit_event_type == ContractPerformance.DL
        assert ceg.attributes.credit_enhancement_guarantee_extent == "NO"

    def test_ceg_contract_creation_with_multiple_covered(
        self, status_date, maturity_date, rf_observer, child_observer, covered_state
    ):
        """Test creating CEG with multiple covered contracts."""
        # Register multiple covered contracts
        child_observer.register_child("LOAN-001", state=covered_state)
        child_observer.register_child("LOAN-002", state=covered_state)

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

        ceg = CreditEnhancementGuaranteeContract(attrs, rf_observer, child_observer)

        assert ceg.attributes.contract_type == ContractType.CEG

    def test_ceg_requires_child_observer(self, status_date, maturity_date, rf_observer):
        """Test that CEG requires a child contract observer."""
        attrs = ContractAttributes(
            contract_id="CEG-001",
            contract_type=ContractType.CEG,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=0.8,
            credit_event_type=ContractPerformance.DL,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001"}',
            currency="USD",
        )

        with pytest.raises(ValueError, match="child_contract_observer is required"):
            CreditEnhancementGuaranteeContract(attrs, rf_observer, None)

    def test_ceg_requires_contract_structure(
        self, status_date, maturity_date, rf_observer, child_observer
    ):
        """Test that CEG requires contract_structure with covered contract."""
        attrs = ContractAttributes(
            contract_id="CEG-001",
            contract_type=ContractType.CEG,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=0.8,
            credit_event_type=ContractPerformance.DL,
            credit_enhancement_guarantee_extent="NO",
            contract_structure=None,
            currency="USD",
        )

        with pytest.raises(ValueError, match="contract_structure.*is required"):
            CreditEnhancementGuaranteeContract(attrs, rf_observer, child_observer)

    def test_ceg_requires_covered_contract_key(
        self, status_date, maturity_date, rf_observer, child_observer
    ):
        """Test that contract_structure must contain CoveredContract key."""
        attrs = ContractAttributes(
            contract_id="CEG-001",
            contract_type=ContractType.CEG,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=0.8,
            credit_event_type=ContractPerformance.DL,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"Underlying": "LOAN-001"}',  # Wrong key
            currency="USD",
        )

        with pytest.raises(ValueError, match="CoveredContract"):
            CreditEnhancementGuaranteeContract(attrs, rf_observer, child_observer)

    def test_ceg_requires_coverage(self, status_date, maturity_date, rf_observer, child_observer):
        """Test that CEG requires coverage amount (CECV)."""
        attrs = ContractAttributes(
            contract_id="CEG-001",
            contract_type=ContractType.CEG,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=None,  # Missing
            credit_event_type=ContractPerformance.DL,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001"}',
            currency="USD",
        )

        with pytest.raises(ValueError, match="coverage.*is required"):
            CreditEnhancementGuaranteeContract(attrs, rf_observer, child_observer)

    def test_ceg_requires_credit_event_type(
        self, status_date, maturity_date, rf_observer, child_observer
    ):
        """Test that CEG requires credit event type (CET)."""
        attrs = ContractAttributes(
            contract_id="CEG-001",
            contract_type=ContractType.CEG,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=0.8,
            credit_event_type=None,  # Missing
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001"}',
            currency="USD",
        )

        with pytest.raises(ValueError, match="credit_event_type.*is required"):
            CreditEnhancementGuaranteeContract(attrs, rf_observer, child_observer)

    def test_ceg_requires_guarantee_extent(
        self, status_date, maturity_date, rf_observer, child_observer
    ):
        """Test that CEG requires guarantee extent (CEGE)."""
        attrs = ContractAttributes(
            contract_id="CEG-001",
            contract_type=ContractType.CEG,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=0.8,
            credit_event_type=ContractPerformance.DL,
            credit_enhancement_guarantee_extent=None,  # Missing
            contract_structure='{"CoveredContract": "LOAN-001"}',
            currency="USD",
        )

        with pytest.raises(ValueError, match="credit_enhancement_guarantee_extent.*is required"):
            CreditEnhancementGuaranteeContract(attrs, rf_observer, child_observer)

    def test_ceg_validates_guarantee_extent_values(
        self, status_date, maturity_date, rf_observer, child_observer
    ):
        """Test that CEGE must be NO, NI, or MV."""
        attrs = ContractAttributes(
            contract_id="CEG-001",
            contract_type=ContractType.CEG,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=0.8,
            credit_event_type=ContractPerformance.DL,
            credit_enhancement_guarantee_extent="INVALID",
            contract_structure='{"CoveredContract": "LOAN-001"}',
            currency="USD",
        )

        with pytest.raises(ValueError, match="must be NO, NI, or MV"):
            CreditEnhancementGuaranteeContract(attrs, rf_observer, child_observer)


# ==================== Test: Coverage Amount Calculation ====================


class TestCEGCoverageCalculation:
    """Test CEG coverage amount calculations for different CEGE modes."""

    def test_ceg_coverage_notional_only(
        self, status_date, maturity_date, rf_observer, child_observer, covered_state
    ):
        """Test coverage calculation for CEGE='NO' (notional only)."""
        child_observer.register_child("LOAN-001", state=covered_state)

        attrs = ContractAttributes(
            contract_id="CEG-001",
            contract_type=ContractType.CEG,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=0.8,
            credit_event_type=ContractPerformance.DL,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001"}',
            currency="USD",
        )

        ceg = CreditEnhancementGuaranteeContract(attrs, rf_observer, child_observer)
        state = ceg.initialize_state()

        # Coverage = 0.8 * 100000 (notional only)
        expected_coverage = 0.8 * 100000.0
        assert float(state.nt) == pytest.approx(expected_coverage, abs=0.01)

    def test_ceg_coverage_notional_plus_interest(
        self, status_date, maturity_date, rf_observer, child_observer, covered_state
    ):
        """Test coverage calculation for CEGE='NI' (notional + interest)."""
        child_observer.register_child("LOAN-001", state=covered_state)

        attrs = ContractAttributes(
            contract_id="CEG-001",
            contract_type=ContractType.CEG,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=0.8,
            credit_event_type=ContractPerformance.DL,
            credit_enhancement_guarantee_extent="NI",
            contract_structure='{"CoveredContract": "LOAN-001"}',
            currency="USD",
        )

        ceg = CreditEnhancementGuaranteeContract(attrs, rf_observer, child_observer)
        state = ceg.initialize_state()

        # Coverage = 0.8 * (100000 + 5000)
        expected_coverage = 0.8 * (100000.0 + 5000.0)
        assert float(state.nt) == pytest.approx(expected_coverage, abs=0.01)

    def test_ceg_coverage_market_value(
        self, status_date, maturity_date, rf_observer, child_observer, covered_state
    ):
        """Test coverage calculation for CEGE='MV' (market value)."""
        child_observer.register_child("LOAN-001", state=covered_state)

        attrs = ContractAttributes(
            contract_id="CEG-001",
            contract_type=ContractType.CEG,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=0.8,
            credit_event_type=ContractPerformance.DL,
            credit_enhancement_guarantee_extent="MV",
            contract_structure='{"CoveredContract": "LOAN-001"}',
            currency="USD",
        )

        ceg = CreditEnhancementGuaranteeContract(attrs, rf_observer, child_observer)
        state = ceg.initialize_state()

        # MV approximated as notional for now
        expected_coverage = 0.8 * 100000.0
        assert float(state.nt) == pytest.approx(expected_coverage, abs=0.01)

    def test_ceg_coverage_with_contract_without_nt(
        self, status_date, maturity_date, rf_observer, child_observer
    ):
        """Test coverage calculation when contract has no notional attribute."""
        # Create a minimal state without nt attribute
        minimal_state = ContractState(
            tmd=maturity_date,
            sd=status_date,
            nt=jnp.array(0.0, dtype=jnp.float32),  # Zero notional
            ipnr=jnp.array(0.0, dtype=jnp.float32),
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
            prf=ContractPerformance.PF,
        )

        child_observer.register_child("LOAN-001", state=minimal_state)

        attrs = ContractAttributes(
            contract_id="CEG-001",
            contract_type=ContractType.CEG,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=0.8,
            credit_event_type=ContractPerformance.DL,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001"}',
            currency="USD",
        )

        ceg = CreditEnhancementGuaranteeContract(attrs, rf_observer, child_observer)
        state = ceg.initialize_state()

        # Coverage should be zero
        assert float(state.nt) == pytest.approx(0.0, abs=0.01)

    def test_ceg_coverage_multiple_contracts(
        self, status_date, maturity_date, rf_observer, child_observer, covered_state
    ):
        """Test coverage calculation for multiple covered contracts."""
        # Register two contracts with different notionals
        state1 = ContractState(
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

        state2 = ContractState(
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

        child_observer.register_child("LOAN-001", state=state1)
        child_observer.register_child("LOAN-002", state=state2)

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

        ceg = CreditEnhancementGuaranteeContract(attrs, rf_observer, child_observer)
        state = ceg.initialize_state()

        # Coverage = 0.8 * (100000 + 50000)
        expected_coverage = 0.8 * 150000.0
        assert float(state.nt) == pytest.approx(expected_coverage, abs=0.01)


# ==================== Test: Event Schedule Generation ====================


class TestCEGEventSchedule:
    """Test CEG event schedule generation."""

    def test_ceg_generates_maturity_event(
        self, status_date, maturity_date, rf_observer, child_observer, covered_state
    ):
        """Test that CEG generates maturity event."""
        child_observer.register_child("LOAN-001", state=covered_state)

        attrs = ContractAttributes(
            contract_id="CEG-001",
            contract_type=ContractType.CEG,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=0.8,
            credit_event_type=ContractPerformance.DL,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001"}',
            currency="USD",
        )

        ceg = CreditEnhancementGuaranteeContract(attrs, rf_observer, child_observer)
        schedule = ceg.generate_event_schedule()

        # Should have MD event
        md_events = [e for e in schedule.events if e.event_type == EventType.MD]
        assert len(md_events) == 1
        assert md_events[0].event_time == maturity_date

    def test_ceg_generates_fee_payment_events(
        self, status_date, maturity_date, rf_observer, child_observer, covered_state
    ):
        """Test that CEG generates fee payment events if fees are specified."""
        child_observer.register_child("LOAN-001", state=covered_state)

        attrs = ContractAttributes(
            contract_id="CEG-001",
            contract_type=ContractType.CEG,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=0.8,
            credit_event_type=ContractPerformance.DL,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001"}',
            fee_rate=0.01,  # 1% annual fee
            fee_payment_cycle="P1Y",
            currency="USD",
        )

        ceg = CreditEnhancementGuaranteeContract(attrs, rf_observer, child_observer)
        schedule = ceg.generate_event_schedule()

        # Should have FP events (5 years = 5 annual payments)
        fp_events = [e for e in schedule.events if e.event_type == EventType.FP]
        assert len(fp_events) > 0  # At least one fee payment

    def test_ceg_with_analysis_dates(
        self, status_date, maturity_date, rf_observer, child_observer, covered_state
    ):
        """Test CEG with analysis dates."""
        child_observer.register_child("LOAN-001", state=covered_state)

        ad1 = ActusDateTime(2025, 1, 1, 0, 0, 0)
        ad2 = ActusDateTime(2026, 1, 1, 0, 0, 0)

        attrs = ContractAttributes(
            contract_id="CEG-001",
            contract_type=ContractType.CEG,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=0.8,
            credit_event_type=ContractPerformance.DL,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001"}',
            analysis_dates=[ad1, ad2],
            currency="USD",
        )

        ceg = CreditEnhancementGuaranteeContract(attrs, rf_observer, child_observer)
        schedule = ceg.generate_event_schedule()

        # Should have AD events
        ad_events = [e for e in schedule.events if e.event_type == EventType.AD]
        assert len(ad_events) == 2


# ==================== Test: State Initialization ====================


class TestCEGStateInitialization:
    """Test CEG state initialization."""

    def test_ceg_initialize_state(
        self, status_date, maturity_date, rf_observer, child_observer, covered_state
    ):
        """Test CEG state initialization."""
        child_observer.register_child("LOAN-001", state=covered_state)

        attrs = ContractAttributes(
            contract_id="CEG-001",
            contract_type=ContractType.CEG,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=0.8,
            credit_event_type=ContractPerformance.DL,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001"}',
            currency="USD",
        )

        ceg = CreditEnhancementGuaranteeContract(attrs, rf_observer, child_observer)
        state = ceg.initialize_state()

        assert state.tmd == maturity_date
        assert state.sd == status_date
        assert float(state.nt) > 0  # Coverage amount calculated
        assert state.prf == ContractPerformance.PF


# ==================== Test: Edge Cases ====================


class TestCEGEdgeCases:
    """Test CEG edge cases and error handling."""

    def test_ceg_invalid_json_structure(
        self, status_date, maturity_date, rf_observer, child_observer
    ):
        """Test that invalid JSON in contract_structure raises error."""
        attrs = ContractAttributes(
            contract_id="CEG-001",
            contract_type=ContractType.CEG,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=0.8,
            credit_event_type=ContractPerformance.DL,
            credit_enhancement_guarantee_extent="NO",
            contract_structure="not valid json",
            currency="USD",
        )

        with pytest.raises(ValueError, match="valid JSON"):
            CreditEnhancementGuaranteeContract(attrs, rf_observer, child_observer)

    def test_ceg_non_dict_json_structure(
        self, status_date, maturity_date, rf_observer, child_observer
    ):
        """Test that non-dict JSON in contract_structure raises error."""
        attrs = ContractAttributes(
            contract_id="CEG-001",
            contract_type=ContractType.CEG,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=0.8,
            credit_event_type=ContractPerformance.DL,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='["array", "not", "dict"]',  # Array instead of dict
            currency="USD",
        )

        with pytest.raises(ValueError, match="must be a JSON object"):
            CreditEnhancementGuaranteeContract(attrs, rf_observer, child_observer)

    def test_ceg_wrong_contract_type(self, status_date, maturity_date, rf_observer, child_observer):
        """Test that wrong contract type raises error."""
        attrs = ContractAttributes(
            contract_id="CEG-001",
            contract_type=ContractType.PAM,  # Wrong type
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=0.8,
            credit_event_type=ContractPerformance.DL,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001"}',
            currency="USD",
        )

        with pytest.raises(ValueError, match="Expected contract_type=CEG"):
            CreditEnhancementGuaranteeContract(attrs, rf_observer, child_observer)

    def test_ceg_with_termination_date(
        self, status_date, maturity_date, rf_observer, child_observer, covered_state
    ):
        """Test CEG with termination date."""
        child_observer.register_child("LOAN-001", state=covered_state)

        termination = ActusDateTime(2025, 6, 1, 0, 0, 0)

        attrs = ContractAttributes(
            contract_id="CEG-001",
            contract_type=ContractType.CEG,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            termination_date=termination,
            coverage=0.8,
            credit_event_type=ContractPerformance.DL,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001"}',
            currency="USD",
        )

        ceg = CreditEnhancementGuaranteeContract(attrs, rf_observer, child_observer)
        schedule = ceg.generate_event_schedule()

        # Should have TD event
        td_events = [e for e in schedule.events if e.event_type == EventType.TD]
        assert len(td_events) == 1
        assert td_events[0].event_time == termination

    def test_ceg_zero_coverage(
        self, status_date, maturity_date, rf_observer, child_observer, covered_state
    ):
        """Test CEG with zero coverage amount."""
        child_observer.register_child("LOAN-001", state=covered_state)

        attrs = ContractAttributes(
            contract_id="CEG-001",
            contract_type=ContractType.CEG,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=0.0,  # Zero coverage
            credit_event_type=ContractPerformance.DL,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001"}',
            currency="USD",
        )

        ceg = CreditEnhancementGuaranteeContract(attrs, rf_observer, child_observer)
        state = ceg.initialize_state()

        # Coverage should be zero
        assert float(state.nt) == pytest.approx(0.0, abs=0.01)

    def test_ceg_full_coverage(
        self, status_date, maturity_date, rf_observer, child_observer, covered_state
    ):
        """Test CEG with 100% coverage."""
        child_observer.register_child("LOAN-001", state=covered_state)

        attrs = ContractAttributes(
            contract_id="CEG-001",
            contract_type=ContractType.CEG,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=1.0,  # 100% coverage
            credit_event_type=ContractPerformance.DL,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001"}',
            currency="USD",
        )

        ceg = CreditEnhancementGuaranteeContract(attrs, rf_observer, child_observer)
        state = ceg.initialize_state()

        # Coverage should equal full notional
        assert float(state.nt) == pytest.approx(100000.0, abs=0.01)


# ==================== Test: Factory Creation ====================


class TestCEGFactory:
    """Test CEG creation through factory."""

    def test_ceg_factory_creation(
        self, status_date, maturity_date, rf_observer, child_observer, covered_state
    ):
        """Test creating CEG through contract factory."""
        from jactus.contracts import create_contract

        child_observer.register_child("LOAN-001", state=covered_state)

        attrs = ContractAttributes(
            contract_id="CEG-001",
            contract_type=ContractType.CEG,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=0.8,
            credit_event_type=ContractPerformance.DL,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001"}',
            currency="USD",
        )

        ceg = create_contract(attrs, rf_observer, child_observer)

        assert isinstance(ceg, CreditEnhancementGuaranteeContract)
        assert ceg.attributes.contract_type == ContractType.CEG


# ==================== Test: Payoff and State Functions ====================


class TestCEGPayoffAndState:
    """Test CEG payoff and state transition functions."""

    def test_ceg_payoff_function(
        self, status_date, maturity_date, rf_observer, child_observer, covered_state
    ):
        """Test getting payoff function."""
        child_observer.register_child("LOAN-001", state=covered_state)

        attrs = ContractAttributes(
            contract_id="CEG-001",
            contract_type=ContractType.CEG,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=0.8,
            credit_event_type=ContractPerformance.DL,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001"}',
            currency="USD",
        )

        ceg = CreditEnhancementGuaranteeContract(attrs, rf_observer, child_observer)
        payoff_fn = ceg.get_payoff_function(EventType.FP)

        assert payoff_fn is not None
        # Test payoff calculation
        state = ceg.initialize_state()
        payoff = payoff_fn.calculate_payoff(EventType.FP, state, attrs, maturity_date, rf_observer)
        assert float(payoff) == 0.0  # Returns zero (actual payoffs in events)

    def test_ceg_state_transition_function(
        self, status_date, maturity_date, rf_observer, child_observer, covered_state
    ):
        """Test getting state transition function."""
        child_observer.register_child("LOAN-001", state=covered_state)

        attrs = ContractAttributes(
            contract_id="CEG-001",
            contract_type=ContractType.CEG,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=0.8,
            credit_event_type=ContractPerformance.DL,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001"}',
            currency="USD",
        )

        ceg = CreditEnhancementGuaranteeContract(attrs, rf_observer, child_observer)
        stf_fn = ceg.get_state_transition_function(EventType.FP)

        assert stf_fn is not None
        # Test state transition
        state = ceg.initialize_state()
        new_state = stf_fn.transition_state(EventType.FP, state, attrs, maturity_date, rf_observer)
        assert new_state.sd == maturity_date

    def test_ceg_credit_event_detection(
        self, status_date, maturity_date, rf_observer, child_observer
    ):
        """Test credit event detection."""
        # Create state with default performance
        default_state = ContractState(
            tmd=maturity_date,
            sd=status_date,
            nt=jnp.array(100000.0, dtype=jnp.float32),
            ipnr=jnp.array(0.05, dtype=jnp.float32),
            ipac=jnp.array(5000.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
            prf=ContractPerformance.DL,  # Default
        )

        child_observer.register_child("LOAN-001", state=default_state)

        attrs = ContractAttributes(
            contract_id="CEG-001",
            contract_type=ContractType.CEG,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=0.8,
            credit_event_type=ContractPerformance.DL,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001"}',
            currency="USD",
        )

        ceg = CreditEnhancementGuaranteeContract(attrs, rf_observer, child_observer)

        # Should detect credit event
        has_event = ceg._detect_credit_event(status_date)
        assert has_event is True

    def test_ceg_no_credit_event(
        self, status_date, maturity_date, rf_observer, child_observer, covered_state
    ):
        """Test no credit event detected when performance is performing."""
        child_observer.register_child("LOAN-001", state=covered_state)

        attrs = ContractAttributes(
            contract_id="CEG-001",
            contract_type=ContractType.CEG,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=0.8,
            credit_event_type=ContractPerformance.DL,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContract": "LOAN-001"}',
            currency="USD",
        )

        ceg = CreditEnhancementGuaranteeContract(attrs, rf_observer, child_observer)

        # Should not detect credit event (state is PF, looking for DL)
        has_event = ceg._detect_credit_event(status_date)
        assert has_event is False

    def test_ceg_invalid_covered_contracts_type(
        self, status_date, maturity_date, rf_observer, child_observer
    ):
        """Test error handling for invalid CoveredContracts type."""
        attrs = ContractAttributes(
            contract_id="CEG-001",
            contract_type=ContractType.CEG,
            contract_role=ContractRole.RPA,
            status_date=status_date,
            maturity_date=maturity_date,
            coverage=0.8,
            credit_event_type=ContractPerformance.DL,
            credit_enhancement_guarantee_extent="NO",
            contract_structure='{"CoveredContracts": 123}',  # Invalid type (number)
            currency="USD",
        )

        ceg = CreditEnhancementGuaranteeContract(attrs, rf_observer, child_observer)

        # Should raise error when trying to get covered contract IDs
        with pytest.raises(ValueError, match="must be list or string"):
            ceg._get_covered_contract_ids()
