"""Unit tests for FXOUT (Foreign Exchange Outright) contract."""

import pytest

from jactus.contracts import create_contract
from jactus.contracts.fxout import (
    FXOutrightContract,
)
from jactus.core import (
    ActusDateTime,
    ContractAttributes,
    ContractRole,
    ContractType,
    EventType,
)
from jactus.observers import ConstantRiskFactorObserver


class TestFXOutrightInitialization:
    """Test FXOUT contract initialization and validation."""

    def test_fxout_initialization_success(self):
        """Test successful FXOUT initialization with all required attributes."""
        attrs = ContractAttributes(
            contract_id="FXOUT-001",
            contract_type=ContractType.FXOUT,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 7, 1, 0, 0, 0),
            currency="EUR",
            currency_2="USD",
            notional_principal=100000.0,
            notional_principal_2=110000.0,
            delivery_settlement="D",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=1.10)
        contract = FXOutrightContract(attrs, rf_obs)

        assert contract.attributes.contract_type == ContractType.FXOUT
        assert contract.attributes.currency == "EUR"
        assert contract.attributes.currency_2 == "USD"
        assert contract.attributes.notional_principal == 100000.0
        assert contract.attributes.notional_principal_2 == 110000.0

    def test_fxout_validation_contract_type(self):
        """Test that FXOUT rejects non-FXOUT contract type."""
        attrs = ContractAttributes(
            contract_id="PAM-001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 7, 1, 0, 0, 0),
            currency="EUR",
            currency_2="USD",
            notional_principal=100000.0,
            notional_principal_2=110000.0,
            delivery_settlement="D",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=1.10)

        with pytest.raises(ValueError, match="Contract type must be FXOUT"):
            FXOutrightContract(attrs, rf_obs)

    def test_fxout_validation_nt_required(self):
        """Test that NT (notional_principal) is required."""
        attrs = ContractAttributes(
            contract_id="FXOUT-001",
            contract_type=ContractType.FXOUT,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 7, 1, 0, 0, 0),
            currency="EUR",
            currency_2="USD",
            # notional_principal missing
            notional_principal_2=110000.0,
            delivery_settlement="D",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=1.10)

        with pytest.raises(ValueError, match="notional_principal .* required"):
            FXOutrightContract(attrs, rf_obs)

    def test_fxout_validation_nt2_required(self):
        """Test that NT2 (notional_principal_2) is required."""
        attrs = ContractAttributes(
            contract_id="FXOUT-001",
            contract_type=ContractType.FXOUT,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 7, 1, 0, 0, 0),
            currency="EUR",
            currency_2="USD",
            notional_principal=100000.0,
            # notional_principal_2 missing
            delivery_settlement="D",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=1.10)

        with pytest.raises(ValueError, match="notional_principal_2 .* required"):
            FXOutrightContract(attrs, rf_obs)

    def test_fxout_validation_currency_2_required(self):
        """Test that currency_2 is required."""
        attrs = ContractAttributes(
            contract_id="FXOUT-001",
            contract_type=ContractType.FXOUT,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 7, 1, 0, 0, 0),
            currency="EUR",
            # currency_2 missing
            notional_principal=100000.0,
            notional_principal_2=110000.0,
            delivery_settlement="D",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=1.10)

        with pytest.raises(ValueError, match="currency_2 .* required"):
            FXOutrightContract(attrs, rf_obs)

    def test_fxout_validation_currencies_different(self):
        """Test that CUR and CUR2 must be different."""
        attrs = ContractAttributes(
            contract_id="FXOUT-001",
            contract_type=ContractType.FXOUT,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 7, 1, 0, 0, 0),
            currency="EUR",
            currency_2="EUR",  # Same as currency
            notional_principal=100000.0,
            notional_principal_2=110000.0,
            delivery_settlement="D",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=1.10)

        with pytest.raises(ValueError, match="Currencies must be different"):
            FXOutrightContract(attrs, rf_obs)

    def test_fxout_validation_ds_required(self):
        """Test that DS (delivery_settlement) is required."""
        attrs = ContractAttributes(
            contract_id="FXOUT-001",
            contract_type=ContractType.FXOUT,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 7, 1, 0, 0, 0),
            currency="EUR",
            currency_2="USD",
            notional_principal=100000.0,
            notional_principal_2=110000.0,
            # delivery_settlement missing
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=1.10)

        with pytest.raises(ValueError, match="delivery_settlement .* required"):
            FXOutrightContract(attrs, rf_obs)

    def test_fxout_validation_ds_valid_values(self):
        """Test that DS must be 'D' or 'S'."""
        attrs = ContractAttributes(
            contract_id="FXOUT-001",
            contract_type=ContractType.FXOUT,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 7, 1, 0, 0, 0),
            currency="EUR",
            currency_2="USD",
            notional_principal=100000.0,
            notional_principal_2=110000.0,
            delivery_settlement="X",  # Invalid value
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=1.10)

        with pytest.raises(ValueError, match="must be 'D' or 'S'"):
            FXOutrightContract(attrs, rf_obs)

    def test_fxout_validation_maturity_required(self):
        """Test that maturity_date or settlement_date is required."""
        attrs = ContractAttributes(
            contract_id="FXOUT-001",
            contract_type=ContractType.FXOUT,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            # maturity_date and settlement_date both missing
            currency="EUR",
            currency_2="USD",
            notional_principal=100000.0,
            notional_principal_2=110000.0,
            delivery_settlement="D",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=1.10)

        with pytest.raises(ValueError, match="maturity_date .* or settlement_date .* required"):
            FXOutrightContract(attrs, rf_obs)


class TestFXOutrightEventSchedule:
    """Test FXOUT event schedule generation."""

    def test_fxout_event_schedule_delivery_mode(self):
        """Test event schedule generation for delivery (gross) settlement mode."""
        attrs = ContractAttributes(
            contract_id="FXOUT-001",
            contract_type=ContractType.FXOUT,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 7, 1, 0, 0, 0),
            currency="EUR",
            currency_2="USD",
            notional_principal=100000.0,
            notional_principal_2=110000.0,
            delivery_settlement="D",  # Gross settlement (no SP)
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=1.10)
        contract = FXOutrightContract(attrs, rf_obs)

        schedule = contract.generate_event_schedule()

        # Should have 2 MD events for gross settlement (one per currency leg)
        md_events = [e for e in schedule.events if e.event_type == EventType.MD]
        assert len(md_events) == 2
        assert md_events[0].event_time == ActusDateTime(2024, 7, 1, 0, 0, 0)
        assert md_events[1].event_time == ActusDateTime(2024, 7, 1, 0, 0, 0)
        currencies = {md_events[0].currency, md_events[1].currency}
        assert currencies == {"EUR", "USD"}

    def test_fxout_event_schedule_dual_mode(self):
        """Test event schedule generation for dual (gross) settlement mode."""
        attrs = ContractAttributes(
            contract_id="FXOUT-001",
            contract_type=ContractType.FXOUT,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 7, 1, 0, 0, 0),
            currency="EUR",
            currency_2="USD",
            notional_principal=100000.0,
            notional_principal_2=110000.0,
            delivery_settlement="S",  # Dual mode
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=1.10)
        contract = FXOutrightContract(attrs, rf_obs)

        schedule = contract.generate_event_schedule()

        # Should have 2 MD events for dual mode (one per currency)
        md_events = [e for e in schedule.events if e.event_type == EventType.MD]
        assert len(md_events) == 2
        assert md_events[0].event_time == ActusDateTime(2024, 7, 1, 0, 0, 0)
        assert md_events[1].event_time == ActusDateTime(2024, 7, 1, 0, 0, 0)
        # One in EUR, one in USD
        currencies = {md_events[0].currency, md_events[1].currency}
        assert currencies == {"EUR", "USD"}

    def test_fxout_event_schedule_with_purchase(self):
        """Test event schedule includes purchase date if defined."""
        attrs = ContractAttributes(
            contract_id="FXOUT-001",
            contract_type=ContractType.FXOUT,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            purchase_date=ActusDateTime(2024, 2, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 7, 1, 0, 0, 0),
            currency="EUR",
            currency_2="USD",
            notional_principal=100000.0,
            notional_principal_2=110000.0,
            delivery_settlement="D",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=1.10)
        contract = FXOutrightContract(attrs, rf_obs)

        schedule = contract.generate_event_schedule()

        # Should have PRD event
        prd_events = [e for e in schedule.events if e.event_type == EventType.PRD]
        assert len(prd_events) == 1
        assert prd_events[0].event_time == ActusDateTime(2024, 2, 1, 0, 0, 0)

    def test_fxout_event_schedule_with_termination(self):
        """Test event schedule includes termination date if defined."""
        attrs = ContractAttributes(
            contract_id="FXOUT-001",
            contract_type=ContractType.FXOUT,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            termination_date=ActusDateTime(2024, 6, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 7, 1, 0, 0, 0),
            currency="EUR",
            currency_2="USD",
            notional_principal=100000.0,
            notional_principal_2=110000.0,
            delivery_settlement="D",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=1.10)
        contract = FXOutrightContract(attrs, rf_obs)

        schedule = contract.generate_event_schedule()

        # Should have TD event
        td_events = [e for e in schedule.events if e.event_type == EventType.TD]
        assert len(td_events) == 1
        assert td_events[0].event_time == ActusDateTime(2024, 6, 1, 0, 0, 0)


class TestFXOutrightStateInitialization:
    """Test FXOUT state initialization."""

    def test_fxout_state_initialization(self):
        """Test state initialization with maturity date."""
        attrs = ContractAttributes(
            contract_id="FXOUT-001",
            contract_type=ContractType.FXOUT,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 7, 1, 0, 0, 0),
            currency="EUR",
            currency_2="USD",
            notional_principal=100000.0,
            notional_principal_2=110000.0,
            delivery_settlement="D",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=1.10)
        contract = FXOutrightContract(attrs, rf_obs)

        state = contract.initialize_state()

        assert state.tmd == ActusDateTime(2024, 7, 1, 0, 0, 0)
        assert state.sd == ActusDateTime(2024, 1, 1, 0, 0, 0)

    def test_fxout_state_initialization_with_settlement_date(self):
        """Test state initialization prefers settlement_date over maturity_date."""
        attrs = ContractAttributes(
            contract_id="FXOUT-001",
            contract_type=ContractType.FXOUT,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 7, 1, 0, 0, 0),
            settlement_date=ActusDateTime(2024, 6, 15, 0, 0, 0),
            currency="EUR",
            currency_2="USD",
            notional_principal=100000.0,
            notional_principal_2=110000.0,
            delivery_settlement="D",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=1.10)
        contract = FXOutrightContract(attrs, rf_obs)

        state = contract.initialize_state()

        # Should use settlement_date
        assert state.tmd == ActusDateTime(2024, 6, 15, 0, 0, 0)


class TestFXOutrightPayoffs:
    """Test FXOUT payoff calculations."""

    def test_fxout_payoff_std_delivery_mode_profit(self):
        """Test MD payoffs for gross settlement (DS='D', no settlement period)."""
        attrs = ContractAttributes(
            contract_id="FXOUT-001",
            contract_type=ContractType.FXOUT,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 7, 1, 0, 0, 0),
            currency="EUR",
            currency_2="USD",
            notional_principal=100000.0,  # Buy 100k EUR
            notional_principal_2=110000.0,  # Sell 110k USD (forward rate = 1.10)
            delivery_settlement="D",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=1.12)
        contract = FXOutrightContract(attrs, rf_obs)

        result = contract.simulate()

        # Gross settlement: 2 MD events (EUR leg + USD leg)
        md_events = [e for e in result.events if e.event_type == EventType.MD]
        assert len(md_events) == 2

        # EUR leg: role_sign * NT = 1 * 100000 = 100000
        eur_event = [e for e in md_events if e.currency == "EUR"][0]
        assert float(eur_event.payoff) == pytest.approx(100000.0, abs=1.0)

        # USD leg: -role_sign * NT2 = -1 * 110000 = -110000
        usd_event = [e for e in md_events if e.currency == "USD"][0]
        assert float(usd_event.payoff) == pytest.approx(-110000.0, abs=1.0)

    def test_fxout_payoff_prd(self):
        """Test PRD (purchase) payoff."""
        attrs = ContractAttributes(
            contract_id="FXOUT-001",
            contract_type=ContractType.FXOUT,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            purchase_date=ActusDateTime(2024, 2, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 7, 1, 0, 0, 0),
            price_at_purchase_date=1000.0,
            currency="EUR",
            currency_2="USD",
            notional_principal=100000.0,
            notional_principal_2=110000.0,
            delivery_settlement="D",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=1.10)
        contract = FXOutrightContract(attrs, rf_obs)

        result = contract.simulate()

        # Find PRD event
        prd_events = [e for e in result.events if e.event_type == EventType.PRD]
        assert len(prd_events) == 1

        # Payoff should be negative (paying premium)
        prd_payoff = float(prd_events[0].payoff)
        assert prd_payoff == pytest.approx(-1000.0, abs=0.01)


class TestFXOutrightSimulation:
    """Test complete FXOUT simulation."""

    def test_fxout_simulation_delivery_mode(self):
        """Test complete simulation in delivery mode."""
        attrs = ContractAttributes(
            contract_id="FXOUT-001",
            contract_type=ContractType.FXOUT,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 7, 1, 0, 0, 0),
            currency="EUR",
            currency_2="USD",
            notional_principal=100000.0,
            notional_principal_2=110000.0,
            delivery_settlement="D",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=1.10)
        contract = FXOutrightContract(attrs, rf_obs)

        result = contract.simulate()

        assert len(result.events) >= 1  # At least STD event
        assert result.initial_state is not None
        assert result.final_state is not None

    def test_fxout_simulation_dual_mode(self):
        """Test complete simulation in dual settlement mode."""
        attrs = ContractAttributes(
            contract_id="FXOUT-001",
            contract_type=ContractType.FXOUT,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 7, 1, 0, 0, 0),
            currency="EUR",
            currency_2="USD",
            notional_principal=100000.0,
            notional_principal_2=110000.0,
            delivery_settlement="S",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=1.10)
        contract = FXOutrightContract(attrs, rf_obs)

        result = contract.simulate()

        # Should have 2 MD events (one per currency)
        md_events = [e for e in result.events if e.event_type == EventType.MD]
        assert len(md_events) == 2

    def test_fxout_factory_creation(self):
        """Test creating FXOUT via factory."""
        attrs = ContractAttributes(
            contract_id="FXOUT-001",
            contract_type=ContractType.FXOUT,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 7, 1, 0, 0, 0),
            currency="EUR",
            currency_2="USD",
            notional_principal=100000.0,
            notional_principal_2=110000.0,
            delivery_settlement="D",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=1.10)
        contract = create_contract(attrs, rf_obs)

        assert isinstance(contract, FXOutrightContract)
        result = contract.simulate()
        assert len(result.events) >= 1
