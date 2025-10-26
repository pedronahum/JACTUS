"""Unit tests for Option Contract (OPTNS) implementation."""

import pytest

from jactus.contracts import create_contract
from jactus.contracts.optns import OptionContract
from jactus.core import ActusDateTime, ContractAttributes, ContractRole, ContractType, EventType
from jactus.observers import ConstantRiskFactorObserver


class TestOptionInitialization:
    """Test OPTNS contract initialization and validation."""

    def test_optns_initialization_success_call(self):
        """Test successful OPTNS initialization for call option."""
        attrs = ContractAttributes(
            contract_id="OPT-CALL-001",
            contract_type=ContractType.OPTNS,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            currency="USD",
            notional_principal=100.0,  # 100 shares
            option_type="C",  # Call option
            option_strike_1=100.0,  # Strike price $100
            option_exercise_type="E",  # European
            price_at_purchase_date=5.0,  # Premium $5/share
            contract_structure="AAPL",  # Underlier
        )
        rf_obs = ConstantRiskFactorObserver(constant_value=110.0)  # Stock at $110

        contract = OptionContract(
            attributes=attrs,
            risk_factor_observer=rf_obs,
        )

        assert contract is not None
        assert contract.attributes.option_type == "C"
        assert contract.attributes.option_strike_1 == 100.0

    def test_optns_initialization_success_put(self):
        """Test successful OPTNS initialization for put option."""
        attrs = ContractAttributes(
            contract_id="OPT-PUT-001",
            contract_type=ContractType.OPTNS,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            currency="USD",
            notional_principal=100.0,
            option_type="P",  # Put option
            option_strike_1=100.0,
            option_exercise_type="E",
            price_at_purchase_date=4.0,
            contract_structure="AAPL",
        )
        rf_obs = ConstantRiskFactorObserver(constant_value=90.0)  # Stock at $90

        contract = OptionContract(
            attributes=attrs,
            risk_factor_observer=rf_obs,
        )

        assert contract is not None
        assert contract.attributes.option_type == "P"

    def test_optns_validation_contract_type(self):
        """Test that contract_type must be OPTNS."""
        attrs = ContractAttributes(
            contract_id="WRONG-001",
            contract_type=ContractType.PAM,  # Wrong type
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            currency="USD",
            option_type="C",
            option_strike_1=100.0,
            option_exercise_type="E",
            contract_structure="AAPL",
        )
        rf_obs = ConstantRiskFactorObserver(constant_value=110.0)

        with pytest.raises(ValueError, match="Expected contract_type=OPTNS"):
            OptionContract(attributes=attrs, risk_factor_observer=rf_obs)

    def test_optns_validation_option_type_required(self):
        """Test that option_type is required."""
        attrs = ContractAttributes(
            contract_id="OPT-001",
            contract_type=ContractType.OPTNS,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            currency="USD",
            option_type=None,  # Missing
            option_strike_1=100.0,
            option_exercise_type="E",
            contract_structure="AAPL",
        )
        rf_obs = ConstantRiskFactorObserver(constant_value=110.0)

        with pytest.raises(ValueError, match="option_type must be"):
            OptionContract(attributes=attrs, risk_factor_observer=rf_obs)

    def test_optns_validation_option_type_valid_values(self):
        """Test that option_type must be 'C', 'P', or 'CP'."""
        attrs = ContractAttributes(
            contract_id="OPT-001",
            contract_type=ContractType.OPTNS,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            currency="USD",
            option_type="X",  # Invalid
            option_strike_1=100.0,
            option_exercise_type="E",
            contract_structure="AAPL",
        )
        rf_obs = ConstantRiskFactorObserver(constant_value=110.0)

        with pytest.raises(ValueError, match="option_type must be"):
            OptionContract(attributes=attrs, risk_factor_observer=rf_obs)

    def test_optns_validation_strike_1_required(self):
        """Test that option_strike_1 is required."""
        attrs = ContractAttributes(
            contract_id="OPT-001",
            contract_type=ContractType.OPTNS,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            currency="USD",
            option_type="C",
            option_strike_1=None,  # Missing
            option_exercise_type="E",
            contract_structure="AAPL",
        )
        rf_obs = ConstantRiskFactorObserver(constant_value=110.0)

        with pytest.raises(ValueError, match="option_strike_1.*required"):
            OptionContract(attributes=attrs, risk_factor_observer=rf_obs)

    def test_optns_validation_strike_2_required_for_collar(self):
        """Test that option_strike_2 is required for collar options."""
        attrs = ContractAttributes(
            contract_id="OPT-COLLAR-001",
            contract_type=ContractType.OPTNS,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            currency="USD",
            option_type="CP",  # Collar
            option_strike_1=100.0,
            option_strike_2=None,  # Missing (required for collar)
            option_exercise_type="E",
            contract_structure="AAPL",
        )
        rf_obs = ConstantRiskFactorObserver(constant_value=110.0)

        with pytest.raises(ValueError, match="option_strike_2.*required for collar"):
            OptionContract(attributes=attrs, risk_factor_observer=rf_obs)

    def test_optns_validation_exercise_type_required(self):
        """Test that option_exercise_type is required."""
        attrs = ContractAttributes(
            contract_id="OPT-001",
            contract_type=ContractType.OPTNS,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            currency="USD",
            option_type="C",
            option_strike_1=100.0,
            option_exercise_type=None,  # Missing
            contract_structure="AAPL",
        )
        rf_obs = ConstantRiskFactorObserver(constant_value=110.0)

        with pytest.raises(ValueError, match="option_exercise_type must be"):
            OptionContract(attributes=attrs, risk_factor_observer=rf_obs)

    def test_optns_validation_underlier_required(self):
        """Test that contract_structure (underlier) is required."""
        attrs = ContractAttributes(
            contract_id="OPT-001",
            contract_type=ContractType.OPTNS,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            currency="USD",
            option_type="C",
            option_strike_1=100.0,
            option_exercise_type="E",
            contract_structure=None,  # Missing
        )
        rf_obs = ConstantRiskFactorObserver(constant_value=110.0)

        with pytest.raises(ValueError, match="contract_structure.*required"):
            OptionContract(attributes=attrs, risk_factor_observer=rf_obs)

    def test_optns_validation_maturity_required(self):
        """Test that maturity_date is required."""
        attrs = ContractAttributes(
            contract_id="OPT-001",
            contract_type=ContractType.OPTNS,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=None,  # Missing
            currency="USD",
            option_type="C",
            option_strike_1=100.0,
            option_exercise_type="E",
            contract_structure="AAPL",
        )
        rf_obs = ConstantRiskFactorObserver(constant_value=110.0)

        with pytest.raises(ValueError, match="maturity_date.*required"):
            OptionContract(attributes=attrs, risk_factor_observer=rf_obs)


class TestOptionEventSchedule:
    """Test OPTNS event schedule generation."""

    def test_optns_event_schedule_european(self):
        """Test event schedule for European option."""
        attrs = ContractAttributes(
            contract_id="OPT-001",
            contract_type=ContractType.OPTNS,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            purchase_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            currency="USD",
            option_type="C",
            option_strike_1=100.0,
            option_exercise_type="E",
            price_at_purchase_date=5.0,
            contract_structure="AAPL",
        )
        rf_obs = ConstantRiskFactorObserver(constant_value=110.0)
        contract = OptionContract(attributes=attrs, risk_factor_observer=rf_obs)

        schedule = contract.generate_event_schedule()
        event_types = [e.event_type for e in schedule.events]

        assert EventType.PRD in event_types  # Purchase
        assert EventType.MD in event_types  # Maturity
        assert EventType.STD in event_types  # Settlement

    def test_optns_event_schedule_with_termination(self):
        """Test event schedule with early termination."""
        attrs = ContractAttributes(
            contract_id="OPT-001",
            contract_type=ContractType.OPTNS,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            termination_date=ActusDateTime(2024, 6, 30, 0, 0, 0),
            currency="USD",
            option_type="P",
            option_strike_1=100.0,
            option_exercise_type="E",
            contract_structure="AAPL",
        )
        rf_obs = ConstantRiskFactorObserver(constant_value=90.0)
        contract = OptionContract(attributes=attrs, risk_factor_observer=rf_obs)

        schedule = contract.generate_event_schedule()
        event_types = [e.event_type for e in schedule.events]

        assert EventType.TD in event_types  # Termination


class TestOptionStateInitialization:
    """Test OPTNS state initialization."""

    def test_optns_state_initialization(self):
        """Test that OPTNS initializes state with xa=0."""
        attrs = ContractAttributes(
            contract_id="OPT-001",
            contract_type=ContractType.OPTNS,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            currency="USD",
            option_type="C",
            option_strike_1=100.0,
            option_exercise_type="E",
            contract_structure="AAPL",
        )
        rf_obs = ConstantRiskFactorObserver(constant_value=110.0)
        contract = OptionContract(attributes=attrs, risk_factor_observer=rf_obs)

        state = contract.initialize_state()

        assert state.sd == attrs.status_date
        assert state.tmd == attrs.maturity_date
        assert state.xa is not None
        assert float(state.xa) == 0.0  # No exercise yet


class TestOptionPayoffs:
    """Test OPTNS payoff calculations."""

    def test_optns_payoff_prd_premium(self):
        """Test POF_PRD (premium payment)."""
        attrs = ContractAttributes(
            contract_id="OPT-001",
            contract_type=ContractType.OPTNS,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            purchase_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100.0,  # 100 shares
            option_type="C",
            option_strike_1=100.0,
            option_exercise_type="E",
            price_at_purchase_date=5.0,  # $5 premium per share
            contract_structure="AAPL",
        )
        rf_obs = ConstantRiskFactorObserver(constant_value=110.0)
        contract = OptionContract(attributes=attrs, risk_factor_observer=rf_obs)

        result = contract.simulate()

        # Find PRD event
        prd_event = next(e for e in result.events if e.event_type == EventType.PRD)

        # Premium = -5 * 100 = -500 (negative for buyer)
        assert prd_event.payoff == pytest.approx(-500.0, abs=0.01)

    def test_optns_payoff_std_call_itm(self):
        """Test POF_STD for in-the-money call option."""
        attrs = ContractAttributes(
            contract_id="OPT-CALL-ITM",
            contract_type=ContractType.OPTNS,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            currency="USD",
            notional_principal=100.0,  # 100 shares
            option_type="C",  # Call
            option_strike_1=100.0,  # Strike $100
            option_exercise_type="E",
            contract_structure="AAPL",
        )
        rf_obs = ConstantRiskFactorObserver(constant_value=110.0)  # Stock at $110
        contract = OptionContract(attributes=attrs, risk_factor_observer=rf_obs)

        result = contract.simulate()

        # Find STD event
        std_event = next(e for e in result.events if e.event_type == EventType.STD)

        # Exercise amount: max(110 - 100, 0) * 100 = 1000
        assert std_event.payoff == pytest.approx(1000.0, abs=0.01)

    def test_optns_payoff_std_put_itm(self):
        """Test POF_STD for in-the-money put option."""
        attrs = ContractAttributes(
            contract_id="OPT-PUT-ITM",
            contract_type=ContractType.OPTNS,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            currency="USD",
            notional_principal=100.0,  # 100 shares
            option_type="P",  # Put
            option_strike_1=100.0,  # Strike $100
            option_exercise_type="E",
            contract_structure="AAPL",
        )
        rf_obs = ConstantRiskFactorObserver(constant_value=90.0)  # Stock at $90
        contract = OptionContract(attributes=attrs, risk_factor_observer=rf_obs)

        result = contract.simulate()

        # Find STD event
        std_event = next(e for e in result.events if e.event_type == EventType.STD)

        # Exercise amount: max(100 - 90, 0) * 100 = 1000
        assert std_event.payoff == pytest.approx(1000.0, abs=0.01)

    def test_optns_payoff_std_call_otm(self):
        """Test POF_STD for out-of-the-money call option (expires worthless)."""
        attrs = ContractAttributes(
            contract_id="OPT-CALL-OTM",
            contract_type=ContractType.OPTNS,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            currency="USD",
            notional_principal=100.0,
            option_type="C",  # Call
            option_strike_1=100.0,  # Strike $100
            option_exercise_type="E",
            contract_structure="AAPL",
        )
        rf_obs = ConstantRiskFactorObserver(constant_value=90.0)  # Stock at $90 (OTM)
        contract = OptionContract(attributes=attrs, risk_factor_observer=rf_obs)

        result = contract.simulate()

        # Find STD event
        std_event = next(e for e in result.events if e.event_type == EventType.STD)

        # Exercise amount: max(90 - 100, 0) * 100 = 0
        assert std_event.payoff == pytest.approx(0.0, abs=0.01)


class TestOptionStateTransitions:
    """Test OPTNS state transition functions."""

    def test_optns_stf_md_european_call_itm(self):
        """Test STF_MD for European call ITM (automatic exercise)."""
        attrs = ContractAttributes(
            contract_id="OPT-001",
            contract_type=ContractType.OPTNS,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            currency="USD",
            notional_principal=100.0,
            option_type="C",  # Call
            option_strike_1=100.0,
            option_exercise_type="E",  # European
            contract_structure="AAPL",
        )
        rf_obs = ConstantRiskFactorObserver(constant_value=110.0)  # ITM
        contract = OptionContract(attributes=attrs, risk_factor_observer=rf_obs)

        result = contract.simulate()

        # After MD event, state should have xa=10 (intrinsic value per share)
        md_state = next(
            s for e, s in zip(result.events, result.states, strict=False) if e.event_type == EventType.MD
        )

        assert md_state.xa is not None
        assert float(md_state.xa) == pytest.approx(10.0, abs=0.01)  # max(110 - 100, 0)

    def test_optns_stf_md_european_call_otm(self):
        """Test STF_MD for European call OTM (no exercise)."""
        attrs = ContractAttributes(
            contract_id="OPT-001",
            contract_type=ContractType.OPTNS,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            currency="USD",
            notional_principal=100.0,
            option_type="C",
            option_strike_1=100.0,
            option_exercise_type="E",
            contract_structure="AAPL",
        )
        rf_obs = ConstantRiskFactorObserver(constant_value=90.0)  # OTM
        contract = OptionContract(attributes=attrs, risk_factor_observer=rf_obs)

        result = contract.simulate()

        # After MD event, state should have xa=0 (no intrinsic value)
        md_state = next(
            s for e, s in zip(result.events, result.states, strict=False) if e.event_type == EventType.MD
        )

        assert md_state.xa is not None
        assert float(md_state.xa) == pytest.approx(0.0, abs=0.01)


class TestOptionSimulation:
    """Test complete OPTNS simulation."""

    def test_optns_simulation_european_call_itm(self):
        """Test complete simulation of European call option in-the-money."""
        attrs = ContractAttributes(
            contract_id="OPT-CALL-ITM",
            contract_type=ContractType.OPTNS,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            purchase_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100.0,  # 100 shares
            option_type="C",
            option_strike_1=100.0,  # Strike $100
            option_exercise_type="E",
            price_at_purchase_date=5.0,  # Premium $5/share
            contract_structure="AAPL",
        )
        rf_obs = ConstantRiskFactorObserver(constant_value=110.0)  # Stock at $110
        contract = OptionContract(attributes=attrs, risk_factor_observer=rf_obs)

        result = contract.simulate()

        # Should have PRD, MD, STD events
        event_types = [e.event_type for e in result.events]
        assert EventType.PRD in event_types
        assert EventType.MD in event_types
        assert EventType.STD in event_types

        # PRD: Pay premium -$500
        prd_payoff = next(e.payoff for e in result.events if e.event_type == EventType.PRD)
        assert prd_payoff == pytest.approx(-500.0, abs=0.01)

        # STD: Receive exercise amount $1000
        std_payoff = next(e.payoff for e in result.events if e.event_type == EventType.STD)
        assert std_payoff == pytest.approx(1000.0, abs=0.01)

        # Net profit: 1000 - 500 = 500
        total_payoff = sum(e.payoff for e in result.events)
        assert total_payoff == pytest.approx(500.0, abs=0.01)

    def test_optns_simulation_european_put_itm(self):
        """Test complete simulation of European put option in-the-money."""
        attrs = ContractAttributes(
            contract_id="OPT-PUT-ITM",
            contract_type=ContractType.OPTNS,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            purchase_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100.0,
            option_type="P",  # Put
            option_strike_1=100.0,
            option_exercise_type="E",
            price_at_purchase_date=4.0,  # Premium $4/share
            contract_structure="AAPL",
        )
        rf_obs = ConstantRiskFactorObserver(constant_value=90.0)  # Stock at $90
        contract = OptionContract(attributes=attrs, risk_factor_observer=rf_obs)

        result = contract.simulate()

        # STD: Receive exercise amount $1000 (max(100-90,0) * 100)
        std_payoff = next(e.payoff for e in result.events if e.event_type == EventType.STD)
        assert std_payoff == pytest.approx(1000.0, abs=0.01)

        # Net profit: 1000 - 400 = 600
        total_payoff = sum(e.payoff for e in result.events)
        assert total_payoff == pytest.approx(600.0, abs=0.01)

    def test_optns_simulation_european_call_otm(self):
        """Test complete simulation of European call option out-of-the-money (expires worthless)."""
        attrs = ContractAttributes(
            contract_id="OPT-CALL-OTM",
            contract_type=ContractType.OPTNS,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            purchase_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100.0,
            option_type="C",
            option_strike_1=100.0,
            option_exercise_type="E",
            price_at_purchase_date=5.0,
            contract_structure="AAPL",
        )
        rf_obs = ConstantRiskFactorObserver(constant_value=90.0)  # Stock at $90 (OTM)
        contract = OptionContract(attributes=attrs, risk_factor_observer=rf_obs)

        result = contract.simulate()

        # STD: No exercise, payoff=0
        std_payoff = next(e.payoff for e in result.events if e.event_type == EventType.STD)
        assert std_payoff == pytest.approx(0.0, abs=0.01)

        # Net loss: 0 - 500 = -500 (lose premium)
        total_payoff = sum(e.payoff for e in result.events)
        assert total_payoff == pytest.approx(-500.0, abs=0.01)

    def test_optns_factory_creation(self):
        """Test creating OPTNS via factory."""
        attrs = ContractAttributes(
            contract_id="OPT-FACTORY",
            contract_type=ContractType.OPTNS,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            currency="USD",
            notional_principal=100.0,
            option_type="C",
            option_strike_1=100.0,
            option_exercise_type="E",
            contract_structure="AAPL",
        )
        rf_obs = ConstantRiskFactorObserver(constant_value=110.0)

        contract = create_contract(attrs, rf_obs)

        assert isinstance(contract, OptionContract)
        result = contract.simulate()
        assert len(result.events) > 0
