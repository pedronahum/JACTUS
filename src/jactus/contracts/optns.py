"""Option Contract (OPTNS) implementation.

This module implements the OPTNS contract type - a vanilla option contract
with call, put, and collar options supporting European, American, and Bermudan
exercise styles.

ACTUS Reference:
    ACTUS v1.1 Section 7.15 - OPTNS: Option

Key Features:
    - Option types: Call ('C'), Put ('P'), Collar ('CP')
    - Exercise types: European ('E'), American ('A'), Bermudan ('B')
    - Underlier reference via contract_structure (CTST)
    - Exercise decision logic based on intrinsic value
    - Settlement after exercise
    - Premium payment at purchase

Exercise Mechanics:
    - European: Exercise only at maturity date
    - American: Exercise anytime before expiration
    - Bermudan: Exercise on specific dates only

Settlement:
    - Exercise Date (XD): Calculate exercise amount Xa
    - Settlement Date (STD): Receive Xa (after settlement period)

Example:
    >>> from jactus.contracts.optns import OptionContract
    >>> from jactus.core import ContractAttributes, ContractType, ContractRole
    >>> from jactus.observers import ConstantRiskFactorObserver
    >>>
    >>> # European call option on stock with $100 strike
    >>> attrs = ContractAttributes(
    ...     contract_id="OPT-CALL-001",
    ...     contract_type=ContractType.OPTNS,
    ...     contract_role=ContractRole.RPA,
    ...     status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
    ...     maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
    ...     currency="USD",
    ...     notional_principal=100.0,  # Number of shares
    ...     option_type="C",  # Call option
    ...     option_strike_1=100.0,  # Strike price
    ...     option_exercise_type="E",  # European
    ...     price_at_purchase_date=5.0,  # Premium per share
    ...     contract_structure="AAPL",  # Underlier (stock ticker)
    ... )
    >>>
    >>> # Risk factor observer for stock price
    >>> rf_obs = ConstantRiskFactorObserver(constant_value=110.0)  # Stock at $110
    >>> contract = OptionContract(
    ...     attributes=attrs,
    ...     risk_factor_observer=rf_obs
    ... )
    >>> result = contract.simulate()
"""

from typing import Any

import jax.numpy as jnp

from jactus.contracts.base import BaseContract
from jactus.contracts.utils.exercise_logic import calculate_intrinsic_value
from jactus.contracts.utils.underlier_valuation import get_underlier_market_value
from jactus.core import (
    ActusDateTime,
    ContractAttributes,
    ContractEvent,
    ContractState,
    ContractType,
    EventSchedule,
    EventType,
)
from jactus.functions import BasePayoffFunction, BaseStateTransitionFunction
from jactus.observers import ChildContractObserver, RiskFactorObserver
from jactus.utilities import contract_role_sign


class OptionPayoffFunction(BasePayoffFunction):
    """Payoff function for OPTNS contracts.

    Implements all OPTNS payoff functions according to ACTUS specification.

    ACTUS Reference:
        ACTUS v1.1 Section 7.15 - OPTNS Payoff Functions

    Events:
        AD: Analysis Date (0.0)
        PRD: Purchase Date (pay premium)
        TD: Termination Date (receive termination price)
        MD: Maturity Date (automatic exercise if in-the-money)
        XD: Exercise Date (exercise decision)
        STD: Settlement Date (receive exercise amount)
        CE: Credit Event (contract default)

    State Variables Used:
        xa: Exercise amount (calculated at XD)
        prf: Contract performance (default status)
    """

    def calculate_payoff(
        self,
        event_type: Any,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """Calculate payoff for OPTNS events.

        Dispatches to specific payoff function based on event type.

        Args:
            event_type: Type of event
            state: Current contract state
            attributes: Contract attributes
            time: Event time
            risk_factor_observer: Risk factor observer

        Returns:
            Payoff amount as JAX array
        """
        # Map event types to payoff functions
        if event_type == EventType.AD:
            return self._pof_ad(state, attributes, time, risk_factor_observer)
        if event_type == EventType.IED:
            return self._pof_ied(state, attributes, time, risk_factor_observer)
        if event_type == EventType.PRD:
            return self._pof_prd(state, attributes, time, risk_factor_observer)
        if event_type == EventType.TD:
            return self._pof_td(state, attributes, time, risk_factor_observer)
        if event_type == EventType.MD:
            return self._pof_md(state, attributes, time, risk_factor_observer)
        if event_type == EventType.XD:
            return self._pof_xd(state, attributes, time, risk_factor_observer)
        if event_type == EventType.STD:
            return self._pof_std(state, attributes, time, risk_factor_observer)
        if event_type == EventType.CE:
            return self._pof_ce(state, attributes, time, risk_factor_observer)
        # Unknown event type, return zero
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_ad(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_AD_OPTNS: Analysis Date payoff.

        Analysis dates have zero payoff.

        Returns:
            0.0
        """
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_ied(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_IED_OPTNS: Initial Exchange Date payoff.

        Not used for OPTNS (options start at PRD).

        Returns:
            0.0
        """
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_prd(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_PRD_OPTNS: Purchase Date payoff.

        Pay option premium (negative cashflow for buyer).

        Formula:
            POF_PRD = R(CNTRL) × (-PPRD)

        Returns:
            Premium payment (negative for buyer, positive for seller)
        """
        pprd = attributes.price_at_purchase_date or 0.0
        role_sign = contract_role_sign(attributes.contract_role)

        return jnp.array(role_sign * (-pprd), dtype=jnp.float32)

    def _pof_td(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_TD_OPTNS: Termination Date payoff.

        Receive termination price.

        Formula:
            POF_TD = R(CNTRL) × PTD

        Returns:
            Termination payment
        """
        ptd = attributes.price_at_termination_date or 0.0
        role_sign = contract_role_sign(attributes.contract_role)

        return jnp.array(role_sign * ptd, dtype=jnp.float32)

    def _pof_md(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_MD_OPTNS: Maturity Date payoff.

        Zero payoff at maturity (actual payoff at STD if exercised).

        Returns:
            0.0
        """
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_xd(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_XD_OPTNS: Exercise Date payoff.

        Zero payoff at exercise (Xa calculated, payoff at STD).

        Returns:
            0.0
        """
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_std(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_STD_OPTNS: Settlement Date payoff.

        Receive exercise amount if option was exercised.

        Formula:
            POF_STD = R(CNTRL) × Xa

        where Xa is the exercise amount calculated at XD.

        Returns:
            Exercise amount (0 if not exercised)
        """
        xa = float(state.xa) if hasattr(state, "xa") else 0.0
        role_sign = contract_role_sign(attributes.contract_role)

        return jnp.array(role_sign * xa, dtype=jnp.float32)

    def _pof_ce(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_CE_OPTNS: Credit Event payoff.

        Zero payoff on credit event (option worthless if counterparty defaults).

        Returns:
            0.0
        """
        return jnp.array(0.0, dtype=jnp.float32)


class OptionStateTransitionFunction(BaseStateTransitionFunction):
    """State transition function for OPTNS contracts.

    Handles state transitions for option contracts, including:
    - Exercise decision logic
    - Exercise amount calculation
    - State updates after settlement
    """

    def transition_state(
        self,
        event_type: EventType,
        state_pre: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """Calculate state transition for a given event.

        Args:
            event_type: Type of event triggering the transition
            state_pre: Current contract state (before event)
            attributes: Contract attributes
            time: Event time
            risk_factor_observer: Observer for market data

        Returns:
            Updated contract state (after event)
        """
        # Create a dummy event for compatibility with helper methods
        event = ContractEvent(
            event_type=event_type,
            event_time=time,
            payoff=0.0,
            currency=attributes.currency,
            sequence=0,
        )

        if event_type == EventType.XD:
            return self._stf_xd(state_pre, event, attributes, risk_factor_observer)
        if event_type == EventType.MD:
            return self._stf_md(state_pre, event, attributes, risk_factor_observer)
        if event_type == EventType.STD:
            return self._stf_std(state_pre, event, attributes)
        # No state change for other events
        return state_pre

    def _stf_xd(
        self,
        state: ContractState,
        event: ContractEvent,
        attributes: ContractAttributes,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_XD_OPTNS: Exercise Date state transition.

        Calculate exercise amount based on underlier price and intrinsic value.

        Formula:
            Call:   Xa = max(S_t - OPS1, 0)
            Put:    Xa = max(OPS1 - S_t, 0)
            Collar: Xa = max(S_t - OPS1, 0) + max(OPS2 - S_t, 0)

        Returns:
            State with updated xa (exercise amount)
        """
        # Get underlier price
        underlier_ref = attributes.contract_structure
        if underlier_ref is None:
            raise ValueError("contract_structure (underlier) required for OPTNS")

        spot_price = get_underlier_market_value(
            underlier_ref, event.event_time, risk_factor_observer
        )

        # Calculate intrinsic value
        option_type = attributes.option_type
        strike_1 = attributes.option_strike_1
        strike_2 = attributes.option_strike_2

        intrinsic = calculate_intrinsic_value(option_type, float(spot_price), strike_1, strike_2)

        # Update state with exercise amount
        return ContractState(
            sd=state.sd,
            tmd=state.tmd,
            nt=state.nt,
            ipnr=state.ipnr,
            ipac=state.ipac,
            feac=state.feac,
            nsc=state.nsc,
            isc=state.isc,
            prf=state.prf,
            xa=jnp.array(float(intrinsic), dtype=jnp.float32),
        )

    def _stf_md(
        self,
        state: ContractState,
        event: ContractEvent,
        attributes: ContractAttributes,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_MD_OPTNS: Maturity Date state transition.

        Maturity just updates status date. Exercise logic is handled at XD.

        Returns:
            State with updated status date
        """
        return ContractState(
            sd=event.event_time,
            tmd=state.tmd,
            nt=state.nt,
            ipnr=state.ipnr,
            ipac=state.ipac,
            feac=state.feac,
            nsc=state.nsc,
            isc=state.isc,
            prf=state.prf,
            xa=state.xa if hasattr(state, "xa") else jnp.array(0.0, dtype=jnp.float32),
        )

    def _stf_std(
        self,
        state: ContractState,
        event: ContractEvent,
        attributes: ContractAttributes,
    ) -> ContractState:
        """STF_STD_OPTNS: Settlement Date state transition.

        Reset exercise amount after settlement.

        Returns:
            State with xa reset to 0
        """
        return ContractState(
            sd=state.sd,
            tmd=state.tmd,
            nt=state.nt,
            ipnr=state.ipnr,
            ipac=state.ipac,
            feac=state.feac,
            nsc=state.nsc,
            isc=state.isc,
            prf=state.prf,
            xa=jnp.array(0.0, dtype=jnp.float32),
        )


class OptionContract(BaseContract):
    """Option Contract (OPTNS) implementation.

    Represents a vanilla option contract with call, put, or collar payoffs.
    Supports European, American, and Bermudan exercise styles.

    Attributes:
        option_type (OPTP): 'C' (call), 'P' (put), 'CP' (collar)
        option_strike_1 (OPS1): Primary strike price
        option_strike_2 (OPS2): Secondary strike (collar only)
        option_exercise_type (OPXT): 'E' (European), 'A' (American), 'B' (Bermudan)
        contract_structure (CTST): Underlier reference
        notional_principal (NT): Number of units (e.g., shares)
        price_at_purchase_date (PPRD): Premium per unit
        maturity_date (MD): Option expiration date
        settlement_period (STPD): Period from exercise to settlement

    Example:
        >>> # European call option
        >>> attrs = ContractAttributes(
        ...     contract_type=ContractType.OPTNS,
        ...     option_type="C",
        ...     option_strike_1=100.0,
        ...     option_exercise_type="E",
        ...     contract_structure="AAPL",
        ...     ...
        ... )
        >>> contract = OptionContract(attrs, rf_obs)
        >>> events = contract.generate_event_schedule()
    """

    def __init__(
        self,
        attributes: ContractAttributes,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: ChildContractObserver | None = None,
    ):
        """Initialize OPTNS contract.

        Args:
            attributes: Contract attributes
            risk_factor_observer: Risk factor observer for market data
            child_contract_observer: Observer for underlier contracts (optional)

        Raises:
            ValueError: If validation fails
        """
        # Validate contract type
        if attributes.contract_type != ContractType.OPTNS:
            raise ValueError(f"Expected contract_type=OPTNS, got {attributes.contract_type}")

        # Validate option type
        if attributes.option_type not in ["C", "P", "CP"]:
            raise ValueError(f"option_type must be 'C', 'P', or 'CP', got {attributes.option_type}")

        # Validate strike prices
        if attributes.option_strike_1 is None:
            raise ValueError("option_strike_1 (OPS1) is required for OPTNS")

        if attributes.option_type == "CP" and attributes.option_strike_2 is None:
            raise ValueError("option_strike_2 (OPS2) required for collar options")

        # Validate exercise type
        if attributes.option_exercise_type not in ["E", "A", "B"]:
            raise ValueError(
                f"option_exercise_type must be 'E', 'A', or 'B', got {attributes.option_exercise_type}"
            )

        # Validate underlier reference
        if attributes.contract_structure is None:
            raise ValueError("contract_structure (CTST) required for OPTNS (underlier reference)")

        # Validate maturity date
        if attributes.maturity_date is None:
            raise ValueError("maturity_date is required for OPTNS")

        super().__init__(attributes, risk_factor_observer, child_contract_observer)

    def _is_pre_exercised(self) -> bool:
        """Check if contract was already exercised before simulation start."""
        return (
            self.attributes.exercise_date is not None
            and self.attributes.exercise_amount is not None
        )

    def generate_event_schedule(self) -> EventSchedule:
        """Generate event schedule for OPTNS contract.

        Events depend on exercise type:
        - European: AD (optional), PRD, MD, XD, STD
        - American: AD (optional), PRD, XD (multiple), MD, STD
        - Bermudan: AD (optional), PRD, XD (specific dates), MD, STD
        - Pre-exercised (exercise_date + exercise_amount set): STD only

        Returns:
            Event schedule with all contract events
        """
        events = []

        # Pre-exercised: only generate STD at exercise_date + settlement period
        if self._is_pre_exercised():
            settlement_date = self._apply_settlement_period(self.attributes.exercise_date)
            events.append(
                ContractEvent(
                    event_type=EventType.STD,
                    event_time=settlement_date,
                    payoff=0.0,
                    currency=self.attributes.currency,
                    sequence=0,
                )
            )
            return EventSchedule(
                contract_id=self.attributes.contract_id,
                events=events,
            )

        # Analysis dates (if specified)
        if self.attributes.analysis_dates:
            for ad_time in self.attributes.analysis_dates:
                events.append(
                    ContractEvent(
                        event_type=EventType.AD,
                        event_time=ad_time,
                        payoff=0.0,
                        currency=self.attributes.currency,
                        sequence=0,
                    )
                )

        # Purchase date (if specified)
        if self.attributes.purchase_date:
            events.append(
                ContractEvent(
                    event_type=EventType.PRD,
                    event_time=self.attributes.purchase_date,
                    payoff=0.0,  # Calculated by payoff function
                    currency=self.attributes.currency,
                    sequence=1,
                )
            )

        # Termination date (if specified)
        if self.attributes.termination_date:
            events.append(
                ContractEvent(
                    event_type=EventType.TD,
                    event_time=self.attributes.termination_date,
                    payoff=0.0,
                    currency=self.attributes.currency,
                    sequence=2,
                )
            )

        # Exercise dates (XD) depend on exercise type
        if self.attributes.option_exercise_type == "E":
            # European: single XD at maturity date (after MD)
            events.append(
                ContractEvent(
                    event_type=EventType.XD,
                    event_time=self.attributes.maturity_date,
                    payoff=jnp.array(0.0, dtype=jnp.float32),
                    currency=self.attributes.currency,
                    sequence=4,
                )
            )
        elif self.attributes.option_exercise_type == "A":
            # American: generate monthly XD events from purchase/status to maturity
            from jactus.utilities.schedules import generate_schedule

            xd_start = self.attributes.purchase_date or self.attributes.status_date
            xd_dates = generate_schedule(
                start=xd_start,
                cycle="1M",
                end=self.attributes.maturity_date,
            )
            for xd_time in xd_dates[1:]:
                events.append(
                    ContractEvent(
                        event_type=EventType.XD,
                        event_time=xd_time,
                        payoff=jnp.array(0.0, dtype=jnp.float32),
                        currency=self.attributes.currency,
                        sequence=4,
                    )
                )
        elif self.attributes.option_exercise_type == "B":
            # Bermudan: exercise on specific dates from exercise end date schedule
            if self.attributes.option_exercise_end_date:
                events.append(
                    ContractEvent(
                        event_type=EventType.XD,
                        event_time=self.attributes.option_exercise_end_date,
                        payoff=jnp.array(0.0, dtype=jnp.float32),
                        currency=self.attributes.currency,
                        sequence=4,
                    )
                )

        # Maturity date
        events.append(
            ContractEvent(
                event_type=EventType.MD,
                event_time=self.attributes.maturity_date,
                payoff=0.0,
                currency=self.attributes.currency,
                sequence=3,
            )
        )

        # Settlement date (after maturity + settlement period, with BDC adjustment)
        settlement_date = self._apply_settlement_period(self.attributes.maturity_date)
        settlement_date = self._apply_bdc(settlement_date)
        events.append(
            ContractEvent(
                event_type=EventType.STD,
                event_time=settlement_date,
                payoff=0.0,
                currency=self.attributes.currency,
                sequence=5,
            )
        )

        # Sort events by time and sequence
        events.sort(key=lambda e: (e.event_time.to_iso(), e.sequence))

        return EventSchedule(
            contract_id=self.attributes.contract_id,
            events=events,
        )

    def _apply_settlement_period(self, base_date: ActusDateTime) -> ActusDateTime:
        """Apply settlement period offset to a date.

        Args:
            base_date: The date to offset from (e.g., maturity/exercise date)

        Returns:
            Offset date (base_date + settlement_period)
        """
        sp = self.attributes.settlement_period
        if not sp or sp == "P0D":
            return base_date

        from datetime import timedelta

        from jactus.core.time import parse_cycle

        # Strip ISO 8601 duration prefix (P3D → 3D)
        sp_clean = sp.lstrip("P") if sp.startswith("P") else sp
        mult, period, _ = parse_cycle(sp_clean)
        if period == "D":
            delta = timedelta(days=mult)
        elif period == "W":
            delta = timedelta(weeks=mult)
        elif period == "M":
            from dateutil.relativedelta import relativedelta

            py_dt = base_date.to_datetime() + relativedelta(months=mult)
            return ActusDateTime(py_dt.year, py_dt.month, py_dt.day, py_dt.hour, py_dt.minute, py_dt.second)
        else:
            delta = timedelta(days=mult)

        py_dt = base_date.to_datetime() + delta
        return ActusDateTime(py_dt.year, py_dt.month, py_dt.day, py_dt.hour, py_dt.minute, py_dt.second)

    def _apply_bdc(self, date: ActusDateTime) -> ActusDateTime:
        """Apply business day convention adjustment to a date."""
        from jactus.utilities.calendars import MondayToFridayCalendar

        bdc = self.attributes.business_day_convention
        cal = self.attributes.calendar
        if not bdc or bdc == "NULL" or not cal or cal in ("NO_CALENDAR", "NC"):
            return date
        calendar = MondayToFridayCalendar()
        bdc_val = bdc.value if hasattr(bdc, "value") else str(bdc)
        if bdc_val in ("CSF", "SCF", "CSMF", "SCMF"):
            return calendar.next_business_day(date)
        if bdc_val in ("CSP", "SCP", "CSMP", "SCMP"):
            return calendar.previous_business_day(date)
        return date

    def initialize_state(self) -> ContractState:
        """Initialize contract state at status date.

        Returns:
            Initial contract state with xa set to exercise_amount if pre-exercised
        """
        prf = self.attributes.contract_performance
        if prf is None:
            prf = "PF"  # Default: performing

        # Pre-exercised: xa is the known exercise amount
        xa = self.attributes.exercise_amount if self._is_pre_exercised() else 0.0

        return ContractState(
            sd=self.attributes.status_date,
            tmd=self.attributes.maturity_date,
            nt=jnp.array(0.0, dtype=jnp.float32),
            ipnr=jnp.array(0.0, dtype=jnp.float32),
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
            prf=prf,
            xa=jnp.array(xa, dtype=jnp.float32),  # Exercise amount
        )

    def get_payoff_function(self, event_type: Any) -> OptionPayoffFunction:
        """Get payoff function for OPTNS contract.

        Args:
            event_type: The event type (for compatibility with BaseContract)

        Returns:
            OptionPayoffFunction instance
        """
        return OptionPayoffFunction(
            contract_role=self.attributes.contract_role,
            currency=self.attributes.currency,
        )

    def get_state_transition_function(self, event_type: Any) -> OptionStateTransitionFunction:
        """Get state transition function for OPTNS contract.

        Args:
            event_type: The event type (for compatibility with BaseContract)

        Returns:
            OptionStateTransitionFunction instance
        """
        return OptionStateTransitionFunction()
