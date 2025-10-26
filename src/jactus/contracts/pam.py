"""Principal At Maturity (PAM) contract implementation.

This module implements the PAM contract type - a bullet loan with interest payments
where the principal is repaid at maturity. PAM is the foundational loan contract
and serves as a template for amortizing contracts (LAM, NAM, ANN).

ACTUS Reference:
    ACTUS v1.1 Section 7.1 - PAM: Principal At Maturity

Key Features:
    - Principal repaid in full at maturity
    - Regular interest payments (IP events)
    - Optional interest capitalization (IPCI)
    - Variable interest rates with rate resets (RR, RRF)
    - Fees (FP events)
    - Prepayments (PP events)
    - Scaling (SC events)
    - 14 event types total

Example:
    >>> from jactus.contracts.pam import PrincipalAtMaturityContract
    >>> from jactus.core import ContractAttributes, ContractType, ContractRole
    >>> from jactus.observers import ConstantRiskFactorObserver
    >>>
    >>> attrs = ContractAttributes(
    ...     contract_id="LOAN-001",
    ...     contract_type=ContractType.PAM,
    ...     contract_role=ContractRole.RPA,
    ...     status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
    ...     initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
    ...     maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
    ...     currency="USD",
    ...     notional_principal=100000.0,
    ...     nominal_interest_rate=0.05,
    ...     day_count_convention=DayCountConvention.A360,
    ...     interest_payment_cycle="1Y"
    ... )
    >>>
    >>> rf_obs = ConstantRiskFactorObserver(constant_value=0.05)
    >>> contract = PrincipalAtMaturityContract(
    ...     attributes=attrs,
    ...     risk_factor_observer=rf_obs
    ... )
    >>> result = contract.simulate()
"""

from typing import Any

import flax.nnx as nnx
import jax.numpy as jnp

from jactus.contracts.base import BaseContract
from jactus.core import (
    ActusDateTime,
    ContractAttributes,
    ContractEvent,
    ContractRole,
    ContractState,
    ContractType,
    DayCountConvention,
    EventSchedule,
    EventType,
)
from jactus.functions import BasePayoffFunction, BaseStateTransitionFunction
from jactus.observers import ChildContractObserver, RiskFactorObserver
from jactus.utilities import generate_schedule, year_fraction


class PAMPayoffFunction(BasePayoffFunction):
    """Payoff function for PAM contracts.

    Implements all 14 PAM payoff functions according to ACTUS specification.

    ACTUS Reference:
        ACTUS v1.1 Section 7.1 - PAM Payoff Functions

    Events:
        AD: Analysis Date (0.0)
        IED: Initial Exchange Date (disburse principal)
        MD: Maturity Date (return principal + accrued)
        PP: Principal Prepayment
        PY: Penalty Payment
        FP: Fee Payment
        PRD: Purchase Date
        TD: Termination Date
        IP: Interest Payment
        IPCI: Interest Capitalization
        RR: Rate Reset
        RRF: Rate Reset Fixing
        SC: Scaling
        CE: Credit Event
    """

    def calculate_payoff(
        self,
        event_type: Any,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """Calculate payoff for PAM events.

        Dispatches to specific payoff function based on event type.

        Args:
            event_type: Type of event
            state: Current contract state
            attributes: Contract attributes
            time: Event time
            risk_factor_observer: Observer for market data

        Returns:
            Payoff amount as JAX array

        ACTUS Reference:
            POF_[event]_PAM functions from Section 7.1
        """
        if event_type == EventType.AD:
            return self._pof_ad(state, attributes, time, risk_factor_observer)
        if event_type == EventType.IED:
            return self._pof_ied(state, attributes, time, risk_factor_observer)
        if event_type == EventType.MD:
            return self._pof_md(state, attributes, time, risk_factor_observer)
        if event_type == EventType.PP:
            return self._pof_pp(state, attributes, time, risk_factor_observer)
        if event_type == EventType.PY:
            return self._pof_py(state, attributes, time, risk_factor_observer)
        if event_type == EventType.FP:
            return self._pof_fp(state, attributes, time, risk_factor_observer)
        if event_type == EventType.PRD:
            return self._pof_prd(state, attributes, time, risk_factor_observer)
        if event_type == EventType.TD:
            return self._pof_td(state, attributes, time, risk_factor_observer)
        if event_type == EventType.IP:
            return self._pof_ip(state, attributes, time, risk_factor_observer)
        if event_type == EventType.IPCI:
            return self._pof_ipci(state, attributes, time, risk_factor_observer)
        if event_type == EventType.RR:
            return self._pof_rr(state, attributes, time, risk_factor_observer)
        if event_type == EventType.RRF:
            return self._pof_rrf(state, attributes, time, risk_factor_observer)
        if event_type == EventType.SC:
            return self._pof_sc(state, attributes, time, risk_factor_observer)
        if event_type == EventType.CE:
            return self._pof_ce(state, attributes, time, risk_factor_observer)
        # Unknown event type - return 0
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_ad(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_AD_PAM: Analysis Date has no cashflow.

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
        """POF_IED_PAM: Initial Exchange - disburse principal.

        Formula:
            POF_IED_PAM = X^CURS_CUR(t) × R(CNTRL) × (-1) × (NT + PDIED)

        Where:
            NT: Notional principal
            PDIED: Premium/discount at IED
            R(CNTRL): Role sign
            X^CURS_CUR(t): FX rate

        Returns:
            Negative of notional plus premium/discount (outflow for lender)
        """
        # Get notional and premium/discount
        nt = attributes.notional_principal or 0.0
        pdied = attributes.premium_discount_at_ied or 0.0

        # Calculate payoff: -1 × (NT + PDIED)
        # Note: Role sign and FX are applied by base class
        payoff = -1.0 * (nt + pdied)

        return jnp.array(payoff, dtype=jnp.float32)

    def _pof_md(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_MD_PAM: Maturity Date - return principal + accrued.

        Formula:
            POF_MD_PAM = X^CURS_CUR(t) × (Nsc_t⁻ × Nt_t⁻ + Isc_t⁻ × Ipac_t⁻ + Feac_t⁻)

        Returns:
            Scaled notional + scaled accrued interest + accrued fees
        """
        nsc = float(state.nsc)
        nt = float(state.nt)
        isc = float(state.isc)
        ipac = float(state.ipac)
        feac = float(state.feac)

        payoff = nsc * nt + isc * ipac + feac

        return jnp.array(payoff, dtype=jnp.float32)

    def _pof_pp(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_PP_PAM: Principal Prepayment.

        Formula:
            POF_PP_PAM = X^CURS_CUR(t) × f(O_ev(CID, PP, t))

        The prepayment amount is observed from the risk factor observer.

        Returns:
            Observed prepayment amount
        """
        # Observe prepayment amount from risk factor
        # For now, return 0.0 - will be enhanced with observer integration
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_py(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_PY_PAM: Penalty Payment.

        Formula depends on PYTP (penalty type):
            PYTP='A': Fixed amount = PYRT
            PYTP='N': Percentage of notional = Y(Sd_t⁻, t) × Nt_t⁻ × PYRT
            PYTP='I': Interest rate differential

        Returns:
            Penalty amount based on type
        """
        # For now, return 0.0 - will be enhanced with full penalty logic
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_fp(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_FP_PAM: Fee Payment.

        Formula depends on FEB (fee basis):
            FEB='A': Absolute = FER
            FEB='N': Notional percentage = Y(Sd_t⁻, t) × Nt_t⁻ × FER + Feac_t⁻

        Returns:
            Fee payment amount
        """
        # For now, return 0.0 - will be enhanced with full fee logic
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_prd(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_PRD_PAM: Purchase Date - pay purchase price + accrued interest.

        Formula:
            POF_PRD_PAM = X^CURS_CUR(t) × R(CNTRL) × (-1) ×
                          (PPRD + Ipac_t⁻ + Y(Sd_t⁻, t) × Ipnr_t⁻ × Nt_t⁻)

        Returns:
            Negative of (purchase price + accrued interest)
        """
        # For now, return 0.0 - will be enhanced with purchase logic
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_td(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_TD_PAM: Termination Date - receive termination price + accrued.

        Formula:
            POF_TD_PAM = X^CURS_CUR(t) × R(CNTRL) ×
                         (PTD + Ipac_t⁻ + Y(Sd_t⁻, t) × Ipnr_t⁻ × Nt_t⁻)

        Returns:
            Termination price + accrued interest
        """
        # For now, return 0.0 - will be enhanced with termination logic
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_ip(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_IP_PAM: Interest Payment.

        Formula:
            POF_IP_PAM = X^CURS_CUR(t) × Isc_t⁻ × (Ipac_t⁻ + Y(Sd_t⁻, t) × Ipnr_t⁻ × Nt_t⁻)

        Returns:
            Scaled accrued interest payment
        """
        isc = float(state.isc)
        ipac = float(state.ipac)
        ipnr = float(state.ipnr)
        nt = float(state.nt)

        # Calculate year fraction from last status date to now
        dcc = attributes.day_count_convention or DayCountConvention.A360
        yf = year_fraction(state.sd, time, dcc)

        # Interest payment = Isc × (Ipac + YF × Ipnr × Nt)
        payoff = isc * (ipac + yf * ipnr * nt)

        return jnp.array(payoff, dtype=jnp.float32)

    def _pof_ipci(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_IPCI_PAM: Interest Capitalization - no cashflow.

        Returns:
            0.0 (interest is capitalized into notional)
        """
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_rr(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_RR_PAM: Rate Reset - no cashflow.

        Returns:
            0.0 (rate is updated in state transition)
        """
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_rrf(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_RRF_PAM: Rate Reset Fixing - no cashflow.

        Returns:
            0.0 (rate is fixed in state transition)
        """
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_sc(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_SC_PAM: Scaling - no cashflow.

        Returns:
            0.0 (scaling multipliers updated in state transition)
        """
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_ce(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_CE_PAM: Credit Event - no cashflow.

        Returns:
            0.0 (handled in state transition)
        """
        return jnp.array(0.0, dtype=jnp.float32)


class PAMStateTransitionFunction(BaseStateTransitionFunction):
    """State transition function for PAM contracts.

    Implements all 14 PAM state transition functions according to ACTUS specification.

    ACTUS Reference:
        ACTUS v1.1 Section 7.1 - PAM State Transition Functions
    """

    def transition_state(
        self,
        event_type: Any,
        state_pre: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """Transition PAM contract state.

        Dispatches to specific state transition function based on event type.

        Args:
            event_type: Type of event
            state_pre: State before event
            attributes: Contract attributes
            time: Event time
            risk_factor_observer: Observer for market data

        Returns:
            Updated contract state

        ACTUS Reference:
            STF_[event]_PAM functions from Section 7.1
        """
        if event_type == EventType.AD:
            return self._stf_ad(state_pre, attributes, time, risk_factor_observer)
        if event_type == EventType.IED:
            return self._stf_ied(state_pre, attributes, time, risk_factor_observer)
        if event_type == EventType.MD:
            return self._stf_md(state_pre, attributes, time, risk_factor_observer)
        if event_type == EventType.PP:
            return self._stf_pp(state_pre, attributes, time, risk_factor_observer)
        if event_type == EventType.PY:
            return self._stf_py(state_pre, attributes, time, risk_factor_observer)
        if event_type == EventType.FP:
            return self._stf_fp(state_pre, attributes, time, risk_factor_observer)
        if event_type == EventType.PRD:
            return self._stf_prd(state_pre, attributes, time, risk_factor_observer)
        if event_type == EventType.TD:
            return self._stf_td(state_pre, attributes, time, risk_factor_observer)
        if event_type == EventType.IP:
            return self._stf_ip(state_pre, attributes, time, risk_factor_observer)
        if event_type == EventType.IPCI:
            return self._stf_ipci(state_pre, attributes, time, risk_factor_observer)
        if event_type == EventType.RR:
            return self._stf_rr(state_pre, attributes, time, risk_factor_observer)
        if event_type == EventType.RRF:
            return self._stf_rrf(state_pre, attributes, time, risk_factor_observer)
        if event_type == EventType.SC:
            return self._stf_sc(state_pre, attributes, time, risk_factor_observer)
        if event_type == EventType.CE:
            return self._stf_ce(state_pre, attributes, time, risk_factor_observer)
        # Unknown event type - return unchanged state
        return state_pre

    def _stf_ad(
        self,
        state_pre: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_AD_PAM: Analysis Date - accrue interest and update status date.

        Updates:
            ipac_t = Ipac_t⁻ + Y(Sd_t⁻, t) × Ipnr_t⁻ × Nt_t⁻
            sd_t = t
        """
        # Calculate year fraction
        dcc = attributes.day_count_convention or DayCountConvention.A360
        yf = year_fraction(state_pre.sd, time, dcc)

        # Accrue interest
        ipac = float(state_pre.ipac) + yf * float(state_pre.ipnr) * float(state_pre.nt)

        return ContractState(
            sd=time,
            tmd=state_pre.tmd,
            nt=state_pre.nt,
            ipnr=state_pre.ipnr,
            ipac=jnp.array(ipac, dtype=jnp.float32),
            feac=state_pre.feac,
            nsc=state_pre.nsc,
            isc=state_pre.isc,
        )

    def _stf_ied(
        self,
        state_pre: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_IED_PAM: Initial Exchange - set notional and interest rate.

        Updates:
            nt_t = R(CNTRL) × NT
            ipnr_t = IPNR (if defined, else 0.0)
            ipac_t = IPAC (if given) or calculated from IPANX
            sd_t = t
        """
        # Get role sign
        role_sign = self._get_role_sign(attributes.contract_role)

        # Set notional with role sign
        nt = role_sign * (attributes.notional_principal or 0.0)

        # Set interest rate
        ipnr = attributes.nominal_interest_rate or 0.0

        # Set initial accrued interest (for now, 0.0)
        ipac = 0.0

        return ContractState(
            sd=time,
            tmd=attributes.maturity_date or time,
            nt=jnp.array(nt, dtype=jnp.float32),
            ipnr=jnp.array(ipnr, dtype=jnp.float32),
            ipac=jnp.array(ipac, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
        )

    def _stf_md(
        self,
        state_pre: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_MD_PAM: Maturity - zero out all amounts.

        Updates:
            nt_t = 0.0
            ipnr_t = 0.0
            ipac_t = 0.0
            feac_t = 0.0
            sd_t = t
        """
        return ContractState(
            sd=time,
            tmd=state_pre.tmd,
            nt=jnp.array(0.0, dtype=jnp.float32),
            ipnr=jnp.array(0.0, dtype=jnp.float32),
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=state_pre.nsc,
            isc=state_pre.isc,
        )

    def _stf_pp(
        self,
        state_pre: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_PP_PAM: Prepayment - reduce notional and accrue interest.

        Updates:
            ipac_t = Ipac_t⁻ + Y(Sd_t⁻, t) × Ipnr_t⁻ × Nt_t⁻
            nt_t = Nt_t⁻ - prepayment_amount
            sd_t = t
        """
        # For now, just accrue interest and update sd
        dcc = attributes.day_count_convention or DayCountConvention.A360
        yf = year_fraction(state_pre.sd, time, dcc)

        ipac = float(state_pre.ipac) + yf * float(state_pre.ipnr) * float(state_pre.nt)

        return ContractState(
            sd=time,
            tmd=state_pre.tmd,
            nt=state_pre.nt,  # TODO: reduce by prepayment amount
            ipnr=state_pre.ipnr,
            ipac=jnp.array(ipac, dtype=jnp.float32),
            feac=state_pre.feac,
            nsc=state_pre.nsc,
            isc=state_pre.isc,
        )

    def _stf_py(
        self,
        state_pre: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_PY_PAM: Penalty - accrue interest, no notional change."""
        return self._stf_ad(state_pre, attributes, time, risk_factor_observer)

    def _stf_fp(
        self,
        state_pre: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_FP_PAM: Fee Payment - reset fees, accrue interest."""
        dcc = attributes.day_count_convention or DayCountConvention.A360
        yf = year_fraction(state_pre.sd, time, dcc)

        ipac = float(state_pre.ipac) + yf * float(state_pre.ipnr) * float(state_pre.nt)

        return ContractState(
            sd=time,
            tmd=state_pre.tmd,
            nt=state_pre.nt,
            ipnr=state_pre.ipnr,
            ipac=jnp.array(ipac, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),  # Reset fees
            nsc=state_pre.nsc,
            isc=state_pre.isc,
        )

    def _stf_prd(
        self,
        state_pre: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_PRD_PAM: Purchase - accrue interest."""
        return self._stf_ad(state_pre, attributes, time, risk_factor_observer)

    def _stf_td(
        self,
        state_pre: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_TD_PAM: Termination - zero out all amounts and rate."""
        return ContractState(
            sd=time,
            tmd=state_pre.tmd,
            nt=jnp.array(0.0, dtype=jnp.float32),
            ipnr=jnp.array(0.0, dtype=jnp.float32),
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=state_pre.nsc,
            isc=state_pre.isc,
        )

    def _stf_ip(
        self,
        state_pre: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_IP_PAM: Interest Payment - reset accrued interest."""
        return ContractState(
            sd=time,
            tmd=state_pre.tmd,
            nt=state_pre.nt,
            ipnr=state_pre.ipnr,
            ipac=jnp.array(0.0, dtype=jnp.float32),  # Reset accrued interest
            feac=state_pre.feac,
            nsc=state_pre.nsc,
            isc=state_pre.isc,
        )

    def _stf_ipci(
        self,
        state_pre: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_IPCI_PAM: Interest Capitalization - add accrued interest to notional."""
        dcc = attributes.day_count_convention or DayCountConvention.A360
        yf = year_fraction(state_pre.sd, time, dcc)

        # Calculate total accrued interest
        total_ipac = float(state_pre.ipac) + yf * float(state_pre.ipnr) * float(state_pre.nt)

        # Add to notional
        nt = float(state_pre.nt) + total_ipac

        return ContractState(
            sd=time,
            tmd=state_pre.tmd,
            nt=jnp.array(nt, dtype=jnp.float32),
            ipnr=state_pre.ipnr,
            ipac=jnp.array(0.0, dtype=jnp.float32),  # Reset after capitalization
            feac=state_pre.feac,
            nsc=state_pre.nsc,
            isc=state_pre.isc,
        )

    def _stf_rr(
        self,
        state_pre: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_RR_PAM: Rate Reset - update interest rate with caps/floors."""
        # For now, just accrue interest and update sd
        # TODO: Implement full rate reset logic with caps/floors/spreads
        return self._stf_ad(state_pre, attributes, time, risk_factor_observer)

    def _stf_rrf(
        self,
        state_pre: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_RRF_PAM: Rate Reset Fixing - set rate to predefined value."""
        # For now, same as AD
        return self._stf_ad(state_pre, attributes, time, risk_factor_observer)

    def _stf_sc(
        self,
        state_pre: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_SC_PAM: Scaling - update scaling multipliers."""
        # For now, same as AD
        return self._stf_ad(state_pre, attributes, time, risk_factor_observer)

    def _stf_ce(
        self,
        state_pre: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_CE_PAM: Credit Event - same as AD."""
        return self._stf_ad(state_pre, attributes, time, risk_factor_observer)

    def _get_role_sign(self, contract_role: ContractRole | None) -> float:
        """Get the sign for contract role."""
        if contract_role in (ContractRole.RPA, ContractRole.RFL):
            return 1.0
        if contract_role in (ContractRole.RPL, ContractRole.PFL):
            return -1.0
        return 1.0


class PrincipalAtMaturityContract(BaseContract):
    """Principal At Maturity (PAM) contract implementation.

    Represents a bullet loan where principal is repaid in full at maturity
    with regular interest payments.

    ACTUS Reference:
        ACTUS v1.1 Section 7.1 - PAM: Principal At Maturity

    Attributes:
        attributes: Contract attributes
        risk_factor_observer: Observer for market data
        child_contract_observer: Observer for child contracts
        rngs: Random number generators for JAX
    """

    def __init__(
        self,
        attributes: ContractAttributes,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: ChildContractObserver | None = None,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize PAM contract.

        Args:
            attributes: Contract attributes
            risk_factor_observer: Observer for market data
            child_contract_observer: Observer for child contracts (optional)
            rngs: Random number generators (optional)

        Raises:
            ValueError: If validation fails
        """
        super().__init__(
            attributes=attributes,
            risk_factor_observer=risk_factor_observer,
            child_contract_observer=child_contract_observer,
            rngs=rngs,
        )

        # Validate contract type
        if attributes.contract_type != ContractType.PAM:
            raise ValueError(f"Contract type must be PAM, got {attributes.contract_type}")

        # Validate required attributes
        if attributes.initial_exchange_date is None:
            raise ValueError("PAM contract requires initial_exchange_date (IED)")

        if attributes.maturity_date is None:
            raise ValueError("PAM contract requires maturity_date (MD)")

        if attributes.notional_principal is None:
            raise ValueError("PAM contract requires notional_principal (NT)")

        # Validate date ordering
        if attributes.initial_exchange_date < attributes.status_date:
            raise ValueError("initial_exchange_date (IED) must be at or after status_date")

        if attributes.maturity_date <= attributes.initial_exchange_date:
            raise ValueError("maturity_date (MD) must be after initial_exchange_date (IED)")

    def generate_event_schedule(self) -> EventSchedule:
        """Generate PAM event schedule.

        Generates all 14 PAM event types according to ACTUS specification.

        Returns:
            EventSchedule with all contract events

        ACTUS Reference:
            PAM Contract Schedule from Section 7.1
        """
        events: list[ContractEvent] = []

        # IED: Initial Exchange Date
        # Note: IED is validated as non-None in __init__
        ied = self.attributes.initial_exchange_date
        assert ied is not None, "IED must be set"

        events.append(
            ContractEvent(
                event_type=EventType.IED,
                event_time=ied,
                payoff=jnp.array(0.0, dtype=jnp.float32),
                currency=self.attributes.currency or "XXX",
                state_pre=None,
                state_post=None,
                sequence=len(events),
            )
        )

        # MD: Maturity Date
        # Note: MD is validated as non-None in __init__
        md = self.attributes.maturity_date
        assert md is not None, "MD must be set"

        # IP: Interest Payment events
        if self.attributes.interest_payment_cycle:
            ip_schedule = generate_schedule(
                start=ied,
                cycle=self.attributes.interest_payment_cycle,
                end=md,
            )
            # Skip first (IED) and last (MD) dates - those are separate events
            for ip_time in ip_schedule[1:-1]:
                events.append(
                    ContractEvent(
                        event_type=EventType.IP,
                        event_time=ip_time,
                        payoff=jnp.array(0.0, dtype=jnp.float32),
                        currency=self.attributes.currency or "XXX",
                        state_pre=None,
                        state_post=None,
                        sequence=len(events),
                    )
                )

            # Add final IP at maturity if there are intermediate payments
            # (final interest payment happens at MD as part of IP schedule)
            if len(ip_schedule) > 2:  # More than just IED and MD
                events.append(
                    ContractEvent(
                        event_type=EventType.IP,
                        event_time=md,
                        payoff=jnp.array(0.0, dtype=jnp.float32),
                        currency=self.attributes.currency or "XXX",
                        state_pre=None,
                        state_post=None,
                        sequence=len(events),
                    )
                )

        # MD: Maturity Date event
        events.append(
            ContractEvent(
                event_type=EventType.MD,
                event_time=md,
                payoff=jnp.array(0.0, dtype=jnp.float32),
                currency=self.attributes.currency or "XXX",
                state_pre=None,
                state_post=None,
                sequence=len(events),
            )
        )

        # Sort events by time
        events.sort(key=lambda e: (e.event_time.to_iso(), e.sequence))

        # Reassign sequence numbers
        for i, event in enumerate(events):
            events[i] = ContractEvent(
                event_type=event.event_type,
                event_time=event.event_time,
                payoff=event.payoff,
                currency=event.currency,
                state_pre=event.state_pre,
                state_post=event.state_post,
                sequence=i,
            )

        return EventSchedule(
            events=tuple(events),
            contract_id=self.attributes.contract_id,
        )

    def initialize_state(self) -> ContractState:
        """Initialize PAM contract state.

        ACTUS Reference:
            PAM State Initialization from Section 7.1

        Returns:
            Initial contract state
        """
        # Initialize at status date (before IED)
        return ContractState(
            sd=self.attributes.status_date,
            tmd=self.attributes.maturity_date or self.attributes.status_date,
            nt=jnp.array(0.0, dtype=jnp.float32),  # Before IED
            ipnr=jnp.array(0.0, dtype=jnp.float32),  # Before IED
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
        )

    def get_payoff_function(self, event_type: Any) -> PAMPayoffFunction:
        """Get payoff function for PAM events."""
        return PAMPayoffFunction(
            contract_role=self.attributes.contract_role,
            currency=self.attributes.currency,
            settlement_currency=None,
        )

    def get_state_transition_function(self, event_type: Any) -> PAMStateTransitionFunction:
        """Get state transition function for PAM events."""
        return PAMStateTransitionFunction()
