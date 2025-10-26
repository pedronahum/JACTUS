"""Payoff function framework for ACTUS contracts.

This module implements the payoff function (POF) framework as defined in
Section 2.7 and 3.9 of the ACTUS specification v1.1.

The payoff function f(e, S, M, t, o_rf) calculates the cashflow amount for
a contract event, applying contract role sign and FX rate adjustments.

References:
    ACTUS Technical Specification v1.1:
    - Section 2.7: Payoff Functions
    - Section 3.9: Canonical Contract Payoff Function F(x,t)
    - Section 3.10: Settlement Currency FX Rate X^CURS_CUR(t)
"""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import jax.numpy as jnp

from jactus.core import ActusDateTime, ContractAttributes, ContractState
from jactus.core.types import ContractRole, EventType
from jactus.utilities import contract_role_sign


@runtime_checkable
class PayoffFunction(Protocol):
    """Protocol for payoff functions.

    A payoff function calculates the cashflow amount for a contract event.
    All concrete POF implementations must implement this protocol.

    The payoff function signature is:
        f(e, S, M, t, o_rf) -> payoff

    Where:
        e = event type
        S = pre-event state
        M = contract attributes
        t = event time
        o_rf = risk factor observer

    Returns:
        Payoff amount as JAX array (scalar)
    """

    def __call__(
        self,
        event_type: EventType,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: "RiskFactorObserver",  # type: ignore # noqa: F821
    ) -> jnp.ndarray:
        """Calculate payoff for an event.

        Args:
            event_type: Type of contract event
            state: Pre-event contract state
            attributes: Contract attributes/terms
            time: Event time
            risk_factor_observer: Observer for market data

        Returns:
            Payoff amount as JAX array (scalar)

        References:
            ACTUS v1.1 Section 2.7
        """
        ...


class BasePayoffFunction(ABC):
    """Base class for payoff functions with common logic.

    This abstract base class implements the common payoff calculation pipeline:
    1. Calculate base payoff amount (contract-specific, abstract)
    2. Apply contract role sign R(CNTRL)
    3. Apply FX rate X^CURS_CUR(t) if settlement currency differs

    Subclasses must implement calculate_payoff() with contract-specific logic.

    Attributes:
        contract_role: Contract role (RPA, RPL, etc.)
        currency: Contract currency
        settlement_currency: Settlement currency (None = same as contract currency)

    References:
        ACTUS v1.1 Section 2.7, 3.10
    """

    def __init__(
        self,
        contract_role: ContractRole,
        currency: str,
        settlement_currency: str | None = None,
    ):
        """Initialize base payoff function.

        Args:
            contract_role: Contract role for sign adjustment
            currency: Contract currency (e.g., "USD")
            settlement_currency: Settlement currency (None = same as contract)
        """
        self.contract_role = contract_role
        self.currency = currency
        self.settlement_currency = settlement_currency

    @abstractmethod
    def calculate_payoff(
        self,
        event_type: EventType,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: "RiskFactorObserver",  # type: ignore # noqa: F821
    ) -> jnp.ndarray:
        """Calculate base payoff before role sign and FX adjustments.

        This is the contract-specific payoff logic that subclasses must implement.

        Args:
            event_type: Type of contract event
            state: Pre-event contract state
            attributes: Contract attributes/terms
            time: Event time
            risk_factor_observer: Observer for market data

        Returns:
            Base payoff amount as JAX array (scalar)
        """
        ...

    def apply_role_sign(self, amount: jnp.ndarray) -> jnp.ndarray:
        """Apply contract role sign R(CNTRL).

        The contract role determines the sign of cashflows:
        - RPA, LG, BUY, etc.: +1 (receive cashflows)
        - RPL, ST, SEL, etc.: -1 (pay cashflows)

        Args:
            amount: Unsigned payoff amount

        Returns:
            Signed payoff amount

        Formula:
            signed_amount = amount * R(CNTRL)

        References:
            ACTUS v1.1 Table 1 (Contract Role Signs)
        """
        sign = contract_role_sign(self.contract_role)
        return amount * jnp.array(sign, dtype=jnp.float32)

    def apply_fx_rate(
        self,
        amount: jnp.ndarray,
        time: ActusDateTime,
        risk_factor_observer: "RiskFactorObserver",  # type: ignore # noqa: F821
    ) -> jnp.ndarray:
        """Apply FX rate X^CURS_CUR(t) if settlement currency differs.

        If the settlement currency differs from the contract currency, the payoff
        must be converted using the FX rate observed at the event time.

        Args:
            amount: Payoff in contract currency
            time: Event time
            risk_factor_observer: Observer for FX rates

        Returns:
            Payoff in settlement currency

        Formula:
            If CURS != CUR:
                payoff_settlement = payoff_contract * X^CURS_CUR(t)
            Else:
                payoff_settlement = payoff_contract

        References:
            ACTUS v1.1 Section 3.10
        """
        fx_rate = settlement_currency_fx_rate(
            time=time,
            contract_currency=self.currency,
            settlement_currency=self.settlement_currency,
            risk_factor_observer=risk_factor_observer,
        )
        return amount * fx_rate

    def __call__(
        self,
        event_type: EventType,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: "RiskFactorObserver",  # type: ignore # noqa: F821
    ) -> jnp.ndarray:
        """Calculate complete payoff with role sign and FX adjustments.

        This method implements the complete payoff calculation pipeline:
        1. Calculate base payoff (contract-specific)
        2. Apply contract role sign
        3. Apply FX rate if needed

        Args:
            event_type: Type of contract event
            state: Pre-event contract state
            attributes: Contract attributes/terms
            time: Event time
            risk_factor_observer: Observer for market data

        Returns:
            Final payoff amount as JAX array (scalar)

        References:
            ACTUS v1.1 Section 2.7
        """
        # Step 1: Calculate base payoff
        payoff = self.calculate_payoff(event_type, state, attributes, time, risk_factor_observer)

        # Step 2: Apply contract role sign
        payoff = self.apply_role_sign(payoff)

        # Step 3: Apply FX rate
        return self.apply_fx_rate(payoff, time, risk_factor_observer)


def settlement_currency_fx_rate(
    time: ActusDateTime,
    contract_currency: str,
    settlement_currency: str | None,
    risk_factor_observer: "RiskFactorObserver",  # type: ignore # noqa: F821
) -> jnp.ndarray:
    """Get FX rate X^CURS_CUR(t) for settlement currency conversion.

    Returns the FX rate to convert from contract currency (CUR) to settlement
    currency (CURS) at the given time.

    Args:
        time: Time at which to observe FX rate
        contract_currency: Contract currency code (e.g., "USD")
        settlement_currency: Settlement currency code (None = same as contract)
        risk_factor_observer: Observer for FX rate data

    Returns:
        FX rate as JAX array (1.0 if currencies are same)

    Logic:
        If settlement_currency is None: return 1.0
        If settlement_currency == contract_currency: return 1.0
        Otherwise: observe FX rate "contract_currency/settlement_currency"

    Example:
        >>> # Contract in EUR, settled in USD
        >>> fx_rate = settlement_currency_fx_rate(
        ...     time=t,
        ...     contract_currency="EUR",
        ...     settlement_currency="USD",
        ...     risk_factor_observer=observer
        ... )
        >>> # Returns EUR/USD rate, e.g., 1.18

    References:
        ACTUS v1.1 Section 3.10
    """
    # If no settlement currency specified, or same as contract currency
    if settlement_currency is None or settlement_currency == contract_currency:
        return jnp.array(1.0, dtype=jnp.float32)

    # Observe FX rate from risk factor observer
    # Convention: "BASE/QUOTE" where BASE is contract currency
    fx_identifier = f"{contract_currency}/{settlement_currency}"

    return risk_factor_observer.observe_risk_factor(identifier=fx_identifier, time=time)  # type: ignore[no-any-return]


def canonical_contract_payoff(
    contract: "BaseContract",  # type: ignore # noqa: F821
    time: ActusDateTime,
    risk_factor_observer: "RiskFactorObserver",  # type: ignore # noqa: F821
) -> jnp.ndarray:
    """Calculate canonical contract payoff F(x, t).

    The canonical contract payoff is the sum of all future event payoffs at
    time t, evaluated using the current risk factor conditions.

    This function is used for contract valuation and mark-to-market calculations.

    Args:
        contract: Contract instance (must have get_events() and payoff_function)
        time: Valuation time
        risk_factor_observer: Observer for risk factors

    Returns:
        Total payoff of all future events as JAX array (scalar)

    Formula:
        F(x, t) = Σ f(e_i, S_i, M, t_i, o_rf) for all events e_i where t_i >= t

    Where:
        - e_i = i-th future event
        - S_i = state at event time
        - M = contract attributes
        - t_i = event time
        - o_rf = risk factor observer (frozen at current state)

    Note:
        This uses current risk factor conditions for all future events, which
        may differ from the actual risk factors at those event times.

    Example:
        >>> contract = PAMContract(attributes)
        >>> observer = MockRiskFactorObserver({'LIBOR': {t: 0.03}})
        >>> f_xt = canonical_contract_payoff(contract, t, observer)
        >>> print(f"Contract value: {f_xt}")

    References:
        ACTUS v1.1 Section 3.9
    """
    # Get all events for the contract
    event_schedule = contract.get_events(risk_factor_observer)

    # Filter to future events (t_i >= t)
    future_events = [event for event in event_schedule.events if event.time >= time]

    # If no future events, return zero
    if not future_events:
        return jnp.array(0.0, dtype=jnp.float32)

    # Sum payoffs of all future events
    # Note: This requires simulating the contract to get states at each event
    # For now, we use a simplified approach
    # We need to simulate to get states - for a complete implementation,
    # this would call contract.simulate() and extract payoffs
    # For now, we return a placeholder
    # TODO: Implement full simulation-based canonical payoff

    return jnp.array(0.0, dtype=jnp.float32)
