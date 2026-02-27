"""Prepayment behavioral risk model for ACTUS contracts.

This module implements a 2D surface-based prepayment model that computes
prepayment rates as a function of:

- **Spread** (dimension 1): The difference between the contract's nominal
  interest rate and the current market reference rate. A positive spread
  means the borrower has an incentive to refinance.
- **Loan age** (dimension 2): Time elapsed since the initial exchange date.
  Prepayment behavior typically follows a seasoning pattern, peaking in
  the middle years of a loan's life.

The model returns a **Multiplicative Reduction Delta (MRD)** — a fraction
by which the notional principal is reduced at each prepayment observation time.

This mirrors the ``TwoDimensionalPrepaymentModel`` from the ACTUS risk service,
with the addition of JAX compatibility for automatic differentiation.

References:
    ACTUS Risk Service v2.0 - TwoDimensionalPrepaymentModel
    ACTUS Technical Specification v1.1 - PP (Principal Prepayment) events
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from jactus.observers.behavioral import BaseBehaviorRiskFactorObserver, CalloutEvent
from jactus.utilities.surface import Surface2D

if TYPE_CHECKING:
    from jactus.core import ActusDateTime, ContractAttributes, ContractState
    from jactus.core.types import EventType


class PrepaymentSurfaceObserver(BaseBehaviorRiskFactorObserver):
    """Prepayment model using a 2D surface (spread x loan age).

    At each prepayment observation time, the model:
    1. Computes the **spread** = ``state.ipnr - market_rate(time)``
    2. Computes the **loan age** = years since ``attributes.initial_exchange_date``
    3. Looks up the prepayment rate from the 2D surface
    4. Returns the prepayment rate as a JAX array

    The market reference rate is obtained from a companion market observer
    or from a fixed reference rate.

    Attributes:
        surface: 2D surface mapping (spread, age) to prepayment rate.
        market_rate_id: Identifier for the market reference rate (e.g.,
            ``"UST-5Y"``). If not provided, ``fixed_market_rate`` is used.
        market_observer: Optional companion market risk factor observer for
            fetching the current market rate.
        fixed_market_rate: Fixed market rate used when no market observer
            is provided (default 0.0).
        prepayment_cycle: Cycle string for prepayment event frequency
            (e.g., ``"6M"`` for semi-annual). Used by ``contract_start()``
            to generate callout events.
        model_id: Identifier for this prepayment model instance.

    Example:
        >>> import jax.numpy as jnp
        >>> from jactus.utilities.surface import Surface2D
        >>> surface = Surface2D(
        ...     x_margins=jnp.array([-5.0, 0.0, 1.0, 2.0, 3.0]),
        ...     y_margins=jnp.array([0.0, 1.0, 2.0, 3.0, 5.0, 10.0]),
        ...     values=jnp.array([
        ...         [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # spread=-5%
        ...         [0.00, 0.00, 0.01, 0.00, 0.00, 0.00],  # spread= 0%
        ...         [0.00, 0.01, 0.02, 0.00, 0.00, 0.00],  # spread= 1%
        ...         [0.00, 0.02, 0.05, 0.03, 0.005, 0.00], # spread= 2%
        ...         [0.01, 0.05, 0.10, 0.07, 0.02, 0.00],  # spread= 3%
        ...     ]),
        ... )
        >>> observer = PrepaymentSurfaceObserver(
        ...     surface=surface,
        ...     fixed_market_rate=0.04,
        ...     prepayment_cycle="6M",
        ... )
    """

    def __init__(
        self,
        surface: Surface2D,
        market_rate_id: str | None = None,
        market_observer: Any | None = None,
        fixed_market_rate: float = 0.0,
        prepayment_cycle: str = "6M",
        model_id: str = "prepayment-model",
        name: str | None = None,
    ):
        """Initialize prepayment surface observer.

        Args:
            surface: 2D surface mapping (spread, age) to prepayment rate.
            market_rate_id: Identifier for the market reference rate.
            market_observer: Optional market risk factor observer.
            fixed_market_rate: Fixed market rate when no observer is provided.
            prepayment_cycle: Cycle for prepayment observation frequency.
            model_id: Unique model identifier.
            name: Optional observer name for debugging.
        """
        super().__init__(name or f"PrepaymentSurface({model_id})")
        self.surface = surface
        self.market_rate_id = market_rate_id
        self.market_observer = market_observer
        self.fixed_market_rate = fixed_market_rate
        self.prepayment_cycle = prepayment_cycle
        self.model_id = model_id

    def _get_market_rate(self, time: ActusDateTime) -> float:
        """Get current market reference rate."""
        if self.market_observer is not None and self.market_rate_id is not None:
            return float(self.market_observer.observe_risk_factor(self.market_rate_id, time))
        return self.fixed_market_rate

    def _get_risk_factor(
        self,
        identifier: str,
        time: ActusDateTime,
        state: ContractState | None,
        attributes: ContractAttributes | None,
    ) -> jnp.ndarray:
        """Compute prepayment rate from surface.

        Uses the contract's current nominal interest rate and loan age
        to look up the prepayment rate from the 2D surface.

        Args:
            identifier: Risk factor identifier (typically the model_id).
            time: Current simulation time.
            state: Current contract state (must contain ``ipnr``).
            attributes: Contract attributes (must contain ``initial_exchange_date``).

        Returns:
            Prepayment rate as JAX array (MRD value).
        """
        if state is None or attributes is None:
            return jnp.array(0.0, dtype=jnp.float32)

        # Compute spread: contract rate - market rate
        contract_rate = float(state.ipnr)
        market_rate = self._get_market_rate(time)
        spread = contract_rate - market_rate

        # Compute loan age in years
        ied = attributes.initial_exchange_date
        if ied is None:
            return jnp.array(0.0, dtype=jnp.float32)
        age_years = ied.days_between(time) / 365.25

        # Look up prepayment rate from surface
        return self.surface.evaluate(spread, age_years)

    def _get_event_data(
        self,
        identifier: str,
        event_type: EventType,
        time: ActusDateTime,
        state: ContractState | None,
        attributes: ContractAttributes | None,
    ) -> Any:
        """Prepayment observer does not provide event data.

        Raises:
            KeyError: Always.
        """
        raise KeyError(
            f"PrepaymentSurfaceObserver does not support event data for '{identifier}'"
        )

    def contract_start(
        self,
        attributes: ContractAttributes,
    ) -> list[CalloutEvent]:
        """Generate prepayment observation events over the contract life.

        Creates callout events at the specified prepayment cycle interval
        from the initial exchange date to the maturity date.

        Args:
            attributes: Contract attributes.

        Returns:
            List of CalloutEvent objects with callout_type ``"MRD"``.
        """
        from jactus.core.time import add_period

        ied = attributes.initial_exchange_date
        md = attributes.maturity_date
        if ied is None or md is None:
            return []

        events: list[CalloutEvent] = []
        current = add_period(ied, self.prepayment_cycle)

        while current < md:
            events.append(
                CalloutEvent(
                    model_id=self.model_id,
                    time=current,
                    callout_type="MRD",
                )
            )
            current = add_period(current, self.prepayment_cycle)

        return events
