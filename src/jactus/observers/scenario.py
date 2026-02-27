"""Scenario management for ACTUS contract simulation.

A ``Scenario`` bundles together all the risk factor observers (both market
and behavioral) needed for a simulation run into a single, named,
reusable configuration.

This provides a higher-level abstraction over individual observers,
enabling:
- Named, reusable simulation configurations
- Consistent bundling of market data with behavioral models
- Easy scenario comparison (base case vs. stress scenarios)

The scenario automatically composes its market and behavioral observers
using a ``CompositeRiskFactorObserver`` so that a single unified observer
can be passed to the simulation engine.

References:
    ACTUS Risk Service v2.0 - Scenario API
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jactus.observers.behavioral import BehaviorRiskFactorObserver, CalloutEvent
from jactus.observers.risk_factor import CompositeRiskFactorObserver, RiskFactorObserver

if TYPE_CHECKING:
    from jactus.core import ContractAttributes


@dataclass
class Scenario:
    """A named simulation scenario bundling market and behavioral observers.

    A scenario acts as a simulation environment, declaring which market data
    sources and behavioral models are available. It provides a unified
    risk factor observer that the simulation engine can use directly.

    Attributes:
        scenario_id: Unique identifier for this scenario.
        description: Human-readable description.
        market_observers: Dictionary mapping identifiers to market risk
            factor observers. These handle pure market data lookups
            (time series, curves, constants).
        behavior_observers: Dictionary mapping identifiers to behavioral
            risk factor observers. These are state-aware and can inject
            callout events into the simulation timeline.

    Example:
        >>> from jactus.observers import (
        ...     ConstantRiskFactorObserver,
        ...     TimeSeriesRiskFactorObserver,
        ... )
        >>> from jactus.observers.prepayment import PrepaymentSurfaceObserver
        >>>
        >>> scenario = Scenario(
        ...     scenario_id="base-case",
        ...     description="Base case with 5Y UST falling and moderate prepayment",
        ...     market_observers={
        ...         "rates": TimeSeriesRiskFactorObserver({
        ...             "UST-5Y": [
        ...                 (ActusDateTime(2024, 1, 1), 0.045),
        ...                 (ActusDateTime(2025, 1, 1), 0.035),
        ...             ],
        ...         }),
        ...     },
        ...     behavior_observers={
        ...         "prepayment": prepayment_observer,
        ...     },
        ... )
        >>> # Get unified observer for simulation
        >>> observer = scenario.get_observer()
        >>> contract.simulate(risk_factor_observer=observer)
    """

    scenario_id: str
    description: str = ""
    market_observers: dict[str, RiskFactorObserver] = field(default_factory=dict)
    behavior_observers: dict[str, BehaviorRiskFactorObserver] = field(default_factory=dict)

    def get_observer(self) -> RiskFactorObserver:
        """Get a unified risk factor observer that combines all market observers.

        Returns a ``CompositeRiskFactorObserver`` that chains all market
        observers in order, providing a single observer for the simulation
        engine.

        If only one market observer is configured, it is returned directly.
        If no market observers are configured, raises ValueError.

        Returns:
            Unified RiskFactorObserver.

        Raises:
            ValueError: If no market observers are configured.
        """
        observers = list(self.market_observers.values())
        if not observers:
            raise ValueError(
                f"Scenario '{self.scenario_id}' has no market observers configured"
            )
        if len(observers) == 1:
            return observers[0]
        return CompositeRiskFactorObserver(observers, name=f"Scenario({self.scenario_id})")

    def get_callout_events(
        self,
        attributes: ContractAttributes,
    ) -> list[CalloutEvent]:
        """Collect callout events from all behavioral observers.

        Calls ``contract_start()`` on each behavioral observer and
        aggregates the returned callout events.

        Args:
            attributes: Contract attributes.

        Returns:
            Combined list of callout events from all behavioral models,
            sorted by time.
        """
        all_events: list[CalloutEvent] = []
        for observer in self.behavior_observers.values():
            events = observer.contract_start(attributes)
            all_events.extend(events)
        return sorted(all_events, key=lambda e: e.time)

    def add_market_observer(self, identifier: str, observer: RiskFactorObserver) -> None:
        """Add or replace a market risk factor observer.

        Args:
            identifier: Observer identifier.
            observer: Market risk factor observer.
        """
        self.market_observers[identifier] = observer

    def add_behavior_observer(
        self, identifier: str, observer: BehaviorRiskFactorObserver
    ) -> None:
        """Add or replace a behavioral risk factor observer.

        Args:
            identifier: Observer identifier.
            observer: Behavioral risk factor observer.
        """
        self.behavior_observers[identifier] = observer

    def list_risk_factors(self) -> dict[str, str]:
        """List all configured risk factor sources.

        Returns:
            Dictionary mapping identifiers to their observer type names.
        """
        result: dict[str, str] = {}
        for name, obs in self.market_observers.items():
            result[name] = type(obs).__name__
        for name, obs in self.behavior_observers.items():
            result[name] = type(obs).__name__
        return result
