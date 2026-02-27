"""Deposit transaction behavioral risk model for ACTUS contracts.

This module implements a deposit transaction model for UMP (Undefined Maturity
Profile) contracts. It models deposit inflows and outflows as a function of:

- **Dimension 1**: Contract identifier (which specific deposit account)
- **Dimension 2**: Date/time of the transaction

The model uses a labeled 2D surface where the x-axis is the contract ID
and the y-axis is a date label, returning the transaction amount (an
**Absolute Funded Delta** — the absolute change in the deposit balance).

This mirrors the ``TwoDimensionalDepositTrxModel`` from the ACTUS risk
service.

References:
    ACTUS Risk Service v2.0 - TwoDimensionalDepositTrxModel
    ACTUS Technical Specification v1.1 - UMP contract type
"""

from __future__ import annotations

import bisect
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from jactus.observers.behavioral import BaseBehaviorRiskFactorObserver, CalloutEvent
from jactus.utilities.surface import LabeledSurface2D

if TYPE_CHECKING:
    from jactus.core import ActusDateTime, ContractAttributes, ContractState
    from jactus.core.types import EventType


class DepositTransactionObserver(BaseBehaviorRiskFactorObserver):
    """Deposit transaction model for UMP contracts.

    Models deposit inflows and outflows using a schedule of known or
    projected transactions. Each contract identifier has its own
    transaction schedule, looked up from a labeled surface or a
    simpler time-series mapping.

    At each transaction observation time, the model returns the
    **Absolute Funded Delta (AFD)** — the change in the deposit balance
    (positive for inflows, negative for outflows).

    Attributes:
        transactions: Dictionary mapping contract identifiers to a sorted
            list of ``(ActusDateTime, float)`` pairs (time, amount).
        model_id: Identifier for this deposit model instance.

    Example:
        >>> from jactus.core import ActusDateTime
        >>> observer = DepositTransactionObserver(
        ...     transactions={
        ...         "DEPOSIT-001": [
        ...             (ActusDateTime(2024, 1, 15), 10000.0),
        ...             (ActusDateTime(2024, 4, 15), -2000.0),
        ...             (ActusDateTime(2024, 7, 15), 5000.0),
        ...         ],
        ...         "DEPOSIT-002": [
        ...             (ActusDateTime(2024, 2, 1), 50000.0),
        ...             (ActusDateTime(2024, 8, 1), -10000.0),
        ...         ],
        ...     },
        ... )
    """

    def __init__(
        self,
        transactions: dict[str, list[tuple[ActusDateTime, float]]],
        model_id: str = "deposit-trx-model",
        name: str | None = None,
    ):
        """Initialize deposit transaction observer.

        Args:
            transactions: Mapping of contract IDs to transaction schedules.
                Each schedule is a list of (time, amount) pairs, sorted by time.
            model_id: Unique model identifier.
            name: Optional observer name for debugging.
        """
        super().__init__(name or f"DepositTransaction({model_id})")
        self.model_id = model_id
        # Sort each transaction list by time
        self._transactions: dict[str, list[tuple[ActusDateTime, float]]] = {}
        for contract_id, trx_list in transactions.items():
            self._transactions[contract_id] = sorted(trx_list, key=lambda x: x[0])

    @classmethod
    def from_labeled_surface(
        cls,
        surface: LabeledSurface2D,
        date_parser: Any = None,
        model_id: str = "deposit-trx-model",
        name: str | None = None,
    ) -> DepositTransactionObserver:
        """Create from a LabeledSurface2D.

        The x-axis labels are contract IDs and y-axis labels are date strings.

        Args:
            surface: Labeled 2D surface with contract IDs and date labels.
            date_parser: Optional callable to parse date labels into ActusDateTime.
                Defaults to ``ActusDateTime.from_iso``.
            model_id: Unique model identifier.
            name: Optional observer name.

        Returns:
            New DepositTransactionObserver instance.
        """
        from jactus.core.time import ActusDateTime

        parse = date_parser or ActusDateTime.from_iso
        transactions: dict[str, list[tuple[ActusDateTime, float]]] = {}

        for contract_id in surface.x_labels:
            trx_list = []
            for date_label in surface.y_labels:
                amount = float(surface.get(contract_id, date_label))
                if abs(amount) > 1e-10:  # Skip zero transactions
                    trx_list.append((parse(date_label), amount))
            if trx_list:
                transactions[contract_id] = trx_list

        return cls(transactions=transactions, model_id=model_id, name=name)

    def _get_risk_factor(
        self,
        identifier: str,
        time: ActusDateTime,
        state: ContractState | None,  # noqa: ARG002
        attributes: ContractAttributes | None,  # noqa: ARG002
    ) -> jnp.ndarray:
        """Get deposit transaction amount at the given time.

        Returns the transaction amount scheduled for the given contract ID
        at the exact time, or 0.0 if no transaction is scheduled.

        For time-matching, uses the closest scheduled transaction within
        the same day (comparing dates only, not times).

        Args:
            identifier: Contract identifier (deposit account ID).
            time: Current simulation time.
            state: Contract state (unused for this model).
            attributes: Contract attributes (unused for this model).

        Returns:
            Transaction amount (AFD) as JAX array.

        Raises:
            KeyError: If contract identifier is not found.
        """
        if identifier not in self._transactions:
            raise KeyError(
                f"Contract '{identifier}' not found in deposit transaction model '{self.name}'"
            )

        trx_list = self._transactions[identifier]
        if not trx_list:
            return jnp.array(0.0, dtype=jnp.float32)

        # Find exact or nearest match by date
        times = [t for t, _ in trx_list]
        idx = bisect.bisect_left(times, time)

        # Check for exact match
        if idx < len(times) and times[idx] == time:
            return jnp.array(trx_list[idx][1], dtype=jnp.float32)

        # No exact match — return 0
        return jnp.array(0.0, dtype=jnp.float32)

    def _get_event_data(
        self,
        identifier: str,
        event_type: EventType,
        time: ActusDateTime,
        state: ContractState | None,
        attributes: ContractAttributes | None,
    ) -> Any:
        """Deposit transaction observer does not provide event data.

        Raises:
            KeyError: Always.
        """
        raise KeyError(
            f"DepositTransactionObserver does not support event data for '{identifier}'"
        )

    def contract_start(
        self,
        attributes: ContractAttributes,
    ) -> list[CalloutEvent]:
        """Generate callout events for all scheduled transactions.

        Returns a callout event for each transaction time associated with
        the contract's ``contract_id``.

        Args:
            attributes: Contract attributes (uses ``contract_id`` to look up
                the transaction schedule).

        Returns:
            List of CalloutEvent objects with callout_type ``"AFD"``.
        """
        contract_id = attributes.contract_id
        if contract_id not in self._transactions:
            return []

        return [
            CalloutEvent(
                model_id=self.model_id,
                time=trx_time,
                callout_type="AFD",
            )
            for trx_time, _ in self._transactions[contract_id]
        ]

    def get_transaction_schedule(self, contract_id: str) -> list[tuple[ActusDateTime, float]]:
        """Get the full transaction schedule for a contract.

        Args:
            contract_id: Contract identifier.

        Returns:
            Sorted list of (time, amount) pairs.

        Raises:
            KeyError: If contract identifier not found.
        """
        if contract_id not in self._transactions:
            raise KeyError(
                f"Contract '{contract_id}' not found in deposit transaction model '{self.name}'"
            )
        return list(self._transactions[contract_id])
