"""Result types and validation for array-mode PAM simulation.

This module provides structured result containers for single-contract and
batched array-mode simulations, plus a validation helper to check whether
a contract is compatible with array-mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from jactus.contracts.pam_array import PAMArrayState


@dataclass(frozen=True)
class ArraySimulationResult:
    """Result from a single array-mode PAM simulation.

    Attributes:
        payoffs: ``(num_events,)`` float32 — payoff at each event.
        final_state: Final ``PAMArrayState`` after all events.
        event_mask: ``(num_events,)`` float32 — 1.0 for real events, 0.0 for NOP padding.
    """

    payoffs: jnp.ndarray
    final_state: PAMArrayState
    event_mask: jnp.ndarray

    @property
    def num_events(self) -> int:
        return int(jnp.sum(self.event_mask))

    def total_cashflow(self) -> jnp.ndarray:
        """Sum of masked payoffs."""
        return jnp.sum(self.payoffs * self.event_mask)


@dataclass(frozen=True)
class BatchSimulationResult:
    """Result from a batched array-mode PAM simulation.

    Attributes:
        payoffs: ``(batch, max_events)`` float32 — payoffs per contract per event.
        final_states: Batched ``PAMArrayState`` (each field is ``(batch,)``).
        masks: ``(batch, max_events)`` float32 — 1.0 for real events.
        contract_ids: Optional list of contract identifiers.
    """

    payoffs: jnp.ndarray
    final_states: PAMArrayState
    masks: jnp.ndarray
    contract_ids: list[str] | None = None

    @property
    def num_contracts(self) -> int:
        return self.payoffs.shape[0]

    def total_cashflows(self) -> jnp.ndarray:
        """Masked sum of payoffs per contract — ``(batch,)``."""
        return jnp.sum(self.payoffs * self.masks, axis=1)

    def present_values(
        self,
        discount_rate: float,
        year_fractions: jnp.ndarray,
    ) -> jnp.ndarray:
        """Vectorized present value per contract — ``(batch,)``.

        Args:
            discount_rate: Annual discount rate.
            year_fractions: ``(batch, max_events)`` cumulative year fractions
                from valuation date to each event.
        """
        discount_factors = 1.0 / (1.0 + discount_rate * year_fractions)
        return jnp.sum(self.payoffs * self.masks * discount_factors, axis=1)


def validate_pam_for_array_mode(attrs) -> list[str]:
    """Check whether contract attributes are compatible with array-mode.

    Returns a list of warning/error messages. Empty list means compatible.
    """
    from jactus.core import ContractType

    errors: list[str] = []
    if attrs.contract_type != ContractType.PAM:
        errors.append(f"Array mode only supports PAM, got {attrs.contract_type}")
    if attrs.status_date is None:
        errors.append("status_date is required")
    if attrs.maturity_date is None:
        errors.append("maturity_date is required for PAM array mode")
    return errors
