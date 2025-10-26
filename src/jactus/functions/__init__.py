"""Payoff and state transition functions for ACTUS contracts."""

from jactus.functions.composition import (
    ContractReference,
    MergeOperation,
    filter_ctst_by_role,
    filter_ctst_by_type,
    get_ctst_reference,
    merge_congruent_events,
    merge_events,
    parse_ctst,
)
from jactus.functions.payoff import (
    BasePayoffFunction,
    PayoffFunction,
    canonical_contract_payoff,
    settlement_currency_fx_rate,
)
from jactus.functions.state import (
    BaseStateTransitionFunction,
    StateTransitionFunction,
    create_state_pre,
    validate_state_transition,
)

__all__ = [
    # Payoff functions
    "PayoffFunction",
    "BasePayoffFunction",
    "canonical_contract_payoff",
    "settlement_currency_fx_rate",
    # State transition functions
    "StateTransitionFunction",
    "BaseStateTransitionFunction",
    "create_state_pre",
    "validate_state_transition",
    # Composition utilities
    "ContractReference",
    "MergeOperation",
    "parse_ctst",
    "filter_ctst_by_type",
    "filter_ctst_by_role",
    "get_ctst_reference",
    "merge_events",
    "merge_congruent_events",
]
