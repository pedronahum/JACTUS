"""Simulation and portfolio engines for contract evaluation."""

from jactus.engine.lifecycle import (
    ContractPhase,
    calculate_contract_end,
    filter_events_by_lifecycle,
    get_contract_phase,
    is_contract_active,
)
from jactus.engine.simulator import (
    ContractSimulator,
    SimulationResult,
    create_cashflow_matrix,
    simulate_contracts,
)

__all__ = [
    # Lifecycle management
    "ContractPhase",
    "calculate_contract_end",
    "filter_events_by_lifecycle",
    "get_contract_phase",
    "is_contract_active",
    # Simulation
    "ContractSimulator",
    "SimulationResult",
    "create_cashflow_matrix",
    "simulate_contracts",
]
