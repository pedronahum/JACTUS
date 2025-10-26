"""Utility functions for contract implementations.

This module provides shared utilities for derivative contracts, including:
- Exercise logic for options and futures
- Underlier valuation functions
- Contract composition helpers

These utilities are designed to be JAX-compatible for automatic differentiation
and JIT compilation.
"""

from jactus.contracts.utils.exercise_logic import (
    ExerciseDecision,
    calculate_intrinsic_value,
    should_exercise,
)
from jactus.contracts.utils.underlier_valuation import (
    get_underlier_market_value,
)

__all__ = [
    # Exercise logic
    "ExerciseDecision",
    "calculate_intrinsic_value",
    "should_exercise",
    # Underlier valuation
    "get_underlier_market_value",
]
