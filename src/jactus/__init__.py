"""JACTUS: JAX implementation of the ACTUS financial contract standard.

This package provides a high-performance implementation of the ACTUS (Algorithmic
Contract Types Unified Standards) specification using JAX for automatic
differentiation and GPU acceleration.

Basic usage:
    >>> import jactus
    >>> print(jactus.__version__)
    0.1.2

For more information, see the documentation at:
https://pedronahum.github.io/JACTUS/
"""

__version__ = "0.1.2"
__author__ = "Pedro N. Rodriguez"
__license__ = "Apache-2.0"

# Import core exceptions for convenient access
from jactus.exceptions import (
    ActusException,
    ConfigurationError,
    ContractValidationError,
    ConventionError,
    DateTimeError,
    EngineError,
    ObserverError,
    PayoffCalculationError,
    ScheduleGenerationError,
    StateTransitionError,
)
from jactus.logging_config import configure_logging, get_logger

# Public API
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    # Exceptions
    "ActusException",
    "ConfigurationError",
    "ContractValidationError",
    "ConventionError",
    "DateTimeError",
    "EngineError",
    "ObserverError",
    "PayoffCalculationError",
    "ScheduleGenerationError",
    "StateTransitionError",
    # Logging
    "configure_logging",
    "get_logger",
]


def __getattr__(name: str) -> object:
    """Lazy import for optional dependencies.

    This allows the package to be imported even if some optional
    dependencies are not installed.
    """
    # Future: Add lazy imports for heavy modules here
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
