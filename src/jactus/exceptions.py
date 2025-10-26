"""Custom exception classes for ACTUS-specific errors.

This module defines a hierarchy of exceptions used throughout the actus_jax package.
All exceptions inherit from ActusException, which provides common functionality
for error handling and context preservation.
"""

from typing import Any


class ActusException(Exception):
    """Base exception for all ACTUS-related errors.

    This is the base class for all custom exceptions in the actus_jax package.
    It provides common functionality for storing context information about
    errors.

    Attributes:
        message: Human-readable error description
        context: Additional context information about the error
    """

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error description
            context: Optional dictionary with additional error context
                    (e.g., contract_id, event_type, timestamp)
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __str__(self) -> str:
        """Return string representation of the exception."""
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


class ContractValidationError(ActusException):
    """Exception raised for invalid contract attributes or configuration.

    This exception should be raised when:
    - Required contract attributes are missing
    - Attribute values are outside valid ranges
    - Attribute combinations are inconsistent
    - Contract type is not supported

    Example:
        >>> raise ContractValidationError(
        ...     "Nominal value must be positive",
        ...     context={"contract_id": "PAM-001", "nominal": -1000}
        ... )
    """


class ScheduleGenerationError(ActusException):
    """Exception raised for errors during schedule generation.

    This exception should be raised when:
    - Schedule generation algorithm fails
    - Date sequences are invalid
    - Calendar or business day adjustments fail
    - Cycle specifications are inconsistent

    Example:
        >>> raise ScheduleGenerationError(
        ...     "Invalid cycle specification for interest payment",
        ...     context={"cycle": "P0M", "anchor_date": "2024-01-01"}
        ... )
    """


class StateTransitionError(ActusException):
    """Exception raised for invalid state transitions.

    This exception should be raised when:
    - State transition function encounters invalid inputs
    - Numerical instability in state calculations
    - State values violate invariants
    - JAX compilation issues in state transitions

    Example:
        >>> raise StateTransitionError(
        ...     "Negative interest rate sensitivity detected",
        ...     context={"event": "IP", "time": "2024-06-01", "state": {...}}
        ... )
    """


class PayoffCalculationError(ActusException):
    """Exception raised for errors in payoff calculations.

    This exception should be raised when:
    - Payoff function produces invalid results (NaN, Inf)
    - Required market data is missing
    - Numerical overflow/underflow occurs
    - Business rules are violated

    Example:
        >>> raise PayoffCalculationError(
        ...     "Division by zero in fee calculation",
        ...     context={"event": "FP", "time": "2024-03-15", "fee_basis": 0}
        ... )
    """


class ObserverError(ActusException):
    """Exception raised for risk factor or child contract observation errors.

    This exception should be raised when:
    - Risk factor observer cannot provide required data
    - Market data is unavailable for the requested time
    - Child contract evaluation fails
    - Observer state is inconsistent

    Example:
        >>> raise ObserverError(
        ...     "No exchange rate data available for requested date",
        ...     context={"currency_pair": "EUR/USD", "date": "2024-01-15"}
        ... )
    """


class DateTimeError(ActusException):
    """Exception raised for date/time parsing or calculation errors.

    This exception should be raised when:
    - Date string parsing fails
    - Timezone conversion issues occur
    - Date arithmetic produces invalid results
    - Calendar operations fail

    Example:
        >>> raise DateTimeError(
        ...     "Unable to parse ISO date string",
        ...     context={"date_string": "2024-13-45", "format": "ISO8601"}
        ... )
    """


class ConventionError(ActusException):
    """Exception raised for day count or business day convention errors.

    This exception should be raised when:
    - Unknown day count convention specified
    - Business day adjustment fails
    - End-of-month rules produce invalid dates
    - Convention application is ambiguous

    Example:
        >>> raise ConventionError(
        ...     "Unsupported day count convention",
        ...     context={"convention": "ACT/999", "supported": ["30E/360", "ACT/365"]}
        ... )
    """


class ConfigurationError(ActusException):
    """Exception raised for configuration and initialization errors.

    This exception should be raised when:
    - Package configuration is invalid
    - Required environment variables are missing
    - Initialization parameters are inconsistent
    - Resource loading fails

    Example:
        >>> raise ConfigurationError(
        ...     "Invalid logging configuration",
        ...     context={"log_level": "INVALID", "valid_levels": ["DEBUG", "INFO"]}
        ... )
    """


class EngineError(ActusException):
    """Exception raised for simulation and portfolio engine errors.

    This exception should be raised when:
    - Portfolio simulation fails
    - Batch processing encounters errors
    - Resource allocation issues occur
    - Parallel execution fails

    Example:
        >>> raise EngineError(
        ...     "Portfolio simulation failed due to memory constraints",
        ...     context={"num_contracts": 10000, "memory_gb": 8}
        ... )
    """
