"""Child contract observer for composite ACTUS contracts.

This module implements observers for child contracts in composite contract structures.
In ACTUS, composite contracts can observe events, states, and attributes of their
child contracts to determine their own behavior.

References:
    ACTUS v1.1 Section 2.10 - Child Contract Observer
    ACTUS v1.1 Section 3.4 - Composite Contract Types
"""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import jax.numpy as jnp

from jactus.core import ActusDateTime, ContractAttributes, ContractEvent, ContractState


@runtime_checkable
class ChildContractObserver(Protocol):
    """Protocol for observing child contract data in composite contracts.

    The child contract observer provides access to events, states, and attributes
    of child contracts within a composite structure. This enables parent contracts
    to make decisions based on child contract behavior.

    Methods correspond to ACTUS observer functions:
    - observe_events: U_ev(i, t, a) - Observe child events at time t
    - observe_state: U_sv(i, t, x, a) - Observe child state at time t
    - observe_attribute: U_ca(i, x) - Observe child attribute value

    Example:
        >>> observer = MockChildContractObserver()
        >>> observer.register_child("loan1", child_contract)
        >>> events = observer.observe_events("loan1", current_time, attributes)
        >>> state = observer.observe_state("loan1", current_time, None, attributes)
        >>> notional = observer.observe_attribute("loan1", "notional_principal")

    Note:
        This protocol uses runtime_checkable to allow isinstance() checks.
        Implementations should handle missing child contracts gracefully.

    References:
        ACTUS v1.1 Section 2.10 - Child Contract Observer
    """

    def observe_events(
        self,
        identifier: str,
        time: ActusDateTime,
        attributes: ContractAttributes | None = None,
    ) -> list[ContractEvent]:
        """Observe events from a child contract.

        Returns all events from the identified child contract that occur at or
        after the specified time. This allows parent contracts to react to
        child contract events.

        Args:
            identifier: Child contract identifier
            time: Observation time
            attributes: Optional parent contract attributes for filtering

        Returns:
            List of child contract events

        Example:
            >>> events = observer.observe_events("child1", t0, parent_attrs)
            >>> principal_events = [e for e in events if e.event_type == EventType.PR]
        """
        ...

    def observe_state(
        self,
        identifier: str,
        time: ActusDateTime,
        state: ContractState | None = None,
        attributes: ContractAttributes | None = None,
    ) -> ContractState:
        """Observe state from a child contract.

        Returns the state of the identified child contract at the specified time.
        This allows parent contracts to access child contract state variables.

        Args:
            identifier: Child contract identifier
            time: Observation time
            state: Optional parent state for context
            attributes: Optional parent attributes for context

        Returns:
            Child contract state at the specified time

        Example:
            >>> child_state = observer.observe_state("child1", t0, parent_state, parent_attrs)
            >>> child_notional = child_state.nt
        """
        ...

    def observe_attribute(
        self,
        identifier: str,
        attribute_name: str,
    ) -> jnp.ndarray:
        """Observe an attribute value from a child contract.

        Returns the value of a specific attribute from the identified child contract.
        This allows parent contracts to access child contract configuration.

        Args:
            identifier: Child contract identifier
            attribute_name: Name of the attribute to observe

        Returns:
            Attribute value as JAX array

        Example:
            >>> notional = observer.observe_attribute("child1", "notional_principal")
            >>> rate = observer.observe_attribute("child1", "nominal_interest_rate")
        """
        ...


class BaseChildContractObserver(ABC):
    """Abstract base class for child contract observers.

    Provides a common implementation pattern with error handling and validation.
    Subclasses implement the abstract methods to define observation behavior.

    Example:
        >>> class MyObserver(BaseChildContractObserver):
        ...     def _get_events(self, identifier, time, attributes):
        ...         return self.children[identifier].get_events()
        ...     def _get_state(self, identifier, time, state, attributes):
        ...         return self.children[identifier].get_state(time)
        ...     def _get_attribute(self, identifier, attribute_name):
        ...         return getattr(self.children[identifier].attributes, attribute_name)

    References:
        ACTUS v1.1 Section 2.10 - Child Contract Observer
    """

    @abstractmethod
    def _get_events(
        self,
        identifier: str,  # noqa: ARG002
        time: ActusDateTime,  # noqa: ARG002
        attributes: ContractAttributes | None,  # noqa: ARG002
    ) -> list[ContractEvent]:
        """Abstract method to retrieve events from child contract.

        Subclasses must implement this to define how events are retrieved.

        Args:
            identifier: Child contract identifier
            time: Observation time
            attributes: Optional parent attributes

        Returns:
            List of child contract events
        """
        raise NotImplementedError

    @abstractmethod
    def _get_state(
        self,
        identifier: str,  # noqa: ARG002
        time: ActusDateTime,  # noqa: ARG002
        state: ContractState | None,  # noqa: ARG002
        attributes: ContractAttributes | None,  # noqa: ARG002
    ) -> ContractState:
        """Abstract method to retrieve state from child contract.

        Subclasses must implement this to define how state is retrieved.

        Args:
            identifier: Child contract identifier
            time: Observation time
            state: Optional parent state
            attributes: Optional parent attributes

        Returns:
            Child contract state
        """
        raise NotImplementedError

    @abstractmethod
    def _get_attribute(
        self,
        identifier: str,  # noqa: ARG002
        attribute_name: str,  # noqa: ARG002
    ) -> jnp.ndarray:
        """Abstract method to retrieve attribute from child contract.

        Subclasses must implement this to define how attributes are retrieved.

        Args:
            identifier: Child contract identifier
            attribute_name: Attribute name

        Returns:
            Attribute value as JAX array
        """
        raise NotImplementedError

    def observe_events(
        self,
        identifier: str,
        time: ActusDateTime,
        attributes: ContractAttributes | None = None,
    ) -> list[ContractEvent]:
        """Observe events from a child contract with error handling.

        Wrapper that adds validation and error handling around _get_events.

        Args:
            identifier: Child contract identifier
            time: Observation time
            attributes: Optional parent attributes

        Returns:
            List of child contract events

        Raises:
            KeyError: If child contract not found
            ValueError: If observation fails
        """
        try:
            events = self._get_events(identifier, time, attributes)
            if not isinstance(events, list):
                raise ValueError(f"Expected list of events, got {type(events)}")
            return events
        except KeyError as e:
            raise KeyError(f"Child contract not found: {identifier}") from e

    def observe_state(
        self,
        identifier: str,
        time: ActusDateTime,
        state: ContractState | None = None,
        attributes: ContractAttributes | None = None,
    ) -> ContractState:
        """Observe state from a child contract with error handling.

        Wrapper that adds validation and error handling around _get_state.

        Args:
            identifier: Child contract identifier
            time: Observation time
            state: Optional parent state
            attributes: Optional parent attributes

        Returns:
            Child contract state

        Raises:
            KeyError: If child contract not found
            ValueError: If observation fails
        """
        try:
            child_state = self._get_state(identifier, time, state, attributes)
            if not isinstance(child_state, ContractState):
                raise ValueError(f"Expected ContractState, got {type(child_state)}")
            return child_state
        except KeyError as e:
            raise KeyError(f"Child contract not found: {identifier}") from e

    def observe_attribute(
        self,
        identifier: str,
        attribute_name: str,
    ) -> jnp.ndarray:
        """Observe attribute from a child contract with error handling.

        Wrapper that adds validation and error handling around _get_attribute.

        Args:
            identifier: Child contract identifier
            attribute_name: Attribute name

        Returns:
            Attribute value as JAX array

        Raises:
            KeyError: If child contract not found
            AttributeError: If attribute not found
        """
        try:
            value = self._get_attribute(identifier, attribute_name)
            return jnp.asarray(value, dtype=jnp.float32)
        except KeyError as e:
            raise KeyError(f"Child contract not found: {identifier}") from e
        except AttributeError as e:
            raise AttributeError(
                f"Attribute '{attribute_name}' not found in child contract '{identifier}'"
            ) from e


class MockChildContractObserver(BaseChildContractObserver):
    """Mock implementation for child contract observation.

    This observer stores child contract data in dictionaries and provides
    simple observation capabilities. Useful for testing and development.

    Attributes:
        child_events: Dictionary mapping child IDs to event lists
        child_states: Dictionary mapping child IDs to states
        child_attributes: Dictionary mapping child IDs to attribute dicts

    Example:
        >>> observer = MockChildContractObserver()
        >>> # Register child contract data
        >>> observer.register_child("loan1",
        ...     events=[event1, event2],
        ...     state=loan_state,
        ...     attributes={"notional_principal": 100000.0}
        ... )
        >>> # Observe child data
        >>> events = observer.observe_events("loan1", t0)
        >>> state = observer.observe_state("loan1", t0)
        >>> notional = observer.observe_attribute("loan1", "notional_principal")

    Note:
        This is a simple mock implementation. Real implementations would
        integrate with actual contract simulation engines.

    References:
        ACTUS v1.1 Section 2.10 - Child Contract Observer
    """

    def __init__(self) -> None:
        """Initialize empty child contract observer."""
        self.child_events: dict[str, list[ContractEvent]] = {}
        self.child_states: dict[str, ContractState] = {}
        self.child_attributes: dict[str, dict[str, float]] = {}

    def register_child(
        self,
        identifier: str,
        events: list[ContractEvent] | None = None,
        state: ContractState | None = None,
        attributes: dict[str, float] | None = None,
    ) -> None:
        """Register a child contract with its data.

        Args:
            identifier: Child contract identifier
            events: Optional list of child events
            state: Optional child state
            attributes: Optional dictionary of attribute name -> value

        Example:
            >>> observer.register_child("child1",
            ...     events=[ContractEvent(...)],
            ...     state=ContractState(...),
            ...     attributes={"notional_principal": 100000.0}
            ... )
        """
        if events is not None:
            self.child_events[identifier] = events
        if state is not None:
            self.child_states[identifier] = state
        if attributes is not None:
            self.child_attributes[identifier] = attributes

    def _get_events(
        self,
        identifier: str,
        time: ActusDateTime,
        attributes: ContractAttributes | None,  # noqa: ARG002
    ) -> list[ContractEvent]:
        """Retrieve events from child contract.

        Filters events to return only those at or after the specified time.

        Args:
            identifier: Child contract identifier
            time: Observation time
            attributes: Optional parent attributes (unused in mock)

        Returns:
            List of child events at or after time
        """
        all_events = self.child_events[identifier]
        # Filter events at or after the observation time
        return [e for e in all_events if e.event_time >= time]

    def _get_state(
        self,
        identifier: str,
        time: ActusDateTime,  # noqa: ARG002
        state: ContractState | None,  # noqa: ARG002
        attributes: ContractAttributes | None,  # noqa: ARG002
    ) -> ContractState:
        """Retrieve state from child contract.

        Args:
            identifier: Child contract identifier
            time: Observation time (unused in mock)
            state: Optional parent state (unused in mock)
            attributes: Optional parent attributes (unused in mock)

        Returns:
            Child contract state
        """
        return self.child_states[identifier]

    def _get_attribute(
        self,
        identifier: str,
        attribute_name: str,
    ) -> jnp.ndarray:
        """Retrieve attribute from child contract.

        Args:
            identifier: Child contract identifier
            attribute_name: Attribute name

        Returns:
            Attribute value as JAX array

        Raises:
            AttributeError: If attribute not found
        """
        if attribute_name not in self.child_attributes[identifier]:
            raise AttributeError(f"Attribute '{attribute_name}' not found in child '{identifier}'")
        value = self.child_attributes[identifier][attribute_name]
        return jnp.array(value, dtype=jnp.float32)

    def apply_conditions(
        self,
        attributes: ContractAttributes,
        overrides: dict[str, float],
    ) -> ContractAttributes:
        """Apply conditional attribute overrides to contract attributes.

        This method temporarily modifies contract attributes based on child
        contract observations. Used in composite contracts where parent
        attributes depend on child state.

        Args:
            attributes: Original contract attributes
            overrides: Dictionary of attribute names to new values

        Returns:
            New ContractAttributes with overrides applied

        Example:
            >>> # Override notional based on child contract
            >>> child_notional = observer.observe_attribute("child1", "notional_principal")
            >>> new_attrs = observer.apply_conditions(
            ...     parent_attrs,
            ...     {"notional_principal": float(child_notional)}
            ... )

        Note:
            This creates a new ContractAttributes instance rather than
            modifying the original (immutability).
        """
        # Create a dictionary of current attribute values
        attr_dict = attributes.model_dump()

        # Apply overrides
        for key, value in overrides.items():
            if key in attr_dict:
                attr_dict[key] = value

        # Create new ContractAttributes with updated values
        return ContractAttributes(**attr_dict)
