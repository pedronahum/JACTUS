"""Composition utilities for composite ACTUS contracts.

This module provides utilities for working with composite contracts - contracts
that reference and aggregate behavior from child contracts (e.g., CLM, STK).

Key Features:
- CTST (Contract Structure) parsing and filtering
- ContractReference dataclass for child contract metadata
- Event merging operations (ADD, SUB, MUL, DIV)
- Congruent event merging for same-time events

ACTUS References:
    ACTUS v1.1 Section 2.3 - Composite Contracts
    ACTUS v1.1 Appendix A.2 - Contract Structure (CTST)
    ACTUS v1.1 Section 4.7 - Event Aggregation

Example:
    >>> from jactus.functions.composition import parse_ctst, merge_events
    >>> from jactus.core import EventType
    >>>
    >>> # Parse contract structure
    >>> ctst_string = "ID1|CLM|Long,ID2|PAM|Short"
    >>> references = parse_ctst(ctst_string)
    >>> print(f"Found {len(references)} child contracts")
    >>>
    >>> # Merge events from child contracts
    >>> merged = merge_events(events1, events2, operation="ADD")
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import jax.numpy as jnp

from jactus.core import ActusDateTime, ContractEvent, ContractRole, ContractType


class MergeOperation(str, Enum):
    """Operations for merging child contract events.

    Used when aggregating cashflows from multiple child contracts.

    Attributes:
        ADD: Add cashflows (default for long positions)
        SUB: Subtract cashflows (for short positions)
        MUL: Multiply cashflows (for leverage)
        DIV: Divide cashflows (for splits)
    """

    ADD = "ADD"
    SUB = "SUB"
    MUL = "MUL"
    DIV = "DIV"


@dataclass
class ContractReference:
    """Reference to a child contract in a composite structure.

    Used in CTST (Contract Structure) attribute to specify child contracts
    that contribute to a composite contract's behavior.

    Attributes:
        object: Identifier or reference to the child contract
        type: Contract type of the child (e.g., PAM, ANN, CLM)
        role: Role of the child contract (Long/Short, RPA/RPL, etc.)

    ACTUS Reference:
        ACTUS v1.1 Appendix A.2 - Contract Structure (CTST)

    Example:
        >>> ref = ContractReference(
        ...     object="BOND_001",
        ...     type=ContractType.PAM,
        ...     role=ContractRole.RPA
        ... )
        >>> print(ref.to_dict())
    """

    object: str
    type: ContractType | str
    role: ContractRole | str

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with object, type, and role as strings

        Example:
            >>> ref = ContractReference("ID1", ContractType.PAM, ContractRole.RPA)
            >>> d = ref.to_dict()
            >>> print(d)  # {"object": "ID1", "type": "PAM", "role": "RPA"}
        """
        return {
            "object": str(self.object),
            "type": str(self.type.value if isinstance(self.type, ContractType) else self.type),
            "role": str(self.role.value if isinstance(self.role, ContractRole) else self.role),
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "ContractReference":
        """Create ContractReference from dictionary.

        Args:
            data: Dictionary with 'object', 'type', and 'role' keys

        Returns:
            New ContractReference instance

        Example:
            >>> data = {"object": "ID1", "type": "PAM", "role": "RPA"}
            >>> ref = ContractReference.from_dict(data)
        """
        # Try to convert to enums, fall back to strings
        contract_type: ContractType | str
        try:
            contract_type = ContractType(data["type"])
        except (KeyError, ValueError):
            contract_type = data.get("type", "")

        contract_role: ContractRole | str
        try:
            contract_role = ContractRole(data["role"])
        except (KeyError, ValueError):
            contract_role = data.get("role", "")

        return cls(
            object=data["object"],
            type=contract_type,
            role=contract_role,
        )


def parse_ctst(ctst: str | None) -> list[ContractReference]:
    """Parse CTST (Contract Structure) string into ContractReferences.

    CTST format: "ID1|TYPE1|ROLE1,ID2|TYPE2|ROLE2,..."
    Each reference is separated by comma, fields by pipe.

    Args:
        ctst: Contract structure string, or None

    Returns:
        List of ContractReference objects (empty list if ctst is None)

    ACTUS Reference:
        ACTUS v1.1 Appendix A.2 - Contract Structure (CTST)

    Example:
        >>> ctst = "BOND_001|PAM|RPA,BOND_002|PAM|RPL"
        >>> refs = parse_ctst(ctst)
        >>> print(f"Found {len(refs)} references")  # 2
        >>> print(refs[0].object)  # "BOND_001"
    """
    if ctst is None or ctst.strip() == "":
        return []

    references = []
    entries = ctst.split(",")

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        parts = entry.split("|")
        if len(parts) != 3:
            # Invalid format, skip or raise error
            continue

        object_id, type_str, role_str = parts

        # Try to convert to enums
        contract_type: ContractType | str
        try:
            contract_type = ContractType(type_str.strip())
        except ValueError:
            contract_type = type_str.strip()

        contract_role: ContractRole | str
        try:
            contract_role = ContractRole(role_str.strip())
        except ValueError:
            contract_role = role_str.strip()

        references.append(
            ContractReference(
                object=object_id.strip(),
                type=contract_type,
                role=contract_role,
            )
        )

    return references


def filter_ctst_by_type(
    references: list[ContractReference],
    contract_types: list[ContractType | str],
) -> list[ContractReference]:
    """Filter contract references by contract type.

    Args:
        references: List of contract references
        contract_types: List of contract types to include

    Returns:
        Filtered list of references matching the specified types

    Example:
        >>> refs = parse_ctst("ID1|PAM|RPA,ID2|ANN|RPA,ID3|PAM|RPL")
        >>> pam_refs = filter_ctst_by_type(refs, [ContractType.PAM])
        >>> print(len(pam_refs))  # 2
    """
    type_values = [t.value if isinstance(t, ContractType) else str(t) for t in contract_types]

    filtered = []
    for ref in references:
        ref_type = ref.type.value if isinstance(ref.type, ContractType) else str(ref.type)
        if ref_type in type_values:
            filtered.append(ref)

    return filtered


def filter_ctst_by_role(
    references: list[ContractReference],
    contract_roles: list[ContractRole | str],
) -> list[ContractReference]:
    """Filter contract references by contract role.

    Args:
        references: List of contract references
        contract_roles: List of contract roles to include

    Returns:
        Filtered list of references matching the specified roles

    Example:
        >>> refs = parse_ctst("ID1|PAM|RPA,ID2|PAM|RPL,ID3|ANN|RPA")
        >>> rpa_refs = filter_ctst_by_role(refs, [ContractRole.RPA])
        >>> print(len(rpa_refs))  # 2
    """
    role_values = [r.value if isinstance(r, ContractRole) else str(r) for r in contract_roles]

    filtered = []
    for ref in references:
        ref_role = ref.role.value if isinstance(ref.role, ContractRole) else str(ref.role)
        if ref_role in role_values:
            filtered.append(ref)

    return filtered


def get_ctst_reference(
    references: list[ContractReference],
    object_id: str,
) -> ContractReference | None:
    """Get a specific contract reference by object ID.

    Args:
        references: List of contract references
        object_id: Object ID to search for

    Returns:
        ContractReference if found, None otherwise

    Example:
        >>> refs = parse_ctst("ID1|PAM|RPA,ID2|ANN|RPA")
        >>> ref = get_ctst_reference(refs, "ID1")
        >>> print(ref.type)  # ContractType.PAM
    """
    for ref in references:
        if ref.object == object_id:
            return ref
    return None


def merge_events(
    events1: list[ContractEvent],
    events2: list[ContractEvent],
    operation: MergeOperation | str = MergeOperation.ADD,
) -> list[ContractEvent]:
    """Merge two event lists using specified operation.

    Combines events from two contracts. Events are matched by time and type,
    then merged using the specified operation on their payoffs.

    Args:
        events1: First list of events
        events2: Second list of events
        operation: Operation to apply (ADD, SUB, MUL, DIV)

    Returns:
        Merged list of events with combined payoffs

    Note:
        - Only events with matching time and type are merged
        - Non-matching events are included separately
        - Currency must match for merged events

    ACTUS Reference:
        ACTUS v1.1 Section 4.7 - Event Aggregation

    Example:
        >>> from jactus.core import EventType
        >>> # Merge interest payments from two bonds
        >>> merged = merge_events(bond1_events, bond2_events, "ADD")
        >>> # Subtract short position
        >>> hedged = merge_events(long_events, short_events, "SUB")
    """
    if isinstance(operation, str):
        operation = MergeOperation(operation)

    # Create lookup for events2 by (time, type)
    events2_map: dict[tuple[ActusDateTime, Any], list[ContractEvent]] = {}
    for event in events2:
        key = (event.event_time, event.event_type)
        if key not in events2_map:
            events2_map[key] = []
        events2_map[key].append(event)

    merged = []
    matched_keys = set()

    # Process events1 and merge with matching events2
    for event1 in events1:
        key = (event1.event_time, event1.event_type)

        if key in events2_map:
            # Merge with all matching events from events2
            for event2 in events2_map[key]:
                # Check currency compatibility
                if event1.currency != event2.currency:
                    # Can't merge different currencies, keep separate
                    merged.append(event1)
                    continue

                # Apply operation
                if operation == MergeOperation.ADD:
                    merged_payoff = event1.payoff + event2.payoff
                elif operation == MergeOperation.SUB:
                    merged_payoff = event1.payoff - event2.payoff
                elif operation == MergeOperation.MUL:
                    merged_payoff = event1.payoff * event2.payoff
                elif operation == MergeOperation.DIV:
                    # Avoid division by zero
                    merged_payoff = jnp.where(
                        event2.payoff != 0,
                        event1.payoff / event2.payoff,
                        event1.payoff,
                    )
                else:
                    merged_payoff = event1.payoff

                # Create merged event
                merged_event = ContractEvent(
                    event_type=event1.event_type,
                    event_time=event1.event_time,
                    payoff=merged_payoff,
                    currency=event1.currency,
                    state_pre=event1.state_pre,
                    state_post=event1.state_post,
                    sequence=event1.sequence,
                )
                merged.append(merged_event)

            matched_keys.add(key)
        else:
            # No matching event, keep as-is
            merged.append(event1)

    # Add unmatched events from events2
    for event2 in events2:
        key = (event2.event_time, event2.event_type)
        if key not in matched_keys:
            merged.append(event2)

    return merged


def merge_congruent_events(
    events: list[ContractEvent],
    operation: MergeOperation | str = MergeOperation.ADD,
) -> list[ContractEvent]:
    """Merge congruent events (same time and type) within a single event list.

    Useful for consolidating multiple events that occur at the same time
    into a single aggregated event.

    Args:
        events: List of events to merge
        operation: Operation to apply when merging (default: ADD)

    Returns:
        List with congruent events merged

    Example:
        >>> # Merge multiple interest payments at same time
        >>> events = [ip_event1, ip_event2, ip_event3]  # All at 2024-01-15
        >>> consolidated = merge_congruent_events(events, "ADD")
        >>> print(len(consolidated))  # 1
    """
    if isinstance(operation, str):
        operation = MergeOperation(operation)

    # Group events by (time, type, currency)
    groups: dict[tuple[ActusDateTime, Any, str], list[ContractEvent]] = {}

    for event in events:
        key = (event.event_time, event.event_type, event.currency)
        if key not in groups:
            groups[key] = []
        groups[key].append(event)

    # Merge each group
    merged = []
    for (time, event_type, currency), group in groups.items():
        if len(group) == 1:
            # No merging needed
            merged.append(group[0])
        else:
            # Merge all events in group
            result_payoff = group[0].payoff
            for event in group[1:]:
                if operation == MergeOperation.ADD:
                    result_payoff = result_payoff + event.payoff
                elif operation == MergeOperation.SUB:
                    result_payoff = result_payoff - event.payoff
                elif operation == MergeOperation.MUL:
                    result_payoff = result_payoff * event.payoff
                elif operation == MergeOperation.DIV:
                    result_payoff = jnp.where(
                        event.payoff != 0,
                        result_payoff / event.payoff,
                        result_payoff,
                    )

            # Create merged event using first event as template
            merged_event = ContractEvent(
                event_type=event_type,
                event_time=time,
                payoff=result_payoff,
                currency=currency,
                state_pre=group[0].state_pre,
                state_post=group[-1].state_post,  # Use final state
                sequence=group[0].sequence,
            )
            merged.append(merged_event)

    # Sort by time and sequence
    # ActusDateTime is comparable via to_iso() which returns sortable ISO strings
    merged.sort(key=lambda e: (e.event_time.to_iso(), e.sequence))

    return merged
