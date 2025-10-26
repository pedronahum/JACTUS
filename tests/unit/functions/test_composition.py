"""Unit tests for composition utilities.

Tests ContractReference, CTST parsing, and event merging operations for
composite ACTUS contracts.
"""

import jax.numpy as jnp

from jactus.core import (
    ActusDateTime,
    ContractEvent,
    ContractRole,
    ContractState,
    ContractType,
    EventType,
)
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

# ============================================================================
# Test ContractReference
# ============================================================================


class TestContractReference:
    """Test ContractReference dataclass."""

    def test_initialization(self):
        """Test ContractReference can be created."""
        ref = ContractReference(object="BOND_001", type=ContractType.PAM, role=ContractRole.RPA)

        assert ref.object == "BOND_001"
        assert ref.type == ContractType.PAM
        assert ref.role == ContractRole.RPA

    def test_initialization_with_strings(self):
        """Test ContractReference with string types/roles."""
        ref = ContractReference(object="TEST", type="PAM", role="RPA")

        assert ref.object == "TEST"
        assert ref.type == "PAM"
        assert ref.role == "RPA"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        ref = ContractReference(object="BOND_001", type=ContractType.PAM, role=ContractRole.RPA)

        d = ref.to_dict()
        assert d["object"] == "BOND_001"
        assert d["type"] == "PAM"
        assert d["role"] == "RPA"

    def test_to_dict_with_strings(self):
        """Test to_dict with string types/roles."""
        ref = ContractReference(object="TEST", type="CLM", role="Long")

        d = ref.to_dict()
        assert d["object"] == "TEST"
        assert d["type"] == "CLM"
        assert d["role"] == "Long"

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {"object": "BOND_001", "type": "PAM", "role": "RPA"}

        ref = ContractReference.from_dict(data)
        assert ref.object == "BOND_001"
        assert ref.type == ContractType.PAM
        assert ref.role == ContractRole.RPA

    def test_from_dict_with_invalid_enum(self):
        """Test from_dict with invalid enum values falls back to strings."""
        data = {"object": "TEST", "type": "UNKNOWN", "role": "CUSTOM"}

        ref = ContractReference.from_dict(data)
        assert ref.object == "TEST"
        assert ref.type == "UNKNOWN"
        assert ref.role == "CUSTOM"

    def test_roundtrip_serialization(self):
        """Test to_dict and from_dict roundtrip."""
        original = ContractReference(
            object="BOND_001", type=ContractType.PAM, role=ContractRole.RPA
        )

        data = original.to_dict()
        restored = ContractReference.from_dict(data)

        assert restored.object == original.object
        assert restored.type == original.type
        assert restored.role == original.role


# ============================================================================
# Test CTST Parsing
# ============================================================================


class TestParseCTST:
    """Test CTST parsing functions."""

    def test_parse_empty_string(self):
        """Test parsing empty CTST string."""
        refs = parse_ctst("")
        assert len(refs) == 0

    def test_parse_none(self):
        """Test parsing None CTST."""
        refs = parse_ctst(None)
        assert len(refs) == 0

    def test_parse_single_reference(self):
        """Test parsing single contract reference."""
        ctst = "BOND_001|PAM|RPA"
        refs = parse_ctst(ctst)

        assert len(refs) == 1
        assert refs[0].object == "BOND_001"
        assert refs[0].type == ContractType.PAM
        assert refs[0].role == ContractRole.RPA

    def test_parse_multiple_references(self):
        """Test parsing multiple contract references."""
        ctst = "BOND_001|PAM|RPA,BOND_002|ANN|RPL,LOAN_003|NAM|RPA"
        refs = parse_ctst(ctst)

        assert len(refs) == 3
        assert refs[0].object == "BOND_001"
        assert refs[1].object == "BOND_002"
        assert refs[2].object == "LOAN_003"

    def test_parse_with_whitespace(self):
        """Test parsing handles whitespace."""
        ctst = " BOND_001 | PAM | RPA , BOND_002 | ANN | RPL "
        refs = parse_ctst(ctst)

        assert len(refs) == 2
        assert refs[0].object == "BOND_001"
        assert refs[1].object == "BOND_002"

    def test_parse_with_unknown_types(self):
        """Test parsing with unknown contract types."""
        ctst = "TEST|CUSTOM_TYPE|CUSTOM_ROLE"
        refs = parse_ctst(ctst)

        assert len(refs) == 1
        assert refs[0].object == "TEST"
        assert refs[0].type == "CUSTOM_TYPE"
        assert refs[0].role == "CUSTOM_ROLE"

    def test_parse_skips_invalid_entries(self):
        """Test parsing skips malformed entries."""
        ctst = "BOND_001|PAM|RPA,INVALID,BOND_002|ANN|RPL"
        refs = parse_ctst(ctst)

        # Should skip the invalid entry
        assert len(refs) == 2
        assert refs[0].object == "BOND_001"
        assert refs[1].object == "BOND_002"


class TestFilterCTST:
    """Test CTST filtering functions."""

    def test_filter_by_type(self):
        """Test filtering by contract type."""
        refs = [
            ContractReference("ID1", ContractType.PAM, ContractRole.RPA),
            ContractReference("ID2", ContractType.ANN, ContractRole.RPA),
            ContractReference("ID3", ContractType.PAM, ContractRole.RPL),
        ]

        pam_refs = filter_ctst_by_type(refs, [ContractType.PAM])

        assert len(pam_refs) == 2
        assert pam_refs[0].object == "ID1"
        assert pam_refs[1].object == "ID3"

    def test_filter_by_multiple_types(self):
        """Test filtering by multiple contract types."""
        refs = [
            ContractReference("ID1", ContractType.PAM, ContractRole.RPA),
            ContractReference("ID2", ContractType.ANN, ContractRole.RPA),
            ContractReference("ID3", ContractType.NAM, ContractRole.RPA),
        ]

        filtered = filter_ctst_by_type(refs, [ContractType.PAM, ContractType.ANN])

        assert len(filtered) == 2

    def test_filter_by_type_with_strings(self):
        """Test filtering with string type values."""
        refs = [
            ContractReference("ID1", "PAM", "RPA"),
            ContractReference("ID2", "ANN", "RPA"),
        ]

        filtered = filter_ctst_by_type(refs, ["PAM"])

        assert len(filtered) == 1
        assert filtered[0].object == "ID1"

    def test_filter_by_role(self):
        """Test filtering by contract role."""
        refs = [
            ContractReference("ID1", ContractType.PAM, ContractRole.RPA),
            ContractReference("ID2", ContractType.PAM, ContractRole.RPL),
            ContractReference("ID3", ContractType.ANN, ContractRole.RPA),
        ]

        rpa_refs = filter_ctst_by_role(refs, [ContractRole.RPA])

        assert len(rpa_refs) == 2
        assert rpa_refs[0].object == "ID1"
        assert rpa_refs[1].object == "ID3"

    def test_filter_by_multiple_roles(self):
        """Test filtering by multiple contract roles."""
        refs = [
            ContractReference("ID1", ContractType.PAM, ContractRole.RPA),
            ContractReference("ID2", ContractType.PAM, ContractRole.RPL),
            ContractReference("ID3", ContractType.PAM, ContractRole.RFL),
        ]

        filtered = filter_ctst_by_role(refs, [ContractRole.RPA, ContractRole.RPL])

        assert len(filtered) == 2

    def test_get_reference_by_id(self):
        """Test getting specific reference by object ID."""
        refs = [
            ContractReference("ID1", ContractType.PAM, ContractRole.RPA),
            ContractReference("ID2", ContractType.ANN, ContractRole.RPA),
        ]

        ref = get_ctst_reference(refs, "ID1")

        assert ref is not None
        assert ref.object == "ID1"
        assert ref.type == ContractType.PAM

    def test_get_reference_not_found(self):
        """Test getting non-existent reference returns None."""
        refs = [ContractReference("ID1", ContractType.PAM, ContractRole.RPA)]

        ref = get_ctst_reference(refs, "NONEXISTENT")

        assert ref is None


# ============================================================================
# Test Event Merging
# ============================================================================


class TestMergeEvents:
    """Test event merging operations."""

    def _create_event(
        self, time: ActusDateTime, event_type: EventType, payoff: float
    ) -> ContractEvent:
        """Helper to create test events."""
        state = ContractState(
            sd=time,
            tmd=time,
            nt=jnp.array(10000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        return ContractEvent(
            event_type=event_type,
            event_time=time,
            payoff=jnp.array(payoff),
            currency="USD",
            state_pre=state,
            state_post=state,
            sequence=0,
        )

    def test_merge_add_operation(self):
        """Test merging events with ADD operation."""
        time = ActusDateTime(2024, 1, 15, 0, 0, 0)

        events1 = [self._create_event(time, EventType.IP, 100.0)]
        events2 = [self._create_event(time, EventType.IP, 50.0)]

        merged = merge_events(events1, events2, MergeOperation.ADD)

        # Should have 1 merged event
        assert len(merged) == 1
        assert float(merged[0].payoff) == 150.0

    def test_merge_sub_operation(self):
        """Test merging events with SUB operation."""
        time = ActusDateTime(2024, 1, 15, 0, 0, 0)

        events1 = [self._create_event(time, EventType.IP, 100.0)]
        events2 = [self._create_event(time, EventType.IP, 30.0)]

        merged = merge_events(events1, events2, MergeOperation.SUB)

        assert len(merged) == 1
        assert float(merged[0].payoff) == 70.0

    def test_merge_mul_operation(self):
        """Test merging events with MUL operation."""
        time = ActusDateTime(2024, 1, 15, 0, 0, 0)

        events1 = [self._create_event(time, EventType.IP, 100.0)]
        events2 = [self._create_event(time, EventType.IP, 2.0)]

        merged = merge_events(events1, events2, MergeOperation.MUL)

        assert len(merged) == 1
        assert float(merged[0].payoff) == 200.0

    def test_merge_div_operation(self):
        """Test merging events with DIV operation."""
        time = ActusDateTime(2024, 1, 15, 0, 0, 0)

        events1 = [self._create_event(time, EventType.IP, 100.0)]
        events2 = [self._create_event(time, EventType.IP, 2.0)]

        merged = merge_events(events1, events2, MergeOperation.DIV)

        assert len(merged) == 1
        assert float(merged[0].payoff) == 50.0

    def test_merge_non_matching_times(self):
        """Test merging events with different times."""
        time1 = ActusDateTime(2024, 1, 15, 0, 0, 0)
        time2 = ActusDateTime(2024, 2, 15, 0, 0, 0)

        events1 = [self._create_event(time1, EventType.IP, 100.0)]
        events2 = [self._create_event(time2, EventType.IP, 50.0)]

        merged = merge_events(events1, events2, MergeOperation.ADD)

        # Should have 2 separate events (no merge)
        assert len(merged) == 2

    def test_merge_non_matching_types(self):
        """Test merging events with different types."""
        time = ActusDateTime(2024, 1, 15, 0, 0, 0)

        events1 = [self._create_event(time, EventType.IP, 100.0)]
        events2 = [self._create_event(time, EventType.PR, 50.0)]

        merged = merge_events(events1, events2, MergeOperation.ADD)

        # Should have 2 separate events (different types)
        assert len(merged) == 2

    def test_merge_with_string_operation(self):
        """Test merging with operation as string."""
        time = ActusDateTime(2024, 1, 15, 0, 0, 0)

        events1 = [self._create_event(time, EventType.IP, 100.0)]
        events2 = [self._create_event(time, EventType.IP, 50.0)]

        merged = merge_events(events1, events2, "ADD")

        assert len(merged) == 1
        assert float(merged[0].payoff) == 150.0

    def test_merge_empty_lists(self):
        """Test merging empty event lists."""
        merged = merge_events([], [], MergeOperation.ADD)
        assert len(merged) == 0

    def test_merge_multiple_matching_events(self):
        """Test merging when multiple events match."""
        time = ActusDateTime(2024, 1, 15, 0, 0, 0)

        events1 = [
            self._create_event(time, EventType.IP, 100.0),
            self._create_event(time, EventType.IP, 200.0),
        ]
        events2 = [
            self._create_event(time, EventType.IP, 50.0),
            self._create_event(time, EventType.IP, 75.0),
        ]

        merged = merge_events(events1, events2, MergeOperation.ADD)

        # Should merge all combinations
        assert len(merged) == 4


class TestMergeCongruentEvents:
    """Test congruent event merging."""

    def _create_event(
        self, time: ActusDateTime, event_type: EventType, payoff: float, sequence: int = 0
    ) -> ContractEvent:
        """Helper to create test events."""
        state = ContractState(
            sd=time,
            tmd=time,
            nt=jnp.array(10000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        return ContractEvent(
            event_type=event_type,
            event_time=time,
            payoff=jnp.array(payoff),
            currency="USD",
            state_pre=state,
            state_post=state,
            sequence=sequence,
        )

    def test_merge_congruent_same_time_type(self):
        """Test merging events at same time with same type."""
        time = ActusDateTime(2024, 1, 15, 0, 0, 0)

        events = [
            self._create_event(time, EventType.IP, 100.0),
            self._create_event(time, EventType.IP, 50.0),
            self._create_event(time, EventType.IP, 25.0),
        ]

        merged = merge_congruent_events(events, MergeOperation.ADD)

        # Should merge into 1 event
        assert len(merged) == 1
        assert float(merged[0].payoff) == 175.0

    def test_merge_congruent_different_times(self):
        """Test congruent merge preserves events at different times."""
        time1 = ActusDateTime(2024, 1, 15, 0, 0, 0)
        time2 = ActusDateTime(2024, 2, 15, 0, 0, 0)

        events = [
            self._create_event(time1, EventType.IP, 100.0),
            self._create_event(time2, EventType.IP, 50.0),
        ]

        merged = merge_congruent_events(events, MergeOperation.ADD)

        # Should keep 2 separate events
        assert len(merged) == 2

    def test_merge_congruent_different_types(self):
        """Test congruent merge preserves events with different types."""
        time = ActusDateTime(2024, 1, 15, 0, 0, 0)

        events = [
            self._create_event(time, EventType.IP, 100.0),
            self._create_event(time, EventType.PR, 50.0),
        ]

        merged = merge_congruent_events(events, MergeOperation.ADD)

        # Should keep 2 separate events (different types)
        assert len(merged) == 2

    def test_merge_congruent_sub_operation(self):
        """Test congruent merge with SUB operation."""
        time = ActusDateTime(2024, 1, 15, 0, 0, 0)

        events = [
            self._create_event(time, EventType.IP, 100.0),
            self._create_event(time, EventType.IP, 30.0),
        ]

        merged = merge_congruent_events(events, MergeOperation.SUB)

        assert len(merged) == 1
        assert float(merged[0].payoff) == 70.0

    def test_merge_congruent_single_event(self):
        """Test congruent merge with single event."""
        time = ActusDateTime(2024, 1, 15, 0, 0, 0)
        events = [self._create_event(time, EventType.IP, 100.0)]

        merged = merge_congruent_events(events, MergeOperation.ADD)

        # Should return unchanged
        assert len(merged) == 1
        assert float(merged[0].payoff) == 100.0

    def test_merge_congruent_empty_list(self):
        """Test congruent merge with empty list."""
        merged = merge_congruent_events([], MergeOperation.ADD)
        assert len(merged) == 0

    def test_merge_congruent_preserves_sorting(self):
        """Test congruent merge maintains time ordering."""
        time1 = ActusDateTime(2024, 1, 15, 0, 0, 0)
        time2 = ActusDateTime(2024, 2, 15, 0, 0, 0)
        time3 = ActusDateTime(2024, 3, 15, 0, 0, 0)

        events = [
            self._create_event(time2, EventType.IP, 100.0, sequence=1),
            self._create_event(time1, EventType.IP, 50.0, sequence=0),
            self._create_event(time3, EventType.IP, 75.0, sequence=2),
        ]

        merged = merge_congruent_events(events, MergeOperation.ADD)

        # Should be sorted by time
        assert len(merged) == 3
        assert merged[0].event_time == time1
        assert merged[1].event_time == time2
        assert merged[2].event_time == time3


# ============================================================================
# Test MergeOperation Enum
# ============================================================================


class TestMergeOperation:
    """Test MergeOperation enum."""

    def test_merge_operation_values(self):
        """Test MergeOperation enum values."""
        assert MergeOperation.ADD.value == "ADD"
        assert MergeOperation.SUB.value == "SUB"
        assert MergeOperation.MUL.value == "MUL"
        assert MergeOperation.DIV.value == "DIV"

    def test_merge_operation_from_string(self):
        """Test creating MergeOperation from string."""
        op = MergeOperation("ADD")
        assert op == MergeOperation.ADD

        op = MergeOperation("SUB")
        assert op == MergeOperation.SUB
