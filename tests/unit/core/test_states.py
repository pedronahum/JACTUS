"""Unit tests for contract state variables.

Test ID: T1.4
"""

import jax
import jax.numpy as jnp
import pytest

from jactus.core.states import ContractState, initialize_state
from jactus.core.time import ActusDateTime
from jactus.core.types import ContractPerformance


class TestContractStateCreation:
    """Test ContractState creation."""

    def test_minimal_creation(self):
        """Test creating state with required fields."""
        state = ContractState(
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        assert state.tmd == ActusDateTime(2029, 1, 15, 0, 0, 0)
        assert state.sd == ActusDateTime(2024, 1, 1, 0, 0, 0)
        assert float(state.nt) == pytest.approx(100000.0)
        assert float(state.ipnr) == pytest.approx(0.05)
        assert state.prf == ContractPerformance.PF

    def test_with_optional_fields(self):
        """Test creating state with optional fields."""
        state = ContractState(
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(500.0),
            feac=jnp.array(100.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
            ipac1=jnp.array(250.0),
            ipac2=jnp.array(250.0),
            prnxt=jnp.array(5000.0),
            xd=ActusDateTime(2025, 1, 1, 0, 0, 0),
            xa=jnp.array(10000.0),
        )

        assert float(state.ipac1) == 250.0
        assert float(state.ipac2) == 250.0
        assert float(state.prnxt) == 5000.0
        assert state.xd == ActusDateTime(2025, 1, 1, 0, 0, 0)
        assert float(state.xa) == 10000.0

    def test_initialize_state_helper(self):
        """Test initialize_state convenience function."""
        state = initialize_state(
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=100000.0,
            ipnr=0.05,
        )

        assert float(state.nt) == pytest.approx(100000.0)
        assert float(state.ipnr) == pytest.approx(0.05)
        assert float(state.ipac) == 0.0  # Default
        assert float(state.feac) == 0.0  # Default
        assert float(state.nsc) == 1.0  # Default
        assert float(state.isc) == 1.0  # Default
        assert state.prf == ContractPerformance.PF


class TestImmutability:
    """Test that states are immutable."""

    def test_frozen_dataclass(self):
        """Test that state cannot be modified directly."""
        state = initialize_state(
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=100000.0,
        )

        # Should raise FrozenInstanceError
        try:
            state.nt = jnp.array(150000.0)  # type: ignore
            assert False, "Should not be able to modify frozen dataclass"
        except Exception:
            pass  # Expected

    def test_replace_method(self):
        """Test creating new state with changes."""
        state = initialize_state(
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=100000.0,
            ipnr=0.05,
        )

        # Create new state with changed notional
        new_state = state.replace(nt=jnp.array(90000.0))

        assert float(state.nt) == pytest.approx(100000.0)  # Original unchanged
        assert float(new_state.nt) == 90000.0  # New state has change
        assert float(new_state.ipnr) == pytest.approx(0.05)  # Other fields copied

    def test_replace_multiple_fields(self):
        """Test replacing multiple fields at once."""
        state = initialize_state(
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=100000.0,
            ipnr=0.05,
        )

        new_state = state.replace(
            nt=jnp.array(90000.0),
            ipac=jnp.array(500.0),
            prf=ContractPerformance.DL,
        )

        assert float(new_state.nt) == 90000.0
        assert float(new_state.ipac) == 500.0
        assert new_state.prf == ContractPerformance.DL


class TestSerialization:
    """Test dictionary serialization."""

    def test_to_dict(self):
        """Test converting to dictionary."""
        state = initialize_state(
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=100000.0,
            ipnr=0.05,
        )

        data = state.to_dict()

        assert isinstance(data, dict)
        assert data["tmd"] == "2029-01-15T00:00:00"
        assert data["sd"] == "2024-01-01T00:00:00"
        assert data["nt"] == 100000.0
        assert data["ipnr"] == pytest.approx(0.05)
        assert data["prf"] == "PF"

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "tmd": "2029-01-15T00:00:00",
            "sd": "2024-01-01T00:00:00",
            "nt": 100000.0,
            "ipnr": 0.05,
            "ipac": 0.0,
            "feac": 0.0,
            "nsc": 1.0,
            "isc": 1.0,
            "prf": "PF",
        }

        state = ContractState.from_dict(data)

        assert state.tmd == ActusDateTime(2029, 1, 15, 0, 0, 0)
        assert float(state.nt) == pytest.approx(100000.0)
        assert float(state.ipnr) == pytest.approx(0.05)

    def test_roundtrip_serialization(self):
        """Test that serialization round-trips correctly."""
        original = initialize_state(
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=100000.0,
            ipnr=0.05,
            prf=ContractPerformance.DL,
        )

        # Convert to dict and back
        data = original.to_dict()
        restored = ContractState.from_dict(data)

        assert restored.tmd == original.tmd
        assert restored.sd == original.sd
        assert float(restored.nt) == float(original.nt)
        assert float(restored.ipnr) == float(original.ipnr)
        assert restored.prf == original.prf


class TestEquality:
    """Test equality comparison."""

    def test_equal_states(self):
        """Test that identical states are equal."""
        state1 = initialize_state(
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=100000.0,
            ipnr=0.05,
        )

        state2 = initialize_state(
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=100000.0,
            ipnr=0.05,
        )

        assert state1 == state2

    def test_unequal_states(self):
        """Test that different states are not equal."""
        state1 = initialize_state(
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=100000.0,
        )

        state2 = initialize_state(
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=90000.0,  # Different
        )

        assert state1 != state2

    def test_hash_equal_states(self):
        """Test that equal states have equal hashes."""
        state1 = initialize_state(
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=100000.0,
        )

        state2 = initialize_state(
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=100000.0,
        )

        assert hash(state1) == hash(state2)

    def test_state_in_dict(self):
        """Test using state as dictionary key."""
        state = initialize_state(
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=100000.0,
        )

        d = {state: "value"}
        assert d[state] == "value"


class TestJAXCompatibility:
    """Test JAX pytree registration."""

    def test_pytree_flatten_unflatten(self):
        """Test that state can be flattened and unflattened."""
        state = initialize_state(
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=100000.0,
            ipnr=0.05,
        )

        flat, tree_def = jax.tree_util.tree_flatten(state)
        reconstructed = jax.tree_util.tree_unflatten(tree_def, flat)

        assert reconstructed == state

    def test_pytree_in_structure(self):
        """Test state in nested structure."""
        state1 = initialize_state(
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=100000.0,
        )

        state2 = initialize_state(
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=90000.0,
        )

        data = {"state1": state1, "state2": state2, "count": 2}

        flat, tree_def = jax.tree_util.tree_flatten(data)
        reconstructed = jax.tree_util.tree_unflatten(tree_def, flat)

        assert reconstructed["state1"] == state1
        assert reconstructed["state2"] == state2

    def test_jax_tree_map(self):
        """Test using jax.tree_map with states."""
        state = initialize_state(
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=100000.0,
            ipnr=0.05,
        )

        # Tree map should work on the JAX arrays in the state
        # This verifies pytree registration works correctly
        flat_values, _ = jax.tree_util.tree_flatten(state)
        assert len(flat_values) > 0  # Should have extracted arrays


class TestStateOperations:
    """Test common state operations."""

    def test_update_notional(self):
        """Test updating notional principal."""
        state = initialize_state(
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=100000.0,
        )

        # Simulate principal redemption
        new_nt = state.nt - 5000.0
        new_state = state.replace(nt=new_nt)

        assert float(new_state.nt) == 95000.0

    def test_accrue_interest(self):
        """Test accruing interest."""
        state = initialize_state(
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=100000.0,
            ipnr=0.05,
        )

        # Accrue interest for one day
        interest_per_day = state.nt * state.ipnr / 365.0
        new_state = state.replace(ipac=state.ipac + interest_per_day)

        assert float(new_state.ipac) > 0.0
        assert float(new_state.ipac) < float(state.nt)  # Reasonable amount

    def test_change_performance_status(self):
        """Test changing performance status."""
        state = initialize_state(
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=100000.0,
            prf=ContractPerformance.PF,
        )

        # Mark as delayed
        new_state = state.replace(prf=ContractPerformance.DL)

        assert state.prf == ContractPerformance.PF
        assert new_state.prf == ContractPerformance.DL

    def test_scaling_multipliers(self):
        """Test using scaling multipliers."""
        state = initialize_state(
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=100000.0,
            ipnr=0.05,
        )

        # Apply scaling
        new_state = state.replace(nsc=jnp.array(1.1), isc=jnp.array(1.05))

        assert float(new_state.nsc) == pytest.approx(1.1)
        assert float(new_state.isc) == pytest.approx(1.05)
        # Effective notional would be nt * nsc
        assert float(new_state.nt * new_state.nsc) == 110000.0
