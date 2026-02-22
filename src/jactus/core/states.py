"""Contract state variables for ACTUS contracts.

This module provides the ContractState dataclass for representing the mutable
state of a contract at a point in time. State variables are immutable and
JAX-compatible for functional programming.

References:
    ACTUS Technical Specification v1.1, Section 6 (State Variables)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp

from jactus.core.time import ActusDateTime
from jactus.core.types import ContractPerformance


@dataclass(frozen=True)
class ContractState:
    """Immutable contract state variables.

    Represents the time-varying state of an ACTUS contract. All numerical
    values use JAX arrays for compatibility with automatic differentiation
    and JIT compilation.

    State follows ACTUS naming conventions (lowercase versions of attribute names).
    See ACTUS documentation for detailed meaning of each state variable.

    Attributes:
        tmd: Maturity date state (Md)
        nt: Notional principal
        ipnr: Nominal interest rate
        ipac: Accrued interest
        ipac1: Accrued interest leg 1 (for swaps)
        ipac2: Accrued interest leg 2 (for swaps)
        feac: Accrued fees
        nsc: Notional scaling multiplier
        isc: Interest scaling multiplier
        prf: Contract performance status
        sd: Status date
        prnxt: Next principal redemption amount
        ipcb: Interest calculation base
        xd: Exercise date (options/futures)
        xa: Exercise amount (options/futures)

    Example:
        >>> import jax.numpy as jnp
        >>> state = ContractState(
        ...     tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
        ...     nt=jnp.array(100000.0),
        ...     ipnr=jnp.array(0.05),
        ...     ipac=jnp.array(0.0),
        ...     feac=jnp.array(0.0),
        ...     nsc=jnp.array(1.0),
        ...     isc=jnp.array(1.0),
        ...     prf=ContractPerformance.PF,
        ...     sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
        ... )

    References:
        ACTUS Technical Specification v1.1, Section 6
    """

    # Required date states
    tmd: ActusDateTime
    sd: ActusDateTime

    # Required numerical states (JAX arrays)
    nt: jnp.ndarray  # Notional
    ipnr: jnp.ndarray  # Interest rate
    ipac: jnp.ndarray  # Accrued interest
    feac: jnp.ndarray  # Accrued fees
    nsc: jnp.ndarray  # Notional scaling multiplier
    isc: jnp.ndarray  # Interest scaling multiplier

    # Optional date states
    xd: ActusDateTime | None = None

    # Optional numerical states
    ipac1: jnp.ndarray | None = None  # Accrued interest leg 1
    ipac2: jnp.ndarray | None = None  # Accrued interest leg 2
    prnxt: jnp.ndarray | None = None  # Next principal redemption
    ipcb: jnp.ndarray | None = None  # Interest calculation base
    xa: jnp.ndarray | None = None  # Exercise amount

    # Performance status (with default)
    prf: ContractPerformance = ContractPerformance.PF

    def replace(self, **changes: Any) -> ContractState:
        """Create a new state with specified changes.

        Since states are immutable, this creates a new ContractState instance
        with the specified fields replaced.

        Args:
            **changes: Field names and new values

        Returns:
            New ContractState with changes applied

        Example:
            >>> new_state = state.replace(nt=jnp.array(90000.0))
            >>> new_state.nt
            Array(90000., dtype=float32)
        """
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation with all state variables

        Example:
            >>> data = state.to_dict()
            >>> data['nt']
            100000.0
        """
        return {
            # Dates
            "tmd": self.tmd.to_iso() if self.tmd else None,
            "sd": self.sd.to_iso() if self.sd else None,
            "xd": self.xd.to_iso() if self.xd else None,
            # Numerical values (convert JAX arrays to Python floats)
            "nt": float(self.nt) if self.nt is not None else None,
            "ipnr": float(self.ipnr) if self.ipnr is not None else None,
            "ipac": float(self.ipac) if self.ipac is not None else None,
            "feac": float(self.feac) if self.feac is not None else None,
            "nsc": float(self.nsc) if self.nsc is not None else None,
            "isc": float(self.isc) if self.isc is not None else None,
            "ipac1": float(self.ipac1) if self.ipac1 is not None else None,
            "ipac2": float(self.ipac2) if self.ipac2 is not None else None,
            "prnxt": float(self.prnxt) if self.prnxt is not None else None,
            "ipcb": float(self.ipcb) if self.ipcb is not None else None,
            "xa": float(self.xa) if self.xa is not None else None,
            # Performance
            "prf": self.prf.value if self.prf else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContractState:
        """Create ContractState from dictionary.

        Args:
            data: Dictionary with state variable values

        Returns:
            New ContractState instance

        Example:
            >>> data = {'tmd': '2029-01-15T00:00:00', 'nt': 100000.0, ...}
            >>> state = ContractState.from_dict(data)
        """
        tmd = ActusDateTime.from_iso(data["tmd"]) if data.get("tmd") else ActusDateTime(1970, 1, 1)
        sd = ActusDateTime.from_iso(data["sd"]) if data.get("sd") else ActusDateTime(1970, 1, 1)
        return cls(
            # Dates
            tmd=tmd,
            sd=sd,
            xd=ActusDateTime.from_iso(data["xd"]) if data.get("xd") else None,
            # Numerical values (convert to JAX arrays)
            nt=jnp.array(data["nt"]) if data.get("nt") is not None else jnp.array(0.0),
            ipnr=jnp.array(data["ipnr"]) if data.get("ipnr") is not None else jnp.array(0.0),
            ipac=jnp.array(data["ipac"]) if data.get("ipac") is not None else jnp.array(0.0),
            feac=jnp.array(data["feac"]) if data.get("feac") is not None else jnp.array(0.0),
            nsc=jnp.array(data["nsc"]) if data.get("nsc") is not None else jnp.array(1.0),
            isc=jnp.array(data["isc"]) if data.get("isc") is not None else jnp.array(1.0),
            ipac1=jnp.array(data["ipac1"]) if data.get("ipac1") is not None else None,
            ipac2=jnp.array(data["ipac2"]) if data.get("ipac2") is not None else None,
            prnxt=jnp.array(data["prnxt"]) if data.get("prnxt") is not None else None,
            ipcb=jnp.array(data["ipcb"]) if data.get("ipcb") is not None else None,
            xa=jnp.array(data["xa"]) if data.get("xa") is not None else None,
            # Performance
            prf=ContractPerformance(data["prf"]) if data.get("prf") else ContractPerformance.PF,
        )

    def __eq__(self, other: object) -> bool:
        """Check equality with another ContractState."""
        if not isinstance(other, ContractState):
            return NotImplemented

        # Compare dates
        if self.tmd != other.tmd or self.sd != other.sd or self.xd != other.xd:
            return False

        # Compare performance
        if self.prf != other.prf:
            return False

        # Compare numerical values (use jnp.allclose for floating point)
        def arrays_equal(a: jnp.ndarray | None, b: jnp.ndarray | None) -> bool:
            if a is None and b is None:
                return True
            if a is None or b is None:
                return False
            return bool(jnp.allclose(a, b))

        return (
            arrays_equal(self.nt, other.nt)
            and arrays_equal(self.ipnr, other.ipnr)
            and arrays_equal(self.ipac, other.ipac)
            and arrays_equal(self.feac, other.feac)
            and arrays_equal(self.nsc, other.nsc)
            and arrays_equal(self.isc, other.isc)
            and arrays_equal(self.ipac1, other.ipac1)
            and arrays_equal(self.ipac2, other.ipac2)
            and arrays_equal(self.prnxt, other.prnxt)
            and arrays_equal(self.ipcb, other.ipcb)
            and arrays_equal(self.xa, other.xa)
        )

    def __hash__(self) -> int:
        """Hash for use in dicts/sets."""
        # Hash based on dates and performance (arrays aren't hashable)
        return hash((self.tmd, self.sd, self.xd, self.prf))


def initialize_state(
    tmd: ActusDateTime,
    sd: ActusDateTime,
    nt: float = 0.0,
    ipnr: float = 0.0,
    prf: ContractPerformance = ContractPerformance.PF,
) -> ContractState:
    """Initialize a contract state with default values.

    Convenience function for creating a new state with sensible defaults.
    All accrued amounts start at zero, scaling multipliers at 1.0.

    Args:
        tmd: Maturity date
        sd: Status date
        nt: Notional principal
        ipnr: Nominal interest rate
        prf: Performance status

    Returns:
        New ContractState with initialized values

    Example:
        >>> state = initialize_state(
        ...     tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
        ...     sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
        ...     nt=100000.0,
        ...     ipnr=0.05,
        ... )

    References:
        ACTUS Technical Specification v1.1, Section 6.2
    """
    return ContractState(
        tmd=tmd,
        sd=sd,
        nt=jnp.array(nt),
        ipnr=jnp.array(ipnr),
        ipac=jnp.array(0.0),
        feac=jnp.array(0.0),
        nsc=jnp.array(1.0),
        isc=jnp.array(1.0),
        prf=prf,
    )


# Register ContractState as a JAX pytree
def _state_flatten(state: ContractState) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Flatten ContractState for JAX pytree registration.

    Separates JAX arrays (children) from non-array data (auxiliary).
    """
    # Children: all JAX arrays
    arrays = [
        state.nt,
        state.ipnr,
        state.ipac,
        state.feac,
        state.nsc,
        state.isc,
        state.ipac1,
        state.ipac2,
        state.prnxt,
        state.ipcb,
        state.xa,
    ]

    # Auxiliary data: dates and enums
    aux = {
        "tmd": state.tmd,
        "sd": state.sd,
        "xd": state.xd,
        "prf": state.prf,
    }

    return (tuple(arrays), aux)


def _state_unflatten(aux: dict[str, Any], arrays: tuple[Any, ...]) -> ContractState:
    """Unflatten ContractState for JAX pytree registration."""
    return ContractState(
        tmd=aux["tmd"],
        sd=aux["sd"],
        xd=aux["xd"],
        nt=arrays[0],
        ipnr=arrays[1],
        ipac=arrays[2],
        feac=arrays[3],
        nsc=arrays[4],
        isc=arrays[5],
        ipac1=arrays[6],
        ipac2=arrays[7],
        prnxt=arrays[8],
        ipcb=arrays[9],
        xa=arrays[10],
        prf=aux["prf"],
    )


# Register with JAX
jax.tree_util.register_pytree_node(  # type: ignore[type-var]
    ContractState,
    _state_flatten,
    _state_unflatten,
)
