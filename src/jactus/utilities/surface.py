"""2D surface interpolation for behavioral risk models.

This module provides JAX-compatible 2D surface interpolation, used primarily
by behavioral risk models such as prepayment and deposit transaction models.

A ``Surface2D`` represents a function f(x, y) defined on a grid of (x, y)
margin values with corresponding z values. Given arbitrary query points
(x_q, y_q), it returns interpolated values using bilinear interpolation.

Key features:
- Bilinear interpolation on 2D grids
- Configurable extrapolation: ``"constant"`` (nearest edge) or ``"raise"``
- JAX array storage for automatic differentiation compatibility
- Support for both numeric and label-based margins via ``LabeledSurface2D``

References:
    ACTUS Risk Service v2.0 - TimeSeries<TimeSeries<Double>> surface model
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp


@dataclass(frozen=True)
class Surface2D:
    """A 2D interpolation surface defined on a rectangular grid.

    The surface is specified by:
    - ``x_margins``: Sorted 1D array of x-axis breakpoints (e.g., spread values)
    - ``y_margins``: Sorted 1D array of y-axis breakpoints (e.g., age values)
    - ``values``: 2D array of shape ``(len(x_margins), len(y_margins))``

    Interpolation is bilinear within the grid, and extrapolation behavior is
    configurable.

    Attributes:
        x_margins: Sorted 1D array of x-axis breakpoints.
        y_margins: Sorted 1D array of y-axis breakpoints.
        values: 2D array of shape ``(len(x_margins), len(y_margins))``.
        extrapolation: Extrapolation method: ``"constant"`` (use nearest edge
            value) or ``"raise"`` (raise ValueError).

    Example:
        >>> import jax.numpy as jnp
        >>> # Prepayment surface: spread (rows) × age (columns)
        >>> surface = Surface2D(
        ...     x_margins=jnp.array([0.0, 1.0, 2.0, 3.0]),   # spread %
        ...     y_margins=jnp.array([0.0, 1.0, 3.0, 5.0]),    # age years
        ...     values=jnp.array([
        ...         [0.00, 0.00, 0.00, 0.00],  # spread=0%
        ...         [0.00, 0.01, 0.02, 0.00],  # spread=1%
        ...         [0.00, 0.02, 0.05, 0.01],  # spread=2%
        ...         [0.01, 0.05, 0.10, 0.02],  # spread=3%
        ...     ]),
        ... )
        >>> rate = surface.evaluate(1.5, 2.0)  # Bilinear interpolation
    """

    x_margins: jnp.ndarray
    y_margins: jnp.ndarray
    values: jnp.ndarray
    extrapolation: str = "constant"

    def __post_init__(self) -> None:
        """Validate surface dimensions and parameters."""
        if self.extrapolation not in ("constant", "raise"):
            raise ValueError(
                f"extrapolation must be 'constant' or 'raise', got '{self.extrapolation}'"
            )
        if self.values.ndim != 2:
            raise ValueError(f"values must be 2D, got shape {self.values.shape}")
        if self.values.shape[0] != self.x_margins.shape[0]:
            raise ValueError(
                f"values rows ({self.values.shape[0]}) must match "
                f"x_margins length ({self.x_margins.shape[0]})"
            )
        if self.values.shape[1] != self.y_margins.shape[0]:
            raise ValueError(
                f"values columns ({self.values.shape[1]}) must match "
                f"y_margins length ({self.y_margins.shape[0]})"
            )
        if self.x_margins.shape[0] < 2:
            raise ValueError("x_margins must have at least 2 values")
        if self.y_margins.shape[0] < 2:
            raise ValueError("y_margins must have at least 2 values")

    def evaluate(self, x: float, y: float) -> jnp.ndarray:
        """Evaluate the surface at a point using bilinear interpolation.

        For points within the grid, standard bilinear interpolation is used.
        For points outside the grid, behavior depends on ``self.extrapolation``.

        Args:
            x: X-axis query value.
            y: Y-axis query value.

        Returns:
            Interpolated value as a scalar JAX array.

        Raises:
            ValueError: If ``extrapolation="raise"`` and point is outside the grid.

        Example:
            >>> value = surface.evaluate(1.5, 2.0)
        """
        x_arr = [float(v) for v in self.x_margins]
        y_arr = [float(v) for v in self.y_margins]

        # Handle extrapolation
        x_clamped, y_clamped = float(x), float(y)

        if x_clamped < x_arr[0] or x_clamped > x_arr[-1]:
            if self.extrapolation == "raise":
                raise ValueError(
                    f"x={x} is outside the grid [{x_arr[0]}, {x_arr[-1]}]"
                )
            x_clamped = max(x_arr[0], min(x_arr[-1], x_clamped))

        if y_clamped < y_arr[0] or y_clamped > y_arr[-1]:
            if self.extrapolation == "raise":
                raise ValueError(
                    f"y={y} is outside the grid [{y_arr[0]}, {y_arr[-1]}]"
                )
            y_clamped = max(y_arr[0], min(y_arr[-1], y_clamped))

        # Find x interval
        xi = bisect.bisect_right(x_arr, x_clamped) - 1
        xi = max(0, min(xi, len(x_arr) - 2))

        # Find y interval
        yi = bisect.bisect_right(y_arr, y_clamped) - 1
        yi = max(0, min(yi, len(y_arr) - 2))

        # Bilinear interpolation fractions
        x0, x1 = x_arr[xi], x_arr[xi + 1]
        y0, y1 = y_arr[yi], y_arr[yi + 1]

        x_frac = (x_clamped - x0) / (x1 - x0) if x1 != x0 else 0.0
        y_frac = (y_clamped - y0) / (y1 - y0) if y1 != y0 else 0.0

        # Four corner values
        q00 = float(self.values[xi, yi])
        q10 = float(self.values[xi + 1, yi])
        q01 = float(self.values[xi, yi + 1])
        q11 = float(self.values[xi + 1, yi + 1])

        # Bilinear interpolation
        result = (
            q00 * (1 - x_frac) * (1 - y_frac)
            + q10 * x_frac * (1 - y_frac)
            + q01 * (1 - x_frac) * y_frac
            + q11 * x_frac * y_frac
        )

        return jnp.array(result, dtype=jnp.float32)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Surface2D:
        """Create a Surface2D from a dictionary representation.

        Args:
            data: Dictionary with keys ``"x_margins"``, ``"y_margins"``,
                ``"values"``, and optionally ``"extrapolation"``.

        Returns:
            New Surface2D instance.

        Example:
            >>> surface = Surface2D.from_dict({
            ...     "x_margins": [0.0, 1.0, 2.0],
            ...     "y_margins": [0.0, 1.0, 3.0],
            ...     "values": [
            ...         [0.0, 0.01, 0.02],
            ...         [0.01, 0.03, 0.05],
            ...         [0.02, 0.05, 0.10],
            ...     ],
            ...     "extrapolation": "constant",
            ... })
        """
        return Surface2D(
            x_margins=jnp.array(data["x_margins"], dtype=jnp.float32),
            y_margins=jnp.array(data["y_margins"], dtype=jnp.float32),
            values=jnp.array(data["values"], dtype=jnp.float32),
            extrapolation=data.get("extrapolation", "constant"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize surface to a dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "x_margins": [float(v) for v in self.x_margins],
            "y_margins": [float(v) for v in self.y_margins],
            "values": [[float(v) for v in row] for row in self.values],
            "extrapolation": self.extrapolation,
        }


@dataclass
class LabeledSurface2D:
    """A 2D surface with string-labeled margins.

    Used for models where one or both dimensions are categorical rather than
    numeric (e.g., deposit transaction models where dimension 1 is contract ID
    and dimension 2 is a date label).

    Each label maps to a numeric index. The underlying data is stored in a
    regular ``Surface2D`` for interpolation.

    Attributes:
        x_labels: Ordered list of labels for the x-axis.
        y_labels: Ordered list of labels for the y-axis.
        values: 2D array of shape ``(len(x_labels), len(y_labels))``.

    Example:
        >>> surface = LabeledSurface2D(
        ...     x_labels=["DEPOSIT-001", "DEPOSIT-002"],
        ...     y_labels=["2024-01-01", "2024-07-01", "2025-01-01"],
        ...     values=jnp.array([
        ...         [1000.0, -500.0, 2000.0],
        ...         [5000.0, -1000.0, 3000.0],
        ...     ]),
        ... )
        >>> amount = surface.get(x_label="DEPOSIT-001", y_label="2024-07-01")
    """

    x_labels: list[str]
    y_labels: list[str]
    values: jnp.ndarray

    def __post_init__(self) -> None:
        """Validate dimensions."""
        if self.values.ndim != 2:
            raise ValueError(f"values must be 2D, got shape {self.values.shape}")
        if self.values.shape[0] != len(self.x_labels):
            raise ValueError(
                f"values rows ({self.values.shape[0]}) must match "
                f"x_labels length ({len(self.x_labels)})"
            )
        if self.values.shape[1] != len(self.y_labels):
            raise ValueError(
                f"values columns ({self.values.shape[1]}) must match "
                f"y_labels length ({len(self.y_labels)})"
            )
        self._x_index = {label: i for i, label in enumerate(self.x_labels)}
        self._y_index = {label: i for i, label in enumerate(self.y_labels)}

    def get(self, x_label: str, y_label: str) -> jnp.ndarray:
        """Get the value at exact label coordinates.

        Args:
            x_label: X-axis label.
            y_label: Y-axis label.

        Returns:
            Value as a scalar JAX array.

        Raises:
            KeyError: If either label is not found.

        Example:
            >>> value = surface.get("DEPOSIT-001", "2024-07-01")
        """
        if x_label not in self._x_index:
            raise KeyError(f"x_label '{x_label}' not found. Available: {self.x_labels}")
        if y_label not in self._y_index:
            raise KeyError(f"y_label '{y_label}' not found. Available: {self.y_labels}")

        xi = self._x_index[x_label]
        yi = self._y_index[y_label]
        return self.values[xi, yi]

    def get_row(self, x_label: str) -> jnp.ndarray:
        """Get all values for a given x-axis label.

        Args:
            x_label: X-axis label.

        Returns:
            1D array of values for that row.

        Raises:
            KeyError: If label not found.
        """
        if x_label not in self._x_index:
            raise KeyError(f"x_label '{x_label}' not found. Available: {self.x_labels}")
        return self.values[self._x_index[x_label]]

    def get_column(self, y_label: str) -> jnp.ndarray:
        """Get all values for a given y-axis label.

        Args:
            y_label: Y-axis label.

        Returns:
            1D array of values for that column.

        Raises:
            KeyError: If label not found.
        """
        if y_label not in self._y_index:
            raise KeyError(f"y_label '{y_label}' not found. Available: {self.y_labels}")
        return self.values[:, self._y_index[y_label]]

    @staticmethod
    def from_dict(data: dict[str, Any]) -> LabeledSurface2D:
        """Create from dictionary representation.

        Args:
            data: Dictionary with keys ``"x_labels"``, ``"y_labels"``, ``"values"``.

        Returns:
            New LabeledSurface2D instance.
        """
        return LabeledSurface2D(
            x_labels=data["x_labels"],
            y_labels=data["y_labels"],
            values=jnp.array(data["values"], dtype=jnp.float32),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "x_labels": self.x_labels,
            "y_labels": self.y_labels,
            "values": [[float(v) for v in row] for row in self.values],
        }
