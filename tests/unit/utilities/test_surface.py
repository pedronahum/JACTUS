"""Unit tests for 2D surface interpolation.

Tests for:
- Surface2D (numeric margins with bilinear interpolation)
- LabeledSurface2D (string-labeled margins)
"""

import jax.numpy as jnp
import pytest

from jactus.utilities.surface import LabeledSurface2D, Surface2D

# ============================================================================
# Surface2D tests
# ============================================================================


class TestSurface2DInit:
    """Test Surface2D initialization and validation."""

    def test_valid_init(self):
        """Create valid Surface2D."""
        surface = Surface2D(
            x_margins=jnp.array([0.0, 1.0]),
            y_margins=jnp.array([0.0, 1.0]),
            values=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        )
        assert surface.values.shape == (2, 2)

    def test_invalid_extrapolation(self):
        """Raises ValueError for invalid extrapolation."""
        with pytest.raises(ValueError, match="extrapolation"):
            Surface2D(
                x_margins=jnp.array([0.0, 1.0]),
                y_margins=jnp.array([0.0, 1.0]),
                values=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
                extrapolation="invalid",
            )

    def test_values_must_be_2d(self):
        """Raises ValueError if values is not 2D."""
        with pytest.raises(ValueError, match="2D"):
            Surface2D(
                x_margins=jnp.array([0.0, 1.0]),
                y_margins=jnp.array([0.0, 1.0]),
                values=jnp.array([1.0, 2.0]),
            )

    def test_x_margins_shape_mismatch(self):
        """Raises ValueError if x_margins doesn't match values rows."""
        with pytest.raises(ValueError, match="x_margins"):
            Surface2D(
                x_margins=jnp.array([0.0, 1.0, 2.0]),  # 3 values
                y_margins=jnp.array([0.0, 1.0]),
                values=jnp.array([[1.0, 2.0], [3.0, 4.0]]),  # 2 rows
            )

    def test_y_margins_shape_mismatch(self):
        """Raises ValueError if y_margins doesn't match values columns."""
        with pytest.raises(ValueError, match="y_margins"):
            Surface2D(
                x_margins=jnp.array([0.0, 1.0]),
                y_margins=jnp.array([0.0, 1.0, 2.0]),  # 3 values
                values=jnp.array([[1.0, 2.0], [3.0, 4.0]]),  # 2 columns
            )

    def test_minimum_margins(self):
        """Requires at least 2 values in each margin."""
        with pytest.raises(ValueError, match="at least 2"):
            Surface2D(
                x_margins=jnp.array([0.0]),
                y_margins=jnp.array([0.0, 1.0]),
                values=jnp.array([[1.0, 2.0]]),
            )


class TestSurface2DEvaluate:
    """Test Surface2D evaluation (bilinear interpolation)."""

    @pytest.fixture
    def simple_surface(self):
        """2x2 surface for testing interpolation."""
        return Surface2D(
            x_margins=jnp.array([0.0, 1.0]),
            y_margins=jnp.array([0.0, 1.0]),
            values=jnp.array([[0.0, 2.0], [4.0, 6.0]]),
        )

    @pytest.fixture
    def larger_surface(self):
        """3x3 surface for more interpolation tests."""
        return Surface2D(
            x_margins=jnp.array([0.0, 1.0, 2.0]),
            y_margins=jnp.array([0.0, 1.0, 2.0]),
            values=jnp.array([
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
            ]),
        )

    def test_evaluate_at_corner(self, simple_surface):
        """Evaluate at grid corner returns exact value."""
        assert float(simple_surface.evaluate(0.0, 0.0)) == pytest.approx(0.0)
        assert float(simple_surface.evaluate(1.0, 0.0)) == pytest.approx(4.0)
        assert float(simple_surface.evaluate(0.0, 1.0)) == pytest.approx(2.0)
        assert float(simple_surface.evaluate(1.0, 1.0)) == pytest.approx(6.0)

    def test_evaluate_at_center(self, simple_surface):
        """Bilinear interpolation at center of grid."""
        # At (0.5, 0.5): average of all four corners = (0+2+4+6)/4 = 3.0
        result = float(simple_surface.evaluate(0.5, 0.5))
        assert result == pytest.approx(3.0)

    def test_evaluate_along_x_edge(self, simple_surface):
        """Interpolation along x-axis edge (y=0)."""
        # At (0.5, 0.0): midpoint between 0.0 and 4.0 = 2.0
        result = float(simple_surface.evaluate(0.5, 0.0))
        assert result == pytest.approx(2.0)

    def test_evaluate_along_y_edge(self, simple_surface):
        """Interpolation along y-axis edge (x=0)."""
        # At (0.0, 0.5): midpoint between 0.0 and 2.0 = 1.0
        result = float(simple_surface.evaluate(0.0, 0.5))
        assert result == pytest.approx(1.0)

    def test_evaluate_in_second_cell(self, larger_surface):
        """Interpolation in second grid cell."""
        # At (1.5, 1.5): midpoint of cell (1,1)-(2,2)
        # corners: 4.0, 5.0, 7.0, 8.0 → average = 6.0
        result = float(larger_surface.evaluate(1.5, 1.5))
        assert result == pytest.approx(6.0)

    def test_extrapolation_constant(self, simple_surface):
        """Constant extrapolation clamps to edge values."""
        # Below minimum x
        result = float(simple_surface.evaluate(-1.0, 0.5))
        expected = float(simple_surface.evaluate(0.0, 0.5))
        assert result == pytest.approx(expected)

        # Above maximum x
        result = float(simple_surface.evaluate(2.0, 0.5))
        expected = float(simple_surface.evaluate(1.0, 0.5))
        assert result == pytest.approx(expected)

    def test_extrapolation_raise(self):
        """Raise extrapolation raises ValueError for out-of-bounds."""
        surface = Surface2D(
            x_margins=jnp.array([0.0, 1.0]),
            y_margins=jnp.array([0.0, 1.0]),
            values=jnp.array([[0.0, 1.0], [2.0, 3.0]]),
            extrapolation="raise",
        )
        with pytest.raises(ValueError, match="outside the grid"):
            surface.evaluate(-0.1, 0.5)
        with pytest.raises(ValueError, match="outside the grid"):
            surface.evaluate(0.5, 1.5)

    def test_returns_jax_array(self, simple_surface):
        """evaluate returns a JAX array."""
        result = simple_surface.evaluate(0.5, 0.5)
        assert isinstance(result, jnp.ndarray)


class TestSurface2DSerialization:
    """Test Surface2D serialization."""

    def test_to_dict(self):
        """to_dict produces valid dictionary."""
        surface = Surface2D(
            x_margins=jnp.array([0.0, 1.0]),
            y_margins=jnp.array([0.0, 1.0]),
            values=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        )
        data = surface.to_dict()
        assert data["x_margins"] == [0.0, 1.0]
        assert data["y_margins"] == [0.0, 1.0]
        assert data["values"] == [[1.0, 2.0], [3.0, 4.0]]
        assert data["extrapolation"] == "constant"

    def test_from_dict(self):
        """from_dict recreates equivalent surface."""
        data = {
            "x_margins": [0.0, 1.0],
            "y_margins": [0.0, 1.0],
            "values": [[1.0, 2.0], [3.0, 4.0]],
            "extrapolation": "constant",
        }
        surface = Surface2D.from_dict(data)
        assert float(surface.evaluate(0.0, 0.0)) == pytest.approx(1.0)
        assert float(surface.evaluate(1.0, 1.0)) == pytest.approx(4.0)

    def test_roundtrip(self):
        """to_dict → from_dict preserves values."""
        original = Surface2D(
            x_margins=jnp.array([0.0, 1.0, 2.0]),
            y_margins=jnp.array([0.0, 0.5, 1.0]),
            values=jnp.array([
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]),
        )
        restored = Surface2D.from_dict(original.to_dict())
        for x in [0.0, 0.5, 1.0, 1.5, 2.0]:
            for y in [0.0, 0.25, 0.5, 0.75, 1.0]:
                assert float(restored.evaluate(x, y)) == pytest.approx(
                    float(original.evaluate(x, y)), abs=1e-5
                )


# ============================================================================
# LabeledSurface2D tests
# ============================================================================


class TestLabeledSurface2D:
    """Test LabeledSurface2D."""

    @pytest.fixture
    def labeled_surface(self):
        """Sample labeled surface."""
        return LabeledSurface2D(
            x_labels=["DEPOSIT-001", "DEPOSIT-002"],
            y_labels=["2024-Q1", "2024-Q2", "2024-Q3"],
            values=jnp.array([
                [1000.0, -500.0, 2000.0],
                [5000.0, -1000.0, 3000.0],
            ]),
        )

    def test_get_exact_value(self, labeled_surface):
        """Get value at exact label coordinates."""
        result = labeled_surface.get("DEPOSIT-001", "2024-Q1")
        assert float(result) == pytest.approx(1000.0)

    def test_get_withdrawal(self, labeled_surface):
        """Get negative (withdrawal) value."""
        result = labeled_surface.get("DEPOSIT-002", "2024-Q2")
        assert float(result) == pytest.approx(-1000.0)

    def test_get_unknown_x_label(self, labeled_surface):
        """Raises KeyError for unknown x label."""
        with pytest.raises(KeyError, match="DEPOSIT-999"):
            labeled_surface.get("DEPOSIT-999", "2024-Q1")

    def test_get_unknown_y_label(self, labeled_surface):
        """Raises KeyError for unknown y label."""
        with pytest.raises(KeyError, match="2024-Q4"):
            labeled_surface.get("DEPOSIT-001", "2024-Q4")

    def test_get_row(self, labeled_surface):
        """Get all values for an x label."""
        row = labeled_surface.get_row("DEPOSIT-001")
        assert row.shape == (3,)
        assert float(row[0]) == pytest.approx(1000.0)

    def test_get_column(self, labeled_surface):
        """Get all values for a y label."""
        col = labeled_surface.get_column("2024-Q1")
        assert col.shape == (2,)
        assert float(col[0]) == pytest.approx(1000.0)
        assert float(col[1]) == pytest.approx(5000.0)

    def test_dimension_mismatch(self):
        """Raises ValueError for dimension mismatch."""
        with pytest.raises(ValueError, match="x_labels"):
            LabeledSurface2D(
                x_labels=["A", "B", "C"],  # 3 labels
                y_labels=["X", "Y"],
                values=jnp.array([[1.0, 2.0], [3.0, 4.0]]),  # 2 rows
            )

    def test_serialization_roundtrip(self, labeled_surface):
        """to_dict → from_dict preserves data."""
        data = labeled_surface.to_dict()
        restored = LabeledSurface2D.from_dict(data)
        assert restored.x_labels == labeled_surface.x_labels
        assert restored.y_labels == labeled_surface.y_labels
        assert float(restored.get("DEPOSIT-001", "2024-Q1")) == pytest.approx(1000.0)
