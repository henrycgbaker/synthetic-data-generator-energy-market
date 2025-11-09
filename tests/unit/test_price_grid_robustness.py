"""
Test equilibrium finding with different price grid configurations.
Ensures solver works across different resolutions, ranges, and scales.
"""

import numpy as np
import pandas as pd
import pytest
from synthetic_data_pkg.simulate import find_equilibrium
from synthetic_data_pkg.demand import DemandCurve
from synthetic_data_pkg.supply import SupplyCurve
from synthetic_data_pkg.config import DemandConfig, TopConfig


@pytest.mark.unit
class TestPriceGridResolution:
    """Test equilibrium finding with different grid resolutions"""

    @pytest.mark.parametrize("grid_step", [1, 2, 5, 10, 20, 50, 100])
    def test_equilibrium_converges_with_different_resolutions(self, grid_step):
        """Test that finer/coarser grids produce consistent equilibria"""
        # Setup standard scenario
        demand_cfg = DemandConfig(
            inelastic=False,
            base_intercept=200.0,
            slope=-0.006,
            daily_seasonality=False,
            annual_seasonality=False,
        )
        demand = DemandCurve(demand_cfg)

        config = TopConfig(
            start_ts="2024-01-01",
            days=1,
            supply_regime_planner={"mode": "local_only"},
            renewable_availability_mode="weather_simulation",
            variables={
                "fuel.gas": {"regimes": [{"name": "s", "dist": {"kind": "const", "v": 30.0}}]},
                "fuel.coal": {"regimes": [{"name": "s", "dist": {"kind": "const", "v": 25.0}}]},
            },
        )
        supply = SupplyCurve(config, rng_seed=42)

        vals = _get_standard_vals()
        ts = pd.Timestamp("2024-01-01 12:00")

        # Test with different grid resolutions
        price_grid = np.array(list(range(-100, 201, grid_step)), dtype=float)

        q_star, p_star = find_equilibrium(ts, demand, supply, vals, price_grid)

        # Assertions
        assert not np.isnan(q_star), f"NaN quantity with grid_step={grid_step}"
        assert not np.isnan(p_star), f"NaN price with grid_step={grid_step}"
        assert q_star > 0, f"Non-positive quantity with grid_step={grid_step}"
        assert price_grid[0] <= p_star <= price_grid[-1], f"Price outside grid with step={grid_step}"

    @pytest.mark.parametrize("grid_step", [1, 5, 10, 20])
    def test_equilibrium_price_consistency_across_resolutions(self, grid_step):
        """Test that equilibrium prices are consistent across different resolutions (within tolerance)"""
        # Run equilibrium with different resolutions
        demand_cfg = DemandConfig(inelastic=False, base_intercept=200.0, slope=-0.006,
                                   daily_seasonality=False, annual_seasonality=False)
        demand = DemandCurve(demand_cfg)
        config = TopConfig(
            start_ts="2024-01-01", days=1,
            supply_regime_planner={"mode": "local_only"},
            renewable_availability_mode="weather_simulation",
            variables={
                "fuel.gas": {"regimes": [{"name": "s", "dist": {"kind": "const", "v": 30.0}}]},
                "fuel.coal": {"regimes": [{"name": "s", "dist": {"kind": "const", "v": 25.0}}]},
            },
        )
        supply = SupplyCurve(config, rng_seed=42)
        vals = _get_standard_vals()
        ts = pd.Timestamp("2024-01-01 12:00")

        price_grid = np.array(list(range(-100, 201, grid_step)), dtype=float)
        q_star, p_star = find_equilibrium(ts, demand, supply, vals, price_grid)

        # Compare with baseline (step=10)
        baseline_grid = np.array(list(range(-100, 201, 10)), dtype=float)
        q_baseline, p_baseline = find_equilibrium(ts, demand, supply, vals, baseline_grid)

        # Prices should be within one grid step of each other
        tolerance = max(grid_step, 10) + 5  # Allow for interpolation differences
        assert abs(p_star - p_baseline) < tolerance, \
            f"Price difference {abs(p_star - p_baseline)} exceeds tolerance {tolerance} for step={grid_step}"

        # Quantities should be similar (within 5%)
        assert abs(q_star - q_baseline) / q_baseline < 0.05, \
            f"Quantity difference too large for step={grid_step}"


@pytest.mark.unit
class TestPriceGridRange:
    """Test equilibrium finding with different price ranges"""

    @pytest.mark.parametrize("price_min,price_max", [
        (-200, 100),   # Shifted negative
        (0, 300),      # All positive
        (-500, 500),   # Wide range
        (-50, 150),    # Narrow range
        (20, 120),     # Above zero only
    ])
    def test_equilibrium_with_different_price_ranges(self, price_min, price_max):
        """Test equilibrium finding with different price range configurations"""
        demand_cfg = DemandConfig(inelastic=False, base_intercept=200.0, slope=-0.006,
                                   daily_seasonality=False, annual_seasonality=False)
        demand = DemandCurve(demand_cfg)
        config = TopConfig(
            start_ts="2024-01-01", days=1,
            supply_regime_planner={"mode": "local_only"},
            renewable_availability_mode="weather_simulation",
            variables={
                "fuel.gas": {"regimes": [{"name": "s", "dist": {"kind": "const", "v": 30.0}}]},
                "fuel.coal": {"regimes": [{"name": "s", "dist": {"kind": "const", "v": 25.0}}]},
            },
        )
        supply = SupplyCurve(config, rng_seed=42)
        vals = _get_standard_vals()
        ts = pd.Timestamp("2024-01-01 12:00")

        price_grid = np.array(list(range(price_min, price_max + 1, 10)), dtype=float)

        q_star, p_star = find_equilibrium(ts, demand, supply, vals, price_grid)

        # Should find valid equilibrium if range contains it
        assert not np.isnan(q_star), f"Failed with range [{price_min}, {price_max}]"
        assert not np.isnan(p_star), f"Failed with range [{price_min}, {price_max}]"
        assert price_min <= p_star <= price_max, f"Price {p_star} outside range [{price_min}, {price_max}]"


@pytest.mark.unit
class TestPriceGridEdgeCases:
    """Test edge cases in price grid configuration"""

    def test_very_coarse_grid(self):
        """Test with very coarse grid (step=100) - should still find equilibrium"""
        demand_cfg = DemandConfig(inelastic=False, base_intercept=500.0, slope=-0.01,
                                   daily_seasonality=False, annual_seasonality=False)
        demand = DemandCurve(demand_cfg)
        config = TopConfig(
            start_ts="2024-01-01", days=1,
            supply_regime_planner={"mode": "local_only"},
            renewable_availability_mode="weather_simulation",
            variables={
                "fuel.gas": {"regimes": [{"name": "s", "dist": {"kind": "const", "v": 30.0}}]},
                "fuel.coal": {"regimes": [{"name": "s", "dist": {"kind": "const", "v": 25.0}}]},
            },
        )
        supply = SupplyCurve(config, rng_seed=42)
        vals = _get_standard_vals()
        ts = pd.Timestamp("2024-01-01 12:00")

        price_grid = np.array([0, 100, 200, 300, 400, 500], dtype=float)
        q_star, p_star = find_equilibrium(ts, demand, supply, vals, price_grid)

        assert not np.isnan(q_star)
        assert not np.isnan(p_star)

    def test_very_fine_grid(self):
        """Test with very fine grid (step=0.1) - should handle precision"""
        demand_cfg = DemandConfig(inelastic=False, base_intercept=200.0, slope=-0.006,
                                   daily_seasonality=False, annual_seasonality=False)
        demand = DemandCurve(demand_cfg)
        config = TopConfig(
            start_ts="2024-01-01", days=1,
            supply_regime_planner={"mode": "local_only"},
            renewable_availability_mode="weather_simulation",
            variables={
                "fuel.gas": {"regimes": [{"name": "s", "dist": {"kind": "const", "v": 30.0}}]},
                "fuel.coal": {"regimes": [{"name": "s", "dist": {"kind": "const", "v": 25.0}}]},
            },
        )
        supply = SupplyCurve(config, rng_seed=42)
        vals = _get_standard_vals()
        ts = pd.Timestamp("2024-01-01 12:00")

        price_grid = np.arange(0, 200.1, 0.1)
        q_star, p_star = find_equilibrium(ts, demand, supply, vals, price_grid)

        assert not np.isnan(q_star)
        assert not np.isnan(p_star)


def _get_standard_vals():
    """Helper to get standard variable values for testing"""
    return {
        "cap.nuclear": 6000.0,
        "avail.nuclear": 0.95,
        "cap.wind": 7000.0,
        "cap.solar": 5000.0,
        "cap.coal": 8000.0,
        "avail.coal": 0.90,
        "cap.gas": 12000.0,
        "avail.gas": 0.95,
        "fuel.coal": 25.0,
        "fuel.gas": 30.0,
        "eta_lb.coal": 0.33,
        "eta_ub.coal": 0.38,
        "eta_lb.gas": 0.48,
        "eta_ub.gas": 0.55,
        "bid.nuclear.min": -200.0,
        "bid.nuclear.max": -50.0,
        "bid.wind.min": -200.0,
        "bid.wind.max": -50.0,
        "bid.solar.min": -200.0,
        "bid.solar.max": -50.0,
    }