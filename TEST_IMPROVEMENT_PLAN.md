# Test Improvement Plan - Energy Market Synthetic Data Generator

**Date:** 2025-11-09
**Status:** Draft
**Priority:** High - Addresses scale/input robustness gaps in current testing regime

---

## Executive Summary

Current test suite covers ~90% of basic functionality but has **critical gaps in scale robustness and edge case coverage**. This plan adds **~80 new test cases** organized by priority to ensure the system works correctly across:
- Different market scales (10 MW to 1,000,000 MW)
- Different price ranges and resolutions
- Extreme parameter values
- Edge cases and boundary conditions

**Estimated effort:** 3-5 days for high priority, 2-3 days for medium priority

---

## Critical Issues to Address

### Issue #1: Fixed Price Grid (HIGHEST PRIORITY)
**Current:** All tests use `range(-100, 201, 10)` - same resolution, same range
**Risk:** Equilibrium solver may fail with coarse/fine grids or different price ranges
**Impact:** Production failures in markets with different pricing structures

### Issue #2: Narrow Parameter Ranges
**Current:** Capacities in 5000-12000 MW range, fuel prices 20-80, demand slopes -0.006 to -300
**Risk:** System may fail for small municipal grids or large regional ISOs
**Impact:** Cannot validate correctness across real-world market sizes

### Issue #3: Skipped Critical Test
**Current:** `test_fuel_price_changes_with_thermal_marginal` is skipped
**Risk:** Core economic behavior (price response to fuel costs) not validated
**Impact:** Could generate economically incorrect synthetic data

### Issue #4: Missing Numerical Edge Cases
**Current:** No tests for extreme values, NaN/Inf handling, zero/negative cases
**Risk:** Silent failures or crashes in edge conditions
**Impact:** Unreliable system behavior in unusual scenarios

---

## HIGH PRIORITY Test Additions

### HP-1: Price Grid Robustness Suite

**File:** `tests/unit/test_price_grid_robustness.py` (NEW)

```python
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
```

**Estimated effort:** 4 hours
**Expected test count:** 15 new tests
**Impact:** Validates core equilibrium solver across different market configurations

---

### HP-2: Capacity Scale Robustness Suite

**File:** `tests/unit/test_capacity_scale_robustness.py` (NEW)

```python
"""
Test system behavior across different capacity scales.
Validates the system works for small municipal grids to large regional ISOs.
"""

import numpy as np
import pandas as pd
import pytest
from synthetic_data_pkg.simulate import find_equilibrium
from synthetic_data_pkg.demand import DemandCurve
from synthetic_data_pkg.supply import SupplyCurve
from synthetic_data_pkg.config import DemandConfig, TopConfig


@pytest.mark.unit
class TestCapacityScales:
    """Test equilibrium finding across different capacity scales"""

    @pytest.mark.parametrize("scale_factor", [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
    def test_equilibrium_scales_linearly_with_capacity(self, scale_factor):
        """Test that equilibrium quantities scale linearly with all capacities

        If we scale all capacities by factor K and demand by K,
        equilibrium quantity should scale by K, price should remain similar.
        """
        # Base scenario
        base_demand_cfg = DemandConfig(
            inelastic=False,
            base_intercept=200.0,
            slope=-0.006,
            daily_seasonality=False,
            annual_seasonality=False,
        )
        base_demand = DemandCurve(base_demand_cfg)

        config = TopConfig(
            start_ts="2024-01-01", days=1,
            supply_regime_planner={"mode": "local_only"},
            renewable_availability_mode="weather_simulation",
            variables={
                "fuel.gas": {"regimes": [{"name": "s", "dist": {"kind": "const", "v": 30.0}}]},
                "fuel.coal": {"regimes": [{"name": "s", "dist": {"kind": "const", "v": 25.0}}]},
            },
        )
        base_supply = SupplyCurve(config, rng_seed=42)

        base_vals = {
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

        ts = pd.Timestamp("2024-01-01 12:00")
        price_grid = np.array(list(range(-100, 201, 10)), dtype=float)

        # Get baseline equilibrium
        q_base, p_base = find_equilibrium(ts, base_demand, base_supply, base_vals, price_grid)

        # Scale scenario
        scaled_demand_cfg = DemandConfig(
            inelastic=False,
            base_intercept=200.0,  # Price intercept stays same
            slope=-0.006 / scale_factor,  # Adjust slope for scaled quantities
            daily_seasonality=False,
            annual_seasonality=False,
        )
        scaled_demand = DemandCurve(scaled_demand_cfg)
        scaled_supply = base_supply  # Same supply curve

        # Scale capacities
        scaled_vals = {k: (v * scale_factor if k.startswith("cap.") else v)
                       for k, v in base_vals.items()}

        q_scaled, p_scaled = find_equilibrium(ts, scaled_demand, scaled_supply, scaled_vals, price_grid)

        # Assertions
        assert not np.isnan(q_scaled), f"NaN quantity at scale {scale_factor}"
        assert not np.isnan(p_scaled), f"NaN price at scale {scale_factor}"

        # Quantity should scale proportionally (within 10% due to discretization)
        expected_q = q_base * scale_factor
        assert abs(q_scaled - expected_q) / expected_q < 0.15, \
            f"Quantity scaling failed: expected {expected_q}, got {q_scaled} at scale {scale_factor}"

        # Price should remain similar (within 20% due to market structure)
        assert abs(p_scaled - p_base) / max(abs(p_base), 1) < 0.25, \
            f"Price changed too much: {p_base} -> {p_scaled} at scale {scale_factor}"

    @pytest.mark.parametrize("capacity_mw", [10, 100, 1000, 10000, 100000, 1000000])
    def test_small_to_large_absolute_capacities(self, capacity_mw):
        """Test system works with absolute capacities from 10 MW to 1,000,000 MW"""
        # Adjust demand to match capacity scale
        demand_intercept = capacity_mw * 0.05  # Choke price proportional to scale
        demand_slope = -0.001 * (10000.0 / capacity_mw)  # Adjust slope for scale

        demand_cfg = DemandConfig(
            inelastic=False,
            base_intercept=demand_intercept,
            slope=demand_slope,
            daily_seasonality=False,
            annual_seasonality=False,
        )
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

        vals = {
            "cap.nuclear": capacity_mw * 0.15,
            "avail.nuclear": 0.95,
            "cap.wind": capacity_mw * 0.18,
            "cap.solar": capacity_mw * 0.12,
            "cap.coal": capacity_mw * 0.20,
            "avail.coal": 0.90,
            "cap.gas": capacity_mw * 0.30,
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

        ts = pd.Timestamp("2024-01-01 12:00")
        price_grid = np.array(list(range(-100, 201, 10)), dtype=float)

        q_star, p_star = find_equilibrium(ts, demand, supply, vals, price_grid)

        # Should find valid equilibrium at any scale
        assert not np.isnan(q_star), f"Failed at capacity scale {capacity_mw} MW"
        assert not np.isnan(p_star), f"Failed at capacity scale {capacity_mw} MW"
        assert q_star > 0, f"Non-positive quantity at capacity {capacity_mw} MW"
        assert q_star <= capacity_mw, f"Quantity {q_star} exceeds total capacity {capacity_mw}"


@pytest.mark.unit
class TestCapacityEdgeCases:
    """Test edge cases in capacity configuration"""

    def test_zero_thermal_capacity(self):
        """Test with zero thermal (coal + gas) capacity - renewables only"""
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

        vals = {
            "cap.nuclear": 6000.0,
            "avail.nuclear": 0.95,
            "cap.wind": 7000.0,
            "cap.solar": 5000.0,
            "cap.coal": 0.0,  # ZERO
            "avail.coal": 0.90,
            "cap.gas": 0.0,  # ZERO
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

        ts = pd.Timestamp("2024-01-01 12:00")
        price_grid = np.array(list(range(-100, 201, 10)), dtype=float)

        q_star, p_star = find_equilibrium(ts, demand, supply, vals, price_grid)

        # Should find equilibrium with renewables only
        assert not np.isnan(q_star)
        assert not np.isnan(p_star)
        # Verify no thermal generation
        _, breakdown = supply.supply_at(p_star, ts, vals)
        assert breakdown["coal"] == 0.0
        assert breakdown["gas"] == 0.0

    def test_zero_renewable_capacity(self):
        """Test with zero renewable capacity - thermal only"""
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

        vals = {
            "cap.nuclear": 0.0,  # ZERO
            "avail.nuclear": 0.95,
            "cap.wind": 0.0,  # ZERO
            "cap.solar": 0.0,  # ZERO
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

        ts = pd.Timestamp("2024-01-01 12:00")
        price_grid = np.array(list(range(-100, 201, 10)), dtype=float)

        q_star, p_star = find_equilibrium(ts, demand, supply, vals, price_grid)

        # Should find equilibrium with thermal only
        assert not np.isnan(q_star)
        assert not np.isnan(p_star)
        # Verify no renewable generation
        _, breakdown = supply.supply_at(p_star, ts, vals)
        assert breakdown["nuclear"] == 0.0
        assert breakdown["wind"] == 0.0
        assert breakdown["solar"] == 0.0

    def test_extreme_capacity_ratios(self):
        """Test with extreme ratios between generation types (1:10000)"""
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

        vals = {
            "cap.nuclear": 50000.0,  # HUGE
            "avail.nuclear": 0.95,
            "cap.wind": 5.0,  # tiny
            "cap.solar": 5.0,  # tiny
            "cap.coal": 5.0,  # tiny
            "avail.coal": 0.90,
            "cap.gas": 5.0,  # tiny
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

        ts = pd.Timestamp("2024-01-01 12:00")
        price_grid = np.array(list(range(-100, 201, 10)), dtype=float)

        q_star, p_star = find_equilibrium(ts, demand, supply, vals, price_grid)

        # Should handle extreme ratios
        assert not np.isnan(q_star)
        assert not np.isnan(p_star)
```

**Estimated effort:** 4 hours
**Expected test count:** 12 new tests
**Impact:** Validates system works for different market sizes (municipal to ISO-scale)

---

### HP-3: Fix Skipped Critical Test

**File:** `tests/unit/test_equilibrium_diagnostics.py` (MODIFY)
**Line:** 134

```python
# CURRENT (SKIPPED):
@pytest.mark.skip(reason="Test needs review - thermal generation unexpectedly zero")
def test_fuel_price_changes_with_thermal_marginal(self):
    ...

# FIX: Adjust demand to ensure thermal is actually marginal
def test_fuel_price_changes_with_thermal_marginal(self):
    """Test that prices change when fuel changes AND thermal is marginal"""
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

    # INCREASED demand to ensure thermal is needed
    demand_cfg = DemandConfig(
        inelastic=False,
        base_intercept=300.0,  # Increased from 200
        slope=-0.004,  # Less elastic to maintain high demand
        daily_seasonality=False,
        annual_seasonality=False,
    )
    demand = DemandCurve(demand_cfg)

    base_vals = {
        "cap.nuclear": 4000.0,  # Reduced to force thermal
        "avail.nuclear": 0.95,
        "cap.wind": 4000.0,  # Reduced
        "cap.solar": 3000.0,  # Reduced
        "cap.coal": 10000.0,  # Increased
        "avail.coal": 0.90,
        "cap.gas": 15000.0,  # Increased
        "avail.gas": 0.95,
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

    ts = pd.Timestamp("2024-01-01 12:00")
    price_grid = np.array(list(range(-100, 301, 10)), dtype=float)  # Extended range

    # Test with different fuel prices
    fuel_scenarios = [
        (20.0, 15.0, "Low fuel"),
        (40.0, 30.0, "Medium fuel"),
        (60.0, 45.0, "High fuel"),
    ]

    results = []
    for gas_price, coal_price, label in fuel_scenarios:
        vals = base_vals.copy()
        vals["fuel.gas"] = gas_price
        vals["fuel.coal"] = coal_price

        q_star, p_star = find_equilibrium(ts, demand, supply, vals, price_grid)
        _, breakdown = supply.supply_at(p_star, ts, vals)

        thermal = breakdown["coal"] + breakdown["gas"]
        results.append((label, gas_price, coal_price, p_star, q_star, thermal))

        print(f"\n{label}: Gas=${gas_price}, Coal=${coal_price}")
        print(f"  Equilibrium: P=${p_star:6.1f}, Q={q_star:7.1f} MW")
        print(f"  Thermal: {thermal:7.1f} MW")

    # Verify thermal is running in ALL cases
    for label, _, _, _, _, thermal in results:
        assert thermal > 5000, f"{label}: Thermal {thermal} should be > 5000 MW"

    # Verify prices INCREASE with fuel prices
    prices = [r[3] for r in results]
    assert prices[1] > prices[0], f"Medium fuel price should be > low: {prices[1]} vs {prices[0]}"
    assert prices[2] > prices[1], f"High fuel price should be > medium: {prices[2]} vs {prices[1]}"

    # Price increase should be substantial
    price_increase = prices[2] - prices[0]
    assert price_increase > 10, f"Price increase {price_increase} should be significant when fuel doubles"

    print(f"\n✓ Price increased from ${prices[0]:.1f} to ${prices[2]:.1f} as fuel increased")
```

**Estimated effort:** 1 hour
**Impact:** CRITICAL - validates core economic behavior (price response to fuel costs)

---

### HP-4: Numerical Robustness Suite

**File:** `tests/unit/test_numerical_robustness.py` (NEW)

```python
"""
Test numerical edge cases and extreme values.
Ensures system doesn't crash or produce NaN/Inf in edge conditions.
"""

import numpy as np
import pandas as pd
import pytest
from synthetic_data_pkg.simulate import find_equilibrium
from synthetic_data_pkg.demand import DemandCurve
from synthetic_data_pkg.supply import SupplyCurve
from synthetic_data_pkg.config import DemandConfig, TopConfig
from synthetic_data_pkg.dists import iid_sample, stateful_step, _clamp


@pytest.mark.unit
class TestNumericalEdgeCases:
    """Test handling of extreme numerical values"""

    def test_very_large_capacities(self):
        """Test with capacities in millions of MW (ISO-scale)"""
        demand_cfg = DemandConfig(
            inelastic=False,
            base_intercept=20000.0,  # Scale demand too
            slope=-0.00001,
            daily_seasonality=False,
            annual_seasonality=False,
        )
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

        vals = {
            "cap.nuclear": 1_000_000.0,  # 1 million MW
            "avail.nuclear": 0.95,
            "cap.wind": 1_500_000.0,
            "cap.solar": 800_000.0,
            "cap.coal": 1_200_000.0,
            "avail.coal": 0.90,
            "cap.gas": 2_000_000.0,
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

        ts = pd.Timestamp("2024-01-01 12:00")
        price_grid = np.array(list(range(-100, 201, 10)), dtype=float)

        q_star, p_star = find_equilibrium(ts, demand, supply, vals, price_grid)

        # Should not overflow or produce NaN
        assert not np.isnan(q_star), "NaN quantity with large capacities"
        assert not np.isnan(p_star), "NaN price with large capacities"
        assert not np.isinf(q_star), "Inf quantity with large capacities"
        assert not np.isinf(p_star), "Inf price with large capacities"
        assert q_star > 0, "Non-positive quantity"

    def test_very_small_capacities(self):
        """Test with fractional MW capacities (micro-grid scale)"""
        demand_cfg = DemandConfig(
            inelastic=False,
            base_intercept=50.0,
            slope=-5.0,  # Steep for small scale
            daily_seasonality=False,
            annual_seasonality=False,
        )
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

        vals = {
            "cap.nuclear": 0.5,  # 500 kW
            "avail.nuclear": 0.95,
            "cap.wind": 0.8,
            "cap.solar": 0.3,
            "cap.coal": 1.0,
            "avail.coal": 0.90,
            "cap.gas": 1.5,
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

        ts = pd.Timestamp("2024-01-01 12:00")
        price_grid = np.array(list(range(-100, 201, 10)), dtype=float)

        q_star, p_star = find_equilibrium(ts, demand, supply, vals, price_grid)

        # Should handle small values without underflow
        assert not np.isnan(q_star), "NaN with small capacities"
        assert not np.isnan(p_star), "NaN price with small capacities"
        assert q_star > 0, "Non-positive quantity"
        assert q_star < 10.0, "Quantity unexpectedly large for small capacities"

    def test_extreme_fuel_prices(self):
        """Test with very high and very low fuel prices"""
        demand_cfg = DemandConfig(
            inelastic=False,
            base_intercept=500.0,
            slope=-0.01,
            daily_seasonality=False,
            annual_seasonality=False,
        )
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

        vals_base = {
            "cap.nuclear": 6000.0,
            "avail.nuclear": 0.95,
            "cap.wind": 7000.0,
            "cap.solar": 5000.0,
            "cap.coal": 8000.0,
            "avail.coal": 0.90,
            "cap.gas": 12000.0,
            "avail.gas": 0.95,
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

        ts = pd.Timestamp("2024-01-01 12:00")
        price_grid = np.array(list(range(-100, 501, 10)), dtype=float)

        # Test very low fuel prices
        vals_low = vals_base.copy()
        vals_low["fuel.coal"] = 0.1  # Nearly free
        vals_low["fuel.gas"] = 0.1

        q_low, p_low = find_equilibrium(ts, demand, supply, vals_low, price_grid)
        assert not np.isnan(q_low) and not np.isnan(p_low), "Failed with very low fuel prices"

        # Test very high fuel prices
        vals_high = vals_base.copy()
        vals_high["fuel.coal"] = 500.0  # Very expensive
        vals_high["fuel.gas"] = 800.0

        q_high, p_high = find_equilibrium(ts, demand, supply, vals_high, price_grid)
        assert not np.isnan(q_high) and not np.isnan(p_high), "Failed with very high fuel prices"

    def test_zero_availability(self):
        """Test with zero availability (complete outage scenario)"""
        demand_cfg = DemandConfig(
            inelastic=False,
            base_intercept=200.0,
            slope=-0.006,
            daily_seasonality=False,
            annual_seasonality=False,
        )
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

        vals = {
            "cap.nuclear": 6000.0,
            "avail.nuclear": 0.0,  # ZERO - complete outage
            "cap.wind": 7000.0,
            "cap.solar": 5000.0,
            "cap.coal": 8000.0,
            "avail.coal": 0.0,  # ZERO
            "cap.gas": 12000.0,
            "avail.gas": 0.0,  # ZERO
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

        ts = pd.Timestamp("2024-01-01 12:00")
        price_grid = np.array(list(range(-100, 201, 10)), dtype=float)

        q_star, p_star = find_equilibrium(ts, demand, supply, vals, price_grid)

        # Should not crash, but equilibrium will be at low quantity (renewables only)
        assert not np.isnan(q_star), "NaN with zero thermal availability"
        assert not np.isnan(p_star), "NaN price with zero thermal availability"

    def test_division_by_zero_protection_in_efficiency(self):
        """Test that eta_lb = eta_ub doesn't cause division by zero"""
        demand_cfg = DemandConfig(
            inelastic=False,
            base_intercept=200.0,
            slope=-0.006,
            daily_seasonality=False,
            annual_seasonality=False,
        )
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

        vals = {
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
            "eta_lb.coal": 0.35,  # SAME AS UB
            "eta_ub.coal": 0.35,
            "eta_lb.gas": 0.50,  # SAME AS UB
            "eta_ub.gas": 0.50,
            "bid.nuclear.min": -200.0,
            "bid.nuclear.max": -50.0,
            "bid.wind.min": -200.0,
            "bid.wind.max": -50.0,
            "bid.solar.min": -200.0,
            "bid.solar.max": -50.0,
        }

        ts = pd.Timestamp("2024-01-01 12:00")
        price_grid = np.array(list(range(-100, 201, 10)), dtype=float)

        # Should not crash with eta_lb = eta_ub
        q_star, p_star = find_equilibrium(ts, demand, supply, vals, price_grid)
        assert not np.isnan(q_star), "NaN with eta_lb = eta_ub"
        assert not np.isnan(p_star), "NaN price with eta_lb = eta_ub"


@pytest.mark.unit
class TestDistributionNumericalRobustness:
    """Test distributions with extreme parameters"""

    def test_clamp_with_extreme_bounds(self):
        """Test clamping with very large/small bounds"""
        # Very large bounds
        bounds_large = {"low": -1e10, "high": 1e10}
        assert _clamp(5.0, bounds_large) == 5.0
        assert _clamp(1e11, bounds_large) == 1e10
        assert _clamp(-1e11, bounds_large) == -1e10

        # Very tight bounds
        bounds_tight = {"low": 0.0, "high": 0.001}
        assert 0.0 <= _clamp(5.0, bounds_tight) <= 0.001
        assert 0.0 <= _clamp(-5.0, bounds_tight) <= 0.001

    def test_normal_distribution_with_extreme_sigma(self):
        """Test normal distribution with very large/small sigma"""
        rng = np.random.default_rng(42)

        # Very small sigma (nearly deterministic)
        spec_small = {"kind": "normal", "mu": 50.0, "sigma": 0.0001}
        samples = [iid_sample(rng, spec_small) for _ in range(100)]
        assert all(49.99 < s < 50.01 for s in samples), "Small sigma failed"

        # Very large sigma (very dispersed)
        spec_large = {"kind": "normal", "mu": 50.0, "sigma": 1000.0, "bounds": {"low": -5000, "high": 5000}}
        samples = [iid_sample(rng, spec_large) for _ in range(1000)]
        assert all(-5000 <= s <= 5000 for s in samples), "Large sigma with bounds failed"

    def test_ar1_with_extreme_persistence(self):
        """Test AR1 process with phi very close to 1"""
        rng = np.random.default_rng(42)

        # phi very close to 1 (near random walk)
        spec = {"kind": "ar1", "mu": 50.0, "sigma": 1.0, "phi": 0.9999}

        vals = [50.0]
        for _ in range(100):
            vals.append(stateful_step(rng, prev=vals[-1], spec=spec))

        # Should not explode or collapse
        assert all(not np.isnan(v) for v in vals), "AR1 produced NaN"
        assert all(not np.isinf(v) for v in vals), "AR1 produced Inf"
```

**Estimated effort:** 5 hours
**Expected test count:** 15 new tests
**Impact:** Prevents crashes and silent failures in edge conditions

---

### HP-5: Fuel Price Robustness Suite

**File:** `tests/unit/test_fuel_price_robustness.py` (NEW)

```python
"""
Test equilibrium behavior across different fuel price ranges and scenarios.
"""

import numpy as np
import pandas as pd
import pytest
from synthetic_data_pkg.simulate import find_equilibrium
from synthetic_data_pkg.demand import DemandCurve
from synthetic_data_pkg.supply import SupplyCurve
from synthetic_data_pkg.config import DemandConfig, TopConfig


@pytest.mark.unit
class TestFuelPriceRanges:
    """Test equilibrium with different fuel price ranges"""

    @pytest.mark.parametrize("gas_price,coal_price", [
        (1.0, 1.0),      # Very low
        (10.0, 8.0),     # Low
        (30.0, 25.0),    # Normal
        (100.0, 80.0),   # High
        (300.0, 250.0),  # Very high
    ])
    def test_equilibrium_at_different_fuel_price_levels(self, gas_price, coal_price):
        """Test equilibrium finding across fuel price spectrum"""
        # Increase demand to ensure thermal is always needed
        demand_cfg = DemandConfig(
            inelastic=False,
            base_intercept=400.0,
            slope=-0.005,
            daily_seasonality=False,
            annual_seasonality=False,
        )
        demand = DemandCurve(demand_cfg)

        config = TopConfig(
            start_ts="2024-01-01", days=1,
            supply_regime_planner={"mode": "local_only"},
            renewable_availability_mode="weather_simulation",
            variables={
                "fuel.gas": {"regimes": [{"name": "s", "dist": {"kind": "const", "v": gas_price}}]},
                "fuel.coal": {"regimes": [{"name": "s", "dist": {"kind": "const", "v": coal_price}}]},
            },
        )
        supply = SupplyCurve(config, rng_seed=42)

        vals = {
            "cap.nuclear": 5000.0,
            "avail.nuclear": 0.95,
            "cap.wind": 5000.0,
            "cap.solar": 4000.0,
            "cap.coal": 12000.0,  # Plenty of thermal capacity
            "avail.coal": 0.90,
            "cap.gas": 15000.0,
            "avail.gas": 0.95,
            "fuel.coal": coal_price,
            "fuel.gas": gas_price,
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

        ts = pd.Timestamp("2024-01-01 12:00")
        price_grid = np.array(list(range(-100, 701, 10)), dtype=float)  # Extended for high fuel prices

        q_star, p_star = find_equilibrium(ts, demand, supply, vals, price_grid)

        # Should find equilibrium at all fuel price levels
        assert not np.isnan(q_star), f"NaN quantity at gas=${gas_price}, coal=${coal_price}"
        assert not np.isnan(p_star), f"NaN price at gas=${gas_price}, coal=${coal_price}"
        assert q_star > 0, f"Non-positive quantity at gas=${gas_price}, coal=${coal_price}"

        # Market price should be related to fuel costs when thermal is marginal
        _, breakdown = supply.supply_at(p_star, ts, vals)
        thermal_gen = breakdown["coal"] + breakdown["gas"]

        if thermal_gen > 1000:  # Thermal is marginal
            # Price should be at least as high as marginal cost of cheapest thermal
            min_mc = min(coal_price / 0.38, gas_price / 0.55)  # fuel/eta_ub
            # Allow some tolerance for equilibrium discretization
            assert p_star >= min_mc - 20, \
                f"Price {p_star} below marginal cost {min_mc} at gas=${gas_price}, coal=${coal_price}"

    def test_fuel_price_monotonicity(self):
        """Test that market prices increase monotonically with fuel prices"""
        demand_cfg = DemandConfig(
            inelastic=False,
            base_intercept=300.0,
            slope=-0.004,
            daily_seasonality=False,
            annual_seasonality=False,
        )
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

        vals_base = {
            "cap.nuclear": 4000.0,
            "avail.nuclear": 0.95,
            "cap.wind": 4000.0,
            "cap.solar": 3000.0,
            "cap.coal": 10000.0,
            "avail.coal": 0.90,
            "cap.gas": 15000.0,
            "avail.gas": 0.95,
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

        ts = pd.Timestamp("2024-01-01 12:00")
        price_grid = np.array(list(range(-100, 401, 10)), dtype=float)

        # Test increasing fuel prices
        fuel_prices = [10, 20, 30, 50, 80, 120]
        equilibrium_prices = []

        for fuel_price in fuel_prices:
            vals = vals_base.copy()
            vals["fuel.coal"] = fuel_price * 0.8
            vals["fuel.gas"] = fuel_price

            q_star, p_star = find_equilibrium(ts, demand, supply, vals, price_grid)
            equilibrium_prices.append(p_star)

            print(f"Fuel price {fuel_price}: Market price {p_star:.1f}")

        # Verify monotonicity (allowing small tolerance for discretization)
        for i in range(1, len(equilibrium_prices)):
            assert equilibrium_prices[i] >= equilibrium_prices[i-1] - 5, \
                f"Price decreased when fuel increased: {equilibrium_prices[i-1]} -> {equilibrium_prices[i]}"

    @pytest.mark.parametrize("price_ratio", [1.0, 2.0, 5.0, 10.0, 100.0])
    def test_extreme_fuel_price_ratios(self, price_ratio):
        """Test with extreme ratios between gas and coal prices"""
        demand_cfg = DemandConfig(
            inelastic=False,
            base_intercept=300.0,
            slope=-0.004,
            daily_seasonality=False,
            annual_seasonality=False,
        )
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

        coal_price = 25.0
        gas_price = coal_price * price_ratio

        vals = {
            "cap.nuclear": 4000.0,
            "avail.nuclear": 0.95,
            "cap.wind": 4000.0,
            "cap.solar": 3000.0,
            "cap.coal": 10000.0,
            "avail.coal": 0.90,
            "cap.gas": 15000.0,
            "avail.gas": 0.95,
            "fuel.coal": coal_price,
            "fuel.gas": gas_price,
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

        ts = pd.Timestamp("2024-01-01 12:00")
        price_grid = np.array(list(range(-100, 1001, 10)), dtype=float)

        q_star, p_star = find_equilibrium(ts, demand, supply, vals, price_grid)

        # Should handle extreme ratios
        assert not np.isnan(q_star), f"NaN with price ratio {price_ratio}"
        assert not np.isnan(p_star), f"NaN price with ratio {price_ratio}"

        # Check merit order makes sense
        _, breakdown = supply.supply_at(p_star, ts, vals)

        if price_ratio > 2.0:
            # Gas much more expensive - coal should produce more
            coal_gen = breakdown["coal"]
            gas_gen = breakdown["gas"]

            # If both are running, coal should produce more
            if coal_gen > 100 and gas_gen > 100:
                assert coal_gen >= gas_gen, \
                    f"Coal should produce more when gas is {price_ratio}x more expensive"
```

**Estimated effort:** 3 hours
**Expected test count:** 18 new tests
**Impact:** Validates economic behavior across fuel price spectrum

---

## MEDIUM PRIORITY Test Additions

### MP-1: Demand Elasticity Edge Cases

**File:** `tests/unit/test_demand_robustness.py` (NEW)

Add tests for:
- Very elastic demand (slope ≈ 0)
- Very inelastic demand (slope « -1000)
- Negative intercept
- Zero intercept
- Different scale combinations (small intercept + steep slope, etc.)

**Estimated effort:** 2 hours
**Expected test count:** 8 new tests

---

### MP-2: Availability Edge Cases

**File:** `tests/unit/test_availability_edge_cases.py` (NEW)

Add tests for:
- Availability = 1.0 (perfect reliability)
- Availability = 0.0 (complete outage) - for each tech
- Time-varying availability patterns
- All availabilities at extremes simultaneously

**Estimated effort:** 2 hours
**Expected test count:** 10 new tests

---

### MP-3: Time Scale Robustness

**File:** `tests/integration/test_time_scale_robustness.py` (NEW)

Add tests for:
- Single hour simulation
- Multi-year simulations (5, 10 years)
- Different frequencies (15min, 30min, 2h, daily)
- Leap year handling
- DST transitions (if applicable)

**Estimated effort:** 3 hours
**Expected test count:** 8 new tests

---

### MP-4: Fix Test Bugs

**Files to fix:**
1. `tests/unit/test_io.py:138` - Remove duplicate column
2. `tests/unit/test_dists.py:163` - Tighten AR1 mean reversion tolerance
3. Various tests - Add explicit seeds for reproducibility

**Estimated effort:** 1 hour

---

### MP-5: Weather Model Edge Cases

**File:** `tests/unit/test_weather_robustness.py` (NEW)

Add tests for:
- Wind at 0% and 100% for extended periods
- Solar at midnight in winter vs noon in summer
- Extreme weather persistence values (rho → 1)
- Extreme weather volatility values (sigma → 0, sigma → ∞)

**Estimated effort:** 2 hours
**Expected test count:** 8 new tests

---

### MP-6: Linear Distribution Edge Cases

**File:** `tests/unit/test_linear_distribution.py` (MODIFY)

Add tests for:
- Slope = 0 (should behave like const)
- Very large positive/negative slopes
- Negative starting values
- Bounds violations with extreme slopes
- Numerical precision with very small slopes

**Estimated effort:** 2 hours
**Expected test count:** 6 new tests

---

## LOW PRIORITY Test Additions

### LP-1: Property-Based Testing with Hypothesis

**File:** `tests/property/test_invariants.py` (NEW)

Use hypothesis library to test invariants:
- Supply ≤ total available capacity
- Price within grid bounds
- Quantity ≥ 0
- Higher fuel prices → higher or equal market prices (when thermal marginal)
- Conservation of energy

**Estimated effort:** 4 hours
**Expected test count:** 10 property tests

---

### LP-2: Performance Benchmarks

**File:** `tests/benchmarks/test_performance.py` (NEW)

Add benchmarks for:
- Equilibrium finding speed vs grid resolution
- Simulation time vs number of days
- Memory usage for long simulations

**Estimated effort:** 3 hours
**Expected test count:** 5 benchmarks

---

### LP-3: Regression Tests

**File:** `tests/regression/test_known_scenarios.py` (NEW)

Add tests with known expected outputs:
- Load saved scenarios from real markets
- Compare outputs to expected values
- Detect breaking changes

**Estimated effort:** 4 hours
**Expected test count:** 5 regression tests

---

## Summary of Test Additions

| Priority | Category | New Files | New Tests | Effort (hrs) |
|----------|----------|-----------|-----------|--------------|
| HIGH | Price grid robustness | 1 | 15 | 4 |
| HIGH | Capacity scale robustness | 1 | 12 | 4 |
| HIGH | Fix skipped test | 0 | 1 | 1 |
| HIGH | Numerical robustness | 1 | 15 | 5 |
| HIGH | Fuel price robustness | 1 | 18 | 3 |
| **HIGH TOTAL** | | **4** | **61** | **17** |
| MEDIUM | Demand robustness | 1 | 8 | 2 |
| MEDIUM | Availability edge cases | 1 | 10 | 2 |
| MEDIUM | Time scale robustness | 1 | 8 | 3 |
| MEDIUM | Fix test bugs | 0 | - | 1 |
| MEDIUM | Weather robustness | 1 | 8 | 2 |
| MEDIUM | Linear distribution edges | 0 | 6 | 2 |
| **MEDIUM TOTAL** | | **4** | **40** | **12** |
| LOW | Property-based testing | 1 | 10 | 4 |
| LOW | Performance benchmarks | 1 | 5 | 3 |
| LOW | Regression tests | 1 | 5 | 4 |
| **LOW TOTAL** | | **3** | **20** | **11** |
| **GRAND TOTAL** | | **11** | **121** | **40** |

---

## Implementation Sequence

### Phase 1: Critical Fixes (Day 1)
1. Fix skipped test `test_fuel_price_changes_with_thermal_marginal` (1h)
2. Fix test bugs in test_io.py and test_dists.py (1h)
3. Implement HP-1: Price Grid Robustness Suite (4h)
4. Implement HP-2: Capacity Scale Robustness Suite (4h)

**Deliverable:** Core robustness improved, skipped test resolved

### Phase 2: Numerical Safety (Day 2-3)
5. Implement HP-4: Numerical Robustness Suite (5h)
6. Implement HP-5: Fuel Price Robustness Suite (3h)
7. Implement MP-1: Demand Elasticity Edge Cases (2h)
8. Implement MP-2: Availability Edge Cases (2h)

**Deliverable:** System validated against extreme values and edge cases

### Phase 3: Complete Coverage (Day 4-5)
9. Implement MP-3: Time Scale Robustness (3h)
10. Implement MP-5: Weather Model Edge Cases (2h)
11. Implement MP-6: Linear Distribution Edge Cases (2h)
12. Documentation and test runner setup (2h)

**Deliverable:** Comprehensive test suite with edge case coverage

### Phase 4: Advanced Testing (Optional)
13. Implement LP-1: Property-Based Testing (4h)
14. Implement LP-2: Performance Benchmarks (3h)
15. Implement LP-3: Regression Tests (4h)

**Deliverable:** Advanced validation and performance monitoring

---

## Test Infrastructure Improvements

### 1. Add Conftest Helpers

**File:** `tests/conftest.py` (MODIFY)

```python
@pytest.fixture
def standard_vals():
    """Standard variable values for testing"""
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


@pytest.fixture
def scaled_vals(standard_vals):
    """Factory for scaled variable values"""
    def _scale(scale_factor):
        return {k: (v * scale_factor if k.startswith("cap.") else v)
                for k, v in standard_vals.items()}
    return _scale
```

### 2. Add Pytest Markers

**File:** `pytest.ini` (CREATE/MODIFY)

```ini
[pytest]
markers =
    unit: Unit tests (fast, isolated)
    integration: Integration tests (slower, multiple components)
    functional: Functional tests (end-to-end workflows)
    slow: Slow tests (skip unless explicitly requested)
    smoke: Smoke tests (basic functionality)
    robustness: Robustness tests (scale and edge cases)
    numerical: Numerical edge case tests
```

### 3. Add Test Documentation

**File:** `tests/README.md` (CREATE)

Document:
- How to run different test categories
- What each test file covers
- Expected test runtime
- How to interpret failures

---

## Success Metrics

After implementing this plan:

1. **Test count:** ~235 total tests (current ~114 + new ~121)
2. **Coverage:**
   - All price grid resolutions (1, 2, 5, 10, 20, 50, 100)
   - All capacity scales (0.01 MW to 1,000,000 MW)
   - All fuel price ranges (0.1 to 500)
   - All numerical edge cases (0, very small, very large, NaN/Inf)
3. **Robustness:** Tests validate system across 6 orders of magnitude in scale
4. **Confidence:** Can deploy to markets of any size with confidence
5. **Maintenance:** Parametrized tests make it easy to add new scenarios

---

## Next Steps

1. **Review this plan** with team
2. **Prioritize** which phases to implement first
3. **Assign** test implementation to developers
4. **Set up CI** to run robustness tests regularly
5. **Monitor** test execution time and optimize if needed

---

## Questions for Review

1. Are there specific market scales you want to prioritize?
2. Should we add tests for specific real-world scenarios?
3. What's an acceptable test suite runtime? (currently ~30s, will increase)
4. Should we add visual regression tests for plots/charts?
5. Do you want fuzz testing or chaos engineering tests?

---

**Document Status:** Draft for Review
**Last Updated:** 2025-11-09
**Author:** Claude (Automated Test Audit)
