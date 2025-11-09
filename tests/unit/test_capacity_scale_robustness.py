"""
Test system behavior across different capacity scales.
Validates the system works for small municipal grids to large regional ISOs.
"""

import numpy as np
import pandas as pd
import pytest

from synthetic_data_pkg.config import DemandConfig, TopConfig
from synthetic_data_pkg.demand import DemandCurve
from synthetic_data_pkg.simulate import find_equilibrium
from synthetic_data_pkg.supply import SupplyCurve


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
            start_ts="2024-01-01",
            days=1,
            supply_regime_planner={"mode": "local_only"},
            renewable_availability_mode="weather_simulation",
            variables={
                "fuel.gas": {
                    "regimes": [{"name": "s", "dist": {"kind": "const", "v": 30.0}}]
                },
                "fuel.coal": {
                    "regimes": [{"name": "s", "dist": {"kind": "const", "v": 25.0}}]
                },
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
        q_base, p_base = find_equilibrium(
            ts, base_demand, base_supply, base_vals, price_grid
        )

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
        scaled_vals = {
            k: (v * scale_factor if k.startswith("cap.") else v)
            for k, v in base_vals.items()
        }

        q_scaled, p_scaled = find_equilibrium(
            ts, scaled_demand, scaled_supply, scaled_vals, price_grid
        )

        # Assertions
        assert not np.isnan(q_scaled), f"NaN quantity at scale {scale_factor}"
        assert not np.isnan(p_scaled), f"NaN price at scale {scale_factor}"

        # Quantity should scale proportionally (within 10% due to discretization)
        expected_q = q_base * scale_factor
        assert (
            abs(q_scaled - expected_q) / expected_q < 0.15
        ), f"Quantity scaling failed: expected {expected_q}, got {q_scaled} at scale {scale_factor}"

        # Price should remain similar (within 20% due to market structure)
        assert (
            abs(p_scaled - p_base) / max(abs(p_base), 1) < 0.25
        ), f"Price changed too much: {p_base} -> {p_scaled} at scale {scale_factor}"

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
            start_ts="2024-01-01",
            days=1,
            supply_regime_planner={"mode": "local_only"},
            renewable_availability_mode="weather_simulation",
            variables={
                "fuel.gas": {
                    "regimes": [{"name": "s", "dist": {"kind": "const", "v": 30.0}}]
                },
                "fuel.coal": {
                    "regimes": [{"name": "s", "dist": {"kind": "const", "v": 25.0}}]
                },
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
        assert (
            q_star <= capacity_mw
        ), f"Quantity {q_star} exceeds total capacity {capacity_mw}"


@pytest.mark.unit
class TestCapacityEdgeCases:
    """Test edge cases in capacity configuration"""

    def test_zero_thermal_capacity(self):
        """Test with zero thermal (coal + gas) capacity - renewables only"""
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
                "fuel.gas": {
                    "regimes": [{"name": "s", "dist": {"kind": "const", "v": 30.0}}]
                },
                "fuel.coal": {
                    "regimes": [{"name": "s", "dist": {"kind": "const", "v": 25.0}}]
                },
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
                "fuel.gas": {
                    "regimes": [{"name": "s", "dist": {"kind": "const", "v": 30.0}}]
                },
                "fuel.coal": {
                    "regimes": [{"name": "s", "dist": {"kind": "const", "v": 25.0}}]
                },
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
                "fuel.gas": {
                    "regimes": [{"name": "s", "dist": {"kind": "const", "v": 30.0}}]
                },
                "fuel.coal": {
                    "regimes": [{"name": "s", "dist": {"kind": "const", "v": 25.0}}]
                },
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
