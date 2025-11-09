"""
Test numerical edge cases and extreme values.
Ensures system doesn't crash or produce NaN/Inf in edge conditions.
"""

import numpy as np
import pandas as pd
import pytest

from synthetic_data_pkg.config import DemandConfig, TopConfig
from synthetic_data_pkg.demand import DemandCurve
from synthetic_data_pkg.dists import _clamp, iid_sample, stateful_step
from synthetic_data_pkg.simulate import find_equilibrium
from synthetic_data_pkg.supply import SupplyCurve


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
        assert not np.isnan(q_low) and not np.isnan(
            p_low
        ), "Failed with very low fuel prices"

        # Test very high fuel prices
        vals_high = vals_base.copy()
        vals_high["fuel.coal"] = 500.0  # Very expensive
        vals_high["fuel.gas"] = 800.0

        q_high, p_high = find_equilibrium(ts, demand, supply, vals_high, price_grid)
        assert not np.isnan(q_high) and not np.isnan(
            p_high
        ), "Failed with very high fuel prices"

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
        spec_large = {
            "kind": "normal",
            "mu": 50.0,
            "sigma": 1000.0,
            "bounds": {"low": -5000, "high": 5000},
        }
        samples = [iid_sample(rng, spec_large) for _ in range(1000)]
        assert all(
            -5000 <= s <= 5000 for s in samples
        ), "Large sigma with bounds failed"

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
