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