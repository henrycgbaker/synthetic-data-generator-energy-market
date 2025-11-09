"""
Unit tests for simulate module.
Tests equilibrium finding and simulation logic.
"""

import numpy as np
import pandas as pd
import pytest

from synthetic_data_pkg.config import DemandConfig, TopConfig
from synthetic_data_pkg.demand import DemandCurve
from synthetic_data_pkg.simulate import find_equilibrium
from synthetic_data_pkg.supply import SupplyCurve


@pytest.mark.unit
class TestEquilibriumFinding:
    """Unit tests for equilibrium finding"""

    def test_equilibrium_with_elastic_demand(self):
        """Test equilibrium finding with elastic demand"""
        # Create demand curve: P = 200 - 0.005*Q
        demand_cfg = DemandConfig(
            inelastic=False,
            base_intercept=200.0,  # Choke price
            slope=-0.005,  # dP/dQ
            daily_seasonality=False,
            annual_seasonality=False,
        )
        demand = DemandCurve(demand_cfg)

        # Create supply curve
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

        # Setup vals
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

        # Check equilibrium is valid
        assert not np.isnan(q_star), "Equilibrium quantity is NaN"
        assert not np.isnan(p_star), "Equilibrium price is NaN"
        assert q_star > 0, "Equilibrium quantity should be positive"
        # Price should be within grid bounds (but not necessarily on grid - continuous equilibrium)
        assert (
            price_grid[0] <= p_star <= price_grid[-1]
        ), f"Equilibrium price {p_star} outside grid range"

    def test_equilibrium_with_inelastic_demand(self):
        """Test equilibrium finding with inelastic demand"""
        # Create inelastic demand curve
        demand_cfg = DemandConfig(
            inelastic=True,
            base_intercept=15000.0,  # Fixed demand
            daily_seasonality=False,
            annual_seasonality=False,
        )
        demand = DemandCurve(demand_cfg)

        # Create supply curve
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

        # With inelastic demand, quantity should equal fixed demand
        expected_q = 15000.0
        assert q_star == pytest.approx(expected_q, rel=0.01)
        assert p_star > 0  # Price should be positive to meet this demand

    def test_equilibrium_responds_to_fuel_price_changes(self):
        """Test that equilibrium price changes when fuel prices change"""
        # Standard form: P = 200 - 0.01*Q
        # At Q=10,000: P=100
        demand_cfg = DemandConfig(
            inelastic=False,
            base_intercept=200.0,  # Choke price
            slope=-0.01,  # dP/dQ
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

        base_vals = {
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
        price_grid = np.array(list(range(-100, 201, 10)), dtype=float)

        # Low fuel prices
        vals_low = base_vals.copy()
        vals_low["fuel.coal"] = 20.0
        vals_low["fuel.gas"] = 25.0
        q1, p1 = find_equilibrium(ts, demand, supply, vals_low, price_grid)

        # High fuel prices
        vals_high = base_vals.copy()
        vals_high["fuel.coal"] = 40.0
        vals_high["fuel.gas"] = 50.0
        q2, p2 = find_equilibrium(ts, demand, supply, vals_high, price_grid)

        # Higher fuel prices should lead to higher market prices
        # (unless all demand is met by must-run renewables)
        print(f"Low fuel price equilibrium: p={p1}, q={q1}")
        print(f"High fuel price equilibrium: p={p2}, q={q2}")

        # At minimum, prices should not be identical
        # If they are, there's a bug in how fuel prices affect equilibrium
        if p1 == p2:
            # Check if thermal is actually running
            _, br1 = supply.supply_at(p1, ts, vals_low)
            _, br2 = supply.supply_at(p2, ts, vals_high)
            print(f"Breakdown at low prices: {br1}")
            print(f"Breakdown at high prices: {br2}")

            # If thermal is running, prices should differ
            if br1["coal"] > 0 or br1["gas"] > 0:
                pytest.fail(
                    f"Prices should change when fuel prices change and thermal is marginal, but got p1={p1}, p2={p2}"
                )

    def test_equilibrium_at_price_floor(self):
        """Test equilibrium when demand is very low"""
        # Standard form: P = 50 - 0.002*Q
        # At Q=25,000: P=0 (hits floor before this)
        demand_cfg = DemandConfig(
            inelastic=False,
            base_intercept=50.0,  # Low choke price
            slope=-0.002,  # dP/dQ
            daily_seasonality=False,
            annual_seasonality=False,
        )
        demand = DemandCurve(demand_cfg)

        config = TopConfig(
            start_ts="2024-01-01",
            days=1,
            renewable_availability_mode="weather_simulation",
            supply_regime_planner={"mode": "local_only"},
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
            "eta_lb.coal": 0.33,
            "eta_ub.coal": 0.38,
            "eta_lb.gas": 0.48,
            "eta_ub.gas": 0.55,
            "bid.nuclear.min": -300.0,  # Well below price floor
            "bid.nuclear.max": -250.0,  # So nuclear is fully available at floor
            "bid.wind.min": -300.0,
            "bid.wind.max": -250.0,
            "bid.solar.min": -300.0,
            "bid.solar.max": -250.0,
        }

        ts = pd.Timestamp("2024-01-01 12:00")
        price_grid = np.array(list(range(-100, 201, 10)), dtype=float)

        q_star, p_star = find_equilibrium(ts, demand, supply, vals, price_grid)

        # With low choke price (50) and limited cheap supply (renewables + nuclear),
        # equilibrium settles where demand intersects the flat supply segment
        # After fuel price bug fix, thermal plants correctly bid at marginal cost (~55-76)
        # so they don't produce at low prices
        print(f"\nLow demand equilibrium: P={p_star:.2f}, Q={q_star:.0f}")

        # Price should be below the choke price (low demand scenario)
        # With ~10,600 MW of cheap renewables/nuclear, equilibrium is around P=28-30
        assert (
            p_star < demand_cfg.base_intercept * 0.7
        ), f"Price {p_star} should be below choke price {demand_cfg.base_intercept}"

        # Price must be within grid bounds (can be negative in high renewable scenarios!)
        assert (
            price_grid[0] <= p_star <= price_grid[-1]
        ), f"Price {p_star} outside grid bounds"

    def test_equilibrium_at_price_ceiling(self):
        """Test equilibrium when demand exceeds supply"""
        # Standard form: P = 500 - 0.01*Q
        # Very high choke price ensures demand exceeds low supply capacity
        demand_cfg = DemandConfig(
            inelastic=False,
            base_intercept=500.0,  # High choke price
            slope=-0.01,  # dP/dQ
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
            "cap.nuclear": 1000.0,  # Low capacity
            "avail.nuclear": 0.95,
            "cap.wind": 1000.0,
            "cap.solar": 1000.0,
            "cap.coal": 1000.0,
            "avail.coal": 0.90,
            "cap.gas": 1000.0,
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

        # With demand exceeding supply, price should be at ceiling
        assert p_star == price_grid[-1]
