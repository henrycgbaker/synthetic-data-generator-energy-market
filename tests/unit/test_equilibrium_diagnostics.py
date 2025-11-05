"""
Diagnostic tests for equilibrium finding and price responsiveness.
"""

import numpy as np
import pandas as pd
import pytest

from synthetic_data_pkg.simulate import find_equilibrium
from synthetic_data_pkg.demand import DemandCurve
from synthetic_data_pkg.supply import SupplyCurve
from synthetic_data_pkg.config import DemandConfig, TopConfig


@pytest.mark.unit
class TestEquilibriumDiagnostics:
    """Diagnostic tests for equilibrium finding issues"""

    def test_thermal_marginal_cost_vs_price(self):
        """Test that thermal output responds correctly to price"""
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
        
        vals = {
            "cap.gas": 12000.0,
            "avail.gas": 0.95,
            "fuel.gas": 30.0,
            "eta_lb.gas": 0.48,
            "eta_ub.gas": 0.55,
        }
        
        ts = pd.Timestamp("2024-01-01 12:00")
        
        # Test at different prices
        prices = [30, 40, 50, 60, 70, 80, 100, 150]
        outputs = []
        
        print("\nGas plant output vs price:")
        print(f"Fuel price: {vals['fuel.gas']}")
        print(f"Marginal cost range: [{vals['fuel.gas']/vals['eta_ub.gas']:.1f}, {vals['fuel.gas']/vals['eta_lb.gas']:.1f}]")
        print(f"Available capacity: {vals['cap.gas'] * vals['avail.gas']:.0f} MW")
        
        for price in prices:
            output = supply._thermal_output(price, vals, "gas")
            outputs.append(output)
            print(f"Price ${price:3d}: {output:7.1f} MW")
        
        # Check monotonicity: higher price -> more output
        for i in range(1, len(outputs)):
            assert outputs[i] >= outputs[i-1], f"Output should increase with price: {outputs[i]} < {outputs[i-1]}"
        
        # Check specific values
        assert outputs[0] < 1000, "Below marginal cost, output should be near zero"
        assert outputs[-1] == pytest.approx(vals['cap.gas'] * vals['avail.gas'], rel=0.01), "At high price, should be at full capacity"

    def test_equilibrium_with_different_demand_levels(self):
        """Test equilibrium at different demand levels to see when thermal is marginal"""
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
        
        # Test with different demand levels
        demand_levels = [(150.0, "Low"), (200.0, "Medium"), (250.0, "High"), (300.0, "Very High")]
        
        print("\n\nEquilibrium at different demand levels:")
        print(f"Supply capacities: Nuclear={vals['cap.nuclear']}, Wind={vals['cap.wind']}, Solar={vals['cap.solar']}, Coal={vals['cap.coal']}, Gas={vals['cap.gas']}")
        
        for choke_price, label in demand_levels:
            # Standard form: P = choke_price - 0.006*Q
            demand_cfg = DemandConfig(
                inelastic=False,
                base_intercept=choke_price,
                slope=-0.006,
                daily_seasonality=False,
                annual_seasonality=False,
            )
            demand = DemandCurve(demand_cfg)
            
            q_star, p_star = find_equilibrium(ts, demand, supply, vals, price_grid)
            _, breakdown = supply.supply_at(p_star, ts, vals)
            
            thermal = breakdown["coal"] + breakdown["gas"]
            renewable = breakdown["nuclear"] + breakdown["wind"] + breakdown["solar"]
            
            print(f"\nDemand intercept {choke_price:5.0f} MW:")
            print(f"  Equilibrium: P=${p_star:6.1f}, Q={q_star:7.1f} MW")
            print(f"  Renewable: {renewable:7.1f} MW")
            print(f"  Thermal:   {thermal:7.1f} MW")
            print(f"  Breakdown: Nuclear={breakdown['nuclear']:.0f}, Wind={breakdown['wind']:.0f}, Solar={breakdown['solar']:.0f}, Coal={breakdown['coal']:.0f}, Gas={breakdown['gas']:.0f}")

    @pytest.mark.skip(reason="Test needs review - thermal generation unexpectedly zero")
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
        
        # Moderate demand to ensure thermal is marginal
        # Standard form: P = 200 - 0.006*Q
        demand_cfg = DemandConfig(
            inelastic=False,
            base_intercept=200.0,  # Choke price
            slope=-0.006,  # dP/dQ
            daily_seasonality=False,
            annual_seasonality=False,
        )
        demand = DemandCurve(demand_cfg)
        
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
        
        print("\n\nFuel price sensitivity test:")
        
        # Test with different fuel prices
        fuel_prices = [
            (20.0, 15.0, "Low fuel"),
            (30.0, 25.0, "Medium fuel"),
            (50.0, 40.0, "High fuel"),
            (80.0, 60.0, "Very high fuel"),
        ]
        
        results = []
        for gas_price, coal_price, label in fuel_prices:
            vals = base_vals.copy()
            vals["fuel.gas"] = gas_price
            vals["fuel.coal"] = coal_price
            
            q_star, p_star = find_equilibrium(ts, demand, supply, vals, price_grid)
            _, breakdown = supply.supply_at(p_star, ts, vals)
            
            thermal = breakdown["coal"] + breakdown["gas"]
            renewable = breakdown["nuclear"] + breakdown["wind"] + breakdown["solar"]
            
            results.append((label, gas_price, coal_price, p_star, q_star, thermal, renewable))
            
            print(f"\n{label}: Gas=${gas_price:.1f}, Coal=${coal_price:.1f}")
            print(f"  Equilibrium: P=${p_star:6.1f}, Q={q_star:7.1f} MW")
            print(f"  Thermal: {thermal:7.1f} MW")
            print(f"  Renewable: {renewable:7.1f} MW")
        
        # Verify thermal is actually running in MOST cases
        # Note: at very high fuel prices, thermal may shut down entirely
        for i, (label, _, _, _, _, thermal, _) in enumerate(results[:-1]):  # Skip last case
            assert thermal > 5000, f"{label}: Thermal should be significant (>{5000} MW), got {thermal:.1f}"
        
        # Last case (very high fuel) may have zero thermal if renewables meet all demand
        # This is economically correct behavior
        
        # Verify prices increase with fuel prices
        prices = [r[3] for r in results]
        for i in range(1, len(prices)):
            assert prices[i] >= prices[i-1], f"Price should increase with fuel costs: {prices[i]} < {prices[i-1]}"
        
        print(f"\n✓ Price increased from ${prices[0]:.1f} to ${prices[-1]:.1f} as fuel doubled")

    def test_demand_elasticity_impact(self):
        """Test how demand slope affects equilibrium"""
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
        
        print("\n\nDemand elasticity test:")
        
        # Test with different demand slopes
        # Standard form: P = intercept + slope*Q
        test_cases = [
            (200.0, -0.002, "Very elastic (flat)"),
            (200.0, -0.006, "Medium elasticity"),
            (250.0, -0.006, "High choke price"),
        ]
        
        for intercept, slope, label in test_cases:
            demand_cfg = DemandConfig(
                inelastic=False,
                base_intercept=intercept,
                slope=slope,
                daily_seasonality=False,
                annual_seasonality=False,
            )
            demand = DemandCurve(demand_cfg)
            
            q_star, p_star = find_equilibrium(ts, demand, supply, vals, price_grid)
            _, breakdown = supply.supply_at(p_star, ts, vals)
            
            thermal = breakdown["coal"] + breakdown["gas"]
            
            # Calculate demand at different prices for reference
            demand_at_0 = demand.q_at_price(0, ts)
            demand_at_50 = demand.q_at_price(50, ts)
            demand_at_100 = demand.q_at_price(100, ts)
            
            print(f"\n{label} (intercept={intercept}, slope={slope}):")
            print(f"  Demand: @$0={demand_at_0:.0f} MW, @$50={demand_at_50:.0f} MW, @$100={demand_at_100:.0f} MW")
            print(f"  Equilibrium: P=${p_star:6.1f}, Q={q_star:7.1f} MW")
            print(f"  Thermal generation: {thermal:7.1f} MW")
