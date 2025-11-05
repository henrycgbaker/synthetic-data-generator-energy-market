"""
Runs the simulation of demand, supply, and equilibrium over time -> synthetic market time series.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from tqdm import tqdm

from .config import DemandConfig
from .demand import DemandCurve
from .supply import SupplyCurve


def find_equilibrium(
    ts: pd.Timestamp,
    demand: DemandCurve,
    supply: SupplyCurve,
    vals: Dict[str, float],
    price_grid: np.ndarray,
) -> Tuple[float, float]:
    """
    Find market equilibrium, with clipping for edge cases:
    - If demand too low: use minimum price
    - If demand too high (> total supply): use maximum price
    - For inelastic demand: find price where supply meets fixed demand
    """
    # Calculate upper bound: maximum possible supply at highest price
    # Use the supply curve itself to get this value
    p_max = float(price_grid[-1])
    q_upper, _ = supply.supply_at(p_max, ts, vals)

    # Handle inelastic demand separately
    if demand.cfg.inelastic:
        # Fixed demand quantity
        q_demand = demand.q_at_price(0.0, ts)  # Price doesn't matter for inelastic

        # Find price where supply equals this fixed demand
        # If demand exceeds total supply, clip at max price
        if q_demand > q_upper:
            q_supply_at_max, _ = supply.supply_at(p_max, ts, vals)
            return float(q_supply_at_max), p_max

        # Find the price where supply = demand
        try:

            def f_inelastic(p):
                q_supply, _ = supply.supply_at(p, ts, vals)
                return q_supply - q_demand

            # Search for equilibrium price
            p_min = float(price_grid[0])
            p_star = brentq(f_inelastic, p_min, p_max, maxiter=300)
            return float(q_demand), float(p_star)
        except ValueError:
            # If no equilibrium found, clip at boundaries
            q_supply_at_min, _ = supply.supply_at(price_grid[0], ts, vals)
            if q_demand <= q_supply_at_min:
                # Demand can be met at minimum price
                return float(q_demand), float(price_grid[0])
            else:
                # Demand exceeds supply even at max price
                q_supply_at_max, _ = supply.supply_at(price_grid[-1], ts, vals)
                return float(q_supply_at_max), float(price_grid[-1])

    # Elastic demand: standard equilibrium finding
    # First check if we're at boundary conditions
    p_min = float(price_grid[0])
    p_max = float(price_grid[-1])
    
    q_demand_at_min = demand.q_at_price(p_min, ts)
    q_supply_at_min, _ = supply.supply_at(p_min, ts, vals)
    
    # If supply exceeds demand even at minimum price, clip at floor
    # Add small tolerance for floating point comparison
    if q_supply_at_min >= q_demand_at_min * 0.999:
        return float(q_demand_at_min), p_min
    
    q_demand_at_max = demand.q_at_price(p_max, ts)
    q_supply_at_max, _ = supply.supply_at(p_max, ts, vals)
    
    # If demand exceeds supply even at maximum price, clip at ceiling
    if q_demand_at_max >= q_supply_at_max * 1.001:  # Small tolerance
        return float(q_supply_at_max), p_max

    def f(q):
        ps = supply.supply_price_at_quantity(q, ts, vals, price_grid)
        pdq = demand.p_at_quantity(q, ts)
        return ps - pdq

    try:
        # Find equilibrium quantity where supply price = demand price
        q_min = max(0.0, min(q_supply_at_min, q_demand_at_min) * 0.9)
        q_max = min(q_supply_at_max, q_demand_at_max) * 1.1
        
        # Ensure valid bounds
        if q_max <= q_min:
            # Edge case: return midpoint
            q_star = (q_supply_at_max + q_demand_at_max) / 2
            p_star = demand.p_at_quantity(q_star, ts)
            return float(q_star), float(p_star)
        
        q_star = brentq(f, q_min, q_max, maxiter=300)
        p_star = demand.p_at_quantity(q_star, ts)
        return float(q_star), float(p_star)
    except (ValueError, RuntimeError) as e:
        # Solver failed - return boundary condition
        # If demand is high relative to supply, hit ceiling
        if q_demand_at_max > q_supply_at_max * 0.8:
            return float(q_supply_at_max), p_max
        # Otherwise hit floor  
        return float(q_demand_at_min), p_min


def simulate_timeseries(
    *,
    start_ts: str,
    hours: int,
    demand_cfg: Any,
    schedules: Dict[str, Any],
    price_grid: np.ndarray,
    seed: int,
    config: Any,
    planned_outages_cfg: Any = None,
) -> pd.DataFrame:
    """
    Simulates full market over given horizon
    Draws values from all schedules, computing equilibrium,
    returns a df of results with both outputs and underlying drivers.
    """
    demand = DemandCurve(DemandConfig(**demand_cfg))
    supply = SupplyCurve(config=config, rng_seed=seed)

    rows = []
    for h in tqdm(range(hours), desc="Simulating timesteps", unit="hr"):
        ts = pd.Timestamp(start_ts) + pd.Timedelta(hours=h)
        vals: Dict[str, float] = {}
        labs: Dict[str, str] = {}
        for name, sched in schedules.items():
            v, lab = sched.value_at(ts)
            vals[name] = float(v)
            labs[f"{name}_regime"] = lab

        # Apply planned outages to availability
        if planned_outages_cfg and planned_outages_cfg.get("enabled", True):
            month = ts.month
            outage_months = planned_outages_cfg.get("months", [5, 6, 7, 8, 9])
            if month in outage_months:
                for tech in ["nuclear", "coal", "gas"]:
                    avail_key = f"avail.{tech}"
                    reduction_key = f"{tech}_reduction"
                    if avail_key in vals:
                        reduction = planned_outages_cfg.get(reduction_key, 0.0)
                        vals[avail_key] = max(0.0, vals[avail_key] * (1.0 - reduction))

        # In weather_simulation mode, add wind/solar availability to vals
        # for consistency in output (even though calculated internally)
        if config.renewable_availability_mode == "weather_simulation":
            vals["avail.wind"] = supply._get_wind_availability(ts, vals)
            vals["avail.solar"] = supply._get_solar_availability(ts, vals)

        # sanity: require fuels present each hour
        for req in ("fuel.coal", "fuel.gas"):
            if req not in vals:
                raise RuntimeError(f"Missing {req} at {ts}")

        q_star, p_star = find_equilibrium(ts, demand, supply, vals, price_grid)
        total, br = supply.supply_at(p_star, ts, vals)

        row = {
            "timestamp": ts,
            "price": p_star,
            "q_cleared": q_star,
            "Q_wind": br["wind"],
            "Q_solar": br["solar"],
            "Q_nuclear": br["nuclear"],
            "Q_coal": br["coal"],
            "Q_gas": br["gas"],
        }
        # store drivers & regimes (wide)
        for k, v in vals.items():
            row[k] = v
        row.update(labs)
        rows.append(row)

    return pd.DataFrame(rows)
