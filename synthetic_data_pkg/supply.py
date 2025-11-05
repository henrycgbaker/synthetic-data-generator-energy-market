from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import linear_ramp


class WindWeatherModel:
    """AR(1) model for wind capacity factors"""

    def __init__(self, params: Dict, rng_seed: int):
        self.base_cf = params.get("base_capacity_factor", 0.45)
        self.rho = params.get("persistence", 0.85)
        self.sigma = params.get("volatility", 0.15)
        self._rng = np.random.default_rng(rng_seed)
        self._cache: Dict[pd.Timestamp, float] = {}
        self._last_ts: Optional[pd.Timestamp] = None

    def availability_at(self, ts: pd.Timestamp) -> float:
        """Calculate wind availability (capacity factor) at given timestamp"""
        key = ts.floor("h")
        if key in self._cache:
            return self._cache[key]

        # New day: reinitialize
        if (self._last_ts is None) or (self._last_ts.floor("h").date() != key.date()):
            cf = np.clip(self._rng.normal(self.base_cf, 0.10), 0.0, 1.0)
        else:
            # AR(1): cf_t = base_cf + rho*(cf_{t-1} - base_cf) + sigma*epsilon
            prev_cf = self._cache.get(self._last_ts.floor("h"), self.base_cf)
            cf = (
                self.base_cf
                + self.rho * (prev_cf - self.base_cf)
                + self.sigma * self._rng.normal()
            )
            cf = np.clip(cf, 0.0, 1.0)

        self._cache[key] = float(cf)
        self._last_ts = key
        return float(cf)


class SolarWeatherModel:
    """Sinusoidal daily pattern for solar capacity factors"""

    def __init__(self, params: Dict):
        self.sunrise = params.get("sunrise_hour", 6)
        self.sunset = params.get("sunset_hour", 20)
        self.peak_cf = params.get("peak_capacity_factor", 0.35)

    def availability_at(self, ts: pd.Timestamp) -> float:
        """Calculate solar availability (capacity factor) at given timestamp"""
        hour = ts.hour
        if self.sunrise <= hour < self.sunset:
            # Sinusoidal shape from sunrise to sunset
            x = (hour - self.sunrise) / (self.sunset - self.sunrise)
            shape = np.sin(np.pi * x)
            return float(self.peak_cf * shape)
        return 0.0


class SupplyCurve:
    """
    Data-driven supply curve with two modes for renewable availability:
    1. weather_simulation: Uses weather models to calculate avail.wind/solar
    2. direct: Uses avail.wind/solar from RV schedules/empirical data

    ALL technologies: output = capacity * availability
    """

    def __init__(self, config, rng_seed: int = 42):
        self._rng = np.random.default_rng(rng_seed)
        self._mode = config.renewable_availability_mode

        # Initialize weather models if in simulation mode
        if self._mode == "weather_simulation":
            self._wind_weather = WindWeatherModel(
                config.weather_simulation.wind.params, rng_seed
            )
            self._solar_weather = SolarWeatherModel(
                config.weather_simulation.solar.params
            )
        else:
            self._wind_weather = None
            self._solar_weather = None

    def _get_wind_availability(self, ts: pd.Timestamp, vals: Dict[str, float]) -> float:
        """Get wind availability based on mode"""
        if self._mode == "weather_simulation":
            return self._wind_weather.availability_at(ts)
        else:  # direct mode
            return vals.get("avail.wind", 0.0)

    def _get_solar_availability(
        self, ts: pd.Timestamp, vals: Dict[str, float]
    ) -> float:
        """Get solar availability based on mode"""
        if self._mode == "weather_simulation":
            return self._solar_weather.availability_at(ts)
        else:  # direct mode
            return vals.get("avail.solar", 0.0)

    def _wind_output(self, ts: pd.Timestamp, vals: Dict[str, float]) -> float:
        """Wind output = capacity * availability"""
        cap = vals.get("cap.wind", 0.0)
        if cap <= 0:
            return 0.0
        avail = self._get_wind_availability(ts, vals)
        return float(cap * avail)

    def _solar_output(self, ts: pd.Timestamp, vals: Dict[str, float]) -> float:
        """Solar output = capacity * availability"""
        cap = vals.get("cap.solar", 0.0)
        if cap <= 0:
            return 0.0
        avail = self._get_solar_availability(ts, vals)
        return float(cap * avail)

    @staticmethod
    def _mc_bounds(
        fuel_price: float, eta_lb: float, eta_ub: float
    ) -> Tuple[float, float]:
        if eta_lb <= 0 or eta_ub <= 0:
            return float("inf"), float("inf")
        return fuel_price / eta_ub, fuel_price / eta_lb

    def _thermal_output(self, price: float, vals: Dict[str, float], tech: str) -> float:
        """Thermal output with marginal cost bid curve"""
        cap = vals.get(f"cap.{tech}", 0.0) * vals.get(f"avail.{tech}", 0.0)
        if cap <= 0:
            return 0.0
        p_low, p_high = self._mc_bounds(
            vals[f"fuel.{tech}"],
            vals.get(f"eta_lb.{tech}", 0.0),
            vals.get(f"eta_ub.{tech}", 0.0),
        )
        return linear_ramp(price, p_low, p_high, cap)

    def _nuclear_output(self, vals: Dict[str, float]) -> float:
        """Nuclear output = capacity * availability (must-run)"""
        return vals.get("cap.nuclear", 0.0) * vals.get("avail.nuclear", 0.0)

    def _renewable_output(
        self, price: float, vals: Dict[str, float], tech: str, base_output: float
    ) -> float:
        """
        Apply linear bid curve to renewable/nuclear output.

        If price >= bid_max: full output (base_output)
        If price < bid_min: zero output
        If bid_min <= price < bid_max: linear interpolation
        """
        if base_output <= 0:
            return 0.0

        bid_min = vals.get(f"bid.{tech}.min", -200.0)
        bid_max = vals.get(f"bid.{tech}.max", -50.0)

        return linear_ramp(price, bid_min, bid_max, base_output)

    def supply_at(
        self, price: float, ts: pd.Timestamp, vals: Dict[str, float]
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate total supply and breakdown at given price and time"""
        # Calculate base outputs (capacity * availability)
        nuc_base = self._nuclear_output(vals)
        wind_base = self._wind_output(ts, vals)
        solar_base = self._solar_output(ts, vals)

        # Apply downward-sloping bid curves to renewables/nuclear
        nuc_q = self._renewable_output(price, vals, "nuclear", nuc_base)
        wind_q = self._renewable_output(price, vals, "wind", wind_base)
        solar_q = self._renewable_output(price, vals, "solar", solar_base)

        # Thermal plants have upward-sloping marginal cost curves
        coal_q = self._thermal_output(price, vals, "coal")
        gas_q = self._thermal_output(price, vals, "gas")

        br = {
            "wind": wind_q,
            "solar": solar_q,
            "nuclear": nuc_q,
            "coal": coal_q,
            "gas": gas_q,
        }
        return sum(br.values()), br

    def curve_for_time(
        self, ts: pd.Timestamp, vals: Dict[str, float], price_grid
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Generate full supply curve across price grid"""
        totals = []
        comp = {k: [] for k in ["wind", "solar", "nuclear", "coal", "gas"]}
        for p in price_grid:
            q, br = self.supply_at(float(p), ts, vals)
            totals.append(q)
            for k in comp:
                comp[k].append(br[k])
        return np.array(totals), {k: np.array(v) for k, v in comp.items()}

    def supply_price_at_quantity(
        self, q: float, ts: pd.Timestamp, vals: Dict[str, float], price_grid
    ) -> float:
        """Find price where supply equals given quantity (inverse supply curve)"""
        Q, _ = self.curve_for_time(ts, vals, price_grid)
        idx = np.searchsorted(Q, q, side="left")
        if idx == 0:
            return float(price_grid[0])
        if idx >= len(price_grid):
            return float(price_grid[-1])
        q0, q1 = Q[idx - 1], Q[idx]
        p0, p1 = price_grid[idx - 1], price_grid[idx]
        if q1 == q0:
            return float(p1)
        w = (q - q0) / (q1 - q0)
        return float(p0 * (1 - w) + p1 * w)
