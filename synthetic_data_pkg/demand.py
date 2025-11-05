"""
this module defines the exogenous demand curve class.
wraps around a passed config and provides methods to compute demand given prices (and vice versa).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DemandConfig


class DemandCurve:
    def __init__(self, cfg: DemandConfig):
        self.cfg = cfg

    def _season(self, ts: pd.Timestamp) -> float:
        """Daily and weekly seasonality (hour of day and weekend effect)"""
        if not self.cfg.daily_seasonality:
            return 1.0

        h = ts.hour
        dow = ts.dayofweek
        day_bump = 1.0 + self.cfg.day_amp * np.cos(
            (h - self.cfg.day_peak_hour) / 12 * np.pi
        )
        weekend = 1.0 - (self.cfg.weekend_drop if dow >= 5 else 0.0)
        return max(0.0, day_bump * weekend)

    def _annual_season(self, ts: pd.Timestamp) -> float:
        """
        Annual seasonality with smooth interpolation between winter and summer peaks.
        Uses a cosine function to create smooth transitions:
        - Winter peak (Dec-Feb): maximum multiplier
        - Summer trough (Jun-Aug): minimum multiplier
        - Spring/Fall: smooth interpolation

        Returns a multiplier to apply to base demand.
        """
        if not self.cfg.annual_seasonality:
            return 1.0

        # Day of year
        doy = ts.dayofyear
        days_in_year = 366 if ts.is_leap_year else 365

        # Convert to radians, with peak in winter (Jan 15 ≈ day 15)
        # Offset by 15 days so peak is mid-January
        angle = 2 * np.pi * (doy - 15) / days_in_year

        # Cosine wave: +1 in winter, -1 in summer
        seasonal_wave = np.cos(angle)

        # Scale by amplitudes: winter_amp when +1, summer_amp when -1
        # Average the two amplitudes and scale the wave
        avg_amp = (self.cfg.winter_amp - self.cfg.summer_amp) / 2
        offset = (self.cfg.winter_amp + self.cfg.summer_amp) / 2

        multiplier = 1.0 + offset + avg_amp * seasonal_wave

        return max(0.0, multiplier)

    def q_at_price(self, p: float, ts: pd.Timestamp) -> float:
        """
        Returns quantity demanded at a given price.
        
        Standard inverse demand: P = intercept + slope * Q
        Solving for Q: Q = (P - intercept) / slope
        
        For inelastic demand, returns fixed quantity regardless of price.
        """
        if self.cfg.inelastic:
            # Inelastic: vertical demand curve at base_intercept level
            # Apply both daily and annual seasonality to the fixed quantity
            daily_multiplier = self._season(ts)
            annual_multiplier = self._annual_season(ts)
            # Use base_intercept as the fixed demand level
            fixed_demand = (
                self.cfg.base_intercept * daily_multiplier * annual_multiplier
            )
            return max(0.0, fixed_demand)

        # Standard downward-sloping demand curve: P = intercept + slope * Q
        # Solve for Q: Q = (P - intercept) / slope
        daily_multiplier = self._season(ts)
        annual_multiplier = self._annual_season(ts)
        price_intercept = self.cfg.base_intercept * daily_multiplier * annual_multiplier

        # Q = (P - intercept) / slope
        # For downward sloping, slope is negative, so this gives positive Q when P < intercept
        q = (p - price_intercept) / self.cfg.slope
        return max(0.0, q)

    def p_at_quantity(self, q: float, ts: pd.Timestamp) -> float:
        """
        Returns price at a given quantity demanded.
        
        Standard inverse demand: P = intercept + slope * Q
        
        For inelastic demand, this returns a very high price if q doesn't match fixed demand,
        or a reference price if it does match.
        """
        if self.cfg.inelastic:
            # For inelastic demand, the inverse is not well-defined
            # Return a very high price to signal that demand is fixed
            # This is mainly used in equilibrium finding where we compare supply vs demand prices
            daily_multiplier = self._season(ts)
            annual_multiplier = self._annual_season(ts)
            fixed_demand = (
                self.cfg.base_intercept * daily_multiplier * annual_multiplier
            )

            # If quantity matches fixed demand (within tolerance), return base price
            if abs(q - fixed_demand) < 0.01:
                return self.cfg.base_intercept
            # Otherwise return extreme price to signal mismatch
            elif q < fixed_demand:
                return 1e6  # Very high price (shortage)
            else:
                return -1e6  # Very low price (surplus)

        # Standard inverse demand curve: P = intercept + slope * Q
        daily_multiplier = self._season(ts)
        annual_multiplier = self._annual_season(ts)
        price_intercept = self.cfg.base_intercept * daily_multiplier * annual_multiplier

        return float(price_intercept + self.cfg.slope * q)
