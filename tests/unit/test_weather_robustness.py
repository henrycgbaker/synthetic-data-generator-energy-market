"""
Test weather models with extreme parameters and edge cases.
"""

import numpy as np
import pandas as pd
import pytest
from synthetic_data_pkg.supply import WindWeatherModel, SolarWeatherModel


@pytest.mark.unit
class TestWindWeatherEdgeCases:
    """Test wind weather model edge cases"""

    def test_wind_at_zero_for_extended_period(self):
        """Test wind can stay at zero for long periods"""
        params = {
            "base_capacity_factor": 0.001,  # Very low base
            "persistence": 0.99,  # Very high persistence
            "volatility": 0.001,  # Very low volatility
        }
        model = WindWeatherModel(params, rng_seed=42)

        # Simulate for many hours
        vals = []
        for h in range(500):
            ts = pd.Timestamp("2024-01-01") + pd.Timedelta(hours=h)
            avail = model.availability_at(ts)
            vals.append(avail)

        # Should have many near-zero values
        near_zero = sum(1 for v in vals if v < 0.1)
        assert near_zero > 100, "Should have many near-zero availability periods"

    def test_wind_at_max_for_extended_period(self):
        """Test wind can stay at maximum for long periods"""
        params = {
            "base_capacity_factor": 0.999,  # Very high base
            "persistence": 0.99,  # Very high persistence
            "volatility": 0.001,  # Very low volatility
        }
        model = WindWeatherModel(params, rng_seed=42)

        # Simulate for many hours
        vals = []
        for h in range(500):
            ts = pd.Timestamp("2024-01-01") + pd.Timedelta(hours=h)
            avail = model.availability_at(ts)
            vals.append(avail)

        # Should have many near-max values
        near_max = sum(1 for v in vals if v > 0.9)
        assert near_max > 100, "Should have many near-maximum availability periods"

    def test_wind_extreme_persistence(self):
        """Test wind with persistence very close to 1 (near random walk)"""
        params = {
            "base_capacity_factor": 0.5,
            "persistence": 0.9999,  # Nearly 1
            "volatility": 0.01,
        }
        model = WindWeatherModel(params, rng_seed=42)

        vals = []
        for h in range(1000):
            ts = pd.Timestamp("2024-01-01") + pd.Timedelta(hours=h)
            vals.append(model.availability_at(ts))

        # With very high persistence, consecutive values should be very similar
        diffs = np.abs(np.diff(vals))
        assert np.mean(diffs) < 0.02, "High persistence should mean small changes"

    def test_wind_zero_volatility(self):
        """Test wind with zero volatility (constant)"""
        params = {
            "base_capacity_factor": 0.45,
            "persistence": 0.85,
            "volatility": 0.0,  # ZERO
        }
        model = WindWeatherModel(params, rng_seed=42)

        vals = []
        for h in range(100):
            ts = pd.Timestamp("2024-01-01") + pd.Timedelta(hours=h)
            vals.append(model.availability_at(ts))

        # With zero volatility, should be nearly constant
        # (may still have small changes due to mean reversion)
        assert np.std(vals) < 0.05, "Zero volatility should mean nearly constant values"

    def test_wind_extreme_volatility(self):
        """Test wind with very high volatility"""
        params = {
            "base_capacity_factor": 0.5,
            "persistence": 0.5,
            "volatility": 0.5,  # VERY HIGH
        }
        model = WindWeatherModel(params, rng_seed=42)

        vals = []
        for h in range(500):
            ts = pd.Timestamp("2024-01-01") + pd.Timedelta(hours=h)
            vals.append(model.availability_at(ts))

        # With high volatility, should see large changes
        diffs = np.abs(np.diff(vals))
        # Should have some large jumps
        large_jumps = sum(1 for d in diffs if d > 0.3)
        assert large_jumps > 10, "High volatility should cause large jumps"


@pytest.mark.unit
class TestSolarWeatherEdgeCases:
    """Test solar weather model edge cases"""

    def test_solar_midnight_winter(self):
        """Test solar at midnight in winter is zero"""
        params = {
            "sunrise_hour": 7,
            "sunset_hour": 17,  # Short winter day
            "peak_capacity_factor": 0.25,
        }
        model = SolarWeatherModel(params)

        ts = pd.Timestamp("2024-01-15 00:00")  # Winter midnight
        avail = model.availability_at(ts)
        assert avail == 0.0

    def test_solar_noon_summer(self):
        """Test solar at noon in summer is at peak"""
        params = {
            "sunrise_hour": 5,
            "sunset_hour": 21,  # Long summer day
            "peak_capacity_factor": 0.40,
        }
        model = SolarWeatherModel(params)

        # Midpoint between 5 and 21 is 13:00
        ts = pd.Timestamp("2024-06-15 13:00")  # Summer noon
        avail = model.availability_at(ts)
        assert avail == pytest.approx(0.40, abs=0.02)

    def test_solar_with_very_short_day(self):
        """Test solar with very short day (polar winter)"""
        params = {
            "sunrise_hour": 10,
            "sunset_hour": 14,  # Only 4 hours of daylight
            "peak_capacity_factor": 0.15,
        }
        model = SolarWeatherModel(params)

        # Should be zero outside 10-14
        for hour in [0, 5, 9, 14, 15, 20, 23]:
            ts = pd.Timestamp(f"2024-01-01 {hour:02d}:00")
            avail = model.availability_at(ts)
            if hour < 10 or hour >= 14:
                assert avail == 0.0, f"Hour {hour} should be zero"

        # Should be positive during 10-13
        for hour in [10, 11, 12, 13]:
            ts = pd.Timestamp(f"2024-01-01 {hour:02d}:00")
            avail = model.availability_at(ts)
            assert avail > 0.0, f"Hour {hour} should be positive"

    def test_solar_with_very_long_day(self):
        """Test solar with very long day (polar summer)"""
        params = {
            "sunrise_hour": 2,
            "sunset_hour": 22,  # 20 hours of daylight
            "peak_capacity_factor": 0.45,
        }
        model = SolarWeatherModel(params)

        # Should be zero for 22-02
        for hour in [22, 23, 0, 1]:
            ts = pd.Timestamp(f"2024-06-15 {hour:02d}:00")
            avail = model.availability_at(ts)
            assert avail == 0.0, f"Hour {hour} should be zero"

        # Should be positive during daylight
        daylight_count = 0
        for hour in range(2, 22):
            ts = pd.Timestamp(f"2024-06-15 {hour:02d}:00")
            avail = model.availability_at(ts)
            if avail > 0:
                daylight_count += 1

        assert daylight_count > 15, "Most daylight hours should have positive availability"

    def test_solar_zero_peak_capacity_factor(self):
        """Test solar with zero peak capacity factor"""
        params = {
            "sunrise_hour": 6,
            "sunset_hour": 20,
            "peak_capacity_factor": 0.0,  # ZERO
        }
        model = SolarWeatherModel(params)

        # Should be zero all day
        for hour in range(24):
            ts = pd.Timestamp(f"2024-01-01 {hour:02d}:00")
            avail = model.availability_at(ts)
            assert avail == 0.0

    def test_solar_perfect_peak_capacity_factor(self):
        """Test solar with peak capacity factor of 1.0"""
        params = {
            "sunrise_hour": 6,
            "sunset_hour": 18,
            "peak_capacity_factor": 1.0,  # PERFECT
        }
        model = SolarWeatherModel(params)

        # At solar noon (midpoint), should be 1.0
        ts = pd.Timestamp("2024-01-01 12:00")
        avail = model.availability_at(ts)
        assert avail == pytest.approx(1.0, abs=0.01)