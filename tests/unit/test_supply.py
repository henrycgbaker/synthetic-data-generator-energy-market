"""
Unit tests for supply module.
Tests supply curve and weather models.
"""

import numpy as np
import pandas as pd
import pytest
from synthetic_data_pkg.config import (
    TopConfig,
    WeatherModelConfig,
    WeatherSimulationConfig,
)
from synthetic_data_pkg.supply import SolarWeatherModel, SupplyCurve, WindWeatherModel


@pytest.mark.unit
class TestWindWeatherModel:
    """Unit tests for wind weather model"""

    def test_initialization(self):
        """Test wind weather model initializes correctly"""
        params = {
            "base_capacity_factor": 0.45,
            "persistence": 0.85,
            "volatility": 0.15,
        }
        model = WindWeatherModel(params, rng_seed=42)

        assert model.base_cf == 0.45
        assert model.rho == 0.85
        assert model.sigma == 0.15

    def test_availability_in_range(self):
        """Test wind availability is between 0 and 1"""
        params = {
            "base_capacity_factor": 0.45,
            "persistence": 0.85,
            "volatility": 0.15,
        }
        model = WindWeatherModel(params, rng_seed=42)

        # Test 100 consecutive hours
        for h in range(100):
            ts = pd.Timestamp("2024-01-01") + pd.Timedelta(hours=h)
            avail = model.availability_at(ts)
            assert 0.0 <= avail <= 1.0, f"Availability {avail} out of range at {ts}"

    def test_persistence(self):
        """Test wind has temporal persistence"""
        params = {
            "base_capacity_factor": 0.45,
            "persistence": 0.95,  # High persistence
            "volatility": 0.05,  # Low volatility
        }
        model = WindWeatherModel(params, rng_seed=42)

        vals = []
        for h in range(50):
            ts = pd.Timestamp("2024-01-01") + pd.Timedelta(hours=h)
            vals.append(model.availability_at(ts))

        # Check that consecutive values are similar
        diffs = np.abs(np.diff(vals))
        assert np.mean(diffs) < 0.1  # Small changes hour-to-hour

    def test_caching(self):
        """Test that wind model caches values"""
        params = {
            "base_capacity_factor": 0.45,
            "persistence": 0.85,
            "volatility": 0.15,
        }
        model = WindWeatherModel(params, rng_seed=42)

        ts = pd.Timestamp("2024-01-01 12:00")
        val1 = model.availability_at(ts)
        val2 = model.availability_at(ts)  # Should be cached

        assert val1 == val2


@pytest.mark.unit
class TestSolarWeatherModel:
    """Unit tests for solar weather model"""

    def test_initialization(self):
        """Test solar weather model initializes correctly"""
        params = {
            "sunrise_hour": 6,
            "sunset_hour": 20,
            "peak_capacity_factor": 0.35,
        }
        model = SolarWeatherModel(params)

        assert model.sunrise == 6
        assert model.sunset == 20
        assert model.peak_cf == 0.35

    def test_zero_at_night(self):
        """Test solar availability is zero at night"""
        params = {
            "sunrise_hour": 6,
            "sunset_hour": 20,
            "peak_capacity_factor": 0.35,
        }
        model = SolarWeatherModel(params)

        # Test night hours
        for hour in [0, 1, 2, 3, 4, 5, 20, 21, 22, 23]:
            ts = pd.Timestamp(f"2024-01-01 {hour:02d}:00")
            avail = model.availability_at(ts)
            assert avail == 0.0, f"Solar should be 0 at night hour {hour}, got {avail}"

    def test_positive_during_day(self):
        """Test solar availability is positive during day"""
        params = {
            "sunrise_hour": 6,
            "sunset_hour": 20,
            "peak_capacity_factor": 0.35,
        }
        model = SolarWeatherModel(params)

        # Test day hours (excluding sunset hour which is now exclusive)
        for hour in range(6, 20):
            ts = pd.Timestamp(f"2024-01-01 {hour:02d}:00")
            avail = model.availability_at(ts)
            assert avail >= 0.0, f"Solar negative at hour {hour}: {avail}"
            if 7 <= hour <= 18:  # Should be clearly positive away from edges
                assert (
                    avail > 0.0
                ), f"Solar should be positive at hour {hour}, got {avail}"

    def test_peak_at_noon(self):
        """Test solar peaks around midday"""
        params = {
            "sunrise_hour": 6,
            "sunset_hour": 20,
            "peak_capacity_factor": 0.35,
        }
        model = SolarWeatherModel(params)

        # Midpoint between sunrise and sunset is hour 13
        noon = pd.Timestamp("2024-01-01 13:00")
        morning = pd.Timestamp("2024-01-01 08:00")
        evening = pd.Timestamp("2024-01-01 17:00")

        noon_avail = model.availability_at(noon)
        morning_avail = model.availability_at(morning)
        evening_avail = model.availability_at(evening)

        # Noon should be higher than morning and evening
        assert noon_avail > morning_avail
        assert noon_avail > evening_avail

        # Noon should be close to peak
        assert noon_avail == pytest.approx(0.35, abs=0.01)

    def test_availability_in_range(self):
        """Test solar availability is between 0 and peak"""
        params = {
            "sunrise_hour": 6,
            "sunset_hour": 20,
            "peak_capacity_factor": 0.35,
        }
        model = SolarWeatherModel(params)

        for hour in range(24):
            ts = pd.Timestamp(f"2024-01-01 {hour:02d}:00")
            avail = model.availability_at(ts)
            assert 0.0 <= avail <= 0.35, f"Availability out of range at hour {hour}"


@pytest.mark.unit
class TestSupplyCurve:
    """Unit tests for supply curve"""

    def test_initialization_weather_simulation_mode(self):
        """Test supply curve initializes with weather simulation"""
        # Create minimal config
        config = TopConfig(
            start_ts="2024-01-01",
            days=1,
            supply_regime_planner={"mode": "local_only"},
            renewable_availability_mode="weather_simulation",
            weather_simulation=WeatherSimulationConfig(),
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

        assert supply._mode == "weather_simulation"
        assert supply._wind_weather is not None
        assert supply._solar_weather is not None

    def test_wind_output_with_weather_simulation(self):
        """Test wind output calculation uses weather simulation"""
        config = TopConfig(
            start_ts="2024-01-01",
            days=1,
            supply_regime_planner={"mode": "local_only"},
            renewable_availability_mode="weather_simulation",
            weather_simulation=WeatherSimulationConfig(),
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

        vals = {"cap.wind": 7000.0}
        ts = pd.Timestamp("2024-01-01 12:00")

        output = supply._wind_output(ts, vals)

        # Output should be positive (capacity * availability)
        assert output > 0
        assert output <= 7000.0  # Can't exceed capacity

    def test_solar_output_with_weather_simulation(self):
        """Test solar output calculation uses weather simulation"""
        config = TopConfig(
            start_ts="2024-01-01",
            days=1,
            supply_regime_planner={"mode": "local_only"},
            renewable_availability_mode="weather_simulation",
            weather_simulation=WeatherSimulationConfig(),
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

        vals = {"cap.solar": 5000.0}

        # Daytime
        ts_day = pd.Timestamp("2024-01-01 12:00")
        output_day = supply._solar_output(ts_day, vals)
        assert output_day > 0

        # Nighttime
        ts_night = pd.Timestamp("2024-01-01 23:00")
        output_night = supply._solar_output(ts_night, vals)
        assert output_night == 0.0

    def test_thermal_marginal_cost_calculation(self):
        """Test thermal plant marginal cost bounds"""
        fuel_price = 30.0
        eta_lb = 0.48
        eta_ub = 0.55

        p_low, p_high = SupplyCurve._mc_bounds(fuel_price, eta_lb, eta_ub)

        # p_low = fuel / eta_ub, p_high = fuel / eta_lb
        assert p_low == pytest.approx(30.0 / 0.55, rel=1e-6)
        assert p_high == pytest.approx(30.0 / 0.48, rel=1e-6)
        assert p_low < p_high

    def test_thermal_output_below_marginal_cost(self):
        """Test thermal plant produces nothing below marginal cost"""
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
            "cap.gas": 12000.0,
            "avail.gas": 0.95,
            "fuel.gas": 30.0,
            "eta_lb.gas": 0.48,
            "eta_ub.gas": 0.55,
        }

        # Price below marginal cost
        output = supply._thermal_output(price=40.0, vals=vals, tech="gas")

        # Should produce little or nothing (40 < 30/0.55 = 54.5)
        assert output < 1000.0

    def test_thermal_output_at_high_price(self):
        """Test thermal plant produces full capacity at high price"""
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
            "cap.gas": 12000.0,
            "avail.gas": 0.95,
            "fuel.gas": 30.0,
            "eta_lb.gas": 0.48,
            "eta_ub.gas": 0.55,
        }

        # Price above marginal cost
        output = supply._thermal_output(price=100.0, vals=vals, tech="gas")

        # Should produce at full available capacity
        expected = 12000.0 * 0.95
        assert output == pytest.approx(expected, rel=1e-6)

    def test_supply_at_returns_breakdown(self):
        """Test supply_at returns total and breakdown by technology"""
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
        price = 70.0

        total, breakdown = supply.supply_at(price, ts, vals)

        # Check breakdown has all technologies
        assert "nuclear" in breakdown
        assert "wind" in breakdown
        assert "solar" in breakdown
        assert "coal" in breakdown
        assert "gas" in breakdown

        # Check total equals sum of breakdown
        assert total == pytest.approx(sum(breakdown.values()), rel=1e-6)

        # All values should be non-negative
        for tech, qty in breakdown.items():
            assert qty >= 0, f"{tech} quantity is negative: {qty}"
