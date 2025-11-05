"""
Unit tests for demand module.
Tests the DemandCurve class in isolation.
"""

import pandas as pd
import pytest
from synthetic_data_pkg.config import DemandConfig
from synthetic_data_pkg.demand import DemandCurve


@pytest.mark.unit
class TestDemandCurve:
    """Unit tests for DemandCurve class"""

    def test_initialization(self):
        """Test DemandCurve initializes correctly"""
        cfg = DemandConfig()
        demand = DemandCurve(cfg)
        assert demand.cfg == cfg

    def test_daily_seasonality_flag_off(self):
        """Test that daily_seasonality=False returns flat multiplier"""
        cfg = DemandConfig(
            daily_seasonality=False,
            day_amp=0.5,  # Should be ignored
        )
        demand = DemandCurve(cfg)

        # Test multiple times of day
        ts_morning = pd.Timestamp("2024-01-01 08:00")
        ts_peak = pd.Timestamp("2024-01-01 14:00")
        ts_evening = pd.Timestamp("2024-01-01 20:00")

        assert demand._season(ts_morning) == 1.0
        assert demand._season(ts_peak) == 1.0
        assert demand._season(ts_evening) == 1.0

    def test_daily_seasonality_flag_on(self):
        """Test that daily_seasonality=True applies time-of-day patterns"""
        cfg = DemandConfig(
            daily_seasonality=True,
            day_peak_hour=14,
            day_amp=0.25,
        )
        demand = DemandCurve(cfg)

        ts_peak = pd.Timestamp("2024-01-01 14:00")
        ts_offpeak = pd.Timestamp("2024-01-01 02:00")

        peak_mult = demand._season(ts_peak)
        offpeak_mult = demand._season(ts_offpeak)

        # Peak should be higher than off-peak
        assert peak_mult > offpeak_mult
        # Both should be positive
        assert peak_mult > 0
        assert offpeak_mult > 0

    def test_annual_seasonality_flag_off(self):
        """Test that annual_seasonality=False returns flat multiplier"""
        cfg = DemandConfig(
            annual_seasonality=False,
            winter_amp=0.2,  # Should be ignored
            summer_amp=-0.15,  # Should be ignored
        )
        demand = DemandCurve(cfg)

        ts_winter = pd.Timestamp("2024-01-15")
        ts_summer = pd.Timestamp("2024-07-15")

        assert demand._annual_season(ts_winter) == 1.0
        assert demand._annual_season(ts_summer) == 1.0

    def test_annual_seasonality_flag_on(self):
        """Test that annual_seasonality=True applies seasonal patterns"""
        cfg = DemandConfig(
            annual_seasonality=True,
            winter_amp=0.2,
            summer_amp=-0.15,
        )
        demand = DemandCurve(cfg)

        ts_winter = pd.Timestamp("2024-01-15")
        ts_summer = pd.Timestamp("2024-07-15")

        winter_mult = demand._annual_season(ts_winter)
        summer_mult = demand._annual_season(ts_summer)

        # Winter should be higher than summer
        assert winter_mult > summer_mult
        # Both should be positive
        assert winter_mult > 0
        assert summer_mult > 0

    def test_weekend_effect(self):
        """Test weekend demand reduction"""
        cfg = DemandConfig(
            daily_seasonality=True,
            weekend_drop=0.1,
            day_peak_hour=14,
        )
        demand = DemandCurve(cfg)

        # Monday and Sunday at same hour
        ts_weekday = pd.Timestamp("2024-01-01 14:00")  # Monday
        ts_weekend = pd.Timestamp("2024-01-06 14:00")  # Saturday

        weekday_mult = demand._season(ts_weekday)
        weekend_mult = demand._season(ts_weekend)

        # Weekday should be higher
        assert weekday_mult > weekend_mult

    def test_inelastic_demand(self):
        """Test inelastic demand (quantity doesn't respond to price)"""
        cfg = DemandConfig(
            inelastic=True,
            base_intercept=1000.0,
            daily_seasonality=False,
            annual_seasonality=False,
        )
        demand = DemandCurve(cfg)

        ts = pd.Timestamp("2024-01-01 12:00")

        # Quantity should be same regardless of price
        q_low = demand.q_at_price(p=10.0, ts=ts)
        q_high = demand.q_at_price(p=100.0, ts=ts)

        assert q_low == q_high
        assert q_low == pytest.approx(1000.0, rel=1e-6)

    def test_elastic_demand_downward_sloping(self):
        """Test elastic demand curve is downward sloping"""
        # Standard form: P = 100 - 0.01*Q
        # At Q=0: P=100, At Q=5000: P=50
        cfg = DemandConfig(
            inelastic=False,
            base_intercept=100.0,  # Choke price
            slope=-0.01,  # dP/dQ
            daily_seasonality=False,
            annual_seasonality=False,
        )
        demand = DemandCurve(cfg)

        ts = pd.Timestamp("2024-01-01 12:00")

        q_low = demand.q_at_price(p=90.0, ts=ts)  # P=90: Q=(90-100)/(-0.01)=1000
        q_high = demand.q_at_price(p=50.0, ts=ts)  # P=50: Q=(50-100)/(-0.01)=5000

        # Higher price should give lower quantity
        assert q_low < q_high
        assert q_low == pytest.approx(1000.0, rel=0.01)
        assert q_high == pytest.approx(5000.0, rel=0.01)

    def test_inverse_demand(self):
        """Test p_at_quantity is inverse of q_at_price"""
        # Standard form: P = 200 - 0.005*Q
        cfg = DemandConfig(
            inelastic=False,
            base_intercept=200.0,  # Choke price
            slope=-0.005,  # dP/dQ
            daily_seasonality=False,
            annual_seasonality=False,
        )
        demand = DemandCurve(cfg)

        ts = pd.Timestamp("2024-01-01 12:00")

        # Pick a price
        p_original = 50.0
        q = demand.q_at_price(p=p_original, ts=ts)  # Q = (50-200)/(-0.005) = 30000
        p_recovered = demand.p_at_quantity(q=q, ts=ts)  # P = 200 + (-0.005)*30000 = 50

        assert p_recovered == pytest.approx(p_original, rel=1e-6)

    def test_quantity_non_negative(self):
        """Test that quantity demanded is never negative"""
        # Standard form: P = 200 - 0.01*Q
        cfg = DemandConfig(
            inelastic=False,
            base_intercept=200.0,
            slope=-0.01,
            daily_seasonality=False,
            annual_seasonality=False,
        )
        demand = DemandCurve(cfg)

        ts = pd.Timestamp("2024-01-01 12:00")

        # Very high price (above choke price) should give zero quantity, not negative
        q = demand.q_at_price(p=10000.0, ts=ts)
        assert q == 0.0

    def test_seasonality_multipliers_positive(self):
        """Test that all seasonality multipliers are positive"""
        cfg = DemandConfig(
            daily_seasonality=True,
            annual_seasonality=True,
            day_amp=0.3,
            weekend_drop=0.2,
            winter_amp=0.25,
            summer_amp=-0.2,
        )
        demand = DemandCurve(cfg)

        # Test multiple times throughout year
        for month in range(1, 13):
            for hour in range(0, 24, 6):
                for day in [1, 15]:  # Beginning and middle of month
                    try:
                        ts = pd.Timestamp(f"2024-{month:02d}-{day:02d} {hour:02d}:00")
                        daily = demand._season(ts)
                        annual = demand._annual_season(ts)

                        assert daily >= 0, f"Negative daily multiplier at {ts}"
                        assert annual >= 0, f"Negative annual multiplier at {ts}"
                    except ValueError:
                        # Skip invalid dates (e.g., Feb 30)
                        pass
