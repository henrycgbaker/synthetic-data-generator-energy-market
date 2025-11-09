"""
Test availability edge cases and extreme values.
"""

import numpy as np
import pandas as pd
import pytest

from synthetic_data_pkg.config import DemandConfig, TopConfig
from synthetic_data_pkg.demand import DemandCurve
from synthetic_data_pkg.simulate import find_equilibrium
from synthetic_data_pkg.supply import SupplyCurve


@pytest.mark.unit
class TestAvailabilityEdgeCases:
    """Test supply behavior with extreme availability values"""

    def test_nuclear_perfect_reliability(self):
        """Test with nuclear at 100% availability"""
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
            "avail.nuclear": 1.0,  # PERFECT
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
        price = 50.0

        total, breakdown = supply.supply_at(price, ts, vals)

        # Nuclear should produce at full capacity
        assert breakdown["nuclear"] == 6000.0

    def test_coal_complete_outage(self):
        """Test with coal at 0% availability (complete outage)"""
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
            "avail.coal": 0.0,  # COMPLETE OUTAGE
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
        price = 100.0  # High price

        total, breakdown = supply.supply_at(price, ts, vals)

        # Coal should produce nothing
        assert breakdown["coal"] == 0.0

    def test_gas_complete_outage(self):
        """Test with gas at 0% availability"""
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
            "avail.gas": 0.0,  # COMPLETE OUTAGE
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
        price = 100.0

        total, breakdown = supply.supply_at(price, ts, vals)

        # Gas should produce nothing
        assert breakdown["gas"] == 0.0

    def test_all_thermal_at_zero_availability(self):
        """Test with all thermal generation offline"""
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
            "avail.nuclear": 0.0,  # OFFLINE
            "cap.wind": 7000.0,
            "cap.solar": 5000.0,
            "cap.coal": 8000.0,
            "avail.coal": 0.0,  # OFFLINE
            "cap.gas": 12000.0,
            "avail.gas": 0.0,  # OFFLINE
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

        # Should find equilibrium with renewables only
        assert not np.isnan(q_star)
        assert not np.isnan(p_star)

        _, breakdown = supply.supply_at(p_star, ts, vals)
        assert breakdown["nuclear"] == 0.0
        assert breakdown["coal"] == 0.0
        assert breakdown["gas"] == 0.0

    def test_all_sources_at_perfect_availability(self):
        """Test with all sources at 100% availability"""
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
            "avail.nuclear": 1.0,  # PERFECT
            "cap.wind": 7000.0,
            "cap.solar": 5000.0,
            "cap.coal": 8000.0,
            "avail.coal": 1.0,  # PERFECT
            "cap.gas": 12000.0,
            "avail.gas": 1.0,  # PERFECT
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
        price = 100.0

        total, breakdown = supply.supply_at(price, ts, vals)

        # Nuclear should produce at full capacity
        assert breakdown["nuclear"] == 6000.0
        # Thermal should produce at full capacity
        assert breakdown["coal"] == 8000.0
        assert breakdown["gas"] == 12000.0

    def test_availability_time_varying(self):
        """Test that availability can vary over time"""
        from synthetic_data_pkg.scenario import build_schedules

        # Create time-varying availability
        schedules = build_schedules(
            start_ts="2024-01-01",
            days=2,
            freq="h",
            seed=42,
            supply_regime_planner={"mode": "local_only"},
            variables={
                "fuel.gas": {
                    "regimes": [{"name": "s", "dist": {"kind": "const", "v": 30.0}}]
                },
                "fuel.coal": {
                    "regimes": [{"name": "s", "dist": {"kind": "const", "v": 25.0}}]
                },
                "avail.coal": {
                    "regimes": [
                        {
                            "name": "normal",
                            "dist": {"kind": "const", "v": 0.90},
                            "breakpoints": [
                                {"date": "2024-01-01", "transition_hours": 0}
                            ],
                        },
                        {
                            "name": "outage",
                            "dist": {
                                "kind": "const",
                                "v": 0.0,
                            },  # Complete outage day 2
                            "breakpoints": [
                                {"date": "2024-01-02", "transition_hours": 0}
                            ],
                        },
                    ]
                },
            },
            series_map={},
        )

        # Check day 1
        ts_day1 = pd.Timestamp("2024-01-01 12:00")
        avail_day1, _ = schedules["avail.coal"].value_at(ts_day1)
        assert avail_day1 == 0.90

        # Check day 2
        ts_day2 = pd.Timestamp("2024-01-02 12:00")
        avail_day2, _ = schedules["avail.coal"].value_at(ts_day2)
        assert avail_day2 == 0.0

    @pytest.mark.parametrize("avail", [0.0, 0.25, 0.50, 0.75, 1.0])
    def test_availability_spectrum(self, avail):
        """Test nuclear availability across full spectrum 0% to 100%"""
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

        capacity = 6000.0
        vals = {
            "cap.nuclear": capacity,
            "avail.nuclear": avail,
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

        # At low price, nuclear should bid in fully (must-run)
        price = 0.0  # Above bid.nuclear.max (-50.0)
        total, breakdown = supply.supply_at(price, ts, vals)

        # Nuclear output should equal capacity * availability
        expected_nuclear = capacity * avail
        assert abs(breakdown["nuclear"] - expected_nuclear) < 1.0

    def test_very_low_availability(self):
        """Test with very low but non-zero availability (1%)"""
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
            "avail.nuclear": 0.01,  # 1% only
            "cap.wind": 7000.0,
            "cap.solar": 5000.0,
            "cap.coal": 8000.0,
            "avail.coal": 0.01,  # 1% only
            "cap.gas": 12000.0,
            "avail.gas": 0.01,  # 1% only
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
        price = 100.0

        total, breakdown = supply.supply_at(price, ts, vals)

        # Should produce very little but not crash
        assert breakdown["nuclear"] == pytest.approx(60.0, abs=1.0)  # 6000 * 0.01
        assert breakdown["coal"] == pytest.approx(80.0, abs=1.0)  # 8000 * 0.01
        assert breakdown["gas"] == pytest.approx(120.0, abs=1.0)  # 12000 * 0.01
