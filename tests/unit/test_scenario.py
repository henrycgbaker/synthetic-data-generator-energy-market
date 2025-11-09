"""
Unit tests for scenario module.
Tests scenario building and schedule creation.
"""

import pandas as pd
import pytest

from synthetic_data_pkg.regimes import RegimeSchedule
from synthetic_data_pkg.scenario import build_schedules


@pytest.mark.unit
class TestBuildSchedules:
    """Test schedule building"""

    def test_build_simple_schedule(self):
        """Test building a simple schedule with constant values"""
        variables = {
            "fuel.gas": {
                "regimes": [{"name": "stable", "dist": {"kind": "const", "v": 30.0}}]
            },
            "cap.nuclear": {
                "regimes": [
                    {"name": "constant", "dist": {"kind": "const", "v": 6000.0}}
                ]
            },
        }

        schedules = build_schedules(
            start_ts="2024-01-01",
            days=1,
            freq="h",
            seed=42,
            supply_regime_planner={"mode": "local_only"},
            variables=variables,
            series_map={},
        )

        assert "fuel.gas" in schedules
        assert "cap.nuclear" in schedules
        assert isinstance(schedules["fuel.gas"], RegimeSchedule)

        # Test sampling
        ts = pd.Timestamp("2024-01-01 12:00")
        val, regime = schedules["fuel.gas"].value_at(ts)
        assert val == pytest.approx(30.0, abs=0.1)

    def test_build_schedule_with_linear_growth(self):
        """Test building schedules with linear distributions"""
        variables = {
            "cap.wind": {
                "regimes": [
                    {
                        "name": "growing",
                        "dist": {"kind": "linear", "start": 5000.0, "slope": 10.0},
                    }
                ]
            }
        }

        schedules = build_schedules(
            start_ts="2024-01-01",
            days=1,
            freq="h",
            seed=42,
            supply_regime_planner={"mode": "local_only"},
            variables=variables,
            series_map={},
        )

        # Check values at different times
        ts_0 = pd.Timestamp("2024-01-01 00:00")
        ts_10 = pd.Timestamp("2024-01-01 10:00")

        val_0, _ = schedules["cap.wind"].value_at(ts_0)
        val_10, _ = schedules["cap.wind"].value_at(ts_10)

        assert val_0 == pytest.approx(5000.0, abs=1.0)
        assert val_10 == pytest.approx(5100.0, abs=1.0)  # 5000 + 10*10
        assert val_10 > val_0

    def test_build_schedule_with_multiple_regimes(self):
        """Test building schedules with regime changes"""
        variables = {
            "fuel.gas": {
                "regimes": [
                    {
                        "name": "low",
                        "dist": {"kind": "const", "v": 25.0},
                        "breakpoints": [{"date": "2024-01-01", "transition_hours": 0}],
                    },
                    {
                        "name": "high",
                        "dist": {"kind": "const", "v": 75.0},
                        "breakpoints": [{"date": "2024-01-02", "transition_hours": 0}],
                    },
                ]
            }
        }

        schedules = build_schedules(
            start_ts="2024-01-01",
            days=3,
            freq="h",
            seed=42,
            supply_regime_planner={"mode": "local_only"},
            variables=variables,
            series_map={},
        )

        # Check regime changes
        ts_day1 = pd.Timestamp("2024-01-01 12:00")
        ts_day2 = pd.Timestamp("2024-01-02 12:00")

        val_1, regime_1 = schedules["fuel.gas"].value_at(ts_day1)
        val_2, regime_2 = schedules["fuel.gas"].value_at(ts_day2)

        assert regime_1 == "low"
        assert regime_2 == "high"
        assert val_1 < val_2

    def test_build_all_required_variables(self):
        """Test that all required variables are created"""
        variables = {
            "fuel.gas": {
                "regimes": [{"name": "s", "dist": {"kind": "const", "v": 30.0}}]
            },
            "fuel.coal": {
                "regimes": [{"name": "s", "dist": {"kind": "const", "v": 25.0}}]
            },
            "cap.nuclear": {
                "regimes": [{"name": "s", "dist": {"kind": "const", "v": 6000.0}}]
            },
            "cap.coal": {
                "regimes": [{"name": "s", "dist": {"kind": "const", "v": 8000.0}}]
            },
            "cap.gas": {
                "regimes": [{"name": "s", "dist": {"kind": "const", "v": 12000.0}}]
            },
            "cap.wind": {
                "regimes": [{"name": "s", "dist": {"kind": "const", "v": 7000.0}}]
            },
            "cap.solar": {
                "regimes": [{"name": "s", "dist": {"kind": "const", "v": 5000.0}}]
            },
            "avail.nuclear": {
                "regimes": [{"name": "s", "dist": {"kind": "const", "v": 0.95}}]
            },
            "avail.coal": {
                "regimes": [{"name": "s", "dist": {"kind": "const", "v": 0.90}}]
            },
            "avail.gas": {
                "regimes": [{"name": "s", "dist": {"kind": "const", "v": 0.95}}]
            },
            "eta_lb.coal": {
                "regimes": [{"name": "s", "dist": {"kind": "const", "v": 0.33}}]
            },
            "eta_ub.coal": {
                "regimes": [{"name": "s", "dist": {"kind": "const", "v": 0.38}}]
            },
            "eta_lb.gas": {
                "regimes": [{"name": "s", "dist": {"kind": "const", "v": 0.48}}]
            },
            "eta_ub.gas": {
                "regimes": [{"name": "s", "dist": {"kind": "const", "v": 0.55}}]
            },
            "bid.nuclear.min": {
                "regimes": [{"name": "s", "dist": {"kind": "const", "v": -200.0}}]
            },
            "bid.nuclear.max": {
                "regimes": [{"name": "s", "dist": {"kind": "const", "v": -50.0}}]
            },
            "bid.wind.min": {
                "regimes": [{"name": "s", "dist": {"kind": "const", "v": -200.0}}]
            },
            "bid.wind.max": {
                "regimes": [{"name": "s", "dist": {"kind": "const", "v": -50.0}}]
            },
            "bid.solar.min": {
                "regimes": [{"name": "s", "dist": {"kind": "const", "v": -200.0}}]
            },
            "bid.solar.max": {
                "regimes": [{"name": "s", "dist": {"kind": "const", "v": -50.0}}]
            },
        }

        schedules = build_schedules(
            start_ts="2024-01-01",
            days=1,
            freq="h",
            seed=42,
            supply_regime_planner={"mode": "local_only"},
            variables=variables,
            series_map={},
        )

        # Verify all variables present
        for var_name in variables.keys():
            assert var_name in schedules, f"Missing variable: {var_name}"
            assert isinstance(schedules[var_name], RegimeSchedule)


@pytest.mark.unit
class TestRegimeScheduleValueAt:
    """Test RegimeSchedule.value_at() method"""

    def test_value_at_const_distribution(self):
        """Test value_at with constant distribution"""
        schedule = RegimeSchedule(
            varname="test",
            start_ts=pd.Timestamp("2024-01-01"),
            freq="h",
            segments=[
                {
                    "name": "const",
                    "days": 1,
                    "dist": {"kind": "const", "v": 100.0},
                    "transition_hours": 0,
                }
            ],
            rng=None,
            series_map={},
        )

        ts = pd.Timestamp("2024-01-01 12:00")
        val, regime = schedule.value_at(ts)

        assert val == 100.0
        assert regime == "const"

    def test_value_at_clamped_to_index(self):
        """Test that value_at clamps to schedule index bounds"""
        schedule = RegimeSchedule(
            varname="test",
            start_ts=pd.Timestamp("2024-01-01"),
            freq="h",
            segments=[
                {
                    "name": "s",
                    "days": 1,
                    "dist": {"kind": "const", "v": 100.0},
                    "transition_hours": 0,
                }
            ],
            rng=None,
            series_map={},
        )

        # Request value before start - should clamp
        ts_before = pd.Timestamp("2023-12-31 12:00")
        val, _ = schedule.value_at(ts_before)
        assert val == 100.0  # Should return first value

        # Request value after end - should clamp
        ts_after = pd.Timestamp("2024-01-05 12:00")
        val, _ = schedule.value_at(ts_after)
        assert val == 100.0  # Should return last value
