"""
Test simulation behavior across different time scales and frequencies.
"""

import numpy as np
import pandas as pd
import pytest
from synthetic_data_pkg.scenario import build_schedules
from synthetic_data_pkg.simulate import simulate_timeseries
from synthetic_data_pkg.config import DemandConfig, TopConfig, IOConfig


@pytest.mark.integration
class TestTimeScaleRobustness:
    """Test simulations across different time scales"""

    def test_single_hour_simulation(self, temp_output_dir):
        """Test minimum viable simulation (1 hour)"""
        config = TopConfig(
            start_ts="2024-01-01 00:00",
            days=1/24,  # 1 hour
            freq="h",
            seed=42,
            price_grid=list(range(-100, 201, 10)),
            demand=DemandConfig(
                inelastic=False,
                base_intercept=200.0,
                slope=-0.006,
                daily_seasonality=False,
                annual_seasonality=False,
            ),
            supply_regime_planner={"mode": "local_only"},
            variables={
                "fuel.gas": {"regimes": [{"name": "s", "dist": {"kind": "const", "v": 30.0}}]},
                "fuel.coal": {"regimes": [{"name": "s", "dist": {"kind": "const", "v": 25.0}}]},
                "cap.nuclear": {"regimes": [{"name": "c", "dist": {"kind": "const", "v": 6000.0}}]},
                "cap.coal": {"regimes": [{"name": "c", "dist": {"kind": "const", "v": 8000.0}}]},
                "cap.gas": {"regimes": [{"name": "c", "dist": {"kind": "const", "v": 12000.0}}]},
                "cap.wind": {"regimes": [{"name": "c", "dist": {"kind": "const", "v": 7000.0}}]},
                "cap.solar": {"regimes": [{"name": "c", "dist": {"kind": "const", "v": 5000.0}}]},
                "avail.nuclear": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": 0.95}}]},
                "avail.coal": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": 0.90}}]},
                "avail.gas": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": 0.95}}]},
                "eta_lb.coal": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": 0.33}}]},
                "eta_ub.coal": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": 0.38}}]},
                "eta_lb.gas": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": 0.48}}]},
                "eta_ub.gas": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": 0.55}}]},
                "bid.nuclear.min": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": -200.0}}]},
                "bid.nuclear.max": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": -50.0}}]},
                "bid.wind.min": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": -200.0}}]},
                "bid.wind.max": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": -50.0}}]},
                "bid.solar.min": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": -200.0}}]},
                "bid.solar.max": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": -50.0}}]},
            },
            empirical_series={},
            planned_outages={"enabled": False},
            renewable_availability_mode="weather_simulation",
            io=IOConfig(
                out_dir=str(temp_output_dir),
                dataset_name="test_1hour",
                add_timestamp=False,
                save_csv=False,
            ),
        )

        schedules = build_schedules(
            start_ts=config.start_ts,
            days=config.days,
            freq=config.freq,
            seed=config.seed,
            supply_regime_planner=config.supply_regime_planner.model_dump() if hasattr(config.supply_regime_planner, 'model_dump') else config.supply_regime_planner,
            variables={k: (v.model_dump() if hasattr(v, 'model_dump') else v) for k, v in config.variables.items()},
            series_map={},
        )

        hours = int(config.days * 24)
        df = simulate_timeseries(
            start_ts=config.start_ts,
            hours=hours,
            demand_cfg=config.demand.model_dump() if hasattr(config.demand, 'model_dump') else config.demand,
            schedules=schedules,
            price_grid=np.array(config.price_grid),
            seed=config.seed,
            config=config,
            planned_outages_cfg=config.planned_outages.model_dump() if hasattr(config.planned_outages, 'model_dump') else config.planned_outages,
        )

        # Should have exactly 1 timestep
        assert len(df) == 1
        assert not df["price"].isna().any()
        assert not df["q_cleared"].isna().any()

    @pytest.mark.slow
    def test_five_year_simulation(self, temp_output_dir):
        """Test long-term simulation (5 years)"""
        config = TopConfig(
            start_ts="2024-01-01 00:00",
            days=365 * 5,  # 5 years
            freq="h",
            seed=42,
            price_grid=list(range(-100, 201, 10)),
            demand=DemandConfig(
                inelastic=False,
                base_intercept=200.0,
                slope=-0.006,
                daily_seasonality=False,
                annual_seasonality=False,
            ),
            supply_regime_planner={"mode": "local_only"},
            variables={
                "fuel.gas": {"regimes": [{"name": "s", "dist": {"kind": "const", "v": 30.0}}]},
                "fuel.coal": {"regimes": [{"name": "s", "dist": {"kind": "const", "v": 25.0}}]},
                "cap.nuclear": {"regimes": [{"name": "c", "dist": {"kind": "const", "v": 6000.0}}]},
                "cap.coal": {"regimes": [{"name": "c", "dist": {"kind": "const", "v": 8000.0}}]},
                "cap.gas": {"regimes": [{"name": "c", "dist": {"kind": "const", "v": 12000.0}}]},
                "cap.wind": {"regimes": [{"name": "c", "dist": {"kind": "const", "v": 7000.0}}]},
                "cap.solar": {"regimes": [{"name": "c", "dist": {"kind": "const", "v": 5000.0}}]},
                "avail.nuclear": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": 0.95}}]},
                "avail.coal": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": 0.90}}]},
                "avail.gas": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": 0.95}}]},
                "eta_lb.coal": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": 0.33}}]},
                "eta_ub.coal": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": 0.38}}]},
                "eta_lb.gas": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": 0.48}}]},
                "eta_ub.gas": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": 0.55}}]},
                "bid.nuclear.min": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": -200.0}}]},
                "bid.nuclear.max": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": -50.0}}]},
                "bid.wind.min": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": -200.0}}]},
                "bid.wind.max": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": -50.0}}]},
                "bid.solar.min": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": -200.0}}]},
                "bid.solar.max": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": -50.0}}]},
            },
            empirical_series={},
            planned_outages={"enabled": False},
            renewable_availability_mode="weather_simulation",
            io=IOConfig(
                out_dir=str(temp_output_dir),
                dataset_name="test_5year",
                add_timestamp=False,
                save_csv=False,
            ),
        )

        schedules = build_schedules(
            start_ts=config.start_ts,
            days=config.days,
            freq=config.freq,
            seed=config.seed,
            supply_regime_planner=config.supply_regime_planner.model_dump() if hasattr(config.supply_regime_planner, 'model_dump') else config.supply_regime_planner,
            variables={k: (v.model_dump() if hasattr(v, 'model_dump') else v) for k, v in config.variables.items()},
            series_map={},
        )

        hours = int(config.days * 24)
        df = simulate_timeseries(
            start_ts=config.start_ts,
            hours=hours,
            demand_cfg=config.demand.model_dump() if hasattr(config.demand, 'model_dump') else config.demand,
            schedules=schedules,
            price_grid=np.array(config.price_grid),
            seed=config.seed,
            config=config,
            planned_outages_cfg=config.planned_outages.model_dump() if hasattr(config.planned_outages, 'model_dump') else config.planned_outages,
        )

        # Should have 5 years of hourly data
        expected_hours = 365 * 5 * 24
        assert len(df) == expected_hours
        assert not df["price"].isna().any()
        assert not df["q_cleared"].isna().any()

    @pytest.mark.slow
    def test_ten_year_simulation(self, temp_output_dir):
        """Test very long-term simulation (10 years)"""
        config = TopConfig(
            start_ts="2020-01-01 00:00",
            days=365 * 10,  # 10 years
            freq="h",
            seed=42,
            price_grid=list(range(-100, 201, 10)),
            demand=DemandConfig(
                inelastic=False,
                base_intercept=200.0,
                slope=-0.006,
                daily_seasonality=False,
                annual_seasonality=False,
            ),
            supply_regime_planner={"mode": "local_only"},
            variables={
                "fuel.gas": {"regimes": [{"name": "s", "dist": {"kind": "const", "v": 30.0}}]},
                "fuel.coal": {"regimes": [{"name": "s", "dist": {"kind": "const", "v": 25.0}}]},
                "cap.nuclear": {"regimes": [{"name": "c", "dist": {"kind": "const", "v": 6000.0}}]},
                "cap.coal": {"regimes": [{"name": "c", "dist": {"kind": "const", "v": 8000.0}}]},
                "cap.gas": {"regimes": [{"name": "c", "dist": {"kind": "const", "v": 12000.0}}]},
                "cap.wind": {"regimes": [{"name": "c", "dist": {"kind": "const", "v": 7000.0}}]},
                "cap.solar": {"regimes": [{"name": "c", "dist": {"kind": "const", "v": 5000.0}}]},
                "avail.nuclear": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": 0.95}}]},
                "avail.coal": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": 0.90}}]},
                "avail.gas": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": 0.95}}]},
                "eta_lb.coal": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": 0.33}}]},
                "eta_ub.coal": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": 0.38}}]},
                "eta_lb.gas": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": 0.48}}]},
                "eta_ub.gas": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": 0.55}}]},
                "bid.nuclear.min": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": -200.0}}]},
                "bid.nuclear.max": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": -50.0}}]},
                "bid.wind.min": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": -200.0}}]},
                "bid.wind.max": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": -50.0}}]},
                "bid.solar.min": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": -200.0}}]},
                "bid.solar.max": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": -50.0}}]},
            },
            empirical_series={},
            planned_outages={"enabled": False},
            renewable_availability_mode="weather_simulation",
            io=IOConfig(
                out_dir=str(temp_output_dir),
                dataset_name="test_10year",
                add_timestamp=False,
                save_csv=False,
            ),
        )

        schedules = build_schedules(
            start_ts=config.start_ts,
            days=config.days,
            freq=config.freq,
            seed=config.seed,
            supply_regime_planner=config.supply_regime_planner.model_dump() if hasattr(config.supply_regime_planner, 'model_dump') else config.supply_regime_planner,
            variables={k: (v.model_dump() if hasattr(v, 'model_dump') else v) for k, v in config.variables.items()},
            series_map={},
        )

        hours = int(config.days * 24)
        df = simulate_timeseries(
            start_ts=config.start_ts,
            hours=hours,
            demand_cfg=config.demand.model_dump() if hasattr(config.demand, 'model_dump') else config.demand,
            schedules=schedules,
            price_grid=np.array(config.price_grid),
            seed=config.seed,
            config=config,
            planned_outages_cfg=config.planned_outages.model_dump() if hasattr(config.planned_outages, 'model_dump') else config.planned_outages,
        )

        # Should have 10 years of data
        expected_hours = 365 * 10 * 24
        assert len(df) == expected_hours

    def test_leap_year_handling(self, temp_output_dir):
        """Test simulation spanning a leap year"""
        config = TopConfig(
            start_ts="2023-12-01 00:00",
            days=365 + 60,  # Dec 2023 + all of 2024 (leap year) + Jan 2025
            freq="h",
            seed=42,
            price_grid=list(range(-100, 201, 10)),
            demand=DemandConfig(
                inelastic=False,
                base_intercept=200.0,
                slope=-0.006,
                daily_seasonality=False,
                annual_seasonality=False,
            ),
            supply_regime_planner={"mode": "local_only"},
            variables={
                "fuel.gas": {"regimes": [{"name": "s", "dist": {"kind": "const", "v": 30.0}}]},
                "fuel.coal": {"regimes": [{"name": "s", "dist": {"kind": "const", "v": 25.0}}]},
                "cap.nuclear": {"regimes": [{"name": "c", "dist": {"kind": "const", "v": 6000.0}}]},
                "cap.coal": {"regimes": [{"name": "c", "dist": {"kind": "const", "v": 8000.0}}]},
                "cap.gas": {"regimes": [{"name": "c", "dist": {"kind": "const", "v": 12000.0}}]},
                "cap.wind": {"regimes": [{"name": "c", "dist": {"kind": "const", "v": 7000.0}}]},
                "cap.solar": {"regimes": [{"name": "c", "dist": {"kind": "const", "v": 5000.0}}]},
                "avail.nuclear": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": 0.95}}]},
                "avail.coal": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": 0.90}}]},
                "avail.gas": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": 0.95}}]},
                "eta_lb.coal": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": 0.33}}]},
                "eta_ub.coal": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": 0.38}}]},
                "eta_lb.gas": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": 0.48}}]},
                "eta_ub.gas": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": 0.55}}]},
                "bid.nuclear.min": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": -200.0}}]},
                "bid.nuclear.max": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": -50.0}}]},
                "bid.wind.min": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": -200.0}}]},
                "bid.wind.max": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": -50.0}}]},
                "bid.solar.min": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": -200.0}}]},
                "bid.solar.max": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": -50.0}}]},
            },
            empirical_series={},
            planned_outages={"enabled": False},
            renewable_availability_mode="weather_simulation",
            io=IOConfig(
                out_dir=str(temp_output_dir),
                dataset_name="test_leap",
                add_timestamp=False,
                save_csv=False,
            ),
        )

        schedules = build_schedules(
            start_ts=config.start_ts,
            days=config.days,
            freq=config.freq,
            seed=config.seed,
            supply_regime_planner=config.supply_regime_planner.model_dump() if hasattr(config.supply_regime_planner, 'model_dump') else config.supply_regime_planner,
            variables={k: (v.model_dump() if hasattr(v, 'model_dump') else v) for k, v in config.variables.items()},
            series_map={},
        )

        hours = int(config.days * 24)
        df = simulate_timeseries(
            start_ts=config.start_ts,
            hours=hours,
            demand_cfg=config.demand.model_dump() if hasattr(config.demand, 'model_dump') else config.demand,
            schedules=schedules,
            price_grid=np.array(config.price_grid),
            seed=config.seed,
            config=config,
            planned_outages_cfg=config.planned_outages.model_dump() if hasattr(config.planned_outages, 'model_dump') else config.planned_outages,
        )

        # Verify Feb 29, 2024 exists in the data
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        feb_29_2024 = pd.to_datetime("2024-02-29").date()
        assert feb_29_2024 in df["date"].values, "Leap day (Feb 29, 2024) missing from simulation"

    @pytest.mark.parametrize("days", [1, 7, 30, 90, 180, 365])
    def test_different_simulation_lengths(self, days, temp_output_dir):
        """Test simulations of various lengths"""
        config = TopConfig(
            start_ts="2024-01-01 00:00",
            days=days,
            freq="h",
            seed=42,
            price_grid=list(range(-100, 201, 10)),
            demand=DemandConfig(
                inelastic=False,
                base_intercept=200.0,
                slope=-0.006,
                daily_seasonality=False,
                annual_seasonality=False,
            ),
            supply_regime_planner={"mode": "local_only"},
            variables={
                "fuel.gas": {"regimes": [{"name": "s", "dist": {"kind": "const", "v": 30.0}}]},
                "fuel.coal": {"regimes": [{"name": "s", "dist": {"kind": "const", "v": 25.0}}]},
                "cap.nuclear": {"regimes": [{"name": "c", "dist": {"kind": "const", "v": 6000.0}}]},
                "cap.coal": {"regimes": [{"name": "c", "dist": {"kind": "const", "v": 8000.0}}]},
                "cap.gas": {"regimes": [{"name": "c", "dist": {"kind": "const", "v": 12000.0}}]},
                "cap.wind": {"regimes": [{"name": "c", "dist": {"kind": "const", "v": 7000.0}}]},
                "cap.solar": {"regimes": [{"name": "c", "dist": {"kind": "const", "v": 5000.0}}]},
                "avail.nuclear": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": 0.95}}]},
                "avail.coal": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": 0.90}}]},
                "avail.gas": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": 0.95}}]},
                "eta_lb.coal": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": 0.33}}]},
                "eta_ub.coal": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": 0.38}}]},
                "eta_lb.gas": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": 0.48}}]},
                "eta_ub.gas": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": 0.55}}]},
                "bid.nuclear.min": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": -200.0}}]},
                "bid.nuclear.max": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": -50.0}}]},
                "bid.wind.min": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": -200.0}}]},
                "bid.wind.max": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": -50.0}}]},
                "bid.solar.min": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": -200.0}}]},
                "bid.solar.max": {"regimes": [{"name": "b", "dist": {"kind": "const", "v": -50.0}}]},
            },
            empirical_series={},
            planned_outages={"enabled": False},
            renewable_availability_mode="weather_simulation",
            io=IOConfig(
                out_dir=str(temp_output_dir),
                dataset_name=f"test_{days}days",
                add_timestamp=False,
                save_csv=False,
            ),
        )

        schedules = build_schedules(
            start_ts=config.start_ts,
            days=config.days,
            freq=config.freq,
            seed=config.seed,
            supply_regime_planner=config.supply_regime_planner.model_dump() if hasattr(config.supply_regime_planner, 'model_dump') else config.supply_regime_planner,
            variables={k: (v.model_dump() if hasattr(v, 'model_dump') else v) for k, v in config.variables.items()},
            series_map={},
        )

        hours = int(config.days * 24)
        df = simulate_timeseries(
            start_ts=config.start_ts,
            hours=hours,
            demand_cfg=config.demand.model_dump() if hasattr(config.demand, 'model_dump') else config.demand,
            schedules=schedules,
            price_grid=np.array(config.price_grid),
            seed=config.seed,
            config=config,
            planned_outages_cfg=config.planned_outages.model_dump() if hasattr(config.planned_outages, 'model_dump') else config.planned_outages,
        )

        # Should have correct number of hours
        expected_hours = days * 24
        assert len(df) == expected_hours
        assert not df["price"].isna().any()