"""
Integration tests specifically for price responsiveness.
Tests that market prices respond correctly to changes in fuel prices and capacity.
"""

import numpy as np
import pandas as pd
import pytest
from synthetic_data_pkg.config import DemandConfig, IOConfig, TopConfig
from synthetic_data_pkg.scenario import build_schedules
from synthetic_data_pkg.simulate import simulate_timeseries


def _create_test_config(fuel_gas_price, fuel_coal_price, cap_gas, cap_coal, temp_dir, days=2):
    """Helper to create test config"""
    return TopConfig(
        start_ts="2024-01-01 00:00",
        days=days,
        freq="h",
        seed=42,
        price_grid=list(range(-100, 201, 10)),
        demand=DemandConfig(
            inelastic=False,
            base_intercept=200.0,  # Standard form: P = 200 - 0.006*Q
            slope=-0.006,  # dP/dQ
            daily_seasonality=False,
            annual_seasonality=False,
        ),
        supply_regime_planner={"mode": "local_only"},
        variables={
            "fuel.gas": {"regimes": [{"name": "stable", "dist": {"kind": "const", "v": fuel_gas_price}}]},
            "fuel.coal": {"regimes": [{"name": "stable", "dist": {"kind": "const", "v": fuel_coal_price}}]},
            "cap.nuclear": {"regimes": [{"name": "constant", "dist": {"kind": "const", "v": 5000.0}}]},
            "cap.coal": {"regimes": [{"name": "constant", "dist": {"kind": "const", "v": cap_coal}}]},
            "cap.gas": {"regimes": [{"name": "constant", "dist": {"kind": "const", "v": cap_gas}}]},
            "cap.wind": {"regimes": [{"name": "constant", "dist": {"kind": "const", "v": 4000.0}}]},
            "cap.solar": {"regimes": [{"name": "constant", "dist": {"kind": "const", "v": 3000.0}}]},
            "avail.nuclear": {"regimes": [{"name": "baseline", "dist": {"kind": "beta", "alpha": 30, "beta": 2, "low": 0.9, "high": 0.98}}]},
            "avail.coal": {"regimes": [{"name": "baseline", "dist": {"kind": "beta", "alpha": 25, "beta": 3, "low": 0.85, "high": 0.95}}]},
            "avail.gas": {"regimes": [{"name": "baseline", "dist": {"kind": "beta", "alpha": 28, "beta": 2, "low": 0.9, "high": 0.98}}]},
            "eta_lb.coal": {"regimes": [{"name": "baseline", "dist": {"kind": "const", "v": 0.33}}]},
            "eta_ub.coal": {"regimes": [{"name": "baseline", "dist": {"kind": "const", "v": 0.38}}]},
            "eta_lb.gas": {"regimes": [{"name": "baseline", "dist": {"kind": "const", "v": 0.48}}]},
            "eta_ub.gas": {"regimes": [{"name": "baseline", "dist": {"kind": "const", "v": 0.55}}]},
            "bid.nuclear.min": {"regimes": [{"name": "baseline", "dist": {"kind": "const", "v": -200.0}}]},
            "bid.nuclear.max": {"regimes": [{"name": "baseline", "dist": {"kind": "const", "v": -50.0}}]},
            "bid.wind.min": {"regimes": [{"name": "baseline", "dist": {"kind": "const", "v": -200.0}}]},
            "bid.wind.max": {"regimes": [{"name": "baseline", "dist": {"kind": "const", "v": -50.0}}]},
            "bid.solar.min": {"regimes": [{"name": "baseline", "dist": {"kind": "const", "v": -200.0}}]},
            "bid.solar.max": {"regimes": [{"name": "baseline", "dist": {"kind": "const", "v": -50.0}}]},
        },
        empirical_series={},
        planned_outages={"enabled": False},
        renewable_availability_mode="weather_simulation",
        io=IOConfig(
            out_dir=str(temp_dir),
            dataset_name="test_scenario",
            add_timestamp=False,
            save_pickle=False,
            save_csv=False,
            save_meta=False,
        ),
    )
    

@pytest.mark.integration
class TestPriceResponsiveness:
    """Tests for market price responsiveness to input changes"""

    def test_prices_respond_to_fuel_price_changes(self, temp_output_dir):
        """Test that market prices change when fuel prices change"""
        # Low fuel prices scenario
        config_low = _create_test_config(
            fuel_gas_price=25.0,
            fuel_coal_price=20.0,
            cap_gas=8000.0,
            cap_coal=6000.0,
            temp_dir=temp_output_dir,
            days=2,
        )

        schedules_low = build_schedules(
            start_ts=config_low.start_ts,
            days=config_low.days,
            freq=config_low.freq,
            seed=config_low.seed,
            supply_regime_planner=(
                config_low.supply_regime_planner.model_dump()
                if hasattr(config_low.supply_regime_planner, "model_dump")
                else config_low.supply_regime_planner
            ),
            variables={
                k: (v.model_dump() if hasattr(v, "model_dump") else v)
                for k, v in config_low.variables.items()
            },
            series_map={},
        )

        hours = config_low.days * 24
        df_low = simulate_timeseries(
            start_ts=config_low.start_ts,
            hours=hours,
            demand_cfg=(
                config_low.demand.model_dump()
                if hasattr(config_low.demand, "model_dump")
                else config_low.demand
            ),
            schedules=schedules_low,
            price_grid=np.array(config_low.price_grid),
            seed=config_low.seed,
            config=config_low,
            planned_outages_cfg=(
                config_low.planned_outages.model_dump()
                if hasattr(config_low.planned_outages, "model_dump")
                else config_low.planned_outages
            ),
        )

        # High fuel prices scenario
        config_high = _create_test_config(
            fuel_gas_price=50.0,
            fuel_coal_price=40.0,
            cap_gas=8000.0,
            cap_coal=6000.0,
            temp_dir=temp_output_dir,
            days=2,
        )

        schedules_high = build_schedules(
            start_ts=config_high.start_ts,
            days=config_high.days,
            freq=config_high.freq,
            seed=config_high.seed + 1,
            supply_regime_planner=(
                config_high.supply_regime_planner.model_dump()
                if hasattr(config_high.supply_regime_planner, "model_dump")
                else config_high.supply_regime_planner
            ),
            variables={
                k: (v.model_dump() if hasattr(v, "model_dump") else v)
                for k, v in config_high.variables.items()
            },
            series_map={},
        )

        df_high = simulate_timeseries(
            start_ts=config_high.start_ts,
            hours=hours,
            demand_cfg=(
                config_high.demand.model_dump()
                if hasattr(config_high.demand, "model_dump")
                else config_high.demand
            ),
            schedules=schedules_high,
            price_grid=np.array(config_high.price_grid),
            seed=config_high.seed + 1,
            config=config_high,
            planned_outages_cfg=(
                config_high.planned_outages.model_dump()
                if hasattr(config_high.planned_outages, "model_dump")
                else config_high.planned_outages
            ),
        )

        # Compare prices
        mean_price_low = df_low["price"].mean()
        mean_price_high = df_high["price"].mean()

        print(f"\nLow fuel prices: mean market price = {mean_price_low:.2f}")
        print(f"High fuel prices: mean market price = {mean_price_high:.2f}")

        # Check that thermal plants are actually running
        thermal_low = df_low["Q_coal"].mean() + df_low["Q_gas"].mean()
        thermal_high = df_high["Q_coal"].mean() + df_high["Q_gas"].mean()

        print(f"Low fuel: thermal generation = {thermal_low:.0f} MW")
        print(f"High fuel: thermal generation = {thermal_high:.0f} MW")

        # If thermal is running, prices MUST be different
        if thermal_low > 1000 or thermal_high > 1000:
            assert (
                mean_price_high > mean_price_low
            ), f"Prices should increase with fuel prices when thermal is marginal, but got low={mean_price_low}, high={mean_price_high}"

    def test_prices_respond_to_capacity_changes(self, temp_output_dir):
        """Test that market prices respond to changes in capacity"""
        # High capacity scenario
        config_high_cap = _create_test_config(
            fuel_gas_price=30.0,
            fuel_coal_price=25.0,
            cap_gas=15000.0,
            cap_coal=10000.0,
            temp_dir=temp_output_dir,
            days=2,
        )

        schedules_high_cap = build_schedules(
            start_ts=config_high_cap.start_ts,
            days=config_high_cap.days,
            freq=config_high_cap.freq,
            seed=config_high_cap.seed,
            supply_regime_planner=(
                config_high_cap.supply_regime_planner.model_dump()
                if hasattr(config_high_cap.supply_regime_planner, "model_dump")
                else config_high_cap.supply_regime_planner
            ),
            variables={
                k: (v.model_dump() if hasattr(v, "model_dump") else v)
                for k, v in config_high_cap.variables.items()
            },
            series_map={},
        )

        hours = config_high_cap.days * 24
        df_high_cap = simulate_timeseries(
            start_ts=config_high_cap.start_ts,
            hours=hours,
            demand_cfg=(
                config_high_cap.demand.model_dump()
                if hasattr(config_high_cap.demand, "model_dump")
                else config_high_cap.demand
            ),
            schedules=schedules_high_cap,
            price_grid=np.array(config_high_cap.price_grid),
            seed=config_high_cap.seed,
            config=config_high_cap,
            planned_outages_cfg=(
                config_high_cap.planned_outages.model_dump()
                if hasattr(config_high_cap.planned_outages, "model_dump")
                else config_high_cap.planned_outages
            ),
        )

        # Low capacity scenario
        config_low_cap = _create_test_config(
            fuel_gas_price=30.0,
            fuel_coal_price=25.0,
            cap_gas=5000.0,
            cap_coal=4000.0,
            temp_dir=temp_output_dir,
            days=2,
        )

        schedules_low_cap = build_schedules(
            start_ts=config_low_cap.start_ts,
            days=config_low_cap.days,
            freq=config_low_cap.freq,
            seed=config_low_cap.seed + 1,
            supply_regime_planner=(
                config_low_cap.supply_regime_planner.model_dump()
                if hasattr(config_low_cap.supply_regime_planner, "model_dump")
                else config_low_cap.supply_regime_planner
            ),
            variables={
                k: (v.model_dump() if hasattr(v, "model_dump") else v)
                for k, v in config_low_cap.variables.items()
            },
            series_map={},
        )

        df_low_cap = simulate_timeseries(
            start_ts=config_low_cap.start_ts,
            hours=hours,
            demand_cfg=(
                config_low_cap.demand.model_dump()
                if hasattr(config_low_cap.demand, "model_dump")
                else config_low_cap.demand
            ),
            schedules=schedules_low_cap,
            price_grid=np.array(config_low_cap.price_grid),
            seed=config_low_cap.seed + 1,
            config=config_low_cap,
            planned_outages_cfg=(
                config_low_cap.planned_outages.model_dump()
                if hasattr(config_low_cap.planned_outages, "model_dump")
                else config_low_cap.planned_outages
            ),
        )

        mean_price_high_cap = df_high_cap["price"].mean()
        mean_price_low_cap = df_low_cap["price"].mean()

        print(f"\nHigh capacity: mean price = {mean_price_high_cap:.2f}")
        print(f"Low capacity: mean price = {mean_price_low_cap:.2f}")

        # Check for NaN values
        if pd.isna(mean_price_low_cap) or pd.isna(mean_price_high_cap):
            pytest.fail(f"NaN prices found: high_cap={mean_price_high_cap}, low_cap={mean_price_low_cap}")

        # Lower capacity should lead to higher prices (scarcity)
        assert (
            mean_price_low_cap >= mean_price_high_cap
        ), f"Lower capacity should lead to higher prices, but got high_cap={mean_price_high_cap}, low_cap={mean_price_low_cap}"

    def test_solar_availability_varies_by_hour(self, minimal_config):
        """Test that solar availability varies correctly by hour of day"""
        minimal_config.days = 1

        schedules = build_schedules(
            start_ts=minimal_config.start_ts,
            days=minimal_config.days,
            freq=minimal_config.freq,
            seed=minimal_config.seed,
            supply_regime_planner=(
                minimal_config.supply_regime_planner.model_dump()
                if hasattr(minimal_config.supply_regime_planner, "model_dump")
                else minimal_config.supply_regime_planner.dict()
            ),
            variables={
                k: (v.model_dump() if hasattr(v, "model_dump") else v.dict())
                for k, v in minimal_config.variables.items()
            },
            series_map={},
        )

        hours = 24
        df = simulate_timeseries(
            start_ts=minimal_config.start_ts,
            hours=hours,
            demand_cfg=(
                minimal_config.demand.model_dump()
                if hasattr(minimal_config.demand, "model_dump")
                else minimal_config.demand.dict()
            ),
            schedules=schedules,
            price_grid=np.array(minimal_config.price_grid),
            seed=minimal_config.seed,
            config=minimal_config,
            planned_outages_cfg=(
                minimal_config.planned_outages.model_dump()
                if hasattr(minimal_config.planned_outages, "model_dump")
                else minimal_config.planned_outages.dict()
            ),
        )

        # Extract hour from timestamp
        df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour

        # Check solar output at different times
        night_solar = df[df["hour"].isin([0, 1, 2, 3, 4, 5, 21, 22, 23])][
            "Q_solar"
        ].max()
        day_solar = df[df["hour"].isin([10, 11, 12, 13, 14, 15])]["Q_solar"].max()

        print(f"\nNight solar: {night_solar:.2f} MW")
        print(f"Day solar: {day_solar:.2f} MW")
        print(f"\navail.solar at different hours:")
        for hour in [0, 6, 12, 18, 23]:
            row = df[df["hour"] == hour].iloc[0]
            print(
                f"  Hour {hour:02d}: avail.solar={row.get('avail.solar', 0):.3f}, Q_solar={row['Q_solar']:.1f} MW"
            )

        # Night should be zero
        assert night_solar == 0.0, f"Solar should be zero at night, got {night_solar}"

        # Day should be positive
        assert day_solar > 0.0, f"Solar should be positive during day, got {day_solar}"

        # Check avail.solar column exists and varies
        if "avail.solar" in df.columns:
            night_avail = df[df["hour"].isin([0, 1, 2, 3])]["avail.solar"].max()
            day_avail = df[df["hour"].isin([12, 13, 14])]["avail.solar"].max()

            assert (
                night_avail == 0.0
            ), f"avail.solar should be 0 at night, got {night_avail}"
            assert (
                day_avail > 0.0
            ), f"avail.solar should be >0 during day, got {day_avail}"
