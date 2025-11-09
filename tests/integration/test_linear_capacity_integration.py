"""
Integration diagnostic tests for full scenario with linear distributions.
"""

import numpy as np
import pandas as pd
import pytest

from synthetic_data_pkg.config import DemandConfig
from synthetic_data_pkg.scenario import build_schedules
from synthetic_data_pkg.simulate import simulate_timeseries


@pytest.mark.integration
class TestLinearCapacityScenario:
    """Integration tests for scenarios with linear capacity changes"""

    def test_coal_phaseout_full_integration(self, temp_output_dir):
        """Test complete coal phaseout scenario with linear capacity changes"""
        from synthetic_data_pkg.config import IOConfig, TopConfig

        # Simplified coal phaseout config
        config = TopConfig(
            start_ts="2024-01-01 00:00",
            days=365,  # 1 year for faster testing
            freq="h",
            seed=42,
            price_grid=list(range(-100, 201, 10)),
            demand=DemandConfig(
                inelastic=False,
                base_intercept=25000.0,
                slope=-200.0,
                daily_seasonality=False,
                annual_seasonality=False,
            ),
            supply_regime_planner={"mode": "local_only"},
            variables={
                "fuel.gas": {
                    "regimes": [
                        {"name": "stable", "dist": {"kind": "const", "v": 30.0}}
                    ]
                },
                "fuel.coal": {
                    "regimes": [
                        {"name": "stable", "dist": {"kind": "const", "v": 25.0}}
                    ]
                },
                # LINEAR CAPACITY CHANGES
                "cap.coal": {
                    "regimes": [
                        {
                            "name": "declining",
                            "dist": {
                                "kind": "linear",
                                "start": 8000.0,
                                "slope": -0.913,
                            },
                        }
                    ]
                },  # 8000 -> 0 in 1 year
                "cap.gas": {
                    "regimes": [
                        {
                            "name": "increasing",
                            "dist": {
                                "kind": "linear",
                                "start": 12000.0,
                                "slope": 0.685,
                            },
                        }
                    ]
                },  # 12000 -> 18000 in 1 year
                "cap.wind": {
                    "regimes": [
                        {
                            "name": "building",
                            "dist": {"kind": "linear", "start": 7000.0, "slope": 0.571},
                        }
                    ]
                },  # 7000 -> 12000 in 1 year
                "cap.solar": {
                    "regimes": [
                        {
                            "name": "building",
                            "dist": {"kind": "linear", "start": 5000.0, "slope": 0.571},
                        }
                    ]
                },  # 5000 -> 10000 in 1 year
                # CONSTANT CAPACITIES
                "cap.nuclear": {
                    "regimes": [
                        {"name": "constant", "dist": {"kind": "const", "v": 6000.0}}
                    ]
                },
                # AVAILABILITIES
                "avail.nuclear": {
                    "regimes": [
                        {
                            "name": "baseline",
                            "dist": {
                                "kind": "beta",
                                "alpha": 30,
                                "beta": 2,
                                "low": 0.9,
                                "high": 0.98,
                            },
                        }
                    ]
                },
                "avail.coal": {
                    "regimes": [
                        {
                            "name": "baseline",
                            "dist": {
                                "kind": "beta",
                                "alpha": 25,
                                "beta": 3,
                                "low": 0.85,
                                "high": 0.95,
                            },
                        }
                    ]
                },
                "avail.gas": {
                    "regimes": [
                        {
                            "name": "baseline",
                            "dist": {
                                "kind": "beta",
                                "alpha": 28,
                                "beta": 2,
                                "low": 0.9,
                                "high": 0.98,
                            },
                        }
                    ]
                },
                # EFFICIENCIES
                "eta_lb.coal": {
                    "regimes": [
                        {"name": "baseline", "dist": {"kind": "const", "v": 0.33}}
                    ]
                },
                "eta_ub.coal": {
                    "regimes": [
                        {"name": "baseline", "dist": {"kind": "const", "v": 0.38}}
                    ]
                },
                "eta_lb.gas": {
                    "regimes": [
                        {"name": "baseline", "dist": {"kind": "const", "v": 0.48}}
                    ]
                },
                "eta_ub.gas": {
                    "regimes": [
                        {"name": "baseline", "dist": {"kind": "const", "v": 0.55}}
                    ]
                },
                # BIDS
                "bid.nuclear.min": {
                    "regimes": [
                        {"name": "baseline", "dist": {"kind": "const", "v": -200.0}}
                    ]
                },
                "bid.nuclear.max": {
                    "regimes": [
                        {"name": "baseline", "dist": {"kind": "const", "v": -50.0}}
                    ]
                },
                "bid.wind.min": {
                    "regimes": [
                        {"name": "baseline", "dist": {"kind": "const", "v": -200.0}}
                    ]
                },
                "bid.wind.max": {
                    "regimes": [
                        {"name": "baseline", "dist": {"kind": "const", "v": -50.0}}
                    ]
                },
                "bid.solar.min": {
                    "regimes": [
                        {"name": "baseline", "dist": {"kind": "const", "v": -200.0}}
                    ]
                },
                "bid.solar.max": {
                    "regimes": [
                        {"name": "baseline", "dist": {"kind": "const", "v": -50.0}}
                    ]
                },
            },
            empirical_series={},
            planned_outages={"enabled": False},
            renewable_availability_mode="weather_simulation",
            io=IOConfig(
                out_dir=str(temp_output_dir),
                dataset_name="test_coal_phaseout",
                add_timestamp=False,
                save_pickle=False,
                save_csv=False,
                save_meta=False,
            ),
        )

        # Build schedules
        schedules = build_schedules(
            start_ts=config.start_ts,
            days=config.days,
            freq=config.freq,
            seed=config.seed,
            supply_regime_planner=(
                config.supply_regime_planner.model_dump()
                if hasattr(config.supply_regime_planner, "model_dump")
                else config.supply_regime_planner
            ),
            variables={
                k: (v.model_dump() if hasattr(v, "model_dump") else v)
                for k, v in config.variables.items()
            },
            series_map={},
        )

        # Run simulation
        hours = config.days * 24
        df = simulate_timeseries(
            start_ts=config.start_ts,
            hours=hours,
            demand_cfg=(
                config.demand.model_dump()
                if hasattr(config.demand, "model_dump")
                else config.demand
            ),
            schedules=schedules,
            price_grid=np.array(config.price_grid),
            seed=config.seed,
            config=config,
            planned_outages_cfg=(
                config.planned_outages.model_dump()
                if hasattr(config.planned_outages, "model_dump")
                else config.planned_outages
            ),
        )

        # Test that capacities are in the dataframe
        assert "cap.coal" in df.columns, "cap.coal should be in dataframe"
        assert "cap.gas" in df.columns
        assert "cap.wind" in df.columns
        assert "cap.solar" in df.columns

        # Sample at different points in time
        sample_indices = [0, len(df) // 4, len(df) // 2, 3 * len(df) // 4, len(df) - 1]

        print("\n\nCapacity evolution over 1 year:")
        print(f"{'Hour':>6} {'Day':>4} {'Coal':>8} {'Gas':>8} {'Wind':>8} {'Solar':>8}")

        coal_values = []
        gas_values = []
        wind_values = []
        solar_values = []

        for idx in sample_indices:
            row = df.iloc[idx]
            hour = idx
            day = hour / 24

            coal_values.append(row["cap.coal"])
            gas_values.append(row["cap.gas"])
            wind_values.append(row["cap.wind"])
            solar_values.append(row["cap.solar"])

            print(
                f"{hour:6d} {day:4.0f} {row['cap.coal']:8.1f} {row['cap.gas']:8.1f} {row['cap.wind']:8.1f} {row['cap.solar']:8.1f}"
            )

        # TEST 1: Coal should decline
        print(f"\nCoal: {coal_values[0]:.1f} -> {coal_values[-1]:.1f}")
        assert (
            coal_values[0] > coal_values[-1]
        ), f"Coal should decline: {coal_values[0]} -> {coal_values[-1]}"
        assert coal_values[0] == pytest.approx(
            8000.0, abs=10.0
        ), "Coal should start at 8000"

        # TEST 2: Gas should increase
        print(f"Gas: {gas_values[0]:.1f} -> {gas_values[-1]:.1f}")
        assert (
            gas_values[-1] > gas_values[0]
        ), f"Gas should increase: {gas_values[0]} -> {gas_values[-1]}"
        assert gas_values[0] == pytest.approx(
            12000.0, abs=10.0
        ), "Gas should start at 12000"

        # TEST 3: Wind should increase
        print(f"Wind: {wind_values[0]:.1f} -> {wind_values[-1]:.1f}")
        assert (
            wind_values[-1] > wind_values[0]
        ), f"Wind should increase: {wind_values[0]} -> {wind_values[-1]}"
        assert wind_values[0] == pytest.approx(
            7000.0, abs=10.0
        ), "Wind should start at 7000"

        # TEST 4: Solar should increase
        print(f"Solar: {solar_values[0]:.1f} -> {solar_values[-1]:.1f}")
        assert (
            solar_values[-1] > solar_values[0]
        ), f"Solar should increase: {solar_values[0]} -> {solar_values[-1]}"
        assert solar_values[0] == pytest.approx(
            5000.0, abs=10.0
        ), "Solar should start at 5000"

        # TEST 5: Check monotonicity over ALL hours
        print("\nChecking monotonicity over all hours...")

        # Sample every 24 hours
        sample_freq = 24
        for i in range(0, len(df) - sample_freq, sample_freq):
            curr = df.iloc[i]
            next_day = df.iloc[i + sample_freq]

            assert (
                curr["cap.coal"] >= next_day["cap.coal"]
            ), f"Hour {i}: Coal should be declining"
            assert (
                curr["cap.gas"] <= next_day["cap.gas"]
            ), f"Hour {i}: Gas should be increasing"
            assert (
                curr["cap.wind"] <= next_day["cap.wind"]
            ), f"Hour {i}: Wind should be increasing"
            assert (
                curr["cap.solar"] <= next_day["cap.solar"]
            ), f"Hour {i}: Solar should be increasing"

        print("✓ All capacity trends correct!")

        # TEST 6: Check that the changes are LINEAR (not exponential or step-wise)
        print("\nChecking linearity...")

        # Check coal decline is linear
        coal_series = df["cap.coal"].values[::24]  # Daily samples
        coal_diffs = np.diff(coal_series)
        coal_diff_std = np.std(coal_diffs)
        print(
            f"Coal daily change std dev: {coal_diff_std:.2f} (should be small for linear)"
        )
        assert (
            coal_diff_std < 5.0
        ), "Coal decline should be linear (low std dev in daily changes)"

        # Check gas increase is linear
        gas_series = df["cap.gas"].values[::24]
        gas_diffs = np.diff(gas_series)
        gas_diff_std = np.std(gas_diffs)
        print(
            f"Gas daily change std dev: {gas_diff_std:.2f} (should be small for linear)"
        )
        assert gas_diff_std < 5.0, "Gas increase should be linear"

        print("✓ Capacity changes are linear!")

    def test_schedules_directly(self):
        """Test that build_schedules creates correct RegimeSchedule objects"""
        from synthetic_data_pkg.scenario import build_schedules

        schedules = build_schedules(
            start_ts="2024-01-01 00:00",
            days=10,
            freq="h",
            seed=42,
            supply_regime_planner={"mode": "local_only"},
            variables={
                "cap.coal": {
                    "regimes": [
                        {
                            "name": "declining",
                            "dist": {"kind": "linear", "start": 8000.0, "slope": -10.0},
                        }
                    ]
                },
                "cap.gas": {
                    "regimes": [
                        {"name": "constant", "dist": {"kind": "const", "v": 12000.0}}
                    ]
                },
            },
            series_map={},
        )

        # Test coal schedule
        coal_schedule = schedules["cap.coal"]

        print("\n\nDirect schedule test:")
        print("Coal capacity (should decline by 10 MW/hour):")

        values = []
        for hour in [0, 1, 2, 10, 50, 100]:
            ts = pd.Timestamp("2024-01-01") + pd.Timedelta(hours=hour)
            val, regime = coal_schedule.value_at(ts)
            expected = 8000.0 - 10.0 * hour
            values.append(val)
            print(f"  Hour {hour:3d}: {val:8.1f} MW (expected {expected:8.1f})")

        # Check declining
        for i in range(1, len(values)):
            assert (
                values[i] < values[i - 1]
            ), f"Coal should decline: {values[i]} >= {values[i-1]}"

        # Test gas schedule (should be constant)
        gas_schedule = schedules["cap.gas"]

        print("\nGas capacity (should be constant at 12000):")
        for hour in [0, 1, 2, 10, 50, 100]:
            ts = pd.Timestamp("2024-01-01") + pd.Timedelta(hours=hour)
            val, regime = gas_schedule.value_at(ts)
            print(f"  Hour {hour:3d}: {val:8.1f} MW")
            assert val == 12000.0, f"Gas should be constant at 12000, got {val}"
