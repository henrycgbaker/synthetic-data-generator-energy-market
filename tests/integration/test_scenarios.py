"""
Integration tests for scenario execution.
Tests multiple components working together to generate synthetic data.
"""

import os

import numpy as np
import pandas as pd
import pytest
import yaml

from synthetic_data_pkg.runner import execute_scenario
from synthetic_data_pkg.scenario import build_schedules
from synthetic_data_pkg.simulate import simulate_timeseries


@pytest.mark.integration
class TestScenarioExecution:
    """Integration tests for full scenario execution"""

    def test_minimal_scenario_runs_successfully(self, minimal_config, temp_output_dir):
        """Test that a minimal configuration runs without errors"""
        # Write config to temp file
        config_path = temp_output_dir / "test_config.yaml"

        # Convert config to dict
        config_dict = (
            minimal_config.model_dump()
            if hasattr(minimal_config, "model_dump")
            else minimal_config.dict()
        )

        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        # Execute via runner function
        paths = execute_scenario(str(config_path))

        # Check output was created
        assert paths is not None
        assert len(paths) > 0

    def test_scenario_produces_correct_length(self, minimal_config, temp_output_dir):
        """Test that scenario produces expected number of timesteps"""
        # Build schedules and simulate
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

        hours = minimal_config.days * 24
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

        # 7 days * 24 hours = 168 timesteps
        expected_length = 7 * 24
        assert len(df) == expected_length

    def test_scenario_timestamps_are_sequential(self, minimal_config, temp_output_dir):
        """Test that timestamps are properly sequential"""
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

        hours = minimal_config.days * 24
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

        timestamps = pd.to_datetime(df["timestamp"])

        # Check all timestamps are unique
        assert len(timestamps) == len(timestamps.unique())

        # Check timestamps are sorted
        assert timestamps.is_monotonic_increasing

        # Check hourly frequency
        diffs = timestamps.diff().dropna()
        assert all(diffs == pd.Timedelta(hours=1))

    def test_scenario_prices_are_reasonable(self, minimal_config):
        """Test that generated prices are within reasonable bounds"""
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

        hours = minimal_config.days * 24
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

        prices = df["price"]

        # Prices should be numeric
        assert pd.api.types.is_numeric_dtype(prices)

        # Prices should be within price grid range
        assert prices.min() >= min(minimal_config.price_grid)
        assert prices.max() <= max(minimal_config.price_grid)

    def test_scenario_with_inelastic_demand(self, minimal_config):
        """Test scenario with inelastic demand"""
        minimal_config.demand.inelastic = True
        minimal_config.demand.base_intercept = 15000.0  # Fixed demand

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

        hours = minimal_config.days * 24
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

        assert len(df) > 0

    def test_scenario_with_seasonality_disabled(self, minimal_config):
        """Test scenario with all seasonality disabled"""
        minimal_config.demand.daily_seasonality = False
        minimal_config.demand.annual_seasonality = False

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

        hours = minimal_config.days * 24
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

        assert len(df) > 0


@pytest.mark.integration
class TestScenarioIO:
    """Integration tests for scenario input/output"""

    def test_scenario_saves_csv(self, minimal_config, temp_output_dir):
        """Test that scenario saves CSV output correctly"""
        minimal_config.io.save_csv = True
        minimal_config.io.save_pickle = False
        minimal_config.io.out_dir = str(temp_output_dir)

        config_path = temp_output_dir / "test_config.yaml"
        config_dict = (
            minimal_config.model_dump()
            if hasattr(minimal_config, "model_dump")
            else minimal_config.dict()
        )

        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        paths = execute_scenario(str(config_path))

        # Check CSV file exists
        assert "csv" in paths
        assert os.path.exists(paths["csv"])

        # Check we can read it back
        df_loaded = pd.read_csv(paths["csv"])
        assert len(df_loaded) > 0

    def test_scenario_saves_pickle(self, minimal_config, temp_output_dir):
        """Test that scenario saves pickle output correctly"""
        minimal_config.io.save_csv = False
        minimal_config.io.save_pickle = True
        minimal_config.io.out_dir = str(temp_output_dir)

        config_path = temp_output_dir / "test_config.yaml"
        config_dict = (
            minimal_config.model_dump()
            if hasattr(minimal_config, "model_dump")
            else minimal_config.dict()
        )

        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        paths = execute_scenario(str(config_path))

        # Check pickle file exists
        assert "pickle" in paths
        assert os.path.exists(paths["pickle"])

        # Check we can read it back
        df_loaded = pd.read_pickle(paths["pickle"])
        assert len(df_loaded) > 0

    def test_scenario_saves_meta(self, minimal_config, temp_output_dir):
        """Test that scenario saves metadata correctly"""
        minimal_config.io.save_meta = True
        minimal_config.io.save_csv = True
        minimal_config.io.out_dir = str(temp_output_dir)

        config_path = temp_output_dir / "test_config.yaml"
        config_dict = (
            minimal_config.model_dump()
            if hasattr(minimal_config, "model_dump")
            else minimal_config.dict()
        )

        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        paths = execute_scenario(str(config_path))

        # Check meta file exists
        assert "meta" in paths
        assert os.path.exists(paths["meta"])

        # Check meta contains expected keys
        import json

        with open(paths["meta"]) as f:
            meta = json.load(f)

        assert "config" in meta


@pytest.mark.integration
@pytest.mark.slow
class TestScenarioConfigLoading:
    """Integration tests for loading and running from config files"""

    def test_load_and_run_gas_crisis_scenario(self, temp_output_dir):
        """Test loading and running the gas crisis scenario"""
        config_path = "configs/1_gas_crisis.yaml"

        if not os.path.exists(config_path):
            pytest.skip("Gas crisis config not found")

        # Modify config to use temp directory and shorter run
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        config_dict["io"]["out_dir"] = str(temp_output_dir)
        config_dict["io"]["add_timestamp"] = False
        config_dict["days"] = 7  # Shorter for testing

        # Write modified config
        test_config_path = temp_output_dir / "test_config.yaml"
        with open(test_config_path, "w") as f:
            yaml.dump(config_dict, f)

        # Execute
        paths = execute_scenario(str(test_config_path))

        assert paths is not None
        assert len(paths) > 0

    def test_all_scenario_configs_are_valid(self, all_scenario_configs):
        """Test that all scenario configs load without errors"""
        from synthetic_data_pkg.config import TopConfig

        for config_path in all_scenario_configs:
            if not os.path.exists(config_path):
                continue

            with open(config_path) as f:
                config_dict = yaml.safe_load(f)

            # Try to create TopConfig (this validates the schema)
            try:
                config = TopConfig(**config_dict)
                assert config is not None
            except Exception as e:
                pytest.fail(f"Config {config_path} failed to load: {e}")


@pytest.mark.integration
class TestDemandSupplyEquilibrium:
    """Integration tests for demand-supply equilibrium"""

    def test_equilibrium_price_in_grid_bounds(self, minimal_config):
        """Test that equilibrium price is within price grid bounds"""
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

        hours = minimal_config.days * 24
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

        prices = df["price"]
        price_min = min(minimal_config.price_grid)
        price_max = max(minimal_config.price_grid)

        # All prices should be within grid bounds (continuous equilibrium)
        for p in prices:
            assert (
                price_min <= p <= price_max
            ), f"Price {p} outside grid bounds [{price_min}, {price_max}]"

    def test_quantity_cleared_is_positive(self, minimal_config):
        """Test that cleared quantity is always positive"""
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

        hours = minimal_config.days * 24
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

        quantities = df["q_cleared"]
        # Check all quantities are non-negative (>= 0)
        assert (
            quantities >= 0
        ).all(), f"Found negative quantities: {quantities[quantities < 0]}"
