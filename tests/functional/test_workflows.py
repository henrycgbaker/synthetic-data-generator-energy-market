"""
Functional tests for complete user workflows.
Tests end-to-end scenarios from config to output.
"""

import os
import subprocess
import tempfile

import pandas as pd
import pytest
import yaml


@pytest.mark.functional
@pytest.mark.smoke
@pytest.mark.slow
class TestRunnerScripts:
    """Test that runner scripts execute successfully"""

    @pytest.mark.skip(reason="Slow integration test - run explicitly with 'pytest -m slow'")
    def test_scenario1_runner_executes(self):
        """Test that run_scenario1_gas_crisis.py executes successfully"""
        result = subprocess.run(
            ["python", "run_scenario1_gas_crisis.py"],
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes
            timeout=300,  # 5 minutes
        )

        # Check it ran without error
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        # Check expected output messages
        assert "SCENARIO 1" in result.stdout
        assert "COMPLETE" in result.stdout

    @pytest.mark.skip(reason="Slow integration test - run explicitly with 'pytest -m slow'")
    def test_scenario2_runner_executes(self):
        """Test that run_scenario2_coal_phaseout.py executes successfully"""
        result = subprocess.run(
            ["python", "run_scenario2_coal_phaseout.py"],
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes
            timeout=300,  # 5 minutes
        )

        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert "SCENARIO 2" in result.stdout
        assert "COMPLETE" in result.stdout

    @pytest.mark.skip(reason="Slow integration test - run explicitly with 'pytest -m slow'")
    def test_scenario3_runner_executes(self):
        """Test that run_scenario3_full_seasonality.py executes successfully"""
        result = subprocess.run(
            ["python", "run_scenario3_full_seasonality.py"],
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes
            timeout=300,  # 5 minutes
        )

        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert "SCENARIO 3" in result.stdout
        assert "COMPLETE" in result.stdout


@pytest.mark.functional
class TestCLIInterface:
    """Test CLI interface"""

    @pytest.mark.skip(reason="CLI entrypoint installation issues")
    def test_cli_executes_with_config(self):
        """Test that CLI can execute with a config file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal config
            config = {
                "start_ts": "2024-01-01 00:00",
                "days": 2,
                "freq": "h",
                "seed": 42,
                "price_grid": list(range(-100, 201, 10)),
                "demand": {
                    "inelastic": False,
                    "base_intercept": 1000.0,
                    "slope": -200.0,
                    "daily_seasonality": False,
                    "annual_seasonality": False,
                },
                "supply_regime_planner": {"mode": "local_only"},
                "variables": {
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
                    "cap.nuclear": {
                        "regimes": [
                            {"name": "constant", "dist": {"kind": "const", "v": 5000.0}}
                        ]
                    },
                    "cap.coal": {
                        "regimes": [
                            {"name": "constant", "dist": {"kind": "const", "v": 6000.0}}
                        ]
                    },
                    "cap.gas": {
                        "regimes": [
                            {"name": "constant", "dist": {"kind": "const", "v": 8000.0}}
                        ]
                    },
                    "cap.wind": {
                        "regimes": [
                            {"name": "constant", "dist": {"kind": "const", "v": 4000.0}}
                        ]
                    },
                    "cap.solar": {
                        "regimes": [
                            {"name": "constant", "dist": {"kind": "const", "v": 3000.0}}
                        ]
                    },
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
                "empirical_series": {},
                "planned_outages": {"enabled": False},
                "renewable_availability_mode": "weather_simulation",
                "weather_simulation": {},
                "io": {
                    "out_dir": tmpdir,
                    "dataset_name": "test_cli",
                    "add_timestamp": False,
                    "save_csv": True,
                },
            }

            config_path = os.path.join(tmpdir, "test_config.yaml")
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            # Run CLI
            result = subprocess.run(
                ["synthetic-data", "run", config_path],
                capture_output=True,
                text=True,
                timeout=30,
            )

            assert result.returncode == 0, f"CLI failed: {result.stderr}"

            # Check output file was created
            csv_path = os.path.join(tmpdir, "test_cli_v0.csv")
            assert os.path.exists(csv_path)


@pytest.mark.functional
@pytest.mark.slow
class TestEndToEndWorkflows:
    """Test complete end-to-end workflows"""

    def test_full_year_simulation(self):
        """Test simulating a full year of data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "start_ts": "2024-01-01 00:00",
                "days": 365,
                "freq": "h",
                "seed": 42,
                "price_grid": list(range(-100, 201, 10)),
                "demand": {
                    "inelastic": False,
                    "base_intercept": 1500.0,
                    "slope": -300.0,
                    "daily_seasonality": True,
                    "annual_seasonality": True,
                    "winter_amp": 0.2,
                    "summer_amp": -0.15,
                },
                "supply_regime_planner": {"mode": "local_only"},
                "variables": {
                    "fuel.gas": {
                        "regimes": [
                            {
                                "name": "stable",
                                "dist": {
                                    "kind": "normal",
                                    "mu": 35.0,
                                    "sigma": 5.0,
                                    "bounds": {"low": 15.0, "high": 80.0},
                                },
                            }
                        ]
                    },
                    "fuel.coal": {
                        "regimes": [
                            {
                                "name": "stable",
                                "dist": {
                                    "kind": "normal",
                                    "mu": 28.0,
                                    "sigma": 4.0,
                                    "bounds": {"low": 15.0, "high": 60.0},
                                },
                            }
                        ]
                    },
                    "cap.nuclear": {
                        "regimes": [
                            {"name": "constant", "dist": {"kind": "const", "v": 6000.0}}
                        ]
                    },
                    "cap.coal": {
                        "regimes": [
                            {"name": "constant", "dist": {"kind": "const", "v": 8000.0}}
                        ]
                    },
                    "cap.gas": {
                        "regimes": [
                            {
                                "name": "constant",
                                "dist": {"kind": "const", "v": 12000.0},
                            }
                        ]
                    },
                    "cap.wind": {
                        "regimes": [
                            {"name": "constant", "dist": {"kind": "const", "v": 7000.0}}
                        ]
                    },
                    "cap.solar": {
                        "regimes": [
                            {"name": "constant", "dist": {"kind": "const", "v": 5000.0}}
                        ]
                    },
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
                "empirical_series": {},
                "planned_outages": {"enabled": True, "months": [5, 6, 7, 8, 9]},
                "renewable_availability_mode": "weather_simulation",
                "io": {
                    "out_dir": tmpdir,
                    "dataset_name": "full_year",
                    "add_timestamp": False,
                    "save_csv": True,
                    "save_pickle": True,
                    "save_meta": True,
                },
            }

            config_path = os.path.join(tmpdir, "full_year_config.yaml")
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            # Run simulation
            from synthetic_data_pkg.runner import execute_scenario

            paths = execute_scenario(config_path)

            assert paths is not None
            assert "csv" in paths
            assert "pickle" in paths
            assert "meta" in paths

            # Load and validate output
            df = pd.read_csv(paths["csv"])

            # Should have 365 * 24 = 8760 hours
            assert len(df) == 8760

            # Check seasonal patterns exist
            df["month"] = pd.to_datetime(df["timestamp"]).dt.month

            # Winter months (Dec, Jan, Feb) should have higher average demand
            winter_demand = df[df["month"].isin([12, 1, 2])]["q_cleared"].mean()
            summer_demand = df[df["month"].isin([6, 7, 8])]["q_cleared"].mean()

            # Winter should be higher (accounting for seasonality)
            assert winter_demand > summer_demand * 0.9  # Allow some tolerance

    def test_scenario_with_regime_changes(self):
        """Test scenario with multiple regime changes"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "start_ts": "2024-01-01 00:00",
                "days": 90,
                "freq": "h",
                "seed": 42,
                "price_grid": list(range(-100, 201, 10)),
                "demand": {
                    "inelastic": False,
                    "base_intercept": 200.0, 
                    "slope": -0.006,  # dP/dQ
                    "daily_seasonality": False,
                    "annual_seasonality": False,
                },
                "supply_regime_planner": {"mode": "local_only"},
                "variables": {
                    # Gas price with regime change
                    "fuel.gas": {
                        "regimes": [
                            {
                                "name": "low",
                                "dist": {"kind": "const", "v": 25.0},
                                "breakpoints": [
                                    {"date": "2024-01-01", "transition_hours": 24}
                                ],
                            },
                            {
                                "name": "high",
                                "dist": {"kind": "const", "v": 75.0},
                                "breakpoints": [
                                    {"date": "2024-02-01", "transition_hours": 168}
                                ],
                            },
                        ]
                    },
                    "fuel.coal": {
                        "regimes": [
                            {"name": "stable", "dist": {"kind": "const", "v": 25.0}}
                        ]
                    },
                    "cap.nuclear": {
                        "regimes": [
                            {"name": "constant", "dist": {"kind": "const", "v": 6000.0}}
                        ]
                    },
                    "cap.coal": {
                        "regimes": [
                            {"name": "constant", "dist": {"kind": "const", "v": 8000.0}}
                        ]
                    },
                    "cap.gas": {
                        "regimes": [
                            {
                                "name": "constant",
                                "dist": {"kind": "const", "v": 12000.0},
                            }
                        ]
                    },
                    "cap.wind": {
                        "regimes": [
                            {"name": "constant", "dist": {"kind": "const", "v": 7000.0}}
                        ]
                    },
                    "cap.solar": {
                        "regimes": [
                            {"name": "constant", "dist": {"kind": "const", "v": 5000.0}}
                        ]
                    },
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
                "empirical_series": {},
                "planned_outages": {"enabled": False},
                "renewable_availability_mode": "weather_simulation",
                "io": {
                    "out_dir": tmpdir,
                    "dataset_name": "regime_change",
                    "add_timestamp": False,
                    "save_csv": True,
                },
            }

            config_path = os.path.join(tmpdir, "regime_change_config.yaml")
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            # Run simulation
            from synthetic_data_pkg.runner import execute_scenario

            paths = execute_scenario(config_path)

            assert paths is not None
            df = pd.read_csv(paths["csv"])

            # Check that regime change affects prices
            df["date"] = pd.to_datetime(df["timestamp"]).dt.date

            jan_prices = df[df["date"] < pd.to_datetime("2024-02-01").date()][
                "price"
            ].mean()
            feb_prices = df[df["date"] >= pd.to_datetime("2024-02-01").date()][
                "price"
            ].mean()

            # February prices should be higher (gas price increased)
            assert feb_prices > jan_prices
