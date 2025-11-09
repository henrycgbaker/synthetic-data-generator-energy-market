"""
Shared pytest fixtures for all test modules.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from synthetic_data_pkg.config import (
    DemandConfig,
    IOConfig,
    PlannedOutageConfig,
    RegimePlanner,
    TopConfig,
    VariableRegimeSpec,
    WeatherSimulationConfig,
)


@pytest.fixture
def seed():
    """Fixed random seed for reproducibility"""
    return 42


@pytest.fixture
def rng(seed):
    """Numpy random generator with fixed seed"""
    return np.random.default_rng(seed)


@pytest.fixture
def temp_output_dir():
    """Temporary directory for test outputs"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def simple_demand_config():
    """Basic demand configuration for testing"""
    # Standard form: P = 200 - 0.006*Q
    return DemandConfig(
        inelastic=False,
        base_intercept=200.0,  # Choke price
        slope=-0.006,  # dP/dQ
        daily_seasonality=True,
        day_peak_hour=14,
        day_amp=0.2,
        weekend_drop=0.1,
        annual_seasonality=False,
    )


@pytest.fixture
def minimal_config(temp_output_dir):
    """Minimal valid configuration for testing"""
    return TopConfig(
        start_ts="2024-01-01 00:00",
        days=7,
        freq="h",
        seed=42,
        price_grid=list(range(-100, 201, 10)),
        demand=DemandConfig(
            inelastic=False,
            base_intercept=200.0,  # Standard form: P = 200 - 0.006*Q
            slope=-0.006,  # dP/dQ
            daily_seasonality=True,
            annual_seasonality=False,
        ),
        supply_regime_planner=RegimePlanner(mode="local_only"),
        variables={
            "fuel.gas": VariableRegimeSpec(
                regimes=[
                    {
                        "name": "stable",
                        "dist": {"kind": "const", "v": 30.0},
                    }
                ]
            ),
            "fuel.coal": VariableRegimeSpec(
                regimes=[
                    {
                        "name": "stable",
                        "dist": {"kind": "const", "v": 25.0},
                    }
                ]
            ),
            "cap.nuclear": VariableRegimeSpec(
                regimes=[{"name": "constant", "dist": {"kind": "const", "v": 5000.0}}]
            ),
            "cap.coal": VariableRegimeSpec(
                regimes=[{"name": "constant", "dist": {"kind": "const", "v": 6000.0}}]
            ),
            "cap.gas": VariableRegimeSpec(
                regimes=[{"name": "constant", "dist": {"kind": "const", "v": 8000.0}}]
            ),
            "cap.wind": VariableRegimeSpec(
                regimes=[{"name": "constant", "dist": {"kind": "const", "v": 4000.0}}]
            ),
            "cap.solar": VariableRegimeSpec(
                regimes=[{"name": "constant", "dist": {"kind": "const", "v": 3000.0}}]
            ),
            "avail.nuclear": VariableRegimeSpec(
                regimes=[
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
            ),
            "avail.coal": VariableRegimeSpec(
                regimes=[
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
            ),
            "avail.gas": VariableRegimeSpec(
                regimes=[
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
            ),
            "eta_lb.coal": VariableRegimeSpec(
                regimes=[{"name": "baseline", "dist": {"kind": "const", "v": 0.33}}]
            ),
            "eta_ub.coal": VariableRegimeSpec(
                regimes=[{"name": "baseline", "dist": {"kind": "const", "v": 0.38}}]
            ),
            "eta_lb.gas": VariableRegimeSpec(
                regimes=[{"name": "baseline", "dist": {"kind": "const", "v": 0.48}}]
            ),
            "eta_ub.gas": VariableRegimeSpec(
                regimes=[{"name": "baseline", "dist": {"kind": "const", "v": 0.55}}]
            ),
            "bid.nuclear.min": VariableRegimeSpec(
                regimes=[{"name": "baseline", "dist": {"kind": "const", "v": -200.0}}]
            ),
            "bid.nuclear.max": VariableRegimeSpec(
                regimes=[{"name": "baseline", "dist": {"kind": "const", "v": -50.0}}]
            ),
            "bid.wind.min": VariableRegimeSpec(
                regimes=[{"name": "baseline", "dist": {"kind": "const", "v": -200.0}}]
            ),
            "bid.wind.max": VariableRegimeSpec(
                regimes=[{"name": "baseline", "dist": {"kind": "const", "v": -50.0}}]
            ),
            "bid.solar.min": VariableRegimeSpec(
                regimes=[{"name": "baseline", "dist": {"kind": "const", "v": -200.0}}]
            ),
            "bid.solar.max": VariableRegimeSpec(
                regimes=[{"name": "baseline", "dist": {"kind": "const", "v": -50.0}}]
            ),
        },
        empirical_series={},
        planned_outages=PlannedOutageConfig(enabled=False),
        renewable_availability_mode="weather_simulation",
        weather_simulation=WeatherSimulationConfig(),
        io=IOConfig(
            out_dir=str(temp_output_dir),
            dataset_name="test_scenario",
            add_timestamp=False,
            save_pickle=False,
            save_csv=True,
            save_meta=False,
        ),
    )

@pytest.fixture
def standard_vals():
    """Standard variable values for testing"""
    return {
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


@pytest.fixture
def scaled_vals(standard_vals):
    """Factory for scaled variable values"""
    def _scale(scale_factor):
        return {k: (v * scale_factor if k.startswith("cap.") else v)
                for k, v in standard_vals.items()}
    return _scale

@pytest.fixture
def sample_timeseries():
    """Sample time series for testing"""
    dates = pd.date_range("2024-01-01", periods=168, freq="h")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "value": np.random.randn(len(dates)) * 10 + 50,
        }
    )


@pytest.fixture
def all_scenario_configs():
    """List of all scenario config paths"""
    return [
        "configs/1_gas_crisis.yaml",
        "configs/2_coal_phaseout.yaml",
    ]
