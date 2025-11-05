"""
this module defines the configuration schema for the synthetic data generator.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator

# ------------------------------------------------------------------------------
# Sub-schemas (top level scheme below)
# ------------------------------------------------------------------------------


class DemandConfig(BaseModel):
    # inelastic demand toggle
    inelastic: bool = False  # if True, demand does not respond to price changes

    # if elastic demand:
    base_intercept: float = 45.0
    slope: float = -7.0

    # daily seasonality
    daily_seasonality: bool = True
    # if daily seasonality: (otherwise ignored)
    day_peak_hour: int = 14
    day_amp: float = 0.25
    weekend_drop: float = 0.10

    # annual seasonality (winter/summer)
    annual_seasonality: bool = True
    winter_amp: float = 0.15  # +15% in winter (Dec-Feb)
    summer_amp: float = -0.10  # -10% in summer (Jun-Aug)


class RegimePlanner(BaseModel):
    """Supply regime planner configuration"""

    mode: str = "hybrid"  # "local_only", "global", "hybrid"
    global_settings: Optional[GlobalSettings] = None

    @field_validator("mode")
    def validate_mode(cls, v):
        if v not in ["local_only", "global", "hybrid"]:
            raise ValueError(
                f"mode must be 'local_only', 'global', or 'hybrid', got '{v}'"
            )
        return v

    def model_post_init(self, __context):
        if self.mode in ["global", "hybrid"] and self.global_settings is None:
            raise ValueError(
                f"mode='{self.mode}' requires global_settings to be specified"
            )
        if self.mode == "local_only" and self.global_settings is not None:
            raise ValueError("mode='local_only' should not have global_settings")


class BreakpointSpec(BaseModel):
    """Breakpoint with date and transition hours"""

    date: str
    transition_hours: int = (
        24  # NB this happens pre-breakpoint (e.g. if 24, then transition is 24h before the date)
    )


class RegimeSpec(BaseModel):
    """Single regime specification with optional local breakpoints"""

    name: str
    dist: Dict[str, Any]
    breakpoints: Optional[List[BreakpointSpec]] = (
        None  # Local breakpoints for this regime
    )


class VariableRegimeSpec(BaseModel):
    """Per-variable regime specification"""

    regimes: List[Dict[str, Any]]  # Will be validated/converted

    @field_validator("regimes", mode="before")
    def normalize_regimes(cls, v):
        """Convert old format to new format if needed"""
        if not isinstance(v, list):
            raise ValueError("regimes must be a list")

        normalized = []
        for regime in v:
            if isinstance(regime, dict):
                # Ensure required fields
                if "name" not in regime or "dist" not in regime:
                    raise ValueError("Each regime must have 'name' and 'dist' fields")
                normalized.append(regime)
            else:
                normalized.append(regime)
        return normalized


class StochasticBreakpointConfig(BaseModel):
    """Configuration for stochastic breakpoint generation"""

    enabled: bool = False
    min_segment_days: int = 30
    max_segment_days: int = 180
    transition_hours: Dict[str, Any] = Field(
        default_factory=lambda: {"type": "fixed", "value": 168}
    )


class GlobalSettings(BaseModel):
    """Global regime settings (used in 'global' and 'hybrid' modes)"""

    n_regimes: int = 3
    sync_regimes: bool = True  # All variables transition together

    # Breakpoint specification - choose one approach
    breakpoints: Optional[List[BreakpointSpec]] = None  # Explicit breakpoints
    stochastic_breakpoints: Optional[StochasticBreakpointConfig] = None  # OR stochastic

    # Distribution templates by variable type (for auto-generation)
    distribution_templates: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class WeatherModelConfig(BaseModel):
    """Weather model configuration for a renewable source"""

    model: str
    params: Dict[str, Any] = Field(default_factory=dict)


class WeatherSimulationConfig(BaseModel):
    """Weather simulation parameters for renewables"""

    wind: WeatherModelConfig = Field(
        default_factory=lambda: WeatherModelConfig(
            model="ar1",
            params={
                "base_capacity_factor": 0.45,
                "persistence": 0.85,
                "volatility": 0.15,
            },
        )
    )
    solar: WeatherModelConfig = Field(
        default_factory=lambda: WeatherModelConfig(
            model="sinusoidal",
            params={"sunrise_hour": 6, "sunset_hour": 20, "peak_capacity_factor": 0.35},
        )
    )


class PlannedOutageConfig(BaseModel):
    """Configuration for planned maintenance outages (seasonal)"""

    enabled: bool = True
    # Months where planned outages occur (1=Jan, 12=Dec)
    months: List[int] = Field(default_factory=lambda: [5, 6, 7, 8, 9])  # May-Sep
    # Reduction factor for each technology
    nuclear_reduction: float = 0.10  # 10% reduction
    coal_reduction: float = 0.10
    gas_reduction: float = 0.10


class IOConfig(BaseModel):
    # Output configuration
    out_dir: str = "outputs"
    dataset_name: str = "synthetic_data"
    # optional: timestamp format for filenames
    timestamp_fmt: str = "%Y_%m_%d_%H_%M"
    add_timestamp: bool = True
    version: str = "v0"  # appears in file name and metadata
    # Optional: file format settings
    save_pickle: bool = True
    save_csv: bool = False
    save_parquet: bool = False
    save_feather: bool = False
    save_excel: bool = False
    save_preview_html: bool = False
    save_head_csv: bool = False
    save_meta: bool = False
    head_rows: int = 200


# ------------------------------------------------------------------------------
# Top-level config schema (YAML / JSON)
# ------------------------------------------------------------------------------


class TopConfig(BaseModel):
    start_ts: str = "2025-01-01 00:00"
    days: int = 30
    freq: str = "h"
    seed: int = 42
    price_grid: List[float] = Field(
        default_factory=lambda: list(map(float, range(-100, 301, 3)))
    )
    demand: DemandConfig = Field(default_factory=DemandConfig)
    supply_regime_planner: RegimePlanner = Field(default_factory=RegimePlanner)
    variables: Dict[str, VariableRegimeSpec]  # ALL RVs live here
    empirical_series: Dict[str, str] = Field(
        default_factory=dict
    )  # name -> path to CSV
    planned_outages: PlannedOutageConfig = Field(default_factory=PlannedOutageConfig)

    # Renewable availability mode
    renewable_availability_mode: str = "weather_simulation"  # or "direct"
    weather_simulation: WeatherSimulationConfig = Field(
        default_factory=WeatherSimulationConfig
    )

    io: IOConfig = Field(default_factory=IOConfig)

    @field_validator("start_ts")
    def _ts_ok(cls, v):
        pd.Timestamp(v)  # validate
        return v

    @field_validator("renewable_availability_mode")
    def validate_renewable_mode(cls, v):
        if v not in ["weather_simulation", "direct"]:
            raise ValueError(
                "renewable_availability_mode must be 'weather_simulation' or 'direct'"
            )
        return v

    @field_validator("variables")
    def _require_fuels(cls, v):
        # No baseline fuel prices: require fuel.coal and fuel.gas specs
        req = {"fuel.coal", "fuel.gas"}
        missing = [k for k in req if k not in v]
        if missing:
            raise ValueError(
                f"Missing required variable specs: {missing}. You must provide RVs or empirical for fuels."
            )
        return v

    def model_post_init(self, __context):
        """Validate mode-specific requirements"""
        if self.renewable_availability_mode == "direct":
            # Must have avail.wind and avail.solar specified
            required = ["avail.wind", "avail.solar"]
            missing = [k for k in required if k not in self.variables]
            if missing and not any(k in self.empirical_series for k in required):
                raise ValueError(
                    f"direct mode requires {missing} in variables or empirical_series"
                )
