"""
Core workflow execution logic for running synthetic data scenarios.

This module contains the main execution pipeline that orchestrates:
- Config loading and validation
- Regime schedule building
- Time series simulation
- Dataset saving

Separated from cli.py to keep CLI concerns separate from business logic.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import TopConfig
from .io import load_config, load_empirical_series, save_dataset
from .scenario import build_schedules
from .simulate import simulate_timeseries

# Configure logger
logger = logging.getLogger(__name__)

# Configure default logging format if not already configured
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def execute_scenario(config_path: str | Path) -> dict[str, Path]:
    """
    Execute a complete scenario simulation from a config file.

    Args:
        config_path: Path to YAML/JSON config file (string or Path object)

    Returns:
        Dictionary mapping output type to file path (e.g. {"csv": Path(...), "pickle": Path(...)})

    Raises:
        FileNotFoundError: If config file cannot be found
        ValidationError: If config is invalid
    """

    logger.info("=" * 60)
    logger.info("   SUPPLY CURVES - Synthetic Data Generation")
    logger.info("=" * 60)

    # Use current working directory (where the CLI is run)
    cwd = Path.cwd()
    package_dir = Path(__file__).resolve().parent
    project_root = package_dir.parent  # one level up (synthetic_data)

    config_path = Path(config_path).expanduser()

    possible_paths = [
        config_path,
        cwd / config_path,
        project_root / config_path,
        project_root / "configs" / config_path,
    ]

    # Also try adding .yaml or .json extensions if missing
    candidates = []
    for p in possible_paths:
        if p.suffix:
            candidates.append(p)
        else:
            candidates += [p.with_suffix(ext) for ext in (".yaml", ".yml", ".json")]

    for p in candidates:
        if p.exists():
            config_path = p.resolve()
            break
    else:
        raise FileNotFoundError(
            "Config file not found in any of:\n  "
            + "\n  ".join(str(p) for p in candidates)
        )

    cfg_raw = load_config(config_path)  # loads YAML/JSON to Dict
    logger.info(f"Loading configuration from: {config_path.name}")
    cfg = TopConfig(**cfg_raw)  # wrap in TopConfig class -> validate and attr access
    logger.info("Configuration loaded successfully")
    logger.info(f"  Scenario: {cfg.io.dataset_name}")
    logger.info(f"  Duration: {cfg.days} days ({cfg.days/365:.1f} years)")
    logger.info(f"  Start: {cfg.start_ts}")
    logger.info(f"  Frequency: {cfg.freq}")

    series_map = load_empirical_series(
        cfg.empirical_series
    )  # if user provides empirical series it loads this too
    if cfg.empirical_series:
        logger.info("Loading empirical time series data...")
    if series_map:
        logger.info(f"Loaded {len(series_map)} empirical series")

    # cnvert config objects to dicts
    def to_dict(obj):
        if hasattr(obj, "model_dump"):  # pydantic 2
            return obj.model_dump()
        elif hasattr(obj, "dict"):  # pydantic 1
            return obj.dict()
        else:
            return obj

    logger.info("Building regime schedules...")
    schedules = build_schedules(  # builds regime schedule for every var
        start_ts=cfg.start_ts,
        days=cfg.days,
        freq=cfg.freq,
        seed=cfg.seed,
        supply_regime_planner=to_dict(cfg.supply_regime_planner),
        variables={k: to_dict(v) for k, v in cfg.variables.items()},
        series_map=series_map,
    )
    logger.info(f"Built schedules for {len(schedules)} variables")
    n_regimes = {k: len(v.segments) for k, v in schedules.items()}
    max_regimes = max(n_regimes.values())
    min_regimes = min(n_regimes.values())
    if max_regimes == min_regimes:
        logger.info(f"  All variables have {max_regimes} regime(s)")
    else:
        logger.info(
            f"  Regimes per variable: (min:) {min_regimes} - (max:) {max_regimes}"
        )

    price_grid = np.array(cfg.price_grid, dtype=float)

    hours = (  # number of simulation hours/steps
        cfg.days * 24
        if cfg.freq.lower() == "h"
        else len(pd.date_range(start=cfg.start_ts, periods=cfg.days, freq=cfg.freq))
    )

    logger.info(f"Simulating {hours:,} hourly timesteps...")
    df = simulate_timeseries(
        start_ts=cfg.start_ts,
        hours=hours,
        demand_cfg=to_dict(cfg.demand),  # demand is exogenous
        schedules=schedules,  # supply is modelled
        price_grid=price_grid,
        seed=cfg.seed,
        config=cfg,  # Pass full config for supply curve
        planned_outages_cfg=to_dict(cfg.planned_outages),
    )
    logger.info("Simulation complete")
    logger.info(f"  Output shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    logger.info(f"Saving outputs to: {cfg.io.out_dir}/")
    paths = save_dataset(
        df=df,
        out_dir=cfg.io.out_dir,
        base_name=cfg.io.dataset_name,
        io_cfg=to_dict(cfg.io),
        meta={
            "created_at": pd.Timestamp.utcnow(),
            "version": cfg.io.version,
            "config": to_dict(cfg),
        },
    )
    if paths:
        logger.info("Artifacts written:")
        for k, p in paths.items():
            filename = Path(p).name
            logger.info(f"  {k:12s} -> {filename}")
        logger.info("=" * 60)
        logger.info("   Scenario generation complete!")
        logger.info("=" * 60)
    else:
        logger.warning("No artifacts requested; set io.save_* flags in config.")
        logger.info("=" * 60)

    return paths
