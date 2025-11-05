"""
this module provides I/O utilities for loading configs, empirical data, and saving datasets.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import pandas as pd
import yaml

try:
    from .config import IOConfig  # if you have it
except Exception:
    IOConfig = None


def load_config(path: str) -> Dict:
    """
    Resolve and load a YAML/JSON config.

    Resolution order:
      1) Absolute path or relative to cwd
      2) $SUPPLYCURVES_CONFIG_DIR/path
      3) synthetic_data_pkg/configs/path (package)
      4) synthetic_data/configs/path (repo-level synthetic_data/configs)
      5) legacy repo-root/path (old behaviour)
    """
    path = os.path.expanduser(path)
    p = Path(path)
    tried = []

    # 1) If p is absolute and exists -> use it. Otherwise try relative to cwd.
    if p.is_absolute():
        tried.append(str(p))
        if p.exists():
            return _read_config_file(p)
    # try cwd / p
    candidate = Path.cwd() / p
    tried.append(str(candidate))
    if candidate.exists():
        return _read_config_file(candidate)

    # 2) SUPPLYCURVES_CONFIG_DIR env var
    env_dir = os.environ.get("SUPPLYCURVES_CONFIG_DIR")
    if env_dir:
        candidate = Path(env_dir) / p
        tried.append(str(candidate))
        if candidate.exists():
            return _read_config_file(candidate)

    # 3) package configs (synthetic_data_pkg/configs)
    try:
        import synthetic_data_pkg

        pkg_dir = (
            Path(synthetic_data_pkg.__file__).resolve().parent
        )  # synthetic_data_pkg/
        candidate = pkg_dir / "configs" / p
        tried.append(str(candidate))
        if candidate.exists():
            return _read_config_file(candidate)
    except Exception:
        # package import failed or no package copy – continue
        pass

    # 4) repo-level synthetic_data/configs (your stated location)
    try:
        # from synthetic_data_pkg location walk up to synthetic_data/ then find configs/
        import synthetic_data_pkg

        pkg_dir = (
            Path(synthetic_data_pkg.__file__).resolve().parent
        )  # synthetic_data_pkg/
        synthetic_data_dir = pkg_dir.parent  # synthetic_data/
        candidate = synthetic_data_dir / "configs" / p
        tried.append(str(candidate))
        if candidate.exists():
            return _read_config_file(candidate)
    except Exception:
        pass

    # 5) legacy: repo root + path (two parents above package)
    try:
        import synthetic_data_pkg

        root = Path(synthetic_data_pkg.__file__).resolve().parent  # synthetic_data_pkg/
        root = root.parent  # synthetic_data/
        root = root.parent  # repository root
        candidate = root / p
        tried.append(str(candidate))
        if candidate.exists():
            return _read_config_file(candidate)
    except Exception:
        pass

    tried_str = "\n  - ".join(tried)
    raise FileNotFoundError(
        f"Config file not found. Looked for '{path}' in:\n  - {tried_str}"
    )


def _read_config_file(candidate: Path) -> Dict:
    suffix = candidate.suffix.lower()
    with candidate.open("r") as f:
        if suffix in (".yml", ".yaml"):
            return yaml.safe_load(f)
        elif suffix == ".json":
            return json.load(f)
        else:
            # try YAML first, then JSON
            try:
                f.seek(0)
                return yaml.safe_load(f)
            except Exception:
                f.seek(0)
                return json.load(f)


def load_single_column_csv(
    path: str, value_col: str = "value", ts_col: str = "ts"
) -> pd.Series:
    """
    Load a single-column CSV with timestamps.

    Expected format:
    - Column 1: timestamps (ts_col, default "ts")
    - Column 2: values (value_col, default "value")

    If CSV has only one column, assumes it's values with implicit hourly index.

    Returns: pd.Series with DatetimeIndex
    """
    df = pd.read_csv(path)

    if len(df.columns) == 1:
        # Single column: assume hourly values, create implicit index
        values = df.iloc[:, 0].values
        index = pd.date_range(start="2020-01-01", periods=len(values), freq="h")
        return pd.Series(values, index=index, name=Path(path).stem)

    # Two columns: expect ts and value
    if ts_col not in df.columns:
        # Try to find timestamp column
        for col in df.columns:
            if col.lower() in ("time", "timestamp", "date", "datetime", "ts"):
                ts_col = col
                break
        else:
            raise ValueError(
                f"Could not find timestamp column in {path}. Expected '{ts_col}'"
            )

    if value_col not in df.columns:
        # Use the other column
        value_col = [c for c in df.columns if c != ts_col][0]

    s = pd.Series(
        df[value_col].values, index=pd.to_datetime(df[ts_col]), name=Path(path).stem
    ).sort_index()

    return s


def load_empirical_series(series_map_cfg: Dict[str, str]) -> Dict[str, pd.Series]:
    """
    Load empirical time series from CSV files.

     takes a dictionary, series_map_cfg, where the keys are names of the time series
     (e.g., "fuel_prices", "demand") and the values are file paths to the corresponding CSV files.
     It returns a dictionary where the keys are the same series names,
     and the values are pandas.Series objects representing the loaded time series data.

    Supports:
    - Standard format: ts, value columns
    - Single column format: just values (implicit hourly index)
    - Fuel prices, demand, renewable generation, etc.
    """
    out = {}
    for name, path in series_map_cfg.items():
        try:
            s = load_single_column_csv(path)
            s.name = name
            out[name] = s
        except Exception as e:
            # Try old format for backward compatibility
            try:
                df = pd.read_csv(path)
                s = pd.Series(
                    df["value"].values, index=pd.to_datetime(df["ts"]), name=name
                ).sort_index()
                out[name] = s
            except Exception:
                raise ValueError(
                    f"Failed to load empirical series '{name}' from {path}: {e}"
                )
    return out


def _as_io_obj(io_cfg):
    # Dict? Coerce to IOConfig if available, else a simple namespace
    if isinstance(io_cfg, dict):
        if IOConfig is not None:
            try:
                return IOConfig(**io_cfg)
            except Exception:
                pass
        return SimpleNamespace(**io_cfg)
    raise TypeError("io_cfg must be IOConfig or dict-like")


def _make_dataset_name(base: str, version: str, io_cfg) -> str:
    add_ts = getattr(io_cfg, "add_timestamp", True)
    ts_fmt = getattr(io_cfg, "timestamp_fmt", "%Y_%m_%d_T_%H_%M")

    if add_ts:
        ts_str = pd.Timestamp.utcnow().strftime(ts_fmt)
        return f"{base}_{version}_{ts_str}"
    else:
        return f"{base}_{version}"


def save_dataset(
    df: pd.DataFrame, out_dir: str, base_name: str, io_cfg: Dict, meta: Dict
) -> Dict[str, str]:
    # --- always resolve relative to package root ---
    if not os.path.isabs(out_dir):
        root = os.path.dirname(__file__)  # synthetic_data_pkg/
        root = os.path.abspath(os.path.join(root, ".."))  # synthetic_data/
        out_dir = os.path.join(root, out_dir)

    os.makedirs(out_dir, exist_ok=True)
    io_cfg = _as_io_obj(io_cfg)
    name = _make_dataset_name(base_name, io_cfg.version, io_cfg)
    paths = {}

    if io_cfg.save_csv:
        p = os.path.join(out_dir, f"{name}.csv")
        df.to_csv(p)
        paths["csv"] = p

    if io_cfg.save_parquet:
        p = os.path.join(out_dir, f"{name}.parquet")
        df.to_parquet(p)
        paths["parquet"] = p

    if io_cfg.save_feather:
        p = os.path.join(out_dir, f"{name}.feather")
        df.reset_index().to_feather(p)
        paths["feather"] = p

    if io_cfg.save_pickle:
        p = os.path.join(out_dir, f"{name}.pkl")
        df.reset_index().to_pickle(p)
        paths["pickle"] = p

    if io_cfg.save_preview_html:
        p = os.path.join(out_dir, f"{name}_preview.html")
        html = df.head(io_cfg.head_rows).to_html(index=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(html)
        paths["preview_html"] = p

    # optionally save metadata
    if getattr(io_cfg, "save_meta", False):
        meta_p = os.path.join(out_dir, f"{name}_meta.json")
        with open(meta_p, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        paths["meta"] = meta_p

    return paths
