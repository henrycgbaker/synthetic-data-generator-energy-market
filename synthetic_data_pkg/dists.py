"""
this module provides random variable sampling utilities.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .utils import _clamp

# Uniform spec shape across all RVs:
# {"kind": "...", ...params..., "bounds": {"low": ..., "high": ...}}


def iid_sample(rng: np.random.Generator, spec: Dict[str, Any]) -> float:
    """
    Draws an independent and identically distributed (iid) sample from the specified distribution.

    Args:
        rng (np.random.Generator): Random number generator.
        spec (Dict[str, Any]): Specification of the distribution and its parameters.

    Raises:
        ValueError: If the specified distribution kind is unsupported.

    Returns:
        float: A sample from the specified distribution.
    """
    k = spec["kind"].lower()
    b = spec.get("bounds")
    if k == "const":
        return _clamp(float(spec["v"]), b)
    if k == "uniform":
        low = spec.get("min", 0.0)
        high = spec.get("max", 1.0)
        return _clamp(float(rng.uniform(low, high)), b)
    if k == "normal":
        return _clamp(float(rng.normal(spec["mu"], spec["sigma"])), b)
    if k == "lognormal":
        return _clamp(float(rng.lognormal(spec["mu"], spec["sigma"])), b)
    if k == "beta":
        x = float(rng.beta(spec["alpha"], spec["beta"]))
        val = spec.get("low", 0.0) + x * (spec.get("high", 1.0) - spec.get("low", 0.0))
        return _clamp(val, b)
    if k == "truncnormal":
        low, high = spec["low"], spec["high"]
        for _ in range(1000):
            x = float(rng.normal(spec["mu"], spec["sigma"]))
            if low <= x <= high:
                return _clamp(x, b)
        return _clamp(float(np.clip(x, low, high)), b)
    raise ValueError(f"Unsupported iid dist: {k}")


def stateful_step(
    rng: np.random.Generator, prev: Optional[float], spec: Dict[str, Any]
) -> float:
    """
    Stateful distributions that depend on previous value.
    Supports: AR1, RW, linear (deterministic)
    Args:
        rng (np.random.Generator): Random number generator.
        prev (Optional[float]): Previous value (None if first call).
        spec (Dict[str, Any]): Specification of the distribution and its parameters.

    For AR1:
        x_t = mu + phi * (x_{t-1} - mu) + eps_t,  eps_t ~ N(0, sigma^2)
    For RW:
        x_t = x_{t-1} + drift + eps_t,  eps_t ~ N(0, sigma^2)
    For linear:
        x_t = start + slope * t  (deterministic)
    """
    k = spec["kind"].lower()
    b = spec.get("bounds")

    if k == "ar1":
        # AR(1) process: x_t = mu + phi * (x_{t-1} - mu) + eps_t,  eps_t ~ N(0, sigma^2)
        mu, sigma, phi = spec["mu"], spec.get("sigma", 1.0), spec.get("phi", 0.9)
        xprev = mu if prev is None else prev
        eps = float(rng.normal(0.0, sigma))
        return _clamp(mu + phi * (xprev - mu) + eps, b)

    if k == "rw":
        # Random Walk with drift: x_t = x_{t-1} + drift + eps_t,  eps_t ~ N(0, sigma^2)
        drift, sigma, start = (
            spec.get("drift", 0.0),
            spec.get("sigma", 1.0),
            spec.get("start", 0.0),
        )
        xprev = start if prev is None else prev
        eps = float(rng.normal(0.0, sigma))
        return _clamp(xprev + drift + eps, b)

    if k == "linear":
        # Deterministic linear growth: v = start + slope * step_number
        # Store step count in spec (mutable dict - careful!)
        start = spec.get("start", 0.0)
        slope = spec.get("slope", 0.0)

        if prev is None:
            # First call: initialize
            spec["_step"] = 0
            return _clamp(start, b)

        # Increment step counter
        spec["_step"] = spec.get("_step", 0) + 1
        val = start + slope * spec["_step"]
        return _clamp(val, b)

    raise ValueError(f"Unsupported stateful dist: {k}")


def empirical_at(
    series_map: Dict[str, pd.Series], ts: pd.Timestamp, spec: Dict[str, Any]
) -> float:
    """
    Get empirical value at a specific timestamp.

    Args:
        series_map (Dict[str, pd.Series]): Mapping of series names to pandas Series.
        ts (pd.Timestamp):
        spec (Dict[str, Any]):

    Returns:
        float: Empirical value at the specified timestamp.
    """
    # Extract series name and transform
    name = spec["name"]
    transform = spec.get("transform", "level")

    if name not in series_map:
        raise KeyError(f"Empirical series '{name}' missing")

    # Ensure series is hourly
    s = series_map[name]
    if s.index.freq is None or s.index.freq != "h":
        s = s.asfreq("h", method="pad")

    # Get value at timestamp (or nearest prior)
    val = s.reindex([ts], method="pad").iloc[0]
    if transform == "level":
        out = float(val)
    elif transform == "pct_change":
        prev = s.reindex([ts - pd.Timedelta(hours=1)], method="pad").iloc[0]
        out = float((val / prev) - 1.0) if prev != 0 else 0.0
    elif transform == "diff":
        prev = s.reindex([ts - pd.Timedelta(hours=1)], method="pad").iloc[0]
        out = float(val - prev)
    else:
        raise ValueError(f"Unknown empirical transform: {transform}")

    return _clamp(out, spec.get("bounds"))
