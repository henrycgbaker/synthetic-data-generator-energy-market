from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def random_partition(
    days: int, N: int, min_segment: int = 1, rng: Optional[np.random.Generator] = None
) -> List[int]:
    assert N * min_segment <= days, "min_segment too large for the given DAYS and N"
    rng = rng or np.random.default_rng()
    remaining = days - N * min_segment
    if remaining == 0:
        return [min_segment] * N
    cuts = np.sort(rng.choice(np.arange(remaining + N - 1), size=N - 1, replace=False))
    parts = np.diff(np.r_[[-1], cuts, [remaining + N - 1]]) - 1
    return [int(min_segment + x) for x in parts]


def linear_ramp(price: float, p_low: float, p_high: float, cap: float) -> float:
    if np.isinf(p_low) or np.isinf(p_high) or cap <= 0:
        return 0.0
    if price <= p_low:
        return 0.0
    if price >= p_high:
        return cap
    w = (price - p_low) / (p_high - p_low)
    return float(cap * max(0.0, min(1.0, w)))


def now_stamp() -> str:
    # ISO-like, filename safe
    return pd.Timestamp.utcnow().strftime("%Y_%m_%d_T_%H_%M")


def _clamp(x: float, bounds: Optional[Dict[str, float]]) -> float:
    """
    Clamp a value to the specified bounds.

    Args:
        x (float):
        bounds (Optional[Dict[str, float]]):

    Returns:
        float: Clamped value.
    """
    if not bounds:
        return x
    low = bounds.get("low", -np.inf)
    high = bounds.get("high", np.inf)
    return float(np.clip(x, low, high))
