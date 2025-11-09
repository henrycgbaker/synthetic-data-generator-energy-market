"""
this module defines how regime time series are generated for variables (fuels, availability, etc.), including stochastic draws, empirical series, and smooth transitions.
= low level mechanics for a single variable -> produces a single time series
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .dists import empirical_at, iid_sample, stateful_step
from .utils import _clamp, random_partition


class RegimeSchedule:
    """
    Per-variable schedule with state; supports iid, AR1/RW, empirical; linear blend near regime end.
    """

    def __init__(
        self,
        varname: str,
        start_ts: pd.Timestamp,
        freq: str,
        segments: List[
            Dict[str, Any]
        ],  # [{"name":..., "days":int, "dist":{...}, "transition_hours":int}, ...]
        rng: np.random.Generator,
        series_map: Dict[str, pd.Series],
    ):
        self.varname = varname
        self.rng = rng
        self.series_map = series_map
        self.segments = segments
        hours = int(sum(seg["days"] for seg in segments) * 24)
        self.index = pd.date_range(start=start_ts, periods=hours, freq=freq)

        # Expand labels
        labs = []
        for seg in segments:
            labs.extend([seg["name"]] * (seg["days"] * 24))
        self.labels = pd.Series(labs, index=self.index, name=f"{varname}_regime")

        # stateful memory
        self._last_ts: Optional[pd.Timestamp] = None
        self._last_value: Optional[float] = None
        self._last_seg_idx: Optional[int] = None
        self._step_counter: int = 0

    def _blend(
        self, ts: pd.Timestamp, seg_idx: int
    ) -> Tuple[float, float, Optional[int]]:
        """
        Returns (w_curr, w_next, next_seg_idx) for blending at time ts in segment seg_idx
        """
        seg = self.segments[seg_idx]
        th = int(seg.get("transition_hours", 0))
        if th <= 0 or seg_idx >= len(self.segments) - 1:
            return 1.0, 0.0, None
        seg_end = self.labels[self.labels == seg["name"]].index[-1]
        hours_to_end = int((seg_end - ts) / pd.Timedelta(hours=1))
        if 0 <= hours_to_end < th:
            w_next = 1.0 - (hours_to_end / th)
            return 1.0 - w_next, w_next, seg_idx + 1
        return 1.0, 0.0, None

    def value_at(self, ts: pd.Timestamp) -> Tuple[float, str]:
        """
        Returns value and regime name at a specific timestamp.

        Args:
            ts (pd.Timestamp): The timestamp to query.

        Returns:
            Tuple[float, str]: The value and regime name at the specified timestamp.
        """
        ts = min(max(ts, self.index[0]), self.index[-1])
        seg_name = self.labels.loc[ts]
        seg_idx = next(i for i, s in enumerate(self.segments) if s["name"] == seg_name)
        w_curr, w_next, next_idx = self._blend(ts, seg_idx)
        curr, nxt = (
            self.segments[seg_idx],
            (self.segments[next_idx] if next_idx is not None else None),
        )
        dist_curr, dist_next = curr["dist"], (nxt["dist"] if nxt else None)

        # steps since last tick
        steps = (
            1
            if self._last_ts is None
            else max(1, int((ts - self._last_ts) / pd.Timedelta(hours=1)))
        )

        # Reset state when changing regimes
        if (self._last_seg_idx is not None) and (seg_idx != self._last_seg_idx):
            self._last_value = None
            self._step_counter = 0

        kind = dist_curr["kind"].lower()
        if kind == "empirical":
            v = empirical_at(self.series_map, ts, dist_curr)
        elif kind in ("ar1", "rw", "linear"):
            v = self._last_value

            # Special handling for linear: use absolute time from segment start
            if kind == "linear":
                start = dist_curr.get("start", 0.0)
                slope = dist_curr.get("slope", 0.0)
                bounds = dist_curr.get("bounds")

                # Calculate hours from segment start
                seg_start = self.labels[self.labels == seg_name].index[0]
                hours_from_start = int((ts - seg_start) / pd.Timedelta(hours=1))

                # Linear: value = start + slope * hours
                v = _clamp(start + slope * hours_from_start, bounds)
            else:
                # AR1 and RW: use existing logic with blended params
                for _ in range(steps):
                    p = dist_curr.copy()
                    if dist_next and w_next > 0:
                        p.update(
                            {
                                k: w_curr * dist_curr.get(k, 0)
                                + w_next * dist_next.get(k, 0)
                                for k in (
                                    "mu",
                                    "sigma",
                                    "phi",
                                    "drift",
                                    "start",
                                    "slope",
                                )
                                if (k in dist_curr or (dist_next and k in dist_next))
                            }
                        )
                        if "bounds" in dist_curr or (
                            dist_next and "bounds" in dist_next
                        ):
                            low = min(
                                dist_curr.get("bounds", {}).get("low", -np.inf),
                                dist_next.get("bounds", {}).get("low", -np.inf),
                            )
                            high = max(
                                dist_curr.get("bounds", {}).get("high", np.inf),
                                dist_next.get("bounds", {}).get("high", np.inf),
                            )
                            p["bounds"] = {"low": low, "high": high}
                    v = stateful_step(self.rng, v, p)
        else:
            # iid draw(s), blend values linearly
            if dist_next and w_next > 0:
                v0 = iid_sample(self.rng, dist_curr)
                v1 = iid_sample(self.rng, dist_next)
                v = float(w_curr * v0 + w_next * v1)
            else:
                v = iid_sample(self.rng, dist_curr)

        self._last_ts = ts
        self._last_value = float(v)
        self._last_seg_idx = seg_idx
        return float(v), seg_name


def plan_days(
    start_ts: pd.Timestamp,
    days: int,
    planner: Dict[str, Any],
    varname: str,
    # optional per-var overrides:
    breakpoints: Optional[List[str]] = None,
    N_override: Optional[int] = None,
    min_seg_override: Optional[int] = None,
) -> List[int]:
    """
    Splits a simulation period into regime segments (days), using breakpoints or stochastic partitioning, depending on planner settings.
    """
    start = pd.Timestamp(start_ts).normalize()
    end = (pd.Timestamp(start_ts) + pd.Timedelta(days=days)).normalize()

    def _days_from_bps(bps: List[pd.Timestamp]) -> List[int]:
        B = sorted([pd.Timestamp(b) for b in bps if start <= pd.Timestamp(b) < end])
        if not B or B[0] != start:
            B = [start] + B
        B.append(end)
        outs = [
            max(1, int(round((B[i + 1] - B[i]) / pd.Timedelta(days=1))))
            for i in range(len(B) - 1)
        ]
        need = (end - start).days
        diff = need - sum(outs)
        if diff != 0:
            outs[-1] += diff
        return outs

    sync = planner["sync_regimes"]
    stochastic = planner["stochastic_regimes"]
    if sync and not stochastic:
        global_bps = planner.get("breakpoints", {}).get("GLOBAL")
        if not global_bps:
            raise ValueError(
                "SYNC_REGIMES=True & STOCH_REGIMES=False requires 'planner.breakpoints.GLOBAL'"
            )
        return _days_from_bps(global_bps)

    if not sync and not stochastic:
        return _days_from_bps(breakpoints or [])

    # stochastic cases
    rng = np.random.default_rng(planner.get("seed", 42))
    N = N_override or planner["global_regimes_n"]
    ms = min_seg_override or planner["min_segment_days_global"]
    return random_partition(days, N=N, min_segment=ms, rng=rng)
