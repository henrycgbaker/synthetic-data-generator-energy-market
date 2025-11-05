"""
= high level orchestration of regime schedules for multiple variables
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .regimes import RegimeSchedule
from .utils import random_partition


def _equalish_splits(total_days: int, n: int) -> list[int]:
    base = total_days // n
    rem = total_days % n
    return [base + (1 if i < rem else 0) for i in range(n)]


def _days_from_breakpoints(
    start: pd.Timestamp, days: int, breakpoints: List[Dict[str, Any]]
) -> tuple[List[int], List[int]]:
    """
    Convert list of breakpoint dicts [{date, transition_hours}] to segment lengths and transition hours
    Returns: (seg_days, transition_hours)
    """
    start = pd.Timestamp(start).normalize()
    end = (start + pd.Timedelta(days=days)).normalize()

    bps = [
        (pd.Timestamp(b["date"]).normalize(), b.get("transition_hours", 24))
        for b in breakpoints
    ]
    bps = sorted([(ts, th) for ts, th in bps if start <= ts < end], key=lambda x: x[0])

    if not bps or bps[0][0] != start:
        # Add start timestamp with first transition_hours or default
        th = bps[0][1] if bps else 24
        bps = [(start, th)] + bps

    # Extract timestamps and transition hours
    timestamps = [b[0] for b in bps]
    timestamps.append(end)
    transition_hours = [b[1] for b in bps]

    seg_days = []
    for i in range(len(timestamps) - 1):
        n_days = max(
            1, int(round((timestamps[i + 1] - timestamps[i]) / pd.Timedelta(days=1)))
        )
        seg_days.append(n_days)

    # Adjust last segment to match exactly
    total = sum(seg_days)
    if total != days:
        seg_days[-1] += days - total

    return seg_days, transition_hours


def _generate_stochastic_breakpoints(
    start_ts: pd.Timestamp,
    days: int,
    n_regimes: int,
    min_segment_days: int,
    max_segment_days: int,
    transition_hours_config: Dict[str, Any],
    rng: np.random.Generator,
) -> tuple[List[int], List[int]]:
    """
    Generate stochastic breakpoints and return segment days + transition hours
    """
    seg_days = random_partition(
        days, N=n_regimes, min_segment=min_segment_days, rng=rng
    )

    # Generate transition hours
    th_type = transition_hours_config.get("type", "fixed")
    if th_type == "fixed":
        th_value = transition_hours_config.get("value", 168)
        transition_hours = [th_value] * n_regimes
    elif th_type == "range":
        min_th = transition_hours_config.get("min", 24)
        max_th = transition_hours_config.get("max", 336)
        transition_hours = [
            int(rng.integers(min_th, max_th + 1)) for _ in range(n_regimes)
        ]
    else:
        raise ValueError(f"Unknown transition_hours type: {th_type}")

    return seg_days, transition_hours


def _extract_local_breakpoints(
    regimes: List[Dict[str, Any]],
) -> Optional[List[Dict[str, Any]]]:
    """
    Extract breakpoints from regime specifications (new format)
    Returns None if no local breakpoints found
    """
    breakpoints = []
    for regime in regimes:
        if "breakpoints" in regime and regime["breakpoints"]:
            for bp in regime["breakpoints"]:
                breakpoints.append(bp)

    return breakpoints if breakpoints else None


def build_schedules(
    start_ts,
    days: int,
    freq: str,
    seed: int,
    supply_regime_planner: dict,
    variables: dict,
    series_map: dict,
):
    """
    Build regime schedules for all variables based on mode:
    - local_only: Each variable must have full local spec
    - global: All variables use global settings (can override distributions only)
    - hybrid: Use global by default, but allow full local overrides
    """
    start_ts = pd.Timestamp(start_ts)
    rng = np.random.default_rng(seed)

    mode = supply_regime_planner.get("mode", "hybrid")
    global_settings = supply_regime_planner.get("global_settings", {})

    schedules = {}

    # === GLOBAL MODE ===
    if mode == "global":
        # Compute global segmentation once
        n_regimes = global_settings.get("n_regimes", 3)
        sync_regimes = global_settings.get("sync_regimes", True)

        # Determine breakpoints
        explicit_bps = global_settings.get("breakpoints")
        stochastic_bp_config = global_settings.get("stochastic_breakpoints", {})

        if explicit_bps:
            seg_days_global, th_global = _days_from_breakpoints(
                start_ts, days, explicit_bps
            )
        elif stochastic_bp_config.get("enabled", False):
            seg_days_global, th_global = _generate_stochastic_breakpoints(
                start_ts,
                days,
                n_regimes,
                stochastic_bp_config.get("min_segment_days", 30),
                stochastic_bp_config.get("max_segment_days", 180),
                stochastic_bp_config.get(
                    "transition_hours", {"type": "fixed", "value": 168}
                ),
                rng,
            )
        else:
            # Equal splits
            seg_days_global = _equalish_splits(days, n_regimes)
            th_global = [24] * n_regimes

        # Apply to all variables
        distribution_templates = global_settings.get("distribution_templates", {})

        for varname, varspec in variables.items():
            seg_defs = list(varspec["regimes"])
            target_N = len(seg_days_global)

            # If variable has distributions specified, use them
            # Otherwise try to get from templates
            if len(seg_defs) == 0:
                # No local spec - use template
                if varname in distribution_templates:
                    template_dist = distribution_templates[varname]
                    seg_defs = [
                        {"name": f"{varname}_regime_{i + 1}", "dist": template_dist}
                        for i in range(target_N)
                    ]
                else:
                    raise ValueError(
                        f"In 'global' mode, variable '{varname}' has no regimes and no distribution_template"
                    )

            # Replicate or trim regimes to match global segments
            if len(seg_defs) == 1 and target_N > 1:
                seg_defs = seg_defs * target_N
            elif len(seg_defs) != target_N:
                k = (target_N + len(seg_defs) - 1) // len(seg_defs)
                seg_defs = (seg_defs * k)[:target_N]

            # Build segments
            segments = []
            for i in range(target_N):
                reg = seg_defs[i]
                seg = {
                    "name": reg["name"],
                    "days": int(seg_days_global[i]),
                    "dist": reg["dist"],
                    "transition_hours": int(th_global[i]),
                }
                segments.append(seg)

            schedules[varname] = RegimeSchedule(
                varname=varname,
                start_ts=start_ts,
                freq=freq,
                segments=segments,
                rng=rng,
                series_map=series_map,
            )

    # === LOCAL_ONLY MODE ===
    elif mode == "local_only":
        for varname, varspec in variables.items():
            seg_defs = list(varspec["regimes"])

            if len(seg_defs) == 0:
                raise ValueError(
                    f"In 'local_only' mode, variable '{varname}' must have regimes specified"
                )

            # Extract local breakpoints from regime specs
            local_breakpoints = _extract_local_breakpoints(seg_defs)

            if local_breakpoints:
                # Use local breakpoints
                seg_days, transition_hours = _days_from_breakpoints(
                    start_ts, days, local_breakpoints
                )
            else:
                # Equal split based on number of regimes
                target_N = len(seg_defs)
                seg_days = _equalish_splits(days, target_N)
                transition_hours = [24] * target_N

            target_N = len(seg_days)

            # Align regimes with segments
            if len(seg_defs) != target_N:
                k = (target_N + len(seg_defs) - 1) // len(seg_defs)
                seg_defs = (seg_defs * k)[:target_N]

            segments = []
            for i in range(target_N):
                reg = seg_defs[i]
                seg = {
                    "name": reg["name"],
                    "days": int(seg_days[i]),
                    "dist": reg["dist"],
                    "transition_hours": int(transition_hours[i]),
                }
                segments.append(seg)

            schedules[varname] = RegimeSchedule(
                varname=varname,
                start_ts=start_ts,
                freq=freq,
                segments=segments,
                rng=rng,
                series_map=series_map,
            )

    # === HYBRID MODE ===
    elif mode == "hybrid":
        # Compute global segmentation
        n_regimes = global_settings.get("n_regimes", 3)
        sync_regimes = global_settings.get("sync_regimes", True)

        explicit_bps = global_settings.get("breakpoints")
        stochastic_bp_config = global_settings.get("stochastic_breakpoints", {})

        if explicit_bps:
            seg_days_global, th_global = _days_from_breakpoints(
                start_ts, days, explicit_bps
            )
        elif stochastic_bp_config.get("enabled", False):
            seg_days_global, th_global = _generate_stochastic_breakpoints(
                start_ts,
                days,
                n_regimes,
                stochastic_bp_config.get("min_segment_days", 30),
                stochastic_bp_config.get("max_segment_days", 180),
                stochastic_bp_config.get(
                    "transition_hours", {"type": "fixed", "value": 168}
                ),
                rng,
            )
        else:
            seg_days_global = _equalish_splits(days, n_regimes)
            th_global = [24] * n_regimes

        distribution_templates = global_settings.get("distribution_templates", {})

        for varname, varspec in variables.items():
            seg_defs = list(varspec["regimes"])

            # Check if variable has local breakpoints
            local_breakpoints = (
                _extract_local_breakpoints(seg_defs) if seg_defs else None
            )

            if local_breakpoints:
                # FULL LOCAL OVERRIDE: Variable has local breakpoints
                seg_days, transition_hours = _days_from_breakpoints(
                    start_ts, days, local_breakpoints
                )
                target_N = len(seg_days)

                # Align regimes
                if len(seg_defs) != target_N:
                    k = (target_N + len(seg_defs) - 1) // len(seg_defs)
                    seg_defs = (seg_defs * k)[:target_N]

            elif len(seg_defs) > 0:
                # PARTIAL OVERRIDE: Has distributions but no local breakpoints
                # Use global breakpoints with local distributions
                seg_days = seg_days_global
                transition_hours = th_global
                target_N = len(seg_days)

                if len(seg_defs) == 1 and target_N > 1:
                    seg_defs = seg_defs * target_N
                elif len(seg_defs) != target_N:
                    k = (target_N + len(seg_defs) - 1) // len(seg_defs)
                    seg_defs = (seg_defs * k)[:target_N]

            else:
                # NO LOCAL SPEC: Use global settings + templates
                seg_days = seg_days_global
                transition_hours = th_global
                target_N = len(seg_days)

                if varname in distribution_templates:
                    template_dist = distribution_templates[varname]
                    seg_defs = [
                        {"name": f"{varname}_regime_{i + 1}", "dist": template_dist}
                        for i in range(target_N)
                    ]
                else:
                    raise ValueError(
                        f"In 'hybrid' mode, variable '{varname}' has no regimes and no distribution_template. "
                        f"Either specify regimes locally or provide a distribution_template."
                    )

            # Build segments
            segments = []
            for i in range(target_N):
                reg = seg_defs[i]
                seg = {
                    "name": reg["name"],
                    "days": int(seg_days[i]),
                    "dist": reg["dist"],
                    "transition_hours": int(transition_hours[i]),
                }
                segments.append(seg)

            schedules[varname] = RegimeSchedule(
                varname=varname,
                start_ts=start_ts,
                freq=freq,
                segments=segments,
                rng=rng,
                series_map=series_map,
            )

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return schedules
