"""
Diagnostic tests for linear distribution bug.
"""

import numpy as np
import pandas as pd
import pytest

from synthetic_data_pkg.regimes import RegimeSchedule


@pytest.mark.unit
class TestLinearDistribution:
    """Tests specifically for linear distribution functionality"""

    def test_linear_distribution_increments(self):
        """Test that linear distribution actually increments over time"""
        start_ts = pd.Timestamp("2024-01-01")
        rng = np.random.default_rng(42)

        # Create a schedule with linear distribution
        segments = [
            {
                "name": "linear_growth",
                "days": 10,
                "dist": {"kind": "linear", "start": 100.0, "slope": 1.0},
                "transition_hours": 0,
            }
        ]

        schedule = RegimeSchedule(
            varname="test_var",
            start_ts=start_ts,
            freq="h",
            segments=segments,
            rng=rng,
            series_map={},
        )

        # Sample at different times
        values = []
        for hour in [0, 1, 2, 5, 10, 20, 50, 100]:
            ts = start_ts + pd.Timedelta(hours=hour)
            val, regime = schedule.value_at(ts)
            values.append(val)
            print(f"Hour {hour:3d}: value = {val:8.2f}")

        # Check that values are increasing
        for i in range(1, len(values)):
            assert (
                values[i] > values[i - 1]
            ), f"Value at index {i} ({values[i]}) should be > previous ({values[i-1]})"

        # Check specific values
        # Hour 0: 100 + 0*1 = 100
        # Hour 1: 100 + 1*1 = 101
        # Hour 2: 100 + 2*1 = 102
        assert values[0] == pytest.approx(100.0, abs=0.1)
        assert values[1] == pytest.approx(101.0, abs=0.1)
        assert values[2] == pytest.approx(102.0, abs=0.1)
        assert values[3] == pytest.approx(105.0, abs=0.1)  # Hour 5
        assert values[7] == pytest.approx(200.0, abs=0.1)  # Hour 100

    def test_linear_distribution_with_negative_slope(self):
        """Test linear distribution with declining values"""
        start_ts = pd.Timestamp("2024-01-01")
        rng = np.random.default_rng(42)

        segments = [
            {
                "name": "linear_decline",
                "days": 100,
                "dist": {"kind": "linear", "start": 8000.0, "slope": -0.1826},
                "transition_hours": 0,
            }
        ]

        schedule = RegimeSchedule(
            varname="cap.coal",
            start_ts=start_ts,
            freq="h",
            segments=segments,
            rng=rng,
            series_map={},
        )

        # Sample over time
        values = []
        hours_to_test = [0, 100, 500, 1000, 1500, 2000, 2400]

        for hour in hours_to_test:
            ts = start_ts + pd.Timedelta(hours=hour)
            val, regime = schedule.value_at(ts)
            values.append(val)
            expected = 8000.0 - 0.1826 * hour
            print(
                f"Hour {hour:4d}: value = {val:8.2f}, expected = {expected:8.2f}, diff = {val - expected:8.2f}"
            )

        # Check values are decreasing
        for i in range(1, len(values)):
            assert (
                values[i] < values[i - 1]
            ), f"Value at {hours_to_test[i]} ({values[i]}) should be < previous ({values[i-1]})"

        # Check specific values
        assert values[0] == pytest.approx(8000.0, abs=1.0)
        assert values[-1] < 7600.0  # Should have declined significantly

    def test_linear_distribution_sequential_calls(self):
        """Test that sequential calls maintain state correctly"""
        start_ts = pd.Timestamp("2024-01-01")
        rng = np.random.default_rng(42)

        segments = [
            {
                "name": "linear_growth",
                "days": 5,
                "dist": {"kind": "linear", "start": 1000.0, "slope": 10.0},
                "transition_hours": 0,
            }
        ]

        schedule = RegimeSchedule(
            varname="test_var",
            start_ts=start_ts,
            freq="h",
            segments=segments,
            rng=rng,
            series_map={},
        )

        # Call sequentially hour by hour
        prev_val = None
        for hour in range(24):
            ts = start_ts + pd.Timedelta(hours=hour)
            val, regime = schedule.value_at(ts)
            expected = 1000.0 + 10.0 * hour

            print(
                f"Hour {hour:2d}: value = {val:8.2f}, expected = {expected:8.2f}, match = {abs(val - expected) < 0.1}"
            )

            if prev_val is not None:
                assert val > prev_val, f"Hour {hour}: {val} should be > {prev_val}"
                assert val == pytest.approx(
                    prev_val + 10.0, abs=0.1
                ), f"Hour {hour}: increment should be 10.0"

            assert val == pytest.approx(
                expected, abs=0.1
            ), f"Hour {hour}: expected {expected}, got {val}"
            prev_val = val

    def test_linear_distribution_with_bounds(self):
        """Test that linear distribution respects bounds"""
        start_ts = pd.Timestamp("2024-01-01")
        rng = np.random.default_rng(42)

        segments = [
            {
                "name": "linear_bounded",
                "days": 10,
                "dist": {
                    "kind": "linear",
                    "start": 100.0,
                    "slope": 50.0,
                    "bounds": {"low": 0.0, "high": 300.0},
                },
                "transition_hours": 0,
            }
        ]

        schedule = RegimeSchedule(
            varname="test_var",
            start_ts=start_ts,
            freq="h",
            segments=segments,
            rng=rng,
            series_map={},
        )

        # Sample at times that should exceed bounds
        for hour in [0, 2, 4, 6, 8, 10]:
            ts = start_ts + pd.Timedelta(hours=hour)
            val, regime = schedule.value_at(ts)

            # Should never exceed bounds
            assert (
                0.0 <= val <= 300.0
            ), f"Hour {hour}: value {val} outside bounds [0, 300]"

            # At hour 6+, should hit upper bound (100 + 50*6 = 400, clamped to 300)
            if hour >= 4:
                assert (
                    val == 300.0
                ), f"Hour {hour}: should be at upper bound 300.0, got {val}"

    def test_linear_vs_const_distribution(self):
        """Compare linear to const to verify they behave differently"""
        start_ts = pd.Timestamp("2024-01-01")
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        # Linear schedule
        segments_linear = [
            {
                "name": "linear",
                "days": 10,
                "dist": {"kind": "linear", "start": 100.0, "slope": 1.0},
                "transition_hours": 0,
            }
        ]

        schedule_linear = RegimeSchedule(
            varname="test_var",
            start_ts=start_ts,
            freq="h",
            segments=segments_linear,
            rng=rng1,
            series_map={},
        )

        # Const schedule
        segments_const = [
            {
                "name": "const",
                "days": 10,
                "dist": {"kind": "const", "v": 100.0},
                "transition_hours": 0,
            }
        ]

        schedule_const = RegimeSchedule(
            varname="test_var",
            start_ts=start_ts,
            freq="h",
            segments=segments_const,
            rng=rng2,
            series_map={},
        )

        # Sample at different times
        linear_values = []
        const_values = []

        for hour in [0, 10, 20, 50, 100]:
            ts = start_ts + pd.Timedelta(hours=hour)

            val_linear, _ = schedule_linear.value_at(ts)
            val_const, _ = schedule_const.value_at(ts)

            linear_values.append(val_linear)
            const_values.append(val_const)

            print(
                f"Hour {hour:3d}: linear = {val_linear:8.2f}, const = {val_const:8.2f}"
            )

        # Const should be flat
        assert all(v == 100.0 for v in const_values), "Const values should all be 100.0"

        # Linear should increase
        assert (
            linear_values[0] < linear_values[-1]
        ), "Linear values should increase over time"

        # They should be different (except at hour 0)
        for i in range(1, len(linear_values)):
            assert (
                linear_values[i] != const_values[i]
            ), f"At index {i}, linear and const should differ"

    def test_coal_phaseout_scenario_capacities(self):
        """Test the exact config from coal phaseout scenario"""
        start_ts = pd.Timestamp("2024-01-01")
        days = 1825  # 5 years
        hours = days * 24
        rng = np.random.default_rng(42)

        # Coal capacity: declining
        segments_coal = [
            {
                "name": "declining",
                "days": days,
                "dist": {"kind": "linear", "start": 8000.0, "slope": -0.1826},
                "transition_hours": 0,
            }
        ]

        schedule_coal = RegimeSchedule(
            varname="cap.coal",
            start_ts=start_ts,
            freq="h",
            segments=segments_coal,
            rng=rng,
            series_map={},
        )

        # Sample at key points
        test_points = [0, hours // 4, hours // 2, 3 * hours // 4, hours - 1]

        print("\nCoal capacity over 5 years:")
        for hour in test_points:
            ts = start_ts + pd.Timedelta(hours=hour)
            val, _ = schedule_coal.value_at(ts)
            expected = max(0, 8000.0 - 0.1826 * hour)
            year = hour / (365.25 * 24)
            print(
                f"Year {year:4.1f} (hour {hour:5d}): {val:7.1f} MW (expected {expected:7.1f} MW)"
            )

        # At start: should be 8000
        ts = start_ts
        val, _ = schedule_coal.value_at(ts)
        assert val == pytest.approx(8000.0, abs=1.0), f"Start: expected 8000, got {val}"

        # At end: should be close to 0
        ts = start_ts + pd.Timedelta(hours=hours - 1)
        val, _ = schedule_coal.value_at(ts)
        assert val < 100.0, f"End: expected near 0, got {val}"

        # Check monotonic decrease
        prev_val = None
        sample_hours = np.linspace(0, hours - 1, 20, dtype=int)
        for hour in sample_hours:
            ts = start_ts + pd.Timedelta(hours=int(hour))
            val, _ = schedule_coal.value_at(ts)
            if prev_val is not None:
                assert (
                    val < prev_val
                ), f"Hour {hour}: coal capacity should be decreasing"
            prev_val = val


@pytest.mark.unit
class TestLinearDistributionEdgeCases:
    """Additional edge case tests for linear distribution"""

    def test_linear_with_zero_slope(self):
        """Test that slope=0 behaves like const distribution"""
        start_ts = pd.Timestamp("2024-01-01")
        rng = np.random.default_rng(42)

        segments = [
            {
                "name": "flat",
                "days": 10,
                "dist": {"kind": "linear", "start": 100.0, "slope": 0.0},
                "transition_hours": 0,
            }
        ]

        schedule = RegimeSchedule(
            varname="test_var",
            start_ts=start_ts,
            freq="h",
            segments=segments,
            rng=rng,
            series_map={},
        )

        # Sample at different times - should all be 100.0
        for hour in [0, 10, 50, 100, 200]:
            ts = start_ts + pd.Timedelta(hours=hour)
            val, _ = schedule.value_at(ts)
            assert val == pytest.approx(
                100.0, abs=0.1
            ), f"Slope=0 should be constant at hour {hour}"

    def test_linear_with_very_large_positive_slope(self):
        """Test linear with very large positive slope"""
        start_ts = pd.Timestamp("2024-01-01")
        rng = np.random.default_rng(42)

        segments = [
            {
                "name": "rapid_growth",
                "days": 100,
                "dist": {"kind": "linear", "start": 1000.0, "slope": 100.0},
                "transition_hours": 0,
            }
        ]

        schedule = RegimeSchedule(
            varname="test_var",
            start_ts=start_ts,
            freq="h",
            segments=segments,
            rng=rng,
            series_map={},
        )

        # Check values grow rapidly
        ts_0 = start_ts
        ts_100 = start_ts + pd.Timedelta(hours=100)

        val_0, _ = schedule.value_at(ts_0)
        val_100, _ = schedule.value_at(ts_100)

        assert val_0 == pytest.approx(1000.0, abs=1.0)
        assert val_100 == pytest.approx(11000.0, abs=1.0)  # 1000 + 100*100

    def test_linear_with_very_large_negative_slope(self):
        """Test linear with very large negative slope"""
        start_ts = pd.Timestamp("2024-01-01")
        rng = np.random.default_rng(42)

        segments = [
            {
                "name": "rapid_decline",
                "days": 100,
                "dist": {"kind": "linear", "start": 50000.0, "slope": -100.0},
                "transition_hours": 0,
            }
        ]

        schedule = RegimeSchedule(
            varname="test_var",
            start_ts=start_ts,
            freq="h",
            segments=segments,
            rng=rng,
            series_map={},
        )

        # Check values decline rapidly
        ts_0 = start_ts
        ts_100 = start_ts + pd.Timedelta(hours=100)

        val_0, _ = schedule.value_at(ts_0)
        val_100, _ = schedule.value_at(ts_100)

        assert val_0 == pytest.approx(50000.0, abs=1.0)
        assert val_100 == pytest.approx(40000.0, abs=1.0)  # 50000 - 100*100

    def test_linear_with_negative_starting_value(self):
        """Test linear distribution starting at negative value"""
        start_ts = pd.Timestamp("2024-01-01")
        rng = np.random.default_rng(42)

        segments = [
            {
                "name": "from_negative",
                "days": 10,
                "dist": {"kind": "linear", "start": -100.0, "slope": 5.0},
                "transition_hours": 0,
            }
        ]

        schedule = RegimeSchedule(
            varname="test_var",
            start_ts=start_ts,
            freq="h",
            segments=segments,
            rng=rng,
            series_map={},
        )

        ts_0 = start_ts
        ts_50 = start_ts + pd.Timedelta(hours=50)

        val_0, _ = schedule.value_at(ts_0)
        val_50, _ = schedule.value_at(ts_50)

        assert val_0 == pytest.approx(-100.0, abs=0.1)
        assert val_50 == pytest.approx(150.0, abs=0.1)  # -100 + 5*50

    def test_linear_bounds_violation_with_large_slope(self):
        """Test that bounds are enforced even with large slope"""
        start_ts = pd.Timestamp("2024-01-01")
        rng = np.random.default_rng(42)

        segments = [
            {
                "name": "rapid_bounded",
                "days": 10,
                "dist": {
                    "kind": "linear",
                    "start": 0.0,
                    "slope": 1000.0,  # Very large
                    "bounds": {"low": -100.0, "high": 500.0},
                },
                "transition_hours": 0,
            }
        ]

        schedule = RegimeSchedule(
            varname="test_var",
            start_ts=start_ts,
            freq="h",
            segments=segments,
            rng=rng,
            series_map={},
        )

        # Even with large slope, should respect bounds
        for hour in [0, 1, 2, 5, 10, 20, 50, 100]:
            ts = start_ts + pd.Timedelta(hours=hour)
            val, _ = schedule.value_at(ts)
            assert -100.0 <= val <= 500.0, f"Hour {hour}: value {val} violates bounds"

    def test_linear_numerical_precision_with_small_slope(self):
        """Test numerical precision with very small slope"""
        start_ts = pd.Timestamp("2024-01-01")
        rng = np.random.default_rng(42)

        segments = [
            {
                "name": "tiny_growth",
                "days": 1000,
                "dist": {"kind": "linear", "start": 5000.0, "slope": 0.0001},
                "transition_hours": 0,
            }
        ]

        schedule = RegimeSchedule(
            varname="test_var",
            start_ts=start_ts,
            freq="h",
            segments=segments,
            rng=rng,
            series_map={},
        )

        # Over 1000 hours, should increase by 0.1
        ts_0 = start_ts
        ts_1000 = start_ts + pd.Timedelta(hours=1000)

        val_0, _ = schedule.value_at(ts_0)
        val_1000, _ = schedule.value_at(ts_1000)

        assert val_0 == pytest.approx(5000.0, abs=0.01)
        assert val_1000 == pytest.approx(5000.1, abs=0.01)
        assert val_1000 > val_0, "Should still detect tiny increase"
