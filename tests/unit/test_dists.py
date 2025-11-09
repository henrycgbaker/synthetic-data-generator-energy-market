"""
Unit tests for dists module.
Tests distribution sampling functions.
"""

import numpy as np
import pandas as pd
import pytest
from synthetic_data_pkg.dists import _clamp, empirical_at, iid_sample, stateful_step


@pytest.mark.unit
class TestClamp:
    """Test clamping utility"""

    def test_clamp_no_bounds(self):
        """Test clamp with no bounds returns original value"""
        assert _clamp(5.0, None) == 5.0
        assert _clamp(-100.0, None) == -100.0

    def test_clamp_lower_bound(self):
        """Test clamping to lower bound"""
        bounds = {"low": 0.0, "high": 100.0}
        assert _clamp(-5.0, bounds) == 0.0
        assert _clamp(50.0, bounds) == 50.0

    def test_clamp_upper_bound(self):
        """Test clamping to upper bound"""
        bounds = {"low": 0.0, "high": 100.0}
        assert _clamp(150.0, bounds) == 100.0
        assert _clamp(50.0, bounds) == 50.0

    def test_clamp_only_lower(self):
        """Test clamp with only lower bound"""
        bounds = {"low": 10.0}
        assert _clamp(5.0, bounds) == 10.0
        assert _clamp(50.0, bounds) == 50.0

    def test_clamp_only_upper(self):
        """Test clamp with only upper bound"""
        bounds = {"high": 100.0}
        assert _clamp(150.0, bounds) == 100.0
        assert _clamp(50.0, bounds) == 50.0


@pytest.mark.unit
class TestIIDSample:
    """Test IID sampling distributions"""

    def test_const_distribution(self, rng):
        """Test constant distribution"""
        spec = {"kind": "const", "v": 42.0}
        samples = [iid_sample(rng, spec) for _ in range(100)]
        assert all(s == 42.0 for s in samples)

    def test_uniform_distribution(self, rng):
        """Test uniform distribution"""
        spec = {"kind": "uniform", "min": 10.0, "max": 20.0}
        samples = [iid_sample(rng, spec) for _ in range(1000)]

        assert all(10.0 <= s <= 20.0 for s in samples)
        assert min(samples) < 12.0  # Should explore lower range
        assert max(samples) > 18.0  # Should explore upper range

    def test_normal_distribution(self, rng):
        """Test normal distribution"""
        spec = {"kind": "normal", "mu": 50.0, "sigma": 10.0}
        samples = [iid_sample(rng, spec) for _ in range(1000)]

        # Check approximate mean and std
        assert np.mean(samples) == pytest.approx(50.0, abs=5.0)
        assert np.std(samples) == pytest.approx(10.0, abs=3.0)

    def test_normal_with_bounds(self, rng):
        """Test normal distribution with bounds"""
        spec = {
            "kind": "normal",
            "mu": 50.0,
            "sigma": 10.0,
            "bounds": {"low": 30.0, "high": 70.0},
        }
        samples = [iid_sample(rng, spec) for _ in range(1000)]

        assert all(30.0 <= s <= 70.0 for s in samples)

    def test_beta_distribution(self, rng):
        """Test beta distribution"""
        spec = {
            "kind": "beta",
            "alpha": 2,
            "beta": 5,
            "low": 0.5,
            "high": 1.0,
        }
        samples = [iid_sample(rng, spec) for _ in range(1000)]

        assert all(0.5 <= s <= 1.0 for s in samples)

    def test_lognormal_distribution(self, rng):
        """Test lognormal distribution"""
        spec = {"kind": "lognormal", "mu": 0.0, "sigma": 1.0}
        samples = [iid_sample(rng, spec) for _ in range(1000)]

        # Lognormal should be positive
        assert all(s > 0 for s in samples)

    def test_truncnormal_distribution(self, rng):
        """Test truncated normal distribution"""
        spec = {
            "kind": "truncnormal",
            "mu": 50.0,
            "sigma": 10.0,
            "low": 40.0,
            "high": 60.0,
        }
        samples = [iid_sample(rng, spec) for _ in range(1000)]

        assert all(40.0 <= s <= 60.0 for s in samples)

    def test_unsupported_distribution(self, rng):
        """Test that unsupported distribution raises error"""
        spec = {"kind": "unsupported"}
        with pytest.raises(ValueError, match="Unsupported iid dist"):
            iid_sample(rng, spec)


@pytest.mark.unit
class TestStatefulStep:
    """Test stateful (time-dependent) distributions"""

    def test_ar1_initialization(self, rng):
        """Test AR1 process initialization"""
        spec = {"kind": "ar1", "mu": 50.0, "sigma": 5.0, "phi": 0.8}

        # First step should be near mu
        first_val = stateful_step(rng, prev=None, spec=spec)
        assert first_val == pytest.approx(50.0, abs=20.0)

    def test_ar1_persistence(self, rng):
        """Test AR1 process has persistence"""
        spec = {"kind": "ar1", "mu": 50.0, "sigma": 1.0, "phi": 0.95}

        vals = [50.0]  # Start at mean
        for _ in range(100):
            vals.append(stateful_step(rng, prev=vals[-1], spec=spec))

        # High phi means values should stay close together
        diffs = np.abs(np.diff(vals))
        assert np.mean(diffs) < 5.0  # Should have small changes

    def test_ar1_mean_reversion(self, rng):
        """Test AR1 process reverts to mean"""
        spec = {"kind": "ar1", "mu": 50.0, "sigma": 5.0, "phi": 0.7}

        # Start far from mean
        vals = [100.0]
        for _ in range(500):  # More steps for better convergence
            vals.append(stateful_step(rng, prev=vals[-1], spec=spec))

        # Should trend back towards mean
        assert vals[-1] < vals[0], "Should trend back from 100 towards 50"
        
        # Average of last 100 values should be close to mean (tighter bounds)
        mean_last_100 = np.mean(vals[-100:])
        assert 42.0 < mean_last_100 < 58.0, f"Mean {mean_last_100} should be within ±8 of 50"

    def test_random_walk_initialization(self, rng):
        """Test random walk initialization"""
        spec = {"kind": "rw", "start": 100.0, "drift": 0.0, "sigma": 5.0}

        first_val = stateful_step(rng, prev=None, spec=spec)
        assert first_val == pytest.approx(100.0, abs=20.0)

    def test_random_walk_with_drift(self, rng):
        """Test random walk with positive drift"""
        spec = {"kind": "rw", "start": 50.0, "drift": 1.0, "sigma": 0.1}

        vals = [stateful_step(rng, prev=None, spec=spec)]
        for _ in range(50):
            vals.append(stateful_step(rng, prev=vals[-1], spec=spec))

        # With positive drift, should trend upward
        assert vals[-1] > vals[0]

    def test_linear_deterministic(self, rng):
        """Test linear (deterministic) growth"""
        spec = {"kind": "linear", "start": 10.0, "slope": 2.0}

        vals = [stateful_step(rng, prev=None, spec=spec)]
        for _ in range(10):
            vals.append(stateful_step(rng, prev=vals[-1], spec=spec))

        # Should be perfectly linear: 10, 12, 14, 16, ...
        expected = [10.0 + 2.0 * i for i in range(11)]
        np.testing.assert_array_almost_equal(vals, expected)

    def test_stateful_with_bounds(self, rng):
        """Test stateful distributions respect bounds"""
        spec = {
            "kind": "ar1",
            "mu": 50.0,
            "sigma": 20.0,
            "phi": 0.9,
            "bounds": {"low": 30.0, "high": 70.0},
        }

        vals = [50.0]
        for _ in range(100):
            vals.append(stateful_step(rng, prev=vals[-1], spec=spec))

        assert all(30.0 <= v <= 70.0 for v in vals)

    def test_unsupported_stateful_distribution(self, rng):
        """Test that unsupported stateful distribution raises error"""
        spec = {"kind": "unsupported"}
        with pytest.raises(ValueError, match="Unsupported stateful dist"):
            stateful_step(rng, prev=None, spec=spec)


@pytest.mark.unit
class TestEmpiricalAt:
    """Test empirical series lookup"""

    def test_empirical_level(self):
        """Test empirical lookup with level transform"""
        dates = pd.date_range("2024-01-01", periods=24, freq="h")
        series = pd.Series(range(24), index=dates)
        series_map = {"test_series": series}

        spec = {"kind": "empirical", "name": "test_series", "transform": "level"}

        ts = pd.Timestamp("2024-01-01 10:00")
        val = empirical_at(series_map, ts, spec)

        assert val == 10.0

    def test_empirical_diff(self):
        """Test empirical lookup with diff transform"""
        dates = pd.date_range("2024-01-01", periods=24, freq="h")
        series = pd.Series(range(24), index=dates)
        series_map = {"test_series": series}

        spec = {"kind": "empirical", "name": "test_series", "transform": "diff"}

        ts = pd.Timestamp("2024-01-01 10:00")
        val = empirical_at(series_map, ts, spec)

        # Diff should be 1.0 (since series is [0,1,2,3,...])
        assert val == pytest.approx(1.0)

    def test_empirical_pct_change(self):
        """Test empirical lookup with pct_change transform"""
        dates = pd.date_range("2024-01-01", periods=24, freq="h")
        series = pd.Series([100, 110, 121], index=dates[:3])
        series_map = {"test_series": series}

        spec = {"kind": "empirical", "name": "test_series", "transform": "pct_change"}

        ts = pd.Timestamp("2024-01-01 01:00")
        val = empirical_at(series_map, ts, spec)

        # 110/100 - 1 = 0.1 (10% increase)
        assert val == pytest.approx(0.1)

    def test_empirical_missing_series(self):
        """Test empirical lookup with missing series raises error"""
        series_map = {}
        spec = {"kind": "empirical", "name": "missing_series"}

        ts = pd.Timestamp("2024-01-01 10:00")

        with pytest.raises(KeyError, match="missing_series"):
            empirical_at(series_map, ts, spec)

    def test_empirical_with_bounds(self):
        """Test empirical lookup respects bounds"""
        dates = pd.date_range("2024-01-01", periods=24, freq="h")
        series = pd.Series(range(24), index=dates)
        series_map = {"test_series": series}

        spec = {
            "kind": "empirical",
            "name": "test_series",
            "transform": "level",
            "bounds": {"low": 5.0, "high": 15.0},
        }

        ts_low = pd.Timestamp("2024-01-01 02:00")  # Would be 2
        ts_high = pd.Timestamp("2024-01-01 20:00")  # Would be 20

        val_low = empirical_at(series_map, ts_low, spec)
        val_high = empirical_at(series_map, ts_high, spec)

        assert val_low == 5.0  # Clamped from 2
        assert val_high == 15.0  # Clamped from 20
