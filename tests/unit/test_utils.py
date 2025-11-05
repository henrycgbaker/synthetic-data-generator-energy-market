"""
Unit tests for utils module.
Tests utility functions.
"""

import numpy as np
import pytest
from synthetic_data_pkg.utils import linear_ramp, random_partition, _clamp


@pytest.mark.unit
class TestLinearRamp:
    """Test linear_ramp function"""

    def test_linear_ramp_below_range(self):
        """Test output when price below ramp range"""
        result = linear_ramp(price=10.0, p_low=20.0, p_high=30.0, cap=100.0)
        assert result == 0.0

    def test_linear_ramp_above_range(self):
        """Test output when price above ramp range"""
        result = linear_ramp(price=40.0, p_low=20.0, p_high=30.0, cap=100.0)
        assert result == 100.0

    def test_linear_ramp_at_midpoint(self):
        """Test output at midpoint of ramp range"""
        result = linear_ramp(price=25.0, p_low=20.0, p_high=30.0, cap=100.0)
        assert result == pytest.approx(50.0, abs=0.1)

    def test_linear_ramp_at_boundaries(self):
        """Test output exactly at boundaries"""
        cap = 100.0
        p_low, p_high = 20.0, 30.0
        
        result_low = linear_ramp(price=p_low, p_low=p_low, p_high=p_high, cap=cap)
        result_high = linear_ramp(price=p_high, p_low=p_low, p_high=p_high, cap=cap)
        
        assert result_low == 0.0
        assert result_high == 100.0

    def test_linear_ramp_monotonic(self):
        """Test that output increases monotonically with price"""
        p_low, p_high, cap = 20.0, 30.0, 100.0
        
        prices = [15, 20, 22, 25, 28, 30, 35]
        outputs = [linear_ramp(p, p_low, p_high, cap) for p in prices]
        
        # Check monotonicity
        for i in range(1, len(outputs)):
            assert outputs[i] >= outputs[i-1], f"Not monotonic at index {i}"

    def test_linear_ramp_with_zero_capacity(self):
        """Test with zero capacity"""
        result = linear_ramp(price=25.0, p_low=20.0, p_high=30.0, cap=0.0)
        assert result == 0.0


@pytest.mark.unit
class TestClamp:
    """Test _clamp function"""

    def test_clamp_no_bounds(self):
        """Test clamping with no bounds (should return value unchanged)"""
        assert _clamp(50.0, None) == 50.0
        assert _clamp(-100.0, None) == -100.0
        assert _clamp(1000.0, None) == 1000.0

    def test_clamp_lower_bound(self):
        """Test clamping with lower bound"""
        bounds = {'low': 0.0}
        assert _clamp(-10.0, bounds) == 0.0
        assert _clamp(10.0, bounds) == 10.0
        assert _clamp(0.0, bounds) == 0.0

    def test_clamp_upper_bound(self):
        """Test clamping with upper bound"""
        bounds = {'high': 100.0}
        assert _clamp(150.0, bounds) == 100.0
        assert _clamp(50.0, bounds) == 50.0
        assert _clamp(100.0, bounds) == 100.0

    def test_clamp_both_bounds(self):
        """Test clamping with both bounds"""
        bounds = {'low': 0.0, 'high': 100.0}
        assert _clamp(-10.0, bounds) == 0.0
        assert _clamp(150.0, bounds) == 100.0
        assert _clamp(50.0, bounds) == 50.0
        assert _clamp(0.0, bounds) == 0.0
        assert _clamp(100.0, bounds) == 100.0

    def test_clamp_with_empty_dict(self):
        """Test clamping with empty bounds dict"""
        assert _clamp(50.0, {}) == 50.0


@pytest.mark.unit
class TestRandomPartition:
    """Test random_partition function"""

    def test_random_partition_sum(self):
        """Test that partition sums to total"""
        rng = np.random.default_rng(42)
        total = 365
        partition = random_partition(total, N=5, min_segment=10, rng=rng)
        
        assert sum(partition) == total
        assert len(partition) == 5

    def test_random_partition_min_segment(self):
        """Test that all segments meet minimum size"""
        rng = np.random.default_rng(42)
        min_seg = 10
        partition = random_partition(365, N=5, min_segment=min_seg, rng=rng)
        
        for seg in partition:
            assert seg >= min_seg, f"Segment {seg} below minimum {min_seg}"

    def test_random_partition_all_positive(self):
        """Test that all segments are positive"""
        rng = np.random.default_rng(42)
        partition = random_partition(100, N=10, min_segment=1, rng=rng)
        
        for seg in partition:
            assert seg > 0

    def test_random_partition_single_segment(self):
        """Test with N=1 (single segment)"""
        rng = np.random.default_rng(42)
        partition = random_partition(100, N=1, min_segment=1, rng=rng)
        
        assert len(partition) == 1
        assert partition[0] == 100

    def test_random_partition_reproducible(self):
        """Test that same seed produces same partition"""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        
        partition1 = random_partition(365, N=5, min_segment=10, rng=rng1)
        partition2 = random_partition(365, N=5, min_segment=10, rng=rng2)
        
        assert partition1 == partition2

    def test_random_partition_different_seeds(self):
        """Test that different seeds produce different partitions"""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(123)
        
        partition1 = random_partition(365, N=5, min_segment=10, rng=rng1)
        partition2 = random_partition(365, N=5, min_segment=10, rng=rng2)
        
        # Should be different (with very high probability)
        assert partition1 != partition2
