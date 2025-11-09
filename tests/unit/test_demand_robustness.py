"""
Test demand curve behavior with extreme elasticity parameters.
"""

import numpy as np
import pandas as pd
import pytest
from synthetic_data_pkg.config import DemandConfig
from synthetic_data_pkg.demand import DemandCurve


@pytest.mark.unit
class TestDemandElasticityEdgeCases:
    """Test demand curves with extreme elasticity values"""

    def test_very_elastic_demand(self):
        """Test nearly flat demand curve (very elastic)"""
        # Standard form: P = 200 - 0.0001*Q (nearly horizontal)
        cfg = DemandConfig(
            inelastic=False,
            base_intercept=200.0,
            slope=-0.0001,  # Nearly flat
            daily_seasonality=False,
            annual_seasonality=False,
        )
        demand = DemandCurve(cfg)
        ts = pd.Timestamp("2024-01-01 12:00")

        # Small price change should cause huge quantity change
        q1 = demand.q_at_price(p=100.0, ts=ts)
        q2 = demand.q_at_price(p=99.0, ts=ts)  # 1% price drop

        # Q = (P - 200) / (-0.0001)
        # At P=100: Q = (100-200)/(-0.0001) = 1,000,000
        # At P=99: Q = (99-200)/(-0.0001) = 1,010,000
        assert abs(q1 - 1_000_000) < 1000
        assert abs(q2 - 1_010_000) < 1000
        assert q2 > q1  # Lower price -> higher quantity

    def test_very_inelastic_demand(self):
        """Test nearly vertical demand curve (very inelastic)"""
        # Standard form: P = 200 - 1000*Q (very steep)
        cfg = DemandConfig(
            inelastic=False,
            base_intercept=200.0,
            slope=-1000.0,  # Very steep
            daily_seasonality=False,
            annual_seasonality=False,
        )
        demand = DemandCurve(cfg)
        ts = pd.Timestamp("2024-01-01 12:00")

        # Large price change should cause small quantity change
        q1 = demand.q_at_price(p=150.0, ts=ts)
        q2 = demand.q_at_price(p=100.0, ts=ts)  # 33% price drop

        # Q = (P - 200) / (-1000)
        # At P=150: Q = (150-200)/(-1000) = 0.05
        # At P=100: Q = (100-200)/(-1000) = 0.1
        assert abs(q1 - 0.05) < 0.01
        assert abs(q2 - 0.1) < 0.01
        assert q2 > q1  # Lower price -> higher quantity

    def test_negative_intercept(self):
        """Test demand curve with negative choke price"""
        # Standard form: P = -50 - 0.01*Q
        cfg = DemandConfig(
            inelastic=False,
            base_intercept=-50.0,  # Negative intercept
            slope=-0.01,
            daily_seasonality=False,
            annual_seasonality=False,
        )
        demand = DemandCurve(cfg)
        ts = pd.Timestamp("2024-01-01 12:00")

        # At negative prices, quantity should still be computable
        q = demand.q_at_price(p=-100.0, ts=ts)
        # Q = (-100 - (-50)) / (-0.01) = -50 / -0.01 = 5000
        assert abs(q - 5000) < 10

        # At price above intercept, quantity should be zero
        q_above = demand.q_at_price(p=0.0, ts=ts)
        # Q = (0 - (-50)) / (-0.01) = 50 / -0.01 = -5000 -> clamped to 0
        assert q_above == 0.0

    def test_zero_intercept(self):
        """Test demand curve with zero choke price"""
        # Standard form: P = 0 - 0.01*Q
        cfg = DemandConfig(
            inelastic=False,
            base_intercept=0.0,  # Zero intercept
            slope=-0.01,
            daily_seasonality=False,
            annual_seasonality=False,
        )
        demand = DemandCurve(cfg)
        ts = pd.Timestamp("2024-01-01 12:00")

        # At price = 0, quantity should be zero
        q0 = demand.q_at_price(p=0.0, ts=ts)
        assert q0 == 0.0

        # At negative price, quantity should be positive
        q_neg = demand.q_at_price(p=-10.0, ts=ts)
        # Q = (-10 - 0) / (-0.01) = -10 / -0.01 = 1000
        assert abs(q_neg - 1000) < 10

    def test_small_intercept_steep_slope(self):
        """Test small intercept with steep slope"""
        # Standard form: P = 10 - 100*Q
        cfg = DemandConfig(
            inelastic=False,
            base_intercept=10.0,  # Small
            slope=-100.0,  # Steep
            daily_seasonality=False,
            annual_seasonality=False,
        )
        demand = DemandCurve(cfg)
        ts = pd.Timestamp("2024-01-01 12:00")

        # At P=5: Q = (5-10)/(-100) = -5/-100 = 0.05
        q = demand.q_at_price(p=5.0, ts=ts)
        assert abs(q - 0.05) < 0.01

    def test_large_intercept_flat_slope(self):
        """Test large intercept with flat slope"""
        # Standard form: P = 10000 - 0.001*Q
        cfg = DemandConfig(
            inelastic=False,
            base_intercept=10000.0,  # Large
            slope=-0.001,  # Flat
            daily_seasonality=False,
            annual_seasonality=False,
        )
        demand = DemandCurve(cfg)
        ts = pd.Timestamp("2024-01-01 12:00")

        # At P=5000: Q = (5000-10000)/(-0.001) = -5000/-0.001 = 5,000,000
        q = demand.q_at_price(p=5000.0, ts=ts)
        assert abs(q - 5_000_000) < 1000

    def test_demand_slope_sign_consistency(self):
        """Test that positive slope raises error or behaves correctly"""
        # Demand curves should have negative slopes
        # Test if system handles positive slope gracefully
        cfg = DemandConfig(
            inelastic=False,
            base_intercept=200.0,
            slope=0.01,  # POSITIVE (unusual for demand)
            daily_seasonality=False,
            annual_seasonality=False,
        )
        demand = DemandCurve(cfg)
        ts = pd.Timestamp("2024-01-01 12:00")

        # With positive slope: P = 200 + 0.01*Q
        # Higher price -> higher quantity (supply-like behavior)
        q1 = demand.q_at_price(p=100.0, ts=ts)
        q2 = demand.q_at_price(p=150.0, ts=ts)

        # Q = (P - 200) / 0.01
        # At P=100: Q = (100-200)/0.01 = -10000 -> clamped to 0
        # At P=150: Q = (150-200)/0.01 = -5000 -> clamped to 0
        assert q1 == 0.0
        assert q2 == 0.0

    def test_extreme_elasticity_ratio(self):
        """Test demand curves with extreme elasticity differences"""
        # Very elastic
        cfg_elastic = DemandConfig(
            inelastic=False,
            base_intercept=200.0,
            slope=-0.00001,
            daily_seasonality=False,
            annual_seasonality=False,
        )
        demand_elastic = DemandCurve(cfg_elastic)

        # Very inelastic
        cfg_inelastic = DemandConfig(
            inelastic=False,
            base_intercept=200.0,
            slope=-10000.0,
            daily_seasonality=False,
            annual_seasonivity=False,
        )
        demand_inelastic = DemandCurve(cfg_inelastic)

        ts = pd.Timestamp("2024-01-01 12:00")

        # Compare response to same price change
        price_change = 10.0

        q_elastic_before = demand_elastic.q_at_price(p=100.0, ts=ts)
        q_elastic_after = demand_elastic.q_at_price(p=100.0 - price_change, ts=ts)
        elastic_response = abs(q_elastic_after - q_elastic_before)

        q_inelastic_before = demand_inelastic.q_at_price(p=100.0, ts=ts)
        q_inelastic_after = demand_inelastic.q_at_price(p=100.0 - price_change, ts=ts)
        inelastic_response = abs(q_inelastic_after - q_inelastic_before)

        # Elastic should respond much more strongly
        assert elastic_response > inelastic_response * 1000