#!/usr/bin/env python3
"""
Run Scenario 1: Gas Crisis

Models a temporary spike in gas prices from May to September,
showing how market prices respond to fuel cost shocks.
"""

import os
import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from synthetic_data_pkg.runner import execute_scenario

if __name__ == "__main__":
    print("=" * 70)
    print("SCENARIO 1: GAS CRISIS")
    print("=" * 70)
    print("\nThis scenario models:")
    print("  • Normal gas prices (~$30/unit) Jan-Apr")
    print("  • Crisis gas prices (~$85/unit) May-Aug")
    print("  • Recovery gas prices (~$35/unit) Sep-Dec")
    print("  • Coal prices also elevated during crisis")
    print("\nExpected observations:")
    print("  • Market prices spike during gas crisis")
    print("  • Gas generation decreases, coal increases")
    print("  • Demand response to higher prices")
    print("\n" + "=" * 70 + "\n")

    config_path = Path(__file__).resolve().parent / "configs" / "1_gas_crisis.yaml"

    try:
        paths = execute_scenario(str(config_path))

        print("\n" + "=" * 70)
        print("SCENARIO 1 COMPLETE")
        print("=" * 70)
        print("\nKey outputs:")
        if paths:
            for key, path in paths.items():
                print(f"  • {key}: {path}")
        print("=" * 70 + "\n")

    except Exception as e:
        print("\n❌ ERROR: Scenario failed")
        print(f"   {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
