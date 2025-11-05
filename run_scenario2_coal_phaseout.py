#!/usr/bin/env python3
"""
Run Scenario 2: Coal Phase-Out

Models gradual reduction in coal capacity with gas ramping up to compensate,
showing capacity transition effects on market dynamics.
"""

import os
import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from synthetic_data_pkg.runner import execute_scenario

if __name__ == "__main__":
    print("=" * 70)
    print("SCENARIO 2: COAL PHASE-OUT")
    print("=" * 70)
    print("\nThis scenario models:")
    print("  • Coal capacity declining: 8→5→2 GW (Jan/May/Sep)")
    print("  • Gas capacity increasing: 12→15→18 GW (compensation)")
    print("  • Coal availability degrading as plants age")
    print("  • Stable fuel prices throughout")
    print("\nExpected observations:")
    print("  • Gas generation replaces coal over time")
    print("  • Market prices may rise slightly (gas more expensive)")
    print("  • System reliability maintained during transition")
    print("\n" + "=" * 70 + "\n")

    config_path = Path(__file__).resolve().parent / "configs" / "2_coal_phaseout.yaml"

    try:
        paths = execute_scenario(str(config_path))

        print("\n" + "=" * 70)
        print("SCENARIO 2 COMPLETE")
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
