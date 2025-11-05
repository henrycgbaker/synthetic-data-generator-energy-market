#!/usr/bin/env python3
"""
Run all three scenarios and capture outputs
"""

import subprocess
import sys
import time
from pathlib import Path


def run_scenario(config_path, scenario_name):
    """Run a single scenario and report results"""
    print(f"\n{'=' * 60}")
    print(f"Running: {scenario_name}")
    print(f"Config: {config_path}")
    print(f"{'=' * 60}\n")

    start_time = time.time()

    try:
        result = subprocess.run(
            ["supplycurves", "run", config_path],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        elapsed = time.time() - start_time

        print(result.stdout)

        if result.returncode == 0:
            print(f"\n✅ {scenario_name} completed successfully in {elapsed:.1f}s")
            return True
        else:
            print(f"\n❌ {scenario_name} failed!")
            print(f"Error output:\n{result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"\n❌ {scenario_name} timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"\n❌ {scenario_name} error: {e}")
        return False


def main():
    """Run all scenarios"""

    # Get to synthetic_data directory
    base_dir = Path(__file__).parent

    scenarios = [
        ("configs/1_gas_crisis.yaml", "Scenario 1: Gas Crisis"),
        ("configs/2_coal_phaseout.yaml", "Scenario 2: Coal Phase-out"),
        #("configs/3_full_seasonality.yaml", "Scenario 3: Full Seasonality"),
    ]

    results = {}

    print("\n" + "=" * 60)
    print("SUPPLY CURVES SCENARIO VALIDATION")
    print("=" * 60)
    print(f"\nRunning {len(scenarios)} scenarios...\n")

    for config_path, name in scenarios:
        success = run_scenario(config_path, name)
        results[name] = success

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")

    total = len(results)
    passed = sum(results.values())

    print(f"\nTotal: {passed}/{total} scenarios passed")

    if passed == total:
        print("\n🎉 All scenarios completed successfully!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} scenario(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
