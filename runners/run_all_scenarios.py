#!/usr/bin/env python3
"""
Run all configured scenarios sequentially and report results.

This script runs each scenario config through the CLI and captures results,
providing a comprehensive validation of all example scenarios.
"""

import logging
import subprocess
import sys
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def run_scenario(config_path: str, scenario_name: str) -> bool:
    """
    Run a single scenario and report results.

    Args:
        config_path: Relative path to config file
        scenario_name: Human-readable scenario name

    Returns:
        True if scenario completed successfully, False otherwise
    """
    logger.info("=" * 60)
    logger.info(f"Running: {scenario_name}")
    logger.info(f"Config: {config_path}")
    logger.info("=" * 60)

    start_time = time.time()

    try:
        result = subprocess.run(
            ["generate", "run", config_path],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        elapsed = time.time() - start_time

        # Print stdout from the scenario
        if result.stdout:
            print(result.stdout)

        if result.returncode == 0:
            logger.info(f"{scenario_name} completed successfully in {elapsed:.1f}s")
            return True
        else:
            logger.error(f"{scenario_name} failed!")
            if result.stderr:
                logger.error(f"Error output:\n{result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"{scenario_name} timed out after 10 minutes")
        return False
    except FileNotFoundError:
        logger.error("'generate' command not found. Make sure the package is installed:")
        logger.error("  pip install -e .[dev]")
        return False
    except Exception as e:
        logger.error(f"{scenario_name} error: {e}")
        return False


def main():
    """Run all scenarios and report summary."""

    scenarios = [
        ("configs/1_gas_crisis.yaml", "Scenario 1: Gas Crisis"),
        ("configs/2_coal_phaseout.yaml", "Scenario 2: Coal Phase-Out"),
    ]

    results = {}

    logger.info("")
    logger.info("=" * 60)
    logger.info("SYNTHETIC DATA GENERATION - SCENARIO VALIDATION")
    logger.info("=" * 60)
    logger.info(f"Running {len(scenarios)} scenarios...")
    logger.info("")

    for config_path, name in scenarios:
        success = run_scenario(config_path, name)
        results[name] = success

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    for name, success in results.items():
        status = "PASS" if success else "FAIL"
        logger.info(f"{status:4s} - {name}")

    total = len(results)
    passed = sum(results.values())

    logger.info(f"\nTotal: {passed}/{total} scenarios passed")

    if passed == total:
        logger.info("All scenarios completed successfully!")
        return 0
    else:
        logger.warning(f"{total - passed} scenario(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
