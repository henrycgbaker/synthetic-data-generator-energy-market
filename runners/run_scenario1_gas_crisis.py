#!/usr/bin/env python3
"""
Run Scenario 1: Gas Crisis

Models a temporary spike in gas prices from May to September,
showing how market prices respond to fuel cost shocks.
"""

import logging
import sys
from pathlib import Path

from synthetic_data_pkg.runner import execute_scenario

# Configure logging for the scenario runner
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("SCENARIO 1: GAS CRISIS")
    logger.info("=" * 70)
    logger.info("")
    logger.info("This scenario models:")
    logger.info("  • Normal gas prices (~$30/unit) Jan-Apr")
    logger.info("  • Crisis gas prices (~$85/unit) May-Aug")
    logger.info("  • Recovery gas prices (~$35/unit) Sep-Dec")
    logger.info("  • Coal prices also elevated during crisis")
    logger.info("")
    logger.info("Expected observations:")
    logger.info("  • Market prices spike during gas crisis")
    logger.info("  • Gas generation decreases, coal increases")
    logger.info("  • Demand response to higher prices")
    logger.info("=" * 70)

    config_path = Path(__file__).resolve().parent.parent / "configs" / "1_gas_crisis.yaml"

    try:
        paths = execute_scenario(str(config_path))

        logger.info("")
        logger.info("=" * 70)
        logger.info("SCENARIO 1 COMPLETE")
        logger.info("=" * 70)
        logger.info("")
        logger.info("Key outputs:")
        if paths:
            for key, path in paths.items():
                logger.info(f"  • {key}: {path}")
        logger.info("=" * 70)

    except Exception as e:
        logger.error("")
        logger.error("SCENARIO FAILED")
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)
