#!/usr/bin/env python3
"""
Run Scenario 2: Coal Phase-Out

Models gradual reduction in coal capacity with gas and renewables ramping up,
showing long-term capacity transition effects on market dynamics over 5 years.
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
    logger.info("SCENARIO 2: COAL PHASE-OUT (5 YEARS)")
    logger.info("=" * 70)
    logger.info("")
    logger.info("This scenario models:")
    logger.info("  • Coal capacity declining over 5 years: 8000 → 0 MW")
    logger.info("  • Gas capacity increasing: 12000 → 18000 MW")
    logger.info("  • Wind & solar buildout accelerating over time")
    logger.info("  • Coal availability degrading as plants age")
    logger.info("  • Stable fuel prices throughout")
    logger.info("")
    logger.info("Expected observations:")
    logger.info("  • Gas generation replaces coal over time")
    logger.info("  • Renewable penetration increases")
    logger.info("  • Market prices may rise slightly (gas more expensive)")
    logger.info("  • System reliability maintained during transition")
    logger.info("=" * 70)

    config_path = Path(__file__).resolve().parent.parent / "configs" / "2_coal_phaseout.yaml"

    try:
        paths = execute_scenario(str(config_path))

        logger.info("")
        logger.info("=" * 70)
        logger.info("SCENARIO 2 COMPLETE")
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
