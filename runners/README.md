# Example Scenario Runners

This directory contains standalone Python scripts for running pre-configured scenarios.

## Scripts

### Individual Scenarios

- **`run_scenario1_gas_crisis.py`** - Gas price spike scenario (1 year)
- **`run_scenario2_coal_phaseout.py`** - Coal retirement scenario (5 years)

### Batch Runner

- **`run_all_scenarios.py`** - Runs all scenarios sequentially and reports results

## Usage

### Direct Execution

```bash
# Run individual scenarios
python runners/run_scenario1_gas_crisis.py
python runners/run_scenario2_coal_phaseout.py

# Run all scenarios
python runners/run_all_scenarios.py
```

### Alternative: Use the CLI

The recommended way to run scenarios is using the `generate` CLI:

```bash
generate run configs/1_gas_crisis.yaml
generate run configs/2_coal_phaseout.yaml
```

## Features

- **Logging**: All scripts use Python's `logging` module for structured output
- **Error Handling**: Comprehensive exception handling with detailed error messages
- **Progress Tracking**: Real-time progress updates via `tqdm` progress bars
- **Batch Execution**: `run_all_scenarios.py` provides summary statistics

## Customization

To create your own runner script:

```python
import logging
from pathlib import Path
from synthetic_data_pkg.runner import execute_scenario

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config_path = Path(__file__).parent.parent / "configs" / "my_scenario.yaml"

try:
    paths = execute_scenario(str(config_path))
    logger.info(f"Outputs: {paths}")
except Exception as e:
    logger.error(f"Failed: {e}")
    raise
```

## Output Location

All scenarios save outputs to the `outputs/` directory in the repository root.
