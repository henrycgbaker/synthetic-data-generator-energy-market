# Synthetic Energy Market Data Generator

A Python package for generating realistic synthetic electricity market time series data with regime shifts. This tool simulates wholesale power markets by modeling supply curves (across different generation technologies), demand curves, and computing market equilibrium prices and quantities over time.

## Features

- **Multi-technology supply modeling**: Nuclear, coal, gas, wind, and solar generation
- **Regime-based evolution**: Parameters can shift over time with smooth transitions between regimes
- **Weather-driven renewables**: AR(1) wind model and sinusoidal solar patterns
- **Flexible demand models**: Elastic (price-responsive) or inelastic demand with daily/seasonal patterns
- **Realistic market dynamics**: Merit-order dispatch, planned outages, efficiency curves
- **Configurable scenarios**: YAML-based configuration for easy scenario definition

## Installation

### Prerequisites

- Python 3.9 or higher
- pip

### Install from Source

1. Clone the repository:
```bash
git clone https://github.com/henrycgbaker/synthetic-data-generator-energy-market.git
cd <repo path>
```

2. Install the package in editable mode with development dependencies:
```bash
pip install -e ".[dev]"
```

This installs the package and all dependencies, including testing and linting tools.

### Verify Installation

```bash
# Check that the CLI is available
generate --help

# Run the test suite
make test
```

## Quick Start

### Running a Scenario

The easiest way to get started is to run one of the pre-configured scenarios:

```bash
# use the CLI directly (recommended)
generate run configs/1_gas_crisis.yaml

# or run the gas crisis scenario (1 year simulation)
python run_scenario1_gas_crisis.py
```

This will generate synthetic market data and save it to the `outputs/` directory.

### Output Files

By default, scenarios generate:
- **CSV file**: Time series data with prices, quantities, and generation dispatch
- **Pickle file**: Pandas DataFrame for easy loading in Python
- **Metadata JSON**: Configuration snapshot and run information

Output files are saved in `outputs/` with timestamps and version numbers.

## Configuration Guide

### Configuration Structure

Scenarios are defined using YAML configuration files. Here's the structure:

```yaml
# Time parameters
start_ts: "2024-01-01 00:00"
days: 365
freq: "h"  # hourly timesteps
seed: 42

# Demand curve parameters
demand:
  inelastic: false          # true = fixed quantity, false = price-responsive
  base_intercept: 45.0      # Choke price ($/MWh)
  slope: -7.0               # Price elasticity ($/MW)
  daily_seasonality: true   # Daily peak/off-peak patterns
  annual_seasonality: true  # Winter/summer variations

# Supply regime planner
supply_regime_planner:
  mode: "local_only"  # Options: "local_only", "global", "hybrid"

# Variable definitions (fuel prices, capacities, availabilities, etc.)
variables:
  fuel.gas:
    regimes:
      - name: "normal"
        dist: {kind: "normal", mu: 30.0, sigma: 5.0}
        breakpoints:
          - date: "2024-01-01"
            transition_hours: 48
  # ... more variables

# Output settings
io:
  out_dir: "outputs"
  dataset_name: "my_scenario"
  save_csv: true
  save_pickle: true
```

### Variable Types

Variables follow a dot-notation naming convention:

| Variable Pattern | Description | Example |
|-----------------|-------------|---------|
| `fuel.<tech>` | Fuel prices ($/unit) | `fuel.gas`, `fuel.coal` |
| `cap.<tech>` | Installed capacity (MW) | `cap.nuclear`, `cap.wind` |
| `avail.<tech>` | Availability factor (0-1) | `avail.coal`, `avail.gas` |
| `eta_lb.<tech>`, `eta_ub.<tech>` | Thermal efficiency bounds | `eta_lb.gas`, `eta_ub.coal` |
| `bid.<tech>.min`, `bid.<tech>.max` | Bid price ranges | `bid.nuclear.min` |

**Required variables**: `fuel.coal` and `fuel.gas` must always be defined.

### Distribution Types

Variables are sampled from distributions within each regime:

- **`const`**: Constant value
  ```yaml
  dist: {kind: "const", v: 5000.0}
  ```

- **`normal`**: Normal distribution with optional bounds
  ```yaml
  dist: {kind: "normal", mu: 30.0, sigma: 5.0, bounds: {low: 10.0, high: 100.0}}
  ```

- **`beta`**: Beta distribution scaled to range
  ```yaml
  dist: {kind: "beta", alpha: 30, beta: 2, low: 0.90, high: 0.98}
  ```

- **`uniform`**: Uniform distribution
  ```yaml
  dist: {kind: "uniform", low: 20.0, high: 40.0}
  ```

- **`linear`**: Linear change over time (for capacity transitions)
  ```yaml
  dist: {kind: "linear", start: 8000.0, slope: -0.114, bounds: {low: 0.0, high: 8000.0}}
  ```

### Regime Modes

Three modes control how regime transitions are coordinated:

1. **`local_only`**: Each variable defines its own breakpoints independently
2. **`global`**: All variables transition together at synchronized breakpoints
3. **`hybrid`**: Global synchronization by default, but variables can override locally

### Renewable Availability Modes

Two options for modeling wind and solar availability:

1. **`weather_simulation`** (default): Use built-in weather models
   - Wind: AR(1) autoregressive model with persistence
   - Solar: Sinusoidal daily pattern

2. **`direct`**: Sample availability directly from regime distributions
   - Requires `avail.wind` and `avail.solar` in variables

### Example Scenarios

Check the `configs/` directory for examples:

- **`1_gas_crisis.yaml`**: Gas price spike scenario (365 days)
- **`2_coal_phaseout.yaml`**: Gradual coal retirement over 5 years
- **`_base_template.yaml`**: Fully documented template with all options

## Package Architecture

### Core Modules

| Module | Purpose |
|--------|---------|
| **`runner.py`** | Main orchestration pipeline: config → schedules → simulation → output |
| **`config.py`** | Pydantic configuration schemas with validation |
| **`scenario.py`** | Builds regime schedules for all variables, handles breakpoint logic |
| **`regimes.py`** | `RegimeSchedule` class manages transitions and sampling |
| **`simulate.py`** | Time series simulation loop, finds equilibrium at each timestep |
| **`supply.py`** | `SupplyCurve` class builds merit-order supply curves |
| **`demand.py`** | `DemandCurve` class models elastic/inelastic demand |
| **`dists.py`** | Distribution sampling (normal, beta, uniform, linear, empirical) |
| **`io.py`** | Config loading, data saving, empirical series loading |
| **`utils.py`** | Helper functions (linear ramps, random partitions, etc.) |
| **`cli.py`** | Command-line interface entry point |

### How It Works

1. **Load Configuration**: Parse YAML config and validate with Pydantic schemas
2. **Build Regime Schedules**: Create timeline of regime transitions for each variable
3. **Simulate Time Series**: For each hour:
   - Sample random values from current regimes
   - Build supply curve (merit order by marginal cost)
   - Build demand curve (with seasonality adjustments)
   - Solve for equilibrium price and quantity
   - Record prices, quantities, and generation dispatch
4. **Save Outputs**: Write results to configured formats (CSV, pickle, parquet, etc.)

### Supply Curve Construction

The supply curve uses merit-order dispatch:

1. **Nuclear**: Baseload with negative/low bids (always dispatched first)
2. **Wind & Solar**: Near-zero marginal cost (renewable priority)
3. **Coal**: Marginal cost = `fuel_price / efficiency`
4. **Gas**: Marginal cost = `fuel_price / efficiency` (typically highest)

Each technology contributes `capacity × availability` MW to the curve.

### Equilibrium Solver

Market clearing uses Brent's method to find the price where:
- **Elastic demand**: Supply price = Demand price
- **Inelastic demand**: Supply quantity = Fixed demand quantity

Edge cases (demand too low/high) are handled by clamping to price grid bounds.

## Development

### Running Tests

```bash
# All tests (excluding slow)
make test

# Specific test categories
make test-unit           # Unit tests only
make test-integration    # Integration tests
make test-functional     # End-to-end workflow tests

# With coverage report
make test-coverage

# Single test file
pytest tests/unit/test_demand.py

# Single test function
pytest tests/unit/test_demand.py::test_demand_curve_elastic
```

### Code Quality

```bash
# Format code (black + isort)
make format

# Lint code (ruff + mypy)
make lint

# Clean build artifacts
make clean
```

### Test Organization

- **`tests/unit/`**: Test individual functions/classes in isolation
- **`tests/integration/`**: Test multiple components together
- **`tests/functional/`**: Test complete workflows end-to-end

Fixtures are centralized in `tests/conftest.py`.

## Advanced Usage

### Using Empirical Data

You can override sampled data with empirical time series:

```yaml
empirical_series:
  fuel.gas: "path/to/gas_prices.csv"
  avail.wind: "path/to/wind_generation.csv"
```

CSV format: two columns (timestamp, value) or single column (implicit hourly index).

### Planned Outages

Model seasonal maintenance:

```yaml
planned_outages:
  enabled: true
  months: [5, 6, 7, 8, 9]  # May-September
  nuclear_reduction: 0.12   # 12% capacity reduction
  coal_reduction: 0.10
  gas_reduction: 0.08
```

### Multi-Stage Transitions

Use multiple regimes with breakpoints for complex scenarios:

```yaml
cap.coal:
  regimes:
    - name: "stage1"
      dist: {kind: "linear", start: 8000.0, slope: -0.114}
      breakpoints:
        - {date: "2024-01-01", transition_hours: 168}
    - name: "stage2"
      dist: {kind: "linear", start: 7000.0, slope: -0.228}
      breakpoints:
        - {date: "2025-01-01", transition_hours: 168}
    # ... more stages
```

### Price Grid Configuration

Control the price range for equilibrium search:

```yaml
price_grid: [-100, -50, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 300, 500, 1000]
```

Finer grids improve accuracy but increase computation time.

## Troubleshooting

### Common Issues

**Package installation fails**
- Ensure Python 3.9+: `python --version`
- Try upgrading pip: `pip install --upgrade pip`

**Equilibrium not found**
- Widen price grid bounds
- Check demand parameters (intercept, slope)
- Verify supply capacity is sufficient

**Slow simulation**
- Reduce `days` for testing
- Use coarser price grid
- Disable coverage during testing

**Config validation errors**
- Check required variables: `fuel.coal`, `fuel.gas`
- Verify all capacity/availability/efficiency variables are defined
- Check breakpoint date formats: `"YYYY-MM-DD"`

## Contributing

Contributions welcome! Please:
1. Run tests before submitting: `make test`
2. Format code: `make format`
3. Check linting: `make lint`
4. Add tests for new features