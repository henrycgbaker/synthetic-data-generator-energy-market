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

The package provides a `generate` command-line tool. Use `generate run` followed by the path to a configuration file:

```bash
# RECOMMENDED: CLI 

# Run a pre-configured scenario (recommended way to get started)
generate run configs/1_gas_crisis.yaml

# The command accepts relative paths from your current directory
generate run configs/2_coal_phaseout.yaml

# Or absolute paths
generate run /path/to/my/custom_scenario.yaml
```
```bash
# Alternative: use the runner scripts (BUT inferior UI to CLI)
python runners/run_scenario1_gas_crisis.py
python runners/run_scenario2_coal_phaseout.py

# Run all scenarios at once
python runners/run_all_scenarios.py
```

This will generate synthetic market data and save it to the `outputs/` directory.

**Key Command Format:** `generate run <config_file_path>`

### Output Files

By default, scenarios generate:
- **CSV file**: Time series data with prices, quantities, and generation dispatch
- **Pickle file**: Pandas DataFrame for easy loading in Python
- **Metadata JSON**: Configuration snapshot and run information
- More formats available (see `~/configs/_base_config`)

Output files are saved in `outputs/` with timestamps and version numbers.

## Configuration Guide

All scenarios are defined using YAML configuration files. This section provides a comprehensive reference for all available configuration options.

### Top-Level Parameters

```yaml
# Simulation time parameters
start_ts: "2024-01-01 00:00"  # Start timestamp (YYYY-MM-DD HH:MM format)
days: 365                      # Simulation duration in days
freq: "h"                      # Timestep frequency ("h" for hourly)
seed: 42                       # Random seed for reproducibility

# Price grid for equilibrium solver
price_grid: [-100, -50, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 300]
# Can be specified as a list or range
# Finer grids = more accurate equilibrium but slower computation
# Must span expected price range (renewables can have negative bids)
```

### Demand Configuration

The `demand` section controls the demand curve and seasonality patterns:

```yaml
demand:
  # === DEMAND CURVE TYPE ===
  inelastic: false
  # false = Elastic demand (price-responsive, downward-sloping)
  # true  = Inelastic demand (fixed quantity regardless of price)
  # ⚠️  IMPORTANT: When true, slope parameter is IGNORED

  # === ELASTIC DEMAND PARAMETERS (used only if inelastic=false) ===
  base_intercept: 45.0  # Choke price: price when quantity = 0 ($/MWh)
  slope: -7.0           # dP/dQ ($/MW) - must be negative for downward slope
  # Inverse demand equation: P = base_intercept + slope * Q
  # Example: P = 45 - 7*Q means at Q=0, P=$45/MWh

  # === INELASTIC DEMAND PARAMETER (used only if inelastic=true) ===
  # When inelastic=true, base_intercept defines the FIXED QUANTITY level
  # Seasonality multipliers still apply to this fixed quantity

  # === DAILY SEASONALITY ===
  daily_seasonality: true  # Enable/disable daily peak/off-peak patterns
  # The following parameters are IGNORED if daily_seasonality=false:
  day_peak_hour: 14        # Hour of daily peak (0-23, default 14 = 2pm)
  day_amp: 0.25            # Peak amplitude (0-1, e.g., 0.25 = ±25% variation)
  weekend_drop: 0.10       # Weekend reduction factor (0-1, e.g., 0.10 = 10% lower)
  # Pattern: Cosine wave centered on day_peak_hour
  # Weekends (Sat/Sun) reduced by weekend_drop factor

  # === ANNUAL SEASONALITY ===
  annual_seasonality: true  # Enable/disable winter/summer patterns
  # The following parameters are IGNORED if annual_seasonality=false:
  winter_amp: 0.15          # Winter increase (Dec-Feb), e.g., 0.15 = +15%
  summer_amp: -0.10         # Summer decrease (Jun-Aug), e.g., -0.10 = -10%
  # Pattern: Smooth cosine interpolation between winter peak and summer trough
```

**Demand Curve Behavior:**
- **Elastic** (`inelastic=false`): Standard inverse demand `P = intercept + slope * Q`
  - Higher prices → lower quantity demanded
  - Used for equilibrium finding where supply price = demand price
- **Inelastic** (`inelastic=true`): Vertical demand curve at fixed quantity
  - Quantity doesn't respond to price
  - `base_intercept` defines the fixed quantity level (not price!)
  - Equilibrium finds price where supply meets this fixed quantity

### Supply Regime Planner

Controls how regime transitions are coordinated across variables:

```yaml
supply_regime_planner:
  mode: "local_only"  # Options: "local_only", "global", "hybrid"
  # Mode determines regime coordination strategy

  # global_settings REQUIRED for "global" and "hybrid" modes
  # global_settings MUST NOT be present for "local_only" mode
  global_settings:  # Omit this entire section for local_only mode
    n_regimes: 3           # Number of regimes for all variables
    sync_regimes: true     # All variables transition simultaneously

    # === BREAKPOINT SPECIFICATION (choose ONE approach) ===

    # Option 1: Explicit breakpoints (dates specified manually)
    breakpoints:
      - date: "2024-01-01"
        transition_hours: 48   # Transition window BEFORE this date
      - date: "2024-05-01"
        transition_hours: 168  # 7-day transition
      - date: "2024-09-01"
        transition_hours: 72

    # Option 2: Stochastic breakpoints (randomly generated)
    stochastic_breakpoints:
      enabled: true
      min_segment_days: 30   # Minimum regime duration
      max_segment_days: 180  # Maximum regime duration
      transition_hours:
        type: "fixed"        # "fixed" or "range"
        value: 168          # Hours (if type="fixed")
        # OR for type="range":
        # min: 24
        # max: 336

    # Distribution templates (used when variables don't specify distributions)
    distribution_templates:
      fuel.gas: {kind: "normal", mu: 30.0, sigma: 5.0}
      fuel.coal: {kind: "normal", mu: 25.0, sigma: 3.0}
      # Template applied to all regimes for these variables
```

**Mode Behavior:**

1. **`local_only`**: Each variable is independent
   - Each variable specifies its own regimes with breakpoints
   - Variables can have different numbers of regimes
   - Variables transition at different times
   - `global_settings` must NOT be present

2. **`global`**: All variables synchronized
   - All variables share the same regime breakpoints
   - `global_settings.breakpoints` or `global_settings.stochastic_breakpoints` required
   - Variables only specify distributions (override templates)
   - If variable has no local distributions, uses `distribution_templates`

3. **`hybrid`**: Global by default, local override allowed
   - Default: uses global breakpoints like "global" mode
   - Override: variables with local breakpoints use their own schedule
   - Flexible mix of synchronized and independent variables

### Variable Definitions

All market variables are defined in the `variables` section. **Required variables:** `fuel.coal` and `fuel.gas` must always be defined.

#### Variable Naming Convention

| Variable Pattern | Description | Example |
|-----------------|-------------|---------|
| `fuel.<tech>` | Fuel prices ($/unit) | `fuel.gas`, `fuel.coal` |
| `cap.<tech>` | Installed capacity (MW) | `cap.nuclear`, `cap.wind` |
| `avail.<tech>` | Availability factor (0-1) | `avail.coal`, `avail.gas` |
| `eta_lb.<tech>` | Lower efficiency bound | `eta_lb.gas` = 0.48 |
| `eta_ub.<tech>` | Upper efficiency bound | `eta_ub.coal` = 0.38 |
| `bid.<tech>.min` | Minimum bid price | `bid.nuclear.min` = -200 |
| `bid.<tech>.max` | Maximum bid price | `bid.wind.max` = -50 |

#### Technologies

Supported technologies: `nuclear`, `coal`, `gas`, `wind`, `solar`

#### Variable Structure

```yaml
variables:
  # Example: Fuel price with regime transitions
  fuel.gas:
    regimes:
      - name: "normal"              # Regime name (for tracking)
        dist:                        # Distribution specification
          kind: "normal"
          mu: 30.0
          sigma: 5.0
          bounds: {low: 10.0, high: 100.0}  # Optional bounds
        breakpoints:                 # Optional local breakpoints
          - date: "2024-01-01"      # Regime starts on this date
            transition_hours: 48     # Smooth transition window (hours BEFORE date)

      - name: "crisis"
        dist: {kind: "normal", mu: 85.0, sigma: 15.0}
        breakpoints:
          - date: "2024-05-01"
            transition_hours: 168

  # Example: Constant capacity
  cap.nuclear:
    regimes:
      - name: "constant"
        dist: {kind: "const", v: 6000.0}  # 6000 MW constant

  # Example: Linear capacity change
  cap.coal:
    regimes:
      - name: "declining"
        dist:
          kind: "linear"
          start: 8000.0       # Starting value (MW)
          slope: -0.114       # Change per hour (MW/hr)
          bounds: {low: 0.0, high: 8000.0}
        breakpoints:
          - {date: "2024-01-01", transition_hours: 168}
```

**Breakpoints Behavior:**
- `transition_hours`: Smooth blend window BEFORE the breakpoint date
- Example: date="2024-05-01", transition_hours=168 (7 days)
  - Regime 1 runs pure until 2024-04-24
  - 2024-04-24 to 2024-05-01: gradual blend from Regime 1 → Regime 2
  - Regime 2 runs pure from 2024-05-01 onward

### Distribution Types

Each variable samples values from a distribution. Available types:

#### 1. Constant (`const`)
Fixed value, no randomness.

```yaml
dist:
  kind: "const"
  v: 6000.0  # The constant value
```
**Use for**: Capacities that don't change, fixed parameters

#### 2. Normal (`normal`)
Gaussian distribution with optional bounds.

```yaml
dist:
  kind: "normal"
  mu: 30.0           # Mean
  sigma: 5.0         # Standard deviation
  bounds:            # Optional: clamp values to range
    low: 10.0
    high: 100.0
```
**Use for**: Fuel prices, availability factors with natural variation

#### 3. Beta (`beta`)
Beta distribution scaled to a specific range.

```yaml
dist:
  kind: "beta"
  alpha: 30      # Shape parameter (higher = more concentrated)
  beta: 2        # Shape parameter
  low: 0.85      # Minimum value (scale lower bound)
  high: 0.95     # Maximum value (scale upper bound)
```
**Use for**: Availability factors (naturally bounded 0-1), efficiency values

#### 4. Uniform (`uniform`)
Uniform distribution over a range.

```yaml
dist:
  kind: "uniform"
  low: 20.0      # Minimum value
  high: 40.0     # Maximum value
```
**Use for**: When all values in range are equally likely

#### 5. Linear (`linear`)
Deterministic linear change over time (NOT random).

```yaml
dist:
  kind: "linear"
  start: 8000.0       # Starting value at regime start
  slope: -0.114       # Change per hour (can be negative)
  bounds:             # Optional: clamp to range
    low: 0.0
    high: 8000.0
```
**Use for**: Gradual capacity changes, phase-outs, ramp-ups
**Note**: Value = `start + slope * hours_from_regime_start`

#### 6. Empirical (`empirical`)
Load values from empirical time series data.

```yaml
dist:
  kind: "empirical"
  series_name: "historical_gas_prices"  # Must match key in empirical_series
```
**Requires**: Corresponding entry in `empirical_series` section (see below)

### Renewable Availability Modes

Controls how wind and solar availability is modeled. This is a **top-level parameter**:

```yaml
renewable_availability_mode: "weather_simulation"  # or "direct"
```

#### Mode 1: `weather_simulation` (Recommended)

Uses built-in weather models to generate realistic patterns:

```yaml
renewable_availability_mode: "weather_simulation"

weather_simulation:
  wind:
    model: "ar1"  # AR(1) autoregressive process
    params:
      base_capacity_factor: 0.45  # Mean capacity factor (0-1)
      persistence: 0.85           # Autocorrelation (0-1, higher = more persistent)
      volatility: 0.15            # Hourly noise std dev

  solar:
    model: "sinusoidal"  # Deterministic daily pattern
    params:
      sunrise_hour: 6              # Hour sun rises (0-23)
      sunset_hour: 20              # Hour sun sets (0-23)
      peak_capacity_factor: 0.35  # Peak output (midday)
```

**When to use**: Default choice for realistic renewable patterns
**Behavior**:
- Wind: Persistent multi-day patterns with hourly fluctuations
- Solar: Zero at night, sinusoidal curve during day
**Note**: `avail.wind` and `avail.solar` NOT needed in `variables` section

#### Mode 2: `direct`

Sample wind/solar availability directly from distributions:

```yaml
renewable_availability_mode: "direct"

variables:
  avail.wind:
    regimes:
      - name: "variable"
        dist: {kind: "beta", alpha: 20, beta: 5, low: 0.2, high: 0.8}

  avail.solar:
    regimes:
      - name: "variable"
        dist: {kind: "beta", alpha: 15, beta: 10, low: 0.0, high: 0.6}
```

**When to use**: Custom renewable patterns, specific distributions
**Requires**: `avail.wind` and `avail.solar` must be defined in `variables` or `empirical_series`

### Planned Outages

Models seasonal maintenance outages for thermal and nuclear plants:

```yaml
planned_outages:
  enabled: true      # Set false to disable outages entirely

  months: [5, 6, 7, 8, 9]  # Months with outages (1=Jan, 12=Dec)
  # Typical: May-September (summer maintenance)

  # Reduction factors applied to availability during outage months
  nuclear_reduction: 0.12   # 12% capacity reduction (e.g., 0.95 → 0.836)
  coal_reduction: 0.10      # 10% reduction
  gas_reduction: 0.08       # 8% reduction
```

**Behavior**: During specified months, availability factors are multiplied by `(1 - reduction)`:
- Example: If `avail.nuclear = 0.95` and `nuclear_reduction = 0.12`
- During outage months: `avail.nuclear_effective = 0.95 * (1 - 0.12) = 0.836`
- Other months: `avail.nuclear_effective = 0.95` (unchanged)

**Impact on Wind/Solar**: NOT affected by planned outages (only `nuclear`, `coal`, `gas`)

### Empirical Series

Load real-world time series data instead of sampling from distributions:

```yaml
empirical_series:
  # Map variable name -> path to CSV file
  fuel.gas: "data/historical_gas_prices.csv"
  avail.wind: "data/wind_generation_2023.csv"
  # Paths relative to config file location
```

**CSV Format Requirements:**
- **Two-column format**: `timestamp, value`
  - Column names: `ts`/`time`/`timestamp`/`datetime` for time, anything else for value
- **Single-column format**: Just values (assumes implicit hourly index starting 2020-01-01)

**Example CSV:**
```csv
timestamp,value
2024-01-01 00:00,28.50
2024-01-01 01:00,29.20
2024-01-01 02:00,28.80
```

**Behavior**: When a variable uses empirical data, it OVERRIDES any regime distributions for that variable.

### Output Configuration

The `io` section controls output formats and locations:

```yaml
io:
  # === OUTPUT LOCATION ===
  out_dir: "outputs"               # Directory for output files
  dataset_name: "my_scenario"      # Base name for files
  version: "v1"                    # Version tag (appears in filename)

  # === FILENAME OPTIONS ===
  add_timestamp: true              # Append timestamp to filename
  timestamp_fmt: "%Y_%m_%d_%H_%M"  # Timestamp format (Python strftime)
  # Result: my_scenario_v1_2024_11_06_14_30.csv

  # === OUTPUT FORMATS (all default to false except pickle) ===
  save_pickle: true       # Python pickle (.pkl) - fastest, Python-only
  save_csv: true          # CSV (.csv) - universal, large files
  save_parquet: false     # Parquet (.parquet) - compressed, columnar
  save_feather: false     # Feather (.feather) - fast, cross-language
  save_excel: false       # Excel (.xlsx) - slow, row limit ~1M
  save_preview_html: false  # HTML preview (first N rows)
  save_meta: false        # JSON metadata (config snapshot)
  save_head_csv: false    # CSV with first N rows only

  head_rows: 200          # Rows for preview/head files (if enabled)
```

**Output Filename Pattern:**
```
{out_dir}/{dataset_name}_{version}_{timestamp}.{ext}
```

**Performance Tips:**
- Use `pickle` for speed and Python compatibility
- Use `parquet` for compressed storage and cross-language use
- Avoid `excel` for large datasets (>1M rows)
- Enable `save_meta` to preserve full config for reproducibility

### Complete Configuration Example

Here's a complete, annotated configuration showing all major features:

```yaml
# === TOP-LEVEL PARAMETERS ===
start_ts: "2024-01-01 00:00"
days: 365
freq: "h"
seed: 42
price_grid: [-200, -100, -50, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300]

# === DEMAND ===
demand:
  inelastic: false
  base_intercept: 200.0  # P = 200 - 0.006*Q
  slope: -0.006
  daily_seasonality: true
  day_peak_hour: 14
  day_amp: 0.25
  weekend_drop: 0.10
  annual_seasonality: true
  winter_amp: 0.15
  summer_amp: -0.10

# === REGIME PLANNING ===
supply_regime_planner:
  mode: "local_only"  # Each variable has independent regimes

# === VARIABLES ===
variables:
  # Required: Fuel prices with regime transitions
  fuel.gas:
    regimes:
      - name: "normal"
        dist: {kind: "normal", mu: 30.0, sigma: 5.0, bounds: {low: 10.0, high: 100.0}}
        breakpoints: [{date: "2024-01-01", transition_hours: 48}]
      - name: "crisis"
        dist: {kind: "normal", mu: 85.0, sigma: 15.0, bounds: {low: 40.0, high: 150.0}}
        breakpoints: [{date: "2024-05-01", transition_hours: 168}]
      - name: "recovery"
        dist: {kind: "normal", mu: 35.0, sigma: 6.0, bounds: {low: 15.0, high: 100.0}}
        breakpoints: [{date: "2024-09-01", transition_hours: 168}]

  fuel.coal:
    regimes:
      - name: "stable"
        dist: {kind: "normal", mu: 25.0, sigma: 3.0, bounds: {low: 15.0, high: 60.0}}

  # Capacities (constant or changing)
  cap.nuclear:
    regimes:
      - {name: "constant", dist: {kind: "const", v: 6000.0}}

  cap.coal:
    regimes:
      - name: "declining"
        dist: {kind: "linear", start: 8000.0, slope: -0.114, bounds: {low: 0.0, high: 8000.0}}

  cap.gas:
    regimes:
      - {name: "constant", dist: {kind: "const", v: 12000.0}}

  cap.wind:
    regimes:
      - {name: "constant", dist: {kind: "const", v: 7000.0}}

  cap.solar:
    regimes:
      - {name: "constant", dist: {kind: "const", v: 5000.0}}

  # Availabilities (beta distributions for realism)
  avail.nuclear:
    regimes:
      - name: "baseline"
        dist: {kind: "beta", alpha: 30, beta: 2, low: 0.90, high: 0.98}

  avail.coal:
    regimes:
      - name: "baseline"
        dist: {kind: "beta", alpha: 25, beta: 3, low: 0.85, high: 0.95}

  avail.gas:
    regimes:
      - name: "baseline"
        dist: {kind: "beta", alpha: 28, beta: 2, low: 0.90, high: 0.98}

  # Thermal efficiencies
  eta_lb.coal:
    regimes:
      - {name: "baseline", dist: {kind: "const", v: 0.33}}
  eta_ub.coal:
    regimes:
      - {name: "baseline", dist: {kind: "const", v: 0.38}}
  eta_lb.gas:
    regimes:
      - {name: "baseline", dist: {kind: "const", v: 0.48}}
  eta_ub.gas:
    regimes:
      - {name: "baseline", dist: {kind: "const", v: 0.55}}

  # Renewable bids (negative for priority dispatch)
  bid.nuclear.min:
    regimes:
      - {name: "baseline", dist: {kind: "const", v: -200.0}}
  bid.nuclear.max:
    regimes:
      - {name: "baseline", dist: {kind: "const", v: -50.0}}
  bid.wind.min:
    regimes:
      - {name: "baseline", dist: {kind: "const", v: -200.0}}
  bid.wind.max:
    regimes:
      - {name: "baseline", dist: {kind: "const", v: -50.0}}
  bid.solar.min:
    regimes:
      - {name: "baseline", dist: {kind: "const", v: -200.0}}
  bid.solar.max:
    regimes:
      - {name: "baseline", dist: {kind: "const", v: -50.0}}

# === RENEWABLE AVAILABILITY ===
renewable_availability_mode: "weather_simulation"
weather_simulation:
  wind:
    model: "ar1"
    params:
      base_capacity_factor: 0.45
      persistence: 0.85
      volatility: 0.15
  solar:
    model: "sinusoidal"
    params:
      sunrise_hour: 6
      sunset_hour: 20
      peak_capacity_factor: 0.35

# === PLANNED OUTAGES ===
planned_outages:
  enabled: true
  months: [5, 6, 7, 8, 9]
  nuclear_reduction: 0.12
  coal_reduction: 0.10
  gas_reduction: 0.08

# === EMPIRICAL DATA (optional) ===
empirical_series: {}  # Empty if not using empirical data

# === OUTPUT ===
io:
  out_dir: "outputs"
  dataset_name: "gas_crisis_scenario"
  version: "v1"
  add_timestamp: true
  save_pickle: true
  save_csv: true
  save_meta: true
```

### Common Configuration Patterns

#### Pattern 1: Simple Constant Scenario
Minimal config with constant parameters (no regime changes):

```yaml
supply_regime_planner:
  mode: "local_only"

variables:
  fuel.gas:
    regimes:
      - {name: "constant", dist: {kind: "const", v: 30.0}}
  fuel.coal:
    regimes:
      - {name: "constant", dist: {kind: "const", v: 25.0}}
  # ... all other variables with single constant regime
```

#### Pattern 2: Synchronized Regime Changes
All variables transition together using global mode:

```yaml
supply_regime_planner:
  mode: "global"
  global_settings:
    n_regimes: 3
    breakpoints:
      - {date: "2024-01-01", transition_hours: 48}
      - {date: "2024-05-01", transition_hours: 168}
      - {date: "2024-09-01", transition_hours: 168}
    distribution_templates:
      fuel.gas: {kind: "normal", mu: 30.0, sigma: 5.0}
      fuel.coal: {kind: "normal", mu: 25.0, sigma: 3.0}

variables:
  # Only override distributions if different from template
  fuel.gas:
    regimes:
      - {name: "low", dist: {kind: "normal", mu: 25.0, sigma: 4.0}}
      - {name: "normal", dist: {kind: "normal", mu: 30.0, sigma: 5.0}}
      - {name: "high", dist: {kind: "normal", mu: 40.0, sigma: 6.0}}
```

#### Pattern 3: Mix of Synchronized and Independent
Use hybrid mode for flexibility:

```yaml
supply_regime_planner:
  mode: "hybrid"
  global_settings:
    n_regimes: 2
    breakpoints:
      - {date: "2024-01-01", transition_hours: 48}
      - {date: "2024-07-01", transition_hours: 168}

variables:
  # These use global breakpoints
  fuel.gas:
    regimes:
      - {name: "low", dist: {kind: "normal", mu: 25.0, sigma: 4.0}}
      - {name: "high", dist: {kind: "normal", mu: 50.0, sigma: 8.0}}

  # This overrides with local breakpoints
  cap.coal:
    regimes:
      - name: "stage1"
        dist: {kind: "linear", start: 8000.0, slope: -0.1}
        breakpoints: [{date: "2024-01-01", transition_hours: 168}]
      - name: "stage2"
        dist: {kind: "linear", start: 7000.0, slope: -0.2}
        breakpoints: [{date: "2024-04-01", transition_hours: 168}]
```

### Configuration Validation & Troubleshooting

**Common Errors and Solutions:**

1. **"Missing required variable specs: ['fuel.coal', 'fuel.gas']"**
   - **Cause**: Required fuel price variables not defined
   - **Solution**: Always include `fuel.coal` and `fuel.gas` in `variables` section

2. **"mode='global' requires global_settings to be specified"**
   - **Cause**: Using global/hybrid mode without `global_settings`
   - **Solution**: Add `global_settings` block with `n_regimes` and `breakpoints`

3. **"direct mode requires ['avail.wind', 'avail.solar'] in variables or empirical_series"**
   - **Cause**: Using `renewable_availability_mode: "direct"` without renewable availability data
   - **Solution**: Either add `avail.wind`/`avail.solar` to `variables`, or switch to `weather_simulation` mode

4. **Equilibrium not found / prices at grid boundaries**
   - **Cause**: Price grid doesn't span actual price range
   - **Solution**: Widen `price_grid` bounds (e.g., add -500, -1000 for renewable bids; 500, 1000 for scarcity)

5. **Inelastic demand produces unexpected prices**
   - **Cause**: Misunderstanding of `base_intercept` meaning
   - **Solution**: When `inelastic=true`, `base_intercept` is the QUANTITY level, not price
   - Seasonality multipliers apply: `effective_demand = base_intercept * daily_mult * annual_mult`

6. **Regime transitions too abrupt**
   - **Cause**: `transition_hours` too small
   - **Solution**: Increase `transition_hours` (e.g., 168 = 7 days for smooth blend)

7. **Variables have different number of regimes**
   - **Cause**: Expected in `local_only` mode; error in `global` mode
   - **Solution**:
     - `local_only`: This is normal behavior
     - `global`: All variables must have same number of regimes (set by `n_regimes`)
     - `hybrid`: Variables without local breakpoints use global; others independent

### Example Scenarios

Check the `configs/` directory for working examples:

- **`1_gas_crisis.yaml`**: Gas price spike scenario (365 days)
- **`2_coal_phaseout.yaml`**: Gradual coal retirement over 5 years
- **`_base_template.yaml`**: Fully documented template with all options
- **`example_global_mode.yaml`**: Global regime synchronization
- **`example_hybrid_mode.yaml`**: Hybrid mode with selective overrides
- **`example_direct_renewable_avail_mode.yaml`**: Direct renewable sampling

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