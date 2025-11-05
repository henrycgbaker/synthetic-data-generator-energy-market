# CSV Loaders Documentation

This document explains how to use empirical time series data from CSV files in your supply curve simulations.

## Overview

The synthetic data generator supports loading real-world time series data from CSV files. This allows you to:
- Use actual historical fuel prices instead of stochastic distributions
- Load real demand patterns from grid operators
- Import renewable generation profiles from actual wind/solar farms
- Combine empirical data with synthetic data for hybrid scenarios

## CSV File Formats

The loader supports two CSV formats:

### Format 1: Timestamp + Value (Recommended)
```csv
ts,value
2024-01-01 00:00:00,45.2
2024-01-01 01:00:00,48.3
2024-01-01 02:00:00,52.1
...
```

- **Column 1:** Timestamp (can be named: `ts`, `time`, `timestamp`, `date`, or `datetime`)
- **Column 2:** Value (can be named: `value`, `price`, `demand`, `generation`, etc.)
- Timestamps should be in ISO format or parseable by pandas
- Frequency will be inferred (hourly recommended)

### Format 2: Values Only
```csv
value
45.2
48.3
52.1
...
```

- Single column of values
- Implicit hourly frequency starting from 2020-01-01

## Using Empirical Series in YAML

### Step 1: Declare the CSV files

Add an `empirical_series` section to your YAML config:

```yaml
empirical_series:
  fuel.gas: "example_data/example_gas_prices.csv"
  fuel.coal: "data/historical/coal_prices_2023.csv"
  demand: "example_data/example_demand.csv"
  renewable.wind: "example_data/example_wind_generation.csv"
  renewable.solar: "example_data/example_solar_generation.csv"
```

**Path resolution order:**
1. Absolute path
2. Relative to current working directory
3. Relative to config file location
4. Relative to `synthetic_data/` directory

### Step 2: Reference in variable definitions

Use the `empirical` distribution type to reference loaded series:

```yaml
variables:
  fuel.gas:
    regimes:
      - name: "historical"
        dist:
          kind: "empirical"
          name: "fuel.gas"  # Must match key in empirical_series
          transform: "level"  # Options: "level", "pct_change", "diff"
```

## Distribution Type: `empirical`

When using empirical data, specify:

```yaml
dist:
  kind: "empirical"
  name: "series_name"      # Key from empirical_series
  transform: "level"        # Transformation type
  bounds: {low: 0, high: 100}  # Optional clipping
```

### Transform Options

- **`level`** (default): Use the raw values from the CSV
  ```python
  value_at_time = csv_data[timestamp]
  ```

- **`pct_change`**: Use percentage change from previous hour
  ```python
  value = (current / previous) - 1.0
  ```
  Useful for: Price returns, growth rates

- **`diff`**: Use absolute difference from previous hour
  ```python
  value = current - previous
  ```
  Useful for: Demand changes, generation deltas

## Example Scenarios

### Example 1: Historical Gas Prices

```yaml
empirical_series:
  fuel.gas: "data/gas_prices_2023.csv"

variables:
  fuel.gas:
    regimes:
      - name: "historical"
        dist:
          kind: "empirical"
          name: "fuel.gas"
          transform: "level"
          bounds: {low: 10.0, high: 200.0}
```

### Example 2: Real Wind Generation Profile

```yaml
empirical_series:
  wind.profile: "data/wind_farm_output_2023.csv"

variables:
  avail.wind:  # Use as capacity factor
    regimes:
      - name: "empirical_profile"
        dist:
          kind: "empirical"
          name: "wind.profile"
          transform: "level"
          bounds: {low: 0.0, high: 1.0}
```

### Example 3: Hybrid (Empirical + Synthetic)

Mix empirical data with synthetic regimes using regime breakpoints:

```yaml
empirical_series:
  gas.historical: "data/gas_2020_2023.csv"

variables:
  fuel.gas:
    breakpoints:
      - "2020-01-01"  # Historical period
      - "2024-01-01"  # Forecast period
    regimes:
      - name: "historical"
        dist:
          kind: "empirical"
          name: "gas.historical"
          transform: "level"
      - name: "forecast"
        dist:
          kind: "normal"
          mu: 50.0
          sigma: 8.0
          bounds: {low: 20.0, high: 100.0}
```

### Example 4: Demand with Seasonal Scaling

Use empirical demand as a base, scaled by synthetic seasonality:

```yaml
empirical_series:
  base.demand: "data/demand_2023.csv"

# Use annual_seasonality in demand config to scale the empirical data
demand:
  base_intercept: 50.0  # This gets scaled by empirical data
  slope: -7.0
  annual_seasonality: true
  winter_amp: 0.15
  summer_amp: -0.10

# Then reference in a variable if needed, or use demand curve directly
```

## Data Requirements

### Coverage
- CSV data should cover your entire simulation period
- If gaps exist, forward-fill is used (last value repeated)
- For periods before CSV start date: uses first value
- For periods after CSV end date: uses last value

### Frequency
- Hourly data recommended (`freq="h"`)
- Other frequencies will be resampled to hourly using forward-fill
- Ensure data aligns with your simulation start time

### Quality Checks
- Remove or interpolate missing values before loading
- Ensure reasonable bounds (use `bounds` parameter for clipping)
- Check timestamp format is consistent
- Verify timezone handling (UTC recommended)

## Common Patterns

### Pattern 1: All Fuels from CSV
```yaml
empirical_series:
  fuel.gas: "data/gas.csv"
  fuel.coal: "data/coal.csv"
  fuel.oil: "data/oil.csv"

variables:
  fuel.gas:
    regimes:
      - name: "hist"
        dist: {kind: "empirical", name: "fuel.gas", transform: "level"}
  fuel.coal:
    regimes:
      - name: "hist"
        dist: {kind: "empirical", name: "fuel.coal", transform: "level"}
```

### Pattern 2: Renewables from Actual Production
```yaml
empirical_series:
  wind.2023: "data/wind_production_2023.csv"
  solar.2023: "data/solar_production_2023.csv"

variables:
  avail.wind:
    regimes:
      - name: "actual_2023"
        dist:
          kind: "empirical"
          name: "wind.2023"
          transform: "level"
          bounds: {low: 0.0, high: 1.0}
```

### Pattern 3: Crisis Scenario with Empirical Baseline
```yaml
empirical_series:
  gas.normal: "data/gas_2022.csv"

variables:
  fuel.gas:
    breakpoints:
      - "2024-01-01"  # Normal period
      - "2024-05-01"  # Crisis begins
      - "2024-09-01"  # Recovery
    regimes:
      - name: "historical_baseline"
        dist:
          kind: "empirical"
          name: "gas.normal"
          transform: "level"
      - name: "crisis_synthetic"
        dist:
          kind: "normal"
          mu: 120.0  # Crisis spike
          sigma: 20.0
          bounds: {low: 60.0, high: 200.0}
      - name: "recovery_synthetic"
        dist:
          kind: "normal"
          mu: 55.0
          sigma: 10.0
          bounds: {low: 30.0, high: 100.0}
```

## Troubleshooting

### Error: "Empirical series 'X' missing"
- Check that the key in `dist.name` matches exactly the key in `empirical_series`
- Verify the CSV file path is correct and accessible

### Error: "Could not find timestamp column"
- Ensure CSV has a column named `ts`, `time`, `timestamp`, `date`, or `datetime`
- Or use single-column format (values only)

### Data looks wrong / shifted
- Check timestamp format matches your simulation period
- Verify timezone consistency (recommend UTC)
- Use `transform: "level"` for absolute values, not returns/diffs

### Out of memory errors
- Large CSV files (>1M rows) may cause issues
- Consider downsampling data before loading
- Use only the time range needed for your simulation

## Example Files Provided

The `example_data/` directory contains sample CSV files:

- **`example_gas_prices.csv`**: 24 hours of gas prices (€/MWh)
- **`example_demand.csv`**: 24 hours of system demand (MW)
- **`example_wind_generation.csv`**: 24 hours of wind output (MW)
- **`example_solar_generation.csv`**: 24 hours of solar output (MW)

These files demonstrate the expected format and can be used for testing.

## Best Practices

1. **Keep CSVs clean**: No missing values, consistent formatting
2. **Use hourly frequency**: Matches simulation timestep
3. **Include bounds**: Clip unrealistic values from empirical data
4. **Document sources**: Add comments in YAML about data provenance
5. **Test small first**: Validate with short time periods before full runs
6. **Combine thoughtfully**: Mix empirical and synthetic carefully to maintain realism
7. **Version control data**: Track which CSV versions were used for reproducibility

## See Also

- **Distribution types**: See `dists.py` for all available distributions
- **Regime planning**: See `scenario.py` for breakpoint handling
- **Config validation**: See `config.py` for schema details
- **Example scenarios**: See configs `1_gas_crisis.yaml`, `2_coal_phaseout.yaml`, `3_full_seasonality.yaml`
