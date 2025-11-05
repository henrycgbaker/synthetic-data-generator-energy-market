# Supply Curves Package - Quick Start

Welcome! This guide helps you get started with the synthetic supply curves generator.

## What This Package Does

Generate realistic power market data with:
- **Supply curves** (merit order dispatch)
- **Demand curves** (elastic or inelastic)
- **Regime shifts** (crises, transitions)
- **Seasonality** (winter/summer patterns)
- **Real data integration** (CSV loaders)

## Installation & Setup

```bash
# Navigate to the package directory
cd synthetic_data

# Install in development mode
pip install -e .

# Verify installation
supplycurves --help
```

## Run Your First Simulation

```bash
# Run the gas crisis scenario (1 year simulation)
supplycurves run configs/1_gas_crisis.yaml

# Outputs will be in: synthetic_data/outputs/
```

## Three Ready-to-Use Scenarios

### 1. Gas Crisis (`configs/1_gas_crisis.yaml`)
**Duration:** 1 year
**Features:**
- Gas price spike (crisis period)
- Coal follows with smaller magnitude
- Annual demand seasonality
- Summer planned outages

**Run:** `supplycurves run configs/1_gas_crisis.yaml`

---

### 2. Coal Phase-out (`configs/2_coal_phaseout.yaml`)
**Duration:** 5 years
**Features:**
- Coal capacity decreases to zero
- Wind & solar capacity increases (deterministic buildout)
- Gas expands as backup
- Energy transition dynamics

**Run:** `supplycurves run configs/2_coal_phaseout.yaml`

---

### 3. Full Seasonality (`configs/3_full_seasonality.yaml`)
**Duration:** 2 years
**Features:**
- 4 seasons per year (8 total regimes)
- Seasonal demand, fuel prices, renewable availability
- Summer thermal plant deratings
- All seasonality features combined

**Run:** `supplycurves run configs/3_full_seasonality.yaml`

---

## Key Features

### ✅ Demand Modeling
```yaml
demand:
  base_intercept: 50.0
  slope: -7.5
  inelastic: false           # Toggle vertical demand
  annual_seasonality: true   # Winter/summer variations
  winter_amp: 0.25          # +25% in winter
  summer_amp: -0.15         # -15% in summer
```

### ✅ CSV Data Loaders
```yaml
empirical_series:
  fuel.gas: "data/gas_prices.csv"
  demand: "data/demand_hourly.csv"

variables:
  fuel.gas:
    regimes:
      - name: "historical"
        dist:
          kind: "empirical"
          name: "fuel.gas"
```

### ✅ Deterministic Capacity
```yaml
cap.wind:
  regimes:
    - name: "buildout"
      dist:
        kind: "linear"
        start: 6000.0
        slope: 0.205  # MW per hour
```

### ✅ Per-Variable Regime Timing
```yaml
variables:
  fuel.gas:
    breakpoints: ["2024-01-01", "2024-05-01", "2024-09-01"]
    transition_hours: [48, 72, 48]
    regimes:
      - name: "normal"
        dist: {...}
      - name: "crisis"
        dist: {...}
      - name: "recovery"
        dist: {...}
```

### ✅ Planned Outages
```yaml
planned_outages:
  enabled: true
  months: [5, 6, 7, 8, 9]  # May-September
  nuclear_reduction: 0.15  # 15% capacity reduction
  coal_reduction: 0.12
  gas_reduction: 0.08
```

---

## Working with Outputs

### Load Results in Python

```python
import pandas as pd

# Load pickle (fastest)
df = pd.read_pickle('outputs/scenario_name_v1_2025_10_02_10_14.pkl')

# Or CSV
df = pd.read_csv('outputs/scenario_name_v1_2025_10_02_10_14.csv',
                 index_col=0, parse_dates=True)

# Inspect
print(df.head())
print(df.columns)
```

### Key Output Columns

- `price` - Market clearing price (€/MWh)
- `quantity` - Equilibrium quantity (MW)
- `Q_wind`, `Q_solar`, `Q_nuclear`, `Q_coal`, `Q_gas` - Generation by technology
- `cap.*` - Installed capacity per technology
- `avail.*` - Availability factors per technology
- `fuel.*` - Fuel prices
- `*_regime` - Regime labels per variable

---

## Next Steps

### 1. Explore Example Data
```bash
ls example_data/
# example_gas_prices.csv
# example_demand.csv
# example_wind_generation.csv
# example_solar_generation.csv
```

### 2. Read Documentation
- **`CSV_LOADERS_GUIDE.md`** - Complete CSV loading reference
- **`TODO.md`** - Feature roadmap and status
- **`SPRINT_SUMMARY.md`** - Implementation details

### 3. Create Your Own Scenario
Copy an existing YAML and modify:
```bash
cp configs/1_gas_crisis.yaml configs/my_scenario.yaml
# Edit my_scenario.yaml
supplycurves run configs/my_scenario.yaml
```

### 4. Validate Results
Create a Jupyter notebook to visualize outputs:
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_pickle('outputs/your_scenario.pkl')

# Plot price over time
df['price'].plot(figsize=(14, 5), title='Market Price')
plt.ylabel('Price (€/MWh)')
plt.show()

# Plot generation mix
df[['Q_wind', 'Q_solar', 'Q_nuclear', 'Q_coal', 'Q_gas']].plot(
    kind='area', stacked=True, figsize=(14, 6),
    title='Generation Mix'
)
plt.ylabel('Generation (MW)')
plt.show()
```

---

## Common Tasks

### Change Simulation Length
```yaml
days: 365  # 1 year
days: 1825 # 5 years
```

### Add More Technologies
Define capacity, availability, fuel price, and bid price for new tech.

### Use Real Data
```yaml
empirical_series:
  fuel.gas: "path/to/real_gas_prices.csv"

variables:
  fuel.gas:
    regimes:
      - name: "historical"
        dist:
          kind: "empirical"
          name: "fuel.gas"
```

### Disable Seasonality
```yaml
demand:
  annual_seasonality: false
```

---

## Distribution Types Available

| Type | Use Case | Example |
|------|----------|---------|
| `const` | Fixed values | `{kind: "const", v: 5000}` |
| `normal` | Random variation | `{kind: "normal", mu: 50, sigma: 8}` |
| `beta` | Bounded (0-1) | `{kind: "beta", alpha: 3, beta: 4, low: 0.2, high: 0.8}` |
| `linear` | Deterministic trends | `{kind: "linear", start: 1000, slope: 0.5}` |
| `empirical` | CSV data | `{kind: "empirical", name: "series_key"}` |
| `ar1` | Time series (autoregressive) | `{kind: "ar1", mu: 50, phi: 0.9, sigma: 5}` |
| `rw` | Random walk | `{kind: "rw", drift: 0.1, sigma: 2}` |

See `dists.py` for details.

---

## Troubleshooting

**Error: "Config file not found"**
- Use absolute path or path relative to current directory
- Check file exists: `ls configs/your_file.yaml`

**Error: "Missing required variable specs"**
- Must define `fuel.coal` and `fuel.gas` at minimum
- See example YAMLs for required structure

**Simulation takes too long**
- Reduce `days` for testing
- Use shorter time periods first
- Profile with smaller datasets

**Strange results**
- Check regime breakpoints align with simulation period
- Verify bounds on distributions
- Ensure capacity > demand (roughly)

---

## Support

- **Issues/Bugs:** Check `TODO.md` for known issues
- **Usage Questions:** See `CSV_LOADERS_GUIDE.md`
- **Feature Requests:** Review `TODO.md` roadmap

---

## Quick Reference

```bash
# Run scenario
supplycurves run configs/scenario.yaml

# Or alternative syntax
supplycurves --config configs/scenario.yaml

# Check outputs
ls outputs/

# View latest output
ls -lt outputs/ | head
```

---

**Happy simulating! 🚀**
