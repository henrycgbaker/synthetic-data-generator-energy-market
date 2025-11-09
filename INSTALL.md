# Installation Guide

## Prerequisites

- **Python 3.11** or higher
- **Poetry** (for dependency management)

### Install Poetry (if you don't have it)

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

After installation, restart your terminal or add Poetry to your PATH:
```bash
# Linux/macOS
export PATH="$HOME/.local/bin:$PATH"  
```

Verify Poetry is installed:
```bash
poetry --version
```

---

## Installation Methods

Choose the method that fits your workflow:

### Method 1: Poetry + Venv Manager (recommended)

If using conda or another virtual environment manager, Poetry will install into your active environment:

```bash
# 1. Clone the repository
git clone https://github.com/henrycgbaker/synthetic-data-generator-energy-market.git
cd synthetic-data-generator-energy-market

# 2. Create and activate your conda environment
conda create -n synth_data python=3.11
conda activate synth_data

# 3. Install dependencies (Poetry detects active environment)
poetry install

# 4. Verify it works
synth-data --help

# 5. Run a test scenario
synth-data generate configs/1_gas_crisis.yaml
```

### Method 2: Pure Poetry 

This is the **simplest approach** - Poetry creates and manages a virtual environment for you.

```bash
# 1. Clone the repository
git clone https://github.com/henrycgbaker/synthetic-data-generator-energy-market.git
cd synthetic-data-generator-energy-market

# 2. Install dependencies (creates .venv automatically)
poetry install

# 3. Activate the virtual environment
source .venv/bin/activate  # On Linux/macOS
# OR
.venv\Scripts\activate     # On Windows

# 4. Verify it works
synth-data --help
```

**From now on:** Always activate the environment before working:
```bash
source .venv/bin/activate  # Linux/macOS
```

If the using validation notebooks, you will additionally need to install the Poetry environment as a **persistent Jupyter kernel**:
```bash
python -m ipykernel install --user --name=synth-data-generator --display-name="Synthetic Data Generator (Poetry)"
```
---

### Installation Troubleshooting

**"synth-data: command not found"**
- You need to activate the virtual environment first: `source .venv/bin/activate`
- Or use: `poetry run synth-data --help`

---

## Common Issues

### "synth-data: command not found"

**Problem:** The virtual environment is not activated.

**Solutions:**
- **Method 1 users:** Run `source .venv/bin/activate`
- **Method 2 users:** Run `conda activate synth_data`
- **Method 3 users:** Use `poetry run synth-data ...`

### "ImportError: cannot import name 'field_validator'"

**Problem:** Old pydantic version installed.

**Solution:**
```bash
poetry install --sync  # Force reinstall
```

### Poetry not found after installation

**Problem:** Poetry not in PATH.

**Solution:**
```bash
export PATH="$HOME/.local/bin:$PATH"
# Add this line to ~/.bashrc or ~/.zshrc to make it permanent
```

---

## Updating Dependencies

```bash
# Update a specific package
poetry update pydantic

# Update all packages
poetry update

# Add a new dependency
poetry add numpy

# Add a dev dependency
poetry add --group dev pytest
```

---