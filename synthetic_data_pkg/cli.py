"""
This module provides the CLI interface for the synthetic data generator.

Usage:
    synth-data generate <config_path>          # Run simulation with config file
    synth-data run <config_path>               # Alternative command name
"""

from __future__ import annotations

import typer

from .runner import execute_scenario

# CLI root application
app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode=None,  # Disable rich output to fix Python 3.13 compatibility
    help="""Synthetic Energy Market Data Generator

Generate realistic electricity market time series with:
- Multiple generation technologies (nuclear, coal, gas, wind, solar)
- Regime-based parameter evolution with smooth transitions
- Weather-driven renewable availability
- Elastic/inelastic demand models
- Planned outage modeling

For more information: https://github.com/henrycgbaker/synthetic-data-generator-energy-market
""",
)


@app.command("generate", help="Run a market simulation from a configuration file")
def generate_cmd(
    config: str = typer.Argument(
        ...,
        help="Relative or absolute path to YAML/JSON config file (e.g., configs/1_gas_crisis.yaml)",
    )
):
    """
    Run a market simulation from a configuration file.

    Examples:
        synth-data generate configs/1_gas_crisis.yaml
        synth-data generate /path/to/my_scenario.yaml
    """
    execute_scenario(config)


@app.command("run", help="Run a market simulation from a configuration file")
def run_cmd(
    config: str = typer.Argument(
        ...,
        help="Relative or absolute path to YAML/JSON config file (e.g., configs/1_gas_crisis.yaml)",
    )
):
    """
    Run a market simulation from a configuration file.

    Examples:
        synth-data run configs/1_gas_crisis.yaml
        synth-data run /path/to/my_scenario.yaml
    """
    execute_scenario(config)


def entrypoint():
    """Entry point for the CLI tool."""
    app()


if __name__ == "__main__":
    entrypoint()
