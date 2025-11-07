"""
This module provides the CLI interface for the synthetic data generator.

Usage:
    generate run <config_path>              # Run simulation with config file
    generate run configs/my_scenario.yaml   # Example with relative path
    generate --config <config_path>         # Alternative syntax
"""

from __future__ import annotations

import typer

from .runner import execute_scenario

# CLI root application
app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Synthetic Energy Market Data Generator w/ Regime Changes - model electricity market time series with specied regime changes"
)


@app.command("run")
def run_cmd(
    config: str = typer.Argument(
        ...,
        help="Relative or absolute path to YAML/JSON config file (e.g., configs/1_gas_crisis.yaml)",
        metavar="CONFIG_PATH"
    )
):
    """
    Run a market simulation from a configuration file.

    Examples:
        generate run configs/1_gas_crisis.yaml
        generate run /path/to/my_scenario.yaml
        generate run my_scenario  # Auto-detects .yaml extension
    """
    execute_scenario(config)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    config: str = typer.Option(
        None,
        "--config", "-c",
        help="Path to YAML/JSON config file (alternative to 'run' subcommand)"
    ),
):
    """
    Synthetic Energy Market Data Generator

    Generate realistic electricity market time series with:
    - Multiple generation technologies (nuclear, coal, gas, wind, solar)
    - Regime-based parameter evolution with smooth transitions
    - Weather-driven renewable availability
    - Elastic/inelastic demand models
    - Planned outage modeling

    For more information and examples, see:
        https://github.com/henrycgbaker/synthetic-data-generator-energy-market
    """
    if ctx.invoked_subcommand is None:
        if not config:
            raise typer.Exit(code=0)  # help already shown due to no_args_is_help=True
        execute_scenario(config)


def entrypoint():  # how pkg becomes CLI tool
    app()


if __name__ == "__main__":
    entrypoint()
