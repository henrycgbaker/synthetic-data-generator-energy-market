"""
this script is the user interface / CLI entry point for the package
"""

# synthetic_data/synthetic_data_pkg/cli.py

from __future__ import annotations

import typer

from .runner import execute_scenario

# CLI root:
app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command("run")  # allows: `supplycurves run CONFIG`
def run_cmd(config: str = typer.Argument(..., help="Path to YAML/JSON config file")):
    """Run a simulation from a config."""
    execute_scenario(config)


@app.callback(invoke_without_command=True)  # allows: `supplycurves --config CONFIG` with no subcommand
def main(
    ctx: typer.Context,
    config: str = typer.Option(None, "--config", "-c", help="Path to YAML/JSON config file"),
    ):
    """Allow: `supplycurves --config CONFIG` with no subcommand."""
    if ctx.invoked_subcommand is None:
        if not config:
            raise typer.Exit(code=0)  # help already shown due to no_args_is_help=True
        execute_scenario(config)


def entrypoint():  # how pkg becomes CLI tool
    app()


if __name__ == "__main__":
    entrypoint()
