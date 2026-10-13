"""Typer CLI for the siRNA off-target discovery pipeline."""

from pathlib import Path
from typing import Optional

import typer

from sioff._logging import setup_logging
from sioff.commands import accessibility, off_targets, risearch

# Startup budget: `sioff --help` must only import typer and the command
# signatures. Everything heavy (polars, numpy, pyarrow, omegaconf, loguru, the
# core and services) is imported inside the function that needs it; the test
# `TestCLIStartup` in tests/off-targets/test_cli.py pins this.


app = typer.Typer(
    name="sioff",
    help="siRNA off-target discovery pipeline — analyze RIsearch3 predictions.",
    add_completion=False,
)
app.command(name="accessibility")(accessibility.run)
app.command(name="off-targets")(off_targets.run)
app.command(name="index")(risearch.index)
app.command(name="search")(risearch.search)


def _version_callback(value: bool) -> None:
    """Print the version and exit, before any other option is processed."""
    if value:
        from sioff import __version__

        typer.echo(f"sioff {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML config file. Runs the command specified in the config.",
        exists=True,
        readable=True,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging (DEBUG level).",
    ),
    # Long form only: -v is taken by --verbose. `is_eager` makes this answer
    # before --config is validated, so `sioff --version` works even when a config
    # elsewhere on the command line would fail.
    version: bool = typer.Option(
        False,
        "--version",
        help="Show the installed siOFF version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """
    siRNA off-target discovery pipeline.

    Use 'accessibility' to pre-compute profiles or 'off-targets' to analyze predictions.
    Alternatively, use --config to run from a YAML configuration file.
    """
    setup_logging(verbose)

    if config is not None:
        from sioff.config import config_to_kwargs, load_config

        try:
            cfg = load_config(config)
            kwargs = config_to_kwargs(cfg, cfg.command)

            final_verbose = verbose or cfg.verbose
            setup_logging(final_verbose)
            kwargs["verbose"] = final_verbose

            if cfg.command == "off-targets":
                off_targets.run(**kwargs)
            elif cfg.command == "accessibility":
                accessibility.run(**kwargs)
        except Exception as e:
            typer.echo(f"Error loading config: {e}", err=True)
            raise typer.Exit(code=1)

        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


def entrypoint() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    entrypoint()
