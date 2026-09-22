from __future__ import annotations

import typer

app = typer.Typer(help="NBA prediction data spine and pipeline.")


@app.callback()
def main() -> None:
    """NBA prediction data spine and pipeline."""


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo("predictor 0.1.0")


if __name__ == "__main__":
    app()
