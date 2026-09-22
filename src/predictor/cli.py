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


@app.command("poll-news")
def poll_news() -> None:
    """Fetch all configured NBA news feeds and archive new items."""
    from predictor.config import settings
    from predictor.sources import news_rss

    settings.ensure_dirs()
    for feed, count in news_rss.poll_all().items():
        state = "FAILED" if count < 0 else f"{count} new"
        typer.echo(f"{feed}: {state}")


if __name__ == "__main__":
    app()
