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
    for feed, result in news_rss.poll_all().items():
        if not result.ok:
            typer.echo(f"{feed}: FAILED - {result.error}")
            continue
        parts = [f"{result.new} new"]
        if result.skipped:
            parts.append(f"{result.skipped} skipped")
        if result.conflicts:
            parts.append(f"{result.conflicts} conflicts")
        line = f"{feed}: {', '.join(parts)}"
        if result.warning:
            line += f" (warning: {result.warning})"
        typer.echo(line)


@app.command("backfill-injuries")
def backfill_injuries(
    start: str = typer.Option("2019-12-01", help="ISO start date."),
    end: str = typer.Option(None, help="ISO end date; defaults to today."),
    hours: str = typer.Option("05PM", help="Comma-separated hour slots."),
) -> None:
    """Backfill archived NBA injury reports into the store."""
    from datetime import date as _date

    from predictor import db
    from predictor.config import settings
    from predictor.sources import injury_report

    settings.ensure_dirs()
    con = db.connect()
    db.migrate(con)

    stats = injury_report.backfill_range(
        con,
        _date.fromisoformat(start),
        _date.fromisoformat(end) if end else _date.today(),
        [h.strip() for h in hours.split(",") if h.strip()],
    )
    for key, value in stats.items():
        typer.echo(f"{key}: {value}")

    transient = stats["transient"]
    parse_failed = stats["parse_failed"]
    bad_content = stats["bad_content"]
    if transient:
        typer.echo(
            f"WARNING: {transient} slot(s) could not be checked (rate-limited "
            "or a server error) and are NOT counted as absent. They were not "
            "archived, so re-running this exact command will retry them "
            "automatically."
        )
    if bad_content:
        typer.echo(
            f"WARNING: {bad_content} slot(s) returned a 200 response with a "
            "body that was not a PDF (for example a CDN block or challenge "
            "page). These are NOT counted as absent -- a real 'not "
            "published' response is a confirmed 403/404, not a 200 with "
            "garbage content. They were not archived, so re-running this "
            "exact command will retry them automatically."
        )
    if parse_failed:
        typer.echo(
            f"WARNING: {parse_failed} slot(s) were fetched but could not be "
            "parsed, so nothing was ingested for them. The raw PDF is "
            "archived on disk, but re-running this command will SKIP them "
            "(already archived) rather than retry ingestion -- run "
            "'predictor reingest-injuries' after fixing the parser to "
            "recover them without re-fetching."
        )
    if transient or parse_failed or bad_content:
        typer.echo(
            f"RUN DID NOT FULLY SUCCEED: {transient + parse_failed + bad_content} "
            "slot(s) out of the requested range are missing from the "
            "database. See the warnings above."
        )
        raise typer.Exit(code=1)


@app.command("reingest-injuries")
def reingest_injuries(
    start: str = typer.Option(None, help="ISO start date filter (inclusive)."),
    end: str = typer.Option(None, help="ISO end date filter (inclusive)."),
) -> None:
    """Re-ingest already-archived injury report PDFs, with no network fetch.

    Recovery path for `parse_failed` slots reported by backfill-injuries:
    once a parser bug is fixed, this re-ingests every already-archived
    report (or, with --start/--end, a date-filtered subset of them)
    straight from the raw store. Safe to re-run -- ingestion is idempotent.
    """
    from datetime import date as _date

    from predictor import db
    from predictor.config import settings
    from predictor.sources import injury_report

    settings.ensure_dirs()
    con = db.connect()
    db.migrate(con)

    stats = injury_report.reingest_archived(
        con,
        _date.fromisoformat(start) if start else None,
        _date.fromisoformat(end) if end else None,
    )

    typer.echo(f"found: {stats['found']} archived injury report(s)")
    typer.echo(f"ingested ok: {stats['ingested_ok']}")
    typer.echo(f"still failed to parse: {stats['still_failed']}")
    typer.echo(f"rows written: {stats['rows_written']}")

    if stats["still_failed"]:
        typer.echo(
            f"WARNING: {stats['still_failed']} archived report(s) still "
            "could not be parsed and were NOT ingested. See the log lines "
            "above for which ones; they need manual attention (or another "
            "parser fix and a re-run of this command)."
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
