from __future__ import annotations

import typer

from predictor.backtest.replay import DEFAULT_BUFFER_MINUTES

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
    """Fetch all configured NBA news feeds, archive new items, and load them."""
    from predictor import db
    from predictor.config import settings
    from predictor.sources import news_rss

    settings.ensure_dirs()
    results = news_rss.poll_all()
    for feed, result in results.items():
        if not result.ok:
            typer.echo(f"{feed}: FAILED - {result.error}")
            continue
        parts = [f"{result.new} new"]
        if result.skipped:
            parts.append(f"{result.skipped} skipped")
        if result.conflicts:
            # "conflicts" here means an identifier's content changed and was
            # archived as a new version -- not that anything was discarded.
            parts.append(f"{result.conflicts} versioned")
        line = f"{feed}: {', '.join(parts)}"
        if result.warning:
            line += f" (warning: {result.warning})"
        typer.echo(line)

    con = db.connect()
    db.migrate(con)
    ingest_stats = news_rss.ingest_archived_news(con)
    line = f"news_items rows: {ingest_stats.written}"
    if ingest_stats.skipped_unknown_feed:
        line += (
            f" ({ingest_stats.skipped_unknown_feed} archived item(s) skipped: "
            "feed not in FEEDS)"
        )
    typer.echo(line)

    failed = [feed for feed, result in results.items() if not result.ok]
    if failed:
        typer.echo(
            f"WARNING: {len(failed)} feed(s) failed this run: "
            f"{', '.join(failed)}. See the FAILED line(s) above for the "
            "reason for each one."
        )
        raise typer.Exit(code=1)


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
    typer.echo(
        f"empty slates (no real filings -- not a failure): "
        f"{stats['empty_no_filings']}"
    )
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


@app.command("ingest-season")
def ingest_season_cmd(season: str = typer.Argument(..., help="e.g. 2024-25")) -> None:
    """Ingest all games for one season."""
    from predictor import db
    from predictor.config import settings
    from predictor.sources import nba_stats

    settings.ensure_dirs()
    con = db.connect()
    db.migrate(con)

    dropped: list[nba_stats.DroppedGame] = []
    count = nba_stats.ingest_season(con, season, dropped=dropped)
    typer.echo(f"ingested {count} games for {season}")

    if dropped:
        game_ids = ", ".join(d.game_id for d in dropped)
        typer.echo(
            f"WARNING: {len(dropped)} game(s) for the {season} season could "
            "NOT be saved to the database, because this tool could not "
            "figure out which team was home and which was away for them. "
            f"The affected game ID(s): {game_ids}. See the lines above "
            "starting with 'nba_stats: DROPPED' for the reason for each "
            "one. This season's data is INCOMPLETE until those games are "
            "fixed and re-ingested."
        )
        raise typer.Exit(code=1)


@app.command("ingest-odds")
def ingest_odds_cmd() -> None:
    """Fetch and store one odds snapshot. Budgeted to one call per run."""
    from predictor import db
    from predictor.config import settings
    from predictor.sources import odds

    settings.ensure_dirs()
    con = db.connect()
    db.migrate(con)
    try:
        typer.echo(f"stored {odds.ingest_current(con)} odds rows")
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None
    except odds.OddsQuotaExceeded as exc:
        typer.echo(f"ODDS QUOTA EXCEEDED: {exc}", err=True)
        raise typer.Exit(code=1) from None


@app.command()
def status() -> None:
    """Report data freshness in plain English."""
    from predictor import db
    from predictor import status as status_mod
    from predictor.config import settings

    settings.ensure_dirs()
    con = db.connect()
    db.migrate(con)
    health = status_mod.check_sources(con)
    typer.echo(status_mod.format_report(health))
    # I5: every OTHER command in this CLI exits 1 on a problem; `status`
    # (the one command whose whole purpose is health reporting) did not,
    # so it could never be wired into an external monitor/cron job to
    # alert on staleness -- it had to be read by a human every time. The
    # output text itself is unchanged; only the exit code is new.
    if any(source.stale for source in health):
        raise typer.Exit(code=1)


@app.command()
def setup() -> None:
    """Create directories and initialise the database."""
    from predictor import db
    from predictor.config import settings

    settings.ensure_dirs()
    db.migrate(db.connect())
    typer.echo(f"ready. data dir: {settings.data_dir}")


@app.command("backtest")
def backtest_cmd(
    model: str = typer.Option(
        "always-home", help="Which predictor to score: always-home or coin-flip."
    ),
    season: str = typer.Option(None, help="Limit to one season, e.g. 2024-25."),
    buffer_minutes: int = typer.Option(
        DEFAULT_BUFFER_MINUTES, help="Minutes before tip-off to cut the data off."
    ),
) -> None:
    """Replay real games and score a predictor on what was knowable pre-tipoff."""
    import duckdb

    from predictor import db
    from predictor import status as status_mod
    from predictor.backtest import baselines, replay, report
    from predictor.config import settings

    known = {
        "always-home": baselines.always_home,
        "coin-flip": baselines.fixed_probability(0.5),
    }
    if model not in known:
        typer.echo(
            f"Unknown model '{model}'. Available: {', '.join(sorted(known))}."
        )
        raise typer.Exit(code=1)

    # FIX 1(c): backtest only ever READS the archive -- it must not migrate
    # (rewrite schema across every real row on every run) or take DuckDB's
    # exclusive write lock (which would collide with the live poll-news
    # job). read_only=True enforces both.
    try:
        con = db.connect(read_only=True)
    except duckdb.Error as exc:
        # FIX 18 (final review, part 3): a lock conflict (the scheduled
        # poll-news job holding the write lock -- routine, not an error)
        # used to be reported identically to a missing database file, and
        # the missing-file remedy ("run predictor ingest-season") needs the
        # SAME write lock, so it sends the user to a command that cannot
        # possibly work either. Both cases raise duckdb.Error (an
        # IOException in both DuckDB's implementation), so the exception
        # TYPE alone cannot distinguish them -- matched on message text
        # instead, verified against a real cross-process lock conflict
        # (see tests/test_backtest_cli.py).
        #
        # FIX 25(a) (final review, part 4): the substring "lock" alone is
        # not specific enough -- a MISSING database whose path happens to
        # contain "lock" (e.g. a data directory named "unlocked-data")
        # would match this branch too, telling a user with no database at
        # all to "wait a moment and re-run" forever. DuckDB's actual lock
        # conflict message (verified above, cross-process) is "Conflicting
        # lock is held in <process> ... by user ..."; match that phrase,
        # not the bare word.
        if "conflicting lock is held" in str(exc).lower():
            typer.echo(
                "Could not open the database -- another 'predictor' command "
                "is using it right now, most likely the scheduled "
                "poll-news job. Wait a moment and try 'predictor backtest' "
                "again."
            )
        else:
            typer.echo(
                "No database found to back-test against. Run an ingest "
                "command first (for example 'predictor ingest-season "
                "<season>'), then try 'predictor backtest' again."
            )
        raise typer.Exit(code=1) from None

    try:
        preds, stats = replay.replay(
            con, known[model], season=season, buffer_minutes=buffer_minutes
        )
    except ValueError as exc:
        typer.echo(f"Cannot run the backtest: {exc}.")
        raise typer.Exit(code=1) from None
    if not preds:
        # FIX 22(c) (final review, part 3): this used to name only 3 of the
        # 9 skip buckets `replay.replay` tracks -- with only those three
        # printed, the numbers it prints could fail to add up to
        # `considered`, silently hiding whatever fell into the other six
        # (conflicting metadata, score missing, result already visible,
        # declined, failed). Every nonzero bucket is now named, so this
        # message always accounts for the full `considered` count.
        buckets = [
            (stats.skipped_conflicting_metadata, "had contradictory metadata across "
             "ingested rows"),
            (stats.skipped_buffer_too_early, "had a buffer reaching back past the "
             "harness's reconstructed schedule timestamp"),
            (stats.skipped_no_tipoff, "had no resolvable tip-off time"),
            (stats.skipped_no_result, "had no result yet (not yet played)"),
            (stats.skipped_score_missing, "were played but the archive did not "
             "record the score"),
            (stats.skipped_result_visible, "had the result already visible at the "
             "cutoff (leak guard)"),
            # FIX 25(d) (final review, part 4): these two used to read
            # "18 the predictor declined to predict" and "12 the predictor
            # failed or returned an impossible probability for" -- neither
            # is a sentence once joined with its count (no verb follows the
            # number). Rewritten as predicates, matching every other bucket
            # above, so each line reads as "<n> <predicate>".
            (stats.declined, "were declined by the predictor"),
            (stats.failed, "made the predictor fail or return an impossible "
             "probability"),
        ]
        detail = "; ".join(f"{n:,} {text}" for n, text in buckets if n)
        message = (
            "No games could be scored -- nothing to measure. "
            f"{stats.considered:,} game(s) were considered"
        )
        message += f": {detail}." if detail else "."
        # FIX 12(a): a mistyped --season (e.g. "2024-2025" instead of
        # "2024-25") silently matches zero rows and used to print all
        # zeros with no hint the season string itself was the problem.
        if stats.considered == 0 and season is not None:
            seasons = replay.known_seasons(con)
            if seasons:
                message += (
                    f" No games at all matched season '{season}'. Seasons present "
                    f"in the archive: {', '.join(seasons)}."
                )
            else:
                message += " No games at all exist in the archive yet."
        typer.echo(message)
        raise typer.Exit(code=1)

    # FIX 10: reuse status.py's own determination of whether odds data
    # exists (row_count over the correctly-resolved physical table) rather
    # than duplicating that SQL or hardcoding an assumed cause. Today there
    # are zero odds rows in the archive at all, so market comparison is
    # unavailable regardless of season/buffer -- that fact is checked live,
    # not assumed, so it stays true the moment 'predictor ingest-odds' runs.
    #
    # FIX 22(d) (final review, part 3): a bare `next(...)` here raises
    # StopIteration (an ugly, unhandled crash, not a plain-English message)
    # if "odds_snapshots" were ever missing from `check_sources`'s output.
    # It cannot happen today -- db.POINT_IN_TIME_TABLES always includes it,
    # and check_sources iterates exactly that mapping -- but a default makes
    # that guarantee explicit rather than relying on the caller never
    # changing, and degrades to "market comparison unavailable" instead of
    # crashing if it ever does.
    odds_health = next(
        (h for h in status_mod.check_sources(con) if h.name == "odds_snapshots"), None
    )
    if odds_health is None:
        market_available = False
        market_reason = "odds data health could not be determined"
        market_row_count = 0
    else:
        market_available = odds_health.row_count > 0
        market_row_count = odds_health.row_count
        market_reason = (
            None
            if market_available
            else (
                "no odds data has been collected yet "
                f"({odds_health.row_count} row(s) in the archive)"
            )
        )

    result = report.summarize(
        preds,
        stats,
        model=model,
        buffer_minutes=buffer_minutes,
        market_available=market_available,
        market_reason=market_reason,
        market_row_count=market_row_count,
    )
    typer.echo(report.format_report(result))


if __name__ == "__main__":
    app()
