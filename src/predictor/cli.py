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
    import duckdb

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

    try:
        con = db.connect_with_retry()
    except duckdb.Error as exc:
        typer.echo(
            f"Could not open the database to save news right now ({exc}). The "
            "downloaded items are archived on disk and will be loaded by the "
            "next run."
        )
        raise typer.Exit(code=1) from None
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


@app.command("ingest-schedule")
def ingest_schedule_cmd(
    season: str = typer.Option(
        None, help="Season to fetch, e.g. 2024-25. Defaults to the current season."
    ),
) -> None:
    """Fetch the NBA schedule for one season, archive it, and load it."""
    from datetime import UTC, datetime

    import duckdb
    import requests

    from predictor import db, raw_store
    from predictor.config import previous_season_label, season_label, settings
    from predictor.sources import schedule

    settings.ensure_dirs()

    def fail(target: str, exc: Exception):
        """Explain a failed download/archive/parse in plain English, exit 1."""
        if isinstance(exc, requests.RequestException):
            typer.echo(
                f"Could not download the NBA schedule for {target} ({exc}). "
                "Nothing was saved; the next scheduled run will try again."
            )
        elif isinstance(exc, schedule.ScheduleUnavailable):
            typer.echo(
                f"The {target} schedule could not be fetched: {exc}. Nothing "
                "was saved."
            )
        elif isinstance(exc, raw_store.RawStoreConflict):
            typer.echo(
                f"The {target} schedule download clashes with a copy already "
                f"archived under the same name ({exc}). Nothing was "
                "overwritten or loaded; the next scheduled run will try again."
            )
        else:
            typer.echo(
                f"The {target} schedule was downloaded and archived, but could "
                f"not be read: {exc}. Nothing was loaded into the database."
            )
        raise typer.Exit(code=1)

    expected = (
        requests.RequestException,
        schedule.ScheduleUnavailable,
        raw_store.RawStoreConflict,
        ValueError,
    )

    def fetch(target: str):
        """Download, archive and parse one season; touches no database."""
        try:
            return schedule.fetch_and_archive(target)
        except expected as exc:
            fail(target, exc)

    # Every download happens BEFORE the database is opened: a slow or
    # retried download must never hold the DuckDB write lock the unattended
    # news job needs.
    fetched = []
    if season:
        fetched.append(fetch(season))
    else:
        target = season_label(datetime.now(UTC))
        reason = None
        try:
            first = schedule.fetch_and_archive(target)
        except (schedule.ScheduleUnavailable, requests.RequestException) as exc:
            reason = str(exc)
        except expected as exc:
            fail(target, exc)
        else:
            if first.parsed.rows:
                fetched.append(first)
            else:
                reason = "the NBA listed no games for it"
        if reason is not None:
            # July-August (or later): next season is not published yet.
            # Keep refreshing the previous one so `status` does not cry
            # wolf for weeks.
            previous = previous_season_label(target)
            typer.echo(
                f"The {target} schedule is not available yet ({reason}); "
                f"refreshing {previous} instead."
            )
            fetched.append(fetch(previous))

    try:
        con = db.connect_with_retry()
    except duckdb.Error as exc:
        typer.echo(
            f"Could not open the database to save the schedule ({exc}). The "
            "download is archived on disk; the next scheduled run will try again."
        )
        raise typer.Exit(code=1) from None
    try:
        db.migrate(con)
    except duckdb.Error as exc:
        typer.echo(
            f"Could not save the schedule to the database ({exc}). The download "
            "is archived on disk; the next scheduled run will try again."
        )
        raise typer.Exit(code=1) from None

    results = []
    for item in fetched:
        try:
            results.append(schedule.load(con, item))
        except duckdb.Error as exc:
            typer.echo(
                f"Could not save the {item.season} schedule to the database "
                f"({exc}). Nothing from this download was loaded; it is "
                "archived on disk and the next scheduled run will try again."
            )
            raise typer.Exit(code=1) from None

    mismatched = False
    for result in results:
        typer.echo(f"schedule {result.season}: {result.written:,} games saved")
        if result.no_tipoff:
            typer.echo(
                f"  {len(result.no_tipoff)} game(s) have no tip-off time yet "
                "(the league lists them as TBD) -- a later run picks the time "
                "up once it is announced."
            )
        if result.undetermined:
            typer.echo(
                f"  {len(result.undetermined)} game(s) left out because their "
                "teams are not decided yet (for example NBA Cup knockout "
                "games) -- a later run picks them up."
            )
        if result.mismatches:
            mismatched = True
            typer.echo(
                f"WARNING: {len(result.mismatches)} game(s) in {result.season} "
                "disagree between the schedule and the games table (see the "
                "'schedule: MISMATCH' lines above). The schedule was saved as "
                "published; nothing was changed to make them agree. Usually a "
                "rescheduled game -- re-run 'predictor ingest-season "
                f"{result.season}' and then this command."
            )
    if mismatched:
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


class _SeasonProgress:
    """Wraps a predictor to print one STDERR line each time `replay.replay`
    crosses into a new season (t7-fix1 finding 5).

    The stage1 backtest against the real archive takes about 7.5 minutes
    with no output at all otherwise -- easy to mistake for a hang. Printed
    to STDERR, never STDOUT, since STDOUT is this command's publishable
    report and must not carry progress chatter.
    """

    def __init__(self, inner, total_seasons: int) -> None:
        self._inner = inner
        self._total = total_seasons
        self._seen: set[str] = set()

    def __call__(self, game, view):
        if game.season not in self._seen:
            self._seen.add(game.season)
            typer.echo(
                f"scoring {game.season} ({len(self._seen)} of {self._total} "
                "seasons)...",
                err=True,
            )
        return self._inner(game, view)


@app.command("backtest")
def backtest_cmd(
    model: str = typer.Option(
        "always-home", help="Which predictor to score: always-home, coin-flip, or stage1."
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

    names = ("always-home", "coin-flip", "stage1")
    if model not in names:
        typer.echo(f"Unknown model '{model}'. Available: {', '.join(names)}.")
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

    stage1_predictor = None
    if model == "stage1":
        from predictor.model import settings as model_settings
        from predictor.model import stage1

        try:
            loaded = model_settings.load()
        except model_settings.SettingsError as exc:
            typer.echo(f"Cannot run the stage1 model: {exc}")
            raise typer.Exit(code=1) from None
        stage1_predictor = stage1.Stage1Predictor(con, loaded)
        # Finding 5 (t7-fix1): tell the user which season is being scored,
        # and how far through the run that is, without printing anything
        # to STDOUT that would pollute the publishable report below.
        total_seasons = 1 if season is not None else len(replay.known_seasons(con))
        chosen = _SeasonProgress(stage1_predictor, total_seasons)
    elif model == "always-home":
        chosen = baselines.always_home
    else:
        chosen = baselines.fixed_probability(0.5)

    try:
        preds, stats = replay.replay(
            con, chosen, season=season, buffer_minutes=buffer_minutes
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
            # The "reconstructed" half of this claim is only true for the
            # games whose schedule timestamp actually carries that flag, which
            # `replay` now counts separately -- so say it only of those, the
            # same way report.py does, rather than asserting it of all of them.
            (stats.skipped_buffer_too_early - stats.skipped_buffer_too_early_reconstructed,
             "had a buffer reaching back past the game's earliest recorded "
             "schedule timestamp"),
            (stats.skipped_buffer_too_early_reconstructed,
             "had a buffer reaching back past the harness's RECONSTRUCTED "
             "schedule timestamp for the game (derived, not observed)"),
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

    headline = preds
    scope = None
    if stage1_predictor is not None:
        headline = [p for p in preds if model_settings.season_role(p.season) == "test"]
        if not headline:
            # Finding 2 (t7-fix1): a non-test --season (the test seasons ARE
            # ingested) must name --season as the cause, not tell the user
            # to ingest data that is already there.
            if season is not None and season not in model_settings.TEST_SEASONS:
                typer.echo(
                    f"No test-season games were scored because --season {season} "
                    "filtered them out. The test seasons are "
                    f"{', '.join(model_settings.TEST_SEASONS)} -- drop --season, or "
                    "pass one of those, to see the stage1 headline."
                )
            else:
                typer.echo(
                    "No test-season games were scored "
                    f"({', '.join(model_settings.TEST_SEASONS)}), so there is no honest "
                    "headline to report. Ingest those seasons and try again."
                )
            raise typer.Exit(code=1)
        # Finding 2 (t7-fix1): built from the seasons actually present in the
        # headline, sorted -- not hard-coded to all three test seasons,
        # which is wrong under --season <one test season> or if a test
        # season is missing from the archive.
        headline_seasons = sorted({p.season for p in headline})
        plural = len(headline_seasons) != 1
        scope = (
            f"test season{'s' if plural else ''} {', '.join(headline_seasons)} only -- "
            f"the model's settings were never tuned on {'them' if plural else 'it'}"
        )

    result = report.summarize(
        headline,
        stats,
        model=model,
        buffer_minutes=buffer_minutes,
        market_available=market_available,
        market_reason=market_reason,
        market_row_count=market_row_count,
        scope=scope,
    )
    typer.echo(report.format_report(result))

    if stage1_predictor is not None:
        typer.echo("")
        typer.echo(report.format_season_table(preds, model_settings.season_role))
        # Finding 6 (t7-fix1): the table's "test" role is easy to misread as
        # the only rows that count -- say plainly that the rest are not
        # excluded, they are the seasons the model trained/tuned on.
        typer.echo(
            "  Only the 'test' rows are held out; earlier rows are seasons the "
            "model learned from or was tuned on."
        )
        if stage1_predictor.unknown_cities or stage1_predictor.no_history:
            typer.echo("")
            # Finding 7 (t7-fix1): "Games" undercounted -- each row here is
            # one TEAM's game (home or away), not one game, and the count
            # spans every replayed season, not just the headline's test
            # seasons -- said plainly rather than left ambiguous.
            typer.echo(
                "  Team-games where travel could not be measured (across all "
                "replayed seasons, counted as 0):"
            )
            for city, n in sorted(stage1_predictor.unknown_cities.items()):
                typer.echo(f"    {n:,} team-game(s): {city}")
            typer.echo(
                f"    {stage1_predictor.no_history:,} team-game(s) with no earlier game "
                "(treated as fully rested, no travel)"
            )
        examples = [
            stage1_predictor.breakdowns[p.game_id]
            for p in headline[-3:]
            if p.game_id in stage1_predictor.breakdowns
        ]
        if examples:
            typer.echo("")
            # Finding 4 (t7-fix1): no legend anywhere explained what the
            # signed terms below mean or what unit they are in.
            typer.echo(
                "  Each term is in points of expected margin; positive numbers "
                "favour the home team."
            )
            typer.echo("  Example explanations (most recent test-season games):")
            for b in examples:
                typer.echo(f"    {b.sentence()}")


@app.command("fit-model")
def fit_model_cmd() -> None:
    """Choose the model's settings from past seasons and save them."""
    import duckdb

    from predictor import db
    from predictor.model import fit as fit_mod
    from predictor.model import settings as model_settings

    try:
        con = db.connect(read_only=True)
    except duckdb.Error as exc:
        typer.echo(
            f"Could not open the database ({exc}). If another 'predictor' "
            "command is running, wait a moment and try again."
        )
        raise typer.Exit(code=1) from None
    try:
        chosen = fit_mod.fit(con)
    except fit_mod.FitError as exc:
        typer.echo(f"Cannot fit the model: {exc}.")
        raise typer.Exit(code=1) from None
    except duckdb.Error as exc:
        typer.echo(
            f"The database is missing tables the model needs ({exc}). Run "
            "'predictor ingest-season <season>' and 'predictor ingest-schedule "
            "--season <season>' first."
        )
        raise typer.Exit(code=1) from None
    finally:
        con.close()
    try:
        model_settings.save(chosen, model_settings.SETTINGS_PATH)
    except OSError as exc:
        typer.echo(
            f"Could not save the fitted settings to {model_settings.SETTINGS_PATH} ({exc})."
        )
        raise typer.Exit(code=1) from None
    typer.echo(fit_mod.describe(chosen))
    typer.echo(f"Saved to {model_settings.SETTINGS_PATH}.")


if __name__ == "__main__":
    app()
