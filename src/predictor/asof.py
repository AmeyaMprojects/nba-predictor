from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import duckdb

from predictor import db
from predictor.db import POINT_IN_TIME_TABLES

# Default entity key used by latest() to pick the most recent snapshot per
# logical table, when the caller does not supply one explicitly.
_DEFAULT_LATEST_KEY: dict[str, tuple[str, ...]] = {
    "games": ("game_id",),
    "injury_status": ("team", "player", "game_date"),
    "odds_snapshots": ("game_key", "book"),
    "news_items": ("item_key",),
}


class AsOfError(Exception):
    """Raised when a point-in-time access rule is violated."""


class AsOfView:
    """A point-in-time reader for the tables listed in ``db.POINT_IN_TIME_TABLES``.

    What IS enforced:

    - ``table(name)`` and ``latest(name)`` only accept a LOGICAL name from
      ``db.POINT_IN_TIME_TABLES`` (e.g. ``"games"``); an unknown or crafted
      string raises ``AsOfError`` before any SQL is built.
    - Both methods return exactly the rows whose ``observed_at`` is at or
      before ``as_of`` -- an exact match, never rounded or excluded at the
      boundary.
    - The underlying physical tables are named with a ``_raw`` suffix
      (``games_raw``, ``injury_status_raw``, ...) that is never exposed
      through this class's public API and that no caller would type by
      accident. Because the logical names ("games", "injury_status", ...)
      do not exist as real tables in the DuckDB catalog, a SQL fragment
      passed to ``.project()``, ``.aggregate()``, ``.filter()``, ``.join()``,
      ``.union()`` or a fresh ``con.sql()``/``con.execute()`` call that
      names one of those logical strings cannot resolve to a table and
      raises a DuckDB ``CatalogException`` instead of silently reading
      unfiltered data. Chaining those methods onto the relation this class
      *returns* is still fine and still filtered, because the filter is
      already baked into that relation before it is handed back.
    - ``as_of`` is validated once, at construction, via ``db.require_utc``
      (naive datetimes AND non-zero-UTC-offset datetimes are both
      rejected), and is exposed only as a read-only property -- it cannot be
      reassigned after construction.
    - Passing a non-``datetime`` (``date``, ``str``, ``int``, ``None``, ...)
      as ``as_of`` raises ``AsOfError`` with a plain-English message instead
      of an ``AttributeError`` deep in validation.

    What is NOT enforced (see the task-6 report for the reasoning):

    - A caller can still poison the connection's catalog by ``.query()``-ing
      a relation under an alias that happens to match a physical ``_raw``
      table name, then reading that alias directly from the connection.
      This only ever produces a MORE conservative (i.e. never a leaking)
      result, so it is left alone.
    - ``table()`` does not eagerly materialize its result; repeated calls
      re-run the filter. This is a performance concern for a large
      backtest, not a correctness one.
    - Nothing stops a caller from opening a second, unfiltered
      ``duckdb.DuckDBPyConnection`` directly onto the same database file and
      reading the ``_raw`` tables that way. ``AsOfView`` guards this
      connection's SQL surface; it is not a database-level permission
      system.
    """

    def __init__(self, con: duckdb.DuckDBPyConnection, as_of: datetime) -> None:
        if not isinstance(as_of, datetime):
            # datetime is a subclass of date, so this must be checked before
            # any date-shaped validation -- otherwise a bare `date` (the
            # realistic mis-call: passing a game date instead of a cutoff
            # timestamp) falls through to attribute access on a `date`
            # object and raises a confusing AttributeError instead of a
            # clear AsOfError.
            raise AsOfError(
                "as_of must be a timezone-aware datetime, "
                f"got {type(as_of).__name__}: {as_of!r}"
            )
        try:
            db.require_utc(as_of, "as_of")
        except ValueError as exc:
            # require_utc raises ValueError so it can be shared by every
            # write path in db.py, which has nothing to do with AsOfView.
            # Re-raise as AsOfError so every caller of this class can catch
            # one exception type for every point-in-time access mistake.
            raise AsOfError(str(exc)) from exc
        self.con = con
        self._as_of = as_of

    @property
    def as_of(self) -> datetime:
        return self._as_of

    def table(self, name: str) -> duckdb.DuckDBPyRelation:
        """Every observation of ``name`` at or before the cutoff.

        This can return more than one row per entity (e.g. a game's
        SCHEDULED row and a later IN_PROGRESS row can both be at or before
        the cutoff). That is intentional -- line-movement and
        report-revision features need the full history. Use ``latest()``
        when you want one row per entity instead.
        """
        physical = self._resolve(name)
        return self.con.sql(
            f"SELECT * FROM {physical} WHERE observed_at <= $cutoff",
            params={"cutoff": self._as_of},
        )

    def latest(
        self, name: str, key: Sequence[str] | None = None
    ) -> duckdb.DuckDBPyRelation:
        """Most recent observation per entity, at or before the cutoff.

        ``key`` defaults per logical table (see ``_DEFAULT_LATEST_KEY``) and,
        if supplied, is validated against the table's real columns before
        being used -- it is never interpolated into SQL unchecked.
        """
        physical = self._resolve(name)
        columns = key if key is not None else _DEFAULT_LATEST_KEY[name]
        valid_columns = self._real_columns(physical)
        unknown = [c for c in columns if c not in valid_columns]
        if unknown:
            raise AsOfError(
                f"unknown column(s) {unknown!r} for table {name!r}; "
                f"known columns: {sorted(valid_columns)}"
            )
        partition = ", ".join(columns)
        return self.con.sql(
            f"SELECT * FROM {physical} WHERE observed_at <= $cutoff "
            f"QUALIFY row_number() OVER "
            f"(PARTITION BY {partition} ORDER BY observed_at DESC) = 1",
            params={"cutoff": self._as_of},
        )

    def _resolve(self, name: str) -> str:
        physical = POINT_IN_TIME_TABLES.get(name)
        if physical is None:
            raise AsOfError(
                f"{name!r} is not a point-in-time table; "
                f"known tables: {sorted(POINT_IN_TIME_TABLES)}"
            )
        return physical

    def _real_columns(self, physical_table: str) -> set[str]:
        rows = self.con.execute(
            "SELECT column_name FROM information_schema.columns"
            " WHERE table_name = $table AND table_schema = 'main'",
            {"table": physical_table},
        ).fetchall()
        return {r[0] for r in rows}
