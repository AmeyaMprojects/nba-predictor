from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import duckdb

from predictor import db
from predictor.db import POINT_IN_TIME_TABLES

# Default entity key used by latest() to pick the most recent snapshot per
# logical table, when the caller does not supply one explicitly. Kept in
# sync with db.POINT_IN_TIME_TABLES by
# test_default_latest_key_covers_every_point_in_time_table.
_DEFAULT_LATEST_KEY: dict[str, tuple[str, ...]] = {
    "games": ("game_id",),
    "injury_status": ("team", "player", "game_date"),
    "odds_snapshots": ("game_key", "book"),
    "news_items": ("item_key",),
}


class AsOfError(Exception):
    """Raised when a point-in-time access rule is violated."""


def _quote_ident(name: str) -> str:
    """Double-quote a SQL identifier, escaping any embedded double quotes."""
    return '"' + name.replace('"', '""') + '"'


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
      accident, so a SQL fragment passed to ``.project()``, ``.aggregate()``,
      ``.filter()``, ``.join()``, ``.union()`` etc. that names the LOGICAL
      string (e.g. ``"games"``) cannot resolve to a table and raises a
      DuckDB ``CatalogException`` instead of silently reading unfiltered
      data.
    - Every reference this class emits to a physical table is fully
      qualified with the database name captured once at construction
      (``"<db>"."main"."<table>_raw"``), not just the table name and not
      just schema-qualified. DuckDB resolves an unqualified or only
      schema-qualified table name through the ``temp`` catalog BEFORE the
      real database, so ANY code sharing this connection that runs
      ``CREATE TEMP VIEW games_raw AS ...``, ``con.register("games_raw",
      ...)``, or ``some_relation.query("games_raw", ...)`` would otherwise
      silently replace what every ``table()``/``latest()`` call on this
      connection reads -- including calls made by code that only ever
      touches ``AsOfView`` and never names a table itself. Capturing the
      database name at construction, rather than re-reading it per call,
      also means a later ``USE other_db`` on the same connection cannot
      re-point this view at a different database's tables. The column
      lookup ``latest()`` uses to validate a caller-supplied ``key`` is
      likewise filtered by that captured database name.
    - ``as_of`` is validated once, at construction, via ``db.require_utc``
      (naive datetimes AND non-zero-UTC-offset datetimes are both
      rejected), and is exposed only as a read-only property -- it cannot be
      reassigned after construction.
    - Passing a non-``datetime`` (``date``, ``str``, ``int``, ``None``, ...)
      as ``as_of`` raises ``AsOfError`` with a plain-English message instead
      of an ``AttributeError`` deep in validation.
    - ``latest()``'s caller-supplied ``key`` is validated against the
      table's real columns (via ``information_schema``, filtered to this
      view's captured database) before use, and is never interpolated
      unchecked; an unknown column, an empty key, or a bare string (which
      would otherwise iterate into individual characters) all raise
      ``AsOfError`` instead of reaching SQL.

    What is NOT enforced:

    - A caller who explicitly writes a physical ``_raw`` table name in
      their OWN SQL fragment (rather than the logical name) still reads it
      unfiltered. Nothing in this class can stop that -- it is a
      deliberate act, not an accident. ``tests/test_leakage.py`` pins this
      residual explicitly. The enforcement mechanism for it is a repo-wide
      source scan asserting no physical name appears in a string literal
      outside ``db.py``/``asof.py`` (excluding ``tests/``, which
      legitimately writes physical names to set up fixtures that simulate
      ingestion).
    - Nothing stops a caller from reassigning ``view.con`` to a different
      connection, or mutating the private ``view._as_of`` attribute
      directly. Both require deliberately reaching past the public API,
      unlike the catalog-shadowing issue above, which corrupts every
      well-behaved caller on the SAME connection without any of them doing
      anything wrong.
    - Nothing stops a caller from opening a second, unfiltered
      ``duckdb.DuckDBPyConnection`` directly onto the same database file and
      reading the ``_raw`` tables that way. ``AsOfView`` guards this
      connection's SQL surface; it is not a database-level permission
      system.
    - A relation returned by ``table()``/``latest()`` is evaluated lazily,
      but a *bound* one (this class always binds ``$cutoff`` as a query
      parameter) is effectively a snapshot as of the moment it was
      returned: a row inserted afterwards, even with an ``observed_at`` at
      or before the cutoff, is not picked up by re-fetching the SAME
      relation object, only by calling ``table()``/``latest()`` again.
      This is a caching/performance detail, not a correctness one -- it
      never shows *more* than the cutoff allows, only potentially less
      until re-queried.
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
        # Captured ONCE here, not re-read per query. This is what defeats
        # both catalog shadowing (temp views/registered relations resolve
        # ahead of a bare or schema-qualified name, but never ahead of an
        # explicit database qualifier) and a later `USE other_db` on this
        # same connection silently re-pointing every subsequent query at a
        # different database's tables.
        self._db_name = con.execute("SELECT current_database()").fetchone()[0]

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
        qualified = self._qualify(self._resolve(name))
        return self.con.sql(
            f"SELECT * FROM {qualified} WHERE observed_at <= $cutoff",
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
        if key is None:
            if name not in _DEFAULT_LATEST_KEY:
                # Guards a future table added to POINT_IN_TIME_TABLES but
                # not to _DEFAULT_LATEST_KEY -- without this, latest()
                # would raise a raw KeyError instead of a clear AsOfError.
                raise AsOfError(
                    f"{name!r} has no default latest() key; pass key= explicitly"
                )
            columns = _DEFAULT_LATEST_KEY[name]
        else:
            if isinstance(key, str):
                # A bare string is a Sequence[str] too -- iterating it
                # yields individual characters, which would build a
                # nonsensical (but not unsafe) PARTITION BY. Reject it
                # explicitly rather than failing confusingly later.
                raise AsOfError(
                    "key must be a sequence of column names, not a bare "
                    f"string; got {key!r} -- did you mean ({key!r},)?"
                )
            columns = tuple(key)
            if not columns:
                raise AsOfError("key must not be empty")

        valid_columns = self._real_columns(physical)
        unknown = [c for c in columns if c not in valid_columns]
        if unknown:
            raise AsOfError(
                f"unknown column(s) {unknown!r} for table {name!r}; "
                f"known columns: {sorted(valid_columns)}"
            )
        qualified = self._qualify(physical)
        partition = ", ".join(_quote_ident(c) for c in columns)
        return self.con.sql(
            f"SELECT * FROM {qualified} WHERE observed_at <= $cutoff "
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

    def _qualify(self, physical_table: str) -> str:
        """Fully qualify a physical table name with the captured database.

        Schema-qualifying alone (``main.games_raw``) is NOT sufficient --
        DuckDB still resolves it through the ``temp`` catalog first, which
        also has a ``main`` schema. Only a full three-part reference
        (database.schema.table) is immune to catalog shadowing.
        """
        return (
            f"{_quote_ident(self._db_name)}."
            f"{_quote_ident('main')}."
            f"{_quote_ident(physical_table)}"
        )

    def _real_columns(self, physical_table: str) -> set[str]:
        rows = self.con.execute(
            "SELECT column_name FROM information_schema.columns"
            " WHERE table_catalog = $db AND table_name = $table"
            " AND table_schema = 'main'",
            {"db": self._db_name, "table": physical_table},
        ).fetchall()
        return {r[0] for r in rows}
