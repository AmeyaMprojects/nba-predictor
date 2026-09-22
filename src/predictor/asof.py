from __future__ import annotations

from datetime import datetime

import duckdb

from predictor.db import POINT_IN_TIME_TABLES


class AsOfError(Exception):
    """Raised when a point-in-time access rule is violated."""


class AsOfView:
    """The only sanctioned way to read point-in-time data.

    Every read is filtered to rows whose ``observed_at`` is at or before
    ``as_of``. Feature code must never query these tables directly.
    """

    def __init__(self, con: duckdb.DuckDBPyConnection, as_of: datetime) -> None:
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise AsOfError("as_of must be timezone-aware")
        self.con = con
        self.as_of = as_of

    def table(self, name: str) -> duckdb.DuckDBPyRelation:
        if name not in POINT_IN_TIME_TABLES:
            raise AsOfError(
                f"{name!r} is not a point-in-time table; "
                f"known tables: {sorted(POINT_IN_TIME_TABLES)}"
            )
        return self.con.sql(
            f"SELECT * FROM {name} WHERE observed_at <= $cutoff",
            params={"cutoff": self.as_of},
        )
