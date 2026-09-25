"""Session-wide test isolation from the real data directory.

FIX 1 (final review, part 1): with no conftest.py and no pytest env
configuration, `predictor.config.settings` resolved to
`PROJECT_ROOT / "data"` during every test run -- the same directory that
holds the irreplaceable `predictor.duckdb` archive and a live scheduled
`predictor poll-news` job. Any test that went through `db.connect()` /
`db.migrate()` without explicitly overriding `settings` (as most CLI tests
do via a per-test `Settings(data_dir=tmp_path)`) wrote straight into that
real database.

`predictor.config.settings` is a module-level singleton computed ONCE, at
import time (`settings = Settings(data_dir=_default_data_dir())`), not
re-resolved on each access. So this environment variable must be set
before `predictor.config` is imported for the first time by anything --
conftest.py is imported by pytest before it collects/imports any test
module, which is early enough. Every test that does not explicitly
override `settings` now gets a fresh, empty, throwaway directory instead
of the real one.
"""

import os
import tempfile

os.environ["PREDICTOR_DATA_DIR"] = tempfile.mkdtemp(prefix="predictor-tests-")
