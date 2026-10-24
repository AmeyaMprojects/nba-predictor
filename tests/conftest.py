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

import atexit
import os
import shutil
import tempfile

_TEST_DATA_DIR = tempfile.mkdtemp(prefix="predictor-tests-")
os.environ["PREDICTOR_DATA_DIR"] = _TEST_DATA_DIR

# FIX 22(b) (final review, part 3): nothing ever removed the throwaway
# directory created above, so one accumulated per test run (50 present
# before this fix). Registered at import time -- the directory is created
# once per process, at collection, so process exit is the only point that
# is guaranteed to run after every test has finished with it. `ignore_errors`
# because a DuckDB connection some test forgot to close can still hold a
# file open on some platforms; leaving that one behind on exit is harmless
# and strictly better than crashing pytest's teardown. This only ever
# removes the directory THIS process itself created under the system temp
# root -- never anything under the real, irreplaceable `data/` directory.
atexit.register(shutil.rmtree, _TEST_DATA_DIR, ignore_errors=True)


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _network_is_up(monkeypatch):
    """`ingest-odds` and `capture-results` first wait for DNS to resolve
    (`predictor.net.wait_for_network`). Tests never do a real lookup:
    the network counts as up unless a test says otherwise."""
    from predictor import net

    monkeypatch.setattr(net, "wait_for_network", lambda *a, **k: True)


@pytest.fixture(autouse=True)
def _repo_dir_is_never_the_real_checkout(monkeypatch, tmp_path_factory):
    """`capture-results` and `predict-today` grade and publish
    `predictions/*.jsonl` in `cli._repo_dir()` -- by default the REAL
    checkout. Every test gets an empty, non-git throwaway directory instead
    (tests that need a real temp repo pin their own via `_wire`), so no test
    can ever append to or commit/push the real public prediction log."""
    from predictor import cli

    scratch = tmp_path_factory.mktemp("not-the-real-repo")
    monkeypatch.setattr(cli, "_repo_dir", lambda: scratch)


@pytest.fixture(autouse=True)
def _gh_is_never_run(monkeypatch):
    """`status` asks GitHub about cloud runs with `gh run list`. Tests never
    run the real gh: by default it is "not installed" (status then reports a
    plain "could not check"); tests that need answers install a fake."""
    from predictor import cli

    def no_gh(*args, **kwargs):
        raise FileNotFoundError("gh is not run in tests")

    monkeypatch.setattr(cli, "_gh_runner", no_gh)
