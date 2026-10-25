"""The Mac's weekly history refresh and the cloud run's copy of it.

The Mac (launchd ``com.predictor.history``, ``predictor refresh-history``)
keeps a worktree of branch ``mac-history`` under its data dir. Once a week it
refreshes its DB (schedule, results, player game logs -- every download
before the DB is opened), writes the ``state/nba/`` snapshots and the current
players list into that worktree, appends any news item the cloud copy lacks
to ``state/news/``, and pushes ONE ``state:`` commit to ``mac-history``. It
never touches ``main`` or ``cloud-shadow``.

Every cloud run starts with `sync_mac_history` (before ``rebuild``):

- ``git ls-remote origin refs/heads/mac-history``: no such branch, or the
  same tip as the last one recorded in ``state/mac_history/synced.jsonl``,
  is a quiet no-op with no fetch.
- Otherwise fetch main and mac-history (``--unshallow`` first on a shallow
  checkout; GitHub refuses ``--shallow-exclude``). Every commit in
  ``origin/main..origin/mac-history`` must pass publish's commit-safety
  rule (subject ``state: ``, paths under ``state/`` only), narrowed to
  ``state/nba/`` and ``state/news/``. One commit that does not means nothing
  is copied, the tip is not recorded, and the reason is reported.
- Every ``state/nba/`` file whose content differs is copied into the working
  tree (whole-file snapshots, written only by the Mac). News items are
  never copied as files: the ``item_key`` s the cloud's ``state/news`` lacks
  are appended to it, so cloud lines are never rewritten and nothing is
  loaded twice.

The new tip is recorded only after a successful copy-in. The copies and the
record are committed by the run's own ``state:`` commit.
"""

from __future__ import annotations

import gzip
import io
import os
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from predictor import db, raw_store
from predictor.cloud import state
from predictor.model import publish

MAC_BRANCH = "mac-history"
# Paths a mac-history commit may touch.
MAC_PATHS = ("state/nba/", "state/news/")
_REMOTE_REF = f"refs/remotes/origin/{MAC_BRANCH}"
# Append-only record of each mac-history tip copied in (the last line wins);
# committed with the run's state commit.
SYNCED_PATH = "mac_history/synced.jsonl"
# --unshallow fetches the whole history once a week on a fresh runner.
_UNSHALLOW_TIMEOUT = 300.0
_NEWS_FILE = re.compile(r"^state/news/[^/]+/[^/]+\.jsonl$")
# The branch mac-history was cut from: the Mac's commits are
# origin/main..origin/mac-history.
_BASE_BRANCH = "main"
_BASE_REF = f"refs/remotes/origin/{_BASE_BRANCH}"

PLAYERS_SOURCE = "players"
PLAYERS_COLUMNS = (
    "player_id", "player_name", "team", "team_id", "roster_status", "from_year", "to_year",
    "season",
)


class MacSyncError(RuntimeError):
    """The mac-history worktree could not be prepared; the message is plain
    English and safe to print."""


@dataclass(frozen=True)
class MacSync:
    copied: list[str]
    skipped_reason: str | None
    # News items appended to state/news (not in ``copied``: they are new
    # lines, not copied files).
    news_items: int = 0
    news_files: list[str] = field(default_factory=list, compare=False)


# --- git -----------------------------------------------------------------------


def _git(runner: Callable, repo: Path, args: list[str], *, timeout: float = publish._GIT_TIMEOUT,
         text: bool = True):
    env = {**os.environ, **publish._GIT_ENV_OVERRIDES}
    try:
        return runner(["git", *args], cwd=repo, capture_output=True, text=text,
                      timeout=timeout, stdin=subprocess.DEVNULL, env=env)
    except (subprocess.TimeoutExpired, OSError) as exc:
        empty = "" if text else b""
        return subprocess.CompletedProcess(["git", *args], 1, stdout=empty, stderr=str(exc))


def _err(result) -> str:
    out = result.stderr or result.stdout or ""
    if isinstance(out, bytes):
        out = out.decode("utf-8", "replace")
    return publish._redact(out.strip().splitlines()[0] if out.strip() else "no output")


def _names(result) -> list[str]:
    return [n for n in result.stdout.split("\0") if n]


# --- the cloud side --------------------------------------------------------------


def last_synced(state_dir: Path) -> str | None:
    """The mac-history tip most recently copied in, or None."""
    path = Path(state_dir) / SYNCED_PATH
    if not path.exists():
        return None
    shas = [r["sha"] for r in state.read_records(path) if r.get("sha")]
    return shas[-1] if shas else None


def sync_mac_history(state_repo: Path, runner=subprocess.run) -> MacSync:
    """Copy the Mac's ``state/nba/`` files (and news items) from
    ``origin/mac-history`` into ``state_repo``'s working tree; see the module
    docstring. Never commits, merges or pushes."""
    repo = Path(state_repo)
    probe = _git(runner, repo, ["ls-remote", "--exit-code", "origin", f"refs/heads/{MAC_BRANCH}"],
                 timeout=publish._PUSH_TIMEOUT)
    if probe.returncode == 2:  # --exit-code: no such ref
        return MacSync([], None)
    if probe.returncode != 0 or not probe.stdout.split():
        return MacSync([], f"could not look for {MAC_BRANCH} on origin ({_err(probe)})")
    tip = probe.stdout.split()[0]
    state_dir = repo / state.STATE_DIRNAME
    if last_synced(state_dir) == tip:
        return MacSync([], None)  # nothing new since the last copy-in: no fetch

    # Only when mac-history moved (about weekly). A depth-1 checkout (live.yml)
    # has no history below HEAD, so the range would be wrong: get the full
    # history first. (--shallow-exclude is not used: GitHub refuses it.)
    fetch = ["fetch", "--quiet", "origin", f"+refs/heads/{_BASE_BRANCH}:{_BASE_REF}",
             f"+refs/heads/{MAC_BRANCH}:{_REMOTE_REF}"]
    shallow = _git(runner, repo, ["rev-parse", "--is-shallow-repository"])
    if shallow.returncode == 0 and shallow.stdout.strip() == "true":
        fetch.insert(2, "--unshallow")
    fetched = _git(runner, repo, fetch, timeout=_UNSHALLOW_TIMEOUT)
    if fetched.returncode != 0:
        return MacSync([], f"fetching {MAC_BRANCH} failed ({_err(fetched)})")

    listed = _git(runner, repo, ["rev-list", f"{_BASE_REF}..{_REMOTE_REF}"])
    if listed.returncode != 0:
        return MacSync([], f"could not list the {MAC_BRANCH} commits ({_err(listed)})")
    shas = [s for s in listed.stdout.split() if s]
    if not shas:
        # mac-history holds nothing main lacks (e.g. just pushed from main by
        # hand): nothing to copy, and copying from an old tree could regress.
        state.append_records(state_dir / SYNCED_PATH, [{"sha": tip}])
        return MacSync([], None)
    unsafe = [publish.commit_subject(repo, sha) for sha in shas
              if not publish.is_safe_commit(repo, sha, only_under=MAC_PATHS)]
    if unsafe:
        return MacSync([], (
            f"{MAC_BRANCH} has commit(s) that are not state: commits touching only "
            f"{' and '.join(MAC_PATHS)} ({'; '.join(unsafe)}); nothing copied -- "
            "fix the branch by hand"))

    changed = _git(runner, repo, ["diff", "--name-only", "-z", "--no-renames",
                                  "--diff-filter=AM", "HEAD", _REMOTE_REF, "--", "state/nba"])
    if changed.returncode != 0:
        return MacSync([], f"could not compare {MAC_BRANCH} ({_err(changed)})")
    copies: dict[str, bytes] = {}
    for rel in _names(changed):
        blob = _git(runner, repo, ["show", f"{_REMOTE_REF}:{rel}"], text=False)
        if blob.returncode != 0:
            return MacSync([], f"could not read {rel} from {MAC_BRANCH} ({_err(blob)})")
        target = repo / rel
        if not target.exists() or target.read_bytes() != blob.stdout:
            copies[rel] = blob.stdout

    news_rows: list[dict] = []
    tree = _git(runner, repo, ["ls-tree", "-r", "-z", "--name-only", _REMOTE_REF, "--",
                               "state/news"])
    if tree.returncode != 0:
        return MacSync([], f"could not list {MAC_BRANCH} news ({_err(tree)})")
    for rel in _names(tree):
        if not _NEWS_FILE.match(rel):
            continue
        text = _git(runner, repo, ["show", f"{_REMOTE_REF}:{rel}"])
        if text.returncode != 0:
            return MacSync([], f"could not read {rel} from {MAC_BRANCH} ({_err(text)})")
        news_rows.extend(state.parse_records(text.stdout, f"{MAC_BRANCH}:{rel}"))

    # Read and checked everything first; now write.
    for rel, content in sorted(copies.items()):
        _write_atomic(repo / rel, content)
    news_files, added = append_new_news(state_dir, news_rows)
    # Last: a run that fails before here copies everything again next time.
    state.append_records(state_dir / SYNCED_PATH, [{"sha": tip}])
    return MacSync(
        copied=sorted(copies), skipped_reason=None, news_items=added,
        news_files=[str(p.relative_to(repo)) for p in news_files],
    )


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(content)
    os.replace(tmp, path)


def append_new_news(state_dir: Path, rows: list[dict]) -> tuple[list[Path], int]:
    """Append the news rows whose ``item_key`` ``state_dir/news`` does not
    hold yet (each key once, oldest first). Returns (files touched, rows
    appended). Used by both sides: the Mac exports its history this way,
    and the cloud takes the Mac's items the same way."""
    seen = {r["item_key"] for r in state.read_rows(state_dir, "news")}
    fresh: list[dict] = []
    for row in sorted(rows, key=lambda r: (state._as_utc(r["observed_at"]), r["item_key"])):
        if row["item_key"] in seen:
            continue
        seen.add(row["item_key"])
        fresh.append({c: row.get(c) for c in state.NEWS_COLUMNS})
    if not fresh:
        return [], 0
    return state.append_rows(state_dir, "news", fresh), len(fresh)


# --- the Mac side ------------------------------------------------------------------


def prepare_worktree(repo_dir: Path, path: Path) -> Path:
    """A worktree of ``repo_dir`` at ``path`` on branch mac-history, fast-
    forwarded to ``origin/mac-history``. Added once (from the local branch
    if there is one, else tracking origin's); later calls only pull
    ``--ff-only``. Raises `MacSyncError` with a plain message."""
    repo_dir, path = Path(repo_dir), Path(path)
    fetch = publish._git(repo_dir, ["fetch", "--quiet", "origin",
                                    f"+refs/heads/{MAC_BRANCH}:{_REMOTE_REF}"],
                         timeout=publish._PUSH_TIMEOUT)
    if fetch.returncode != 0:
        raise MacSyncError(
            f"could not fetch {MAC_BRANCH} from origin ({publish._err(fetch)}); the first "
            f"push is made by hand: git push origin main:{MAC_BRANCH}")
    if not (path / ".git").exists():
        if path.exists() and any(path.iterdir()):
            raise MacSyncError(f"{path} exists but is not a git worktree; move it aside")
        path.parent.mkdir(parents=True, exist_ok=True)
        publish._git(repo_dir, ["worktree", "prune"])
        local = publish._git(repo_dir, ["rev-parse", "--verify", "--quiet",
                                        f"refs/heads/{MAC_BRANCH}"])
        if local.returncode == 0:
            args = ["worktree", "add", str(path), MAC_BRANCH]
        else:
            args = ["worktree", "add", "--track", "-b", MAC_BRANCH, str(path), _REMOTE_REF]
        added = publish._git(repo_dir, args)
        if added.returncode != 0:
            raise MacSyncError(f"could not add the {MAC_BRANCH} worktree at {path} "
                               f"({publish._err(added)})")
    branch = publish.current_branch(path)
    if branch != MAC_BRANCH:
        raise MacSyncError(f"{path} is on {branch or 'a detached HEAD'}, not {MAC_BRANCH}; "
                           "fix it by hand")
    pulled = publish._git(path, ["pull", "--ff-only", "--quiet", "origin", MAC_BRANCH],
                          timeout=publish._PUSH_TIMEOUT)
    if pulled.returncode != 0:
        raise MacSyncError(f"could not fast-forward {path} to origin/{MAC_BRANCH} "
                           f"({publish._err(pulled)}); fix it by hand")
    return path


def _fetch_players(season: str):
    from nba_api.stats.endpoints import commonallplayers
    from tenacity import retry, stop_after_attempt, wait_exponential

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=30),
           reraise=True)
    def fetch():
        return commonallplayers.CommonAllPlayers(
            is_only_current_season=1, league_id="00", season=season, timeout=60,
        ).get_data_frames()[0]

    return fetch()


def _text_or_none(value) -> str | None:
    import pandas as pd

    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _int_or_none(value) -> int | None:
    text = _text_or_none(value)
    return None if text is None else int(float(text))


def fetch_current_players(season: str, *, fetch: Callable = _fetch_players,
                          fetched_at: datetime | None = None) -> list[dict]:
    """This season's players (``CommonAllPlayers(is_only_current_season=1)``),
    archived raw-first under source ``players`` and parsed from the archived
    bytes. Rows (sorted by player_id) use `PLAYERS_COLUMNS`."""
    import pandas as pd

    frame = fetch(season)
    fetched_at = db.require_utc(fetched_at if fetched_at is not None else datetime.now(UTC),
                                "fetched_at")
    key = f"{season}_{fetched_at:%Y%m%dT%H%M%S}Z.json.gz"
    payload = frame.to_json(orient="split").encode()
    raw_store.store(PLAYERS_SOURCE, key, gzip.compress(payload, mtime=0), fetched_at,
                    meta={"season": season})
    archived = pd.read_json(io.BytesIO(gzip.decompress(raw_store.load(PLAYERS_SOURCE, key))),
                            orient="split", dtype=False)
    rows = []
    for rec in archived.to_dict("records"):
        rows.append({
            "player_id": _int_or_none(rec["PERSON_ID"]),
            "player_name": _text_or_none(rec.get("DISPLAY_FIRST_LAST")),
            "team": _text_or_none(rec.get("TEAM_ABBREVIATION")),
            "team_id": _int_or_none(rec.get("TEAM_ID")),
            "roster_status": _int_or_none(rec.get("ROSTERSTATUS")),
            "from_year": _text_or_none(rec.get("FROM_YEAR")),
            "to_year": _text_or_none(rec.get("TO_YEAR")),
            "season": season,
        })
    return sorted(rows, key=lambda r: r["player_id"])
