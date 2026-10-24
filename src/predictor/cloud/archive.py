"""Each cloud run's raw responses, archived as a weekly GitHub release asset.

Raw responses go to release assets rather than a branch: the injury PDFs
alone are about 6 MB a day, which on a branch would be gigabytes of history
every runner has to fetch. Release assets add nothing to clone size and
survive across runs.

- One release per ISO week (UTC), tagged ``raw-<iso year>-W<week>``. At 96
  runs a day that is at most 672 assets, under GitHub's 1,000 per release.
- One asset per run, ``run-<YYYYMMDDTHHMMZ>-a<attempt>.tar.gz``: a tar.gz
  of the whole raw dir (the data dir is fresh per run, so every blob in it
  is from this run).
- Append-only. ``gh release upload`` is never given ``--clobber``; a name
  already on the release gets a ``-2``, ``-3``, ... suffix instead, so an
  existing asset is never replaced.
- Secret guard. The release is on a public repo, so every file path and
  file content is searched for the odds key, in any letter case, before
  anything is bundled or uploaded (`assert_no_secret`), and the tarball's
  member names and contents are searched again just before upload. Gzip
  content is decompressed and searched too (up to `GZIP_CAP`); gzip that
  cannot be decompressed is refused as unscannable. A hit raises
  `SecretLeak` and nothing is uploaded; the key itself never reaches a
  message. ``upload``/``archive_run`` require ``secret``: None falls back
  to env ``ODDS_API_KEY`` and, if that is unset, uploads nothing;
  ``secret=""`` is the explicit opt-out.

`gh` failures never raise: they come back as ``ArchiveResult(ok=False)``
with gh's stderr, stripped of credentials.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tarfile
import zlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from predictor.model.publish import _redact

GH_TIMEOUT = 120
# GitHub allows 1,000 assets per release; stop short of it.
MAX_ASSETS = 1000

RELEASE_NOTES = "Append-only raw source responses, one asset per cloud run."

_SUFFIX = ".tar.gz"
_GZIP_MAGIC = b"\x1f\x8b"
# Largest decompressed gzip the guard will search; anything bigger is
# refused as unscannable rather than uploaded unchecked.
GZIP_CAP = 200 * 1024 * 1024
# Gzip inside gzip is decompressed this many levels deep, then refused.
_GZIP_DEPTH = 3
# Environment variables whose values must never appear in a message.
_SECRET_ENV = ("ODDS_API_KEY", "GH_TOKEN", "GITHUB_TOKEN")
# gh must never stop to ask a question or print an update banner.
_GH_ENV = {"GH_PROMPT_DISABLED": "1", "GH_NO_UPDATE_NOTIFIER": "1"}


class SecretLeak(Exception):
    """A file that would be archived or committed contains the odds key,
    or cannot be searched for it."""


@dataclass(frozen=True)
class ArchiveResult:
    ok: bool
    message: str


def _utc(now: datetime) -> datetime:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("archive: a timezone-aware datetime is required")
    return now.astimezone(UTC)


def release_tag(now: datetime) -> str:
    """``raw-<ISO year>-W<ISO week, 2 digits>`` of ``now`` in UTC."""
    iso_year, week, _ = _utc(now).isocalendar()
    return f"raw-{iso_year}-W{week:02d}"


def asset_name(now: datetime, run_attempt: int) -> str:
    return f"run-{_utc(now):%Y%m%dT%H%MZ}-a{run_attempt}{_SUFFIX}"


def _files(paths: Iterable[Path]) -> Iterator[Path]:
    for path in paths:
        path = Path(path)
        if path.is_dir():
            yield from sorted(p for p in path.rglob("*") if p.is_file())
        elif path.is_file():
            yield path


def _scrub(text: str, extra: str | None = None) -> str:
    """``text`` with credentials removed: URL userinfo / access tokens
    (`publish._redact`), and -- in any letter case -- ``extra`` and the
    values of `_SECRET_ENV`."""
    text = _redact(text)
    for value in [extra, *(os.environ.get(name) for name in _SECRET_ENV)]:
        value = (value or "").strip()
        if value:
            text = re.sub(re.escape(value), "***", text, flags=re.IGNORECASE)
    return text


def _gunzip(data: bytes) -> bytes:
    """Decompress gzip ``data`` (all members), refusing past `GZIP_CAP`."""
    out = bytearray()
    while data:
        d = zlib.decompressobj(wbits=31)
        out += d.decompress(data, GZIP_CAP + 1 - len(out))
        if len(out) > GZIP_CAP or d.unconsumed_tail:
            raise OverflowError("over the decompression cap")
        if not d.eof:
            raise zlib.error("truncated gzip stream")
        data = d.unused_data
    return bytes(out)


def _check_name(name: str, secret: str) -> None:
    if secret.lower() in name.lower():
        raise SecretLeak(
            f"{_scrub(name, secret)} has the odds API key in its name; refusing to archive it"
        )


def _check_bytes(data: bytes, secret: str, label: str, depth: int = 0) -> None:
    """Raise `SecretLeak` if ``data`` holds ``secret`` in any letter case,
    searching gzip content too; unscannable gzip is refused."""
    if secret.lower().encode() in data.lower():
        raise SecretLeak(f"{_scrub(label, secret)} contains the odds API key; refusing it")
    if not data.startswith(_GZIP_MAGIC):
        return
    if depth >= _GZIP_DEPTH:
        raise SecretLeak(f"{_scrub(label, secret)} is unscannable (gzip nested too deep); refusing it")
    try:
        inner = _gunzip(data)
    except (OverflowError, zlib.error, EOFError) as exc:
        raise SecretLeak(
            f"{_scrub(label, secret)} is unscannable gzip ({exc}); refusing it"
        ) from None
    _check_bytes(inner, secret, label, depth + 1)


def assert_no_secret(paths: Iterable[Path], secret: str | None) -> None:
    """Raise `SecretLeak` if any file (directories are searched
    recursively) has ``secret`` in its path or content, in any letter case.
    Gzip content is decompressed and searched; gzip that cannot be
    (corrupt, or over `GZIP_CAP`) is refused. A no-op when ``secret`` is
    None or blank."""
    secret = (secret or "").strip()
    if not secret:
        return
    for path in _files(paths):
        _check_name(str(path), secret)
        _check_bytes(path.read_bytes(), secret, str(path))


def _assert_tarball_clean(asset: Path, secret: str) -> None:
    if not secret:
        return
    _check_name(str(asset), secret)
    if not tarfile.is_tarfile(asset):
        assert_no_secret([asset], secret)
        return
    with tarfile.open(asset, "r:*") as tar:
        for member in tar:
            _check_name(member.name, secret)
            if not member.isfile():
                continue
            fh = tar.extractfile(member)
            if fh is not None:
                _check_bytes(fh.read(), secret, f"{asset.name} member {member.name}")


def _resolve_secret(secret: str | None) -> str | None:
    """The key to guard against: ``secret`` itself when given (``""`` is
    the explicit opt-out: no guard); for None, env ``ODDS_API_KEY``; None
    when neither exists (the caller must refuse)."""
    if secret is not None:
        return secret.strip()
    env = os.environ.get("ODDS_API_KEY", "").strip()
    return env or None


_NO_SECRET = (
    "no odds key to guard against (secret=None and ODDS_API_KEY unset); "
    "nothing uploaded -- pass secret='' to archive without a guard"
)


def bundle(raw_dir: Path, out: Path) -> Path | None:
    """Write a tar.gz of the whole raw dir (members under ``raw/``) to
    ``out``. None, and no file, when the raw dir has no files."""
    raw_dir = Path(raw_dir)
    out = Path(out)
    if out.resolve().is_relative_to(raw_dir.resolve()):
        raise ValueError("archive: the bundle cannot be written inside the raw dir")
    files = list(_files([raw_dir])) if raw_dir.is_dir() else []
    if not files:
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        for path in files:
            tar.add(path, arcname=str(Path("raw") / path.relative_to(raw_dir)), recursive=False)
    return out


def _run(runner, argv: list[str]):
    """Run one gh command; never raises. Returns (ok, stdout, error text)."""
    try:
        result = runner(
            argv,
            capture_output=True,
            text=True,
            timeout=GH_TIMEOUT,
            stdin=subprocess.DEVNULL,
            env={**os.environ, **_GH_ENV},
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return False, "", f"{type(exc).__name__}: {exc}"
    out = result.stdout or ""
    err = (result.stderr or "").strip() or out.strip()
    return result.returncode == 0, out, err


def _existing_names(stdout: str) -> set[str] | None:
    """Asset names from ``gh release view --json assets``; None when the
    output is not that shape."""
    try:
        data = json.loads(stdout)
        assets = data["assets"]
        if not isinstance(assets, list):
            return None
        names = {a["name"] for a in assets}
    except (ValueError, TypeError, KeyError, AttributeError):
        return None
    return names if all(isinstance(n, str) for n in names) else None


def _unique(asset: Path, taken: set[str]) -> Path:
    """``asset``, or a suffixed copy of it whose name is not in ``taken``."""
    if asset.name not in taken:
        return asset
    name = asset.name
    stem = name[: -len(_SUFFIX)] if name.endswith(_SUFFIX) else asset.stem
    ext = _SUFFIX if name.endswith(_SUFFIX) else asset.suffix
    n = 2
    while f"{stem}-{n}{ext}" in taken or asset.with_name(f"{stem}-{n}{ext}").exists():
        n += 1
    copy = asset.with_name(f"{stem}-{n}{ext}")
    shutil.copyfile(asset, copy)
    return copy


def upload(tag: str, asset: Path, runner=subprocess.run, *, secret: str | None) -> ArchiveResult:
    """Upload ``asset`` to release ``tag``, creating the release if missing.

    ``secret`` is required: the key every tarball member (and its name) is
    searched for before gh runs -- a hit raises `SecretLeak`. None means
    env ``ODDS_API_KEY``; if that is unset too, nothing is uploaded
    (ok=False). ``secret=""`` is the explicit opt-out: upload unguarded.

    Never passes ``--clobber``: a name already on the release is uploaded
    under a suffixed name instead. gh failures, and view output that
    cannot be read, give ok=False and never raise.
    """
    asset = Path(asset)
    key = _resolve_secret(secret)
    if key is None:
        return ArchiveResult(False, _NO_SECRET)
    _assert_tarball_clean(asset, key)

    def fail(what: str, err: str) -> ArchiveResult:
        return ArchiveResult(False, _scrub(f"{what}: {err}" if err else what, key))

    ok, out, err = _run(runner, ["gh", "release", "view", tag, "--json", "assets"])
    taken: set[str] = set()
    if ok:
        names = _existing_names(out)
        if names is None:
            return fail(f"could not read the asset list of release {tag}; not uploading", "")
        taken = names
        if len(taken) >= MAX_ASSETS:
            return fail(f"release {tag} already has {len(taken)} assets; not uploading", "")
    elif "release not found" in err.lower():
        ok, _, err = _run(runner, [
            "gh", "release", "create", tag,
            "--title", f"Raw archive {tag}",
            "--notes", RELEASE_NOTES,
            "--prerelease", "--latest=false",
        ])
        if not ok:
            return fail(f"could not create release {tag}", err)
    else:
        return fail(f"could not read release {tag}", err)

    target = _unique(asset, taken)
    ok, _, err = _run(runner, ["gh", "release", "upload", tag, str(target)])
    if not ok:
        return fail(f"upload of {target.name} to {tag} failed", err)
    return ArchiveResult(True, f"uploaded {target.name} to {tag}")


def archive_run(
    raw_dir: Path,
    out_dir: Path,
    now: datetime,
    run_attempt: int,
    *,
    secret: str | None,
    runner=subprocess.run,
) -> ArchiveResult:
    """Guard, bundle and upload one run's raw dir. ``secret`` as for
    `upload`. Raises `SecretLeak` (before writing or uploading anything)
    if the key is in any raw file or file name."""
    raw_dir = Path(raw_dir)
    if not (raw_dir.is_dir() and any(_files([raw_dir]))):
        return ArchiveResult(True, "no raw responses this run; nothing archived")
    key = _resolve_secret(secret)
    if key is None:
        return ArchiveResult(False, _NO_SECRET)
    assert_no_secret([raw_dir], key)
    out = bundle(raw_dir, Path(out_dir) / asset_name(now, run_attempt))
    if out is None:
        return ArchiveResult(True, "no raw responses this run; nothing archived")
    return upload(release_tag(now), out, runner=runner, secret=key)
