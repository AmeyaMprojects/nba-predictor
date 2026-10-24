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
- Secret guard. The release is on a public repo, so every file is
  byte-searched for the odds key before anything is bundled or uploaded
  (`assert_no_secret`), and the tarball's members are searched again just
  before upload. A hit raises `SecretLeak` and nothing is uploaded; the
  key itself is never put in the message.

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
_HEX32 = re.compile(rb"(?<![0-9A-Fa-f])[0-9A-Fa-f]{32}(?![0-9A-Fa-f])")
# Environment variables whose values must never appear in a message.
_SECRET_ENV = ("ODDS_API_KEY", "GH_TOKEN", "GITHUB_TOKEN")


class SecretLeak(Exception):
    """A file that would be archived or committed contains the odds key."""


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


def _contains(data: bytes, secret: str) -> bool:
    """True when ``data`` holds ``secret`` byte for byte, or holds a
    standalone 32-hex run equal to it in any letter case."""
    if secret.encode() in data:
        return True
    want = secret.lower().encode()
    return any(m.group().lower() == want for m in _HEX32.finditer(data))


def assert_no_secret(paths: Iterable[Path], secret: str | None) -> None:
    """Raise `SecretLeak` if any file (directories are searched recursively)
    contains ``secret``. A no-op when ``secret`` is None or blank."""
    secret = (secret or "").strip()
    if not secret:
        return
    for path in _files(paths):
        if _contains(path.read_bytes(), secret):
            raise SecretLeak(f"{path} contains the odds API key; refusing to archive or commit it")


def _assert_tarball_clean(asset: Path, secret: str | None) -> None:
    secret = (secret or "").strip()
    if not secret:
        return
    if not tarfile.is_tarfile(asset):
        assert_no_secret([asset], secret)
        return
    with tarfile.open(asset, "r:*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            fh = tar.extractfile(member)
            if fh is not None and _contains(fh.read(), secret):
                raise SecretLeak(
                    f"{asset.name} member {member.name} contains the odds API key; "
                    "refusing to upload it"
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


def _scrub(text: str, extra: str | None = None) -> str:
    text = _redact(text)
    for value in [extra, *(os.environ.get(name) for name in _SECRET_ENV)]:
        value = (value or "").strip()
        if value:
            text = text.replace(value, "***")
    return text


def _run(runner, argv: list[str]):
    """Run one gh command; never raises. Returns (ok, stdout, error text)."""
    try:
        result = runner(
            argv,
            capture_output=True,
            text=True,
            timeout=GH_TIMEOUT,
            stdin=subprocess.DEVNULL,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return False, "", f"{type(exc).__name__}: {exc}"
    out = result.stdout or ""
    err = (result.stderr or "").strip() or out.strip()
    return result.returncode == 0, out, err


def _existing_names(stdout: str) -> set[str] | None:
    try:
        data = json.loads(stdout)
        return {a["name"] for a in data.get("assets") or []}
    except (ValueError, TypeError, KeyError, AttributeError):
        return None


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


def upload(tag: str, asset: Path, runner=subprocess.run, *, secret: str | None = None) -> ArchiveResult:
    """Upload ``asset`` to release ``tag``, creating the release if missing.

    Never passes ``--clobber``: a name already on the release is uploaded
    under a suffixed name instead. With ``secret``, the asset (each tarball
    member) is searched first and `SecretLeak` is raised before gh runs.
    """
    asset = Path(asset)
    _assert_tarball_clean(asset, secret)

    def fail(what: str, err: str) -> ArchiveResult:
        return ArchiveResult(False, _scrub(f"{what}: {err}" if err else what, secret))

    ok, out, err = _run(runner, ["gh", "release", "view", tag, "--json", "assets"])
    taken: set[str] = set()
    if ok:
        names = _existing_names(out)
        if names is not None:
            taken = names
        if len(taken) >= MAX_ASSETS:
            return fail(f"release {tag} already has {len(taken)} assets; not uploading", "")
    elif "not found" in err.lower():
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
    """Guard, bundle and upload one run's raw dir. Raises `SecretLeak`
    (before writing or uploading anything) if the key is in any raw file."""
    raw_dir = Path(raw_dir)
    if raw_dir.is_dir():
        assert_no_secret([raw_dir], secret)
    out = bundle(raw_dir, Path(out_dir) / asset_name(now, run_attempt))
    if out is None:
        return ArchiveResult(True, "no raw responses this run; nothing archived")
    return upload(release_tag(now), out, runner=runner, secret=secret)
