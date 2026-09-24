from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from predictor.config import settings

# Percent-encoding keeps the mapping from raw key -> on-disk name injective:
# distinct keys can never collide on the same blob path. Safe characters pass
# through unchanged (so a human browsing the directory sees readable names for
# the common case); everything else, including '%' itself, is escaped as
# '%XX' over its UTF-8 bytes.
_SAFE_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-@"
)

# Per-source manifest index, keyed by the RESOLVED manifest path (not the
# source name) so tests that monkeypatch `settings` to different temp
# directories never see state leaked from another settings instance.
_index_cache: dict[Path, dict[str, str]] = {}


class RawStoreConflict(Exception):
    """Raised when a key already has different content recorded or on disk.

    The raw store never silently overwrites or drops bytes under an existing
    key; a genuine content mismatch must surface loudly instead of quietly
    publishing stale or wrong data.
    """


@dataclass(frozen=True)
class RawRef:
    source: str
    key: str
    sha256: str
    path: Path
    fetched_at: datetime
    size: int


def _safe(name: str) -> str:
    if name in ("", ".", ".."):
        raise ValueError(f"unusable name: {name!r}")
    out: list[str] = []
    for ch in name:
        if ch in _SAFE_CHARS:
            out.append(ch)
        else:
            out.extend(f"%{byte:02X}" for byte in ch.encode("utf-8"))
    return "".join(out)


def _source_dir(source: str) -> Path:
    d = settings.raw_dir / _safe(source)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _blobs_dir(source: str) -> Path:
    d = _source_dir(source) / "blobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _manifest_path(source: str) -> Path:
    return _source_dir(source) / "_manifest.jsonl"


def blob_path(source: str, key: str) -> Path:
    return _blobs_dir(source) / _safe(key)


def exists(source: str, key: str) -> bool:
    return blob_path(source, key).exists()


def load(source: str, key: str) -> bytes:
    return blob_path(source, key).read_bytes()


def _iter_manifest_lines(source: str, path: Path) -> Iterator[dict]:
    """Shared line-parsing core for `iter_manifest`/`_load_index_from_disk`.

    I4: a torn/malformed manifest line (the deliberate no-fsync decision --
    see Ruling 10 -- is what creates one, if the process is killed mid-
    append) used to raise an uncaught `json.JSONDecodeError` straight out
    of BOTH readers. Because `_refresh_index` re-reads the manifest on
    every `store()` call that appends a new key, that single torn line
    didn't just break iteration -- it broke `store()` ITSELF, permanently,
    for every future write to that source. For the news archiver (the one
    source that cannot be recovered retroactively), that meant one bad
    line silently stopped it from ever archiving anything again. Skipping
    and warning loudly is the fix: every entry on disk before and after
    the torn line is still real, recoverable data, and must not be thrown
    away just because one line in the middle is damaged.
    """
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            print(
                f"raw_store: WARNING -- skipping unparseable manifest line "
                f"{lineno} in {path} -- {exc}. This entry is likely torn "
                "(an interrupted write); everything else in the manifest "
                "is still intact and will be read normally."
            )
            continue


def iter_manifest(source: str) -> Iterator[dict]:
    path = _manifest_path(source)
    if not path.exists():
        return
    yield from _iter_manifest_lines(source, path)


def _load_index_from_disk(source: str) -> dict[str, str]:
    path = _manifest_path(source)
    idx: dict[str, str] = {}
    if path.exists():
        for entry in _iter_manifest_lines(source, path):
            idx[entry["key"]] = entry["sha256"]
    return idx


def _index(source: str) -> dict[str, str]:
    """Return the key -> sha256 index for a source, built once and cached.

    Idempotency decisions are made against this index (manifest membership),
    never against `path.exists()` alone, so a process crash between writing a
    blob and appending its manifest line can never permanently hide that
    blob from later `store()` calls.
    """
    path = _manifest_path(source)
    idx = _index_cache.get(path)
    if idx is None:
        idx = _load_index_from_disk(source)
        _index_cache[path] = idx
    return idx


def _refresh_index(source: str) -> dict[str, str]:
    """Reload a source's cached index from disk truth, in place.

    Only called on the path where `store()` is about to decide whether to
    append a manifest line for a key the in-memory cache does not know
    about. The cache is built lazily and then trusted for the process's
    lifetime, so if another writer (a concurrent process, or an earlier,
    out-of-band write to the manifest) recorded this key after our cache
    was built, a stale cache would wrongly treat it as brand new and append
    a duplicate line. Appends are rare relative to idempotent lookups, so
    paying for a full re-read of the manifest here -- and only here -- keeps
    the idempotency guarantee real without slowing down the common path or
    the backfill-loop case where the key really is already cached.
    """
    path = _manifest_path(source)
    fresh = _load_index_from_disk(source)
    cached = _index_cache.get(path)
    if cached is None:
        _index_cache[path] = fresh
        return fresh
    cached.clear()
    cached.update(fresh)
    return cached


def store(
    source: str,
    key: str,
    content: bytes,
    fetched_at: datetime,
    meta: dict | None = None,
) -> RawRef:
    if fetched_at.tzinfo is None:
        raise ValueError("fetched_at must be timezone-aware")
    offset = fetched_at.utcoffset()
    if offset is not None and offset.total_seconds() != 0:
        raise ValueError(
            "fetched_at must be UTC (zero UTC offset); "
            f"got an offset of {offset} instead"
        )

    digest = hashlib.sha256(content).hexdigest()
    idx = _index(source)
    recorded = idx.get(key)
    blob = blob_path(source, key)

    if recorded is None:
        # We're on the path that may decide to append a manifest line.
        # Re-check against disk truth in case another writer recorded this
        # key since our cache was built, so an idempotent retry can never
        # produce a duplicate manifest entry.
        idx = _refresh_index(source)
        recorded = idx.get(key)

    if recorded is not None:
        if recorded != digest:
            raise RawStoreConflict(
                f"source {source!r} key {key!r} was already archived with "
                f"different content (recorded sha256={recorded}, new "
                f"sha256={digest}); refusing to overwrite archived data"
            )
        if not blob.exists():
            # Self-heal: manifest recorded it, but the blob went missing.
            blob.write_bytes(content)
        return RawRef(source, key, digest, blob, fetched_at, len(content))

    if blob.exists():
        existing_digest = hashlib.sha256(blob.read_bytes()).hexdigest()
        if existing_digest != digest:
            raise RawStoreConflict(
                f"source {source!r} key {key!r} has an unrecorded blob on "
                f"disk with different content (on-disk sha256="
                f"{existing_digest}, new sha256={digest}); refusing to "
                f"overwrite archived data"
            )
    else:
        blob.write_bytes(content)

    entry = {
        "source": source,
        "key": key,
        "sha256": digest,
        "fetched_at": fetched_at.isoformat(),
        "size": len(content),
        "meta": meta or {},
    }
    manifest_path = _manifest_path(source)
    # I4 (second half): a torn trailing fragment left by an interrupted
    # write (see Ruling 10 -- no fsync is deliberate) has NO trailing
    # newline. Appending straight onto that would concatenate our new,
    # well-formed JSON line onto the tail of the torn one, permanently
    # fusing them into a second, different unparseable line -- turning a
    # recoverable one-line scar into un-recoverable, compounding damage.
    # A leading newline before the append is a no-op on a healthy manifest
    # (an empty line is skipped by both readers above) and, on a torn one,
    # isolates our new line as its own line instead of corrupting it too.
    if manifest_path.exists() and manifest_path.stat().st_size > 0:
        with manifest_path.open("rb") as fh:
            fh.seek(-1, 2)
            ends_with_newline = fh.read(1) == b"\n"
        if not ends_with_newline:
            with manifest_path.open("a") as fh:
                fh.write("\n")
    with manifest_path.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    # Self-heal in the other direction: keep the cached index in sync so a
    # missing manifest record (e.g. after a crash) is repaired in place
    # without re-reading and re-parsing the whole manifest file.
    idx[key] = digest

    return RawRef(source, key, digest, blob, fetched_at, len(content))
