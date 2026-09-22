from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from predictor.config import settings

_SAFE = re.compile(r"[^A-Za-z0-9._@-]")


@dataclass(frozen=True)
class RawRef:
    source: str
    key: str
    sha256: str
    path: Path
    fetched_at: datetime
    size: int


def _safe(name: str) -> str:
    cleaned = _SAFE.sub("_", name)
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError(f"unusable name: {name!r}")
    return cleaned


def _source_dir(source: str) -> Path:
    d = settings.raw_dir / _safe(source)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _manifest_path(source: str) -> Path:
    return _source_dir(source) / "_manifest.jsonl"


def blob_path(source: str, key: str) -> Path:
    return _source_dir(source) / _safe(key)


def exists(source: str, key: str) -> bool:
    return blob_path(source, key).exists()


def load(source: str, key: str) -> bytes:
    return blob_path(source, key).read_bytes()


def iter_manifest(source: str) -> Iterator[dict]:
    path = _manifest_path(source)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def store(
    source: str,
    key: str,
    content: bytes,
    fetched_at: datetime,
    meta: dict | None = None,
) -> RawRef:
    if fetched_at.tzinfo is None:
        raise ValueError("fetched_at must be timezone-aware")

    path = blob_path(source, key)
    digest = hashlib.sha256(content).hexdigest()
    already = path.exists()
    if not already:
        path.write_bytes(content)
        entry = {
            "source": source,
            "key": key,
            "sha256": digest,
            "fetched_at": fetched_at.isoformat(),
            "size": len(content),
            "meta": meta or {},
        }
        with _manifest_path(source).open("a") as fh:
            fh.write(json.dumps(entry) + "\n")

    return RawRef(source, key, digest, path, fetched_at, len(content))
