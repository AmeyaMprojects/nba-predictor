"""Raw archive: weekly release assets behind a secret guard.

No test here runs `gh` or touches the network: every `upload` gets a fake
runner that records argv and returns canned results.
"""

import json
import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from predictor.cloud import archive
from predictor.cloud.archive import (
    ArchiveResult,
    SecretLeak,
    archive_run,
    assert_no_secret,
    asset_name,
    bundle,
    release_tag,
    upload,
)

# Built at runtime so no literal 32-hex run sits in a tracked file.
FAKE_KEY = "0123456789abcdef" * 2


def dt(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)


class FakeRunner:
    """Records every argv; answers from a list of (returncode, stdout,
    stderr) per subcommand ("view", "create", "upload")."""

    def __init__(self, view=(0, '{"assets":[]}', ""), create=(0, "", ""),
                 upload=(0, "", ""), raise_on=None):
        self.answers = {"view": view, "create": create, "upload": upload}
        self.raise_on = raise_on or {}
        self.calls: list[list[str]] = []
        self.kwargs: list[dict] = []

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        self.kwargs.append(kwargs)
        sub = argv[2]
        if sub in self.raise_on:
            raise self.raise_on[sub]
        code, out, err = self.answers[sub]
        return subprocess.CompletedProcess(argv, code, out, err)


def make_raw(raw_dir: Path, blobs: dict[str, bytes]) -> None:
    src = raw_dir / "espn_scoreboard"
    (src / "blobs").mkdir(parents=True)
    lines = []
    for key, data in blobs.items():
        (src / "blobs" / key).write_bytes(data)
        lines.append(json.dumps({"key": key}) + "\n")
    (src / "_manifest.jsonl").write_text("".join(lines))


def make_asset(tmp_path: Path, name="run-20261022T1307Z-a1.tar.gz",
               content=b"{}") -> Path:
    raw = tmp_path / "rawsrc" / "raw"
    make_raw(raw, {"20261022": content})
    return bundle(raw, tmp_path / "out" / name)


# --- names -----------------------------------------------------------------


def test_release_tag_is_iso_week():
    assert release_tag(dt("2026-10-22T13:07Z")) == "raw-2026-W43"
    assert release_tag(dt("2027-01-01T00:07Z")) == "raw-2026-W53"
    assert release_tag(dt("2026-01-01T00:00Z")) == "raw-2026-W01"


def test_release_tag_uses_utc():
    # 23:30 on Sunday in New York is already Monday (next ISO week) in UTC.
    from zoneinfo import ZoneInfo

    local = datetime(2026, 10, 25, 23, 30, tzinfo=ZoneInfo("America/New_York"))
    assert release_tag(local) == "raw-2026-W44"


def test_asset_name():
    assert asset_name(dt("2026-10-22T13:07Z"), 1) == "run-20261022T1307Z-a1.tar.gz"


def test_naive_datetime_is_refused():
    with pytest.raises(ValueError):
        release_tag(datetime(2026, 10, 22, 13, 7))


# --- bundle ------------------------------------------------------------------


def test_bundle_contains_manifest_and_blobs(tmp_path):
    raw = tmp_path / "data" / "raw"
    make_raw(raw, {"20261022": b'{"events":[]}', "20261023": b'{"events":[1]}'})
    out = bundle(raw, tmp_path / "out" / "a.tar.gz")
    assert out == tmp_path / "out" / "a.tar.gz"
    with tarfile.open(out, "r:gz") as tar:
        names = set(tar.getnames())
        assert "raw/espn_scoreboard/_manifest.jsonl" in names
        assert "raw/espn_scoreboard/blobs/20261022" in names
        assert "raw/espn_scoreboard/blobs/20261023" in names
        data = tar.extractfile("raw/espn_scoreboard/blobs/20261023").read()
    assert data == b'{"events":[1]}'


def test_bundle_of_empty_or_missing_dir_is_none(tmp_path):
    assert bundle(tmp_path / "missing", tmp_path / "a.tar.gz") is None
    (tmp_path / "empty" / "sub").mkdir(parents=True)
    assert bundle(tmp_path / "empty", tmp_path / "a.tar.gz") is None
    assert not (tmp_path / "a.tar.gz").exists()


def test_bundle_refuses_output_inside_raw_dir(tmp_path):
    raw = tmp_path / "raw"
    make_raw(raw, {"k": b"x"})
    with pytest.raises(ValueError):
        bundle(raw, raw / "a.tar.gz")


# --- secret guard ------------------------------------------------------------


def test_assert_no_secret_is_a_noop_without_a_secret(tmp_path):
    f = tmp_path / "a"
    f.write_text(FAKE_KEY)
    assert_no_secret([f], None)
    assert_no_secret([f], "")


def test_clean_files_pass(tmp_path):
    f = tmp_path / "a"
    f.write_text("sha256 " + "ab" * 32 + "\n")
    assert_no_secret([f], FAKE_KEY)


def test_exact_secret_is_refused_and_not_echoed(tmp_path):
    f = tmp_path / "blob"
    f.write_bytes(b'{"url":"https://x/odds?apiKey=' + FAKE_KEY.encode() + b'"}')
    with pytest.raises(SecretLeak) as exc:
        assert_no_secret([f], FAKE_KEY)
    assert FAKE_KEY not in str(exc.value)
    assert "blob" in str(exc.value)


def test_standalone_hex_equal_to_secret_in_other_case_is_refused(tmp_path):
    f = tmp_path / "blob"
    f.write_text(f"key={FAKE_KEY.upper()}\n")
    with pytest.raises(SecretLeak):
        assert_no_secret([f], FAKE_KEY)


def test_directories_are_scanned_recursively(tmp_path):
    raw = tmp_path / "raw"
    make_raw(raw, {"clean": b"x", "dirty": FAKE_KEY.encode()})
    with pytest.raises(SecretLeak):
        assert_no_secret([raw], FAKE_KEY)


def test_secret_in_raw_blob_refuses_upload(tmp_path):
    raw = tmp_path / "data" / "raw"
    make_raw(raw, {"20261022": b'{"apiKey":"' + FAKE_KEY.encode() + b'"}'})
    runner = FakeRunner()
    with pytest.raises(SecretLeak):
        archive_run(raw, tmp_path / "out", dt("2026-10-22T13:07Z"), 1,
                    secret=FAKE_KEY, runner=runner)
    assert runner.calls == []
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())


def test_upload_scans_the_tarball_members_too(tmp_path):
    asset = make_asset(tmp_path, content=FAKE_KEY.encode())
    runner = FakeRunner()
    with pytest.raises(SecretLeak):
        upload("raw-2026-W43", asset, runner=runner, secret=FAKE_KEY)
    assert runner.calls == []


# --- upload ------------------------------------------------------------------


def test_upload_to_existing_release(tmp_path):
    asset = make_asset(tmp_path)
    runner = FakeRunner()
    result = upload("raw-2026-W43", asset, runner=runner)
    assert result == ArchiveResult(ok=True, message=result.message)
    assert result.ok
    assert [c[:3] for c in runner.calls] == [
        ["gh", "release", "view"],
        ["gh", "release", "upload"],
    ]
    assert runner.calls[1] == ["gh", "release", "upload", "raw-2026-W43", str(asset)]


def test_upload_never_clobbers(tmp_path):
    asset = make_asset(tmp_path)
    for runner in (FakeRunner(), FakeRunner(view=(1, "", "release not found"))):
        upload("raw-2026-W43", asset, runner=runner)
        assert runner.calls
        assert not any("--clobber" in arg for call in runner.calls for arg in call)


def test_every_call_has_a_120s_timeout(tmp_path):
    asset = make_asset(tmp_path)
    runner = FakeRunner(view=(1, "", "release not found"))
    upload("raw-2026-W43", asset, runner=runner)
    assert len(runner.kwargs) == 3
    assert all(kw.get("timeout") == 120 for kw in runner.kwargs)


def test_release_created_once_when_missing(tmp_path):
    asset = make_asset(tmp_path)
    runner = FakeRunner(view=(1, "", "release not found"))
    result = upload("raw-2026-W43", asset, runner=runner)
    assert result.ok
    assert len(runner.calls) == 3
    assert runner.calls[1] == [
        "gh", "release", "create", "raw-2026-W43",
        "--title", "Raw archive raw-2026-W43",
        "--notes", "Append-only raw source responses, one asset per cloud run.",
        "--prerelease", "--latest=false",
    ]
    assert runner.calls[2][:4] == ["gh", "release", "upload", "raw-2026-W43"]


def test_view_failing_for_another_reason_does_not_create(tmp_path):
    asset = make_asset(tmp_path)
    runner = FakeRunner(view=(1, "", "HTTP 401: Bad credentials"))
    result = upload("raw-2026-W43", asset, runner=runner)
    assert not result.ok
    assert "Bad credentials" in result.message
    assert len(runner.calls) == 1


def test_existing_asset_name_is_suffixed_not_replaced(tmp_path):
    asset = make_asset(tmp_path)
    existing = {"assets": [{"name": asset.name},
                           {"name": "run-20261022T1307Z-a1-2.tar.gz"}]}
    runner = FakeRunner(view=(0, json.dumps(existing), ""))
    result = upload("raw-2026-W43", asset, runner=runner)
    assert result.ok
    uploaded = Path(runner.calls[-1][4])
    assert uploaded.name == "run-20261022T1307Z-a1-3.tar.gz"
    assert uploaded.read_bytes() == asset.read_bytes()
    assert not any("--clobber" in a for c in runner.calls for a in c)


def test_full_release_is_not_uploaded_to(tmp_path):
    asset = make_asset(tmp_path)
    full = {"assets": [{"name": f"x{i}"} for i in range(archive.MAX_ASSETS)]}
    runner = FakeRunner(view=(0, json.dumps(full), ""))
    result = upload("raw-2026-W43", asset, runner=runner)
    assert not result.ok
    assert len(runner.calls) == 1


def test_failures_become_results_not_exceptions(tmp_path):
    asset = make_asset(tmp_path)
    r = upload("t", asset, runner=FakeRunner(upload=(1, "", "asset already exists")))
    assert not r.ok and "already exists" in r.message
    r = upload("t", asset, runner=FakeRunner(
        raise_on={"view": subprocess.TimeoutExpired(["gh"], 120)}))
    assert not r.ok
    r = upload("t", asset, runner=FakeRunner(
        raise_on={"view": FileNotFoundError("gh")}))
    assert not r.ok
    r = upload("t", asset, runner=FakeRunner(
        view=(1, "", "release not found"), create=(1, "", "boom")))
    assert not r.ok and "boom" in r.message


def test_credentials_never_reach_the_message(tmp_path, monkeypatch):
    token = "ghs_" + "Z" * 36
    monkeypatch.setenv("GH_TOKEN", token)
    monkeypatch.setenv("ODDS_API_KEY", FAKE_KEY)
    asset = make_asset(tmp_path)
    err = (f"failed https://x-access-token:{token}@github.com/o/r "
           f"token {token} key {FAKE_KEY}")
    r = upload("t", asset, runner=FakeRunner(upload=(1, "", err)))
    assert not r.ok
    assert token not in r.message
    assert FAKE_KEY not in r.message


# --- archive_run ---------------------------------------------------------------


def test_archive_run_bundles_and_uploads(tmp_path):
    raw = tmp_path / "data" / "raw"
    make_raw(raw, {"20261022": b"{}"})
    runner = FakeRunner()
    result = archive_run(raw, tmp_path / "out", dt("2026-10-22T13:07Z"), 2,
                         secret=FAKE_KEY, runner=runner)
    assert result.ok
    assert runner.calls[0][:4] == ["gh", "release", "view", "raw-2026-W43"]
    assert Path(runner.calls[-1][4]).name == "run-20261022T1307Z-a2.tar.gz"


def test_archive_run_with_nothing_raw_uploads_nothing(tmp_path):
    runner = FakeRunner()
    result = archive_run(tmp_path / "raw", tmp_path / "out",
                         dt("2026-10-22T13:07Z"), 1, secret=None, runner=runner)
    assert result.ok
    assert runner.calls == []
