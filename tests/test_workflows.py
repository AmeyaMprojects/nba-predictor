"""Structure checks for the GitHub Actions workflows.

The workflows only run on GitHub, so these tests parse the YAML and check
the parts that matter: schedule, overlap, permissions, the publish branch,
where the odds key may appear, and when the site is rebuilt. They load with
``yaml.BaseLoader`` so ``on`` stays a string key and every value is a string.
"""

from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"
ODDS_SECRET = "${{ secrets.ODDS_API_KEY }}"


def _load(name: str) -> dict:
    return yaml.load((WORKFLOWS / name).read_text(), Loader=yaml.BaseLoader)


def _live() -> dict:
    return _load("live.yml")


def _job() -> dict:
    return _live()["jobs"]["run"]


def _steps() -> list[dict]:
    return _job()["steps"]


def _step(step_id: str) -> dict:
    return next(s for s in _steps() if s.get("id") == step_id)


def _named(name: str) -> dict:
    return next(s for s in _steps() if s.get("name") == name)


def test_live_runs_every_15_minutes_and_by_hand():
    on = _live()["on"]
    assert [entry["cron"] for entry in on["schedule"]] == ["7,22,37,52 * * * *"]
    assert "workflow_dispatch" in on


def test_live_never_overlaps():
    assert _live()["concurrency"] == {"group": "live", "cancel-in-progress": "false"}


def test_live_permissions():
    assert _live()["permissions"] == {"contents": "write", "actions": "write"}


def test_default_branch_is_shadow():
    env = _job()["env"]
    assert env["PREDICTOR_PUBLISH_BRANCH"] == (
        "${{ vars.PREDICTOR_PUBLISH_BRANCH || 'cloud-shadow' }}"
    )


def test_state_checkout_refuses_an_unknown_branch():
    script = _named("State checkout")["run"]
    assert "main|cloud-shadow" in script
    assert "exit 1" in script


def test_odds_key_only_in_the_cloud_run_step_env():
    with_key = [s for s in _steps() if s.get("env", {}).get("ODDS_API_KEY") == ODDS_SECRET]
    assert len(with_key) == 1
    assert with_key[0].get("id") == "cloud"
    for step in _steps():
        assert "ODDS_API_KEY" not in step.get("run", "")
    assert "ODDS_API_KEY" not in _job()["env"]
    assert "env" not in _live()
    # The secret is referenced exactly once in the whole file.
    assert (WORKFLOWS / "live.yml").read_text().count("secrets.") == 1


def test_no_shell_tracing():
    text = (WORKFLOWS / "live.yml").read_text()
    assert "set -x" not in text
    assert "xtrace" not in text


def test_gh_uses_the_job_token_without_prompts():
    env = _job()["env"]
    assert env["GH_TOKEN"] == "${{ github.token }}"
    assert env["GH_REPO"] == "${{ github.repository }}"
    assert env["GH_PROMPT_DISABLED"] == "1"


def test_job_has_a_timeout_and_python_314_via_cached_uv():
    assert int(_job()["timeout-minutes"]) <= 15
    uv = next(s for s in _steps() if s.get("uses", "").startswith("astral-sh/setup-uv@"))
    assert uv["with"]["python-version"] == "3.14"
    assert uv["with"]["enable-cache"] == "true"


def test_shallow_checkout_of_main():
    checkout = next(s for s in _steps() if s.get("uses", "").startswith("actions/checkout@"))
    assert checkout["with"]["ref"] == "main"
    assert checkout["with"]["fetch-depth"] == "1"


def test_state_worktree_lives_outside_the_code_checkout():
    script = _named("State checkout")["run"]
    assert '"$RUNNER_TEMP/state-repo"' in script
    assert "STATE_REPO=$GITHUB_WORKSPACE" in script
    assert "--state-repo \"$STATE_REPO\"" in _step("cloud")["run"]


def test_site_dispatched_only_when_predictions_changed():
    rebuild = _named("Rebuild site")
    condition = rebuild["if"]
    assert "steps.cloud.outputs.site_changed == 'true'" in condition
    assert "env.PREDICTOR_PUBLISH_BRANCH == 'main'" in condition  # never in shadow mode
    assert rebuild["run"].strip() == "gh workflow run site.yml --ref main"
    dispatching = [s for s in _steps() if "gh workflow run" in s.get("run", "")]
    assert dispatching == [rebuild]


def test_raw_kept_as_artifact_only_when_the_release_upload_failed():
    keep = next(s for s in _steps() if s.get("uses", "").startswith("actions/upload-artifact@"))
    assert keep["if"] == "always() && steps.cloud.outputs.raw_upload_failed == 'true'"
    assert keep["with"]["path"] == "${{ runner.temp }}/predictor-data/raw"
    assert _job()["env"]["PREDICTOR_DATA_DIR"] == "${{ runner.temp }}/predictor-data"
    assert keep["with"]["name"] == "raw-${{ github.run_id }}-${{ github.run_attempt }}"
    assert int(keep["with"]["retention-days"]) <= 90


def test_site_yml_still_has_workflow_dispatch():
    site = _load("site.yml")
    assert "workflow_dispatch" in site["on"]
    checkout = next(
        s for s in site["jobs"]["build"]["steps"] if s.get("uses", "").startswith("actions/checkout@")
    )
    assert checkout["with"]["fetch-depth"] == "0"  # git blame needs full history
