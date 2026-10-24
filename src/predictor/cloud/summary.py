"""The run summary (Markdown for ``$GITHUB_STEP_SUMMARY``) and the step
outputs the workflow reads (``$GITHUB_OUTPUT``)."""

from __future__ import annotations

from zoneinfo import ZoneInfo

from predictor.cloud.run import RunReport

IST = ZoneInfo("Asia/Kolkata")


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _result(step) -> str:
    if not step.due:
        return "—" if step.ok else "not run"
    return "ok" if step.ok else "FAILED"


def markdown(report: RunReport) -> str:
    """Heading, one table row per step, the commits and push line, then notes."""
    at = report.started_at
    lines = [
        f"## Cloud run {at:%Y-%m-%d %H:%M} UTC ({at.astimezone(IST):%H:%M} IST)"
        f" — branch {report.branch}",
        "",
        "| step | due | result | detail |",
        "| --- | --- | --- | --- |",
    ]
    for step in report.steps:
        lines.append(
            f"| {step.name} | {'yes' if step.due else 'no'} | {_result(step)} "
            f"| {_cell(step.detail)} |"
        )
    push = "pushed" if report.pushed else "not pushed"
    if report.push_message:
        push += f" ({_cell(report.push_message)})"
    lines += [
        "",
        f"state commit: {report.state_commit or 'none'} · "
        f"predictions commit: {report.predictions_commit or 'none'} · push: {push}",
    ]
    flags = []
    if report.site_changed:
        flags.append("site rebuild requested")
    if not report.raw_archived:
        flags.append("raw responses not archived to a release")
    if report.secret_leak:
        flags.append("SECRET FOUND — the run stopped; nothing more was committed or pushed")
    if report.unexpected:
        flags.append("UNEXPECTED ERROR — the run exits 1")
    if report.publish_error:
        flags.append("PUBLISH ERROR — the run exits 1")
    if flags:
        lines += ["", "; ".join(flags)]
    if report.notes:
        lines += ["", *[f"- {_cell(n)}" for n in report.notes]]
    return "\n".join(lines) + "\n"


def outputs(report: RunReport) -> dict[str, str]:
    """``site_changed`` and ``raw_upload_failed`` as "true"/"false".

    The raw upload counts as failed only when the raw_archive step ran and
    failed -- never after a secret was found, so a raw dir holding a secret
    is never uploaded as a workflow artifact either."""
    raw = next((s for s in report.steps if s.name == "raw_archive"), None)
    failed = raw is not None and raw.due and not raw.ok and not report.secret_leak
    return {
        "site_changed": "true" if report.site_changed else "false",
        "raw_upload_failed": "true" if failed else "false",
    }
