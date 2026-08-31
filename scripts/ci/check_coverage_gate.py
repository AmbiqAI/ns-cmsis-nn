#!/usr/bin/env python3
"""Gate merged line coverage on a floor and on no-regression vs main.

Runs at the end of the coverage-merge-summary job, AFTER the
coverage-merged artifact upload, so the artifact exists even when this
gate fails (AmbiqAI/ns-cmsis-nn#356, AmbiqAI/helia-core-tester#73).

Unlike matrix-summary -- which is pure reporting and wrapped in
continue-on-error at every layer so a summary bug can never redden
`CI Passed` -- this step is a GATE by design: a floor breach or a real
coverage regression MUST fail the job. Do not add continue-on-error here.

Two checks, honest-degradation contract:

1. Floor: current overall line rate must be >= line_floor_pct from the
   checked-in ci/coverage-floor.json. This check always runs and a breach
   always fails. Raising the floor is a reviewed diff to that file.

2. No-regression: current overall line rate must not be more than
   REGRESSION_TOLERANCE_PP below the rate recorded in the newest
   coverage-merged artifact of a successful main-branch run, fetched via
   the GitHub API with the job's GITHUB_TOKEN (needs actions: read, which
   the same-repo read token carries even on fork PRs). When no baseline
   can be fetched -- first run, expired artifacts, API hiccup -- this
   sub-check degrades to a LOUD warning and does not fail: the gate must
   never flake CI red (or green) on artifact retention. A fetched baseline
   that shows a regression fails for real.

Stdlib only. Exit 0 = pass, exit 1 = gate failure.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import urllib.error
import urllib.request
import zipfile

ARTIFACT_NAME = "coverage-merged"
SUMMARY_BASENAME = "coverage_merged_summary.json"
BASELINE_BRANCH = "main"
# Guards float noise between runs (per-leg lcov merge order, gcov rounding).
# A drop within this many percentage points of the baseline is not a failure.
REGRESSION_TOLERANCE_PP = 0.15
# How many main-branch artifacts to consider while looking for one whose
# workflow run actually concluded successfully.
MAX_BASELINE_CANDIDATES = 10
HTTP_TIMEOUT_S = 30
TOP_DROPPERS = 5


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Surface redirects instead of following them.

    The artifact archive_download_url 302-redirects to signed blob storage
    that rejects requests still carrying the GitHub Authorization header,
    so the redirect must be re-issued without it.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


def _api(url: str, token: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download_artifact_zip(url: str, token: str) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(req, timeout=HTTP_TIMEOUT_S) as resp:
            return resp.read()
    except urllib.error.HTTPError as err:
        if err.code in (301, 302, 303, 307, 308):
            location = err.headers.get("Location")
            if not location:
                raise RuntimeError("artifact redirect had no Location header")
            # Signed URL: no Authorization header on purpose (see _NoRedirect).
            with urllib.request.urlopen(location, timeout=HTTP_TIMEOUT_S) as resp:
                return resp.read()
        raise


def fetch_baseline(api_url: str, repo: str, token: str, current_run_id: str) -> tuple[dict | None, str]:
    """Return (baseline summary JSON, note). None + note when unavailable."""
    artifacts_url = (
        f"{api_url}/repos/{repo}/actions/artifacts"
        f"?name={ARTIFACT_NAME}&per_page=50"
    )
    listing = _api(artifacts_url, token)
    candidates = [
        art
        for art in listing.get("artifacts", [])
        if not art.get("expired")
        and (art.get("workflow_run") or {}).get("head_branch") == BASELINE_BRANCH
        and str((art.get("workflow_run") or {}).get("id")) != str(current_run_id)
    ]
    if not candidates:
        return None, f"no unexpired {ARTIFACT_NAME} artifact found on {BASELINE_BRANCH}"

    for art in candidates[:MAX_BASELINE_CANDIDATES]:
        run_id = (art.get("workflow_run") or {}).get("id")
        run = _api(f"{api_url}/repos/{repo}/actions/runs/{run_id}", token)
        if run.get("conclusion") != "success":
            continue
        # head_branch == "main" also matches pull_request runs whose FORK head
        # branch is named main, letting an external PR's artifact become the
        # baseline (phantom regressions, or a vacuous gate). Only runs of the
        # branch itself may anchor the comparison.
        if run.get("event") not in ("push", "workflow_dispatch", "schedule"):
            continue
        blob = _download_artifact_zip(art["archive_download_url"], token)
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            name = next((n for n in zf.namelist() if os.path.basename(n) == SUMMARY_BASENAME), None)
            if name is None:
                return None, f"baseline artifact from run {run_id} lacks {SUMMARY_BASENAME}"
            summary = json.loads(zf.read(name).decode("utf-8"))
        note = (
            f"baseline: run {run_id} ({run.get('head_sha', '?')[:12]}, "
            f"{art.get('created_at', '?')})"
        )
        return summary, note

    return None, (
        f"none of the newest {min(len(candidates), MAX_BASELINE_CANDIDATES)} "
        f"{BASELINE_BRANCH}-branch {ARTIFACT_NAME} artifacts belong to a successful run"
    )


def file_rates(summary: dict) -> dict[str, float]:
    return {
        entry["file"]: float(entry.get("line_rate", 0.0))
        for entry in summary.get("file_coverage", [])
        if isinstance(entry, dict) and "file" in entry
    }


def biggest_droppers(current: dict, previous: dict, limit: int = TOP_DROPPERS) -> list[str]:
    cur, prev = file_rates(current), file_rates(previous)
    drops = []
    for path, prev_rate in prev.items():
        delta = cur.get(path, 0.0) - prev_rate
        if delta < 0:
            gone = path not in cur
            drops.append((delta, path, prev_rate, cur.get(path, 0.0), gone))
    drops.sort()
    lines = []
    for delta, path, prev_rate, cur_rate, gone in drops[:limit]:
        state = "no longer in report" if gone else f"{cur_rate:.2f}%"
        lines.append(f"{path}: {prev_rate:.2f}% -> {state} ({delta:+.2f}pp)")
    return lines


def append_summary(summary_file: str | None, lines: list[str]) -> None:
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if summary_file:
        try:
            with open(summary_file, "a", encoding="utf-8") as fh:
                fh.write(text)
        except OSError as err:
            print(f"warning: could not write step summary: {err}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", required=True, help="current coverage_merged_summary.json")
    parser.add_argument("--floor-file", required=True, help="checked-in ci/coverage-floor.json")
    parser.add_argument("--repo", required=True, help="owner/repo for baseline artifact lookup")
    parser.add_argument("--summary-file", default=os.environ.get("GITHUB_STEP_SUMMARY"))
    args = parser.parse_args()

    with open(args.summary_json, encoding="utf-8") as fh:
        current = json.load(fh)
    with open(args.floor_file, encoding="utf-8") as fh:
        floor_cfg = json.load(fh)

    current_pct = float(current["overall_line_rate"])
    floor_pct = float(floor_cfg["line_floor_pct"])

    failures: list[str] = []
    warnings: list[str] = []
    details: list[str] = []

    # --- Check 1: floor (always enforced) ---
    if current_pct < floor_pct:
        failures.append(
            f"floor breach: merged line coverage {current_pct:.2f}% is below the "
            f"floor {floor_pct:.2f}% from {args.floor_file}"
        )

    # --- Check 2: no-regression vs previous successful main run ---
    previous_pct: float | None = None
    token = os.environ.get("GITHUB_TOKEN", "")
    api_url = os.environ.get("GITHUB_API_URL", "https://api.github.com")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    if not token:
        warnings.append("no GITHUB_TOKEN in the environment")
    else:
        try:
            baseline, note = fetch_baseline(api_url, args.repo, token, run_id)
        except Exception as err:  # noqa: BLE001 -- any fetch failure degrades, never fails
            baseline, note = None, f"baseline fetch failed: {err.__class__.__name__}: {err}"
        if baseline is None:
            warnings.append(note)
        else:
            # Baseline CONSUMPTION degrades like baseline fetch: a malformed
            # baseline (bad key, garbage rate) is the other side's defect and
            # must warn, not fail -- only the CURRENT summary is this gate's
            # own input and stays fail-loud.
            try:
                previous_pct = float(baseline["overall_line_rate"])
                details.append(note)
                if current_pct < previous_pct - REGRESSION_TOLERANCE_PP:
                    msg = (
                        f"regression: merged line coverage {current_pct:.2f}% dropped "
                        f"{previous_pct - current_pct:.2f}pp below the previous successful "
                        f"{BASELINE_BRANCH} run's {previous_pct:.2f}% "
                        f"(tolerance {REGRESSION_TOLERANCE_PP:.2f}pp)"
                    )
                    droppers = biggest_droppers(current, baseline)
                    if droppers:
                        msg += "; biggest file-level drops: " + "; ".join(droppers)
                    failures.append(msg)
            except Exception as err:  # noqa: BLE001 -- degrade, never flake
                previous_pct = None
                warnings.append(
                    f"baseline unusable ({err.__class__.__name__}: {err}); floor-only"
                )

    # --- Verdict, rendered right under the existing coverage table ---
    prev_str = f"{previous_pct:.2f}%" if previous_pct is not None else "unavailable"
    lines = ["", "### Coverage gate", ""]
    if failures:
        lines.append(
            f"FAIL -- current {current_pct:.2f}% | floor {floor_pct:.2f}% | "
            f"previous {BASELINE_BRANCH} {prev_str} | tolerance {REGRESSION_TOLERANCE_PP:.2f}pp"
        )
        lines += [f"- {msg}" for msg in failures]
    else:
        lines.append(
            f"PASS -- current {current_pct:.2f}% | floor {floor_pct:.2f}% | "
            f"previous {BASELINE_BRANCH} {prev_str} | tolerance {REGRESSION_TOLERANCE_PP:.2f}pp"
        )
    for msg in warnings:
        lines.append(
            f"- WARNING: no-regression check skipped ({msg}); "
            f"enforcing the floor only -- fix the baseline source, do not rely on this"
        )
    lines += [f"- {msg}" for msg in details]
    append_summary(args.summary_file, lines)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
