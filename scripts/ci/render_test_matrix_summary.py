#!/usr/bin/env python3
"""Render the per-leg helia-core-tester matrix as a step-summary grid.

Reads the JUnit XML files from the downloaded reports-* artifacts and
writes a CPU x leg status grid, plus the names of any failing cases, to
GITHUB_STEP_SUMMARY. Pure reporting: this script must NEVER fail the
matrix-summary job, because ci.yml gates on the helia-core-tester
reusable workflow as a single unit and a summary bug must not redden
`CI Passed` (AmbiqAI/ns-cmsis-nn#356). Rendering problems are written
INTO the summary instead of raised out of it, and the whole invocation is
wrapped so an unwritable summary file degrades to stdout rather than a
nonzero exit; the workflow's continue-on-error layers backstop what a
script cannot (argparse misuse, interpreter failure).

Stdlib only -- the job runs on a bare ubuntu runner with no container
and must not grow action or pip dependencies.

JUnit schema (observed from real artifacts, not the spec): each
reports-<profile>/tests/<suite>/<cpu>/junit.xml has a single <testsuite>
root (no <testsuites> wrapper) with tests/failures/errors/skipped
attributes; passing <testcase> elements are self-closing, failing ones
carry a <failure> or <error> child. Counts are recomputed from the
testcase elements rather than trusted from the root attributes.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# One entry per artifact the tester workflow uploads: artifact name ->
# (row, column). Rows are CPUs; columns are legs (suite x build flavour).
# A toolchain variant added later (#340) is one COLUMNS entry plus one
# EXPECTED line per CPU running it -- keep the two in sync (an EXPECTED
# column absent from COLUMNS would drop out of the grid; the unreported-
# artifact line below catches the reverse direction). No restructuring.
EXPECTED = {
    "reports-cortex-m0": ("m0", "int"),
    "reports-cortex-m4": ("m4", "int"),
    "reports-cortex-m55": ("m55", "int"),
    "reports-m4-int-shipped": ("m4", "int-shipped"),
    "reports-m55-int-shipped": ("m55", "int-shipped"),
    "reports-m4-f32-fallback": ("m4", "f32-fallback"),
    "reports-m55-f32-fallback": ("m55", "f32-fallback"),
    "reports-m55-f32-mvef": ("m55", "f32-mvef"),
    "reports-m55-f32-shipped": ("m55", "f32-shipped"),
    "reports-m55-f16-fallback": ("m55", "f16-fallback"),
    "reports-m55-f16-mvef": ("m55", "f16-mvef"),
}
ROWS = ["m0", "m4", "m55"]
COLUMNS = [
    "int",
    "int-shipped",
    "f32-fallback",
    "f32-mvef",
    "f32-shipped",
    "f16-fallback",
    "f16-mvef",
]
MAX_NAMED_FAILURES_PER_LEG = 50


def parse_profile(profile_dir: Path):
    """Aggregate every junit.xml under one reports-<profile> directory.

    Returns (passed, failed, skipped, total, failing_names) or None when
    no parseable junit.xml exists -- which the caller renders as a
    missing leg, never as a silently blank cell.
    """
    passed = failed = skipped = total = 0
    failing = []
    found = False
    for junit in sorted(profile_dir.rglob("junit.xml")):
        try:
            root = ET.parse(junit).getroot()
        except ET.ParseError:
            continue
        found = True
        # Accept both a bare <testsuite> root (what the tester emits
        # today) and a <testsuites> wrapper (the common JUnit shape),
        # so a tester-side format change degrades gracefully.
        suites = [root] if root.tag == "testsuite" else root.iter("testsuite")
        for suite in suites:
            for case in suite.iter("testcase"):
                total += 1
                if case.find("failure") is not None or case.find("error") is not None:
                    failed += 1
                    failing.append(case.get("name") or "<unnamed>")
                elif case.find("skipped") is not None:
                    skipped += 1
                else:
                    passed += 1
    return (passed, failed, skipped, total, failing) if found else None


def render(reports_root: Path):
    results = {}  # (row, col) -> parse_profile result or None
    for artifact, cell in EXPECTED.items():
        profile_dir = reports_root / artifact
        results[cell] = parse_profile(profile_dir) if profile_dir.is_dir() else None

    lines = ["## Test matrix", ""]
    lines.append("| CPU | " + " | ".join(COLUMNS) + " |")
    lines.append("|---" * (len(COLUMNS) + 1) + "|")
    in_matrix = set(EXPECTED.values())
    for row in ROWS:
        cells = []
        for col in COLUMNS:
            if (row, col) not in in_matrix:
                cells.append("—")  # leg not in the matrix for this CPU
                continue
            res = results[(row, col)]
            if res is None:
                # Expected artifact absent or unreadable: the leg failed
                # before it could report, or the upload failed. Render
                # loudly -- a missing leg must never look like "not run".
                cells.append("❌ missing")
                continue
            p, f, s, t, _ = res
            note = f" ({s} skipped)" if s else ""
            if t == 0:
                # A parseable junit with zero testcases means the leg ran
                # nothing -- that must not read as a pass.
                cells.append("⚠️ 0 ran")
            elif f:
                cells.append(f"❌ {f}/{t}{note}")
            else:
                cells.append(f"✅ {p}{note}")
        lines.append(f"| {row} | " + " | ".join(cells) + " |")

    failures_section = []
    for (row, col), res in sorted(results.items()):
        if res and res[4]:
            names = res[4]
            failures_section.append(f"\n### Failing cases: {row} / {col} ({len(names)})")
            for name in names[:MAX_NAMED_FAILURES_PER_LEG]:
                failures_section.append(f"- `{name}`")
            extra = len(names) - MAX_NAMED_FAILURES_PER_LEG
            if extra > 0:
                failures_section.append(f"- +{extra} more")
    missing = [f"{r}/{c}" for (r, c), res in sorted(results.items()) if res is None]
    if missing:
        failures_section.append(
            "\n**Missing report artifacts:** " + ", ".join(missing)
        )
    # A reports-* artifact not in EXPECTED means a leg was added to the
    # workflow without a grid mapping: surface it rather than silently
    # omitting the new leg exactly when the matrix grows.
    unknown = sorted(
        d.name
        for d in reports_root.glob("reports-*")
        if d.is_dir() and d.name not in EXPECTED
    ) if reports_root.is_dir() else []
    if unknown:
        failures_section.append(
            "\n**Unreported artifacts (add to EXPECTED in "
            "scripts/ci/render_test_matrix_summary.py):** " + ", ".join(unknown)
        )
    return "\n".join(lines + failures_section) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=os.environ.get("GITHUB_STEP_SUMMARY"),
        help="defaults to $GITHUB_STEP_SUMMARY",
    )
    args = parser.parse_args()
    try:
        text = render(args.reports_root)
    except Exception as exc:  # noqa: BLE001 -- reporting must not fail the job
        text = f"## Test matrix\n\nSummary renderer error (non-gating): `{exc!r}`\n"
    try:
        if args.summary_file:
            with open(args.summary_file, "a", encoding="utf-8") as fh:
                fh.write(text)
        else:
            sys.stdout.write(text)
    except OSError:  # unwritable summary file: degrade to stdout, never fail
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
