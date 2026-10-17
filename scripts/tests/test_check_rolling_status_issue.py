#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
"""Rolling status issue decisions, against a stub gh that records its calls."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "ci" / "rolling_status_issue.sh"
TITLE = "Nightly run failing on main"
STUB = """#!/bin/sh
printf '%s\\n' "$(printf '%s ' "$@" | tr '\\n' ' ')" >> "$GH_LOG"
case "$1 $2" in
  "label list") [ -n "$STUB_LABEL" ] && echo "$STUB_LABEL"; exit 0 ;;
  "issue list") printf '%s\\n' "$STUB_ISSUES"; exit 0 ;;
esac
exit 0
"""


def run(results, label="nightly", issues=()):
    with tempfile.TemporaryDirectory() as directory:
        stub = Path(directory) / "gh"
        stub.write_text(STUB)
        stub.chmod(0o755)
        log = Path(directory) / "log"
        log.touch()
        env = dict(os.environ, PATH=f"{directory}:{os.environ['PATH']}", GH_LOG=str(log), REPO="o/r",
                   STUB_LABEL=label, STUB_ISSUES=json.dumps([{"number": n, "title": t} for n, t in issues]))
        subprocess.run(["bash", str(SCRIPT), "--title", TITLE, "--label", "nightly", "--label-description", "d",
                        "--what", "Nightly run", "--run-url", "https://run/1", *results],
                       env=env, check=True, capture_output=True)
        return [line for line in log.read_text().splitlines() if not line.startswith(("label list", "issue list"))]


class RollingStatusIssue(unittest.TestCase):
    def test_red_without_issue_creates_it_and_the_missing_label(self):
        calls = run(["a=success", "b=failure"], label="")
        self.assertEqual([c.split(" -R")[0] for c in calls], ["label create nightly", "issue create"])
        self.assertIn("- b: failure", calls[1])
        self.assertNotIn("- a:", calls[1])

    def test_red_with_issue_comments_on_the_exact_title_only(self):
        calls = run(["a=cancelled"], issues=[(9, TITLE + " (old)"), (813, TITLE)])
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0].startswith("issue comment 813 "))

    def test_green_closes_the_open_issue(self):
        calls = run(["a=success", "b=success"], issues=[(813, TITLE)])
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0].startswith("issue close 813 "))
        self.assertIn("Nightly run on `main` is green again", calls[0])

    def test_green_without_issue_and_near_matches_do_nothing(self):
        self.assertEqual(run(["a=success"]), [])
        self.assertEqual(run(["a=success"], issues=[(9, TITLE.lower())]), [])

    def test_skipped_callee_counts_as_red(self):
        calls = run(["a=skipped"])
        self.assertTrue(calls[0].startswith("issue create"))


if __name__ == "__main__":
    unittest.main()
