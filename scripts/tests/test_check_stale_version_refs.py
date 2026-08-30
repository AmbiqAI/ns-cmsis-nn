#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Tests for scripts/check_stale_version_refs.py.
#
# The check it exercises is a release-time gate: on an ordinary PR the
# previous release is already absent from the tree, so the check passes
# whether or not its scanner works at all. A gate whose green result carries
# no information until the one day a year it matters is exactly the shape of
# the bug it was written for (#347), so its scanner, its boundary handling
# and its allowlist are pinned here against synthetic trees where both sides
# of every comparison are controlled.
#
# Each test builds a real git repository in a temp dir -- the check reads the
# file set from `git ls-files`, and faking that out would skip the part most
# likely to be wrong (untracked build output must not be scanned, tracked
# files in any directory must be).
#
# The literal versions below (7.30.0 current, 7.29.2 previous) are synthetic
# fixture content, not the live repo's versions, and must not be updated when
# the repo releases.
#
# Run with: python3 scripts/tests/test_check_stale_version_refs.py

from __future__ import annotations

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO / "scripts" / "check_stale_version_refs.py"

HEADER = """\
#define NS_CMSIS_NN_VERSION_MAJOR (7)  /* x-release-please-major */
#define NS_CMSIS_NN_VERSION_MINOR (30) /* x-release-please-minor */
#define NS_CMSIS_NN_VERSION_PATCH (0)  /* x-release-please-patch */
"""

CHANGELOG = """\
# Changelog

## [7.30.0](https://github.com/AmbiqAI/ns-cmsis-nn/compare/v7.29.2...v7.30.0) (2026-08-28)

### Features

* something

## [7.29.2](https://github.com/AmbiqAI/ns-cmsis-nn/compare/v7.29.1...v7.29.2) (2026-08-09)

### Bug Fixes

* something else
"""


def _load_module():
    """Load the check under a private module name so each test gets a fresh
    copy of its module-level `failures` list and can monkeypatch
    REPO/TYPES_H/CHANGELOG/ALLOWED without affecting any other import."""
    spec = importlib.util.spec_from_file_location("check_stale_version_refs_under_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StaleVersionRefs(unittest.TestCase):
    def setUp(self):
        self.module = _load_module()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        self.module.REPO = self.root
        self.module.TYPES_H = self.root / "Include" / "arm_nn_types.h"
        self.module.CHANGELOG = self.root / "CHANGELOG.md"
        self.module.ALLOWED = {"CHANGELOG.md": "the release history itself"}
        self.module.failures = []
        self.write("Include/arm_nn_types.h", HEADER)
        self.write("CHANGELOG.md", CHANGELOG)

    def write(self, rel: str, content: str, track: bool = True) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        if track:
            subprocess.run(["git", "-C", str(self.root), "add", "-f", rel], check=True)

    def run_check(self) -> list[str]:
        canonical = self.module.parse_header_version(
            self.module.TYPES_H.read_text(encoding="utf-8")
        )
        prev = self.module.previous_version(
            self.module.CHANGELOG.read_text(encoding="utf-8"), canonical
        )
        if prev is not None:
            self.module.scan(prev)
        return list(self.module.failures)

    # -- accepted -------------------------------------------------------

    def test_clean_tree_passes(self):
        self.write("docs/x.md", "VERSION=7.30.0 # x-release-please-version\n")
        self.assertEqual(self.run_check(), [])

    def test_allowlisted_file_may_mention_the_previous_version(self):
        # CHANGELOG.md itself names 7.29.2 in both its own heading and the
        # 7.30.0 compare link, and must not be flagged for it.
        self.assertEqual(self.run_check(), [])

    def test_older_release_is_not_flagged(self):
        # Only the immediately previous release is evidence of a missed
        # bump; a doc comment naming a much older release is normal.
        self.write("Include/other.h", '#  error "needs ns-cmsis-nn >= 7.24.0"\n')
        self.assertEqual(self.run_check(), [])

    def test_longer_versions_containing_the_previous_one_are_not_flagged(self):
        self.write("docs/x.md", "a 17.29.2 b 7.29.20 c\n")
        self.assertEqual(self.run_check(), [])

    def test_untracked_file_is_not_scanned(self):
        # Build output and downloaded fixtures routinely contain old
        # versions; only tracked files are the repo's to keep correct.
        self.write("build/stale.txt", "7.29.2\n", track=False)
        self.assertEqual(self.run_check(), [])

    # -- rejected -------------------------------------------------------

    def test_unbumped_doc_is_rejected(self):
        # The #347 bug: an extra-file release-please silently skipped, so it
        # still carries the previous release after the bump.
        self.write("docs/guides/toolchains.md", '  "version": "7.29.2",\n')
        failures = self.run_check()
        self.assertTrue(
            any("toolchains.md:1" in f and "7.29.2" in f for f in failures), failures
        )

    def test_tag_form_is_rejected(self):
        self.write("docs/y.md", "      revision: v7.29.2\n")
        failures = self.run_check()
        self.assertTrue(any("docs/y.md:1" in f for f in failures), failures)

    def test_filename_suffix_form_is_rejected(self):
        # Documentation/build.md's `libns-cmsis-nn-cortex-m4-7.29.2.a`: the
        # version is followed by `.a`, which a naive right boundary of
        # "not a digit or dot" would let through.
        self.write("z.md", "  LIBRARY ${DIR}/libns-cmsis-nn-cortex-m4-7.29.2.a\n")
        failures = self.run_check()
        self.assertTrue(any("z.md:1" in f for f in failures), failures)

    def test_line_number_and_offending_line_are_reported(self):
        self.write("docs/w.md", "intro\n\nsee 7.29.2 here\n")
        failures = self.run_check()
        self.assertTrue(any("docs/w.md:3" in f for f in failures), failures)
        self.assertTrue(any("see 7.29.2 here" in f for f in failures), failures)

    def test_changelog_disagreeing_with_header_is_rejected(self):
        self.write(
            "Include/arm_nn_types.h",
            HEADER.replace("NS_CMSIS_NN_VERSION_MINOR (30)", "NS_CMSIS_NN_VERSION_MINOR (31)"),
        )
        failures = self.run_check()
        self.assertTrue(
            any("newest heading is '7.30.0'" in f and "7.31.0" in f for f in failures), failures
        )

    def test_changelog_with_one_release_is_rejected(self):
        self.write("CHANGELOG.md", "# Changelog\n\n## [7.30.0](x) (2026-08-28)\n")
        failures = self.run_check()
        self.assertTrue(any("need at least two" in f for f in failures), failures)

    def test_allowlist_entry_that_left_the_tree_is_rejected(self):
        self.module.ALLOWED = dict(self.module.ALLOWED)
        self.module.ALLOWED["docs/renamed-away.md"] = "why"
        failures = self.run_check()
        self.assertTrue(
            any("renamed-away.md" in f and "not in the tree" in f for f in failures), failures
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
