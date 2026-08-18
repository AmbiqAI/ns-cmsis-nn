#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2024-2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Tests for scripts/check_pdsc.py's check_extra_files_annotations().
#
# release-please's own generic updater (src/updaters/generic.js) leaves a
# line untouched if nothing on it matches the scope's value regex -- no
# warning, no error -- and its manifest strategy only *warns* (does not
# fail) about an extra-files path that no longer exists. Both mean a
# broken or inert extra-files entry ships silently, which is the exact
# failure class check_pdsc.py exists to catch everywhere else. A check
# that can't fail is worse than no check, so this pins both the "did we
# even wire this up" cases and the "does the annotated value agree with
# reality" cross-check against Include/arm_nn_types.h.
#
# Unlike test_check_float_cmsis_components.py's subprocess-over-a-copied-
# real-tree approach, this imports check_pdsc.py as a module and calls
# check_extra_files_annotations() directly against a small synthetic
# fixture tree, monkeypatching the module's REPO/RP_CONFIG/TYPES_H
# globals. check_pdsc.py's other checks (pack identity, licenses, Source/
# coverage) shell out to `git ls-files` and expect a full pdsc + Source/
# tree; running the whole script end-to-end per test would mean faking a
# git repo and copying the entire Source/ tree just to exercise one
# function, coupling these tests to unrelated parts of the script for no
# benefit. The values used below (7.29.2, x-release-please-* forms) are
# deliberately literal, synthetic fixture content rather than the live
# repo files: this is the one place in the repo where hardcoding a
# version string is correct, since the whole point is to control both
# sides of the comparison and assert on the outcome.
#
# Run with: python3 scripts/tests/test_check_pdsc.py

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO / "scripts" / "check_pdsc.py"

HEADER_TEMPLATE = """\
#define NS_CMSIS_NN_VERSION_MAJOR ({major})  /* x-release-please-major */
#define NS_CMSIS_NN_VERSION_MINOR ({minor}) /* x-release-please-minor */
#define NS_CMSIS_NN_VERSION_PATCH ({patch})  /* x-release-please-patch */
"""


def _load_module():
    """Load scripts/check_pdsc.py under a private module name so each
    test gets a fresh copy of its module-level `failures` list and can
    freely monkeypatch REPO/RP_CONFIG/TYPES_H without affecting any other
    import of the real script."""
    spec = importlib.util.spec_from_file_location("check_pdsc_under_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CheckExtraFilesAnnotations(unittest.TestCase):
    """Builds the minimal tree check_extra_files_annotations() needs --
    just release-please-config.json and Include/arm_nn_types.h, plus
    whatever extra-files targets a test declares -- rather than the full
    pdsc/licenses/Source/ tree the rest of check_pdsc.py needs."""

    def setUp(self):
        self.module = _load_module()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.module.REPO = self.root
        self.module.RP_CONFIG = self.root / "release-please-config.json"
        self.module.TYPES_H = self.root / "Include" / "arm_nn_types.h"
        self.module.failures = []
        # Canonical version for every test unless a test overwrites
        # Include/arm_nn_types.h itself.
        self.write("Include/arm_nn_types.h", HEADER_TEMPLATE.format(major="7", minor="29", patch="2"))

    def write(self, rel: str, content: str) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def write_config(self, extra_files: list) -> None:
        self.write(
            "release-please-config.json",
            json.dumps({"packages": {".": {"extra-files": extra_files}}}),
        )

    def run_check(self) -> list[str]:
        self.module.check_extra_files_annotations()
        return list(self.module.failures)

    # -- accepted: every annotation form actually used in this repo -----

    def test_version_annotation_matching_canonical_is_accepted(self):
        self.write("docs/x.md", "VERSION=7.29.2 # x-release-please-version\n")
        self.write_config([{"type": "generic", "path": "docs/x.md"}])
        self.assertEqual(self.run_check(), [])

    def test_major_minor_patch_annotations_matching_canonical_are_accepted(self):
        # Include/arm_nn_types.h checking itself: still confirms the
        # annotation-scanning + regex plumbing actually runs, not just
        # that comparing a file to itself is trivially true.
        self.write_config([{"type": "generic", "path": "Include/arm_nn_types.h"}])
        self.assertEqual(self.run_check(), [])

    def test_xml_style_trailing_comment_annotation_is_accepted(self):
        self.write(
            "pack.xml",
            '<component Cversion="7.29.2"/> <!-- x-release-please-version -->\n',
        )
        self.write_config([{"type": "generic", "path": "pack.xml"}])
        self.assertEqual(self.run_check(), [])

    def test_v_prefixed_version_is_accepted(self):
        # zephyr.md's `revision: v7.29.2` -- the "v" must survive as part
        # of the line; only the M.m.p substring is compared.
        self.write("west.yml", "      revision: v7.29.2 # x-release-please-version\n")
        self.write_config([{"type": "generic", "path": "west.yml"}])
        self.assertEqual(self.run_check(), [])

    def test_directory_name_embedded_version_is_accepted(self):
        # cmake.md's third_party/ns-cmsis-nn-cortex-m4-atfe-7.29.2 form.
        self.write(
            "cmake.md",
            'list(APPEND X "third_party/ns-cmsis-nn-cortex-m4-atfe-7.29.2") '
            "# x-release-please-version\n",
        )
        self.write_config([{"type": "generic", "path": "cmake.md"}])
        self.assertEqual(self.run_check(), [])

    def test_literal_only_allowlist_entry_matching_canonical_is_accepted(self):
        self.write("docs/guides/toolchains.md", '  "version": "7.29.2",\n')
        self.write_config([{"type": "generic", "path": "docs/guides/toolchains.md"}])
        self.assertEqual(self.run_check(), [])

    def test_non_generic_object_entry_is_existence_checked_but_not_annotation_checked(self):
        # A jsonpath-based (type: json/yaml/...) entry doesn't use
        # x-release-please comments at all -- must not be flagged as
        # inert -- but must still be existence-checked like any other
        # entry.
        self.write("package.json", '{"version": "0.0.0"}')
        self.write_config([{"type": "json", "path": "package.json", "jsonpath": "$.version"}])
        self.assertEqual(self.run_check(), [])

    def test_bare_string_entry_is_existence_checked_but_not_annotation_checked(self):
        self.write("VERSION.txt", "0.0.0\n")
        self.write_config(["VERSION.txt"])
        self.assertEqual(self.run_check(), [])

    # -- rejected: the failure modes release-please itself has ----------

    def test_missing_generic_extra_file_is_rejected(self):
        self.write_config([{"type": "generic", "path": "docs/does-not-exist.md"}])
        failures = self.run_check()
        self.assertTrue(
            any("does not exist on disk" in f and "does-not-exist.md" in f for f in failures),
            failures,
        )

    def test_missing_non_generic_extra_file_is_also_rejected(self):
        self.write_config([{"type": "json", "path": "missing.json", "jsonpath": "$.version"}])
        failures = self.run_check()
        self.assertTrue(
            any("does not exist on disk" in f and "missing.json" in f for f in failures), failures
        )

    def test_missing_bare_string_extra_file_is_also_rejected(self):
        self.write_config(["missing.txt"])
        failures = self.run_check()
        self.assertTrue(
            any("does not exist on disk" in f and "missing.txt" in f for f in failures), failures
        )

    def test_zero_annotation_entry_is_rejected(self):
        # The exact bug this check exists to catch: a file listed in
        # extra-files with no annotation at all, so release-please's
        # generic updater changes 0 lines in it, forever, silently.
        self.write("docs/inert.md", "VERSION=7.29.2\n")  # no annotation comment
        self.write_config([{"type": "generic", "path": "docs/inert.md"}])
        failures = self.run_check()
        self.assertTrue(
            any("zero x-release-please annotations" in f and "inert.md" in f for f in failures),
            failures,
        )

    def test_annotated_version_drifted_from_canonical_is_rejected(self):
        self.write("docs/x.md", "VERSION=7.30.0 # x-release-please-version\n")
        self.write_config([{"type": "generic", "path": "docs/x.md"}])
        failures = self.run_check()
        self.assertTrue(any("'7.30.0' != canonical '7.29.2'" in f for f in failures), failures)

    def test_annotated_minor_drifted_from_canonical_is_rejected(self):
        self.write("Include/other.h", "#define X_MINOR (99) /* x-release-please-minor */\n")
        self.write_config([{"type": "generic", "path": "Include/other.h"}])
        failures = self.run_check()
        self.assertTrue(
            any("annotated minor '99' != canonical '29'" in f for f in failures), failures
        )

    def test_literal_only_entry_drifted_from_canonical_is_rejected(self):
        self.write("docs/guides/toolchains.md", '  "version": "7.25.0",\n')
        self.write_config([{"type": "generic", "path": "docs/guides/toolchains.md"}])
        failures = self.run_check()
        self.assertTrue(
            any(
                "literal version '7.25.0' does not match canonical '7.29.2'" in f
                for f in failures
            ),
            failures,
        )

    def test_literal_only_pattern_does_not_match_nested_toolchain_version(self):
        # Regression guard for the exact hazard this PR's review caught:
        # docs/guides/toolchains.md's manifest.json example also shows an
        # unrelated ATfE compiler version ("toolchain": {"version":
        # "19.1.5"}) a few lines below the package version. The allowlist
        # pattern is anchored to exactly two leading spaces (the
        # top-level field) so it cannot pick up the nested one (four
        # leading spaces) -- if it did, this test would see a spurious
        # "does not match canonical" failure comparing 19.1.5 to 7.29.2.
        self.write(
            "docs/guides/toolchains.md",
            "{\n"
            '  "version": "7.29.2",\n'
            '  "toolchain": {\n'
            '    "version": "19.1.5"\n'
            "  }\n"
            "}\n",
        )
        self.write_config([{"type": "generic", "path": "docs/guides/toolchains.md"}])
        self.assertEqual(self.run_check(), [])

    def test_literal_only_pattern_missing_is_reported_not_silently_skipped(self):
        # If toolchains.md is ever reformatted so the anchor no longer
        # matches anything, that must fail loudly (wrong field could
        # otherwise be silently compared, or the drift could go
        # unnoticed) rather than being treated as "nothing to check".
        self.write("docs/guides/toolchains.md", '   "version": "7.29.2",\n')  # 3 spaces, not 2
        self.write_config([{"type": "generic", "path": "docs/guides/toolchains.md"}])
        failures = self.run_check()
        self.assertTrue(
            any("found no match" in f and "toolchains.md" in f for f in failures), failures
        )

    def test_empty_extra_files_is_rejected(self):
        self.write_config([])
        failures = self.run_check()
        self.assertTrue(any("no extra-files entries" in f for f in failures), failures)

    def test_unparsable_config_is_reported_cleanly_not_a_crash(self):
        self.write("release-please-config.json", "{not valid json")
        failures = self.run_check()
        self.assertTrue(any("could not read" in f for f in failures), failures)


if __name__ == "__main__":
    unittest.main(verbosity=2)
