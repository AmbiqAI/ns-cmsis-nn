#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Mutation tests for scripts/check_release_asset_manifest.py and the
# expansion in scripts/ci/release_assets.py.
#
# The defect these guard against had already happened and shipped: the
# required-asset list was transcribed by hand into release.yml's
# release-verify job, scripts/ci/audit_release_assets.sh and the table in
# docs/guides/releases.md, with nothing relating the three
# (AmbiqAI/ns-cmsis-nn#376). The copies agreed by luck, and an asset added to
# release.yml alone would have been verified once at release time and never
# audited again.
#
# A guard is only worth having if it can fail, so this suite pins that it:
#
#   - fires when the manifest declares an asset the guide's table does not
#     list, and vice versa -- the two directions of the original drift;
#   - fires when the two agree on an asset but disagree on whether it is
#     required, which is the difference between a release that is broken and
#     one that is merely incomplete;
#   - fires when the asset counts quoted in the guide's prose no longer match
#     what the manifest expands to, since a number in prose goes stale in
#     silence;
#   - fires when a consumer stops expanding the manifest, or regrows the
#     hand-built list it used to carry, because a manifest that is
#     authoritative for the docs only is not a single source;
#   - fails loudly rather than vacuously when discovery breaks -- a renamed
#     heading, a table it cannot parse, a manifest whose per-cpu rows expand
#     once instead of three times -- because a check that passes over nothing
#     is a false assurance; and
#   - does NOT fire on the real tree, which is the only thing standing
#     between this suite and "the check accidentally always passes".
#
# Run with: python3 scripts/tests/test_check_release_asset_manifest.py

from __future__ import annotations

import copy
import importlib.util
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "check_release_asset_manifest.py"
GENERATOR = REPO / "scripts" / "ci" / "release_assets.py"
REAL_MANIFEST = REPO / "ci" / "release-assets.json"


def load_checker():
    spec = importlib.util.spec_from_file_location("check_release_asset_manifest", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_generator():
    spec = importlib.util.spec_from_file_location("release_assets_under_test", GENERATOR)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def real_manifest() -> dict:
    return json.loads(REAL_MANIFEST.read_text(encoding="utf-8"))


def docs_from_manifest(manifest: dict, base: int, promoted: int) -> str:
    """A releases.md shaped like the real one: the heading this gate reads,
    a table agreeing with `manifest`, and the prose sentences that quote the
    required-asset counts."""
    rows = []
    for row in manifest["rows"]:
        status = "**Required**" if row["class"] == "required" else "**Optional**"
        rows.append(f"| {row['docs_asset']} | {status} | why |")
    for row in manifest.get("non_asset_rows", []):
        status = "**Required**" if row["class"] == "required" else "**Optional**"
        rows.append(f"| {row['docs_asset']} | {status} | why |")
    table = "\n".join(rows)
    return (
        "# Versioning & Releases\n\n"
        "## Required vs optional assets\n\n"
        "| Asset | Status | Why |\n"
        "| --- | --- | --- |\n"
        f"{table}\n\n"
        "Set it and `release-verify` promotes them, taking the required bar "
        f"from {base} assets to {promoted}; leave it unset and the bar stays "
        f"at {base}.\n\n"
        "## Next section\n"
    )


class ReleaseAssetManifestCheckTest(unittest.TestCase):
    def setUp(self):
        self.mod = load_checker()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

        self.manifest_path = self.root / "release-assets.json"
        self.doc_path = self.root / "releases.md"
        self.workflow = self.root / "release.yml"
        self.audit = self.root / "audit_release_assets.sh"

    # -- fixtures -----------------------------------------------------------

    def write_all(self, manifest=None, doc=None, workflow=None, audit=None):
        manifest = real_manifest() if manifest is None else manifest
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        gen = load_generator()
        base = len(gen.asset_names("0.0.0", "required", False, manifest))
        promoted = len(gen.asset_names("0.0.0", "required", True, manifest))
        self.doc_path.write_text(
            docs_from_manifest(manifest, base, promoted) if doc is None else doc,
            encoding="utf-8",
        )
        self.workflow.write_text(
            "run: python3 scripts/ci/release_assets.py \"$VERSION\"\n"
            if workflow is None
            else workflow,
            encoding="utf-8",
        )
        self.audit.write_text(
            "python3 \"${REPO_ROOT}/scripts/ci/release_assets.py\" \"$1\"\n"
            if audit is None
            else audit,
            encoding="utf-8",
        )

    def consumers(self):
        return (
            (self.workflow, re.compile(r"^\s*required=\(", re.MULTILINE), "the hand-built array"),
            (self.audit, re.compile(r"\bgcc\s+atfe\b"), "the hand-rolled loop"),
        )

    def run_check(self):
        self.mod.failures.clear()
        self.mod._stats.clear()
        self.mod.check_release_asset_manifest(
            manifest_path=self.manifest_path,
            doc=self.doc_path,
            consumers=self.consumers(),
        )
        return list(self.mod.failures)

    def assertFails(self, needle):
        failures = self.run_check()
        self.assertTrue(failures, f"expected a failure mentioning {needle!r}, got none")
        joined = "\n".join(failures)
        self.assertIn(needle, joined)

    def assertClean(self):
        failures = self.run_check()
        self.assertEqual(failures, [], f"expected no failures, got:\n" + "\n".join(failures))

    # -- the control --------------------------------------------------------

    def test_fixture_is_clean(self):
        """Nothing below means anything if the aligned fixture already fails."""
        self.write_all()
        self.assertClean()

    def test_real_tree_passes(self):
        mod = load_checker()
        self.assertEqual(mod.main(), 0)

    # -- the two directions of the original drift ---------------------------

    def test_asset_in_manifest_but_not_in_the_docs_table(self):
        manifest = real_manifest()
        doc = docs_from_manifest(manifest, 17, 25)
        manifest["rows"].append(
            {
                "id": "sdk-tarball-newtc",
                "template": "ns-cmsis-nn-{cpu}-newtc-{version}.tar.gz",
                "class": "required",
                "sha256_sidecar": True,
                "per_cpu": True,
                "docs_asset": "`ns-cmsis-nn-<cpu>-newtc-<version>.tar.gz`",
            }
        )
        self.write_all(manifest=manifest, doc=doc)
        self.assertFails("which the asset table in")

    def test_row_in_the_docs_table_but_not_in_the_manifest(self):
        manifest = real_manifest()
        with_ghost = copy.deepcopy(manifest)
        with_ghost["non_asset_rows"].append(
            {
                "id": "ghost",
                "class": "required",
                "docs_asset": "`ns-cmsis-nn-<cpu>-ghost-<version>.tar.gz`",
            }
        )
        self.write_all(manifest=manifest, doc=docs_from_manifest(with_ghost, 17, 25))
        self.assertFails("which is not registered in")

    def test_required_and_optional_disagree(self):
        manifest = real_manifest()
        doc = docs_from_manifest(manifest, 17, 25).replace(
            "| `ns-cmsis-nn-<cpu>-atfe-<version>.tar.gz` | **Required** |",
            "| `ns-cmsis-nn-<cpu>-atfe-<version>.tar.gz` | **Optional** |",
        )
        self.write_all(manifest=manifest, doc=doc)
        self.assertFails("is 'required' in")

    # -- the numbers quoted in prose ----------------------------------------

    def test_stale_required_count_in_prose(self):
        manifest = real_manifest()
        doc = docs_from_manifest(manifest, 16, 25)
        self.write_all(manifest=manifest, doc=doc)
        self.assertFails("expands to (17, 25)")

    def test_prose_sentence_removed_entirely(self):
        manifest = real_manifest()
        doc = docs_from_manifest(manifest, 17, 25)
        doc = doc.replace("the bar stays at 17", "the bar does not move")
        self.write_all(manifest=manifest, doc=doc)
        self.assertFails("no longer contains a sentence matching")

    # -- the consumers ------------------------------------------------------

    def test_consumer_that_stops_expanding_the_manifest(self):
        self.write_all(workflow="run: echo no generator here\n")
        self.assertFails("does not invoke scripts/ci/release_assets.py")

    def test_workflow_that_regrows_its_hand_built_array(self):
        self.write_all(
            workflow=(
                "run: python3 scripts/ci/release_assets.py \"$VERSION\"\n"
                "     required=( \"Ambiq.NS-CMSIS-NN.${VERSION}.pack\" )\n"
            )
        )
        self.assertFails("has grown the hand-built array back")

    def test_audit_that_regrows_its_toolchain_loop(self):
        self.write_all(
            audit=(
                "python3 \"${REPO_ROOT}/scripts/ci/release_assets.py\" \"$1\"\n"
                "for tc in gcc atfe; do :; done\n"
            )
        )
        self.assertFails("has grown the hand-rolled loop back")

    # -- discovery must break loudly, never vacuously -----------------------

    def test_renamed_table_heading_is_not_a_silent_pass(self):
        manifest = real_manifest()
        doc = docs_from_manifest(manifest, 17, 25).replace(
            "## Required vs optional assets", "## Assets"
        )
        self.write_all(manifest=manifest, doc=doc)
        self.assertFails("has no '## Required vs optional assets' heading")

    def test_heading_with_no_table_under_it(self):
        manifest = real_manifest()
        self.write_all(
            manifest=manifest,
            doc="# Releases\n\n## Required vs optional assets\n\nNo table here.\n",
        )
        self.assertFails("no table rows were parsed")

    def test_missing_manifest_is_reported_not_raised(self):
        self.write_all()
        self.manifest_path.unlink()
        self.assertFails("does not exist")

    # -- manifest validation ------------------------------------------------

    def test_per_cpu_row_whose_template_does_not_expand_per_cpu(self):
        """A per-cpu row that expands once silently drops two thirds of its
        assets from every check that reads the manifest, and every consumer
        would still report a complete release."""
        manifest = real_manifest()
        for row in manifest["rows"]:
            if row["id"] == "sdk-tarball-gcc":
                row["template"] = "ns-cmsis-nn-gcc-{version}.tar.gz"
        self.write_all(manifest=manifest, doc=docs_from_manifest(real_manifest(), 17, 25))
        self.assertFails("sets per_cpu=True")

    def test_row_class_outside_required_optional(self):
        manifest = real_manifest()
        manifest["rows"][0]["class"] = "probably"
        self.write_all(manifest=manifest, doc=docs_from_manifest(real_manifest(), 17, 25))
        self.assertFails("expected one of required, optional")

    def test_duplicate_row_id(self):
        manifest = real_manifest()
        manifest["rows"].append(copy.deepcopy(manifest["rows"][0]))
        self.write_all(manifest=manifest, doc=docs_from_manifest(real_manifest(), 17, 25))
        self.assertFails("more than once")


class ReleaseAssetGeneratorTest(unittest.TestCase):
    """Pins the expansion itself. The checker only proves the docs and the
    manifest agree; if the expansion were wrong they would agree on the wrong
    contract."""

    def setUp(self):
        self.gen = load_generator()

    def test_required_expansion_is_the_shipped_contract(self):
        names = self.gen.asset_names("7.31.0", "required")
        self.assertEqual(len(names), 17)
        self.assertEqual(len(set(names)), len(names), "expansion produced a duplicate name")
        self.assertIn("Ambiq.NS-CMSIS-NN.7.31.0.pack", names)
        for cpu in ("cortex-m0", "cortex-m4", "cortex-m55"):
            for tc in ("gcc", "atfe"):
                self.assertIn(f"ns-cmsis-nn-{cpu}-{tc}-7.31.0.tar.gz", names)
                self.assertIn(f"ns-cmsis-nn-{cpu}-{tc}-7.31.0.tar.gz.sha256", names)

    def test_armclang_is_optional_until_promoted(self):
        optional = self.gen.asset_names("7.31.0", "optional")
        self.assertEqual(len(optional), 8)
        self.assertTrue(all("armclang" in name for name in optional))
        promoted = self.gen.asset_names("7.31.0", "required", armclang_required=True)
        self.assertEqual(len(promoted), 25)
        self.assertEqual(
            self.gen.asset_names("7.31.0", "optional", armclang_required=True),
            [],
            "promotion must leave nothing optional, or a consumer double-counts",
        )

    def test_all_is_required_plus_optional(self):
        self.assertEqual(
            sorted(self.gen.asset_names("7.31.0", "all")),
            sorted(
                self.gen.asset_names("7.31.0", "required")
                + self.gen.asset_names("7.31.0", "optional")
            ),
        )

    def test_a_leading_v_is_rejected(self):
        """The names embed the version verbatim, so 'v7.31.0' would generate
        a contract no release can ever satisfy."""
        with self.assertRaises(SystemExit) as ctx:
            self.gen.main(["v7.31.0"])
        self.assertNotEqual(ctx.exception.code, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
