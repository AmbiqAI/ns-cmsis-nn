#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2024-2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Mutation tests for scripts/check_toolchain_manifest_sync.py.
#
# The defect this guard exists for had already happened and was sitting in
# main: ci/tools/manifest.json pinned armclang 6.24.0 / ATfE 20.1.0 for the
# published dev container, while ci/toolchains/*.json pinned armclang 6.23.32
# / ATfE 19.1.5 for the compilers the shipped static libraries are actually
# built with. Nothing anywhere related the two files, so nothing noticed.
#
# A guard is only worth having if it can fail, so this suite pins that it:
#
#   - fires on exactly the historical skew (test_mutation_historical_skew_is_caught,
#     the required mutation case) and names both files AND both values, which
#     is the difference between a message someone can act on and one that
#     sends them hunting;
#   - fires on either compiler independently, so a one-sided bump is caught
#     (test_mutation_armclang_only / test_mutation_atfe_only);
#   - fires when the versions agree but the sha256 does not -- same version
#     label, different bytes, which a version-only check waves through and
#     which reintroduces the whole reproducibility hole
#     (test_mutation_same_version_different_sha_is_caught);
#   - does NOT fire on the aligned tree (test_repo_is_clean), which is the
#     only thing standing between this suite and "the check accidentally
#     always passes"; and
#   - fails loudly rather than vacuously when discovery breaks -- a renamed
#     tool id, a missing manifest, malformed JSON -- because a guard that
#     passes over nothing is worse than no guard, it is a false assurance.
#
# Run with: python3 scripts/tests/test_check_toolchain_manifest_sync.py

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "check_toolchain_manifest_sync.py"


def load_checker():
    spec = importlib.util.spec_from_file_location("check_toolchain_manifest_sync", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# The real, aligned values. Both sha256 values were verified by downloading
# the artifacts and hashing them, not copied from a neighbouring file.
ARMCLANG_VERSION = "6.23.32"
ARMCLANG_SHA = "48b2f02ac8941dcac97a5a792b5ddbc9cefcde7b95a39b681ee13f9b321fffe8"
ATFE_VERSION = "19.1.5"
ATFE_SHA = "34ee877aadc78c5e9f067e603a1bc9745ed93ca7ae5dbfc9b4406508dc153920"

# The exact skew that was sitting in main before this change.
HISTORICAL_ARMCLANG_SKEW = "6.24.0"
HISTORICAL_ATFE_SKEW = "20.1.0"


def tools_manifest(armclang_version=ARMCLANG_VERSION, armclang_sha=ARMCLANG_SHA,
                   atfe_version=ATFE_VERSION, atfe_sha=ATFE_SHA) -> dict:
    """A ci/tools/manifest.json shaped like the real one, trimmed to the two
    entries this check looks at plus one it must ignore."""
    return {
        "schema_version": 1,
        "platform": "linux-x86_64",
        "tools": [
            {
                "id": "cmake",
                "version": "3.31.5",
                "url": "https://example.invalid/cmake.tar.gz",
                "sha256": "0" * 64,
                "archive": "tar.gz",
                "strip_components": 1,
                "probe": "bin/cmake",
                "path": "bin",
                "license": "BSD-3-Clause",
            },
            {
                "id": "llvm-embedded",
                "version": atfe_version,
                "url": "https://example.invalid/atfe.tar.xz",
                "sha256": atfe_sha,
                "archive": "tar.xz",
                "strip_components": 1,
                "probe": "bin/clang",
                "path": "bin",
                "license": "Apache-2.0 WITH LLVM-exception",
            },
            {
                "id": "armclang",
                "version": armclang_version,
                "url": "https://example.invalid/armclang.tar.gz",
                "sha256": armclang_sha,
                "archive": "tar.gz",
                "strip_components": 0,
                "probe": "bin/armclang",
                "path": "bin",
                "license": "Arm Compiler for Embedded EULA",
            },
        ],
    }


def toolchain_json(tool_id: str, version: str, sha256: str) -> dict:
    return {
        "id": tool_id,
        "version": version,
        "compiler_id": "ARMClang" if tool_id == "armclang" else "Clang",
        "platforms": {
            "linux-x86_64": {
                "archive": f"{tool_id}-archive",
                "url": f"https://example.invalid/{tool_id}",
                "sha256": sha256,
                "probe": "bin/armclang" if tool_id == "armclang" else "bin/clang",
            }
        },
    }


class ToolchainManifestSyncCase(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = load_checker()
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.tree = Path(holder.name)
        self.toolchain_dir = self.tree / "ci" / "toolchains"
        self.toolchain_dir.mkdir(parents=True)
        self.manifest_path = self.tree / "ci" / "tools" / "manifest.json"
        self.manifest_path.parent.mkdir(parents=True)

    # -- fixture helpers ---------------------------------------------------

    def write_manifest(self, data) -> None:
        self.manifest_path.write_text(
            data if isinstance(data, str) else json.dumps(data, indent=2),
            encoding="utf-8",
        )

    def write_toolchains(self, armclang_version=ARMCLANG_VERSION, armclang_sha=ARMCLANG_SHA,
                         atfe_version=ATFE_VERSION, atfe_sha=ATFE_SHA) -> None:
        (self.toolchain_dir / "armclang.json").write_text(
            json.dumps(toolchain_json("armclang", armclang_version, armclang_sha), indent=2),
            encoding="utf-8",
        )
        (self.toolchain_dir / "atfe.json").write_text(
            json.dumps(toolchain_json("atfe", atfe_version, atfe_sha), indent=2),
            encoding="utf-8",
        )

    def write_aligned(self) -> None:
        self.write_manifest(tools_manifest())
        self.write_toolchains()

    def run_check(self) -> list[str]:
        self.mod.failures.clear()
        self.mod._stats.clear()
        self.mod.check_toolchain_manifest_sync(
            tools_manifest=self.manifest_path, toolchain_dir=self.toolchain_dir
        )
        return list(self.mod.failures)

    def assertClean(self) -> None:
        got = self.run_check()
        self.assertEqual(got, [], f"expected no failures but got: {got}")

    def assertFails(self, *fragments: str) -> list[str]:
        got = self.run_check()
        self.assertTrue(got, "expected a failure but check passed")
        joined = " ".join(got)
        for fragment in fragments:
            self.assertIn(fragment, joined, f"{fragment!r} missing from failure message: {got}")
        return got

    # -- the real tree -----------------------------------------------------

    def test_repo_is_clean(self):
        """The real ci/tools/manifest.json against the real
        ci/toolchains/*.json. This is the actual regression pin: it is what
        fails if the container and the shipped compilers ever drift apart
        again."""
        mod = load_checker()
        mod.failures.clear()
        mod._stats.clear()
        mod.check_toolchain_manifest_sync()
        self.assertEqual(mod.failures, [])
        # Both pairs must actually have been compared -- one silently skipped
        # pair would mean this passes while covering half the surface.
        self.assertEqual(mod._stats.get("pairs_checked"), 2)

    def test_real_manifests_agree_independently_of_the_checker(self):
        """Belt-and-braces: read the three real files directly and compare,
        without going through the check's own pairing logic. If the checker
        had a bug that made it compare a value with itself, test_repo_is_clean
        would still pass and this would not."""
        manifest = json.loads((REPO / "ci" / "tools" / "manifest.json").read_text(encoding="utf-8"))
        by_id = {t["id"]: t for t in manifest["tools"]}
        for tool_id, filename in (("armclang", "armclang.json"), ("llvm-embedded", "atfe.json")):
            shipped = json.loads(
                (REPO / "ci" / "toolchains" / filename).read_text(encoding="utf-8")
            )
            self.assertEqual(
                by_id[tool_id]["version"],
                shipped["version"],
                f"{tool_id}: container manifest and {filename} disagree on version",
            )
            self.assertEqual(
                by_id[tool_id]["sha256"].lower(),
                shipped["platforms"]["linux-x86_64"]["sha256"].lower(),
                f"{tool_id}: container manifest and {filename} point at different archives",
            )

    def test_aligned_fixture_is_clean(self):
        """Confirms the mutation tests below are detecting the skew they
        inject, not tripping over the fixture itself."""
        self.write_aligned()
        self.assertClean()
        self.assertEqual(self.mod._stats.get("pairs_checked"), 2)

    # -- the required mutation case ----------------------------------------

    def test_mutation_historical_skew_is_caught(self):
        """The exact state that was sitting in main: container on armclang
        6.24.0 / ATfE 20.1.0, shipped on 6.23.32 / 19.1.5. Both compilers
        must be reported, not just the first one found -- otherwise fixing
        this becomes a two-round game of whack-a-mole."""
        self.write_manifest(
            tools_manifest(
                armclang_version=HISTORICAL_ARMCLANG_SKEW,
                atfe_version=HISTORICAL_ATFE_SKEW,
            )
        )
        self.write_toolchains()
        got = self.assertFails()
        self.assertEqual(len(got), 2, f"expected both compilers reported, got: {got}")

    def test_failure_message_names_both_files_and_both_values(self):
        """The message has to identify both manifests and both versions. A
        bare 'toolchain versions differ' leaves the reader to go and find
        which two of the repo's several version pins are meant."""
        self.write_manifest(tools_manifest(armclang_version=HISTORICAL_ARMCLANG_SKEW))
        self.write_toolchains()
        self.assertFails(
            "ci/tools/manifest.json",
            "ci/toolchains/armclang.json",
            HISTORICAL_ARMCLANG_SKEW,
            ARMCLANG_VERSION,
        )

    def test_failure_message_states_the_safe_direction(self):
        """Whoever trips this gate has to choose which side to move. The
        message must say that moving the shipped compiler is a real toolchain
        upgrade needing re-validation, or the cheap fix looks like editing
        whichever file is closer to hand."""
        self.write_manifest(tools_manifest(armclang_version=HISTORICAL_ARMCLANG_SKEW))
        self.write_toolchains()
        self.assertFails("artifact re-validation", "verified", "sha256")

    def test_mutation_armclang_only(self):
        """A one-sided bump of just armclang is still drift."""
        self.write_manifest(tools_manifest(armclang_version="6.25.0"))
        self.write_toolchains()
        got = self.assertFails("armclang", "6.25.0")
        self.assertEqual(len(got), 1, got)

    def test_mutation_atfe_only(self):
        """ATfE is the entry most likely to be missed, because the two files
        call it by different names -- 'llvm-embedded' in the container
        manifest, 'atfe' in the shipped one."""
        self.write_manifest(tools_manifest(atfe_version="20.1.0"))
        self.write_toolchains()
        got = self.assertFails("atfe.json", "20.1.0", ATFE_VERSION)
        self.assertEqual(len(got), 1, got)

    def test_mutation_shipped_side_moved_is_also_caught(self):
        """Symmetry: the guard must fire regardless of which file moved.
        Bumping ci/toolchains/ and forgetting the container is just as much
        drift as the reverse, and is the more dangerous direction."""
        self.write_manifest(tools_manifest())
        self.write_toolchains(armclang_version="6.25.0", armclang_sha="a" * 64)
        self.assertFails("armclang", "6.25.0")

    # -- same version, different bytes -------------------------------------

    def test_mutation_same_version_different_sha_is_caught(self):
        """Equal version labels pointing at different archives is the same
        defect in disguise, and a version-only check passes it. This is what
        a silent re-spin of a vendor artifact looks like."""
        self.write_manifest(tools_manifest(armclang_sha="b" * 64))
        self.write_toolchains()
        self.assertFails("different archives", "b" * 64, ARMCLANG_SHA)

    def test_sha_comparison_is_case_insensitive(self):
        """Hex case is not a semantic difference; failing on it would be a
        false positive, and a gate that cries wolf gets deleted."""
        self.write_manifest(tools_manifest(armclang_sha=ARMCLANG_SHA.upper()))
        self.write_toolchains()
        self.assertClean()

    # -- must fail loudly, not silently ------------------------------------

    def test_renamed_tool_id_fails_loudly(self):
        """If someone renames the 'llvm-embedded' entry, the pair silently
        stops being compared. That must be an error, not a quiet pass over a
        shrinking surface -- the exact fail-open shape this guard exists to
        avoid reproducing one layer up."""
        manifest = tools_manifest()
        for tool in manifest["tools"]:
            if tool["id"] == "llvm-embedded":
                tool["id"] = "atfe-renamed"
        self.write_manifest(manifest)
        self.write_toolchains()
        self.assertFails("llvm-embedded", "no tool with id")

    def test_missing_toolchain_file_fails_loudly(self):
        self.write_manifest(tools_manifest())
        self.write_toolchains()
        (self.toolchain_dir / "atfe.json").unlink()
        self.assertFails("ci/toolchains", "atfe.json", "does not exist")

    def test_missing_tools_manifest_fails_loudly(self):
        self.write_toolchains()
        self.assertFails("does not exist")

    def test_malformed_json_fails_loudly_rather_than_raising(self):
        """A traceback out of a merge gate is indistinguishable from a broken
        runner, so malformed JSON is reported as a failure instead."""
        self.write_manifest("{ not json at all")
        self.write_toolchains()
        self.assertFails("not valid JSON")

    def test_duplicate_tool_id_fails_loudly(self):
        """Two entries with the same id means one of them is dead config and
        nobody can tell which one the container actually installed."""
        manifest = tools_manifest()
        armclang = next(t for t in manifest["tools"] if t["id"] == "armclang")
        manifest["tools"].append(copy.deepcopy(armclang))
        self.write_manifest(manifest)
        self.write_toolchains()
        self.assertFails("more than once")

    def test_manifest_without_tools_array_fails_loudly(self):
        self.write_manifest({"schema_version": 1, "platform": "linux-x86_64"})
        self.write_toolchains()
        self.assertFails("no 'tools' array")

    def test_missing_version_field_fails_loudly(self):
        manifest = tools_manifest()
        for tool in manifest["tools"]:
            if tool["id"] == "armclang":
                del tool["version"]
        self.write_manifest(manifest)
        self.write_toolchains()
        self.assertFails("no usable 'version'")

    def test_toolchain_file_without_platform_entry_fails_loudly(self):
        """ci/toolchains/*.json nests the sha256 under
        platforms['linux-x86_64']. If that structure changes, the sha
        comparison must not silently degrade to 'nothing to compare'."""
        self.write_manifest(tools_manifest())
        self.write_toolchains()
        broken = toolchain_json("armclang", ARMCLANG_VERSION, ARMCLANG_SHA)
        del broken["platforms"]["linux-x86_64"]
        (self.toolchain_dir / "armclang.json").write_text(
            json.dumps(broken, indent=2), encoding="utf-8"
        )
        self.assertFails("could not read a sha256")

    def test_empty_pairs_would_fail_rather_than_pass_vacuously(self):
        """If PAIRS is ever emptied -- by a bad merge, or by someone
        'temporarily' commenting an entry out -- the check must say it
        compared nothing, not print OK."""
        self.mod.PAIRS = ()
        self.write_aligned()
        self.assertFails("no armclang/ATfE version pairs were compared")

    # -- exit code contract -------------------------------------------------

    def test_main_returns_zero_on_the_real_tree(self):
        mod = load_checker()
        self.assertEqual(mod.main(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
