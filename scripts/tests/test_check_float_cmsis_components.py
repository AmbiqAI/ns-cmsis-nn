#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2024-2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Mutation tests for scripts/check_float_cmsis_components.py.
#
# That script is a merge gate, and a gate that passes when it should not
# is worse than no gate. Both of its historical failure modes are easy
# to reintroduce: the original text-matching version silently accepted
# regressions spelled with a trailing comment, and the structural
# rewrite that replaced it briefly accepted a `for-context` list that
# named the generic targets alongside the SSE-300 one.
#
# Each test copies the real csolution/clayer/cproject tree into a temp
# directory, applies one mutation, and asserts the checker's verdict.
# Cases that must stay GREEN matter as much as the ones that must fail:
# a check that cries wolf on valid input gets disabled by the next
# person to hit it.
#
# Nothing here hardcodes a pack version or a clayer line: the fixtures
# are the live files, so release-please bumps and ordinary maintenance
# edits must not turn this suite red. Anchors are located structurally
# and asserted to exist, so genuine fixture drift fails loudly rather
# than silently mutating nothing.
#
# Run with: python3 scripts/tests/test_check_float_cmsis_components.py

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT_REL = "scripts/check_float_cmsis_components.py"
CMSIS_REL = "Tests/UnitTest/cmsis"
CSOLUTION_REL = f"{CMSIS_REL}/cmsis_nn_unit_tests_flt.csolution.yml"
CLAYER_REL = f"{CMSIS_REL}/corstone300_unittest.clayer.yml"
CPROJECT_REL = f"{CMSIS_REL}/test_arm_convolve_flt/test_arm_convolve_flt.cproject.yml"

PACK_PIN = re.compile(r"Ambiq::NS-CMSIS-NN@\S+")
SSE300_COMPONENTS = (
    "ARM::Device:Definition",
    "ARM::Device:Native Driver:Timeout",
    "ARM::Device:Native Driver:SysCounter",
    "ARM::Device:Native Driver:SysTimer",
)
SELECTOR = "Ambiq::Machine Learning:NN Lib:heliaCORE&Source"


class CheckerCase(unittest.TestCase):
    """Runs the checker against a mutated copy of the real test wiring."""

    def build_tree(self) -> Path:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        tree = Path(holder.name) / "repo"
        (tree / "scripts").mkdir(parents=True)
        shutil.copy2(REPO / SCRIPT_REL, tree / SCRIPT_REL)
        shutil.copy2(REPO / "Ambiq.NS-CMSIS-NN.pdsc", tree / "Ambiq.NS-CMSIS-NN.pdsc")
        shutil.copytree(REPO / CMSIS_REL, tree / CMSIS_REL)
        return tree

    def run_checker(self, tree: Path) -> tuple[int, str]:
        proc = subprocess.run(
            [sys.executable, str(tree / SCRIPT_REL)],
            capture_output=True,
            text=True,
        )
        return proc.returncode, proc.stdout + proc.stderr

    # -- fixture access -------------------------------------------------
    # Read out of the copied tree rather than hardcoded, so a
    # release-please version bump or a routine pack-pin bump cannot turn
    # this suite red on an unrelated PR.

    def pack_pin(self, tree: Path) -> str:
        text = (tree / CSOLUTION_REL).read_text(encoding="utf-8")
        match = PACK_PIN.search(text)
        if match is None:
            self.fail("fixture drift: no `Ambiq::NS-CMSIS-NN@<version>` pin found")
        return match.group(0)

    def component_block(self, tree: Path, component: str) -> str:
        """The exact source lines declaring `component`, with its nested keys."""
        lines = (tree / CLAYER_REL).read_text(encoding="utf-8").splitlines(keepends=True)
        start = next(
            (i for i, ln in enumerate(lines) if ln.strip() == f"- component: {component}"),
            None,
        )
        if start is None:
            self.fail(f"fixture drift: no `- component: {component}` in the clayer")
        indent = len(lines[start]) - len(lines[start].lstrip())
        end = start + 1
        while end < len(lines):
            line = lines[end]
            if line.strip() and (len(line) - len(line.lstrip())) <= indent:
                break
            end += 1
        return "".join(lines[start:end])

    # -- mutation ------------------------------------------------------

    def edit(self, tree: Path, rel: str, old: str, new: str) -> None:
        path = tree / rel
        text = path.read_text(encoding="utf-8")
        if old not in text:
            self.fail(f"fixture drift: {rel} no longer contains {old.strip()[:80]!r}")
        path.write_text(text.replace(old, new, 1), encoding="utf-8")

    def replace_component(self, tree: Path, component: str, new: str) -> None:
        self.edit(tree, CLAYER_REL, self.component_block(tree, component), new)

    def add_pack_entry(self, tree: Path, entry: str) -> None:
        """Insert a pack entry directly after the Ambiq pin, without
        depending on any third-party pack's pinned version."""
        text = (tree / CSOLUTION_REL).read_text(encoding="utf-8")
        pin_line = next(
            (ln for ln in text.splitlines(keepends=True) if PACK_PIN.search(ln)), None
        )
        if pin_line is None:
            self.fail("fixture drift: no pack pin line found")
        self.edit(tree, CSOLUTION_REL, pin_line, pin_line + entry)

    # -- assertions ----------------------------------------------------

    def assertRejected(self, tree: Path, *expected: str) -> str:
        """Assert a nonzero exit and that every fragment appears. Callers
        pass the offending file or component too, so a test cannot pass
        because some unrelated finding happened to match."""
        code, out = self.run_checker(tree)
        self.assertNotEqual(code, 0, f"checker accepted bad input:\n{out}")
        self.assertNotIn("Traceback", out, f"checker crashed instead of reporting:\n{out}")
        for fragment in expected:
            self.assertIn(fragment.lower(), out.lower(), f"unexpected diagnostic:\n{out}")
        return out

    def assertAccepted(self, tree: Path) -> None:
        code, out = self.run_checker(tree)
        self.assertEqual(code, 0, f"checker rejected valid input:\n{out}")


class TestBaseline(CheckerCase):
    def test_unmodified_tree_is_accepted(self):
        self.assertAccepted(self.build_tree())

    def test_summary_reports_what_was_verified(self):
        tree = self.build_tree()
        code, out = self.run_checker(tree)
        self.assertEqual(code, 0, out)
        version = self.pack_pin(tree).split("@", 1)[1]
        self.assertIn(f"pack pinned at {version}", out)
        self.assertIn(f"{len(SSE300_COMPONENTS)} SSE-300 components gated", out)


class TestPackPin(CheckerCase):
    def test_pin_drifted_from_pdsc(self):
        tree = self.build_tree()
        pin = self.pack_pin(tree)
        self.edit(tree, CSOLUTION_REL, pin, "Ambiq::NS-CMSIS-NN@0.0.1")
        self.assertRejected(tree, "but pdsc", "0.0.1", "csolution.yml")

    def test_range_specifier_is_not_an_exact_pin(self):
        tree = self.build_tree()
        pin = self.pack_pin(tree)
        version = pin.split("@", 1)[1]
        self.edit(tree, CSOLUTION_REL, pin, f"Ambiq::NS-CMSIS-NN@>={version}")
        self.assertRejected(tree, "exact pin", "csolution.yml")

    def test_unpinned_pack_is_rejected(self):
        tree = self.build_tree()
        self.edit(tree, CSOLUTION_REL, self.pack_pin(tree), "Ambiq::NS-CMSIS-NN")
        self.assertRejected(tree, "must be pinned as", "csolution.yml")

    def test_path_override_is_rejected(self):
        # A path pack competes with the cpackget .Local registration even
        # when it is the only entry, so this must fail on its own.
        tree = self.build_tree()
        pin_line = next(
            ln
            for ln in (tree / CSOLUTION_REL).read_text(encoding="utf-8").splitlines(
                keepends=True
            )
            if PACK_PIN.search(ln)
        )
        self.edit(tree, CSOLUTION_REL, pin_line, pin_line + "      path: ../../..\n")
        self.assertRejected(tree, "must not use a `path:` override", "csolution.yml")

    def test_duplicate_pack_entry_is_rejected(self):
        tree = self.build_tree()
        self.add_pack_entry(tree, "    - pack: Ambiq::NS-CMSIS-NN@0.0.1\n")
        self.assertRejected(tree, "declared 2 times", "csolution.yml")


class TestComponentSelector(CheckerCase):
    def test_bare_selector_with_qualified_name_only_in_a_comment(self):
        tree = self.build_tree()
        self.edit(
            tree,
            CPROJECT_REL,
            f"    - component: {SELECTOR}\n",
            f"    # TODO restore {SELECTOR}\n"
            "    - component: Machine Learning:NN Lib:heliaCORE\n",
        )
        self.assertRejected(tree, "not the fully-qualified", "test_arm_convolve_flt")

    def test_selector_missing_entirely(self):
        tree = self.build_tree()
        self.edit(tree, CPROJECT_REL, f"    - component: {SELECTOR}\n", "")
        self.assertRejected(tree, "missing component selector", "test_arm_convolve_flt")

    def test_registered_project_missing_from_disk(self):
        tree = self.build_tree()
        (tree / CPROJECT_REL).unlink()
        self.assertRejected(tree, "does not exist on disk", "test_arm_convolve_flt")

    def test_project_on_disk_but_not_registered(self):
        tree = self.build_tree()
        src = tree / CMSIS_REL / "test_arm_convolve_flt"
        dst = tree / CMSIS_REL / "test_arm_newop_flt"
        shutil.copytree(src, dst)
        (dst / "test_arm_convolve_flt.cproject.yml").rename(
            dst / "test_arm_newop_flt.cproject.yml"
        )
        self.assertRejected(tree, "not registered", "test_arm_newop_flt")

    def test_project_registered_twice(self):
        tree = self.build_tree()
        text = (tree / CSOLUTION_REL).read_text(encoding="utf-8")
        entry = next(ln for ln in text.splitlines(keepends=True) if "- project:" in ln)
        self.edit(tree, CSOLUTION_REL, entry, entry + entry)
        self.assertRejected(tree, "registered more than once")


class TestDeviceGating(CheckerCase):
    """The invariant most prone to silent regression, in both directions.

    Every case runs against all four SSE-300-only components, so the
    guarded set cannot be quietly narrowed to whichever one the tests
    happened to name.
    """

    def test_ungated_component_with_trailing_comment(self):
        for component in SSE300_COMPONENTS:
            with self.subTest(component=component):
                tree = self.build_tree()
                self.replace_component(
                    tree, component, f"    - component: {component}  # SSE-300 only\n"
                )
                self.assertRejected(tree, "must be gated", component)

    def test_ungated_component_with_quoted_name(self):
        for component in SSE300_COMPONENTS:
            with self.subTest(component=component):
                tree = self.build_tree()
                self.replace_component(
                    tree, component, f'    - component: "{component}"\n'
                )
                self.assertRejected(tree, "must be gated", component)

    def test_component_missing_entirely(self):
        for component in SSE300_COMPONENTS:
            with self.subTest(component=component):
                tree = self.build_tree()
                self.replace_component(tree, component, "")
                self.assertRejected(tree, "is not declared", component)

    def test_for_context_list_that_also_names_generic_targets(self):
        # Regression: membership-not-equality accepted this, which is
        # exactly the ARMCM0/ARMCM4 resolution failure being guarded.
        for component in SSE300_COMPONENTS:
            with self.subTest(component=component):
                tree = self.build_tree()
                self.replace_component(
                    tree,
                    component,
                    f"    - component: {component}\n      for-context:\n"
                    "        - +Corstone-300-FVP\n        - +Cortex-M0-FVP\n",
                )
                self.assertRejected(tree, "must be gated with exactly", component)

    def test_ungated_duplicate_before_gated_entry(self):
        # Regression: folding components into a dict kept only the last
        # declaration, so an ungated duplicate ahead of it vanished.
        for component in SSE300_COMPONENTS:
            with self.subTest(component=component):
                tree = self.build_tree()
                gated = self.component_block(tree, component)
                self.replace_component(
                    tree, component, f"    - component: {component}\n" + gated
                )
                self.assertRejected(tree, "must be gated with exactly", component)

    def test_gated_component_with_define_block_is_accepted(self):
        for component in SSE300_COMPONENTS:
            with self.subTest(component=component):
                tree = self.build_tree()
                self.replace_component(
                    tree,
                    component,
                    f"    - component: {component}\n"
                    "      define:\n        - SOME_FLAG\n"
                    "      for-context: +Corstone-300-FVP\n",
                )
                self.assertAccepted(tree)

    def test_gated_component_with_trailing_comment_is_accepted(self):
        for component in SSE300_COMPONENTS:
            with self.subTest(component=component):
                tree = self.build_tree()
                self.replace_component(
                    tree,
                    component,
                    f"    - component: {component}\n"
                    "      for-context: +Corstone-300-FVP  # SSE-300 only\n",
                )
                self.assertAccepted(tree)


class TestDiagnostics(CheckerCase):
    def test_malformed_yaml_is_reported_cleanly(self):
        tree = self.build_tree()
        path = tree / CLAYER_REL
        path.write_text(path.read_text(encoding="utf-8") + "\n  : : bad\n")
        self.assertRejected(tree, "is not valid yaml", "corstone300_unittest.clayer.yml")

    def test_missing_file_does_not_mask_other_findings(self):
        tree = self.build_tree()
        (tree / CLAYER_REL).unlink()
        self.edit(tree, CSOLUTION_REL, self.pack_pin(tree), "Ambiq::NS-CMSIS-NN@0.0.1")
        self.assertRejected(
            tree,
            "corstone300_unittest.clayer.yml: cannot be read",
            "but pdsc",
        )

    def test_one_malformed_key_does_not_flood_the_report(self):
        # A typo in `solution:` used to report every project as
        # unregistered, burying the single real cause under 20 lines.
        tree = self.build_tree()
        self.edit(tree, CSOLUTION_REL, "\nsolution:", "\nsoluton:")
        out = self.assertRejected(tree, "missing top-level `solution:` mapping")
        findings = [ln for ln in out.splitlines() if ln.strip().startswith("- ")]
        self.assertEqual(
            len(findings), 1, f"one bad key produced {len(findings)} findings:\n{out}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
