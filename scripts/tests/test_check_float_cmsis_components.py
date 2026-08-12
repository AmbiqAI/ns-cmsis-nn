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
# Run with: python3 scripts/tests/test_check_float_cmsis_components.py

from __future__ import annotations

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
CPROJECT_REL = f"{CMSIS_REL}/test_arm_softmax_flt/test_arm_softmax_flt.cproject.yml"

GATED_DEFINITION = (
    "    - component: ARM::Device:Definition\n      for-context: +Corstone-300-FVP\n"
)


class CheckerCase(unittest.TestCase):
    """Runs the checker against a mutated copy of the real test wiring."""

    def build_tree(self) -> Path:
        tree = Path(self.enterContext(tempfile.TemporaryDirectory())) / "repo"
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

    def edit(self, tree: Path, rel: str, old: str, new: str) -> None:
        path = tree / rel
        text = path.read_text(encoding="utf-8")
        self.assertIn(old, text, f"fixture drift: {rel} no longer contains {old!r}")
        path.write_text(text.replace(old, new, 1), encoding="utf-8")

    def assertRejected(self, tree: Path, expected: str) -> None:
        code, out = self.run_checker(tree)
        self.assertNotEqual(code, 0, f"checker accepted bad input:\n{out}")
        self.assertIn(expected.lower(), out.lower(), f"unexpected diagnostic:\n{out}")

    def assertAccepted(self, tree: Path) -> None:
        code, out = self.run_checker(tree)
        self.assertEqual(code, 0, f"checker rejected valid input:\n{out}")


class TestBaseline(CheckerCase):
    def test_unmodified_tree_is_accepted(self):
        self.assertAccepted(self.build_tree())


class TestPackPin(CheckerCase):
    def test_pin_drifted_from_pdsc(self):
        tree = self.build_tree()
        self.edit(tree, CSOLUTION_REL, "NS-CMSIS-NN@7.29.2", "NS-CMSIS-NN@7.27.0")
        self.assertRejected(tree, "but pdsc")

    def test_range_specifier_is_not_an_exact_pin(self):
        tree = self.build_tree()
        self.edit(tree, CSOLUTION_REL, "NS-CMSIS-NN@7.29.2", "NS-CMSIS-NN@>=7.29.2")
        self.assertRejected(tree, "exact pin")

    def test_duplicate_pack_entry_with_path_override(self):
        tree = self.build_tree()
        self.edit(
            tree,
            CSOLUTION_REL,
            "    - pack: ARM::CMSIS@6.3.0\n",
            "    - pack: Ambiq::NS-CMSIS-NN\n      path: ../../..\n"
            "    - pack: ARM::CMSIS@6.3.0\n",
        )
        self.assertRejected(tree, "declared 2 times")


class TestComponentSelector(CheckerCase):
    def test_bare_selector_with_qualified_name_only_in_a_comment(self):
        tree = self.build_tree()
        self.edit(
            tree,
            CPROJECT_REL,
            "    - component: Ambiq::Machine Learning:NN Lib:heliaCORE&Source\n",
            "    # TODO restore Ambiq::Machine Learning:NN Lib:heliaCORE&Source\n"
            "    - component: Machine Learning:NN Lib:heliaCORE\n",
        )
        self.assertRejected(tree, "not the fully-qualified")

    def test_registered_project_missing_from_disk(self):
        tree = self.build_tree()
        (tree / CPROJECT_REL).unlink()
        self.assertRejected(tree, "does not exist on disk")

    def test_project_on_disk_but_not_registered(self):
        tree = self.build_tree()
        src = tree / CMSIS_REL / "test_arm_softmax_flt"
        dst = tree / CMSIS_REL / "test_arm_newop_flt"
        shutil.copytree(src, dst)
        (dst / "test_arm_softmax_flt.cproject.yml").rename(
            dst / "test_arm_newop_flt.cproject.yml"
        )
        self.assertRejected(tree, "not registered")

    def test_project_registered_twice(self):
        tree = self.build_tree()
        entry = "    - project: ./test_arm_pad_flt/test_arm_pad_flt.cproject.yml\n"
        self.edit(tree, CSOLUTION_REL, entry, entry + entry)
        self.assertRejected(tree, "registered more than once")


class TestDeviceGating(CheckerCase):
    """The invariant most prone to silent regression, in both directions."""

    def test_ungated_component_with_trailing_comment(self):
        tree = self.build_tree()
        self.edit(
            tree,
            CLAYER_REL,
            GATED_DEFINITION,
            "    - component: ARM::Device:Definition  # SSE-300 driver\n",
        )
        self.assertRejected(tree, "must be gated")

    def test_ungated_component_with_quoted_name(self):
        tree = self.build_tree()
        self.edit(
            tree,
            CLAYER_REL,
            GATED_DEFINITION,
            '    - component: "ARM::Device:Definition"\n',
        )
        self.assertRejected(tree, "must be gated")

    def test_component_missing_entirely(self):
        tree = self.build_tree()
        self.edit(tree, CLAYER_REL, GATED_DEFINITION, "")
        self.assertRejected(tree, "is not declared")

    def test_for_context_list_that_also_names_generic_targets(self):
        # Regression: membership-not-equality accepted this, which is
        # exactly the ARMCM0/ARMCM4 resolution failure being guarded.
        tree = self.build_tree()
        self.edit(
            tree,
            CLAYER_REL,
            GATED_DEFINITION,
            "    - component: ARM::Device:Definition\n      for-context:\n"
            "        - +Corstone-300-FVP\n        - +Cortex-M0-FVP\n",
        )
        self.assertRejected(tree, "must be gated with exactly")

    def test_ungated_duplicate_before_gated_entry(self):
        # Regression: folding components into a dict kept only the last
        # declaration, so an ungated duplicate ahead of it vanished.
        tree = self.build_tree()
        self.edit(
            tree,
            CLAYER_REL,
            GATED_DEFINITION,
            "    - component: ARM::Device:Definition\n" + GATED_DEFINITION,
        )
        self.assertRejected(tree, "must be gated with exactly")

    def test_gated_component_with_define_block_is_accepted(self):
        tree = self.build_tree()
        self.edit(
            tree,
            CLAYER_REL,
            GATED_DEFINITION,
            "    - component: ARM::Device:Definition\n"
            "      define:\n        - SOME_FLAG\n"
            "      for-context: +Corstone-300-FVP\n",
        )
        self.assertAccepted(tree)

    def test_gated_component_with_trailing_comment_is_accepted(self):
        tree = self.build_tree()
        self.edit(
            tree,
            CLAYER_REL,
            GATED_DEFINITION,
            "    - component: ARM::Device:Definition\n"
            "      for-context: +Corstone-300-FVP  # SSE-300 only\n",
        )
        self.assertAccepted(tree)


class TestDiagnostics(CheckerCase):
    def test_malformed_yaml_is_reported_cleanly(self):
        tree = self.build_tree()
        path = tree / CLAYER_REL
        path.write_text(path.read_text(encoding="utf-8") + "\n  : : bad\n")
        code, out = self.run_checker(tree)
        self.assertNotEqual(code, 0)
        self.assertIn("not valid yaml", out.lower())
        self.assertNotIn("Traceback", out)

    def test_missing_file_does_not_mask_other_findings(self):
        tree = self.build_tree()
        (tree / CLAYER_REL).unlink()
        self.edit(tree, CSOLUTION_REL, "NS-CMSIS-NN@7.29.2", "NS-CMSIS-NN@1.0.0")
        code, out = self.run_checker(tree)
        self.assertNotEqual(code, 0)
        self.assertIn("cannot be read", out)
        self.assertIn("but pdsc", out)
        self.assertNotIn("Traceback", out)

    def test_one_malformed_key_does_not_flood_the_report(self):
        # A typo in `solution:` used to report every project as
        # unregistered, burying the single real cause under 20 lines.
        tree = self.build_tree()
        self.edit(tree, CSOLUTION_REL, "solution:", "soluton:")
        code, out = self.run_checker(tree)
        self.assertNotEqual(code, 0)
        findings = [ln for ln in out.splitlines() if ln.strip().startswith("- ")]
        self.assertLessEqual(
            len(findings), 2, f"one bad key produced {len(findings)} findings:\n{out}"
        )
        self.assertIn("missing top-level `solution:` mapping", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
