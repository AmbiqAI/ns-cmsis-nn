#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2024-2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Mutation tests for check #10 (registered unit-test suites exist and
# their relative includes resolve) in scripts/check_pdsc.py.
#
# The bug class this check guards is invisible everywhere else in CI, so
# nothing else would notice the check going quiet. 36 float suites were
# registered in Tests/UnitTest/CMakeLists.txt against
# `../TestData/<name>/test_data.h` paths that their `*_settings_flt.py`
# generators do produce, but only into a gitignored `TestData/` tree --
# the data was never checked in, so the suites were unbuildable in every
# checkout. Because no PR-gating job builds the float suites at all
# either way (ARM_NN_ENABLE_F32/F16 default OFF in the legacy build),
# they were never configured, never compiled, and never run -- while
# still reading as coverage on the tin. That is how the transpose-conv
# output-shift bug reached a release (#253, #256).
#
# A check whose failure mode is "silently passes" needs its own gate, so
# the mutations here are the shapes that would make it go quiet:
#
#   - a dangling `../TestData/...` include in a registered suite (the
#     #256 shape itself) must be red;
#   - `test_data_is_resolved_relative_to_the_including_file` pins the
#     resolution rule, because resolving relative to the suite root
#     instead of the file would call the real (healthy) integer suites
#     broken and the real broken ones fine -- the check would be exactly
#     inverted while still "working";
#   - a bare-name include such as `"unity.h"` must stay green: it is
#     found on the compiler's include path, and flagging it would make
#     the check unusable and get it deleted;
#   - the allowlist is keyed by exact resolved include path, not by
#     suite name, so it must suppress only its own listed paths -- a new
#     dangling include in the same allowlisted suite (the shape a PR
#     would add if it shipped a new float dataset without checking the
#     data in) must still be red -- and must go red once a listed path
#     stops dangling or its suite is unregistered, so an exemption cannot
#     silently grow or outlive its reason. The live allowlist is empty
#     (its one #236-era entry, test_arm_convolve_f16, was deleted with
#     the suite), so these cases run on synthetic allowlists.
#
# Run with: python3 scripts/tests/test_check_pdsc_unit_test_suites.py

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "check_pdsc.py"


def load_checker():
    spec = importlib.util.spec_from_file_location("check_pdsc", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Shaped like the real Tests/UnitTest/CMakeLists.txt: the library build
# and Unity are registered alongside the suites, so the parser is
# exercised through the same shapes it must skip in the real file.
CMAKE_TEMPLATE = """\
cmake_minimum_required(VERSION 3.15.6)
project(cmsis_nn_unit_tests VERSION 0.0.1)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. cmsis-nn)

__SUITES__

if(ARM_NN_ENABLE_F32)
__F32_SUITES__
endif()

add_subdirectory(Unity)
"""

PREFIX = "Tests/UnitTest/TestCases/"

# A healthy integer suite: the .c sits at the suite root, so
# "../TestData/add_s8/test_data.h" resolves to the *shared*
# TestCases/TestData tree, not to one inside the suite. Getting that
# wrong in either direction is the inversion the tests below pin.
HEALTHY = {
    f"{PREFIX}test_arm_add_s8/test_arm_add_s8.c": '#include "../TestData/add_s8/test_data.h"\n',
    f"{PREFIX}test_arm_add_s8/Unity/unity_test_arm_add_s8.c": (
        '#include "unity.h"\n#include "../test_arm_add_s8.c"\n'
    ),
    f"{PREFIX}TestData/add_s8/test_data.h": "// golden\n",
}

# The #256 shape: registered, but its ../TestData path does not exist.
DECOY = {
    f"{PREFIX}test_arm_softmax_f32/test_arm_softmax_f32.c": (
        '#include "../TestData/softmax_f32/test_data.h"\n'
    ),
}


class SuiteDataCase(unittest.TestCase):
    def run_check(
        self,
        files: dict[str, str],
        suites: list[str],
        f32_suites: list[str] | None = None,
        allowlist: dict[str, dict[str, str]] | None = None,
        tracked_extra: list[str] | None = None,
    ) -> list[str]:
        """Run check #10 over a synthetic tree; return its failures."""
        mod = load_checker()
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        root = Path(holder.name)

        for rel, text in files.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

        cmake = root / "Tests" / "UnitTest" / "CMakeLists.txt"
        cmake.parent.mkdir(parents=True, exist_ok=True)
        cmake.write_text(
            CMAKE_TEMPLATE.replace(
                "__SUITES__",
                "\n".join(f"add_subdirectory(TestCases/{s})" for s in suites),
            ).replace(
                "__F32_SUITES__",
                "\n".join(f"    add_subdirectory(TestCases/{s})" for s in (f32_suites or [])),
            ),
            encoding="utf-8",
        )

        mod.REPO = root
        mod.UNIT_TEST_CMAKE = cmake
        mod.tracked_repo_files = lambda: set(files) | set(tracked_extra or [])
        mod.UNBUILDABLE_SUITE_ALLOWLIST = {} if allowlist is None else allowlist
        mod.failures.clear()
        mod.check_unit_test_suite_data()
        return list(mod.failures)

    def assertGreen(self, **kw) -> None:
        got = self.run_check(**kw)
        self.assertEqual(got, [], f"should pass check #10 but failed: {got}")

    def assertRed(self, needle: str = "", **kw) -> list[str]:
        got = self.run_check(**kw)
        self.assertTrue(got, "should fail check #10 but passed")
        if needle:
            self.assertIn(needle, " ".join(got))
        return got

    # -- the real tree ---------------------------------------------------

    def test_repo_is_clean(self):
        """The live tree passes: every registered suite resolves, with
        no allowlist entry covering for anything."""
        mod = load_checker()
        mod.failures.clear()
        mod.check_unit_test_suite_data()
        self.assertEqual(mod.failures, [])

    def test_real_tree_registers_many_suites(self):
        """A parser that silently matched nothing would make this check
        vacuous while still exiting 0."""
        mod = load_checker()
        mod.failures.clear()
        suites = mod.registered_unit_test_suites()
        self.assertIsNotNone(suites)
        self.assertEqual(mod.failures, [])
        self.assertGreater(len(suites), 100)
        self.assertIn("test_arm_add_s8", suites)
        # The library build and Unity must not be mistaken for suites.
        self.assertNotIn("Unity", suites)

    def test_allowlist_is_empty(self):
        """No suite is exempt. The dict exists so a documented, PR-bound
        exception is possible, but the last one (test_arm_convolve_f16,
        parked by #256 while #236 was open) left with its suite; if this
        grows again, the guard has become a parking spot."""
        mod = load_checker()
        self.assertEqual(mod.UNBUILDABLE_SUITE_ALLOWLIST, {})

    # -- must stay green -------------------------------------------------

    def test_healthy_suite_passes(self):
        self.assertGreen(files=HEALTHY, suites=["test_arm_add_s8"])

    def test_bare_include_is_not_flagged(self):
        """"unity.h" comes off the compiler's include path. Flagging it
        would make the check unusable, so it must stay invisible."""
        files = dict(HEALTHY)
        files[f"{PREFIX}test_arm_add_s8/test_arm_add_s8.c"] += '#include "arm_nnfunctions.h"\n'
        self.assertGreen(files=files, suites=["test_arm_add_s8"])

    def test_unregistered_broken_suite_is_ignored(self):
        """Only registration makes a suite claim to be coverage. An
        unregistered directory is dead weight, not a build break."""
        self.assertGreen(files={**HEALTHY, **DECOY}, suites=["test_arm_add_s8"])

    # -- the #256 shape ---------------------------------------------------

    def test_dangling_testdata_include_is_caught(self):
        got = self.assertRed(
            files={**HEALTHY, **DECOY},
            suites=["test_arm_add_s8"],
            f32_suites=["test_arm_softmax_f32"],
            needle="does not resolve",
        )
        joined = " ".join(got)
        self.assertIn("test_arm_softmax_f32", joined)
        self.assertIn("#256", joined)
        # The healthy sibling must not be implicated.
        self.assertNotIn("test_arm_add_s8", joined)

    def test_suite_registered_behind_a_dtype_gate_is_still_checked(self):
        """The #256 suites were all inside `if(ARM_NN_ENABLE_F32/F16)`.
        A parser that only read top-level registrations would have seen
        none of them."""
        self.assertRed(
            files={**HEALTHY, **DECOY},
            suites=["test_arm_add_s8"],
            f32_suites=["test_arm_softmax_f32"],
            needle="test_arm_softmax_f32",
        )

    def test_registered_but_missing_directory_is_caught(self):
        """Deleting a suite without unregistering it is a hard CMake
        error in a configure nobody runs."""
        self.assertRed(
            files=HEALTHY,
            suites=["test_arm_add_s8", "test_arm_deleted_f32"],
            needle="has no tracked source files",
        )

    def test_data_is_resolved_relative_to_the_including_file(self):
        """The inversion canary. TestCases/test_arm_add_s8/test_arm_add_s8.c
        includes "../TestData/..." meaning the *shared* TestCases/TestData
        tree. Resolving relative to the suite root instead would call this
        healthy suite broken -- and, symmetrically, would have called the
        36 real decoys fine. Same include text, two different resolutions,
        so only the rule itself distinguishes them."""
        # Same include, but placed one directory deeper: now it must
        # resolve inside the suite, and there is nothing there.
        files = {
            f"{PREFIX}test_arm_x_f32/Unity/unity_test_arm_x_f32.c": (
                '#include "../TestData/x_f32/test_data.h"\n'
            ),
            f"{PREFIX}TestData/x_f32/test_data.h": "// shared tree, wrong level\n",
        }
        got = self.assertRed(files=files, suites=["test_arm_x_f32"], needle="does not resolve")
        self.assertIn("test_arm_x_f32/TestData/x_f32/test_data.h", " ".join(got))
        # And the healthy suite, whose include sits at the suite root,
        # must resolve against the shared tree.
        self.assertGreen(files=HEALTHY, suites=["test_arm_add_s8"])

    # -- the allowlist (synthetic entries; the live dict is empty) ---------

    def test_allowlist_suppresses_its_own_entry(self):
        self.assertGreen(
            files={**HEALTHY, **DECOY},
            suites=["test_arm_add_s8"],
            f32_suites=["test_arm_softmax_f32"],
            allowlist={
                "test_arm_softmax_f32": {
                    f"{PREFIX}TestData/softmax_f32/test_data.h": "rides with some PR",
                },
            },
        )

    def test_allowlist_does_not_suppress_anything_else(self):
        """A broad allowlist would re-open the whole #256 hole."""
        other = {
            f"{PREFIX}test_arm_other_f32/test_arm_other_f32.c": (
                '#include "../TestData/other_f32/test_data.h"\n'
            ),
        }
        got = self.assertRed(
            files={**HEALTHY, **DECOY, **other},
            suites=["test_arm_add_s8"],
            f32_suites=["test_arm_softmax_f32", "test_arm_other_f32"],
            allowlist={
                "test_arm_softmax_f32": {
                    f"{PREFIX}TestData/softmax_f32/test_data.h": "rides with some PR",
                },
            },
        )
        joined = " ".join(got)
        self.assertIn("test_arm_other_f32", joined)
        self.assertNotIn("test_arm_softmax_f32", joined)

    def test_allowlist_does_not_cover_a_new_dangling_path_in_the_same_suite(self):
        """Adding a second dataset to an already-allowlisted suite without
        checking its data in must still be red, because the allowlist is
        keyed by the exact path, not by suite name."""
        files = dict(DECOY)
        files[f"{PREFIX}test_arm_softmax_f32/test_arm_softmax_f32.c"] += (
            '#include "../TestData/softmax_new_case_f32/test_data.h"\n'
        )
        got = self.assertRed(
            files={**HEALTHY, **files},
            suites=["test_arm_add_s8"],
            f32_suites=["test_arm_softmax_f32"],
            allowlist={
                "test_arm_softmax_f32": {
                    f"{PREFIX}TestData/softmax_f32/test_data.h": "rides with some PR",
                },
            },
            needle="softmax_new_case_f32",
        )
        joined = " ".join(got)
        self.assertIn(
            "an unrelated existing entry for this suite does not cover a new path", joined
        )
        # The listed path stays suppressed; only the unlisted one fires.
        self.assertNotIn("does not resolve (Tests/UnitTest/TestCases/TestData/softmax_f32/", joined)

    def test_clean_allowlist_entry_is_caught(self):
        """Once the exempted path is fixed (checked in, or the include
        removed), the entry must go -- an exemption that outlives its
        reason is how allowlists rot."""
        self.assertRed(
            files=HEALTHY,
            suites=["test_arm_add_s8"],
            allowlist={
                "test_arm_add_s8": {
                    f"{PREFIX}TestData/nonexistent/test_data.h": "stale",
                },
            },
            needle="no longer dangle",
        )

    def test_unregistered_allowlist_entry_is_caught(self):
        """If a suite is deleted, its entry must go too."""
        self.assertRed(
            files=HEALTHY,
            suites=["test_arm_add_s8"],
            allowlist={
                "test_arm_gone_f16": {
                    f"{PREFIX}TestData/whatever/test_data.h": "deleted elsewhere",
                },
            },
            needle="no longer registered",
        )

    # -- an unreadable tracked source must fail(), not raise -------------

    def test_unreadable_tracked_source_is_a_clean_failure(self):
        """`sources` comes from `tracked_repo_files()` (git ls-files), not
        the working tree. A tracked file the working tree no longer has
        (deleted locally, broken symlink, ...) must surface as a normal
        fail() entry, not an uncaught FileNotFoundError traceback."""
        got = self.assertRed(
            files=HEALTHY,
            suites=["test_arm_add_s8"],
            tracked_extra=[f"{PREFIX}test_arm_add_s8/ghost.c"],
            needle="could not be read",
        )
        self.assertIn("test_arm_add_s8/ghost.c", " ".join(got))
        # The real, readable source must still be checked -- one bad file
        # must not swallow the rest of the suite's sources.
        self.assertIn("test_arm_add_s8", " ".join(got))

    # -- the parser must refuse what it cannot parse -----------------------

    def test_unmodelled_add_subdirectory_is_loud(self):
        got = self.run_check(
            files=HEALTHY, suites=["test_arm_add_s8"], f32_suites=[]
        )
        self.assertEqual(got, [])
        mod = load_checker()
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        root = Path(holder.name)
        cmake = root / "Tests" / "UnitTest" / "CMakeLists.txt"
        cmake.parent.mkdir(parents=True, exist_ok=True)
        cmake.write_text(
            "add_subdirectory(TestCases/test_arm_add_s8)\nadd_subdirectory(SomethingElse)\n",
            encoding="utf-8",
        )
        mod.REPO = root
        mod.UNIT_TEST_CMAKE = cmake
        mod.failures.clear()
        self.assertIsNone(mod.registered_unit_test_suites())
        self.assertIn("unmodelled", " ".join(mod.failures))

    def test_no_registrations_at_all_is_loud(self):
        """If the file changes shape and the parser stops seeing suites,
        that must fail rather than pass vacuously."""
        mod = load_checker()
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        root = Path(holder.name)
        cmake = root / "Tests" / "UnitTest" / "CMakeLists.txt"
        cmake.parent.mkdir(parents=True, exist_ok=True)
        cmake.write_text("add_subdirectory(Unity)\n", encoding="utf-8")
        mod.REPO = root
        mod.UNIT_TEST_CMAKE = cmake
        mod.failures.clear()
        self.assertIsNone(mod.registered_unit_test_suites())
        self.assertIn("no `add_subdirectory(TestCases/...)` registrations", " ".join(mod.failures))

    def test_commented_out_registration_is_ignored(self):
        """A `#`-commented registration is not a registration."""
        mod = load_checker()
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        root = Path(holder.name)
        cmake = root / "Tests" / "UnitTest" / "CMakeLists.txt"
        cmake.parent.mkdir(parents=True, exist_ok=True)
        cmake.write_text(
            "add_subdirectory(TestCases/test_arm_add_s8)\n"
            "# add_subdirectory(TestCases/test_arm_softmax_f32)\n",
            encoding="utf-8",
        )
        mod.REPO = root
        mod.UNIT_TEST_CMAKE = cmake
        mod.failures.clear()
        self.assertEqual(mod.registered_unit_test_suites(), ["test_arm_add_s8"])
        self.assertEqual(mod.failures, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
