#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2024-2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Mutation tests for check #9 (SSoT/pdsc source-list agreement) in
# scripts/check_pdsc.py.
#
# That check reads cmake/ns_cmsis_nn.cmake — CMake, not data — with a
# small hand-rolled parser, and a parser guarding against silent drift
# is worthless the moment it silently ignores something. Two failure
# shapes are pinned here because a reviewer demonstrated both against
# the first draft:
#
#   - `list(REMOVE_ITEM extras "...")` was ignored along with every
#     list() subcommand except APPEND, so a removed source still counted
#     as reachable and real drift passed green.
#   - An argument yielding no double-quoted literal (`set(patterns
#     *_s8.c)`, `list(APPEND extras ${VAR})`) resolved to *nothing*, and
#     the resulting error then named an innocent, reachable kernel as
#     pdsc-only and told the maintainer to allowlist it — following that
#     advice would have permanently blinded the check.
#
# Both are now hard failures, so the tests assert not just "fails" but
# "fails saying the right thing" — `test_unquoted_argument_does_not_
# misdiagnose` is the one that matters, since a loud failure with
# misleading advice is barely better than silence.
#
# Set-equality alone also cannot see a *misplaced* gate: moving
# arm_softmax_f32.c under `if(ARM_NN_ENABLE_F16)` leaves the union
# untouched while handing an F32-only consumer a fresh #268. The check
# resolves each ARM_NN_ENABLE_F32/F16 combination separately and compares
# membership against each file's dtype tag; `test_wrong_gate_placement`
# is that exact mutation and must stay red in both directions.
#
# Run with: python3 scripts/tests/test_check_pdsc_ssot.py

from __future__ import annotations

import importlib.util
import re
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "check_pdsc.py"
SSOT_CMAKE_REAL = REPO / "cmake" / "ns_cmsis_nn.cmake"


def load_checker():
    spec = importlib.util.spec_from_file_location("check_pdsc", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# A one-group stand-in for the real SSoT module, shaped exactly like it
# (same command vocabulary, same PARENT_SCOPE hand-back) so the parser is
# exercised through its real entry point rather than a stub.
SSOT_TEMPLATE = """\
set(_NS_CMSIS_NN_GROUPS
    softmax
)

set(_NS_CMSIS_NN_DTYPES s4 s8 s16 s32 s64 q7 q15 f16 f32)

function(_ns_cmsis_nn_group_def group out_subdir out_patterns out_extras)
  set(subdir "")
  set(patterns "")
  set(extras "")

  if(group STREQUAL "softmax")
    set(subdir   "SoftmaxFunctions")
    set(patterns "*_s8.c")
__BODY__
  else()
    message(FATAL_ERROR "ns_cmsis_nn: unknown group '${group}'")
  endif()

  set(${out_subdir}   "${subdir}"   PARENT_SCOPE)
  set(${out_patterns} "${patterns}" PARENT_SCOPE)
  set(${out_extras}   "${extras}"   PARENT_SCOPE)
endfunction()
"""

TRACKED = [
    "Source/SoftmaxFunctions/arm_softmax_s8.c",
    "Source/SoftmaxFunctions/arm_softmax_u8.c",
    "Source/SoftmaxFunctions/arm_softmax_f32.c",
    "Source/SoftmaxFunctions/arm_softmax_f16.c",
]

GOOD = """\
    set(extras   "arm_softmax_u8.c")
    if(ARM_NN_ENABLE_F32)
      list(APPEND extras "arm_softmax_f32.c")
    endif()
    if(ARM_NN_ENABLE_F16)
      list(APPEND extras "arm_softmax_f16.c")
    endif()
"""

# The reviewer's mutation: the union is unchanged, only the gate moved.
WRONG_GATE = """\
    set(extras   "arm_softmax_u8.c")
    if(ARM_NN_ENABLE_F16)
      list(APPEND extras "arm_softmax_f32.c")
      list(APPEND extras "arm_softmax_f16.c")
    endif()
"""

F16_GATE_BLOCK = """\
    if(ARM_NN_ENABLE_F16)
      list(APPEND extras "arm_softmax_f16.c")
    endif()
"""
MISSING_F16 = GOOD.replace(F16_GATE_BLOCK, "")

REMOVE_ITEM = GOOD + '    list(REMOVE_ITEM extras "arm_softmax_u8.c")\n'
LIST_FILTER = GOOD + '    list(FILTER extras EXCLUDE REGEX "u8")\n'
UNQUOTED_SET = GOOD.replace('set(patterns "*_s8.c")', "set(patterns *_s8.c)").replace(
    'set(extras   "arm_softmax_u8.c")', "set(extras   arm_softmax_u8.c)"
)
VARIABLE_APPEND = GOOD.replace(
    'list(APPEND extras "arm_softmax_f32.c")', "list(APPEND extras ${SOFTMAX_F32_SOURCES})"
)
HASH_IN_STRING = GOOD.replace('"arm_softmax_u8.c"', '"arm_softmax_#u8.c"')
UNKNOWN_COMMAND = GOOD + '    file(GLOB extra_sources "${dir}/*.c")\n'
UNMODELLED_CONDITION = GOOD + """\
    if(ARM_NN_ENABLE_BF16)
      list(APPEND extras "arm_softmax_bf16.c")
    endif()
"""
# Reviewer fragment 1: an elseif dtype condition read as a nested if()
# is AND-ed with the sibling branch, so arm_softmax_f16.c resolves only
# in F32+F16 and assertion B blames correct CMake for a parser defect.
ELSEIF_GATE = """\
    set(extras   "arm_softmax_u8.c")
    if(ARM_NN_ENABLE_F32)
      list(APPEND extras "arm_softmax_f32.c")
    elseif(ARM_NN_ENABLE_F16)
      list(APPEND extras "arm_softmax_f16.c")
    endif()
"""

# Reviewer fragment 2, the worse one: the single endif() pops only the
# leaked elseif gate, so the trailing ungated append resolves as if it
# were F32-gated. arm_softmax_s16.c carries an integer tag, so assertion
# B skips it and assertion A still sees it in the union — a wrong model
# that exits 0.
ELSEIF_LEAK = ELSEIF_GATE + '    list(APPEND extras "arm_softmax_s16.c")\n'
S16 = "Source/SoftmaxFunctions/arm_softmax_s16.c"

SET_INSIDE_GATE = """\
    set(extras   "arm_softmax_u8.c")
    if(ARM_NN_ENABLE_F32)
      set(extras "arm_softmax_f32.c")
    endif()
    if(ARM_NN_ENABLE_F16)
      list(APPEND extras "arm_softmax_f16.c")
    endif()
"""


class SsotAgreementCase(unittest.TestCase):
    def run_check(
        self,
        body: str,
        pdsc: list[str] | None = None,
        tracked: list[str] | None = None,
        allowlist: dict[str, str] | None = None,
    ) -> list[str]:
        """Run check #9 over a synthetic SSoT module; return its failures."""
        mod = load_checker()
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        path = Path(holder.name) / "ns_cmsis_nn.cmake"
        path.write_text(SSOT_TEMPLATE.replace("__BODY__", body), encoding="utf-8")
        mod.SSOT_CMAKE = path
        mod.tracked_source_files = lambda: list(TRACKED if tracked is None else tracked)
        if allowlist is not None:
            mod.SSOT_PDSC_ALLOWLIST = allowlist
        mod.failures.clear()
        mod.check_ssot_pdsc_agreement(
            [("source", p) for p in (TRACKED if pdsc is None else pdsc)]
        )
        return list(mod.failures)

    def assertGreen(self, body: str, **kw) -> None:
        got = self.run_check(body, **kw)
        self.assertEqual(got, [], f"should pass check #9 but failed: {got}")

    def assertRed(self, body: str, needle: str = "", **kw) -> list[str]:
        got = self.run_check(body, **kw)
        self.assertTrue(got, "should fail check #9 but passed")
        if needle:
            self.assertIn(needle, " ".join(got))
        return got

    # -- the real tree ---------------------------------------------------

    def test_repo_is_clean(self):
        """The live pdsc and the live SSoT agree, gates included."""
        mod = load_checker()
        pkg, _ = mod.parse_pdsc_text()
        comp = mod.find_component(pkg)
        entries = mod.collect_file_entries(comp)
        mod.failures.clear()
        mod.check_ssot_pdsc_agreement(entries)
        self.assertEqual(mod.failures, [])

    def test_real_ssot_resolves_every_configuration(self):
        """Each float configuration resolves, and each is distinct — a
        parser that collapsed them would make the gate check vacuous."""
        mod = load_checker()
        parsed = mod.parse_ssot_sources()
        self.assertIsNotNone(parsed)
        per_config, dtypes = parsed
        self.assertEqual(set(per_config), set(mod.SSOT_CONFIGS))
        self.assertIn("f16", dtypes)
        sizes = {name: len(src) for name, src in per_config.items()}
        self.assertLess(sizes["integer-only"], sizes["F32-only"])
        self.assertLess(sizes["integer-only"], sizes["F16-only"])
        self.assertLess(sizes["F32-only"], sizes["F32+F16"])
        self.assertLess(sizes["F16-only"], sizes["F32+F16"])

    # -- must stay green -------------------------------------------------

    def test_baseline_agrees(self):
        self.assertGreen(GOOD)

    def test_untagged_source_is_gate_exempt(self):
        """arm_softmax_u8.c carries no float tag, so being reachable in
        every configuration is correct, not drift."""
        got = self.run_check(GOOD)
        self.assertEqual(got, [])

    def test_allowlist_suppresses_documented_exception(self):
        self.assertGreen(
            GOOD,
            pdsc=TRACKED + ["Source/SoftmaxFunctions/arm_softmax_s4.c"],
            tracked=TRACKED + ["Source/SoftmaxFunctions/arm_softmax_s4.c"],
            allowlist={"Source/SoftmaxFunctions/arm_softmax_s4.c": "documented, see #000"},
        )

    # -- drift, the #268 shape -------------------------------------------

    def test_pdsc_only_source_is_caught(self):
        self.assertRed(
            MISSING_F16, "unreachable from cmake/ns_cmsis_nn.cmake under any dtype gate"
        )

    def test_ssot_only_source_is_caught(self):
        self.assertRed(
            GOOD + '    list(APPEND extras "arm_softmax_typo.c")\n',
            "referenced by cmake/ns_cmsis_nn.cmake but not shipped by the pdsc",
        )

    def test_stale_allowlist_entry_is_caught(self):
        self.assertRed(
            GOOD,
            "drop the allowlist entry",
            allowlist={"Source/SoftmaxFunctions/arm_softmax_u8.c": "stale"},
        )

    # -- the parser must refuse what it cannot parse ---------------------

    def test_remove_item_is_loud(self):
        got = self.assertRed(REMOVE_ITEM, "list(REMOVE_ITEM ...)")
        self.assertIn("silently miss", " ".join(got))

    def test_other_list_subcommand_is_loud(self):
        self.assertRed(LIST_FILTER, "list(FILTER ...)")

    def test_unquoted_set_argument_is_loud(self):
        self.assertRed(UNQUOTED_SET, "unquoted or variable argument in group 'softmax'")

    def test_variable_append_is_loud(self):
        self.assertRed(VARIABLE_APPEND, "unquoted or variable argument in group 'softmax'")

    def test_unquoted_argument_does_not_misdiagnose(self):
        """The failure must not name a live kernel as pdsc-only, nor tell
        the maintainer to allowlist one — following that advice would
        blind the check permanently."""
        for body in (UNQUOTED_SET, VARIABLE_APPEND):
            got = " ".join(self.run_check(body))
            self.assertNotIn("SSOT_PDSC_ALLOWLIST", got)
            self.assertNotIn("unreachable from cmake/ns_cmsis_nn.cmake", got)

    def test_hash_inside_quoted_string_is_loud(self):
        """Comment stripping is not quote-aware; the resulting paren
        desync must surface as a hard failure, never a truncated parse."""
        self.assertRed(HASH_IN_STRING, "unbalanced parentheses")

    def test_unknown_command_is_loud(self):
        self.assertRed(UNKNOWN_COMMAND, "unrecognized command `file(...)`")

    def test_unmodelled_condition_is_loud(self):
        self.assertRed(UNMODELLED_CONDITION, "unmodelled condition in group 'softmax'")

    def test_set_inside_dtype_gate_is_loud(self):
        self.assertRed(SET_INSIDE_GATE, "inside a dtype gate")

    def test_elseif_inside_group_branch_is_loud(self):
        self.assertRed(ELSEIF_GATE, "inside a group branch is not modelled")

    def test_elseif_leaked_gate_does_not_resolve_silently(self):
        """The dangerous shape: read as a nested if(), the leaked gate
        outlives the single endif() and silently re-gates the append that
        follows it, while the check still exits 0. The whole parse must be
        refused — not partially resolved and reported on."""
        got = self.run_check(ELSEIF_LEAK, pdsc=TRACKED + [S16], tracked=TRACKED + [S16])
        self.assertEqual(
            len(got),
            1,
            f"a refused parse must publish nothing else, got: {got}",
        )
        self.assertIn("inside a group branch is not modelled", got[0])
        self.assertNotIn("arm_softmax_s16.c", got[0])

    def test_filter_dtypes_special_cases_are_mirrored(self):
        """Canary. _ssot_dtype_tag() hand-mirrors cmake's
        _ns_cmsis_nn_filter_dtypes() and nothing links the two, so a
        second special case added on the CMake side (an `_fp32` remap,
        say) would silently divorce them: assertion B would skip the
        affected files and a misplaced gate would pass green. Pin the
        CMake side's shape so that change has to come here too."""
        mod = load_checker()
        text = mod.CMAKE_COMMENT_RE.sub("", SSOT_CMAKE_REAL.read_text(encoding="utf-8"))
        body = mod._cmake_function_body(text, "_ns_cmsis_nn_filter_dtypes")
        self.assertIsNotNone(body, "_ns_cmsis_nn_filter_dtypes() not found in the SSoT")
        advice = (
            "cmake/ns_cmsis_nn.cmake's _ns_cmsis_nn_filter_dtypes() changed how it "
            "derives a file's dtype tag. _ssot_dtype_tag() in scripts/check_pdsc.py "
            "hand-mirrors that algorithm and must be updated in the same change, or "
            "check #9's gate-placement assertion will silently skip the affected "
            "files."
        )
        # Exactly one literal special case, then the generic dtype loop.
        self.assertEqual(
            re.findall(r'MATCHES\s+"([^"]*)"', body),
            ["_fp16([._]|$)", "_${dt}([._]|$)"],
            advice,
        )
        # ... and it is still break-on-first-hit, which is why
        # arm_quantize_f32_s8.c tags as s8 rather than f32.
        self.assertIn("break()", body, advice)

    # -- gate placement ---------------------------------------------------

    def test_wrong_gate_placement(self):
        """arm_softmax_f32.c under the F16 gate: union-equality is blind
        to it, per-configuration resolution is not. Both directions must
        be reported."""
        got = self.assertRed(WRONG_GATE, "wrong ARM_NN_ENABLE_* gate")
        joined = " ".join(got)
        self.assertIn("arm_softmax_f32.c", joined)
        # F32-only / F32+F16 enable F32 but no longer reach the file.
        self.assertIn("unreachable from cmake/ns_cmsis_nn.cmake in the F32-only", joined)
        # F16-only reaches it with ARM_NN_ENABLE_F32 off.
        self.assertIn("reachable from cmake/ns_cmsis_nn.cmake in the F16-only", joined)
        # The f16 sibling is correctly placed and must not be implicated.
        self.assertNotIn("arm_softmax_f16.c", joined)

    def test_ungated_float_source_is_caught(self):
        """A float source with no gate at all is reachable in the
        integer-only build, which cannot compile it."""
        self.assertRed(
            '    set(extras   "arm_softmax_u8.c" "arm_softmax_f32.c" "arm_softmax_f16.c")\n',
            "integer-only configuration",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
