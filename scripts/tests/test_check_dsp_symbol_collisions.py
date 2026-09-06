#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2024-2026 Ambiq <opensource@ambiq.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# Mutation tests for scripts/check_dsp_symbol_collisions.py (#282).
#
# This check is a merge gate for a bug class that is invisible everywhere
# else: two libraries Ambiq ships and links together (ns-cmsis-nn and
# CMSIS-DSP) can each declare a same-named, differently-signed public
# function, and the linker silently picks whichever one resolves first --
# no compiler warning, no build failure. arm_abs_f16/arm_abs_f32 shipped in
# that state for one full development cycle (#240) before being caught by
# hand in review (#281) and renamed. This suite pins that the check:
#
#   - actually catches a new collision (test_mutation_new_collision_is_caught,
#     the required mutation case);
#   - reproduces the historical #240/#281 collision against a minimal
#     fixture shaped like it, so a regression in the check's core logic
#     would be caught here even if the real Include/*.h and
#     scripts/data/cmsis_dsp_symbols.txt ever drifted back into agreement
#     by coincidence (test_reproduces_historical_abs_collision);
#   - does NOT flag a symbol on the allowlist
#     (test_allowlist_suppresses_flagged_name);
#   - does NOT false-positive on macros, comments, struct fields, or
#     function-pointer typedefs, which share the "identifier followed by
#     '('" shape with a real declaration but are not one;
#   - correctly tolerates CMSIS-DSP's indented declaration style, without
#     which the check would be blind to the exact collision (arm_abs_f32)
#     it exists to catch (test_matches_indented_dsp_style_declaration);
#   - does not scan Include/Internal/ -- the deliberate scope decision, see
#     check_dsp_symbol_collisions.py's header comment
#     (test_internal_headers_are_not_scanned);
#   - catches a collision written as a SPLIT-LINE declaration (return type
#     alone on one line, name starting the next), not just the same-line
#     form -- an earlier version of DECL_RE required a same-line prefix and
#     was blind to this shape, missing 17 real public symbols including
#     arm_sqrt_s16 itself (found in #285 review; see
#     test_mutation_split_line_collision_is_caught and
#     test_adjacent_same_line_and_split_line_declarations_both_found,
#     which mirrors the real, ten-lines-apart arm_sqrt_s8/arm_sqrt_s16
#     shape in Include/arm_nnfunctions.h); and
#   - ALLOWLIST stays empty (test_allowlist_is_currently_empty) -- any
#     addition must be a deliberate edit to this test, not something that
#     creeps in unreviewed;
#   - catches a collision whose name contains an UPPERCASE letter after
#     "arm_", in BOTH same-line and split-line spelling -- an earlier
#     version of DECL_RE's captured tail was lowercase-only ([a-z0-9_]+),
#     so a real declaration like arm_circularRead_f32 or
#     arm_biquad_cascade_df2T_f32 failed to match at all (not a truncated
#     match -- the character right after the truncation point is the
#     uppercase letter itself, never '(' or whitespace, so the whole line
#     yields zero match). 17 real CMSIS-DSP declarations were invisible to
#     the snapshot as a result (found independently in a second #285
#     review round, converging with the split-line finding above). Every
#     fixture before this fix used a lowercase-only name, which is exactly
#     why this survived a full review round untouched -- see
#     test_mutation_mixed_case_same_line_collision_is_caught,
#     test_mutation_mixed_case_split_line_collision_is_caught, and
#     test_same_line_and_split_line_mixed_case_spellings_are_equivalent;
#     and
#   - does NOT read a declaration-shaped line INSIDE a comment as real
#     code -- extract_symbols() strips /* */ and // comments first
#     (test_ignores_prose_inside_block_comment), closing a residual,
#     synthetic-only gap flagged in the same review round.
#
# Also covers list_hazards()/--list-hazards (test_list_hazards_*): the
# derived, not hand-maintained, "stems" (dtype-suffix stripped names) our
# public API shares with CMSIS-DSP -- the mechanism AGENTS.md's naming
# rule points at instead of a comment that can silently go stale.
#
# Run with: python3 scripts/tests/test_check_dsp_symbol_collisions.py

from __future__ import annotations

import importlib.util
import re
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "check_dsp_symbol_collisions.py"


def load_checker():
    spec = importlib.util.spec_from_file_location("check_dsp_symbol_collisions", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# A real declaration shape lifted straight from Include/arm_nnfunctions_flt.h.
ABS_F32_DECL = (
    "arm_cmsis_nn_status arm_abs_f32(const float32_t *input, "
    "float32_t *output, int32_t block_size);\n"
)
ABS_F16_DECL = (
    "arm_cmsis_nn_status arm_abs_f16(const float16_t *input, "
    "float16_t *output, int32_t block_size);\n"
)

# CMSIS-DSP's actual declaration style: two-space indented, inside an
# `extern "C"` block, params wrapped onto following lines. Lifted from
# upstream Include/dsp/basic_math_functions.h.
DSP_STYLE_INDENTED = """\
#ifdef   __cplusplus
extern "C"
{
#endif

  /**
   * @brief   Floating-point vector absolute value.
   */
  void arm_abs_f32(
  const float32_t * pSrc,
        float32_t * pDst,
        uint32_t blockSize);

#ifdef   __cplusplus
}
#endif
"""

# Include/Internal/arm_conv1x1_opt_f16.h's actual shape: a function-pointer
# typedef whose return type would, without the (?!\\s*\\*) guard, be
# misread as a declared function named arm_cmsis_nn_status.
FUNCTION_POINTER_TYPEDEF = (
    "typedef arm_cmsis_nn_status (*arm_conv1x1_call_f16)(const int32_t *a, "
    "const int32_t *b);\n"
)

# The #285 review finding: a mutation written as a SPLIT-LINE declaration
# (return type alone on its own line, name starting the next line with
# nothing before it). Same collision as ABS_F32_DECL-style fixtures above,
# just spelled the other way clang-format's 120-column limit produces it.
SQRT_F32_SAME_LINE_DECL = (
    "arm_cmsis_nn_status arm_sqrt_f32(const float32_t *input, "
    "float32_t *output, int32_t block_size);\n"
)
SQRT_F32_SPLIT_DECL = (
    "arm_cmsis_nn_status\n"
    "arm_sqrt_f32(const float32_t *input, float32_t *output, "
    "int32_t block_size);\n"
)

# The REAL shape in Include/arm_nnfunctions.h, ten lines apart: arm_sqrt_s8
# same-line, arm_sqrt_s16 split-line. Both must be found, not just
# whichever style happens to come first.
REAL_SQRT_PAIR_S8_S16 = (
    "arm_cmsis_nn_status arm_sqrt_s8(const int8_t *input, "
    "const cmsis_nn_dims *input_dims, int8_t *output, int8_t *sqrt_lut);\n"
    "\n"
    "arm_cmsis_nn_status\n"
    "arm_sqrt_s16(const int16_t *input, const cmsis_nn_dims *input_dims, "
    "int16_t *output, const int16_t *sqrt_lut);\n"
)

# Second #285-review finding: DECL_RE's captured tail was lowercase-only
# ([a-z0-9_]+), so a real, mixed-case CMSIS-DSP declaration -- e.g.
# arm_circularRead_f32 (__STATIC_FORCEINLINE, Include/dsp/filtering_
# functions.h) -- silently failed to match at all: the capture truncates
# at "arm_circular" (stopping before the uppercase 'R'), and the very next
# character is 'R', not '(' or whitespace, so `\s*\(` never matches and
# the whole line yields nothing. All 20 tests before this fix used
# lowercase-only names, which is exactly why this survived a full review
# round. Both spellings, mirroring the two shapes CMSIS-DSP and this repo
# actually use.
CIRCULAR_READ_F32_SAME_LINE_DECL = (
    "__STATIC_FORCEINLINE void arm_circularRead_f32(int32_t *circBuffer, "
    "int32_t L, int32_t *readOffset, int32_t bufferInc, int32_t *dst);\n"
)
CIRCULAR_READ_F32_SPLIT_DECL = (
    "__STATIC_FORCEINLINE void\n"
    "arm_circularRead_f32(int32_t *circBuffer, int32_t L, "
    "int32_t *readOffset, int32_t bufferInc, int32_t *dst);\n"
)

# Residual gap the same review flagged (lower severity, demonstrated only
# synthetically): a free-form line inside a /* ... */ block comment, with
# no leading '*' or '//', that happens to start with `arm_something(` --
# an @code example or prose describing a call -- must NOT be read as a
# real declaration. Closed by _strip_comments() in extract_symbols().
PROSE_INSIDE_BLOCK_COMMENT = """\
/*
This paragraph is not code. It describes what
arm_sqrt_f32(input, output, size) does, in prose,
with no leading '*' on this continuation line.
*/
"""


class DspSymbolCollisionCase(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = load_checker()
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.tree = Path(holder.name)
        self.include_dir = self.tree / "Include"
        self.include_dir.mkdir()
        self.dsp_file = self.tree / "cmsis_dsp_symbols.txt"

    def write_header(self, name: str, body: str, subdir: str = "") -> Path:
        target_dir = self.include_dir / subdir if subdir else self.include_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / name
        path.write_text(body, encoding="utf-8")
        return path

    def write_dsp_symbols(self, *names: str, header: str = "") -> None:
        text = header + "\n".join(names) + "\n"
        self.dsp_file.write_text(text, encoding="utf-8")

    def run_check(self) -> list[str]:
        self.mod.failures.clear()
        self.mod._stats.clear()
        self.mod.check_dsp_symbol_collisions(
            include_dir=self.include_dir, dsp_symbols_file=self.dsp_file
        )
        return list(self.mod.failures)

    def assertClean(self) -> None:
        got = self.run_check()
        self.assertEqual(got, [], f"expected no collisions but got: {got}")

    def assertCollides(self, *names: str) -> list[str]:
        got = self.run_check()
        self.assertTrue(got, "expected a collision to be reported but check passed")
        joined = " ".join(got)
        for name in names:
            self.assertIn(name, joined, f"{name!r} missing from failure message: {got}")
        return got

    # -- the real tree ---------------------------------------------------

    def test_repo_is_clean(self):
        """The real Include/*.h against the real checked-in DSP snapshot."""
        mod = load_checker()
        mod.failures.clear()
        mod._stats.clear()
        mod.check_dsp_symbol_collisions()
        self.assertEqual(mod.failures, [])
        self.assertGreater(mod._stats.get("ours", 0), 0)
        self.assertGreater(mod._stats.get("dsp", 0), 0)

    # -- the required mutation case ---------------------------------------

    def test_mutation_new_collision_is_caught(self):
        """Introducing a name CMSIS-DSP already owns must fail the check."""
        self.write_header(
            "arm_nnfunctions_flt.h",
            "arm_cmsis_nn_status arm_sqrt_f32(const float32_t *input, "
            "float32_t *output, int32_t block_size);\n",
        )
        self.write_dsp_symbols("arm_sqrt_f32", "arm_unrelated_f32")
        self.assertCollides("arm_sqrt_f32")

    def test_reproduces_historical_abs_collision(self):
        """Minimal fixture shaped like the real #240/#281 arm_abs_f16/f32
        collision. Pinned independently of the live repo tree and the live
        data file, so a regression here is caught even if those two ever
        drift back into (coincidental) agreement."""
        self.write_header("arm_nnfunctions_flt.h", ABS_F32_DECL + ABS_F16_DECL)
        self.write_dsp_symbols("arm_abs_f32", "arm_abs_f16", "arm_add_f32")
        self.assertCollides("arm_abs_f16", "arm_abs_f32")

    # -- must stay green ---------------------------------------------------

    def test_clean_tree_has_no_failures(self):
        self.write_header(
            "arm_nnfunctions_flt.h",
            "arm_cmsis_nn_status arm_elementwise_sub_f32(const float32_t *a, "
            "const float32_t *b, float32_t *out, int32_t size);\n",
        )
        self.write_dsp_symbols("arm_sub_f32", "arm_add_f32")
        self.assertClean()

    def test_allowlist_suppresses_flagged_name(self):
        """A symbol on ALLOWLIST must not be reported even if it collides."""
        self.write_header(
            "arm_nnfunctions_flt.h",
            "arm_cmsis_nn_status arm_sqrt_f32(const float32_t *input, "
            "float32_t *output, int32_t block_size);\n",
        )
        self.write_dsp_symbols("arm_sqrt_f32")
        original = self.mod.ALLOWLIST
        self.mod.ALLOWLIST = frozenset({"arm_sqrt_f32"})
        try:
            self.assertClean()
        finally:
            self.mod.ALLOWLIST = original

    def test_internal_headers_are_not_scanned(self):
        """Include/Internal/ is out of scope (deliberate design decision) --
        a colliding name declared only there must not be flagged."""
        self.write_header(
            "arm_nnfunctions_flt.h",
            "arm_cmsis_nn_status arm_elementwise_sub_f32(const float32_t *a, "
            "const float32_t *b, float32_t *out, int32_t size);\n",
        )
        self.write_header(
            "arm_conv1x1_opt_f16.h",
            "arm_cmsis_nn_status arm_sqrt_f32(const float32_t *input, "
            "float32_t *output, int32_t block_size);\n",
            subdir="Internal",
        )
        self.write_dsp_symbols("arm_sqrt_f32")
        self.assertClean()

    def test_ignores_macro_definitions(self):
        self.write_header(
            "arm_nnfunctions_flt.h",
            "#define arm_sqrt_f32(x) (call_the_real_thing(x))\n",
        )
        self.write_dsp_symbols("arm_sqrt_f32")
        self.assertClean()

    def test_ignores_line_comments(self):
        self.write_header(
            "arm_nnfunctions_flt.h",
            "// arm_cmsis_nn_status arm_sqrt_f32(float32_t *x, int32_t n);\n"
            "/* arm_sqrt_f32(x) was considered and rejected */\n",
        )
        self.write_dsp_symbols("arm_sqrt_f32")
        self.assertClean()

    def test_ignores_struct_field(self):
        """A type name followed by a plain field, not '(', is not a
        declaration -- the design note's 'struct fields' case."""
        self.write_header(
            "arm_nnfunctions_flt.h",
            "typedef struct {\n"
            "  arm_cmsis_nn_status arm_sqrt_f32_status;\n"
            "} arm_sqrt_result_f32;\n",
        )
        self.write_dsp_symbols("arm_sqrt_f32", "arm_sqrt_f32_status")
        self.assertClean()

    def test_ignores_function_pointer_typedef(self):
        """The (?!\\s*\\*) guard: a function-pointer typedef's return type
        must not be misread as a declared function name. Real shape from
        Include/Internal/arm_conv1x1_opt_f16.h (kept here even though
        Internal/ is out of scope, in case that scope ever widens)."""
        self.write_header("arm_nnfunctions_flt.h", FUNCTION_POINTER_TYPEDEF)
        self.write_dsp_symbols("arm_cmsis_nn_status")
        self.assertClean()

    def test_matches_indented_dsp_style_declaration(self):
        """extract_symbols() must tolerate CMSIS-DSP's own declaration
        style (two-space indented, inside extern "C", params on following
        lines) -- this is what the scripts/data/cmsis_dsp_symbols.txt
        refresh recipe relies on. Without it, arm_abs_f32 itself -- the
        symbol this whole check exists to catch -- would never make it
        into the generated snapshot."""
        path = self.write_header("basic_math_functions.h", DSP_STYLE_INDENTED)
        names = self.mod.extract_symbols([path])
        self.assertEqual(names, {"arm_abs_f32"})

    # -- must stay red -------------------------------------------------

    def test_missing_data_file_fails_clearly(self):
        self.write_header("arm_nnfunctions_flt.h", "")
        missing = self.tree / "does_not_exist.txt"
        self.mod.failures.clear()
        self.mod._stats.clear()
        self.mod.check_dsp_symbol_collisions(
            include_dir=self.include_dir, dsp_symbols_file=missing
        )
        self.assertTrue(self.mod.failures)
        self.assertIn("not found", " ".join(self.mod.failures))

    def test_data_file_comment_lines_are_not_symbols(self):
        """The '#'-prefixed header block in the checked-in data file must
        not itself be parsed as symbol names."""
        self.write_header(
            "arm_nnfunctions_flt.h",
            "arm_cmsis_nn_status arm_elementwise_sub_f32(const float32_t *a, "
            "const float32_t *b, float32_t *out, int32_t size);\n",
        )
        self.write_dsp_symbols(
            "arm_sub_f32",
            header="# generated from upstream CMSIS-DSP\n# refresh monthly\n\n",
        )
        self.assertClean()

    # -- #285 review: split-line declarations (the blocking finding) -----

    def test_mutation_split_line_collision_is_caught(self):
        """THE required regression case: a collision introduced as a
        split-line declaration (return type alone on its own line) must
        fail the check exactly like the same-line spelling does. Before
        the #285 fix, this mutation passed silently -- DECL_RE required a
        same-line prefix, so a same-line arm_sqrt_f32 was caught but a
        split-line one was invisible. clang-format's 120-column limit is
        what produces the split form for a long float prototype, so this
        is the realistic shape a future arm_sqrt_f32 port would take."""
        self.write_header("arm_nnfunctions_flt.h", SQRT_F32_SPLIT_DECL)
        self.write_dsp_symbols("arm_sqrt_f32")
        self.assertCollides("arm_sqrt_f32")

    def test_same_line_and_split_line_spellings_are_equivalent(self):
        """The same declaration, spelled two ways, must extract to the
        identical symbol -- the split-line path is not a weaker check
        than the same-line path."""
        same_path = self.write_header("a_same.h", SQRT_F32_SAME_LINE_DECL)
        split_path = self.write_header("b_split.h", SQRT_F32_SPLIT_DECL)
        same = self.mod.extract_symbols([same_path])
        split = self.mod.extract_symbols([split_path])
        self.assertEqual(same, {"arm_sqrt_f32"})
        self.assertEqual(same, split)

    def test_adjacent_same_line_and_split_line_declarations_both_found(self):
        """Mirrors the real shape in Include/arm_nnfunctions.h: arm_sqrt_s8
        (same-line) and arm_sqrt_s16 (split-line) declared ten lines apart
        in the same file. Both must be extracted, not just whichever style
        happens to come first -- this is the exact case the #285 review
        caught: arm_sqrt_s8 was found, arm_sqrt_s16 was silently missed."""
        path = self.write_header("arm_nnfunctions.h", REAL_SQRT_PAIR_S8_S16)
        names = self.mod.extract_symbols([path])
        self.assertEqual(names, {"arm_sqrt_s8", "arm_sqrt_s16"})

    # -- ALLOWLIST must stay empty (or grow only deliberately) -----------

    def test_allowlist_is_currently_empty(self):
        """ALLOWLIST's own comment requires an approving issue/PR before
        any entry is added. Nothing else enforces that -- this test does:
        growing ALLOWLIST means deliberately editing this assertion too,
        not something that can happen silently alongside an unrelated
        change."""
        self.assertEqual(self.mod.ALLOWLIST, frozenset())

    # -- list_hazards() / --list-hazards: derived, not hand-maintained ----

    def test_list_hazards_derives_shared_stems(self):
        """arm_abs_s8/s16 (ours) and arm_abs_f32/arm_abs_q7 (dsp) share
        the stem 'arm_abs' even though neither individual name collides --
        this is the #285 review finding that a hand-maintained hazard list
        missed (an earlier draft named four families and dropped 'abs',
        since #281 already resolved abs's own float collision)."""
        self.write_header(
            "arm_nnfunctions_flt.h",
            "arm_cmsis_nn_status arm_abs_s8(const int8_t *a, int8_t *b, "
            "int32_t c);\n"
            "arm_cmsis_nn_status arm_abs_s16(const int16_t *a, int16_t *b, "
            "int32_t c);\n"
            "arm_cmsis_nn_status arm_unrelated_s8(void);\n",
        )
        self.write_dsp_symbols("arm_abs_f32", "arm_abs_q7", "arm_other_f32")
        hazards = self.mod.list_hazards(
            include_dir=self.include_dir, dsp_symbols_file=self.dsp_file
        )
        self.assertEqual(set(hazards), {"arm_abs"})
        our_names, dsp_names = hazards["arm_abs"]
        self.assertEqual(our_names, ["arm_abs_s16", "arm_abs_s8"])
        self.assertEqual(dsp_names, ["arm_abs_f32", "arm_abs_q7"])

    def test_list_hazards_empty_when_no_shared_stems(self):
        self.write_header(
            "arm_nnfunctions_flt.h",
            "arm_cmsis_nn_status arm_totally_unique_s8(void);\n",
        )
        self.write_dsp_symbols("arm_something_else_f32")
        hazards = self.mod.list_hazards(
            include_dir=self.include_dir, dsp_symbols_file=self.dsp_file
        )
        self.assertEqual(hazards, {})

    def test_list_hazards_on_real_repo_finds_five_stems(self):
        """The real tree today: arm_abs/add/mean/sqrt/sub are exactly the
        five stems ns-cmsis-nn and CMSIS-DSP currently share (verified by
        #285 review as the corrected, complete list)."""
        hazards = self.mod.list_hazards()
        self.assertEqual(
            set(hazards),
            {"arm_abs", "arm_add", "arm_mean", "arm_sqrt", "arm_sub"},
        )

    # -- #285 review round 2: mixed-case tail (the second blocking bug) --

    def test_mutation_mixed_case_same_line_collision_is_caught(self):
        """A real, same-line, mixed-case CMSIS-DSP declaration
        (arm_circularRead_f32) must be caught exactly like an all-
        lowercase one."""
        self.write_header(
            "arm_nnfunctions_flt.h", CIRCULAR_READ_F32_SAME_LINE_DECL
        )
        self.write_dsp_symbols("arm_circularRead_f32")
        self.assertCollides("arm_circularRead_f32")

    def test_mutation_mixed_case_split_line_collision_is_caught(self):
        """THE required combination: mixed-case AND split-line at once.
        Before the case-tail fix, this mutation was invisible for two
        independent reasons layered on top of each other; it must fail
        now that both are fixed."""
        self.write_header(
            "arm_nnfunctions_flt.h", CIRCULAR_READ_F32_SPLIT_DECL
        )
        self.write_dsp_symbols("arm_circularRead_f32")
        self.assertCollides("arm_circularRead_f32")

    def test_same_line_and_split_line_mixed_case_spellings_are_equivalent(self):
        """Mixed-case name, spelled two ways -- both extract to the
        identical symbol, with the uppercase 'R' intact (not truncated,
        not dropped)."""
        same_path = self.write_header(
            "a_same.h", CIRCULAR_READ_F32_SAME_LINE_DECL
        )
        split_path = self.write_header(
            "b_split.h", CIRCULAR_READ_F32_SPLIT_DECL
        )
        same = self.mod.extract_symbols([same_path])
        split = self.mod.extract_symbols([split_path])
        self.assertEqual(same, {"arm_circularRead_f32"})
        self.assertEqual(same, split)

    def test_lowercase_only_tail_would_truncate_the_capture_not_match_it(self):
        """Documents the exact failure mode: a lowercase-only tail class
        does not capture "arm_circularRead_f32" as a shorter, wrong name
        (e.g. "arm_circular") -- it fails to match the line at all, because
        nothing immediately follows the truncated capture except the
        uppercase letter itself, never '(' or whitespace."""
        lowercase_only = re.compile(
            r"^[ \t]*(?:[A-Za-z_][A-Za-z0-9_ \*\t]*\b)?(arm_[a-z0-9_]+)"
            r"\s*\((?!\s*\*)",
            re.M,
        )
        self.assertEqual(
            lowercase_only.findall(CIRCULAR_READ_F32_SAME_LINE_DECL), []
        )

    def test_ignores_prose_inside_block_comment(self):
        """A doc-comment line with no leading '*' that happens to start
        with an arm_* call shape must not be read as a real declaration."""
        path = self.write_header(
            "arm_nnfunctions_flt.h", PROSE_INSIDE_BLOCK_COMMENT
        )
        names = self.mod.extract_symbols([path])
        self.assertEqual(names, set())


if __name__ == "__main__":
    unittest.main(verbosity=2)
