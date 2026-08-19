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
#     it exists to catch (test_matches_indented_dsp_style_declaration); and
#   - does not scan Include/Internal/ -- the deliberate scope decision, see
#     check_dsp_symbol_collisions.py's header comment
#     (test_internal_headers_are_not_scanned).
#
# Run with: python3 scripts/tests/test_check_dsp_symbol_collisions.py

from __future__ import annotations

import importlib.util
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
