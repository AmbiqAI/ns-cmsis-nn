#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2024-2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Mutation tests for check #7 (float dtype gating) in scripts/check_pdsc.py.
#
# That check is a merge gate for a bug class that is invisible everywhere
# else: the pdsc ships float sources unconditionally, while
# cmake/ns_cmsis_nn.cmake adds them only under the matching
# ARM_NN_ENABLE_F32/F16 block. An ungated float kernel therefore builds
# green in CI and breaks CMSIS-Pack consumers only (#264).
#
# The check has two easy-to-regress directions, so both are pinned here:
#
#   - Too narrow. The first draft matched `_(f16|f32)\.c$` and silently
#     waved through arm_convolve_f16_fast_small_kernel.c, the very file
#     from #236 that reintroduced the bug — its dtype token is an infix,
#     not a suffix. `test_infix_dtype_name_without_gate` is that exact
#     shape and must stay red.
#   - Too broad. QuantizationFunctions/ takes float32_t across an integer
#     API by design and is built in integer-only configurations. A check
#     that flags it gets disabled by the next person to hit it, so it is
#     pinned green. (The legacy *_fp16.c sources that once self-guarded on
#     ARM_FLOAT16_SUPPORTED are gone or gated now, so no exemption remains.)
#
# A gate is not enough on its own: the file must be an *empty translation
# unit* when the dtype is off, so stray code before the gate or after its
# #endif must also fail. Both gate placements in the tree are pinned
# green (gate above the includes for sources that pull a non-self-guarding
# Internal/*_opt_f16.h header, gate below them otherwise).
#
# Run with: python3 scripts/tests/test_check_pdsc_float_gating.py

from __future__ import annotations

import importlib.util
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


GATED_BELOW = """\
#include "arm_nnfunctions.h"

#if ARM_NN_ENABLE_F16

arm_cmsis_nn_status arm_thing_f16(void) { return 0; }

#endif /* ARM_NN_ENABLE_F16 */
"""

GATED_ABOVE = """\
#include "arm_nn_types.h"

#if ARM_NN_ENABLE_F16

    #include "Internal/arm_conv_opt_f16.h"

arm_cmsis_nn_status arm_thing_f16(void) { return 0; }

#endif /* ARM_NN_ENABLE_F16 */
"""

UNGATED = """\
#include "arm_nnfunctions.h"

arm_cmsis_nn_status arm_thing_f16(void) { return 0; }
"""

WRONG_MACRO = GATED_BELOW.replace("ARM_NN_ENABLE_F16", "ARM_NN_ENABLE_F32")

CODE_AFTER_ENDIF = GATED_BELOW + "\nint arm_thing_f16_stray;\n"

CODE_BEFORE_GATE = """\
#include "arm_nnfunctions.h"

int arm_thing_f16_stray;

#if ARM_NN_ENABLE_F16

arm_cmsis_nn_status arm_thing_f16(void) { return 0; }

#endif /* ARM_NN_ENABLE_F16 */
"""

UNCLOSED = """\
#include "arm_nnfunctions.h"

#if ARM_NN_ENABLE_F16

arm_cmsis_nn_status arm_thing_f16(void) { return 0; }
"""

# A gate whose #endif closes an inner directive rather than the gate must
# not be mistaken for the gate's own #endif.
NESTED_OK = """\
#include "arm_nnfunctions.h"

#if ARM_NN_ENABLE_F16

    #if defined(ARM_MATH_MVE_FLOAT16)
arm_cmsis_nn_status arm_thing_f16(void) { return 1; }
    #else
arm_cmsis_nn_status arm_thing_f16(void) { return 0; }
    #endif

#endif /* ARM_NN_ENABLE_F16 */
"""

# Doxygen blocks and // comments are not declarations and must not count
# as code sitting outside the gate.
COMMENTS_OUTSIDE_OK = """\
#include "arm_nnfunctions.h"

/**
 * @addtogroup groupElementwise
 * @{
 */

#if ARM_NN_ENABLE_F16

arm_cmsis_nn_status arm_thing_f16(void) { return 0; }

#endif /* ARM_NN_ENABLE_F16 */

// trailing note
/* @} */
"""


class FloatGatingCase(unittest.TestCase):
    def check(self, rel: str, body: str) -> list[str]:
        """Run check #7 over a single synthetic source; return its failures."""
        mod = load_checker()
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        tree = Path(holder.name)
        path = tree / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        mod.REPO = tree
        mod.failures.clear()
        mod.check_float_source_gating([("source", rel)])
        return list(mod.failures)

    def assertGreen(self, rel: str, body: str) -> None:
        got = self.check(rel, body)
        self.assertEqual(got, [], f"{rel} should pass check #7 but failed: {got}")

    def assertRed(self, rel: str, body: str, needle: str = "") -> None:
        got = self.check(rel, body)
        self.assertTrue(got, f"{rel} should fail check #7 but passed")
        if needle:
            self.assertIn(needle, " ".join(got))

    # -- the real tree ---------------------------------------------------

    def test_repo_is_clean(self):
        """Every float source the live pdsc ships is correctly gated."""
        mod = load_checker()
        pkg, _ = mod.parse_pdsc_text()
        comp = mod.find_component(pkg)
        entries = mod.collect_file_entries(comp)
        mod.failures.clear()
        mod.check_float_source_gating(entries)
        self.assertEqual(mod.failures, [])

    # -- must stay green -------------------------------------------------

    def test_gate_below_includes(self):
        self.assertGreen("Source/BasicMathFunctions/arm_thing_f16.c", GATED_BELOW)

    def test_gate_above_includes(self):
        self.assertGreen("Source/ConvolutionFunctions/arm_convolve_f16.c", GATED_ABOVE)

    def test_nested_directives(self):
        self.assertGreen("Source/BasicMathFunctions/arm_thing_f16.c", NESTED_OK)

    def test_comments_outside_gate(self):
        self.assertGreen("Source/BasicMathFunctions/arm_thing_f16.c", COMMENTS_OUTSIDE_OK)

    def test_legacy_fp16_name_is_not_exempt(self):
        """The last legacy *_fp16.c source is gated like any other float file; no exemption remains."""
        self.assertRed("Source/BasicMathFunctions/arm_elementwise_add_fp16.c", UNGATED)

    def test_quantization_dir_exempt(self):
        """float32_t across an integer API; built in integer-only configs."""
        for rel in (
            "Source/QuantizationFunctions/arm_dequantize_s8_f32.c",
            "Source/QuantizationFunctions/arm_quantize_f32_s8.c",
        ):
            self.assertGreen(rel, UNGATED)

    def test_integer_source_ignored(self):
        self.assertGreen("Source/BasicMathFunctions/arm_elementwise_add_s8.c", UNGATED)

    # -- must stay red ---------------------------------------------------

    def test_missing_gate(self):
        self.assertRed(
            "Source/BasicMathFunctions/arm_thing_f16.c", UNGATED, "has no `#if ARM_NN_ENABLE"
        )

    def test_infix_dtype_name_without_gate(self):
        """The #236 shape: dtype token is an infix, not a suffix."""
        self.assertRed(
            "Source/ConvolutionFunctions/arm_convolve_f16_fast_small_kernel.c",
            UNGATED,
            "has no `#if ARM_NN_ENABLE",
        )

    def test_wrong_macro(self):
        self.assertRed(
            "Source/BasicMathFunctions/arm_thing_f16.c", WRONG_MACRO, "expected ARM_NN_ENABLE_F16"
        )

    def test_code_after_endif(self):
        self.assertRed(
            "Source/BasicMathFunctions/arm_thing_f16.c", CODE_AFTER_ENDIF, "outside the"
        )

    def test_code_before_gate(self):
        self.assertRed(
            "Source/BasicMathFunctions/arm_thing_f16.c", CODE_BEFORE_GATE, "outside the"
        )

    def test_unclosed_gate(self):
        self.assertRed("Source/BasicMathFunctions/arm_thing_f16.c", UNCLOSED, "never closed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
