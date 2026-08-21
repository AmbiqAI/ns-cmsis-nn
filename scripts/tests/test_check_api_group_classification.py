#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2024-2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Mutation tests for scripts/check_api_group_classification.py (#283).
#
# This check is a merge gate whose own logic is easy to regress in either
# direction -- silently under-cover (a public kernel goes unflagged again,
# reproducing #283 itself) or silently no-op (the whole check passes
# vacuously because header discovery quietly found nothing). This suite
# pins that the check:
#
#   - actually catches a public kernel with no matching GROUP_PATTERNS
#     entry (test_mutation_ungrouped_public_kernel_is_caught, the required
#     mutation case);
#   - does NOT flag a function declared only outside the public header
#     glob, e.g. a support-function header
#     (test_non_public_header_is_not_scanned);
#   - still treats a kernel guarded by `#if ARM_NN_ENABLE_F32` as public
#     and checks it -- nn.dxy.in predefines that macro for Doxygen, so
#     this check's notion of "public" must agree
#     (test_conditionally_compiled_float_declaration_is_still_public);
#   - fails loudly, not silently, when public header discovery looks
#     broken: zero matching files, fewer than the expected minimum (the
#     #283-review scenario of exactly one header having been renamed
#     away), and zero names extracted from files that did match
#     (test_missing_include_dir_fails_loudly and friends); and
#   - reproduces cleanly against the real repo tree
#     (test_repo_is_clean), which is the only thing standing between this
#     suite and "the check accidentally always passes".
#
# Run with: python3 scripts/tests/test_check_api_group_classification.py

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "check_api_group_classification.py"
API_GROUP_INDEX = REPO / "docs" / "_ext" / "api_group_index.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: api_group_index.py's ApiFunction is a
    # @dataclass, whose class-processing looks the defining module up in
    # sys.modules by name -- skip this and it raises AttributeError on a
    # module that is perfectly valid. See check_api_group_classification.py's
    # load_api_group_index() for the same fix on the non-test path.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_checker():
    return load_module("check_api_group_classification", SCRIPT)


# A real declaration shape lifted from Include/arm_nnfunctions_flt.h,
# including the ARM_NN_ENABLE_F32 guard exactly as it appears there.
CONDITIONAL_FLOAT_DECL = """\
#if ARM_NN_ENABLE_F32

arm_cmsis_nn_status arm_depthwise_nhwc_conv_f32(const cmsis_nn_context *ctx,
                                                const cmsis_nn_dims *input_dims,
                                                const float32_t *input);

#endif
"""


class ApiGroupClassificationCase(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = load_checker()
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.tree = Path(holder.name)
        self.include_dir = self.tree / "Include"
        self.include_dir.mkdir()

    def write_header(self, name: str, body: str) -> Path:
        path = self.include_dir / name
        path.write_text(body, encoding="utf-8")
        return path

    def stub_other_public_header(self) -> None:
        """Fill in an empty arm_nnfunctions.h / arm_nnfunctions_flt.h for
        whichever of the two a test didn't itself write, so
        MIN_EXPECTED_PUBLIC_HEADERS is satisfied and tests that are not
        about header *discovery* aren't tripped up by it. Only fills in
        what's missing -- never overwrites a file the test already wrote."""
        for name in ("arm_nnfunctions.h", "arm_nnfunctions_flt.h"):
            path = self.include_dir / name
            if not path.exists():
                path.write_text("", encoding="utf-8")

    def run_check(
        self,
        include_dir: Path | None = None,
        api_group_index_path: Path = API_GROUP_INDEX,
    ) -> list[str]:
        self.mod.failures.clear()
        self.mod._stats.clear()
        self.mod.check_api_group_classification(
            include_dir=include_dir if include_dir is not None else self.include_dir,
            api_group_index_path=api_group_index_path,
        )
        return list(self.mod.failures)

    def assertClean(self, **kwargs) -> None:
        got = self.run_check(**kwargs)
        self.assertEqual(got, [], f"expected no failures but got: {got}")

    def assertFails(self, *fragments: str, **kwargs) -> list[str]:
        got = self.run_check(**kwargs)
        self.assertTrue(got, "expected a failure but check passed")
        joined = " ".join(got)
        for fragment in fragments:
            self.assertIn(fragment, joined, f"{fragment!r} missing from failure message: {got}")
        return got

    # -- the real tree ----------------------------------------------------

    def test_repo_is_clean(self):
        """The real Include/arm_nnfunctions*.h against the real
        GROUP_PATTERNS -- the actual regression pin. Everything else in
        this suite uses synthetic fixtures precisely so a real-tree
        drift can't mask a break in the check's own logic, or vice
        versa."""
        mod = load_checker()
        mod.failures.clear()
        mod._stats.clear()
        mod.check_api_group_classification()
        self.assertEqual(mod.failures, [])
        self.assertGreater(mod._stats.get("public", 0), 0)

    def test_fp16_spelling_lands_in_the_f16_dtype_bucket(self):
        """arm_fully_connected_fp16 is a real float16_t kernel -- declared
        under ARM_FLOAT16_SUPPORTED, which nn.dxy.in predefines to 1, so it
        renders on the page -- that spells the dtype `fp16` rather than
        `f16`. api-filter.js compares dtype for exact equality against the
        chip values in api-groups.md, so any return other than "f16" leaves
        the kernel unreachable from every chip and badged wrong."""
        mod = load_module("api_group_index", API_GROUP_INDEX)
        self.assertEqual(mod._dtype("arm_fully_connected_fp16"), "f16")
        self.assertEqual(mod._dtype("arm_avg_pool_f16"), "f16")
        self.assertEqual(mod._dtype("arm_avg_pool_f32"), "f32")
        self.assertEqual(mod._dtype("arm_convolve_s8"), "s8")

    # -- the required mutation case ----------------------------------------

    def test_mutation_ungrouped_public_kernel_is_caught(self):
        """A public kernel whose name matches no GROUP_PATTERNS entry must
        fail the check -- this is #283 itself."""
        self.write_header(
            "arm_nnfunctions.h",
            "arm_cmsis_nn_status arm_totally_unpatterned_op_s8(const int8_t *input, "
            "int8_t *output, int32_t size);\n",
        )
        self.stub_other_public_header()
        self.assertFails("arm_totally_unpatterned_op_s8")

    # -- scope: only the public header glob is scanned ----------------------

    def test_non_public_header_is_not_scanned(self):
        """A function declared only in a header outside the
        arm_nnfunctions*.h glob (e.g. the support-function headers) must
        not be flagged, even though it would match nothing in
        GROUP_PATTERNS either -- it is not part of the public surface
        this check owns."""
        self.write_header(
            "arm_nnfunctions.h",
            "arm_cmsis_nn_status arm_avg_pool_s8(const cmsis_nn_context *ctx, "
            "const int8_t *input, int8_t *output);\n",
        )
        self.write_header(
            "arm_nnsupportfunctions.h",
            "arm_cmsis_nn_status arm_totally_unpatterned_internal_helper(const int8_t *x);\n",
        )
        self.stub_other_public_header()
        self.assertClean()

    # -- conditional compilation --------------------------------------------

    def test_conditionally_compiled_float_declaration_is_still_public(self):
        """A declaration guarded by `#if ARM_NN_ENABLE_F32` is still
        extracted and checked -- Doxygen documents it unconditionally
        (nn.dxy.in predefines the macro), so this check's notion of
        "public" must agree, not silently exempt everything behind a
        feature flag.

        Uses a throwaway GROUP_PATTERNS-less api_group_index.py fixture
        (not the real one, which already covers this exact kernel) so the
        assertion is about extraction seeing the declaration at all, not
        about today's real pattern coverage -- otherwise this test would
        pass even if the `#if` guard silently defeated DECL_RE, as long as
        *some* real pattern happened to cover the name anyway.
        """
        self.write_header("arm_nnfunctions_flt.h", CONDITIONAL_FLOAT_DECL)
        self.stub_other_public_header()
        empty_patterns = self.tree / "api_group_index_empty.py"
        empty_patterns.write_text(
            "GROUP_PATTERNS = {}\n\n\ndef _matches(name, patterns):\n    return False\n",
            encoding="utf-8",
        )
        self.assertFails(
            "arm_depthwise_nhwc_conv_f32", api_group_index_path=empty_patterns
        )

        # And the real, shipped GROUP_PATTERNS already covers it -- confirms
        # the extraction assertion above isn't testing a kernel that
        # wouldn't actually be relevant to the real check.
        agi = load_module("api_group_index_fixture_ok", API_GROUP_INDEX)
        self.assertTrue(
            any(
                agi._matches("arm_depthwise_nhwc_conv_f32", patterns)
                for patterns in agi.GROUP_PATTERNS.values()
            ),
            "arm_depthwise_nhwc_conv_f32 should already match 'convolution' "
            "in the real GROUP_PATTERNS",
        )
        self.assertClean()

    # -- must stay green -----------------------------------------------------

    def test_clean_tree_has_no_failures(self):
        self.write_header(
            "arm_nnfunctions.h",
            "arm_cmsis_nn_status arm_avg_pool_s8(const cmsis_nn_context *ctx, "
            "const int8_t *input, int8_t *output);\n",
        )
        self.write_header(
            "arm_nnfunctions_flt.h",
            "arm_cmsis_nn_status arm_where_s8(const int8_t *condition, "
            "int64_t *output, int32_t *num_true);\n",
        )
        self.assertClean()

    def test_doc_comment_reference_is_still_a_valid_public_name(self):
        """@copydoc-style mentions are deliberately over-matched (see
        DECL_RE's docstring) -- confirm that over-matching still resolves
        cleanly rather than e.g. producing a name with trailing junk."""
        self.write_header(
            "arm_nnfunctions_flt.h",
            "/**\n * @copydoc arm_avg_pool_f32\n */\n"
            "arm_cmsis_nn_status arm_avg_pool_f32(const cmsis_nn_context *ctx);\n",
        )
        self.stub_other_public_header()
        self.assertClean()

    # -- must fail loudly, not silently --------------------------------------

    def test_missing_include_dir_fails_loudly(self):
        """Include/ not existing at all must not be read as '0 public
        names, 0 gaps, clean' -- the fail-open bug this check exists to
        avoid reproducing one layer up from #283."""
        missing = self.tree / "does_not_exist"
        self.assertFails("expected at least", include_dir=missing)

    def test_one_header_missing_fails_loudly(self):
        """Only arm_nnfunctions.h present, arm_nnfunctions_flt.h renamed
        or removed: still a nonzero, non-obviously-broken name set (the
        base API alone), which is exactly the silent-degradation case a
        bare "did we get zero names" check would miss. MIN_EXPECTED_PUBLIC_HEADERS
        is what catches it instead."""
        self.write_header(
            "arm_nnfunctions.h",
            "arm_cmsis_nn_status arm_avg_pool_s8(const cmsis_nn_context *ctx, "
            "const int8_t *input, int8_t *output);\n",
        )
        self.assertFails("found only 1 public API header")

    def test_zero_names_extracted_fails_loudly(self):
        """Headers matching the glob but containing no arm_* declarations
        (e.g. the extraction regex broke) must fail, not report '0 public
        functions, all compliant'."""
        self.write_header("arm_nnfunctions.h", "// nothing declared here\n")
        self.write_header("arm_nnfunctions_flt.h", "// nothing declared here either\n")
        self.assertFails("yielded zero function names")

    def test_missing_api_group_index_fails_loudly(self):
        self.write_header(
            "arm_nnfunctions.h",
            "arm_cmsis_nn_status arm_avg_pool_s8(const cmsis_nn_context *ctx);\n",
        )
        self.stub_other_public_header()
        missing = self.tree / "does_not_exist.py"
        self.assertFails("not found", api_group_index_path=missing)


if __name__ == "__main__":
    unittest.main(verbosity=2)
