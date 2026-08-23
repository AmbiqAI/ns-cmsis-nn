#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2024-2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Mutation tests for scripts/check_header_placeholders.py (#288).
#
# The check this suite gates is the "never again" guard for a defect class
# that has already shipped heap overflows twice (#269): a literal,
# unsubstituted `{API}_get_buffer_size()` in a public kernel's `ctx` doxygen,
# which every caller then has to resolve by hand, and which two callers
# resolved wrongly. The guard is only worth having if it can fail, so this
# suite pins that it:
#
#   - actually catches an injected `{API}` placeholder in a public header
#     (test_mutation_api_placeholder_is_caught, the required mutation case),
#     and catches other `{TOKEN}` template leftovers too;
#   - catches a cited arm_*_get_buffer_size* that resolves to nothing
#     (test_mutation_cited_sizer_that_does_not_exist_is_caught) -- the rule
#     that covers #269 itself, where the placeholder was resolved to a
#     real-but-wrong name and left no token behind, and the #288 case where
#     8 of 11 sites had no same-named sizer to substitute;
#   - keeps the KNOWN_ABSENT_SIZERS allowlist honest, failing if a name
#     documented as nonexistent ever becomes declared
#     (test_known_absent_sizer_that_becomes_real_is_caught);
#   - does NOT fire on ordinary C that happens to contain braces, e.g.
#     `= {NULL};` -- a merge gate that flags valid code is a gate that gets
#     deleted, which is why scanning is comment-scoped
#     (test_c_initializer_is_not_flagged);
#   - is not fooled by "/*" inside a string literal
#     (test_comment_marker_inside_string_literal_is_not_a_comment);
#   - fails loudly, not silently, when header discovery looks broken --
#     missing Include/, or a required public header renamed away, which
#     would otherwise leave the check passing over nothing at all; and
#   - reproduces cleanly against the real repo tree (test_repo_is_clean),
#     which is the only thing standing between this suite and "the check
#     accidentally always passes".
#
# Run with: python3 scripts/tests/test_check_header_placeholders.py

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "check_header_placeholders.py"


def load_checker():
    spec = importlib.util.spec_from_file_location("check_header_placeholders", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# The exact shape the 17 defective blocks had in Include/arm_nnfunctions.h
# before #288 resolved them.
DEFECTIVE_CTX_BLOCK = """\
/**
 * @brief Basic s8 Fully Connected function.
 *
 * @param[in, out] ctx           Function context (e.g. temporary buffer). Check the function
 *                               definition file to see if an additional buffer is required.
 *                               Optional function {API}_get_buffer_size() provides the buffer
 *                               size if an additional buffer is required.
 */
arm_cmsis_nn_status arm_fully_connected_s8(const cmsis_nn_context *ctx);
"""

# The same block after resolution. Declares the sizer it cites, so that both
# rules are satisfied: rule 1 sees no token, and rule 2 can resolve the name.
RESOLVED_CTX_BLOCK = """\
/**
 * @brief Basic s8 Fully Connected function.
 *
 * @param[in, out] ctx           Function context. Size ctx->buf with
 *                               arm_fully_connected_s8_get_buffer_size(filter_dims).
 */
arm_cmsis_nn_status arm_fully_connected_s8(const cmsis_nn_context *ctx);

int32_t arm_fully_connected_s8_get_buffer_size(const cmsis_nn_dims *filter_dims);
"""


class HeaderPlaceholderCase(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = load_checker()
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.tree = Path(holder.name)
        self.include_dir = self.tree / "Include"
        self.include_dir.mkdir()

    def write_header(self, name: str, body: str) -> Path:
        path = self.include_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        return path

    def stub_required_headers(self) -> None:
        """Fill in whichever of the REQUIRED_HEADERS a test did not write
        itself, so tests that are not about header *discovery* are not
        tripped up by it. Never overwrites a file the test already wrote."""
        for name in self.mod.REQUIRED_HEADERS:
            path = self.include_dir / name
            if not path.exists():
                path.write_text("/* nothing here */\n", encoding="utf-8")

    def run_check(self, include_dir: Path | None = None) -> list[str]:
        self.mod.failures.clear()
        self.mod._stats.clear()
        self.mod.check_header_placeholders(
            include_dir=include_dir if include_dir is not None else self.include_dir
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
        """The real Include/*.h. This is the actual regression pin: it is
        what fails if a `{API}` placeholder is ever reintroduced, and it is
        what would have failed on the tree that shipped #269."""
        mod = load_checker()
        mod.failures.clear()
        mod._stats.clear()
        mod.check_header_placeholders()
        self.assertEqual(mod.failures, [])
        self.assertGreater(mod._stats.get("headers", 0), 0)
        self.assertEqual(mod._stats.get("placeholders"), 0)
        # Rule 2 must actually be looking at something on the real tree --
        # zero citations would mean it passes vacuously.
        self.assertGreater(mod._stats.get("cited_sizers", 0), 0)

    def test_real_header_has_no_api_token_anywhere(self):
        """Belt-and-braces on the specific token from #288, independent of
        the check's own comment-scoping logic: a plain substring search over
        the real public headers must find nothing."""
        for header in sorted((REPO / "Include").glob("*.h")):
            self.assertNotIn(
                "{API}",
                header.read_text(encoding="utf-8", errors="replace"),
                f"{header.name} still carries an unsubstituted {{API}} placeholder",
            )

    # -- the required mutation case ----------------------------------------

    def test_mutation_api_placeholder_is_caught(self):
        """Reintroducing the exact defective block from #288 must fail the
        check. This is the regression that shipped twice."""
        self.write_header("arm_nnfunctions.h", DEFECTIVE_CTX_BLOCK)
        self.stub_required_headers()
        got = self.assertFails("{API}", "arm_nnfunctions.h")
        # Points at the offending line, not just the file.
        self.assertIn(":6:", " ".join(got))

    def test_resolved_block_is_clean(self):
        """The same block after resolution passes -- confirms the mutation
        test above is detecting the placeholder, not the surrounding
        boilerplate."""
        self.write_header("arm_nnfunctions.h", RESOLVED_CTX_BLOCK)
        self.stub_required_headers()
        self.assertClean()

    def test_failure_message_warns_against_hand_guessing(self):
        """The whole lesson of #269 is that resolving the placeholder by
        guessing is what caused the overflow, so the failure message has to
        say that, not just 'placeholder found'."""
        self.write_header("arm_nnfunctions.h", DEFECTIVE_CTX_BLOCK)
        self.stub_required_headers()
        self.assertFails("Do NOT guess", "#269", "#288")

    # -- other template leftovers -------------------------------------------

    def test_other_template_token_is_caught(self):
        """`{API}` is the token behind #288, but any unsubstituted
        `{TOKEN}` in a comment is the same defect shape."""
        self.write_header("arm_nnfunctions.h", "/** @brief Sized by {TYPE}_get_buffer_size(). */\n")
        self.stub_required_headers()
        self.assertFails("{TYPE}")

    def test_placeholder_in_line_comment_is_caught(self):
        self.write_header("arm_nnfunctions.h", "// TODO: call {API}_get_buffer_size()\n")
        self.stub_required_headers()
        self.assertFails("{API}")

    def test_placeholder_in_any_public_header_is_caught(self):
        """Scope is every Include/*.h, not just arm_nnfunctions.h."""
        self.write_header("arm_nnfunctions_flt.h", "/* {API}_get_buffer_size() */\n")
        self.stub_required_headers()
        self.assertFails("arm_nnfunctions_flt.h")

    def test_multiple_placeholders_are_all_reported(self):
        """17 sites existed; reporting only the first would have made the
        cleanup an 17-round game of whack-a-mole."""
        self.write_header(
            "arm_nnfunctions.h",
            "/* {API}_get_buffer_size() */\n/* {OTHER} */\n/* {THIRD} */\n",
        )
        self.stub_required_headers()
        got = self.run_check()
        self.assertEqual(len(got), 3, got)

    def test_literal_api_token_outside_a_comment_is_caught(self):
        """`{API}` is never valid C in these headers, so it is flagged
        regardless of comment context -- catching it cannot cost a false
        positive."""
        self.write_header("arm_nnfunctions.h", "#define SIZER {API}_get_buffer_size\n")
        self.stub_required_headers()
        self.assertFails("{API}")

    # -- false positives: valid C must not be flagged -------------------------

    def test_c_initializer_is_not_flagged(self):
        """Ordinary C with a `{IDENT}` initializer must pass. This is the
        reason scanning is comment-scoped: a gate that fires on valid code
        gets switched off, and then guards nothing."""
        self.write_header(
            "arm_nnfunctions.h",
            "static const cmsis_nn_context empty_ctx = {NULL};\n"
            "static const int8_t table[] = {ZERO};\n",
        )
        self.stub_required_headers()
        self.assertClean()

    def test_comment_marker_inside_string_literal_is_not_a_comment(self):
        """A "/*" inside a string constant must not start a comment region,
        or everything after it gets scanned as comment text and ordinary
        code starts producing false positives."""
        self.write_header(
            "arm_nnfunctions.h",
            'static const char *s = "/* not a comment";\n'
            "static const cmsis_nn_context c = {NULL};\n",
        )
        self.stub_required_headers()
        self.assertClean()

    def test_braced_word_with_spaces_in_prose_is_not_flagged(self):
        """Deliberate boundary: `{ x }` with internal whitespace reads as
        prose or quoted code, not a template placeholder, and is left
        alone. Pinned so the boundary is a decision, not an accident."""
        self.write_header("arm_nnfunctions.h", "/* the set { x } is unused */\n")
        self.stub_required_headers()
        self.assertClean()

    def test_internal_headers_are_out_of_scope(self):
        """Include/Internal/ is excluded as non-customer-facing API, so the
        glob is non-recursive. Note this is a scope choice, not a claim that
        doxygen ignores those headers -- nn.dxy.in INPUTs Include/
        recursively. Pinned so that widening the scope later is a deliberate
        change with a failing test to update, not a surprise."""
        self.write_header("arm_nnfunctions.h", "/* clean */\n")
        self.stub_required_headers()
        internal = self.include_dir / "Internal"
        internal.mkdir()
        (internal / "arm_conv_opt_common.h").write_text(
            "/* {API}_get_buffer_size() */\n", encoding="utf-8"
        )
        self.assertClean()

    def test_unterminated_block_comment_runs_to_end_of_file(self):
        """An unterminated /* is how a compiler would see it too; the
        placeholder after it must still be found rather than silently
        dropped."""
        self.write_header("arm_nnfunctions.h", "/* start of comment\n{API}_get_buffer_size()\n")
        self.stub_required_headers()
        self.assertFails("{API}")

    # -- rule 2: cited sizers must resolve ------------------------------------
    #
    # The token rule cannot catch #269: both of those defects resolved the
    # placeholder to a real-but-wrong name, leaving no token behind. And 8 of
    # the 11 sites in #288 had no same-named sizer, so the "obvious"
    # substitution would have named a function that does not exist. This rule
    # is what covers both.

    def test_mutation_cited_sizer_that_does_not_exist_is_caught(self):
        """A doc citing a sizer that is declared nowhere must fail. This is
        the #288 shape: 8 of 11 sites had no same-named sizer, so the
        mechanical substitution would have produced exactly this."""
        self.write_header(
            "arm_nnfunctions.h",
            "/**\n"
            " * @param[in, out] ctx  Size ctx->buf with\n"
            " *                      arm_max_pool_s8_get_buffer_size_typo().\n"
            " */\n"
            "arm_cmsis_nn_status arm_max_pool_s8(const cmsis_nn_context *ctx);\n",
        )
        self.stub_required_headers()
        self.assertFails("arm_max_pool_s8_get_buffer_size_typo", "not declared", "#269")

    def test_cited_sizer_that_is_declared_passes(self):
        """The same shape, but naming a sizer that really is declared in the
        public headers -- confirms the rule keys on resolution, not on the
        citation merely existing."""
        self.write_header(
            "arm_nnfunctions.h",
            "/**\n"
            " * @param[in, out] ctx  Size ctx->buf with arm_avgpool_s8_get_buffer_size().\n"
            " */\n"
            "int32_t arm_avgpool_s8_get_buffer_size(const int dim_dst_width, const int ch_src);\n",
        )
        self.stub_required_headers()
        self.assertClean()

    def test_declaration_inside_a_comment_does_not_count_as_declaring(self):
        """A name is only "declared" if it appears outside comments. If a
        comment mentioning a name counted, every bad citation would satisfy
        the rule by virtue of citing itself, and rule 2 would be a no-op."""
        self.write_header(
            "arm_nnfunctions.h",
            "/**\n"
            " * int32_t arm_ghost_s8_get_buffer_size(const cmsis_nn_dims *d);\n"
            " * @param[in, out] ctx  Size ctx->buf with arm_ghost_s8_get_buffer_size().\n"
            " */\n",
        )
        self.stub_required_headers()
        self.assertFails("arm_ghost_s8_get_buffer_size", "not declared")

    def test_known_absent_sizer_is_allowed_to_be_cited(self):
        """Docs deliberately name nonexistent sizers to stop the reader
        inventing one ("there is deliberately no arm_max_pool_s8_get_buffer_size()").
        That must not fail the check."""
        self.assertIn("arm_max_pool_s8_get_buffer_size", self.mod.KNOWN_ABSENT_SIZERS)
        self.write_header(
            "arm_nnfunctions.h",
            "/**\n"
            " * @param[in] ctx  No buffer needed; there is deliberately no\n"
            " *                 arm_max_pool_s8_get_buffer_size().\n"
            " */\n",
        )
        self.stub_required_headers()
        self.assertClean()

    def test_known_absent_sizer_that_becomes_real_is_caught(self):
        """The allowlist is self-cleaning: if a name claimed to be
        nonexistent is later actually declared, the prose asserting it does
        not exist has become wrong and must be fixed. Without this the
        allowlist would quietly rot into a blanket exemption."""
        self.write_header(
            "arm_nnfunctions.h",
            "/**\n"
            " * @param[in] ctx  There is deliberately no arm_max_pool_s8_get_buffer_size().\n"
            " */\n"
            "int32_t arm_max_pool_s8_get_buffer_size(const cmsis_nn_dims *d);\n",
        )
        self.stub_required_headers()
        self.assertFails("KNOWN_ABSENT_SIZERS", "now declared")

    def test_sizer_declared_in_a_sibling_public_header_resolves(self):
        """Resolution is across the whole public surface, not per-file: the
        float sizers live in arm_nnfunctions_flt.h but are cited from
        arm_nnfunctions.h."""
        self.write_header(
            "arm_nnfunctions.h",
            "/** @param[in] ctx  See arm_convolve_f32_get_buffer_size(). */\n",
        )
        self.write_header(
            "arm_nnfunctions_flt.h",
            "int32_t arm_convolve_f32_get_buffer_size(const cmsis_nn_conv_params_f32 *p);\n",
        )
        self.assertClean()

    # -- must fail loudly, not silently --------------------------------------

    def test_missing_include_dir_fails_loudly(self):
        """Include/ not existing must not be read as '0 placeholders,
        clean' -- the fail-open bug this check exists to avoid reproducing
        one layer up."""
        self.assertFails("no public headers matching", include_dir=self.tree / "does_not_exist")

    def test_empty_include_dir_fails_loudly(self):
        self.assertFails("no public headers matching")

    def test_renamed_required_header_fails_loudly(self):
        """Headers present, but arm_nnfunctions.h renamed away: a
        smaller-but-nonzero file set, which a bare non-empty check would
        wave through with the entire templated surface silently exempted."""
        self.write_header("arm_nnfunctions_flt.h", "/* clean */\n")
        self.write_header("arm_nn_types.h", "/* clean */\n")
        self.assertFails("arm_nnfunctions.h", "not found")

    def test_stats_count_headers_scanned(self):
        self.write_header("arm_nnfunctions.h", "/* clean */\n")
        self.write_header("arm_nnfunctions_flt.h", "/* clean */\n")
        self.assertClean()
        self.assertEqual(self.mod._stats.get("headers"), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
