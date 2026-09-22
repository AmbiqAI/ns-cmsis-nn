#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""Mutation tests for scripts/check_doxygen_params.py, plus a run over the real headers."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / 'scripts/check_doxygen_params.py'

# Every syntactic shape the real public headers use, all correctly documented.
CLEAN = '''\
/*
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef FIXTURE_H
#define FIXTURE_H

#include <stdint.h>
#define LONG_MACRO(x) \\
    ((x) + 1)

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief A lookup table that must not be credited to the next function.
 */
extern const int8_t fx_table[16];

/**
 * @brief Rounding helper macro.
 */
#define FX_ROUND(x) ((x) + 1)

/** A union with its own doc block. */
union fx_word
{
    int32_t word;
    int8_t bytes[4];
};

/* A plain comment, not a doc block. */
/**
 * @brief Convolution-shaped kernel, see https://example.com/kernels#s8 and the "S8" note.
 * @param[in, out] ctx          Function context that may hold a scratch buffer.
 * @param[in]      params       Kernel parameters.
 * @param[in]      input_data   Input tensor.
 * @param[out]     output_data  Output tensor.
 * @param[in]      size         Number of elements.
 * @return     The function returns <code>ARM_CMSIS_NN_SUCCESS</code>
 */
arm_cmsis_nn_status fx_kernel_s8(const cmsis_nn_context *ctx,
                                 const cmsis_nn_conv_params *params,
                                 const int8_t *input_data, // NHWC
                                 int8_t *output_data,
                                 const int32_t size);

#if defined(ARM_MATH_MVEI)
/**
 * @brief Buffer size for the MVE leg.
 * @copydetails fx_kernel_s8_get_buffer_size
 */
int32_t fx_kernel_s8_get_buffer_size_mve(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims);
#endif

/**
 * @brief Buffer size query.
 * @param[in] input_dims   Input tensor dimensions.
 * @param[in] filter_dims  Filter tensor dimensions.
 * @return     Size in bytes.
 */
int32_t
fx_kernel_s8_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims);

/**
 * @brief Pointer shapes.
 * @param[in,out]  advanced    Pointer that is advanced by the callee.
 * @param[in]      rows        Row pointers, all read-only.
 * @param[out]     outputs     Output row pointers.
 * @param[in]      dims        Fixed-size dimension array.
 * @param[in]      bias        Read-only bias with a const pointer.
 * @param[in]      scalar      Plain scalar.
 */
void fx_pointers(const int8_t **advanced,
                 const int8_t *const *rows,
                 int8_t *const *outputs,
                 const int32_t dims[4],
                 const int32_t *const bias,
                 int32_t scalar);

/**
 * @brief Inline helper whose body has braces in a string, a character literal, and a
 *        conditional whose branches share one closing brace.
 * @param[out] dst         Destination buffer.
 * @param[in]  src         Source buffer.
 * @param[in]  block_size  Number of elements to copy.
 */
__STATIC_FORCEINLINE void
fx_memcpy(int8_t *__RESTRICT dst, const int8_t *__RESTRICT src, uint32_t block_size)
{
#if defined(ARM_MATH_MVEI)
    __asm volatile("   wlstp.8 lr, %[cnt], 1f {  \\n" : : [cnt] "r"(block_size));
    if (block_size) {
#elif defined(ARM_MATH_DSP)
    if (block_size > 1) {
#else
    if (src[0] != '{') {
#endif
        for (uint32_t i = 0; i < block_size; i++) { dst[i] = src[i]; }
    }
}

/**
 * @brief Attribute before the return type, a struct return, and a callback parameter.
 * @param[in]  words     Words to fold.
 * @param[in]  fold      Callback applied to each word.
 * @param[out] result    Receives the folded word.
 * @return     The union that received the result.
 */
__attribute__((warn_unused_result)) union fx_word *
fx_fold(const union fx_word *words, int32_t (*fold)(int32_t, int32_t), union fx_word *result);

/**
 * @brief Inline twin; doxygen cannot resolve a copy directive to a static target, so it carries its own tags.
 * @param[out] dst         Destination buffer.
 * @param[in]  src         Source buffer.
 * @param[in]  block_size  Number of elements to copy.
 */
__STATIC_FORCEINLINE void
fx_memcpy_twin(int8_t *__RESTRICT dst, const int8_t *__RESTRICT src, uint32_t block_size)
{
    fx_memcpy(dst, src, block_size);
}

#ifdef __cplusplus
}
#endif

#endif
'''

TWIN_TAGS = '''\
 * @brief Inline twin; doxygen cannot resolve a copy directive to a static target, so it carries its own tags.
 * @param[out] dst         Destination buffer.
 * @param[in]  src         Source buffer.
 * @param[in]  block_size  Number of elements to copy.
'''


def run(args, cwd=None):
    return subprocess.run([sys.executable, str(CHECKER), *args], capture_output=True, text=True, cwd=cwd)


class FixtureTests(unittest.TestCase):
    def check(self, text, name='fixture.h'):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / name).write_text(text)
            return run(['--include-dir', directory, name])

    def assert_fails(self, text, function, fragment):
        result = self.check(text)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn(function, result.stderr)
        self.assertIn(fragment, result.stderr)
        return result

    def test_clean_fixture_passes(self):
        result = self.check(CLEAN)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('OK: 7 declarations in 1 headers', result.stdout)

    def test_missing_param(self):
        text = CLEAN.replace(' * @param[in]      size         Number of elements.\n', '')
        self.assert_fails(text, 'fx_kernel_s8', 'missing @param size')

    def test_extra_param(self):
        text = CLEAN.replace(' * @param[in]      size         Number of elements.\n',
                             ' * @param[in]      size         Number of elements.\n'
                             ' * @param[in]      bogus        Not a parameter.\n')
        self.assert_fails(text, 'fx_kernel_s8', '@param bogus: no such parameter')

    def test_duplicate_param(self):
        line = ' * @param[in]      size         Number of elements.\n'
        self.assert_fails(CLEAN.replace(line, line + line), 'fx_kernel_s8', 'duplicate @param size')

    def test_missing_direction(self):
        text = CLEAN.replace('@param[in]      size', '@param          size')
        self.assert_fails(text, 'fx_kernel_s8', '@param size: missing direction')

    def test_unrecognized_direction(self):
        text = CLEAN.replace('@param[in]      size', '@param[inout]   size')
        self.assert_fails(text, 'fx_kernel_s8', 'unrecognized direction [inout]')

    def test_direction_versus_type(self):
        cases = (
            ('@param[in]      input_data', '@param[out]     input_data', 'fx_kernel_s8', 'expects [in]'),
            ('@param[out]     output_data', '@param[in]      output_data', 'fx_kernel_s8', 'expects [in,out] or [out]'),
            ('@param[in]      size', '@param[out]     size', 'fx_kernel_s8', 'expects [in]'),
            ('@param[in,out]  advanced', '@param[in]      advanced', 'fx_pointers', 'expects [in,out] or [out]'),
            ('@param[in]      rows', '@param[out]     rows', 'fx_pointers', 'expects [in]'),
            ('@param[out]     outputs', '@param[in]      outputs', 'fx_pointers', 'expects [in,out] or [out]'),
            ('@param[in]      dims', '@param[out]     dims', 'fx_pointers', 'expects [in]'),
            ('@param[in]      bias', '@param[in,out]  bias', 'fx_pointers', 'expects [in]'),
            ('@param[in]  src', '@param[out] src', 'fx_memcpy', 'expects [in]'),
        )
        for old, new, function, fragment in cases:
            with self.subTest(mutation=new):
                self.assertIn(old, CLEAN)
                self.assert_fails(CLEAN.replace(old, new), function, fragment)

    def test_context_pointer_accepts_in_and_inout_only(self):
        for direction, ok in (('[in, out]', True), ('[in]', True), ('[out]', False)):
            with self.subTest(direction=direction):
                text = CLEAN.replace('@param[in, out] ctx', f'@param{direction} ctx')
                result = self.check(text)
                self.assertEqual(result.returncode, 0 if ok else 1, result.stderr)

    def test_copydoc_unknown_target(self):
        text = CLEAN.replace(TWIN_TAGS, ' * @copydoc fx_nonexistent\n')
        self.assert_fails(text, 'fx_memcpy_twin', 'unknown @copydoc target fx_nonexistent')

    def test_copydoc_to_static_target_is_rejected(self):
        text = CLEAN.replace(TWIN_TAGS, ' * @copydoc fx_memcpy\n')
        self.assert_fails(text, 'fx_memcpy_twin', 'target fx_memcpy is a static function')

    def test_copydoc_cycle(self):
        text = CLEAN.replace('@copydetails fx_kernel_s8_get_buffer_size\n',
                             '@copydetails fx_kernel_s8_get_buffer_size_mve\n')
        self.assert_fails(text, 'fx_kernel_s8_get_buffer_size_mve', '@copydoc cycle')

    def test_copydoc_chain_follows_to_terminal_tags(self):
        chain = CLEAN.replace(' * @brief Buffer size for the MVE leg.\n * @copydetails fx_kernel_s8_get_buffer_size\n',
                              ' * @copydoc fx_kernel_s8_get_buffer_size_dsp\n')
        chain = chain.replace('#if defined(ARM_MATH_MVEI)\n',
                              '/**\n * @copydoc fx_kernel_s8_get_buffer_size\n */\n'
                              'int32_t fx_kernel_s8_get_buffer_size_dsp(const cmsis_nn_dims *input_dims, '
                              'const cmsis_nn_dims *filter_dims);\n\n#if defined(ARM_MATH_MVEI)\n')
        self.assertEqual(self.check(chain).returncode, 0)
        broken = chain.replace(' * @param[in] filter_dims  Filter tensor dimensions.\n', '')
        result = self.assert_fails(broken, 'fx_kernel_s8_get_buffer_size_mve', 'missing @param filter_dims')
        self.assertIn('fx_kernel_s8_get_buffer_size_dsp', result.stderr)

    def test_copydoc_target_with_different_parameter_names(self):
        text = CLEAN.replace('fx_kernel_s8_get_buffer_size_mve(const cmsis_nn_dims *input_dims,',
                             'fx_kernel_s8_get_buffer_size_mve(const cmsis_nn_dims *in_dims,')
        result = self.assert_fails(text, 'fx_kernel_s8_get_buffer_size_mve', 'missing @param in_dims')
        self.assertIn('@param input_dims: no such parameter', result.stderr)

    def test_multiple_copy_directives_are_rejected(self):
        text = CLEAN.replace(' * @copydetails fx_kernel_s8_get_buffer_size\n',
                             ' * @copydetails fx_kernel_s8_get_buffer_size\n * @copydoc fx_nonexistent\n')
        self.assert_fails(text, 'fx_kernel_s8_get_buffer_size_mve', 'multiple copy directives')

    def test_copydoc_mixed_with_params(self):
        text = CLEAN.replace(' * @copydetails fx_kernel_s8_get_buffer_size\n',
                             ' * @copydetails fx_kernel_s8_get_buffer_size\n * @param[in] input_dims Dims.\n')
        self.assert_fails(text, 'fx_kernel_s8_get_buffer_size_mve', 'mixes @copydoc with @param')

    def test_no_doc_block(self):
        block = ('/**\n * @brief Buffer size query.\n'
                 ' * @param[in] input_dims   Input tensor dimensions.\n'
                 ' * @param[in] filter_dims  Filter tensor dimensions.\n'
                 ' * @return     Size in bytes.\n */\n')
        self.assertIn(block, CLEAN)
        cases = {
            'deleted': CLEAN.replace(block, ''),
            'plain comment': CLEAN.replace(block, block.replace('/**', '/*', 1)),
            'define between': CLEAN.replace(block, block + '#define FX_BETWEEN 1\n'),
            'block comment between': CLEAN.replace(block, block + '/* not a doc block */\n'),
            'line comment between': CLEAN.replace(block, block + '// not a doc block\n'),
            'blank doc then decl': CLEAN.replace(block, block + '/** @brief orphan */\nextern int fx_var;\n'),
        }
        for label, text in cases.items():
            with self.subTest(case=label):
                self.assert_fails(text, 'fx_kernel_s8_get_buffer_size', 'no doc block')
        conditional = CLEAN.replace(block, block + '#if ARM_NN_ENABLE_F32\n').replace(
            'const cmsis_nn_dims *filter_dims);\n\n/**\n * @brief Pointer shapes.',
            'const cmsis_nn_dims *filter_dims);\n#endif\n\n/**\n * @brief Pointer shapes.')
        self.assertEqual(self.check(conditional).returncode, 0, self.check(conditional).stderr)

    def test_conditional_branch_hides_nothing(self):
        text = CLEAN.replace(' * @brief Buffer size for the MVE leg.\n * @copydetails fx_kernel_s8_get_buffer_size\n',
                             ' * @brief Buffer size for the MVE leg.\n * @param[in] input_dims Dims.\n')
        self.assert_fails(text, 'fx_kernel_s8_get_buffer_size_mve', 'missing @param filter_dims')

    # Each of the following shapes once made the scanner drop a declaration and exit 0. The
    # twin's missing tag is the canary: it is only reported if the scanner reached it.
    TWIN_MISSING_DST = CLEAN.replace(' * @param[out] dst         Destination buffer.\n'
                                     ' * @param[in]  src         Source buffer.\n'
                                     ' * @param[in]  block_size  Number of elements to copy.\n'
                                     ' */\n__STATIC_FORCEINLINE void\nfx_memcpy_twin',
                                     ' * @param[in]  src         Source buffer.\n'
                                     ' * @param[in]  block_size  Number of elements to copy.\n'
                                     ' */\n__STATIC_FORCEINLINE void\nfx_memcpy_twin')

    def assert_twin_still_checked(self, text):
        self.assertNotEqual(text, CLEAN)
        self.assert_fails(text, 'fx_memcpy_twin', 'missing @param dst')

    def test_url_or_quote_in_doc_block_does_not_swallow_the_next_declaration(self):
        for label, tail in (('url on closing line', ' see https://example.com/x */'),
                            ('odd quote on closing line', ' the "S8 path */')):
            with self.subTest(case=label):
                text = self.TWIN_MISSING_DST.replace(' * @brief Inline twin; doxygen cannot resolve a copy '
                                                     'directive to a static target, so it carries its own tags.\n'
                                                     ' * @param[in]  src         Source buffer.\n'
                                                     ' * @param[in]  block_size  Number of elements to copy.\n */',
                                                     ' * @brief Inline twin' + tail)
                result = self.check(text)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn('fx_memcpy_twin', result.stderr)
                self.assertIn('missing @param', result.stderr)

    def test_shared_closing_brace_across_preprocessor_branches(self):
        self.assertIn('#elif defined(ARM_MATH_DSP)\n    if (block_size > 1) {\n', CLEAN)
        self.assert_twin_still_checked(self.TWIN_MISSING_DST)

    def test_character_literal_brace_in_inline_body(self):
        self.assertIn("if (src[0] != '{') {", CLEAN)
        text = self.TWIN_MISSING_DST.replace("if (src[0] != '{') {", "if (src[0] != '\\\\{' && src[1] != '}') {")
        self.assert_twin_still_checked(text)

    def test_unbalanced_braces_fail_loud(self):
        body = '    fx_memcpy(dst, src, block_size);\n}'
        self.assertIn(body, CLEAN)
        cases = (
            ('extra close', body, body + '\n}', 'unexpected closing brace'),
            ('never closed', body, body[:-2], 'extern block opened here is never closed'),
        )
        for label, old, new, fragment in cases:
            with self.subTest(case=label):
                self.assertIn(old, CLEAN)
                result = self.check(CLEAN.replace(old, new))
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn('fixture.h:', result.stderr)
                self.assertIn(fragment, result.stderr)

    def test_struct_and_enum_return_types_are_functions(self):
        self.assertIn('union fx_word *\nfx_fold(', CLEAN)
        text = CLEAN.replace(' * @param[out] result    Receives the folded word.\n', '')
        self.assert_fails(text, 'fx_fold', 'missing @param result')
        text = CLEAN.replace('union fx_word *\nfx_fold(', 'enum fx_kind\nfx_fold(').replace(
            ' * @param[out] result    Receives the folded word.\n', '')
        self.assert_fails(text, 'fx_fold', 'missing @param result')

    def test_code_after_a_same_line_comment_is_still_a_declaration(self):
        # A plain comment is a boundary (the block above no longer counts); a one-line doc
        # block is the declaration's own. Either way the declaration must be seen.
        for label, prefix, fragment in (('plain comment', '/* MVE only */ ', 'no doc block'),
                                        ('one-line doc block', '/** @brief Twin. */ ', 'missing @param dst')):
            with self.subTest(case=label):
                text = CLEAN.replace(TWIN_TAGS + ' */\n__STATIC_FORCEINLINE void\nfx_memcpy_twin',
                                     ' * @brief Orphan block.\n */\n' + prefix + '__STATIC_FORCEINLINE void fx_memcpy_twin')
                self.assertNotEqual(text, CLEAN)
                self.assert_fails(text, 'fx_memcpy_twin', fragment)

    def test_attribute_before_the_return_type_does_not_name_the_function(self):
        self.assertIn('__attribute__((warn_unused_result)) union fx_word *', CLEAN)
        text = CLEAN.replace(' * @param[in]  words     Words to fold.\n', '')
        result = self.assert_fails(text, 'fx_fold', 'missing @param words')
        self.assertNotIn('__attribute__', result.stderr)

    def test_function_pointer_parameter_is_input_only(self):
        self.assertIn('int32_t (*fold)(int32_t, int32_t)', CLEAN)
        text = CLEAN.replace('@param[in]  fold', '@param[out] fold')
        self.assert_fails(text, 'fx_fold', '@param fold (int32_t (*fold)(int32_t, int32_t)): tagged [out], expects [in]')

    def test_unparseable_parameter_is_reported_against_its_declaration(self):
        text = CLEAN.replace('int32_t (*fold)(int32_t, int32_t)', 'int32_t (*)(int32_t, int32_t)')
        result = self.assert_fails(text, 'fx_fold', "cannot parse parameter 'int32_t (*)(int32_t, int32_t)'")
        fold_line = CLEAN.splitlines().index('__attribute__((warn_unused_result)) union fx_word *') + 1
        self.assertIn(f'fixture.h:{fold_line}: fx_fold:', result.stderr)
        # The rest of the header is still checked.
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'fixture.h').write_text(text)
            listing = run(['--include-dir', directory, '--list', 'fixture.h'])
        self.assertIn('fx_memcpy_twin', listing.stdout)
        self.assertIn("fx_fold\t!cannot parse parameter", listing.stdout)

    def test_missing_and_empty_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run(['--include-dir', directory, 'absent.h'])
            self.assertEqual(result.returncode, 1)
            self.assertIn('header not found', result.stderr)
            (Path(directory) / 'empty.h').write_text('/* nothing here */\n')
            result = run(['--include-dir', directory, 'empty.h'])
            self.assertEqual(result.returncode, 1)
            self.assertIn('no function declarations found', result.stderr)

    def test_list_output(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'fixture.h').write_text(CLEAN)
            result = run(['--include-dir', directory, '--list', 'fixture.h'])
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = {line.split('\t')[1]: line for line in result.stdout.splitlines() if '\t' in line}
        self.assertEqual(len(lines), 7)
        self.assertIn('advanced=const int8_t * *:in,out', lines['fx_pointers'])
        self.assertIn('dims=const int32_t:in', lines['fx_pointers'])
        self.assertIn('dst=int8_t *:out', lines['fx_memcpy_twin'])
        self.assertIn('block_size=uint32_t:in', lines['fx_memcpy'])
        kernel_line = CLEAN.splitlines().index('arm_cmsis_nn_status fx_kernel_s8(const cmsis_nn_context *ctx,') + 1
        self.assertTrue(lines['fx_kernel_s8'].startswith(f'fixture.h:{kernel_line}\t')
                        or f'fixture.h:{kernel_line}\t' in lines['fx_kernel_s8'], lines['fx_kernel_s8'])


class RealHeaderTests(unittest.TestCase):
    def test_public_headers_are_clean(self):
        result = run([], cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('OK: ', result.stdout)


if __name__ == '__main__':
    unittest.main()
