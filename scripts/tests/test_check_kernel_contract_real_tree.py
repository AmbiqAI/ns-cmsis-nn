#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""Invariants of the kernel contract exported from the real public headers (S5 of the
kernel-contracts plan). test_check_kernel_contract.py proves the exporter's mechanics on a
fixture; this suite proves what the real tree's export is allowed to contain, so a header
change that widens the vocabulary helia-core-tester binds against (a new return type, a
guard the consumer does not model, a scratch-size query without its kernel) fails here before
the export is committed. Every rule is checked as a function over the document, and each has
a perturbed-header negative proving the rule can fail."""

import re
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from check_doxygen_params import parse_header, public_headers  # noqa: E402
from check_kernel_contract import build_contract  # noqa: E402

INCLUDE = ROOT / 'Include'
FUNCTIONS_HEADERS = ('Include/arm_nnfunctions.h', 'Include/arm_nnfunctions_flt.h')
SUPPORT_HEADERS = ('Include/arm_nnsupportfunctions.h', 'Include/arm_nnsupportfunctions_flt.h')
FLOAT_HEADERS = ('Include/arm_nnfunctions_flt.h', 'Include/arm_nnsupportfunctions_flt.h')

# The consumer models exactly these: a status, a scratch size, nothing, or a pointer into
# the output a support helper advanced. Pointer returns stay in the support header.
RETURN_TYPES = {'arm_cmsis_nn_status', 'int32_t', 'void'}
SUPPORT_RETURN_TYPES = RETURN_TYPES | {'int8_t *', 'int16_t *'}
FLOAT_GUARDS = {'ARM_NN_ENABLE_F16', 'ARM_NN_ENABLE_F32'}
SUPPORT_GUARDS = FLOAT_GUARDS | {'defined(ARM_MATH_DSP)'}
SIZER_RE = re.compile(r'^(?P<base>arm_\w+?)_get_buffer_size(?P<variant>_mve|_dsp)?$')
# A kernel with several scratch buffers names each query after the buffer it sizes.
SIZER_BUFFER_SUFFIX_RE = re.compile(r'_(temp[12]|input_ctx|output_ctx)$')


def sizer_kernel(name):
    """The kernel a scratch-size query sizes, or None when `name` is not a query."""
    match = SIZER_RE.match(name)
    if match is None:
        return None
    return SIZER_BUFFER_SUFFIX_RE.sub('', match.group('base'))


def real_tree_violations(document):
    """Every way the real export may break the vocabulary helia-core-tester binds against."""
    functions = document['functions']
    names = {record['name'] for record in functions}
    problems = []
    for record in functions:
        name, header, guards, returns = (record['name'], record['header'], record['guards'],
                                         record['returns'])
        allowed_returns = SUPPORT_RETURN_TYPES if header in SUPPORT_HEADERS else RETURN_TYPES
        if returns not in allowed_returns:
            problems.append(f'{name}: returns {returns!r}, not one of {sorted(allowed_returns)}')
        allowed_guards = SUPPORT_GUARDS if header in SUPPORT_HEADERS else FLOAT_GUARDS
        for guard in guards:
            if guard not in allowed_guards:
                problems.append(f'{name}: guard {guard!r} is not one of {sorted(allowed_guards)}')
        if header in FLOAT_HEADERS and not set(guards) & FLOAT_GUARDS:
            problems.append(f'{name}: declared in {header} outside any ARM_NN_ENABLE_F16/F32 block')
        if header not in FLOAT_HEADERS and set(guards) & FLOAT_GUARDS:
            problems.append(f'{name}: float-gated declaration outside the _flt.h headers')
        kernel = sizer_kernel(name)
        if kernel is None:
            continue
        if kernel not in names:
            problems.append(f'{name}: sizes {kernel}, which the headers do not declare')
        if returns != 'int32_t':
            problems.append(f'{name}: a scratch-size query must return int32_t, not {returns!r}')
        for param in record['params']:
            if param['direction'] != 'in':
                problems.append(f"{name}: scratch-size query parameter {param['name']} is "
                                f"{param['direction']}, queries only read their arguments")
        variant = SIZER_RE.match(name).group('variant')
        if variant and name[:-len(variant)] not in names:
            problems.append(f'{name}: a {variant[1:]} variant needs the generic query '
                            f'{name[:-len(variant)]}')
    return problems


class RealTreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = build_contract(INCLUDE)

    def test_export_count_matches_the_scanner(self):
        declared = [decl for path in public_headers(INCLUDE) for decl in parse_header(path)
                    if not decl.is_static]
        self.assertEqual(len(self.document['functions']), len(declared))
        self.assertGreater(len(declared), 400)

    def test_real_tree_holds_every_invariant(self):
        self.assertEqual(real_tree_violations(self.document), [])

    def test_vocabulary_is_actually_exercised(self):
        # A rule nobody hits is a rule that cannot catch drift in the other direction.
        functions = self.document['functions']
        self.assertEqual({r['returns'] for r in functions}, SUPPORT_RETURN_TYPES)
        self.assertEqual({g for r in functions for g in r['guards']}, SUPPORT_GUARDS)
        sizers = [r['name'] for r in functions if sizer_kernel(r['name'])]
        self.assertGreater(len(sizers), 80)
        self.assertTrue(any(SIZER_BUFFER_SUFFIX_RE.search(SIZER_RE.match(s).group('base'))
                            for s in sizers))
        self.assertTrue(any(s.endswith('_mve') for s in sizers))
        self.assertTrue(any(s.endswith('_dsp') for s in sizers))


def perturbed_include(edit):
    """A copy of Include/ with `edit(text) -> text` applied to arm_nnfunctions.h."""
    root = Path(tempfile.mkdtemp())
    include = root / 'Include'
    shutil.copytree(INCLUDE, include)
    header = include / 'arm_nnfunctions.h'
    before = header.read_text()
    after = edit(before)
    if after == before:
        raise AssertionError('the perturbation changed nothing')
    header.write_text(after)
    return root, include


class PerturbedTreeTests(unittest.TestCase):
    """Each invariant must be able to fail: perturb the real headers and watch it."""

    def violations_after(self, edit):
        root, include = perturbed_include(edit)
        try:
            return real_tree_violations(build_contract(include))
        finally:
            shutil.rmtree(root)

    def test_sizer_without_its_kernel(self):
        problems = self.violations_after(
            lambda text: text.replace('arm_cmsis_nn_status arm_convolve_s16(',
                                      'arm_cmsis_nn_status arm_convolve_s16x(', 1))
        self.assertTrue(any('arm_convolve_s16_get_buffer_size' in p and 'do not declare' in p for p in problems),
                        problems)

    def test_unlisted_return_type(self):
        problems = self.violations_after(
            lambda text: text.replace('arm_cmsis_nn_status arm_convolve_s16(', 'int64_t arm_convolve_s16(', 1))
        self.assertTrue(any(p.startswith('arm_convolve_s16: returns') for p in problems), problems)

    def test_pointer_return_outside_the_support_header(self):
        problems = self.violations_after(
            lambda text: text.replace('arm_cmsis_nn_status arm_convolve_s16(', 'int8_t * arm_convolve_s16(', 1))
        self.assertTrue(any(p.startswith('arm_convolve_s16: returns') for p in problems), problems)

    def test_unmodelled_guard(self):
        problems = self.violations_after(
            lambda text: text.replace('arm_cmsis_nn_status arm_convolve_s16(',
                                      '#if ARM_NN_ENABLE_EXPERIMENTAL\narm_cmsis_nn_status arm_convolve_s16(', 1)
                                  .replace('arm_cmsis_nn_status arm_convolve_s4(',
                                           '#endif\narm_cmsis_nn_status arm_convolve_s4(', 1))
        self.assertTrue(any("guard 'ARM_NN_ENABLE_EXPERIMENTAL'" in p for p in problems), problems)

    def test_float_gate_outside_the_float_headers(self):
        problems = self.violations_after(
            lambda text: text.replace('arm_cmsis_nn_status arm_convolve_s16(',
                                      '#if ARM_NN_ENABLE_F32\narm_cmsis_nn_status arm_convolve_s16(', 1)
                                  .replace('arm_cmsis_nn_status arm_convolve_s4(',
                                           '#endif\narm_cmsis_nn_status arm_convolve_s4(', 1))
        self.assertTrue(any('float-gated declaration outside' in p for p in problems), problems)

    def test_sizer_returning_the_wrong_type(self):
        problems = self.violations_after(
            lambda text: text.replace('int32_t arm_convolve_s16_get_buffer_size(',
                                      'void arm_convolve_s16_get_buffer_size(', 1))
        self.assertTrue(any('must return int32_t' in p for p in problems), problems)

    def test_variant_without_the_generic_query(self):
        # Renaming the generic query itself would break the variants' @copydoc and stop the
        # export; moving one variant to a name whose generic query does not exist does not.
        problems = self.violations_after(
            lambda text: text.replace('int32_t arm_avgpool_s16_get_buffer_size_dsp(',
                                      'int32_t arm_avgpool_s16x_get_buffer_size_dsp(', 1))
        self.assertIn('arm_avgpool_s16x_get_buffer_size_dsp: a dsp variant needs the generic query '
                      'arm_avgpool_s16x_get_buffer_size', problems)
        self.assertIn('arm_avgpool_s16x_get_buffer_size_dsp: sizes arm_avgpool_s16x, which the headers '
                      'do not declare', problems)

    def test_sizer_kernel_naming(self):
        self.assertEqual(sizer_kernel('arm_convolve_s8_get_buffer_size_mve'), 'arm_convolve_s8')
        self.assertEqual(sizer_kernel('arm_lstm_unidirectional_s8_temp2_get_buffer_size'), 'arm_lstm_unidirectional_s8')
        self.assertEqual(sizer_kernel('arm_svdf_state_s16_s8_input_ctx_get_buffer_size'), 'arm_svdf_state_s16_s8')
        self.assertIsNone(sizer_kernel('arm_convolve_s8'))
        self.assertIsNone(sizer_kernel('arm_get_buffer_size_helper'))


if __name__ == '__main__':
    unittest.main()
