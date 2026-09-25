#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""Tests for scripts/check_kernel_contract.py: the export shape, its fail-loud paths, the
freshness check, and a run over the real headers."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/check_kernel_contract.py'
sys.path.insert(0, str(SCRIPT.parent))
from check_doxygen_params import DIRECTIONS, parse_header, public_headers  # noqa: E402
from check_kernel_contract import SCHEMA, render  # noqa: E402

# Pre-commit's check-added-large-files default; the export must stay reviewable and
# committable without a per-file exemption.
LARGE_FILE_LIMIT = 500 * 1024

FIXTURE = '''\
#ifndef FX_H
#define FX_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Elementwise add.
 * @param[in]  a    First input.
 * @param[out] out  Output.
 * @param[in]  n    Element count.
 * @return status
 */
arm_cmsis_nn_status fx_add(const int8_t *a, int8_t *out, const int32_t n);

#if FX_ENABLE_F16
/**
 * @copydoc fx_add
 */
arm_cmsis_nn_status fx_add_f16(const int8_t *a, int8_t *out, const int32_t n);
#else
/**
 * @brief Reads a dims array.
 * @param[in] dims  Four dimensions.
 */
void fx_dims(const int32_t dims[4]);
#endif

#ifdef FX_HAVE_CB
/**
 * @brief Takes a callback.
 * @param[in] fn  Called once.
 */
void fx_cb(void (*fn)(int32_t));
#endif

/**
 * @brief No parameters.
 * @return something
 */
int32_t fx_void(void);

/**
 * @brief Not exported.
 */
static inline int32_t fx_static(void) { return 0; }

#ifdef __cplusplus
}
#endif

#endif /* FX_H */
'''

FILLER = '''\
/**
 * @brief Filler.
 * @param[in] x  Input.
 */
void {name}(int32_t x);
'''

PUBLIC_HEADER_NAMES = ('arm_nnfunctions.h', 'arm_nnfunctions_flt.h',
                       'arm_nnsupportfunctions.h', 'arm_nnsupportfunctions_flt.h')


def run(args, cwd=None):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, cwd=cwd)


def write_tree(root, main_text=FIXTURE):
    """A fake checkout: <root>/Include with the four public headers; returns the include dir."""
    include = Path(root) / 'Include'
    include.mkdir(parents=True, exist_ok=True)
    (include / PUBLIC_HEADER_NAMES[0]).write_text(main_text)
    for name in PUBLIC_HEADER_NAMES[1:]:
        (include / name).write_text(FILLER.format(name='fx_' + name.replace('.h', '').replace('arm_', '')))
    return include


class ExportTests(unittest.TestCase):
    def export(self, root):
        include, output = write_tree(root), Path(root) / 'out' / 'kernel_contracts.json'
        result = run(['export', '--include-dir', str(include), '--output', str(output)])
        return result, output

    def test_export_shape(self):
        with tempfile.TemporaryDirectory() as root:
            result, output = self.export(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('exported 8 public function signatures', result.stdout)
            raw = output.read_text()
            document = json.loads(raw)
        self.assertEqual(document['schema'], SCHEMA)
        by_name = {record['name']: record for record in document['functions']}
        self.assertNotIn('fx_static', by_name)
        add = by_name['fx_add']
        self.assertEqual(add['header'], 'Include/arm_nnfunctions.h')
        self.assertEqual(add['guards'], [])
        self.assertEqual(add['returns'], 'arm_cmsis_nn_status')
        self.assertEqual(add['params'], [
            {'name': 'a', 'type': 'const int8_t *', 'direction': 'in'},
            {'name': 'out', 'type': 'int8_t *', 'direction': 'out'},
            {'name': 'n', 'type': 'const int32_t', 'direction': 'in'},
        ])
        self.assertEqual(add['line'], FIXTURE.splitlines().index(
            'arm_cmsis_nn_status fx_add(const int8_t *a, int8_t *out, const int32_t n);') + 1)
        # @copydoc resolved into directions; the #if branch recorded as its condition.
        self.assertEqual(by_name['fx_add_f16']['params'], add['params'])
        self.assertEqual(by_name['fx_add_f16']['guards'], ['FX_ENABLE_F16'])
        # The #else twin carries the negated condition and keeps its array extent.
        self.assertEqual(by_name['fx_dims']['guards'], ['!FX_ENABLE_F16'])
        self.assertEqual(by_name['fx_dims']['params'],
                         [{'name': 'dims', 'type': 'const int32_t', 'direction': 'in', 'extent': '[4]'}])
        self.assertEqual(by_name['fx_cb']['guards'], ['defined(FX_HAVE_CB)'])
        self.assertEqual(by_name['fx_cb']['params'][0]['direction'], 'in')
        self.assertIn('(*fn)', by_name['fx_cb']['params'][0]['type'])
        self.assertEqual(by_name['fx_void']['params'], [])
        self.assertEqual(by_name['fx_void']['returns'], 'int32_t')
        # Include guards and the extern "C" wrapper never appear as guards.
        noise = {'defined(FX_H)', '!defined(FX_H)', 'defined(__cplusplus)', '!defined(__cplusplus)'}
        for record in document['functions']:
            self.assertFalse(noise & set(record['guards']), record['name'])
        # Sorted by header then name, canonical rendering.
        keys = [(record['header'], record['name']) for record in document['functions']]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(raw, render(document))
        self.assertEqual(list(document['functions'][0]), ['name', 'header', 'line', 'guards', 'returns', 'params'])

    def test_undocumented_parameter_aborts_with_no_file(self):
        broken = FIXTURE.replace(' * @param[in]  n    Element count.\n', '', 1)
        with tempfile.TemporaryDirectory() as root:
            include = write_tree(root, broken)
            output = Path(root) / 'out' / 'kernel_contracts.json'
            result = run(['export', '--include-dir', str(include), '--output', str(output)])
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertIn('fx_add: missing @param n', result.stderr)
            self.assertFalse(output.exists())
            self.assertEqual(list(output.parent.glob('*')) if output.parent.exists() else [], [])

    def test_unrecognized_declaration_aborts(self):
        broken = FIXTURE.replace('int32_t fx_void(void);', 'int32_t fx_void(void) FX_DEPRECATED;', 1)
        with tempfile.TemporaryDirectory() as root:
            include = write_tree(root, broken)
            result = run(['export', '--include-dir', str(include), '--output', f'{root}/o.json'])
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn('unrecognized declaration', result.stderr)

    def test_duplicate_name_across_headers_aborts(self):
        with tempfile.TemporaryDirectory() as root:
            include = write_tree(root)
            (include / 'arm_nnfunctions_flt.h').write_text(FILLER.format(name='fx_add'))
            result = run(['export', '--include-dir', str(include), '--output', f'{root}/o.json'])
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn('fx_add is declared twice', result.stderr)

    def test_too_few_public_headers_aborts(self):
        with tempfile.TemporaryDirectory() as root:
            include = write_tree(root)
            (include / 'arm_nnsupportfunctions_flt.h').unlink()
            result = run(['export', '--include-dir', str(include), '--output', f'{root}/o.json'])
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn('expected at least 4 public headers', result.stderr)

    @unittest.skipIf(os.geteuid() == 0, 'root ignores file permissions')
    def test_unreadable_header_aborts_with_no_file(self):
        with tempfile.TemporaryDirectory() as root:
            include = write_tree(root)
            (include / 'arm_nnfunctions_flt.h').chmod(0)
            output = Path(root) / 'o.json'
            try:
                result = run(['export', '--include-dir', str(include), '--output', str(output)])
            finally:
                (include / 'arm_nnfunctions_flt.h').chmod(0o644)
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertIn('ERROR:', result.stderr)
            self.assertFalse(output.exists())

    @unittest.skipIf(os.geteuid() == 0, 'root ignores file permissions')
    def test_unwritable_output_directory_aborts(self):
        with tempfile.TemporaryDirectory() as root:
            include = write_tree(root)
            locked = Path(root) / 'locked'
            locked.mkdir()
            locked.chmod(0o500)
            try:
                result = run(['export', '--include-dir', str(include), '--output', str(locked / 'o.json')])
            finally:
                locked.chmod(0o700)
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertIn('ERROR:', result.stderr)
            self.assertEqual(list(locked.glob('*')), [])


class CheckTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.include = write_tree(self.root)
        self.output = Path(self.root) / 'Tests' / 'KernelContracts' / 'kernel_contracts.json'
        result = run(['export', '--include-dir', str(self.include), '--output', str(self.output)])
        self.assertEqual(result.returncode, 0, result.stderr)

    def tearDown(self):
        shutil.rmtree(self.root)

    def check(self):
        return run(['check', '--include-dir', str(self.include), '--output', str(self.output)])

    def test_fresh_export_passes_and_is_the_default_command(self):
        result = run(['--include-dir', str(self.include), '--output', str(self.output)])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('is fresh (8 public function signatures)', result.stdout)

    def test_moved_lines_are_still_fresh(self):
        header = self.include / 'arm_nnfunctions.h'
        header.write_text('\n\n\n' + header.read_text())
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_changed_direction_is_stale(self):
        header = self.include / 'arm_nnfunctions.h'
        header.write_text(header.read_text().replace(' * @param[out] out  Output.', ' * @param[in,out] out  Output.'))
        result = self.check()
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn('stale', result.stderr)
        self.assertIn('~ fx_add', result.stderr)

    def test_added_and_removed_declarations_are_stale(self):
        header = self.include / 'arm_nnsupportfunctions.h'
        header.write_text(FILLER.format(name='fx_new'))
        result = self.check()
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn('+ fx_new', result.stderr)
        self.assertIn('- fx_nnsupportfunctions', result.stderr)

    def test_missing_file_fails(self):
        self.output.unlink()
        result = self.check()
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn('missing', result.stderr)

    def test_corrupt_file_fails(self):
        self.output.write_text(self.output.read_text()[:40])
        result = self.check()
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn('cannot be read as JSON', result.stderr)

    def test_wrong_schema_fails(self):
        document = json.loads(self.output.read_text())
        document['schema'] = 'ns-cmsis-nn/kernel-contracts/0'
        self.output.write_text(render(document))
        result = self.check()
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn('schema', result.stderr)

    def test_non_canonical_rendering_is_stale(self):
        document = json.loads(self.output.read_text())
        self.output.write_text(json.dumps(document, indent=4))
        result = self.check()
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn('not in canonical form', result.stderr)

    def test_record_with_unknown_shape_is_malformed(self):
        document = json.loads(self.output.read_text())
        del document['functions'][0]['returns']
        self.output.write_text(json.dumps(document))
        result = self.check()
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn('malformed record', result.stderr)


class RealHeaderTests(unittest.TestCase):
    def test_real_headers_export(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / 'kernel_contracts.json'
            result = run(['export', '--output', str(output)], cwd=ROOT)
            self.assertEqual(result.returncode, 0, result.stderr)
            document = json.loads(output.read_text())
            size = output.stat().st_size
        self.assertLess(size, LARGE_FILE_LIMIT,
                        'kernel_contracts.json would trip check-added-large-files; split it per header')
        functions = document['functions']
        expected = [decl for path in public_headers(ROOT / 'Include')
                    for decl in parse_header(path) if not decl.is_static]
        self.assertEqual(len(functions), len(expected))
        self.assertEqual(len({record['name'] for record in functions}), len(functions))
        self.assertEqual({record['header'] for record in functions},
                         {f'Include/{name}' for name in PUBLIC_HEADER_NAMES})
        for record in functions:
            self.assertTrue(record['returns'], record['name'])
            for param in record['params']:
                self.assertIn(param['direction'], DIRECTIONS, f"{record['name']}.{param['name']}")
            for guard in record['guards']:
                self.assertRegex(guard, r'^!?(\(?defined\(\w+\)\)?|ARM_NN_ENABLE_F(16|32))$', record['name'])


if __name__ == '__main__':
    unittest.main()
