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
 * @brief Scalar fallback that the docs build never sees.
 * @param[in] x  Input.
 */
void fx_scalar_path(int32_t x);
#endif

/**
 * @brief Reads a dims array.
 * @param[in] dims  Four dimensions.
 */
void fx_dims(const int32_t dims[4]);

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

INDEX_XML = '''\
<?xml version='1.0' encoding='UTF-8' standalone='no'?>
<doxygenindex version="1.17.0">
  <compound refid="arm__nnfunctions_8h" kind="file"><name>arm_nnfunctions.h</name></compound>
  <compound refid="group__fx" kind="group"><name>fx</name></compound>
  <compound refid="structfx__dims" kind="struct"><name>fx_dims</name></compound>
</doxygenindex>
'''


def member(name, ret, params, tags, declfile='Include/arm_nnfunctions.h', static='no', ident=None):
    """One <memberdef kind="function"> with typed params and direction-tagged @param docs."""
    param_xml = ''.join(
        f'<param><type>{ptype}</type><declname>{pname}</declname>'
        + (f'<array>{extent}</array>' if extent else '') + '</param>'
        for pname, ptype, extent in params)
    items = ''.join(
        f'<parameteritem><parameternamelist><parametername direction="{direction}">{pname}'
        '</parametername></parameternamelist><parameterdescription><para>d</para>'
        '</parameterdescription></parameteritem>' if direction else
        f'<parameteritem><parameternamelist><parametername>{pname}</parametername>'
        '</parameternamelist><parameterdescription><para>d</para></parameterdescription></parameteritem>'
        for pname, direction in tags)
    return (f'<memberdef kind="function" id="{ident or "m_" + name}" static="{static}">'
            f'<type>{ret}</type><name>{name}</name>{param_xml}'
            f'<detaileddescription><para><parameterlist kind="param">{items}</parameterlist></para>'
            f'</detaileddescription><location file="/x/{declfile}" line="1" declfile="/x/{declfile}" '
            f'declline="1"/></memberdef>')


def compound(refid, kind, members):
    return (f"<?xml version='1.0' encoding='UTF-8' standalone='no'?>\n<doxygen version=\"1.17.0\">"
            f'<compounddef id="{refid}" kind="{kind}"><compoundname>{refid}</compoundname>'
            f'<sectiondef kind="func">{"".join(members)}</sectiondef></compounddef></doxygen>\n')


FIXTURE_XML = {
    # The int-style function sits in the file compound; the "float" twin only in the
    # group compound, the way @addtogroup places every real float kernel.
    'arm__nnfunctions_8h.xml': compound('arm__nnfunctions_8h', 'file', [
        member('fx_add', 'arm_cmsis_nn_status',
               [('a', 'const int8_t *', ''), ('out', 'int8_t *', ''), ('n', 'const int32_t', '')],
               [('a', 'in'), ('out', 'out'), ('n', 'in')]),
        member('fx_static', 'int32_t', [], [], static='yes'),
        member('fx_internal', 'void', [('p', 'int32_t *', '')], [('p', 'out')],
               declfile='Include/Internal/arm_nn_compiler.h'),
    ]),
    'group__fx.xml': compound('group__fx', 'group', [
        member('fx_add_f16', 'arm_cmsis_nn_status',
               [('a', 'const int8_t *', ''), ('out', 'int8_t *', ''), ('n', 'const int32_t', '')],
               [('a', 'in'), ('out', 'out'), ('n', 'in')]),
        member('fx_dims', 'void', [('dims', 'const int32_t', '[4]')], [('dims', 'in')]),
        member('fx_cb', 'void', [('fn', 'void(*)(int32_t)', '')], [('fn', 'in')]),
        member('fx_void', 'int32_t', [('', 'void', '')], []),
        member('fx_nnfunctions_flt', 'void', [('x', 'int32_t', '')], [('x', 'in')],
               declfile='Include/arm_nnfunctions_flt.h'),
        member('fx_nnsupportfunctions', 'void', [('x', 'int32_t', '')], [('x', 'in')],
               declfile='Include/arm_nnsupportfunctions.h'),
        member('fx_nnsupportfunctions_flt', 'void', [('x', 'int32_t', '')], [('x', 'in')],
               declfile='Include/arm_nnsupportfunctions_flt.h'),
    ]),
    'structfx__dims.xml': compound('structfx__dims', 'struct', []),
}


DOXYFILE = '''\
PROJECT_NAME = fx
PREDEFINED             = FX_ENABLE_F16=1 \\
                         FX_HAVE_CB=1 \\
                         __RESTRICT=
QUIET = YES
'''


def write_xml(root, files=None):
    xml_dir = Path(root) / 'xml'
    xml_dir.mkdir(parents=True, exist_ok=True)
    (xml_dir / 'index.xml').write_text(INDEX_XML)
    for name, text in (files or FIXTURE_XML).items():
        (xml_dir / name).write_text(text)
    return xml_dir


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
            self.assertIn('exported 9 public function signatures', result.stdout)
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
        # The #else twin carries the negated condition; the array extent is kept.
        self.assertEqual(by_name['fx_scalar_path']['guards'], ['!FX_ENABLE_F16'])
        self.assertEqual(by_name['fx_dims']['guards'], [])
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
        self.assertIn('is fresh (9 public function signatures)', result.stdout)

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


class VerifyXmlTests(unittest.TestCase):
    """verify-xml against a synthetic Doxygen XML tree describing the same fixture headers."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.include = write_tree(self.root)
        self.output = Path(self.root) / 'kernel_contracts.json'
        result = run(['export', '--include-dir', str(self.include), '--output', str(self.output)])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.xml_dir = write_xml(self.root)
        self.doxyfile = Path(self.root) / 'nn.dxy.in'
        self.doxyfile.write_text(DOXYFILE)

    def tearDown(self):
        shutil.rmtree(self.root)

    def verify(self):
        return run(['verify-xml', '--xml-dir', str(self.xml_dir), '--output', str(self.output),
                    '--doxyfile', str(self.doxyfile)])

    def rewrite(self, name, transform):
        (self.xml_dir / name).write_text(transform(FIXTURE_XML[name]))

    def test_agreeing_xml_passes(self):
        result = self.verify()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('agrees with the Doxygen XML', result.stdout)
        self.assertIn('(8 functions compared, 1 under #if branches', result.stdout)

    def test_direction_mismatch(self):
        self.rewrite('group__fx.xml', lambda x: x.replace('direction="out">out', 'direction="inout">out'))
        result = self.verify()
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn('fx_add_f16.out: direction out vs XML in,out', result.stderr)

    def test_missing_direction_in_xml(self):
        self.rewrite('arm__nnfunctions_8h.xml', lambda x: x.replace(' direction="in">n<', '>n<'))
        result = self.verify()
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn('fx_add.n: no direction in the Doxygen XML', result.stderr)

    def test_function_only_in_xml(self):
        self.rewrite('group__fx.xml', lambda x: x.replace('</sectiondef>', member(
            'fx_extra', 'void', [('x', 'int32_t', '')], [('x', 'in')]) + '</sectiondef>'))
        result = self.verify()
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn('fx_extra: in the Doxygen XML (arm_nnfunctions.h) but not in the export', result.stderr)

    def test_function_only_in_export(self):
        self.rewrite('group__fx.xml', lambda x: x.replace(member(
            'fx_dims', 'void', [('dims', 'const int32_t', '[4]')], [('dims', 'in')]), ''))
        result = self.verify()
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn('fx_dims: in the export (Include/arm_nnfunctions.h) but not in the Doxygen XML', result.stderr)

    def test_parameter_order_type_extent_and_return_mismatches(self):
        self.rewrite('arm__nnfunctions_8h.xml', lambda x: x
                     .replace('<declname>a</declname>', '<declname>b</declname>', 1)
                     .replace('<type>arm_cmsis_nn_status</type><name>fx_add</name>',
                              '<type>int32_t</type><name>fx_add</name>'))
        self.rewrite('group__fx.xml', lambda x: x
                     .replace('<array>[4]</array>', '<array>[8]</array>')
                     .replace('<type>const int8_t *</type><declname>a</declname>',
                              '<type>const int16_t *</type><declname>a</declname>'))
        result = self.verify()
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("fx_add: returns 'arm_cmsis_nn_status' vs XML 'int32_t'", result.stderr)
        self.assertIn("fx_add: parameters ['a', 'out', 'n'] vs XML ['b', 'out', 'n']", result.stderr)
        self.assertIn("fx_dims.dims: extent '[4]' vs XML '[8]'", result.stderr)
        self.assertIn("fx_add_f16.a: type 'const int8_t *' vs XML 'const int16_t *'", result.stderr)

    def test_pointer_spacing_and_static_and_internal_members_are_tolerated(self):
        # Doxygen spaces `*` differently from the scanner; static and Internal/ members are
        # not part of the contract and must not be reported.
        self.rewrite('arm__nnfunctions_8h.xml', lambda x: x.replace('const int8_t *', 'const int8_t*'))
        result = self.verify()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_index_and_malformed_compound_exit_2(self):
        (self.xml_dir / 'index.xml').unlink()
        result = self.verify()
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn('index.xml: not found', result.stderr)
        (self.xml_dir / 'index.xml').write_text(INDEX_XML)
        (self.xml_dir / 'group__fx.xml').write_text('<doxygen><compounddef>')
        result = self.verify()
        self.assertEqual(result.returncode, 2, result.stdout)

    def test_function_under_an_undefined_branch_must_not_appear_in_xml(self):
        # fx_scalar_path sits under !FX_ENABLE_F16, which the Doxyfile defines as 1, so doxygen
        # cannot have documented it; seeing it means the guard evaluation is wrong.
        self.rewrite('group__fx.xml', lambda x: x.replace('</sectiondef>', member(
            'fx_scalar_path', 'void', [('x', 'int32_t', '')], [('x', 'in')]) + '</sectiondef>'))
        result = self.verify()
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("fx_scalar_path: in the Doxygen XML although its guards ['!FX_ENABLE_F16']", result.stderr)

    def test_guard_the_verifier_cannot_judge_exits_2(self):
        header = self.include / 'arm_nnfunctions.h'
        header.write_text(header.read_text().replace('#ifdef FX_HAVE_CB', '#if defined(FX_HAVE_CB) && FX_LEVEL > 1'))
        result = run(['export', '--include-dir', str(self.include), '--output', str(self.output)])
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.verify()
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn('cannot be evaluated against the Doxyfile', result.stderr)

    def test_doxyfile_defines_parse_continuations_and_empty_values(self):
        self.doxyfile.write_text('X = 1\nPREDEFINED             = A=1 \\\n                         B= \\\n                         C(x)= \\\n                         FX_ENABLE_F16=1 FX_HAVE_CB=1\n\nEXPAND = YES\n')
        result = self.verify()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.doxyfile.write_text('PREDEFINED = FX_HAVE_CB=1\n')
        # Now FX_ENABLE_F16 is undefined: fx_add_f16 must be absent and fx_scalar_path present.
        result = self.verify()
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("fx_add_f16: in the Doxygen XML although its guards ['FX_ENABLE_F16']", result.stderr)
        self.assertIn('fx_scalar_path: in the export (Include/arm_nnfunctions.h) but not in the Doxygen XML', result.stderr)
        self.doxyfile.write_text('QUIET = YES\n')
        result = self.verify()
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn('no PREDEFINED line', result.stderr)

    def test_missing_export_exits_2(self):
        self.output.unlink()
        result = self.verify()
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn('cannot be read', result.stderr)


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
