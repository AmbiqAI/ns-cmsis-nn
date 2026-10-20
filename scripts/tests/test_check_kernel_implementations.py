#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""Tests for scripts/check_kernel_implementations.py: the real manifest is fresh and canonical, and every validation
rule rejects a copied tree mutated to break it, with the exact diagnostic."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/check_kernel_implementations.py'
sys.path.insert(0, str(SCRIPT.parent))
from check_kernel_implementations import (  # noqa: E402
    OUTPUT, SCHEMA, SIGNATURES, SOURCES, ExportError, build_document, check, render)

FC = SOURCES / 'fully_connected_s8.yaml'
DW = SOURCES / 'depthwise_conv_s8.yaml'
POOL = SOURCES / 'pooling_s8.yaml'


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)


class RealTree(unittest.TestCase):
    def test_committed_manifest_is_fresh(self):
        self.assertEqual(check(ROOT, ROOT / OUTPUT), 13)
        result = run('check')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('is fresh (13 kernel implementations)', result.stdout)

    def test_envelope_binds_the_inputs(self):
        document = build_document(ROOT)
        self.assertEqual(document['schema'], SCHEMA)
        self.assertEqual(document['signature_schema'], 'ns-cmsis-nn/kernel-contracts/1')
        self.assertEqual(document['signature_sha256'], hashlib.sha256((ROOT / SIGNATURES).read_bytes()).hexdigest())
        self.assertEqual(document['planner_abi'], 1)
        version = '.'.join(__import__('re').findall(
            r'NS_CMSIS_NN_VERSION_(?:MAJOR|MINOR|PATCH)\s*\((\d+)\)', (ROOT / 'Include/arm_nn_types.h').read_text()))
        self.assertEqual(document['library_version'], version)

    def test_pilot_families_are_covered(self):
        ids = {record['id'] for record in build_document(ROOT)['implementations']}
        self.assertEqual(ids, {
            'fully_connected_s8', 'fully_connected_per_channel_s8', 'fully_connected_per_channel_packed_s8',
            'depthwise_conv_s8', 'depthwise_conv_3x3_s8', 'depthwise_conv_s8_opt', 'depthwise_conv_s8_opt_planar',
            'depthwise_conv_s8_opt_channelwise', 'depthwise_conv_s8_opt_3x3', 'depthwise_conv_s8_opt_3x3_c64_s1',
            'depthwise_conv_s8_to_conv', 'avg_pool_s8', 'max_pool_s8'})

    def test_derived_preparation_fields(self):
        records = {record['id']: record for record in build_document(ROOT)['implementations']}
        packed = records['fully_connected_per_channel_packed_s8']
        self.assertEqual([step['fills'] for step in packed['prepare']], ['kernel_sums', 'packed_weights'])
        self.assertEqual(packed['prepare'][0]['consumes'],
                         ['filter_data', 'fc_params.input_offset', 'fc_params.filter_offset', 'bias_data'])
        self.assertTrue(packed['eligibility']['predicate_static'])
        self.assertFalse(records['depthwise_conv_s8_opt_planar']['eligibility']['predicate_static'])

    def test_canonical_form_survives_javascript_round_trip(self):
        # release-please's json updater rewrites library_version with JSON.stringify(_, null, 2).
        node = shutil.which('node')
        if node is None:
            self.skipTest('node is not installed')
        text = (ROOT / OUTPUT).read_text()
        script = ('const t=require("fs").readFileSync(0,"utf8");'
                  'process.stdout.write(JSON.stringify(JSON.parse(t),null,2)+"\\n")')
        result = subprocess.run([node, '-e', script], input=text, capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout, text)

    def test_release_please_bumps_the_library_version(self):
        config = json.loads((ROOT / 'release-please-config.json').read_text())
        entries = config['packages']['.']['extra-files']
        self.assertIn({'type': 'json', 'path': OUTPUT.as_posix(), 'jsonpath': '$.library_version'}, entries)


class CopiedTree(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        shutil.copytree(ROOT / 'Include', self.root / 'Include')
        shutil.copytree(ROOT / 'Tests/KernelContracts', self.root / 'Tests/KernelContracts')
        shutil.copytree(ROOT / 'Planner/include', self.root / 'Planner/include')

    def tearDown(self):
        self.tmp.cleanup()

    def edit(self, rel, old, new, count=1):
        path = self.root / rel
        text = path.read_text()
        self.assertEqual(text.count(old), count, f'{old!r} occurs {text.count(old)} times in {rel}')
        path.write_text(text.replace(old, new))

    def edit_signature(self, name, change):
        path = self.root / SIGNATURES
        document = json.loads(path.read_text())
        for record in document['functions']:
            if record['name'] == name:
                change(record)
                break
        else:
            self.fail(f'{name} not in the signature export')
        path.write_text(json.dumps(document))

    def assertRejected(self, *fragments):
        with self.assertRaises(ExportError) as caught:
            build_document(self.root)
        for fragment in fragments:
            self.assertIn(fragment, str(caught.exception))
        return str(caught.exception)

    def test_copy_is_accepted_unchanged(self):
        self.assertEqual(check(self.root, self.root / OUTPUT), 13)

    # Unresolved symbols.
    def test_unknown_entry_symbol(self):
        self.edit(FC, 'entry: arm_fully_connected_s8\n', 'entry: arm_fully_connected_s9\n')
        self.assertRejected('fully_connected_s8.entry: arm_fully_connected_s9 is not in the signature export')

    def test_unknown_prepare_symbol(self):
        self.edit(DW, '- symbol: arm_depthwise_convolve_weight_sum\n', '- symbol: arm_depthwise_weight_sum\n', 6)
        self.assertRejected('prepare[0].symbol: arm_depthwise_weight_sum is not in the signature export')

    def test_unknown_static_predicate(self):
        self.edit(FC, 'predicate: arm_nn_fc_packed_s8_supported', 'predicate: arm_nn_fc_packed_s8_ok')
        self.assertRejected('arm_nn_fc_packed_s8_ok is not in the signature export or a static inline helper')

    def test_unknown_wrapper(self):
        self.edit(POOL, '    routed_by: direct\n    requires: []\n    buffers:\n      - name: ctx',
                  '    routed_by: {wrapper: arm_pool_wrapper_s8, route: "pool:max", requires: [], when: always}\n'
                  '    requires: []\n    buffers:\n      - name: ctx')
        self.assertRejected('max_pool_s8.routed_by.wrapper: arm_pool_wrapper_s8 is not in the signature export')

    # Buffer parameters: type and direction.
    def test_prepared_context_documented_in_out(self):
        self.edit_signature('arm_fully_connected_s8', lambda r: r['params'][0].update(direction='in,out'))
        self.assertRejected('a prepared context is only read by the entry, but ctx is [in,out]')

    def test_context_documented_out(self):
        self.edit_signature('arm_avgpool_s8', lambda r: r['params'][0].update(direction='out'))
        self.assertRejected('ctx is documented [out]')

    def test_buffer_on_a_non_context_output(self):
        self.edit(POOL, '      - name: ctx\n        param: ctx\n', '      - name: ctx\n        param: output_data\n')
        self.assertRejected("output_data is int8_t * [out]: a buffer must be a cmsis_nn_context * or a const pointer")

    def test_operand_buffer_written_by_the_entry(self):
        self.edit_signature('arm_fully_connected_per_channel_packed_s8',
                            lambda r: [p.update(direction='in,out') for p in r['params'] if p['name'] == 'packed_data'])
        self.assertRejected('packed_data is const int8_t * [in,out]')

    def test_buffer_param_not_on_entry(self):
        self.edit(POOL, '        param: ctx\n        size_query: none',
                  '        param: scratch\n        size_query: none')
        self.assertRejected("param 'scratch' is not a parameter of arm_max_pool_s8")

    def test_context_param_without_buffer(self):
        self.edit(POOL, '''    buffers:
      - name: ctx
        param: ctx
        size_query: none
        units: bytes
        align: 1
        lifetime: call
        optional: true
        note: No scratch; ctx->buf may be NULL.
''', '    buffers: []\n')
        self.assertRejected('context parameter ctx of arm_max_pool_s8 has no buffer')

    def test_operand_buffer_needs_a_format(self):
        self.edit(FC, '        format: fc_per_channel_packed_s8/1\n', '')
        self.assertRejected('a buffer that replaces an operand needs a format')

    # Size queries.
    def test_sizer_not_returning_int32(self):
        self.edit_signature('arm_avgpool_s8_get_buffer_size', lambda r: r.update(returns='int64_t'))
        self.assertRejected('arm_avgpool_s8_get_buffer_size returns int64_t, expected int32_t')

    def test_sizer_with_a_status_return(self):
        self.edit(POOL, 'symbol: arm_avgpool_s8_get_buffer_size', 'symbol: arm_max_pool_s8')
        self.assertRejected('arm_max_pool_s8 returns arm_cmsis_nn_status, expected int32_t')

    def test_sizer_argument_left_unbound(self):
        self.edit(POOL, 'args: {dim_dst_width: output_dims.w, ch_src: input_dims.c}', 'args: {ch_src: input_dims.c}')
        self.assertRejected("args must bind exactly the parameters of arm_avgpool_s8_get_buffer_size")

    def test_sizer_bound_to_a_buffer(self):
        self.edit(POOL, 'ch_src: input_dims.c}', 'ch_src: buffer:accumulators}')
        self.assertRejected('a buffer cannot be bound here')

    def test_size_query_none_on_a_prepared_buffer(self):
        self.edit(FC, '''        size_query:
          symbol: arm_fully_connected_s8_get_buffer_size
          args: {filter_dims: filter_dims}
        units: bytes
        align: 4
        lifetime: prepared
        optional: true
        note: >-
          Read only''', '''        size_query: none
        units: bytes
        align: 4
        lifetime: prepared
        optional: true
        note: >-
          Read only''')
        self.assertRejected('size_query none is only for an optional call-lifetime context')

    # Bindings.
    def test_unknown_struct_field(self):
        self.edit(POOL, 'output_dims.w', 'output_dims.width')
        self.assertRejected("binding 'output_dims.width': cmsis_nn_dims has no field width")

    def test_nested_struct_field_resolves(self):
        self.edit(POOL, 'output_dims.w', 'pool_params.stride.w')
        self.assertEqual(len(build_document(self.root)['implementations']), 13)

    def test_binding_type_mismatch(self):
        self.edit(FC, 'vector_cols: filter_dims.n', 'vector_cols: filter_data', 3)
        self.assertRejected('filter_data is const int8_t *, which const int32_t does not accept')

    def test_binding_to_an_unknown_operand(self):
        self.edit(FC, 'bias_data: bias_data\n', 'bias_data: bias\n', 3)
        self.assertRejected("binding 'bias' is not an operand of arm_fully_connected_s8")

    def test_null_bound_to_a_scalar(self):
        self.edit(FC, 'rhs_offset: fc_params.filter_offset', 'rhs_offset: null', 3)
        self.assertRejected('null bound to const int32_t, which is not a pointer')

    def test_written_operand(self):
        self.edit(FC, 'packed_data: buffer:packed_weights', 'packed_data: output_data')
        self.assertRejected('packed_data is written, so it must be bound to a buffer, not output_data')

    def test_operand_type_differs_from_entry(self):
        self.edit_signature('arm_fully_connected_per_channel_packed_s8',
                            lambda r: [p.update(type='const int16_t *')
                                       for p in r['params'] if p['name'] == 'input_data'])
        self.assertRejected('entry parameter input_data is const int16_t * but operand input_data is const int8_t *')

    # Preparation.
    def test_buffer_read_before_it_is_filled(self):
        text = (self.root / FC).read_text()
        first = text.index('      - symbol: arm_vector_sum_s8', text.index('id: fully_connected_per_channel_packed_s8'))
        second = text.index('      - symbol: arm_fully_connected_per_channel_packed_s8_pack')
        end = text.index('    declines:', second)
        (self.root / FC).write_text(text[:first] + text[second:end] + text[first:second] + text[end:])
        self.assertRejected('buffer kernel_sums is read before a step fills it')

    def test_prepared_buffer_never_filled(self):
        self.edit(POOL, '        lifetime: call\n        optional: true\n        note: >-',
                  '        lifetime: prepared\n        optional: true\n        note: >-')
        self.edit_signature('arm_avgpool_s8', lambda r: r['params'][0].update(direction='in'))
        self.assertRejected('buffer accumulators has lifetime prepared but no prepare step fills it')

    def test_call_scratch_filled_by_prepare(self):
        self.edit(FC, '''        lifetime: prepared
        optional: true
        note: >-
          Read only''', '''        lifetime: call
        optional: true
        note: >-
          Read only''')
        self.assertRejected('buffer kernel_sums is call scratch, so no prepare step may fill it')

    # Feature vocabulary.
    def test_unknown_feature(self):
        self.edit(FC, "requires: [mvei, '!autovectorize', '!single_rounding']", "requires: [mvei, helium]")
        self.assertRejected("unknown feature token(s) ['helium']")

    def test_contradictory_features(self):
        self.edit(FC, "requires: [mvei, '!autovectorize', '!single_rounding']", "requires: [mvei, '!mvei']")
        self.assertRejected("contradictory feature tokens ['mvei']")

    def test_unknown_guard(self):
        self.edit_signature('arm_max_pool_s8', lambda r: r.update(guards=['defined(ARM_MATH_FOO)']))
        self.assertRejected('declared under #if defined(ARM_MATH_FOO), which has no feature token')

    def test_guard_not_implied_by_requires(self):
        self.edit_signature('arm_max_pool_s8', lambda r: r.update(guards=['defined(ARM_MATH_MVEI)']))
        self.assertRejected('arm_max_pool_s8 is declared under #if defined(ARM_MATH_MVEI), so requires needs mvei')

    def test_float_dtype_without_float_api(self):
        layout = '    layout: {input_data: NHWC, output_data: NHWC}\n    entry: arm_max_pool_s8'
        self.edit(POOL, 'dtypes: {input_data: int8, output_data: int8}\n' + layout,
                  'dtypes: {input_data: float32, output_data: int8}\n' + layout)
        self.assertRejected('a float32 operand needs f32_api in requires')

    def test_float_guard_without_float_api(self):
        self.edit_signature('arm_max_pool_s8', lambda r: r.update(guards=['ARM_NN_ENABLE_F16']))
        self.assertRejected('so requires needs f16_api')

    def test_float_api_without_float_guard(self):
        self.edit(POOL, '    entry: arm_max_pool_s8\n    routed_by: direct\n    requires: []',
                  '    entry: arm_max_pool_s8\n    routed_by: direct\n    requires: [f32_api]')
        self.assertRejected('requires f32_api but no referenced symbol is declared under its ARM_NN_ENABLE_* guard')

    def test_unknown_varies_with(self):
        self.edit(DW, 'varies_with: [toolchain]', 'varies_with: [compiler]', 2)
        self.assertRejected("varies_with must be a list of distinct tokens from ['toolchain']")

    def test_build_decline_without_requirements(self):
        self.edit(POOL, '    declines: []\n    since: 4.0.0\n    numerics:\n      reference: hct_ref:max_pool_s8',
                  '    declines: [{cause: build, when: never}]\n    since: 4.0.0\n    numerics:\n'
                  '      reference: hct_ref:max_pool_s8')
        self.assertRejected('a build decline needs the features it lacks in requires')

    def test_wrapper_only_route_needs_route_features(self):
        self.edit(DW, '    requires: [mvei]\n    varies_with: [toolchain]',
                  '    requires: []\n    varies_with: [toolchain]')
        self.assertRejected("an entry reachable only through its route must require the route's features ['mvei']")

    # Identity and structure.
    def test_duplicate_id(self):
        self.edit(FC, '  - id: fully_connected_per_channel_s8\n', '  - id: fully_connected_s8\n')
        self.assertRejected('duplicate id (also in Tests/KernelContracts/implementations/fully_connected_s8.yaml)')

    def test_duplicate_route(self):
        self.edit(FC, 'route: fc_wrapper_s8:per_channel', 'route: fc_wrapper_s8:per_tensor')
        self.assertRejected('route fc_wrapper_s8:per_tensor of arm_fully_connected_wrapper_s8 is also claimed by')

    def test_route_prefix_shared_across_wrappers(self):
        self.edit(DW, 'route: dw_opt_s8:planar', 'route: dw_wrapper_s8:planar')
        self.assertRejected('route prefix dw_wrapper_s8: already belongs to arm_depthwise_conv_wrapper_s8')

    def test_routed_entry_parameter_missing_on_wrapper(self):
        self.edit_signature('arm_fully_connected_s8', lambda r: r['params'][1].update(name='params'))
        self.assertRejected("arm_fully_connected_s8 parameters ['params'] are not parameters of "
                            'arm_fully_connected_wrapper_s8')

    def test_unknown_record_key(self):
        self.edit(POOL, '  - id: max_pool_s8\n', '  - id: max_pool_s8\n    priority: 1\n')
        self.assertRejected("max_pool_s8: unknown key(s) ['priority']")

    def test_missing_record_key(self):
        self.edit(POOL, '    since: 4.0.0\n    numerics:\n      reference: hct_ref:max_pool_s8',
                  '    numerics:\n      reference: hct_ref:max_pool_s8')
        self.assertRejected("max_pool_s8: missing key(s) ['since']")

    def test_duplicate_yaml_key(self):
        self.edit(POOL, '  - id: max_pool_s8\n', '  - id: max_pool_s8\n    id: max_pool_s8_again\n')
        self.assertRejected("duplicate key 'id'")

    def test_unhashable_value_is_a_diagnostic(self):
        self.edit(POOL, '    op: max_pool\n', '    op: [max_pool]\n')
        self.assertRejected("op ['max_pool'] is not one of")

    def test_family_must_match_file(self):
        self.edit(POOL, 'family: pooling_s8', 'family: pool_s8')
        self.assertRejected("family 'pool_s8' must equal the file stem 'pooling_s8'")

    def test_stray_file_in_sources(self):
        (self.root / SOURCES / 'extra.yml').write_text('family: extra\n')
        self.assertRejected('only <family>.yaml files belong here')

    def test_empty_sources(self):
        for path in (self.root / SOURCES).iterdir():
            path.unlink()
        self.assertRejected('no implementation records')

    def test_non_ascii_source(self):
        self.edit(POOL, 'No scratch; ctx->buf may be NULL.', 'No scratch — ctx->buf may be NULL.')
        self.assertRejected('non-ASCII text')

    # Boundaries.
    def test_alignment_boundaries(self):
        for align, accepted in ((0, False), (1, True), (3, False), (64, True), (128, False), ('true', False)):
            with self.subTest(align=align):
                self.tearDown()
                self.setUp()
                self.edit(POOL, '        align: 1\n', f'        align: {align}\n')
                if accepted:
                    self.assertEqual(len(build_document(self.root)['implementations']), 13)
                else:
                    self.assertRejected('must be a power of two from 1 to 64')

    def test_since_boundaries(self):
        version = build_document(self.root)['library_version']
        major, minor, patch = (int(part) for part in version.split('.'))
        self.edit(POOL, '    since: 4.0.0\n    numerics:\n      reference: hct_ref:max_pool_s8',
                  f'    since: {version}\n    numerics:\n      reference: hct_ref:max_pool_s8')
        build_document(self.root)
        self.edit(POOL, f'since: {version}\n', f'since: {major}.{minor}.{patch + 1}\n')
        self.assertRejected(f'since {major}.{minor}.{patch + 1} is after the library version {version}')

    def test_since_must_be_a_version_string(self):
        self.edit(POOL, '    since: 4.0.0\n    numerics:\n      reference: hct_ref:max_pool_s8',
                  '    since: 4.0\n    numerics:\n      reference: hct_ref:max_pool_s8')
        self.assertRejected('must be a MAJOR.MINOR.PATCH version string')

    # Envelope inputs and freshness.
    def test_planner_abi_is_read_from_the_planner_header(self):
        (self.root / 'Planner/include').mkdir(parents=True, exist_ok=True)
        (self.root / 'Planner/include/arm_nn_plan.h').write_text('#define ARM_NN_PLAN_ABI_VERSION (3)\n')
        self.assertEqual(build_document(self.root)['planner_abi'], 3)
        (self.root / 'Planner/include/arm_nn_plan.h').write_text('/* no version yet */\n')
        self.assertRejected('defines no ARM_NN_PLAN_ABI_VERSION')

    def test_corrupt_signature_export(self):
        (self.root / SIGNATURES).write_text('{"schema": ')
        self.assertRejected('cannot be read as JSON')

    def test_stale_after_a_source_edit(self):
        self.edit(POOL, 'No scratch; ctx->buf may be NULL.', 'No scratch; ctx->buf may be NULL here.')
        with self.assertRaisesRegex(ExportError, r'stale; run .*\n  ~ max_pool_s8'):
            check(self.root, self.root / OUTPUT)

    def test_stale_after_a_signature_change(self):
        path = self.root / SIGNATURES
        path.write_text(path.read_text().replace('"line": 1', '"line": 2', 1))
        with self.assertRaisesRegex(ExportError, r'stale; .*\n  signature_sha256: '):
            check(self.root, self.root / OUTPUT)

    def test_stale_after_a_version_bump(self):
        self.edit('Include/arm_nn_types.h', '#define NS_CMSIS_NN_VERSION_PATCH (0)',
                  '#define NS_CMSIS_NN_VERSION_PATCH (9)')
        with self.assertRaisesRegex(ExportError, r"library_version: '(\d+\.){2}0' -> '(\d+\.){2}9'"):
            check(self.root, self.root / OUTPUT)

    def test_non_canonical_json(self):
        path = self.root / OUTPUT
        path.write_text(json.dumps(json.loads(path.read_text()), indent=4) + '\n')
        with self.assertRaisesRegex(ExportError, 'not in canonical form'):
            check(self.root, path)

    def test_missing_manifest(self):
        (self.root / OUTPUT).unlink()
        with self.assertRaisesRegex(ExportError, 'missing; run'):
            check(self.root, self.root / OUTPUT)

    def test_failed_export_writes_nothing(self):
        before = (self.root / OUTPUT).read_text()
        self.edit(FC, 'entry: arm_fully_connected_s8\n', 'entry: arm_fully_connected_s9\n')
        result = run('export', '--root', str(self.root))
        self.assertEqual(result.returncode, 1)
        self.assertIn('arm_fully_connected_s9 is not in the signature export', result.stderr)
        self.assertEqual((self.root / OUTPUT).read_text(), before)

    def test_export_then_check_round_trip(self):
        self.edit(POOL, 'No scratch; ctx->buf may be NULL.', 'No scratch; ctx->buf may be NULL here.')
        self.assertEqual(run('export', '--root', str(self.root)).returncode, 0)
        self.assertEqual(run('check', '--root', str(self.root)).returncode, 0)
        self.assertEqual((self.root / OUTPUT).read_text(), render(build_document(self.root)))

    def test_every_problem_is_reported_at_once(self):
        self.edit(FC, 'entry: arm_fully_connected_s8\n', 'entry: arm_fully_connected_s9\n')
        self.edit(POOL, '    op: max_pool\n', '    op: min_pool\n')
        message = self.assertRejected('arm_fully_connected_s9', "op 'min_pool'")
        self.assertRegex(message, r'^2 problem\(s\)')


if __name__ == '__main__':
    unittest.main()
