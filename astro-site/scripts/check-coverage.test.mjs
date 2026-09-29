// SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, copyFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { test } from 'node:test';

test('coverage rejects missing and same-count replacement symbols', () => {
  const root = mkdtempSync(join(tmpdir(), 'heliacore-coverage-'));
  try {
    for (const dir of ['Include', 'astro-site/scripts', 'astro-site/src/data', 'astro-site/public/reference/api']) {
      mkdirSync(join(root, dir), { recursive: true });
    }
    const script = join(root, 'astro-site/scripts/check-coverage.mjs');
    copyFileSync(new URL('./check-coverage.mjs', import.meta.url), script);
    writeFileSync(join(root, 'Include/arm_nnfunctions.h'), 'void arm_first_s8(void);\nvoid arm_second_s8(void);\n');
    writeFileSync(join(root, 'astro-site/src/data/coverage.json'), JSON.stringify({
      headers: ['arm_nnfunctions.h'], functionsByHeader: { 'arm_nnfunctions.h': 2 },
      totals: { functions: 2, byDtype: { s8: 2 } }, families: [{ functions: 2 }], dtypes: ['s8'],
    }));
    const run = (names) => {
      writeFileSync(join(root, 'astro-site/public/reference/api/reference.json'), JSON.stringify({
        modules: [{ symbols: names.map((name) => ({ name, kind: 'function', source: { path: 'Include/arm_nnfunctions.h' } })) }],
      }));
      return spawnSync(process.execPath, [script], { encoding: 'utf8' });
    };
    assert.equal(run(['arm_first_s8', 'arm_second_s8']).status, 0);
    for (const names of [['arm_first_s8'], ['arm_first_s8', 'arm_replacement_s8']]) {
      const result = run(names);
      assert.notEqual(result.status, 0);
      assert.match(result.stderr, /API model declarations differ from arm_nnfunctions.h/);
    }
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('public FP16 aliases and unsigned APIs appear in coverage', () => {
  const root = mkdtempSync(join(tmpdir(), 'heliacore-dtypes-'));
  try {
    for (const dir of ['Include', 'cmake', 'astro-site/scripts', 'astro-site/src/data', 'astro-site/public/reference/api']) {
      mkdirSync(join(root, dir), { recursive: true });
    }
    for (const name of ['build-coverage.mjs', 'check-coverage.mjs']) {
      copyFileSync(new URL(`./${name}`, import.meta.url), join(root, 'astro-site/scripts', name));
    }
    const names = ['arm_elementwise_add_fp16', 'arm_avg_pool_f16', 'arm_softmax_u8'];
    writeFileSync(join(root, 'Include/arm_nnfunctions.h'), names.map(name => `void ${name}(void);`).join('\n'));
    writeFileSync(join(root, 'cmake/ns_cmsis_nn.cmake'), 'set(_NS_CMSIS_NN_GROUPS Activation)');
    writeFileSync(join(root, 'astro-site/reference.config.json'), JSON.stringify({
      base: '/', groups: [{ id: 'test', label: 'Test', patterns: ['^arm_'] }],
    }));
    writeFileSync(join(root, 'astro-site/public/reference/api/reference.json'), JSON.stringify({
      modules: [{ symbols: names.map(name => ({ name, kind: 'function', source: { path: 'Include/arm_nnfunctions.h' } })) }],
    }));
    const run = name => spawnSync(process.execPath, [join(root, 'astro-site/scripts', name)], { encoding: 'utf8' });
    const generated = run('build-coverage.mjs');
    assert.equal(generated.status, 0, generated.stderr);
    const file = join(root, 'astro-site/src/data/coverage.json');
    const coverage = JSON.parse(readFileSync(file, 'utf8'));
    assert.equal(coverage.totals.byDtype.f16, 2);
    assert.equal(coverage.totals.byDtype.u8, 1);
    assert.equal(coverage.families[0].counts.f16, 2);
    assert.equal(coverage.families[0].counts.u8, 1);
    assert.equal(coverage.families[0].untyped, 0);
    assert.equal(run('check-coverage.mjs').status, 0);
    coverage.totals.byDtype.f16 = 1;
    writeFileSync(file, JSON.stringify(coverage));
    assert.notEqual(run('check-coverage.mjs').status, 0);
    coverage.totals.byDtype.f16 = 2;
    coverage.dtypes = coverage.dtypes.filter(dtype => dtype !== 'u8');
    writeFileSync(file, JSON.stringify(coverage));
    assert.match(run('check-coverage.mjs').stderr, /Missing API data type: u8/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
