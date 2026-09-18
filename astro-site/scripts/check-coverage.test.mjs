// SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, copyFileSync, rmSync } from 'node:fs';
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
