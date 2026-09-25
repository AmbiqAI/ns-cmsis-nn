// SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../src/content/docs/', import.meta.url));
const markers = /draft copy|pending owner approval|diagram proposal|this page reserves a route|integration status/i;
const failures = [];
function visit(dir) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const file = path.join(dir, entry.name);
    if (entry.isDirectory()) visit(file);
    else if (/\.mdx?$/.test(entry.name)) {
      const source = readFileSync(file, 'utf8');
      if (/^draft:\s*true\s*$/m.test(source)) failures.push(`${path.relative(root, file)}: draft content belongs outside the published collection`);
      source.split('\n').forEach((line, index) => {
        if (markers.test(line)) failures.push(`${path.relative(root, file)}:${index + 1}: internal editorial marker`);
      });
    }
  }
}
visit(root);
if (failures.length) {
  console.error(failures.join('\n'));
  process.exitCode = 1;
} else {
  console.log('Public content check passed.');
}
