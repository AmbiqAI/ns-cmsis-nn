// SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
import { readFileSync, mkdirSync, writeFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../dist/', import.meta.url));
const redirects = JSON.parse(readFileSync(new URL('../redirects.json', import.meta.url), 'utf8'));
const escape = (value) => value.replaceAll('&', '&amp;').replaceAll('"', '&quot;').replaceAll('<', '&lt;');

// Astro's directory output would turn a Sphinx .html URL into a directory.
for (const [route, target] of Object.entries(redirects)) {
  if (!route.endsWith('.html')) continue;
  const file = path.resolve(root, `.${route}`);
  if (!file.startsWith(root) || !target.startsWith('/ns-cmsis-nn/')) {
    throw new Error(`Invalid legacy redirect: ${route} -> ${target}`);
  }
  if (existsSync(file)) throw new Error(`Legacy redirect would overwrite ${file}`);
  mkdirSync(path.dirname(file), { recursive: true });
  const href = escape(target);
  writeFileSync(file, `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="robots" content="noindex"><meta http-equiv="refresh" content="0;url=${href}"><title>Page moved</title></head><body><p>This page has moved. <a href="${href}">Continue to the documentation</a>.</p></body></html>\n`);
}
