#!/usr/bin/env node
// SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';

const read = (relative) => fs.readFileSync(new URL(relative, import.meta.url), 'utf8');
const coverage = JSON.parse(read('../src/data/coverage.json'));
const model = JSON.parse(read('../public/reference/api/reference.json'));
const extracted = new Map();
function visit(module) {
  for (const symbol of module.symbols ?? []) {
    if (symbol.kind !== 'function') continue;
    const header = symbol.source?.path?.replace(/^Include\//, '');
    if (!extracted.has(header)) extracted.set(header, new Set());
    extracted.get(header).add(symbol.name);
  }
  for (const child of module.submodules ?? []) visit(child);
}
for (const module of model.modules ?? []) visit(module);
const names = new Set();
for (const header of coverage.headers) {
  // This independent header scan guards against a thinner extracted API model.
  const text = read(`../../Include/${header}`)
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/\/\/[^\n]*/g, '')
    .replace(/^\s*#.*$/gm, '');
  const declared = new Set([...text.matchAll(/\b(arm_[a-z0-9_]+)\s*\([^;{}]*\)\s*;/g)]
    .map((match) => match[1]));
  assert.equal(coverage.functionsByHeader[header], declared.size, header);
  assert.deepEqual([...(extracted.get(header) ?? [])].sort(), [...declared].sort(),
    `API model declarations differ from ${header}`);
  for (const name of declared) names.add(name);
}
assert.equal(coverage.totals.functions, names.size);
assert.equal(coverage.families.reduce((sum, family) => sum + family.functions, 0), names.size);
const expectedByDtype = new Map();
for (const name of names) {
  const tags = new Set([...name.matchAll(/(?:^|_)(s4|s8|s16|s32|s64|u8|q7|q15|f16|fp16|f32)(?=_|$)/g)]
    .map((match) => match[1] === 'fp16' ? 'f16' : match[1]));
  for (const tag of tags) expectedByDtype.set(tag, (expectedByDtype.get(tag) ?? 0) + 1);
}
for (const tag of expectedByDtype.keys()) {
  assert.ok(coverage.dtypes.includes(tag), `Missing API data type: ${tag}`);
}
for (const dtype of coverage.dtypes) {
  assert.equal(coverage.totals.byDtype[dtype], expectedByDtype.get(dtype) ?? 0, dtype);
}
console.log(`Coverage matches ${names.size} declarations, including multiline prototypes (${fileURLToPath(new URL('../src/data/coverage.json', import.meta.url))}).`);
