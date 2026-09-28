#!/usr/bin/env node
// SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
/*
 * Rewrites the Markdown rendition of each C API page from the model, then
 * rebuilds the site agent bundle from the renditions.
 *
 * The site derives `<route>/index.md` from the MDX source, and an API page
 * holds its parameter tables, enum members and signatures in component props
 * that a source-based rendition cannot recover (AmbiqAI/ns-cmsis-nn#572). The
 * model can, so the API pages are re-rendered from it here.
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  RENDER_DEFAULTS,
  buildIndex,
  flatten,
  renderModuleMarkdown,
  slugSegment,
} from '../node_modules/@ambiqai/helia-ui/scripts/lib/reference-render.mjs';

const dist = fileURLToPath(new URL('../dist/', import.meta.url));
const model = JSON.parse(
  fs.readFileSync(path.join(dist, 'reference/api/reference.json'), 'utf8'),
);
const catalog = JSON.parse(
  fs.readFileSync(path.join(dist, 'content-index.json'), 'utf8'),
);
const options = {
  ...RENDER_DEFAULTS,
  base: catalog.base,
  site: catalog.site,
  routePrefix: 'reference/api',
};
const index = buildIndex(model, options);

const modules = flatten(model);
for (const module of modules) {
  const route = module.path.split('.').map(slugSegment).join('/');
  const target = path.join(dist, options.routePrefix, route, 'index.md');
  fs.writeFileSync(target, renderModuleMarkdown(module, { index, options }));
}

const kernelDir = path.join(dist, 'reference/kernel-index');
const kernelHtml = fs.readFileSync(path.join(kernelDir, 'index.html'), 'utf8');
const sidecar = kernelHtml.match(/<script\b[^>]*data-helia-rendition="reference-browser"[^>]*>([\s\S]*?)<\/script>/);
if (!sidecar) throw new Error('Kernel index is missing its reference-browser rendition');
const kernelRows = sidecar[1].replace(
  /<(\\+)(\/|!--)/g,
  (_, held, opener) => `<${held.slice(1)}${opener}`,
);
const kernelMarkdownPath = path.join(kernelDir, 'index.md');
const kernelIntro = fs.readFileSync(kernelMarkdownPath, 'utf8').split('\n## Functions\n')[0];
fs.writeFileSync(kernelMarkdownPath, `${kernelIntro.trimEnd()}\n\n## Functions\n\n${kernelRows.trim()}\n`);

const bundle = `${catalog.routes
  .map((page) => {
    if (!page.markdown.startsWith(catalog.base)) {
      throw new Error(`Unexpected Markdown route: ${page.markdown}`);
    }
    const markdown = fs.readFileSync(
      path.join(dist, page.markdown.slice(catalog.base.length)),
      'utf8',
    );
    return `<!-- ${page.url} -->\n\n${markdown.trimEnd()}`;
  })
  .join('\n\n---\n\n')}\n`;
fs.writeFileSync(path.join(dist, 'llms-full.txt'), bundle);

console.log(
  `Published model-derived Markdown for ${modules.length} API pages and rebuilt the agent bundle.`,
);
