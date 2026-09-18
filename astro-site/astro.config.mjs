// SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
// @ts-check

import fs from 'node:fs';
import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import tailwindcss from '@tailwindcss/vite';
import starlight from '@astrojs/starlight';
import { heliaStarlight } from '@ambiqai/helia-ui/starlight';
import { satteri } from '@astrojs/markdown-satteri';
import markdownCallouts from './scripts/markdown-callouts.mjs';

/* `site` plus `base` reproduce the Pages URL the Sphinx site already serves,
   so a canonical link and an Open Graph card resolve to the live origin. */
const site = 'https://ambiqai.github.io';
const base = '/ns-cmsis-nn';
const basePath = `${base}/`;

/* The routes the Sphinx site serves today that this structure no longer has,
   plus the three section roots whose sections are a page list rather than a
   landing page. Kept in a data file because the cut-over needs the same list
   outside the build (AmbiqAI/ns-cmsis-nn#524). */
const redirects = JSON.parse(
  fs.readFileSync(new URL('./redirects.json', import.meta.url), 'utf8'),
);

/* Written by scripts/build-reference.mjs next to the generated pages: the
   module pages are not in the checkout to declare by hand. Checked in so an
   `--if-missing` run, which skips generation, still finds it. */
/**
 * @typedef {{ label: string, slug: string }} NavEntry
 * @typedef {{ groups: { label: string, slug: string, modules: NavEntry[] }[],
 *             other: NavEntry[] }} ReferenceNav
 */
/** @type {ReferenceNav} */
const referenceNav = JSON.parse(
  fs.readFileSync(
    new URL('./src/data/reference-nav.json', import.meta.url),
    'utf8',
  ),
);

/* The Reference section's pages: one entry per operator family, its module
   pages one level under it, rather than a wrapper group around the generated
   tree. The generated index keeps the one-page view. */
const referenceItems = [
  { label: 'Overview', slug: 'reference' },
  { label: 'All functions', slug: 'reference/api' },
  { label: 'Kernel index', slug: 'reference/kernel-index' },
  ...referenceNav.groups.map((group) => ({
    label: group.label,
    collapsed: true,
    items: [
      { label: 'Overview', slug: group.slug },
      ...group.modules.map((module) => ({
        label: module.label,
        slug: module.slug,
      })),
    ],
  })),
  {
    label: 'Headers and types',
    collapsed: true,
    items: referenceNav.other.map((module) => ({
      label: module.label,
      slug: module.slug,
    })),
  },
];

export default defineConfig({
  site,
  base,
  redirects: Object.fromEntries(Object.entries(redirects).filter(([route]) => !route.endsWith('.html'))),
  markdown: { processor: satteri({ mdastPlugins: [markdownCallouts] }) },
  integrations: [
    starlight({
      title: 'heliaCORE',
      description: 'Optimized AI kernels for Ambiq Silicon',
      /* The Tailwind entry the package's React lane needs: the Reference
         section's kernel index is the site's one island, and its shadcn
         components are utility classes this tree compiles. */
      customCss: ['./src/styles/tailwind.css'],
      plugins: [
        heliaStarlight({
          accent: 'helia-core',
          sidebar: 'always',
          /* The package header draws the name as text, so the site carries no
             wordmark asset in the bar. Its links are not written here: the
             bar is derived from `sections` below, so the top bar and the left
             sidebar are one definition. */
          header: {
            title: 'heliaCORE',
            hub: { label: 'HELIA', href: 'https://ambiqai.github.io/helia-developer-hub/' },
          },
          sections: [
            { label: 'Home', href: basePath, sidebar: false },
            {
              label: 'Getting started', href: `${basePath}getting-started/`,
              sidebar: [
                { label: 'Overview', slug: 'getting-started' },
                { label: 'Requirements', slug: 'getting-started/requirements' },
                { label: 'Integration', collapsed: false, items: [
                  { label: 'CMake', slug: 'getting-started/cmake' },
                  { label: 'CMSIS-Pack', slug: 'getting-started/cmsis-pack' },
                  { label: 'Zephyr', slug: 'getting-started/zephyr' },
                  { label: 'neuralSPOT-X', slug: 'getting-started/neuralspot-x' },
                ] },
                { label: 'First kernel', slug: 'getting-started/first-kernel' },
              ],
            },
            {
              label: 'User guide', href: `${basePath}guide/`,
              sidebar: [
                { label: 'Overview', slug: 'guide' },
                { label: 'Using kernels', collapsed: false, items: [
                  { label: 'Operator coverage', slug: 'guide/coverage/operator-coverage' },
                  { label: 'Data types', slug: 'guide/architecture/data-types' },
                  { label: 'Quantization', slug: 'guide/using-kernels/quantization' },
                  { label: 'Calling kernels', slug: 'guide/using-kernels/calling-kernels' },
                  { label: 'Memory', slug: 'guide/using-kernels/memory' },
                ] },
                { label: 'Configuration', items: [
                  { label: 'Build options', slug: 'guide/architecture/build-path-selection' },
                  { label: 'Targets', slug: 'guide/architecture/cortex-m-targets' },
                  { label: 'Toolchains', slug: 'guide/architecture/toolchains' },
                ] },
                { label: 'Performance', items: [
                  { label: 'Acceleration', slug: 'guide/architecture/acceleration-paths' },
                  { label: 'Benchmarks', slug: 'guide/performance/kernel-benchmarks' },
                  { label: 'Measurement', slug: 'guide/performance/methodology' },
                  { label: 'Validation', slug: 'guide/performance/validation' },
                ] },
              ],
            },
            { label: 'API reference', href: `${basePath}reference/`, sidebar: referenceItems },
          ],
          /* Stated rather than left to default so the four artifacts this site
             owes a crawler and an agent are visible in review. The JSON-LD
             publisher is Ambiq at https://www.ambiq.com, fixed by the package. */
          discoverability: {
            ogImage: true,
            jsonLd: true,
            markdown: true,
            llms: true,
          },
          footer: {
            links: [
              { label: 'Getting started', href: `${basePath}getting-started/` },
              { label: 'User guide', href: `${basePath}guide/` },
              { label: 'API reference', href: `${basePath}reference/` },
              { label: 'Contributing', href: `${basePath}contributing/` },
              { label: 'About and licenses', href: `${basePath}about/` },
              { label: 'GitHub', href: 'https://github.com/AmbiqAI/ns-cmsis-nn' },
            ],
            /* The copyright line from the Sphinx conf.py, carried verbatim so
               the Arm attribution stays on every page. */
            tagline: 'Ambiq Micro, Inc. Built on Arm CMSIS-NN.',
            logo: 'ambiq',
          },
        }),
      ],
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/AmbiqAI/ns-cmsis-nn',
        },
      ],
      favicon: '/helia-core-icon-color.svg',
    }),
    /* For the kernel index island, and nothing else on the site. */
    react(),
  ],
  vite: {
    plugins: [tailwindcss()],
  },
});
