// SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
// @ts-check
/*
 * The Astro Starlight site, built alongside the Sphinx site in docs/ until the
 * cut-over (AmbiqAI/ns-cmsis-nn#521). Pages still serves the Sphinx build.
 */
import fs from 'node:fs';
import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import tailwindcss from '@tailwindcss/vite';
import starlight from '@astrojs/starlight';
import { heliaStarlight } from '@ambiqai/helia-ui/starlight';

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
  { label: 'Generation notes', slug: 'reference/doxygen' },
];

export default defineConfig({
  site,
  base,
  redirects,
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
          /* The left navigation is the site's table of contents, and the
             landing page is the first place a reader needs it. */
          sidebar: 'always',
          /* The package header draws the name as text, so the site carries no
             wordmark asset in the bar. Its links are not written here: the
             bar is derived from `sections` below, so the top bar and the left
             sidebar are one definition. */
          header: {
            title: 'heliaCORE',
          },
          /* One entry per section of astro-site/SITE-PLAN.md, in the plan's
             order. Each carries the pages of that section; the plugin puts the
             list in the bar and scopes the sidebar to the section the reader
             is in, under the section's name. The three section hrefs that have
             no landing page of their own are redirects, declared in
             redirects.json. */
          sections: [
            {
              label: 'Home',
              href: basePath,
              sidebar: [{ label: 'Overview', slug: '' }],
            },
            {
              label: 'Getting started',
              href: `${basePath}getting-started/`,
              sidebar: [
                { label: 'Overview', slug: 'getting-started' },
                { label: 'CMake (find_package)', slug: 'getting-started/cmake' },
                { label: 'CMSIS-Pack', slug: 'getting-started/cmsis-pack' },
                { label: 'Zephyr module', slug: 'getting-started/zephyr' },
                { label: 'neuralSPOT-X', slug: 'getting-started/neuralspot-x' },
                { label: 'Toolchains', slug: 'getting-started/toolchains' },
              ],
            },
            {
              label: 'Architecture',
              href: `${basePath}architecture/`,
              sidebar: [
                {
                  label: 'Acceleration paths',
                  slug: 'architecture/acceleration-paths',
                },
                {
                  label: 'Cortex-M targets',
                  slug: 'architecture/cortex-m-targets',
                },
                {
                  label: 'Data types and quantization',
                  slug: 'architecture/data-types',
                },
                {
                  label: 'How the build selects a path',
                  slug: 'architecture/build-path-selection',
                },
              ],
            },
            {
              label: 'Coverage',
              href: `${basePath}coverage/`,
              sidebar: [
                { label: 'Operator coverage', slug: 'coverage/operator-coverage' },
                {
                  label: 'Data types by family',
                  slug: 'coverage/data-types-by-family',
                },
                {
                  label: 'Compared with CMSIS-NN',
                  slug: 'coverage/compared-with-cmsis-nn',
                },
              ],
            },
            {
              /* Cross-SoC comparison is built but carries `sidebar.hidden`:
                 the page reserves the route and says what will be published
                 there, and stays out of the navigation until it has values. */
              label: 'Performance',
              href: `${basePath}performance/`,
              sidebar: [
                {
                  label: 'Kernel benchmarks',
                  slug: 'performance/kernel-benchmarks',
                },
                { label: 'Methodology', slug: 'performance/methodology' },
              ],
            },
            {
              label: 'Reference',
              href: `${basePath}reference/`,
              sidebar: referenceItems,
            },
            {
              label: 'Contributing',
              href: `${basePath}contributing/`,
              sidebar: [
                { label: 'Overview', slug: 'contributing' },
                {
                  label: 'Testing and verification',
                  slug: 'contributing/verification',
                },
                { label: 'CI matrix', slug: 'contributing/ci-matrix' },
                {
                  label: 'Publish once after local review',
                  slug: 'contributing/pr-publication',
                },
                { label: 'Releases and versioning', slug: 'contributing/releases' },
              ],
            },
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
              { label: 'Overview', href: basePath },
              { label: 'Getting started', href: `${basePath}getting-started/` },
              { label: 'Architecture', href: `${basePath}architecture/` },
              { label: 'Coverage', href: `${basePath}coverage/` },
              { label: 'Performance', href: `${basePath}performance/` },
              { label: 'Reference', href: `${basePath}reference/` },
              { label: 'Contributing', href: `${basePath}contributing/` },
              { label: 'About', href: `${basePath}about/` },
              {
                label: 'GitHub',
                href: 'https://github.com/AmbiqAI/ns-cmsis-nn',
              },
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
