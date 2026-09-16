// SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
// @ts-check
/*
 * The Astro Starlight site, built alongside the Sphinx site in docs/ until the
 * cut-over (AmbiqAI/ns-cmsis-nn#521). Pages still serves the Sphinx build.
 */
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import { heliaStarlight } from '@ambiqai/helia-ui/starlight';

/* `site` plus `base` reproduce the Pages URL the Sphinx site already serves,
   so a canonical link and an Open Graph card resolve to the live origin. */
const site = 'https://ambiqai.github.io';
const base = '/ns-cmsis-nn';
const basePath = `${base}/`;

export default defineConfig({
  site,
  base,
  integrations: [
    starlight({
      title: 'heliaCORE',
      description: 'Optimized AI kernels for Ambiq Silicon',
      /* The asset names describe the background the wordmark sits on rather
         than the theme that shows it, so the pairing is inverted here: the
         dark wordmark is the one a light page needs. */
      logo: {
        light: './src/assets/helia-core-logo-dark.png',
        dark: './src/assets/helia-core-logo-light.png',
        replacesTitle: true,
      },
      plugins: [
        heliaStarlight({
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
              { label: 'Getting Started', href: `${basePath}getting-started/` },
              { label: 'Guides', href: `${basePath}guides/` },
              { label: 'Reference', href: `${basePath}reference/` },
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
      /* The four sections of the Sphinx toctree. Each directory holds a
         placeholder until the page conversion lands
         (AmbiqAI/ns-cmsis-nn#518). */
      sidebar: [
        { label: 'Overview', slug: '' },
        { label: 'Why heliaCORE', slug: 'why' },
        {
          label: 'Getting Started',
          collapsed: false,
          items: [{ autogenerate: { directory: 'getting-started' } }],
        },
        {
          label: 'Guides',
          collapsed: false,
          items: [{ autogenerate: { directory: 'guides' } }],
        },
        {
          label: 'Reference',
          collapsed: false,
          items: [{ autogenerate: { directory: 'reference' } }],
        },
        { label: 'About', slug: 'about' },
      ],
    }),
  ],
});
