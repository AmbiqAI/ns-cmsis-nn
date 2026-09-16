# HANDOFF: heliaCORE docs migration

For the session that owns this work. A different session owns the HELIA Dev
Hub and the `@ambiqai/helia-ui` package; see "Working with helia-ui" below,
because that boundary matters.

## Goal

Replace the Sphinx site at https://ambiqai.github.io/ns-cmsis-nn/ with an Astro
Starlight site built on `@ambiqai/helia-ui`. This is the first product-site
migration, so it also proves the shared package against real content. heliaRT
is the intended second migration.

## Where the work lives

- Repo `AmbiqAI/ns-cmsis-nn`, branch `docs/helia-ui-migration`, worktree
  `.claude/worktrees/helia-ui-migration`.
- Draft PR #522. Parent issue #516 with sub-issues #517 to #521 and #524.
- Site directory is `astro-site/`, not `site/`: the Sphinx build script deletes
  `site/` on every run.
- The Sphinx docs under `docs/` still build and still serve GitHub Pages. They
  stay until cut-over (#521).

## State

Built and verified: seven sections, 86 pages, zero broken links out of roughly
7,800, `astro check` clean. A generated C reference covering 585 functions
across 17 headers in eight operator families, plus a searchable kernel index.
Benchmark charts on the package chart lane. Redirects for 13 moved routes.
Pinned to helia-ui `v0.1.0-alpha.10`; `v0.1.0-alpha.11` is now published and
worth adopting (see next steps).

## Decisions already taken

- **Structure**, recorded in `astro-site/SITE-PLAN.md`: Home, Getting started,
  Architecture, Coverage, Performance, Reference, Contributing. The Guides
  umbrella is gone and "Why heliaCORE" is folded into Home and Architecture.
- **Shell**: top-bar sections with a section-scoped sidebar, and no sidebar on
  Home. A menu button opens the section list on narrow screens.
- **helia-ui is a base, not the whole.** Site-specific components are fine
  under `astro-site/src/components` provided they build on the package tokens.
  Promote something to the package only when a second site needs it.
- **Benchmark values stay as they are** until the owner captures new ones. Do
  not adjust the numbers or the clock.
- Landing defects the owner ruled on are fixed: no empty icon discs, no "rest
  of the site" section, figures rounded to 12x, 8x, 3x.

## Blocked on the owner

1. **The six landing headings** in `astro-site/LANDING-PAGE-CONTENT.md`. This
   gates the landing rebuild, the last content piece before cut-over.
2. **Three claims that cannot be published unverified**: the MVE per-cycle MAC
   figures (absent from the repo, needs an Arm or Apollo510 source of record),
   "40+ field models" (a deployment claim carried from the old site), and any
   numeric comparison against upstream CMSIS-NN (needs a coverage matrix).
3. **Eight pages carrying draft copy**, each opening with a "Draft copy,
   pending owner approval" aside: the four Architecture pages, two Coverage
   pages, the hidden cross-SoC placeholder, and the kernel index intro.
4. **Cut-over** (#521) needs sign-off on the built site.

## Next steps, in order

1. Bump the pin to `v0.1.0-alpha.11`, adopt `header.hub` and
   `sections[].sidebar: false` for Home, and delete the stopgap
   `astro-site/src/components/LinkCard.astro`, which exists only to suppress
   an empty disc the package now fixes.
2. Rebuild the landing once the owner settles the headings.
3. Walk the owner through the eight draft-copy pages.
4. Cut over: apply the redirect map, switch the Pages job, remove Sphinx.
5. Optional follow-up: the kernel SSoT manifest (#525), which the kernel index
   is already designed to consume.

## Working with helia-ui

Another session owns the package. Do not edit `packages/helia-ui` in the hub
repo from here. When the site needs something the package lacks or gets wrong:

- File an issue on `AmbiqAI/helia-ui` as a sub-issue of #67, with the heliaCORE
  page, header, or function that needs it as the acceptance fixture.
- Meanwhile, either write a site component on the package tokens or leave the
  construct plain. Never work around a package defect with raw CSS.
- Consume the package by pinned git tag only, never a branch.

## Gotchas

- `npm run prebuild` generates the reference and coverage data. A dev server
  started without it returns 500s on the generated pages.
- Product repos are branch-only. Never push to `main`; the owner merges the PR.
- Stage explicitly; never `git commit -a`. No attribution or sign-off lines in
  commit messages.
- Playwright uses the existing headless-shell cache only
  (`PLAYWRIGHT_BROWSERS_PATH=~/Library/Caches/ms-playwright`,
  `PLAYWRIGHT_CHROMIUM_USE_HEADLESS_SHELL=1`). Never `npx playwright install`,
  never headed.
- Never `pkill -f` a dev server; kill by port with `lsof -ti :<port> | xargs kill`.
- American English throughout.
- Doxygen 1.17.0 is pinned in CI for the XML schema the generator expects.

## Key files

| Path | What it is |
|---|---|
| `astro-site/SITE-PLAN.md` | Agreed structure and the reasoning |
| `astro-site/LANDING-PAGE-CONTENT.md` | Landing copy draft awaiting approval |
| `astro-site/astro.config.mjs` | Sections, header, accent, redirects |
| `astro-site/scripts/build-reference.mjs` | Doxygen to MDX plus JSON artifacts |
| `astro-site/scripts/build-coverage.mjs` | Coverage table from the headers |
| `astro-site/reference.config.json` | The eight operator families |
| `astro-site/redirects.json` | Old Sphinx routes to new ones |
