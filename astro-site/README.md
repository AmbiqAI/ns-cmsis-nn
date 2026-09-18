# heliaCORE documentation

The customer documentation site uses Astro/Starlight and an immutable
`@ambiqai/helia-ui` release. Product content and composition live here; shared
components and generators are maintained in `AmbiqAI/helia-ui`.

## Develop and verify

Use Node 24, npm 11, Python 3.11 or newer, and Doxygen 1.17.0. CI verifies the
Doxygen download against its pinned checksum.

```sh
cd astro-site
npm ci
npm run dev
```

Use the npm scripts so API generation runs before the development server.
After changing public headers, run `npm run reference` to refresh the API.

```sh
npm run build
npm run check
npm run test:coverage
npm run test:smoke
```

The browser suite uses Playwright Chromium. CI installs it; local development
uses the cached browser. `npm run build` regenerates the API, validates public
coverage and content, renders the site, writes legacy redirects, and checks
internal links and fragments. Build output is `dist/` and browser evidence is
`test-results/`; neither is committed.

## Content and API

Authored pages live in `src/content/docs/`. Prefer Markdown for prose and MDX
when composing shared parts. Site-specific components live in `src/components/`.
Do not put internal reviews or unfinished planning notes in the public collection.

`reference.config.json` configures API input headers and family grouping.
`scripts/build-reference.mjs` uses the repository's
`Documentation/Doxygen/nn.dxy.in` template to generate XML, then invokes the
installed `helia-ui-doxyref` command. Generated API pages and models are ignored
by Git and reproduced during the build. `scripts/check-coverage.mjs` compares
exact public declaration names with the generated index.

Use `redirects.json` for known moved pages. The build emits physical `.html`
redirect files for older links. Unknown URLs receive `404.html`, which explains
that the page moved and redirects to Home after five seconds. Missing anchors
on existing pages do not trigger a 404; those links land on the existing page.

## Publishing and recovery

`.github/workflows/docs.yml` builds, validates, and browser-tests the site before
uploading its artifact. Pages deploys that exact artifact. Pull requests only
produce review artifacts; pushes to main publish the site.

On a release, `release.yml` calls the same workflow with the exact release
commit, independently of CMSIS-Pack generation. Existing-tag asset recovery
never publishes documentation. CMSIS-Pack retains its own Doxygen generation.

To recover a failed deployment, rerun the failed deployment job against its
existing tested artifact. If the artifact has expired, dispatch `docs.yml`
from main to rebuild and validate the current site. For a content regression,
revert the responsible change through a PR and publish the resulting tested
main commit. Do not redeploy an old release's documentation over the live site.
