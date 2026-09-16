# heliaCORE Astro Starlight site

The Astro Starlight replacement for the Sphinx site in `docs/`, built on the
shared `@ambiqai/helia-ui` design system. It is a scaffold: the page conversion,
the generated C API reference, and the charts land in
AmbiqAI/ns-cmsis-nn#518, #519 and #520.

**GitHub Pages still serves the Sphinx build.** CI builds this site to an
artifact named `astro-site` for review only. Pages switches over in
AmbiqAI/ns-cmsis-nn#521.

## Running it locally

Node 24 and the npm 11 it ships. `engine-strict` makes `engines` a refusal
rather than a warning, so an older pair fails the install instead of producing
a tree nothing has been exercised on.

```sh
cd astro-site
npm ci
npm run dev      # dev server at http://localhost:4321/ns-cmsis-nn/
npm run build    # static output in dist/
npm run preview  # serve dist/ at the base path
npm run check    # astro check
```

The site is served under the base path `/ns-cmsis-nn/`, matching the Pages URL.
Write every link in the content relative (`getting-started/`) so it follows the
base path; a root-relative link (`/getting-started/`) resolves in `astro dev`
and 404s in production.

## The design system is pinned by tag

`@ambiqai/helia-ui` is installed from a git tag, never a branch:

```json
"@ambiqai/helia-ui": "github:AmbiqAI/helia-ui#v0.1.0-alpha.7"
```

To move to a newer release, change the tag in `package.json`, then regenerate
the lockfile and prove a clean install:

```sh
cd astro-site
npx -y npm@11.19.0 install --package-lock-only --ignore-scripts
npm ci
npm run build
```

Commit `package.json` and `package-lock.json` together. Never hand-edit the
lockfile.

Tags are published from
[AmbiqAI/helia-ui](https://github.com/AmbiqAI/helia-ui). Anything the package
cannot do for this site is a helia-ui issue, not a local stylesheet: this site
carries no CSS of its own, and the theme, the shell and the discoverability
metadata all come from `heliaStarlight()`.

## Directory name

The Sphinx build writes its HTML to `site/` at the repository root and clears
that directory on every run (`scripts/docs/build_sphinx_docs.sh`), so this
source tree lives in `astro-site/` instead. The two can be merged once the
Sphinx build is retired.
