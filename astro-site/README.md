# heliaCORE Astro Starlight site

The Astro Starlight replacement for the Sphinx site in `docs/`, built on the
shared `@ambiqai/helia-ui` design system. The handwritten pages are converted
and the C API reference at `/reference/api/` is generated at build time. The
benchmark charts land in AmbiqAI/ns-cmsis-nn#520, which is why the two chart
sections on the benchmarks page have nothing behind them yet.

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
npm run dev        # dev server at http://localhost:4321/ns-cmsis-nn/
npm run build      # static output in dist/
npm run preview    # serve dist/ at the base path
npm run check      # astro check
npm run reference  # regenerate the C API reference on its own
```

Doxygen has to be on `PATH`: `brew install doxygen`, or the pinned tarball the
`astro` job in `.github/workflows/docs.yml` unpacks. `npm run build` refuses
without it.

The site is served under the base path `/ns-cmsis-nn/`, matching the Pages URL.
Write every link in the content relative (`getting-started/`) so it follows the
base path; a root-relative link (`/getting-started/`) resolves in `astro dev`
and 404s in production.

## The generated C API reference

`scripts/build-reference.mjs` runs as `prebuild`, so `npm run build` always has
a current reference. It does three things:

1. Appends an XML-only override block to `Documentation/Doxygen/nn.dxy.in`, the
   Doxyfile the Sphinx site already uses, and runs Doxygen over `Include/` into
   `.cache/reference/xml/`. Doxygen takes the last assignment of a tag, so the
   template stays the one definition of what is extracted and the block only
   redirects the output.
2. Runs `helia-ui-doxyref` over that XML. Pages go to
   `src/content/docs/reference/api/`, guarded by the `.doxyref-manifest.json`
   the generator writes there; `reference.json`, a JSON per module, `llms.txt`
   and `llms-full.txt` go to `public/reference/api/` and ship at
   `/ns-cmsis-nn/reference/api/`.
3. Writes the operator-family index at `reference/api/index.mdx` from
   `reference.config.json`.

`reference.config.json` carries the eight families verbatim from
`GROUP_PATTERNS` in `docs/_ext/api_group_index.py` — same ids, same patterns,
same order — so `scripts/check_api_group_classification.py` guards this page as
well as the Sphinx one. The generator has no grouping option, which is why the
index is assembled here from `reference.json` rather than by the generator;
that, and the two workarounds the script carries, are filed on the helia-ui
tracking issue.

### The output is not committed

Everything under `src/content/docs/reference/api/` and `public/reference/` is
gitignored, along with the `.cache/` scratch. Committing it would put a
thousand-odd generated files in front of every reviewer of an unrelated change
and invite a hand-edit that the next build silently drops. What is committed is
the input: the Doxyfile, the headers, and `reference.config.json`. Two runs from
a clean tree at the same commit produce a byte-identical `reference.json`, so
the build is the record rather than the checkout.

The cost is that `Include/` and Doxygen are required to work on the site at all.
`npm run dev` and `npm run check` generate the reference when it is missing;
`npm run build` regenerates it every time.

The `Reference` sidebar group lists its handwritten pages explicitly, like the
rest of the sidebar, and carries one `autogenerate` sub-entry for
`reference/api` because those pages do not exist until the prebuild runs. The
generated `index.mdx` is titled `API` with `sidebar.order: 0`, so it lands first
inside that sub-entry with the module pages under it.

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
