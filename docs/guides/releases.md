# Versioning & Releases

heliaCORE uses [release-please](https://github.com/googleapis/release-please)
to manage versions. Every merge to `main` triggers `.github/workflows/release.yml`:

1. **release-please** scans conventional-commit subjects since the last
   tag, decides whether to open a "Release PR" (bumping the version),
   or — when that PR is itself merged — to create a tag and a GitHub
   Release.
2. **publish-staticlibs** builds static libraries and SDK tarballs for each
   Cortex-M target across GCC, ATfE, and armclang, then publishes SDK tarballs
   plus per-toolchain staticlib bundles with SHA-256 checksums.
3. **publish-ci-image** builds and pushes
   `ghcr.io/ambiqai/ns-cmsis-nn-ci:vX.Y.Z` so consumers can pin a
   reproducible build environment.
4. **publish-pack** generates `Ambiq.NS-CMSIS-NN.<version>.pack`,
   schema-validates it against `PACK.xsd`, and uploads it.
5. **release-unit-tests** and **release-helia-core-tester** run
   against the freshly-built artifacts to verify the release actually
   works before customers see it.

## Tag scheme

```
v<MAJOR>.<MINOR>.<PATCH>
```

- `MAJOR` bumps on `feat!:` / `fix!:` / `BREAKING CHANGE:` footers.
- `MINOR` bumps on any `feat:`.
- `PATCH` bumps on `fix:` and similar.

Other conventional types (`docs:`, `chore:`, `refactor:`, `test:`,
`ci:`, `build:`, `perf:`) do **not** trigger a release on their own,
but their bodies still appear in `CHANGELOG.md`.

## What's in a release

Every GitHub Release contains:

| Asset                                                | Purpose                       |
|------------------------------------------------------|-------------------------------|
| `Ambiq.NS-CMSIS-NN.<version>.pack`                   | CMSIS-Pack                    |
| `ns-cmsis-nn-<cpu>-<toolchain>-<version>.tar.gz`     | SDK tarball (CMake users)     |
| `ns-cmsis-nn-staticlibs-<toolchain>-<version>.zip`   | Bare static archives by CPU   |
| `*.sha256`                                           | SHA-256 of each artifact      |

`<cpu>` is one of `cortex-m0`, `cortex-m4`, `cortex-m55`.
`<toolchain>` is one of `gcc`, `atfe`, `armclang`.

:::{note} Cortex-M0/M0+ naming
The release asset name uses `cortex-m0` for the baseline ARMv6-M package. Use
that artifact for Cortex-M0/M0+ class Apollo targets.
:::

## Required vs optional assets

Not every asset is equally load-bearing. If a release is missing something,
use this table to know whether that's a release-blocking bug or an expected
gap (see [AmbiqAI/ns-cmsis-nn#228](https://github.com/AmbiqAI/ns-cmsis-nn/issues/228)):

| Asset                                                                | Status       | Why                                                                 |
|-----------------------------------------------------------------------|--------------|----------------------------------------------------------------------|
| `vX.Y.Z` tag + GitHub Release (source)                                | **Required** | Created by release-please; the release exists once this lands.       |
| `Ambiq.NS-CMSIS-NN.<version>.pack`                                     | **Required** | Only depends on the `gcc` static libs.                                |
| `ns-cmsis-nn-<cpu>-gcc-<version>.tar.gz`                               | **Required** | Free toolchain, no license dependency.                                 |
| `ns-cmsis-nn-<cpu>-atfe-<version>.tar.gz`                              | **Required** | Free toolchain (LLVM-Embedded-Toolchain-for-Arm), no license dependency. |
| `ns-cmsis-nn-staticlibs-gcc-<version>.zip`                             | **Required** | Same as above.                                                          |
| `ns-cmsis-nn-staticlibs-atfe-<version>.zip`                            | **Required** | Same as above.                                                          |
| `ns-cmsis-nn-<cpu>-armclang-<version>.tar.gz`                          | **Conditional** | Required when an Arm Compiler for Embedded licence is configured; optional otherwise. |
| `ns-cmsis-nn-staticlibs-armclang-<version>.zip`                        | **Conditional** | Same as above.                                                          |
| `ghcr.io/ambiqai/ns-cmsis-nn-ci:vX.Y.Z` image                          | **Required** | Needed by `release-unit-tests` / `release-helia-core-tester`.           |
| GitHub Pages docs update                                               | **Required** | Rides along with `publish-pack`.                                       |

Armclang is licensed through Arm's **User-Based Licensing**: the
`ARM_UBL_LICENSE_IDENTIFIER` secret is an activation code, redeemed once per
release run by `armlm activate --code` in the `publish-staticlibs-armclang`
job before any compilation happens. That job builds all three Cortex-M
targets on a single runner precisely so the licence is activated once and the
Arm Compiler archive is downloaded once.

With the identifier configured, the eight `armclang` assets are **required**
— `release-verify` promotes them and fails the release if they are missing,
taking the required bar from 17 assets to 25. Without it,
`resolve-release-capabilities` reports the capability as unavailable, every
step of `publish-staticlibs-armclang` self-skips (the job succeeds as a
no-op and self-reports via `::notice::`/`::warning::` annotations), the
`armclang` bundle is omitted from `publish-staticlib-bundles`, and the eight
assets drop back to optional. Either way the required gcc/atfe assets are
unaffected: they come from a different job entirely, and
`publish-staticlibs-armclang` is `continue-on-error: true`, so armclang can
never hold up the tag, pack, gcc/atfe libs, CI image, or tests.

Before v7.30.0 this gate read `ARMLMD_LICENSE_FILE`, the legacy FlexLM
licence-file variable, which has never been configured on this repository —
which is why no release from v7.24.1 onward published any armclang asset
(see [AmbiqAI/ns-cmsis-nn#275](https://github.com/AmbiqAI/ns-cmsis-nn/issues/275)).

## Recovering assets for an existing tag

`release.yml` also accepts a manual, idempotent recovery run via
`workflow_dispatch` with a `recover_tag` input (e.g. `v7.29.2`). This
**never creates, moves, or re-publishes a tag** — it only re-runs the
staticlib/bundle/pack jobs against an already-published release and re-uploads
(`gh release upload --clobber`) any assets that are missing or need updating:

```bash
gh workflow run release.yml --ref main -f recover_tag=v7.29.2
```

The run fails fast if `recover_tag` isn't already a published GitHub Release,
so it can't be used to sneak out a brand-new tag. See
[Contributing → Release recovery](../contributing.md#release-recovery) for
the full recovery runbook.

Every recovery run — and every normal release run — resolves `recover_tag`
(or the freshly-cut tag) to its exact target commit **once**, via the GitHub
API. A recovery dispatch's own triggering ref (whatever branch was selected
when running the workflow, typically `main`) is never used to build release
assets; only the commit the tag itself points at is.

Historical recovery restores customer GitHub Release assets only:
static-library archives/bundles and the CMSIS-Pack. It deliberately skips
`publish-ci-image`, `release-unit-tests`, and `release-helia-core-tester`.
The CI image is build infrastructure rather than a release asset, and its
tool acquisition is qualified independently of historical asset recovery.
Skipping it also makes recovery incapable of publishing an old versioned image
or repointing `:latest`. Normal new releases continue to require the versioned
CI image and both container test suites.

### Immutable CI tool acquisition

Normal-release CI images install their build and simulation tools from
`ci/tools/manifest.json`. Each entry records an exact version, vendor HTTPS
URL, SHA-256 digest, extraction rule, executable probe, environment export,
and license classification. `scripts/install_ci_tools.sh` validates the
manifest, verifies every archive before extraction, and emits
`/opt/ns-cmsis-nn/tool-env.json` with the resolved paths and provenance.

This repository-owned manifest replaces the retired vcpkg-artifacts/vcpkg-ce
registry. The image build has no package-solver or floating-registry dependency.
Armclang remains installed from Arm's vendor-hosted archive, but licensed
execution is capability-gated separately through
`ARM_UBL_LICENSE_IDENTIFIER`.
Updating any tool requires a reviewed manifest version, URL, and digest change
plus a clean image build and version smoke test.

Recovering a genuinely old tag surfaces two more subtleties, both handled
automatically via an explicit **two-tree checkout architecture**: a
*trusted/current* tree (this repo's current `main`, whatever ref/commit
triggered the run) supplies recovery tooling — helper scripts and the
hardened Dockerfile — while a *distinct, immutable/historical* tree (pinned
to the resolved historical commit) supplies the only source payload that is
ever built, packaged, or uploaded. The two trees always use different
on-disk paths (never the same checkout directory), so neither can silently
overwrite the other:

- **CI-image tooling vs. source.** The reusable Docker build workflow
  checks out the repository twice, into two *different* directories: an
  unpinned checkout at `_tooling/` (whatever ref/commit triggered the run)
  purely to obtain `scripts/ci/resolve_image_tags.sh` — a historical tag
  can predate that helper script entirely — and a second checkout pinned to
  the resolved historical commit at `_source/` for the actual Docker build
  context. `docker build` is then invoked as
  `-f _tooling/.devcontainer/Dockerfile _source`: the **Dockerfile always
  comes from the current, hardened tree** (so a historical, pre-fix
  Dockerfile bootstrap bug can never resurface), while the **build context
  is always the immutable historical source** (so the built image still
  reflects exactly what that historical tag shipped). Giving each checkout
  its own `path:` is required, not cosmetic: an earlier revision of this
  recovery path had both checkouts default to the *same* directory, so the
  historical, pinned checkout silently clobbered the current tooling
  checkout's files — including the Dockerfile — right before the build ran
  (AmbiqAI/ns-cmsis-nn#228). Normal push/direct-dispatch/schedule runs are
  unaffected by this split: with no `recover_tag`, both `_tooling` and
  `_source` resolve to the same `github.sha`, so the two trees are
  identical content living at two paths, and the build behaves exactly as
  it always has.
- **Lightweight vs. annotated tags for `gen_pack.sh`.** Release Please
  creates lightweight tags, but Open-CMSIS-Pack's `gen-pack` (run with
  `PACK_CHANGELOG_MODE=tag`) requires an *annotated* tag with a non-empty
  message. `publish-pack`'s only long-lived checkout is pinned directly to
  the resolved historical commit — that historical tree is the *only*
  source `gen_pack.sh` ever packages. Because
  `scripts/ci/ensure_local_tag_annotation.sh` is itself part of the
  *current* repository, `publish-pack` also checks out the current,
  merged repository into a second, unpinned directory (`_tooling/`) purely
  to obtain that helper, then invokes it explicitly against the historical
  checkout's working directory (`bash
  _tooling/scripts/ci/ensure_local_tag_annotation.sh "$TAG"
  "$GITHUB_REPOSITORY" "$GITHUB_WORKSPACE"`). The helper — for a
  lightweight tag only — creates a **local-only** annotated tag at the
  exact same commit (message sourced from the existing GitHub Release
  body, XML-escaped, or a deterministic fallback) so `gen_pack.sh`
  succeeds. It never runs `git push`; the remote/immutable tag is
  untouched, already-annotated tags are left completely unaltered, and
  `_tooling/` is never read by `gen-pack-action` or packaged into the
  `.pack` output — it exists solely to supply the helper script.

## See also

- Maintainer release notes in [Contributing](../contributing.md#maintainer-release-notes)
   explain how to recover when the pipeline fails.
- Toolchain identity is recorded in `manifest.json` inside
  each tarball. See [Toolchain Pinning](toolchains.md).
