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
| `ns-cmsis-nn-<cpu>-armclang-<version>.tar.gz`                          | **Optional** | Requires a commercial Arm Compiler for Embedded license.                |
| `ns-cmsis-nn-staticlibs-armclang-<version>.zip`                        | **Optional** | Same as above.                                                          |
| `ghcr.io/ambiqai/ns-cmsis-nn-ci:vX.Y.Z` image                          | **Required** | Needed by `release-unit-tests` / `release-helia-core-tester`.           |
| GitHub Pages docs update                                               | **Required** | Rides along with `publish-pack`.                                       |

Hosted GitHub runners have no Arm Compiler for Embedded license configured
(no `ARMLMD_LICENSE_FILE` secret), so the `armclang` static-library legs of
`publish-staticlibs` are **skipped intentionally** — they self-report via a
`::notice::`/`::warning::` annotation rather than failing the job — and the
`armclang` bundle is omitted from `publish-staticlib-bundles` the same way.
Every required asset above is unaffected: it is produced by a different
job/toolchain leg entirely, so an unlicensed armclang has no way to hold up
the tag, pack, gcc/atfe libs, CI image, or tests. Provisioning a real
Armclang license (e.g. on a protected self-hosted runner) is tracked
separately; this repo does not claim armclang static-lib publication works
on hosted runners today.

## Recovering assets for an existing tag

`release.yml` also accepts a manual, idempotent recovery run via
`workflow_dispatch` with a `recover_tag` input (e.g. `v7.29.2`). This
**never creates, moves, or re-publishes a tag** — it only re-runs the
staticlib/bundle/pack/CI-image jobs against an already-published release and
re-uploads (`gh release upload --clobber`) any assets that are missing or
need updating:

```bash
gh workflow run release.yml --ref main -f recover_tag=v7.29.2
```

The run fails fast if `recover_tag` isn't already a published GitHub Release,
so it can't be used to sneak out a brand-new tag. See
[Contributing → Release recovery](../contributing.md#release-recovery) for
the full recovery runbook.

Every recovery run — and every normal release run — resolves `recover_tag`
(or the freshly-cut tag) to its exact target commit **once**, via the GitHub
API, and pins every downstream checkout (`publish-staticlibs`,
`publish-pack`, the CI image build, and the release test suites) to that
exact commit. A recovery dispatch's own triggering ref (whatever branch was
selected when running the workflow, typically `main`) is never used to build
release assets; only the commit the tag itself points at is. `:latest` is
also never repointed during a recovery run (`publish_latest` is forced off
whenever `recovery_mode == 'true'`), so recovering an old tag like `v7.29.1`
can never regress what `:latest` resolves to.

Recovering a genuinely old tag surfaces two more subtleties, both handled
automatically:

- **CI-image tooling vs. source.** The reusable Docker build workflow
  checks out the repository twice: once *unpinned* (whatever ref/commit
  triggered the run) purely to obtain `scripts/ci/resolve_image_tags.sh` —
  a historical tag can predate that helper script entirely — and a second
  time pinned to the resolved historical commit for the actual `docker
  build` context/Dockerfile. Only the pinned checkout ever contributes to
  the built image.
- **Lightweight vs. annotated tags for `gen_pack.sh`.** Release Please
  creates lightweight tags, but Open-CMSIS-Pack's `gen-pack` (run with
  `PACK_CHANGELOG_MODE=tag`) requires an *annotated* tag with a non-empty
  message. `publish-pack` runs
  `scripts/ci/ensure_local_tag_annotation.sh` first, which — for a
  lightweight tag only — creates a **local-only** annotated tag at the
  exact same commit (message sourced from the existing GitHub Release
  body, XML-escaped, or a deterministic fallback) so `gen_pack.sh`
  succeeds. It never runs `git push`; the remote/immutable tag is
  untouched, and already-annotated tags are left completely unaltered.

## See also

- Maintainer release notes in [Contributing](../contributing.md#maintainer-release-notes)
   explain how to recover when the pipeline fails.
- Toolchain identity is recorded in `manifest.json` inside
  each tarball. See [Toolchain Pinning](toolchains.md).
