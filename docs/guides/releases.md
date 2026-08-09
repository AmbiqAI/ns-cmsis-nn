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
retired vcpkg-artifacts dependency requires the separate migration tracked by
#233. Skipping it also makes recovery incapable of publishing an old versioned
image or repointing `:latest`. Normal new releases continue to require the
versioned CI image and both container test suites.

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

A third live recovery run against `v7.29.2` (31337539282) got past both of
the issues above and surfaced one more, purely internal to the CI image's
own vcpkg bootstrap (no checkout-path involvement): the Dockerfile pinned
and installed only the `vcpkg-glibc` binary release asset, but
`vcpkg x-update-registry --all` / `vcpkg activate` shell out to Node-based
vcpkg-artifacts tooling (`scripts/vcpkg-tools.json`, `vcpkg-artifacts.mjs`,
`.vcpkg-root`, `triplets/`, …) that ships only in the separate
`vcpkg-standalone-bundle.tar.gz` release asset, so those commands failed
with `.../scripts/vcpkg-tools.json: No such file or directory`. The
Dockerfile now downloads, checksum-verifies, and extracts the pinned
standalone bundle into `VCPKG_ROOT` first, then downloads/checksums/installs
the pinned `vcpkg-glibc` binary on top of it (its executable, not the
bundle's own bootstrap binaries, is what ends up on `PATH`), and asserts at
build time that the bundle's companion files actually landed before any
artifacts command runs — so a future regression here fails the image build
with a specific message instead of an oblique one from deep inside vcpkg's
Node tooling.

> **Resolved: redundant registry-update call removed, plus a downstream
> `VCPKG_DOWNLOADS` bug fixed (AmbiqAI/ns-cmsis-nn#228 follow-up).** An
> earlier revision of the fix above ran `vcpkg x-update-registry --all` as
> a separate step immediately before `vcpkg activate`. Local isolated
> `linux/amd64` Docker reproduction (2026-08) found that call redundant —
> `vcpkg activate` already performs its own registry refresh internally
> (both the default Microsoft `vcpkg-ce-catalog` registry and this repo's
> custom `arm` registry from `vcpkg-configuration.json`) as part of
> resolving artifacts — and it was the exact call that had been observed,
> in earlier local testing, to intermittently fail with an internal
> vcpkg-artifacts error (`z-extract.cpp: Value was null`) or a plain
> network timeout while fetching the catalog archive. Repeated isolated
> trials in this session (one full `vcpkg activate` run installing all
> seven pinned tools, plus three additional fast `x-update-registry --all`
> reruns against a cached pre-activate image) succeeded 3 of 4 times, with
> the one failure being a plain 120s network timeout during a Node.js
> download rather than the `z-extract` error — evidence that the earlier
> failure was upstream/network flakiness rather than a permanent
> post-`vcpkg`-artifacts-sunset breakage. Removing the separate
> `x-update-registry --all` call removes one of the two places that fetch
> could flake, without changing which registries get refreshed (`activate`
> still refreshes both). `scripts/check_vcpkg_bundle_companion_files.sh`
> now asserts `x-update-registry` never reappears anywhere in the
> Dockerfile, so this simplification can't silently regress.
>
> The same reproduction, now reaching further than any previous run,
> exposed a second, unrelated pre-existing bug: the artifacts-resolution
> step ended with `rm -rf "${VCPKG_DOWNLOADS}"` immediately after
> `vcpkg activate` succeeded — deleting the just-installed tool binaries
> (that's exactly where `activate --downloads-root=...` places them)
> before the very next step's build-time smoke check (or, per this file's
> own header comment, before `.github/actions/cmsis-env`'s intended
> replay-via-`vcpkg-env.json` at CI job runtime) could ever use them. This
> was never previously exposed because `x-update-registry --all` had
> always failed before execution reached this far in every prior live or
> local run. It has been removed (not moved or reinstated in any form);
> the installed tool tree under `VCPKG_DOWNLOADS` is meant to persist in
> the built image.
>
> Fixing both of the above let the build-time smoke check reach `armclang
> --version` for the first time ever, which failed on a license checkout
> (`ARMLMD_LICENSE_FILE is not set`) — expected and unrelated to any of
> the above: hosted GitHub runners have no Arm Compiler for Embedded
> license (this is the exact, already-documented condition
> `resolve-release-capabilities` / `publish-staticlibs` treat as an
> optional, best-effort asset elsewhere in `release.yml`, per
> AmbiqAI/ns-cmsis-nn#228), and this Dockerfile's own build never receives
> `ARMLMD_LICENSE_FILE` as a build secret either. The smoke check now
> tolerates only that specific, known error message and still fails the
> build on any other `armclang` failure, so a real regression (e.g. the
> tool failing to install at all) is still caught. With these three fixes
> in place, a full isolated `linux/amd64` Docker build was run end-to-end
> through activation and every smoke tool: `cbuild`, `cpackget`,
> `csolution`, `arm-none-eabi-gcc`, and `FVP_Corstone_SSE-300` all
> succeeded; `armclang` correctly self-skipped on the documented
> no-license condition. This upstream risk is therefore considered
> resolved for this repository's usage, pending confirmation on a live
> `publish-ci-image` run.

## See also

- Maintainer release notes in [Contributing](../contributing.md#maintainer-release-notes)
   explain how to recover when the pipeline fails.
- Toolchain identity is recorded in `manifest.json` inside
  each tarball. See [Toolchain Pinning](toolchains.md).
