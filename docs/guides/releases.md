# Versioning & Releases

heliaCORE uses [release-please](https://github.com/googleapis/release-please)
to manage versions. Every merge to `main` triggers `.github/workflows/release.yml`:

1. **release-please** scans conventional-commit subjects since the last
   tag, decides whether to open a "Release PR" (bumping the version),
   or — when that PR is itself merged — to create a tag and a GitHub
   Release.
2. **publish-staticlibs** builds `libns-cmsis-nn-<cpu>-<version>.a` and
   the SDK tarball for each Cortex-M target, uploads them as Release
   assets, and writes per-arch SHA-256 checksums.
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
| `ns-cmsis-nn-<cpu>-gcc-<version>.tar.gz`             | SDK tarball (CMake users)     |
| `libns-cmsis-nn-<cpu>-<version>.a`                   | Bare static archive           |
| `*.sha256`                                           | SHA-256 of each artifact      |

`<cpu>` is one of `cortex-m0`, `cortex-m4`, `cortex-m55`.

!!! note "Cortex-M0/M0+ naming"
   The release asset name uses `cortex-m0` for the baseline ARMv6-M package.
   Use that artifact for Cortex-M0/M0+ class Apollo targets.

## See also

- Operator runbook: [Releasing](../releasing.md) — how to recover when
  the pipeline fails.
- Toolchain identity is recorded in `ns-cmsis-nn-manifest.json` inside
  each tarball. See [Toolchain Pinning](toolchains.md).
