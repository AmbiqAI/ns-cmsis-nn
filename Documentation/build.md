# Build system & SSoT CMake module

This page documents the build-side single source of truth (SSoT) for
ns-cmsis-nn's per-group source layout, and the three integration recipes
that consume it.

## Why an SSoT module?

ns-cmsis-nn is integrated into three distinct build systems — the standalone
CMake build (`CMakeLists.txt`), the Zephyr module (`zephyr/`) and the NSX
module (`nsx/`). Before #148 each integration kept its own per-group source
list (or a recursive `file(GLOB)`) and they drifted out of sync over time.

`cmake/ns_cmsis_nn.cmake` is the one place that decides, for each operator
group, which `.c` files belong to it. All three integrations include this
module and call its public entry points instead of enumerating sources
themselves.

## Public API

```cmake
include(<repo>/cmake/ns_cmsis_nn.cmake)
```

| Function | Purpose |
|---|---|
| `ns_cmsis_nn_groups(<out>)` | Set `<out>` to the list of all known operator group ids. |
| `ns_cmsis_nn_group_sources(<group> <out>)` | Set `<out>` to the absolute source paths for a single group. |
| `ns_cmsis_nn_attach(<target> [GROUPS …\|ALL] [DTYPES …\|ALL] [INCLUDE_DIRS_VISIBILITY PUBLIC\|PRIVATE\|INTERFACE])` | Add the resolved source set and the public `Include/` directory to an existing `<target>`. |

`ns_cmsis_nn_attach()` does not create a target — the caller owns it.
The include directory is wrapped in `$<BUILD_INTERFACE:…>` so consumers can
safely re-export the target via `install(TARGETS … EXPORT …)`.

## Group taxonomy

| Group id | Source subdirectory |
|---|---|
| `activation`     | `Source/ActivationFunctions/` |
| `basicmath`      | `Source/BasicMathFunctions/` |
| `comparison`     | `Source/ComparisonFunctions/` |
| `concatenation`  | `Source/ConcatenationFunctions/` |
| `convolution`    | `Source/ConvolutionFunctions/` |
| `fullyconnected` | `Source/FullyConnectedFunctions/` |
| `gather`         | `Source/GatherFunctions/` |
| `lstm`           | `Source/LSTMFunctions/` |
| `nnsupport`      | `Source/NNSupportFunctions/` |
| `pad`            | `Source/PadFunctions/` |
| `pooling`        | `Source/PoolingFunctions/` |
| `quantization`   | `Source/QuantizationFunctions/` |
| `reshape`        | `Source/ReshapeFunctions/` |
| `softmax`        | `Source/SoftmaxFunctions/` |
| `stridedslice`   | `Source/StridedSliceFunctions/` |
| `svd`            | `Source/SVDFunctions/` |
| `transpose`      | `Source/TransposeFunctions/` |

The `DTYPES` filter accepts `s4 s8 s16 s32 s64 q7 q15`. Files whose basename
carries no recognized dtype tag (e.g. `arm_nntables.c`) are always kept.

## Consumer recipes

### Standalone CMake

```cmake
add_subdirectory(third_party/ns-cmsis-nn)
target_link_libraries(my_app PRIVATE cmsis-nn)
```

The library is created as `cmsis-nn` (real STATIC target) with `ALIAS`es
`ns-cmsis-nn` and `ns::cmsis-nn`. The real-target name is preserved as
`cmsis-nn` so existing TFLM integrations link unchanged
(`OUTPUT_NAME=cmsis-nn`). Per-group selection is exposed as standard
`option()` knobs (`ACTIVATION`, `CONVOLUTION`, …); see the top-level
[`CMakeLists.txt`](../CMakeLists.txt).

### Zephyr

In your application's `prj.conf`:

```kconfig
CONFIG_NS_CMSIS_NN=y
CONFIG_NS_CMSIS_NN_ALL=y     # or pick individual groups
```

Per-group symbols are listed in [`zephyr/Kconfig`](../zephyr/Kconfig); their
spellings match the standalone `option()` names (e.g.
`CONFIG_NS_CMSIS_NN_BASICMATHSNN`, `CONFIG_NS_CMSIS_NN_STRIDEDSLICE`,
`CONFIG_NS_CMSIS_NN_SVDF`). The Zephyr module translates the enabled
symbols into SSoT group ids and calls `ns_cmsis_nn_attach()` on the
Zephyr library target.

### NSX

In source mode (the default), select groups via `NSX_CMSIS_NN_GROUPS`:

```cmake
# Whole library (default)
add_subdirectory(${CMSIS_NN_REPO}/nsx nsx_cmsis_nn)

# Subset
set(NSX_CMSIS_NN_GROUPS "activation;convolution;fullyconnected" CACHE STRING "")
add_subdirectory(${CMSIS_NN_REPO}/nsx nsx_cmsis_nn)

target_link_libraries(my_app PRIVATE nsx::cmsis_nn)
```

Prebuilt mode (set `NSX_CMSIS_NN_LIB` to an `.a`) bypasses the SSoT
entirely and only exposes headers. When a `manifest.json` sidecar is
discoverable next to the archive (see
["Float (F32/F16) capability manifest"](#float-f32f16-capability-manifest)
below), requesting `NSX_CMSIS_NN_ENABLE_F32`/`F16` is validated against
it at configure time.

### Prebuilt static library

Each GitHub Release ships per-arch prebuilts alongside the `.pack`:

```
libns-cmsis-nn-cortex-m0-<version>.a
libns-cmsis-nn-cortex-m4-<version>.a
libns-cmsis-nn-cortex-m55-<version>.a
```

Plus a `.sha256` next to each. Consume them from a project that has no
visibility into the cmsis-nn source tree via the prebuilt helper:

```cmake
include(<path>/ns-cmsis-nn/cmake/ns-cmsis-nn-prebuilt.cmake)

ns_cmsis_nn_import_prebuilt(
  LIBRARY      ${CMAKE_CURRENT_LIST_DIR}/libns-cmsis-nn-cortex-m4-7.24.1.a
  INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}/ns-cmsis-nn/Include)

target_link_libraries(my_app PRIVATE ns::cmsis-nn)
```

The same target names (`cmsis-nn`, `ns-cmsis-nn`, `ns::cmsis-nn`) the
source build exposes resolve to the imported archive, so consumer code
is build-mode agnostic.

The full SDK tarball (see below) additionally ships a `manifest.json`
and a `find_package(ns-cmsis-nn)`-compatible CMake package config that
auto-forwards the archive's compiled-in float capabilities — prefer
that path over the bare-archive helper above when you need F32/F16.

## Adding a new kernel group

1. Drop the new sources under `Source/<NewGroup>/`.
2. Add a branch to `_ns_cmsis_nn_group_def()` in
   [`cmake/ns_cmsis_nn.cmake`](../cmake/ns_cmsis_nn.cmake) describing the
   subdirectory, the glob patterns, and any explicitly-named extras.
3. Add the group id to the `_NS_CMSIS_NN_GROUPS` list in the same file.
4. (Standalone) Add an `option(NEWGROUP "…" ON)` and a per-group
   `add_subdirectory()` line in the top-level `CMakeLists.txt`.
5. (Zephyr) Add a `config NS_CMSIS_NN_NEWGROUP` in
   `zephyr/Kconfig` and a `"NEWGROUP;newgroup"` pair in the
   translation `foreach(_pair …)` in `zephyr/CMakeLists.txt`.
6. (NSX) Nothing to do — `NSX_CMSIS_NN_GROUPS` accepts any group id the
   SSoT knows about, including the new one.
7. Bump the matching `EXPECTED_COUNT_<group>` (and `EXPECTED_TOTAL`) in
   [`cmake/tests/ssot_contract/CMakeLists.txt`](../cmake/tests/ssot_contract/CMakeLists.txt),
   and add the new group id to `EXPECTED_GROUPS` in the same file.

## SSoT contract test

A small configure-time test pins the SSoT against silent drift (e.g. an
upstream sync that adds or removes a kernel without anyone noticing). It
asserts the group ordering, per-group source counts, the `GROUPS ALL`
total, file existence on disk, and that `ns_cmsis_nn_attach()` /
`DTYPES` filtering still behave. Run it locally with:

```sh
cmake -S cmake/tests/ssot_contract -B build/ssot_contract
```

A green configure means the contract holds. The `SSoT Contract` GitHub
Actions workflow runs the same command on every PR and on `main` push.
When a kernel-set change is intentional, bump the `EXPECTED_*` values in
[`cmake/tests/ssot_contract/CMakeLists.txt`](../cmake/tests/ssot_contract/CMakeLists.txt)
in the same PR — the failure message names the offending group and the
delta.

## Zephyr integration tests

The Zephyr module is verified at the wiring layer, not the firmware
layer: the kernel C is already exercised by the unit-test suite, so the
only Zephyr-specific risk is that `zephyr/CMakeLists.txt` or
`zephyr/Kconfig` plumbs something wrong (forgets a group, mistypes a
knob, drops the `select` glue on a deprecated alias, etc.). Two
configure-time checks pin that contract — no Zephyr SDK, no west, no
cross-compiler required:

1. **Wiring mock** —
   [`cmake/tests/zephyr_wiring/CMakeLists.txt`](../cmake/tests/zephyr_wiring/CMakeLists.txt)
   stubs the `zephyr_library*` and `zephyr_*compile_definitions` macros,
   sets every `CONFIG_NS_CMSIS_NN_*=1`, then `include()`s the real
   `zephyr/CMakeLists.txt` and asserts the captured wiring: 206 sources
   attached, `Include/` exposed globally, `CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY`
   propagated, and every SSoT group covered by the Kconfig translation
   table. It also drives the prebuilt path against a fake extracted SDK
   tarball with a `manifest.json` sidecar, asserting that a requested
   `CONFIG_NS_CMSIS_NN_ENABLE_F32/F16` is forwarded as `ARM_NN_ENABLE_F32/F16`
   compile definitions when the manifest reports that capability, and
   that requesting a capability the manifest does not report aborts
   configure with `FATAL_ERROR` (verified via a child `cmake` probe
   process, since a direct failure would abort the whole test script).
   Run with:

   ```sh
   cmake -S cmake/tests/zephyr_wiring -B build/zephyr_wiring
   ```

2. **Kconfig contract** —
   [`scripts/check_zephyr_kconfig.py`](../scripts/check_zephyr_kconfig.py)
   parses `zephyr/Kconfig` and asserts every renamed knob exists, that
   `NS_CMSIS_NN_ALL` `imply`s all of them, and that each deprecated
   alias `select`s both `DEPRECATED` and its renamed counterpart. Run
   with:

   ```sh
   python3 scripts/check_zephyr_kconfig.py
   ```

The `Zephyr Integration` GitHub Actions workflow runs both on every PR.
A real `west build` against the Zephyr SDK is intentionally not part of
PR gating: it would only re-prove what these wiring checks plus the
existing unit-test suite already cover, at the cost of a multi-GB SDK
pull on every CI run.

## NSX integration tests

The NSX module ([`nsx/CMakeLists.txt`](../nsx/CMakeLists.txt)) is
verified at the same wiring layer as Zephyr — the kernel C itself is
already cross-compiled on m0/m4/m55 by the unit-test suite, so the only
NSX-specific risk is that the module plumbs the SSoT or the install
rules incorrectly. Two checks pin that contract:

1. **Wiring mock** —
   [`cmake/tests/nsx_wiring/CMakeLists.txt`](../cmake/tests/nsx_wiring/CMakeLists.txt)
   stubs `NSX_BOARD_FLAGS_TARGET`, `add_subdirectory()`s the real NSX
   module, and asserts the resulting target graph. Parameterised by
   `-DCASE=`:

   | Case | What it pins |
   |------|--------------|
   | `source_all` | STATIC target, 206 sources, `nsx::cmsis_nn` alias, `EXPORT_NAME=cmsis_nn`, BUILD/INSTALL_INTERFACE split on Include/. |
   | `source_subset` | `NSX_CMSIS_NN_GROUPS="activation;convolution"` resolves to exactly 62 sources. |
   | `inline_asm` | `NSX_CMSIS_NN_USE_REQUANTIZE_INLINE_ASM=ON` propagates `CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY`. |
   | `prebuilt` | `NSX_CMSIS_NN_LIB=…` builds an INTERFACE wrapper + IMPORTED prebuilt target, alias still works. |
   | `prebuilt_manifest_ok` | Prebuilt lib + sibling `manifest.json` reporting `f32=true`; `NSX_CMSIS_NN_ENABLE_F32=ON` succeeds and forwards `ARM_NN_ENABLE_F32=1`. |
   | `prebuilt_manifest_reject` | Manifest reports `f32=false`; requesting `NSX_CMSIS_NN_ENABLE_F32=ON` aborts with `FATAL_ERROR` (via child `cmake` probe). |
   | `prebuilt_manifest_legacy` | Manifest has no `features` block (schema v1); requesting F32 is conservatively rejected the same way. |

   Run any case locally with:

   ```sh
   cmake -S cmake/tests/nsx_wiring -B build/nsx_wiring -DCASE=source_all
   ```

2. **Install contract** —
   [`scripts/check_nsx_install.sh`](../scripts/check_nsx_install.sh)
   drives a real configure → build → `cmake --install` on
   ubuntu-latest stock GCC, then asserts:

   - `nsxTargets.cmake` (+ `-noconfig`) installed at
     `lib/cmake/nsx/`,
   - `INTERFACE_INCLUDE_DIRECTORIES` and `IMPORTED_LOCATION` use
     `${_IMPORT_PREFIX}` (no absolute build-tree path leaks — regression
     guard for #159),
   - all public headers + `libnsx_cmsis_nn.a` land at the expected
     install paths,
   - a downstream `include(nsxTargets.cmake)` consumer can resolve the
     `cmsis_nn` target.

   ```sh
   scripts/check_nsx_install.sh
   ```

The `NSX Integration` GitHub Actions workflow runs the 4-leg wiring
matrix and the install contract on every PR / `main` push.

## PDSC contract test

The CMSIS-Pack manifest (`Ambiq.NS-CMSIS-NN.pdsc`) is the contract the
heliaRT consumer pack imports against. A drift between the manifest and
either the on-disk Source tree or the release-please-driven version
markers silently breaks downstream packs. The legacy
[`check_pdsc.sh`](../check_pdsc.sh) script only diffs the enumerated
`<file category="source">` paths against `Source/**/*.c`; the wider
contract is pinned by
[`scripts/check_pdsc.py`](../scripts/check_pdsc.py), which asserts:

- **Pack identity** — `schemaVersion`, `<name>` and `<vendor>` match the
  values the heliaRT consumer pins.
- **Component identity** — the
  `Cclass="Machine Learning" Cgroup="NN Lib" Csub="heliaCORE" Cvendor="Ambiq"`
  4-tuple is preserved; renaming any of them is a breaking change.
- **Version sync** — the latest `<release version>`, the component
  `Cversion`, the `NS_CMSIS_NN_VERSION_*` defines and `$Revision`
  comment in `Include/arm_nn_types.h`, and the `.` field of
  `.release-please-manifest.json` must all agree. This catches a manual
  bump that forgets one of these.
- **License plumbing** — every `<license name=...>` resolves on disk,
  and both `LICENSE` and `LICENSES/Apache-2.0.txt` are declared.
- **File existence** — every `<file name=...>` exists on disk (with the
  generated `Documentation/html/` artefacts whitelisted).
- **Source coverage** — every `Source/**/*.c` in the repo is enumerated
  as `category="source"`, so adding a kernel without updating the pack
  fails CI (this subsumes `check_pdsc.sh`).

Run locally with:

```sh
python3 scripts/check_pdsc.py
```

The `Verify that PDSC file is up to date` GitHub Actions workflow runs
both `check_pdsc.sh` and `scripts/check_pdsc.py` on every PR.

## Publishing the CMSIS-Pack

The release flow is driven by release-please. When a release-please
release PR is merged, [`release.yml`](../.github/workflows/release.yml)
tags `vX.Y.Z`, creates a GitHub Release, then:

1. Runs `python3 scripts/check_pdsc.py` as a fail-fast pre-flight.
2. Invokes [`Open-CMSIS-Pack/gen-pack-action`](https://github.com/Open-CMSIS-Pack/gen-pack-action)
   which runs Doxygen + [`gen_pack.sh`](../gen_pack.sh) + `packchk`.
3. Uploads `Ambiq.NS-CMSIS-NN.<version>.pack` as a release asset.
4. Publishes the generated Doxygen docs to GitHub Pages.

The pack name and `<version>` are derived from the pdsc vendor/name and
the `<release version>` element, which release-please bumps in lockstep
with `arm_nn_types.h` and `.release-please-manifest.json` (the PDSC
contract test pins this lockstep).

To exercise the pack pipeline without cutting a release — for example
to verify a `gen_pack.sh` change before tagging — run the
[`Pack dry-run`](../.github/workflows/pack-dryrun.yml) workflow from
the Actions tab. It runs the same gen-pack-action and uploads the
resulting `.pack` as a workflow artefact (14 day retention), with a
post-build assertion that the artefact is named
`Ambiq.NS-CMSIS-NN.<version>.pack`.

## Publishing prebuilt static libraries

The same `release.yml` cross-compiles `cmsis-nn` for cortex-m0,
cortex-m4 and cortex-m55 using GNU Arm Embedded
(`arm-none-eabi-gcc`), strips each archive, sha256s it, and uploads
both files to the GitHub Release:

```
libns-cmsis-nn-cortex-m{0,4,55}-<version>.a
libns-cmsis-nn-cortex-m{0,4,55}-<version>.a.sha256
```

A smoke-link step links a tiny TU that references one symbol per
kernel group against the produced `.a` and checks that all expected
symbols resolve. Failure of the smoke step blocks the upload for the
affected target.

Arch flags live in
[`cmake/toolchain/arm-none-eabi-gcc.cmake`](../cmake/toolchain/arm-none-eabi-gcc.cmake)
(also mirrored in [`scripts/smoke/smoke_staticlib.sh`](../scripts/smoke/smoke_staticlib.sh)
for the smoke link):

| target_cpu  | flags |
|-------------|-------|
| `cortex-m0`  | `-mcpu=cortex-m0 -mthumb -mfloat-abi=soft` |
| `cortex-m4`  | `-mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard` |
| `cortex-m55` | `-mcpu=cortex-m55 -mthumb -mfloat-abi=hard` |

Each archive is built as a **superset for its target_cpu/toolchain**:
integer kernels are always included, and `scripts/build_staticlib.sh`
is invoked with the float flags qualified for that combination (see
["Float (F32/F16) capability manifest"](#float-f32f16-capability-manifest)
below). `release.yml` always passes `--enable-f32`; `--enable-f16` is
passed only for `cortex-m55` (the only qualified combination today).
There is still exactly one archive per `target_cpu`/toolchain — no
separate int-only/float variant artifacts are produced.

To exercise the staticlib pipeline without cutting a release, run the
[`Static lib dry-run`](../.github/workflows/staticlib-dryrun.yml)
workflow from the Actions tab. It cross-compiles + smokes all three
targets, packages each into an SDK tarball with the matching flags,
verifies the produced `manifest.json` actually reports the requested
features, and uploads the archives as workflow artefacts (14 day
retention).

## Float (F32/F16) capability manifest

Published archives are **supersets**: each `target_cpu`/toolchain
combination ships exactly one canonical archive containing every float
implementation qualified for it, plus the always-present integer
kernels. Today F32 is enabled on every prebuilt combination; F16 is
enabled only on `cortex-m55` (the only target with qualified MVEF/toolchain
support, mirroring the Zephyr `ARMV8_1_M_MVEF` Kconfig restriction).
`scripts/build_staticlib.sh` fails fast (`exit 2`) if an unsupported
combination is requested (e.g. `--enable-f16` on `cortex-m4`).

**How capabilities are recorded.** `build_staticlib.sh` writes a
`<archive>.buildconfig.json` sidecar next to the `.a` recording exactly
which `ARM_NN_ENABLE_F32`/`ARM_NN_ENABLE_F16`/
`CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY` flags were actually compiled
in — this is the single source of truth. `scripts/build_sdk_tarball.sh`
reads that sidecar (rather than trusting caller-supplied flags) and
renders it into the packaged `manifest.json`:

```json
{
  "schema_version": 2,
  "target_cpu": "cortex-m55",
  "compiler_id": "GNU",
  "features": {
    "integer": true,
    "f32": true,
    "f16": true,
    "requantize_inline_asm": false,
    "verified": true
  }
}
```

`features.verified` is `true` when the values came from the trusted
`buildconfig.json` sidecar, and `false` if `build_sdk_tarball.sh` had
to fall back to caller-supplied CLI flags because no sidecar was found
(e.g. packaging an archive built outside this repo's own pipeline) —
that fallback is unverified and only intended for manual/local use. If
both a sidecar and CLI flags are given and they disagree,
`build_sdk_tarball.sh` aborts rather than silently trusting one over
the other.

**Older manifests.** Any manifest with `schema_version < 2`, or missing
the `features` object entirely, is conservatively interpreted as
**integer-only** (`f32=false`, `f16=false`) by every consumer — NSX,
Zephyr, and the `find_package` CMake config. This is a deliberate,
documented fallback: it never lets an old manifest silently grant a
float capability it never claimed to have.

**Capability vs. request.** The manifest describes what the archive
*contains*; `ARM_NN_ENABLE_F32`/`F16` on a consumer target describes
what that consumer *requests*. Exposing a feature to consumer code that
the linked archive does not actually implement produces undefined
references at final link time (the original bug this system fixes) —
so every prebuilt integration path now validates the request against
the manifest at configure time and fails early with a clear message
instead of deferring to a confusing link-time error.

**How macros reach consumer code, per integration:**

| Consumer | Capability variables | Request variables | Policy |
|----------|----------------------|--------------------|--------|
| `find_package(ns-cmsis-nn)` | `NS_CMSIS_NN_HAS_F32`, `NS_CMSIS_NN_HAS_F16`, `NS_CMSIS_NN_HAS_REQUANTIZE_INLINE_ASM` | *(none — no pre-existing opt-in API)* | Automatic: every capability the manifest reports is unconditionally forwarded onto the `cmsis-nn` imported target's `INTERFACE_COMPILE_DEFINITIONS` as `ARM_NN_ENABLE_F32`/`F16=1`. |
| NSX (`nsx/CMakeLists.txt`) | discovered via `NSX_CMSIS_NN_MANIFEST` → `<lib_dir>/manifest.json` → `<lib_dir>/../manifest.json` → none | `NSX_CMSIS_NN_ENABLE_F32`/`F16` (default `OFF`) | Opt-in preserved: only forwards what's requested, but `FATAL_ERROR`s if the manifest doesn't report that capability. If no manifest is discoverable, the request is honored as before but logged as **unverified**. |
| Zephyr (`zephyr/CMakeLists.txt`) | `${_ns_sdk}/manifest.json` (fixed tarball layout, no discovery needed) | `CONFIG_NS_CMSIS_NN_ENABLE_F32`/`F16` | Same opt-in + validate + fail-fast policy as NSX; existing FP16 arch/toolchain `Kconfig` restrictions (e.g. requires MVEF) are preserved unchanged. |

**Inspecting a package's capabilities** — no CMake configure required:

```sh
tar xOf libns-cmsis-nn-cortex-m55-<version>-sdk.tar.gz manifest.json | jq .features
```

**Archive size vs. firmware size.** A superset archive is larger on
disk than an int-only archive, but static linking only pulls in the
`.o` members a consumer's final link actually references — an
integer-only consumer linking against the F32/F16-enabled superset
archive does not pay for the unused float object code in its firmware
image.

## Why three group-id spellings?


The standalone CMake `option()` names are the historical originals
(`BASICMATHSNN`, `STRIDEDSLICE`, `SVDF`). The SSoT uses the lowercase
versions as group ids (`basicmath`, `stridedslice`, `svd`). The Zephyr
Kconfig symbols mirror the standalone `option()` names for consistency
across `prj.conf` / `cmake -D…` workflows. Old Kconfig spellings
(`BASICMATH`, `STRIDED_SLICE`, `SVD`) are kept as deprecated `select`
aliases for one release cycle.
