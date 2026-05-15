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
entirely and only exposes headers.

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
   table. Run with:

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

## Why three group-id spellings?

The standalone CMake `option()` names are the historical originals
(`BASICMATHSNN`, `STRIDEDSLICE`, `SVDF`). The SSoT uses the lowercase
versions as group ids (`basicmath`, `stridedslice`, `svd`). The Zephyr
Kconfig symbols mirror the standalone `option()` names for consistency
across `prj.conf` / `cmake -D…` workflows. Old Kconfig spellings
(`BASICMATH`, `STRIDED_SLICE`, `SVD`) are kept as deprecated `select`
aliases for one release cycle.
