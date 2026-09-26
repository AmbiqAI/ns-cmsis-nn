# neuralSPOT-X

If you're already on [neuralSPOT-X (NSX)](https://github.com/AmbiqAI/neuralSPOT),
you don't need to integrate heliaCORE — NSX already does it for you.

## What NSX provides

- A CMake target (`nsx::cmsis_nn` via NSX's graph) that resolves to the
  heliaCORE-built archive matched to your NSX target arch.
- Board-flag propagation through the `NSX_BOARD_FLAGS_TARGET` interface
  library — the kernels link against the same `-mcpu`/`-mfpu` flags
  the rest of your firmware uses.
- Automatic linkage of `nsx::cmsis_nn` into your model runtime
  (heliaRT, LiteRT-for-Micro, etc.).

## Pinning a heliaCORE version inside NSX

NSX projects pin heliaCORE the same way they pin any other dependency:
via the NSX manifest or a CMake cache variable. Refer to your NSX
project's docs for the exact knob — typically:

```cmake
set(NSX_CMSIS_NN_VERSION "7.36.0" CACHE STRING "heliaCORE release to use") # x-release-please-version
```

### Source-build settings

Standalone CMake and NSX share configuration logic, but retain their existing
inputs. Set the appropriate options before adding the module:

| Setting | Standalone CMake | NSX |
| --- | --- | --- |
| Optimization | `CMSIS_OPTIMIZATION_LEVEL` | `NSX_CMSIS_NN_OPTIMIZATION` |
| Inline requantize assembly | `CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY` | `NSX_CMSIS_NN_USE_REQUANTIZE_INLINE_ASM` |
| Float kernels | `ARM_NN_ENABLE_F32/F16` | `ARM_NN_ENABLE_F32/F16` |

Optimization defaults to `-Ofast`; inline assembly and both float widths default
to off. The NSX-specific settings are not aliases for standalone options: sibling
libraries can use different settings. `NSX_CMSIS_NN_GROUPS` preserves the supplied
group order; `ALL` or an empty list selects all groups. These source-build options
do not rebuild or change a prebuilt archive.

### Arm Compiler application setup

When using the repository's armclang toolchain, the application owns CMake policy
`CMP0123`. Set it after `cmake_minimum_required()` and before the application's
`project()` enables C or C++:

```cmake
cmake_minimum_required(VERSION 3.19)
if(POLICY CMP0123)
  cmake_policy(SET CMP0123 NEW)
endif()
project(my_firmware C CXX)
# Add the SDK and heliaCORE module after project().
```

Without this setup, older policy behavior adds another `-mcpu` option during
compiler initialization. Including heliaCORE afterward cannot undo that earlier
step. The standalone heliaCORE project already sets the policy; applications
that include the NSX module directly must set it themselves.

To check the policy with the installed Arm Compiler, run the repository's wiring
fixture from the repository root, using separate build directories:

```sh
cmake -S cmake/tests/nsx_wiring -B build/nsx-armclang-new \
  -DCASE=source_subset \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/cmake/toolchain/armclang.cmake" \
  -DNS_CMSIS_NN_TOOLCHAIN_ROOT="$ARMCLANG_ROOT" \
  -DNS_CMSIS_NN_TARGET_CPU=cortex-m55 \
  -DCMAKE_POLICY_DEFAULT_CMP0123=NEW \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

Set `ARMCLANG_ROOT` to the installation containing `bin/armclang` and `bin/armar`.
Compare with a fresh `build/nsx-armclang-old` configured with
`-DCMAKE_POLICY_DEFAULT_CMP0123=OLD`: inspect `compile_commands.json` for one
`-mcpu=cortex-m55` under NEW versus two under OLD. Leaving the policy unset also
produces the CMP0123 developer warning on affected CMake versions. This fixture
uses a modeled board-flags target; it checks compiler setup, not a complete NSX
firmware build.

## Verifying the link

NSX's CI includes a smoke test that links a tiny consumer against
`nsx::cmsis_nn` + the NSX board flags target, the same way real NSX
firmware does. The test config lives at
[`cmake/tests/nsx_prebuilt_real_link/`](https://github.com/AmbiqAI/ns-cmsis-nn/tree/main/cmake/tests/nsx_prebuilt_real_link)
inside this repo and runs on every heliaCORE PR.

For an application build, verify the resolved CMake target and version before
debugging model-level failures:

- Confirm the NSX dependency manifest or cache value resolves to the intended
  heliaCORE release.
- Check the verbose link line for `libns-cmsis-nn.a` or the source-built
  heliaCORE target.
- Confirm the NSX board flags target and heliaCORE package agree on `-mcpu`,
  FPU, and ABI settings.

If you ever see "undefined reference" to a CMSIS-NN symbol from an NSX
project, the most common culprits are:

- A heliaCORE version pinned older than the NSX commit you're on.
- A board-flag mismatch — NSX is targeting a different `-mcpu` than
  the heliaCORE archive was built for. `find_package` will catch
  this; see [Toolchain Pinning](../guides/toolchains.md).
