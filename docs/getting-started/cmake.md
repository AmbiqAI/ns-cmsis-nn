# CMake (`find_package`)

heliaCORE ships a relocatable CMake config package inside each SDK
tarball. After extraction, `find_package(ns-cmsis-nn)` exposes the
target `nsx::cmsis_nn`, plus a set of cache variables describing the
arch and toolchain the archive was built with.

## 1. Download the SDK tarball

Pick the tarball matching your target CPU:

```bash
VERSION=7.25.0
CPU=cortex-m4   # or cortex-m0, cortex-m55
curl -LO https://github.com/AmbiqAI/ns-cmsis-nn/releases/download/v${VERSION}/ns-cmsis-nn-${CPU}-gcc-${VERSION}.tar.gz
curl -LO https://github.com/AmbiqAI/ns-cmsis-nn/releases/download/v${VERSION}/ns-cmsis-nn-${CPU}-gcc-${VERSION}.tar.gz.sha256
sha256sum -c ns-cmsis-nn-${CPU}-gcc-${VERSION}.tar.gz.sha256
tar -xzf ns-cmsis-nn-${CPU}-gcc-${VERSION}.tar.gz -C third_party/
```

## 2. Wire it into your CMake project

```cmake
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/third_party/ns-cmsis-nn-cortex-m4-gcc-7.25.0")

find_package(ns-cmsis-nn 7.25.0 REQUIRED CONFIG)

add_executable(my_firmware main.c)
target_link_libraries(my_firmware PRIVATE nsx::cmsis_nn)
```

That's it. `nsx::cmsis_nn` is an `IMPORTED STATIC` target that already
carries the right `INTERFACE_INCLUDE_DIRECTORIES`.

## 3. Configure-time guardrails

`find_package` will **fail fast** if your project's compile flags don't
match the archive:

- **CPU mismatch** — if your `-mcpu=` differs from the archive's, you
  get a `FATAL_ERROR` with the expected vs actual value.
- **Compiler ID mismatch** — the archive records the `CMAKE_C_COMPILER_ID`
  it was built with (e.g. `GNU` for GCC, `ARMClang` for Arm Compiler 6).
  The packaged CMake config rejects a different consumer compiler ID as a
  conservative provenance check. ATfE can still link a GCC-built C archive when
  the Arm embedded ABI settings match.

This is intentional for packaged SDK consumption: a `.a` built for the wrong CPU
or float ABI can fail only after it reaches a device, and compiler provenance is
part of release qualification. For performance-oriented ATfE builds, building
heliaCORE from source lets ATfE optimize the kernels directly.

## 4. Verify the integration

After configuration, confirm CMake imported the heliaCORE package and selected
the expected archive:

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$PWD/third_party/ns-cmsis-nn-cortex-m4-gcc-7.25.0"
cmake --build build --verbose
```

In the configure or verbose build output, look for:

- `nsx::cmsis_nn` in the link line.
- The extracted package's `include/` directory in the compiler include paths.
- The package's `lib/libns-cmsis-nn.a` in the final link command.

If CMake reports a CPU mismatch, switch to the matching release artifact. If it
reports a compiler mismatch and you want ATfE to optimize the kernels, build
heliaCORE from source with your project's toolchain.

## Reference

- Manifest: `ns-cmsis-nn-manifest.json` (inside each tarball) describes
  the exact toolchain and build flags. See [Toolchain Pinning](../guides/toolchains.md).
- Smoke test: a minimal consumer project lives in
  [`cmake/tests/find_package/`](https://github.com/AmbiqAI/ns-cmsis-nn/tree/main/cmake/tests/find_package).
