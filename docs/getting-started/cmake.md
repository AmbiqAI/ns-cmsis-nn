# CMake (`find_package`)

heliaCORE ships a relocatable CMake config package inside each SDK
tarball. After extraction, `find_package(ns-cmsis-nn)` exposes the
target `ns::cmsis-nn`, plus ordinary CMake variables describing the
architecture and toolchain the archive was built with. These are package metadata,
not cache settings for changing the prebuilt archive.

## 1. Download the SDK tarball

Pick the tarball matching your target CPU:

```bash
VERSION=7.36.0 # x-release-please-version
CPU=cortex-m4   # or cortex-m0, cortex-m55
TOOLCHAIN=atfe # or gcc, armclang
curl -LO https://github.com/AmbiqAI/ns-cmsis-nn/releases/download/v${VERSION}/ns-cmsis-nn-${CPU}-${TOOLCHAIN}-${VERSION}.tar.gz
curl -LO https://github.com/AmbiqAI/ns-cmsis-nn/releases/download/v${VERSION}/ns-cmsis-nn-${CPU}-${TOOLCHAIN}-${VERSION}.tar.gz.sha256
sha256sum -c ns-cmsis-nn-${CPU}-${TOOLCHAIN}-${VERSION}.tar.gz.sha256
mkdir -p third_party
tar -xzf ns-cmsis-nn-${CPU}-${TOOLCHAIN}-${VERSION}.tar.gz -C third_party/
```

## 2. Wire it into your CMake project

```cmake
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/third_party/ns-cmsis-nn-cortex-m4-atfe-7.36.0") # x-release-please-version

find_package(ns-cmsis-nn 7.36.0 REQUIRED CONFIG) # x-release-please-version

add_executable(my_firmware main.c)
target_link_libraries(my_firmware PRIVATE ns::cmsis-nn)
```

That's it. `ns::cmsis-nn` is an `IMPORTED STATIC` target that already
carries the right `INTERFACE_INCLUDE_DIRECTORIES`.

## 3. Configure-time guardrails

The package checks two aspects of the consumer configuration:

- **CPU mismatch:** when `CMAKE_C_FLAGS` contains `-mcpu=`, a value that
  differs from the archive's CPU produces a `FATAL_ERROR`. This check does
  not inspect per-target or configuration-specific compiler options.
- **Compiler ID mismatch:** when both compiler IDs are known, the package
  rejects a consumer `CMAKE_C_COMPILER_ID` that differs from the recorded
  ID (for example, `GNU` or `ARMClang`). This is a provenance check, not a
  compiler-version comparison.

These checks do not validate the FPU or float ABI. Compare your firmware's
verbose compiler command with the package's `manifest.json`: compiler
identity/version, CPU/FPU flags, and float ABI must fit your project.
A successful configure alone does not establish compatibility. Choose a
matching SDK tarball, or build heliaCORE from source when you need different
compiler flags.

## 4. Verify the integration

After configuration, confirm CMake imported the heliaCORE package and selected
the expected archive:

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$PWD/third_party/ns-cmsis-nn-cortex-m4-atfe-7.36.0" # x-release-please-version
cmake --build build --verbose
```

In the configure or verbose build output, look for:

- The extracted package's `include/` directory in the compiler include paths.
- The package's `lib/libns-cmsis-nn.a` in the final link command.

If CMake reports a CPU mismatch, switch to the matching release artifact. If it
reports a compiler mismatch and you want ATfE to optimize the kernels, build
heliaCORE from source with your project's toolchain.

## Reference

- Manifest: `manifest.json` (inside each tarball) describes
  the exact toolchain and build flags. See [Toolchain Pinning](../guides/toolchains.md).
- Smoke test: a minimal consumer project lives in
  [`cmake/tests/find_package/`](https://github.com/AmbiqAI/ns-cmsis-nn/tree/main/cmake/tests/find_package).
