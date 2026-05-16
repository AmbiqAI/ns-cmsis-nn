# Toolchain Pinning

For new source builds and HELIA integrations, prefer **Arm Toolchain for
Embedded (ATfE)**. ATfE is Arm's open LLVM-based embedded toolchain, maps to the
Clang compiler family, and is the preferred choice for Cortex-M55/MVE work where
compiler code generation can materially affect kernel cycles. On representative
optimized kernels, ATfE can reduce cycle counts by roughly 20% compared with GCC,
depending on model shape, data layout, and enabled accelerator path.

The prebuilt artifacts heliaCORE ships today are still **toolchain-stamped** GCC
packages: the file name, the manifest, and the CMake config all record the
compiler ID and version they were built with. Use those prebuilts when you want
the released GCC package. Build from source with ATfE when you want the
recommended performance-oriented compiler path.

## Recommended toolchain

| Use case | Recommended choice | Notes |
|---|---|---|
| New Ambiq/HELIA source builds | Arm Toolchain for Embedded (ATfE) | LLVM/Clang based, open source, and preferred for Cortex-M55/MVE optimization work. |
| Released prebuilt tarballs | GNU Arm Embedded package shipped by the release | Current release assets are named and validated as GCC packages. Do not mix a GCC-built `.a` with a Clang/ATfE application. |
| Existing GCC-based firmware | GNU Arm Embedded | Keep using GCC when that is the qualified project compiler, but treat ATfE as the performance-forward migration path. |

## What gets pinned

Each tarball contains `ns-cmsis-nn-manifest.json`:

```json
{
  "schema_version": 1,
  "name": "ns-cmsis-nn",
  "version": "7.25.0",
  "target_cpu": "cortex-m4",
  "arch_flags": "-mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard",
  "toolchain": {
    "id": "gcc",
    "full_id": "arm-none-eabi-gcc",
    "compiler_id": "GNU",
    "version": "13.2.1"
  }
}
```

And `ns-cmsis-nn-config.cmake` records the same identity so
`find_package` can validate against the consumer's project.

## What `find_package` validates

At configure time, `find_package(ns-cmsis-nn)` checks:

| Check                | Behavior on mismatch                              |
|----------------------|---------------------------------------------------|
| `-mcpu` flag         | `FATAL_ERROR` listing expected vs actual.         |
| `CMAKE_C_COMPILER_ID`| `FATAL_ERROR` listing expected vs actual.         |
| Version (if pinned)  | `FATAL_ERROR` (standard CMake VERSION semantics). |

## CMake `COMPILER_ID` reference

heliaCORE follows the canonical CMake names:

| Toolchain                      | `CMAKE_C_COMPILER_ID` |
|--------------------------------|-----------------------|
| Arm Toolchain for Embedded (ATfE, LLVM/Clang) | `Clang` |
| GNU Arm Embedded (GCC)         | `GNU`                 |
| Arm Compiler 6 (armclang)      | `ARMClang`            |
| Arm Compiler 5 (armcc, EOL)    | `ARMCC`               |
| Other embedded LLVM/Clang builds | `Clang`             |

The prebuilts today are **GCC only**. Building from source under ATfE/Clang or
ARMClang works when your project supplies that toolchain, but mixing a GCC-built
`.a` into an ATfE, Clang, or ARMClang link does not. `find_package` is strict so
this mismatch fails during configure instead of becoming a device-side runtime
fault.

## Why so strict?

A static archive baked with `-mcpu=cortex-m4 -mfloat-abi=hard` that
gets linked into a `cortex-m0+` image will silently produce a binary
that hard-faults at the first FP instruction. The strict check trades
a configure-time error for a hard-to-debug runtime crash on a device
in a customer's hands.
