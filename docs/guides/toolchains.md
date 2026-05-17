# Toolchain Pinning

For new source builds and HELIA integrations, prefer **Arm Toolchain for
Embedded (ATfE)**. ATfE is Arm's open LLVM-based embedded toolchain, maps to the
Clang compiler family, and is the preferred choice for Cortex-M55/MVE work where
compiler code generation can materially affect kernel cycles. On representative
optimized kernels, ATfE can reduce cycle counts by roughly 20% compared with GCC,
depending on model shape, data layout, and enabled accelerator path.

The SDK tarballs heliaCORE ships are **toolchain-stamped** packages: the file
name, manifest, and CMake config all record the compiler ID and version they
were built with. Pick the artifact matching your project compiler when you want
prebuilt kernels. Build from source when you need local flags, qualification, or
toolchain versions that differ from the release.

## Recommended toolchain

| Use case | Recommended choice | Notes |
|---|---|---|
| New Ambiq/HELIA source builds | Arm Toolchain for Embedded (ATfE) | LLVM/Clang based, open source, and preferred for Cortex-M55/MVE optimization work. |
| Released prebuilt tarballs | Match your project compiler: `atfe`, `armclang`, or `gcc` | The CMake package validates compiler ID and CPU flags against the selected archive. |
| Existing GCC-based firmware | GNU Arm Embedded | Keep using GCC when that is the qualified project compiler, but treat ATfE as the performance-forward migration path. |

## What gets pinned

Each tarball contains `manifest.json`:

```json
{
  "schema_version": 1,
  "package": "ns-cmsis-nn",
  "version": "7.25.0",
  "target_cpu": "cortex-m4",
  "toolchain": {
    "id": "atfe",
    "full_id": "arm-toolchain-for-embedded",
    "compiler_id": "Clang",
    "version": "19.1.5"
  },
  "abi": {
    "arch_flags": "-mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard"
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
| `CMAKE_C_COMPILER_ID`| `FATAL_ERROR` for stamped SDK tarballs when the consumer compiler ID differs from the recorded build compiler. This is a conservative provenance check, not a statement that ATfE and GCC objects are ABI-incompatible. |
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

The release publishes SDK tarballs for `gcc`, `atfe`, and `armclang`. The
CMSIS-Pack `Prebuilt` Cvariant remains GCC-built; use the `Source` Cvariant when
your pack-based project needs ATfE or armclang to compile the kernels.

## Why so strict?

A static archive baked with `-mcpu=cortex-m4 -mfloat-abi=hard` that gets linked
into a `cortex-m0+` image can silently produce a binary that hard-faults at the
first unsupported instruction. The strict CPU check trades a configure-time error
for a hard-to-debug runtime crash on a device in a customer's hands. The compiler
ID check is about package provenance and qualification. If you intentionally mix
objects from different Arm embedded toolchains, validate the CPU/FPU, float ABI,
and calling convention as part of your integration.
