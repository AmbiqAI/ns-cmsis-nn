# Toolchain Pinning

For new source builds and HELIA integrations, prefer **Arm Toolchain for
Embedded (ATfE)**. ATfE is Arm's open LLVM-based embedded toolchain, maps to the
Clang compiler family, and is the preferred choice for Cortex-M55/MVE work where
compiler code generation can materially affect kernel cycles. On representative
optimized kernels, ATfE can reduce cycle counts by roughly 20% compared with GCC,
depending on model shape, data layout, and enabled accelerator path.

The prebuilt artifacts heliaCORE ships today are still **toolchain-stamped** GCC
packages: the file name, the manifest, and the CMake config all record the
compiler ID and version they were built with. A GCC-built C static archive can
be linked into an ATfE application when the target CPU, FPU, float ABI, and AAPCS
calling convention match. Build from source with ATfE when you want ATfE to
optimize the heliaCORE kernels themselves.

## Recommended toolchain

| Use case | Recommended choice | Notes |
|---|---|---|
| New Ambiq/HELIA source builds | Arm Toolchain for Embedded (ATfE) | LLVM/Clang based, open source, and preferred for Cortex-M55/MVE optimization work. |
| Released prebuilt tarballs | GNU Arm Embedded package shipped by the release | Current release assets are named and validated as GCC packages. ATfE can link the archive when ABI flags match, but the CMake package records the GCC provenance. |
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

The prebuilts today are **GCC-stamped**. Building from source under ATfE/Clang
or ARMClang works when your project supplies that toolchain. Linking the
GCC-built archive into an ATfE application can also be valid when the archive and
application use compatible Arm embedded ABI settings. The CMake config's
compiler-ID check is intentionally conservative because it cannot prove that the
rest of the application was built with compatible options.

## Why so strict?

A static archive baked with `-mcpu=cortex-m4 -mfloat-abi=hard` that gets linked
into a `cortex-m0+` image can silently produce a binary that hard-faults at the
first unsupported instruction. The strict CPU check trades a configure-time error
for a hard-to-debug runtime crash on a device in a customer's hands. The compiler
ID check is about package provenance and qualification: if you intentionally mix
ATfE application code with a GCC-built heliaCORE archive, validate the CPU/FPU,
float ABI, and calling convention as part of your integration.
