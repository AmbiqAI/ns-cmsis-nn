# Toolchain Pinning

The prebuilt artifacts heliaCORE ships are **toolchain-stamped**: the
file name, the manifest, and the CMake config all record the compiler
ID and version they were built with.

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
| GNU Arm Embedded (GCC)         | `GNU`                 |
| Arm Compiler 6 (armclang)      | `ARMClang`            |
| Arm Compiler 5 (armcc, EOL)    | `ARMCC`               |
| LLVM/Clang (Embedded Clang)    | `Clang`               |

The prebuilts today are **GCC only**. Building from source under
ARMClang works; mixing a GCC-built `.a` into an ARMClang link does
not.

## Why so strict?

A static archive baked with `-mcpu=cortex-m4 -mfloat-abi=hard` that
gets linked into a `cortex-m0+` image will silently produce a binary
that hard-faults at the first FP instruction. The strict check trades
a configure-time error for a hard-to-debug runtime crash on a device
in a customer's hands.
