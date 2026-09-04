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

## `float16` needs a binutils 2.43 assembler

The `float16` kernels are built from the Q-register form of the MVE
`VCVTB`/`VCVTT.F16<->F32` conversions. Assemblers from binutils releases before
2.43 encode that form with the wrong register numbers: low operands read and
write the wrong registers and return silently wrong results, and higher operands
overflow the register field into a word that is architecturally UNDEFINED and
faults the first time it executes. The compiler is not at fault, the assembler
is, so nothing the compiler emits and nothing the linker resolves can reveal it.

A `float16` MVE build therefore requires an assembler from binutils 2.43 or
newer. Arm ships one from **Arm GNU Toolchain 14.2.Rel1**. This applies to
`float16` only: the integer kernels and the `float32` kernels never touch those
conversions, so **GCC 13.x remains fully supported for `s4`/`s8`/`s16` and
`float32` builds with `ARM_NN_ENABLE_F16` off.** With `ARM_NN_ENABLE_F16` on the
refusal is per translation unit rather than per kernel, so every translation
unit that includes the `float16` support header refuses under GCC 13 without a
verified assembler, the integer ones included.

The CMake build assembles one witness instruction at configure time and compares
the bytes, so it validates the assembler that is actually in use rather than
trusting a version number. It assembles with the flags the library target will
really compile with: the target's own compile options, definitions and include
directories, plus the ones it picks up from the interface targets it links. So a
project that carries `-mcpu` on a board flags target is checked the same as one
that sets `CMAKE_C_FLAGS`. Options that inject a header into every translation
unit, `-include` and `-imacros`, are dropped before the witness is assembled and
named in a configure message: they feed C declarations to an assembly file and
cannot change how an instruction encodes. On a broken assembler it fails the
configure with the remediation below. `ARM_NN_ENABLE_F16=OFF` skips the check
entirely.

### The check that cannot be bypassed

The configure-time probe cannot see everything. An architecture flag, or an
include directory a flag depends on, that exists only inside a CMake generator
expression has no value at configure time; a flag added to the library target
after the directory that attached the sources has finished is not there yet when
the probe reads the target; and a build that never runs CMake at all gets no
probe.

Only a flag the probe cannot read at all can stop the witness from assembling,
such as an include directory hidden inside a generator expression that a flag on
the witness command line depends on. When that happens the probe refuses the
configure and prints the flags it used and the assembler's own error, so an
unmeasured assembler is never taken for a good one. Setting
`ARM_NN_SKIP_GAS_F16_PROBE=ON` is the only way past that refusal.

So the probe is not the only guard. When it measures a good assembler it
defines `ARM_NN_GAS_F16_VERIFIED=1` on the target it checked, and
`Include/arm_nnsupportfunctions_flt.h` refuses to compile a `float16` MVE
translation unit on GCC 13 or older without that definition. The shapes the
probe cannot see therefore fail at compile time with a message naming the fix,
rather than building the mis-encoded conversions in silence. Where the probe
does apply it stays authoritative: it fails the configure on any GCC, including
14.2 and newer with a `-B` that points at a broken assembler.

### Building without CMake

The CMSIS-Pack `Source` Cvariant and `module.mk` compile the sources directly,
so nothing probes the assembler for them.

- On **Arm GNU 14.2.Rel1 or newer**, nothing to do. One residual: outside CMake
  the guard keys on the compiler major, so a GCC 14 or newer driver over a
  binutils below 2.43 is not caught. That pair only exists if you assembled it
  yourself; check `as --version` if you did. Under CMake the probe catches it,
  or refuses to configure when your flags keep it from assembling its witness.
- On **GCC 13.x or older**, pass both `-B<dir>/` for a binutils 2.43 or newer
  `arm-none-eabi` assembler and `-DARM_NN_GAS_F16_VERIFIED=1`. The `-B` is what
  fixes the encoding; the define is how you tell the library you did it.

```sh
arm-none-eabi-gcc -mcpu=cortex-m55 -mfloat-abi=hard -DARM_NN_ENABLE_F16=1 \
  -IInclude -IInclude/Internal \
  -B/opt/arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-eabi/arm-none-eabi/bin/ \
  -DARM_NN_GAS_F16_VERIFIED=1 \
  -c Source/ActivationFunctions/arm_hard_swish_f16.c
```

Defining `ARM_NN_GAS_F16_VERIFIED` without the `-B` builds the mis-encoded
conversions. The define asserts that you checked; it does not check anything.

### Keeping a GCC 13.x compiler

If GCC 13.x is your qualified compiler, you do not have to move it. The compiler
and the assembler are separate binaries, and `-B` points the driver at a
different one. Give it the directory that holds the **unprefixed** `as`, which in
an Arm GNU install is `<install root>/arm-none-eabi/bin/`, and keep the trailing
slash:

```sh
CFLAGS="-B/opt/arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-eabi/arm-none-eabi/bin/" \
  cmake -S . -B build \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain/arm-none-eabi-gcc.cmake \
  -DNS_CMSIS_NN_TARGET_CPU=cortex-m55 \
  -DARM_NN_ENABLE_F16=ON
```

Set it through `CFLAGS` on a fresh build directory, or append it to
`CMAKE_C_FLAGS` alongside the architecture flags. A bare
`-DCMAKE_C_FLAGS="-B..."` replaces the flags the toolchain file supplies.

`ARM_NN_SKIP_GAS_F16_PROBE=ON` downgrades the configure failure to a warning and
defines `ARM_NN_GAS_F16_VERIFIED=1` on the target, which lifts the compile-time
guard as well; otherwise the build would still stop. It does the same when the
probe cannot assemble its witness at all, so a build whose flags defeat the
measurement still has a way past. It is a last resort for a build that cannot
be changed any other way: the library it produces contains the mis-encoded
conversions, and the `float16` kernels that use them return wrong results or
fault.

See [#427](https://github.com/AmbiqAI/ns-cmsis-nn/issues/427).

## What gets pinned

Each tarball contains a `manifest.json` recording the identity of the archive.
Its `"version"` field is the heliaCORE release the archive was built from:
tarballs for this release carry `"version": "7.31.0"`. <!-- x-release-please-version -->

The example below is trimmed. The `"version"` field is left out on purpose, so
the block stays valid JSON that no release has to edit; the `"features"`,
`"library"` and `"built_at"` blocks a real manifest also carries are omitted for
brevity. `scripts/build_sdk_tarball.sh` renders the manifest and is the
authority on its exact shape, and `Documentation/build.md` describes the
`"features"` block and how consumers must treat an older `schema_version`.

```json
{
  "schema_version": 2,
  "package": "ns-cmsis-nn",
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

Release CI smoke-links the GCC and ATfE static libraries. The armclang artifacts
are compiled and symbol-verified in CI, with full Arm linker smoke coverage to be
added once the armlink invocation is qualified for this package.

## Why so strict?

A static archive baked with `-mcpu=cortex-m4 -mfloat-abi=hard` that gets linked
into a `cortex-m0+` image can silently produce a binary that hard-faults at the
first unsupported instruction. The strict CPU check trades a configure-time error
for a hard-to-debug runtime crash on a device in a customer's hands. The compiler
ID check is about package provenance and qualification. If you intentionally mix
objects from different Arm embedded toolchains, validate the CPU/FPU, float ABI,
and calling convention as part of your integration.
