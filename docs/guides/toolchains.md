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

## `float16` and the MVE half/single conversions

Three `float16` kernels convert between half and single precision:
`arm_nn_mean_f16` and `arm_reduce_sum_f16` widen to accumulate, and
`arm_softmax_f16` does both through an inlined exponential helper. The natural
instructions are the Q-register form of the MVE `VCVTB`/`VCVTT.F16<->F32`
conversions.

Assemblers from binutils releases before 2.43 encode that form with the wrong
register numbers: each Q operand is written with its D-register alias number, so
low operands read and write the wrong registers and return silently wrong
results, and operands above `q3` overflow the register field into a word that is
architecturally UNDEFINED and faults the first time it executes. The compiler is
not at fault: it emits correct assembly text on every release from 12.x to
15.x, so nothing the compiler diagnoses and nothing the linker resolves can
reveal this. An `objdump` of the same vintage is no help either: it mis-renders
the words the same way its assembler mis-encoded them, so only 2.43 or newer can
display the corruption.

Arm ships binutils 2.43 from **Arm GNU Toolchain 14.2.Rel1**. Releases below that
are still fully supported, for `float16` as well as for the integer and
`float32` kernels. The library neither requires the newer assembler nor refuses
to build on the older one.

### What the library does instead

`Include/Internal/arm_nn_vcvt_f16.h` defines four wrappers, one per conversion
direction, and every `float16` kernel that widens or narrows calls a wrapper
rather than naming an intrinsic. On an assembler that encodes the vector form
correctly the wrappers *are* the intrinsics, and the objects are byte-identical
to a build without them.

On an assembler that does not, each wrapper emits the scalar VFP form of the same
instruction, four of them per vector conversion, one per 32-bit lane. The `S`
registers of a `Q` register are consecutive, so the four scalar conversions cover
exactly the bits the one vector conversion would have. Those are ordinary VFP
mnemonics that every gas from 2.39 to 2.45 encodes identically.

The cost falls only on the affected assemblers, and only inside those three
kernels. Everything else in the library, `float16` included, is unchanged.

One behavioral difference is worth knowing about: the scalar form honors
`FPSCR.DN` for NaN payloads, while the vector form always returns the default
NaN. NaN stays NaN either way. Only NaN inputs are affected, only in their
payload bits, and only on an assembler that needs the scalar form.

The wrappers exist for as long as a GCC 13 release is in the support matrix;
their removal is tracked by
[#435](https://github.com/AmbiqAI/ns-cmsis-nn/issues/435).

### How the form is chosen

The preprocessor cannot see the assembler, so the CMake build asks it. At
configure time it compiles a witness and compares the bytes, so it measures the
assembler actually in use rather than trusting a version number.

It compiles with the flags the library target will really compile with: the
target's own compile options, definitions and include directories, plus the ones
it picks up from the interface targets it links. A project that carries `-mcpu`
on a board flags target is measured the same as one that sets `CMAKE_C_FLAGS`.
Options that inject a header into every translation unit, `-include` and
`-imacros`, are dropped before the witness is compiled, and named in a configure
message: no header can change how an instruction encodes, and the directory that
holds one is often carried in a generator expression the probe cannot read.

The witness is a C file whose body is a single top-level `asm`, not a `.S`. The
compiler opens every object it writes with `.arch`/`.fpu`/`.arch_extension`
directives derived from `-mcpu`, and those override what the driver forwards to
the assembler on the command line. A hand-written `.S` has no such prologue, so
a project naming both `-mcpu=cortex-m55` and an explicit `-mfpu=` that predates
MVE would assemble the kernels fine and fail on the witness. Going through the C
path puts the witness on the same footing as the kernels. One thing the C path cannot see: an assembler command line that those directives override, such as a stray `-Wa,-march` flag. Nothing in this repository is hand-written assembly, so every conversion the library emits goes through the same compiler prologue as the witness; a consumer that assembles its own MVE `.S` files gets no verdict about them from this probe.

The verdict is a compile definition on the target that was measured:

| Definition | Meaning |
|---|---|
| `ARM_NN_GAS_F16_VERIFIED=1` | the vector conversions encode correctly; the wrappers are the intrinsics |
| `ARM_NN_GAS_VCVT_F16_BROKEN=1` | they do not; the wrappers emit the scalar form |

Because the probe measures the assembler and not the compiler, it lands correctly
on a GCC 13 driver handed a newer assembler with `-B`, and equally on a GCC 14
driver paired by hand with an older binutils. Neither verdict stops a build.

`ARM_NN_ENABLE_F16=OFF` skips the check entirely.

### When the witness will not compile

The probe cannot see everything. An architecture flag, or an include directory a
flag depends on, that exists only inside a CMake generator expression has no
value at configure time; a flag added to the library target after the directory
that attached the sources has finished is not there yet when the probe reads the
target; and a build that never runs CMake at all gets no probe. In those shapes
the probe reports that it does not apply, and the header falls back to the guard
described under [Building without CMake](#building-without-cmake).

Separately, the witness can fail to compile, most often because a flag the
probe cannot read is missing, such as an include directory hidden inside a
generator expression that another flag depends on, or because a flag on the
target is invalid for this compiler. That is a hard configure error, printing the
flags used and the toolchain's own message, so an unmeasured assembler is never
quietly taken for a good one. `ARM_NN_SKIP_GAS_F16_PROBE=ON` downgrades it to a
warning; the build then defines neither verdict and falls back to the same
compiler-major guard.

### Building without CMake

The CMSIS-Pack `Source` Cvariant and `module.mk` compile the sources directly, so
nothing probes the assembler for them. With neither definition present, the
header keys on the compiler major instead: GCC 13 and older take the scalar form,
GCC 14 and newer take the vector form. Clang and armclang encode MVE themselves
and always take the vector form.

That proxy is right for every Arm GNU release as shipped, because each one pairs
a compiler with the binutils of its era. It is conservative in one direction, since a
GCC 13 driver over a 2.43 assembler gets a scalar form it did not need, and
blind in the other: **a GCC 14 or newer driver paired by hand with a binutils
below 2.43 is not caught.** That pair only exists if you assembled it yourself;
check `as --version` if you did, and pass `-DARM_NN_GAS_VCVT_F16_BROKEN=1` if it
is older than 2.43. Under CMake the probe catches this case.

### Getting the vector conversions on a GCC 13.x compiler

Nothing forces this, since the scalar form is correct, but it is one instruction
per conversion instead of four, and the compiler and the assembler are separate
binaries, so you do not have to move your qualified compiler to get it. `-B`
points the driver at a different assembler. Give it the directory holding the
**unprefixed** `as`, which in an Arm GNU install is
`<install root>/arm-none-eabi/bin/`, and keep the trailing slash:

```sh
CFLAGS="-B/opt/arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-eabi/arm-none-eabi/bin/" \
  cmake -S . -B build \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain/arm-none-eabi-gcc.cmake \
  -DNS_CMSIS_NN_TARGET_CPU=cortex-m55 \
  -DARM_NN_ENABLE_F16=ON
```

Set it through `CFLAGS` on a fresh build directory, or append it to
`CMAKE_C_FLAGS` alongside the architecture flags. A bare
`-DCMAKE_C_FLAGS="-B..."` replaces the flags the toolchain file supplies. The
probe will report that the conversions encode correctly, and the wrappers become
the intrinsics.

Outside CMake, pass the same `-B` together with `-DARM_NN_GAS_F16_VERIFIED=1`:

```sh
arm-none-eabi-gcc -mcpu=cortex-m55 -mfloat-abi=hard -DARM_NN_ENABLE_F16=1 \
  -IInclude -IInclude/Internal \
  -B/opt/arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-eabi/arm-none-eabi/bin/ \
  -DARM_NN_GAS_F16_VERIFIED=1 \
  -c Source/BasicMathFunctions/arm_nn_mean_f16.c
```

Defining `ARM_NN_GAS_F16_VERIFIED` without the `-B` builds the mis-encoded
conversions. The define asserts that you checked; it does not check anything.

### The other mis-encoded family

A sweep of the MVE instruction space across binutils 2.39 to 2.45 found one other
family that encodes differently between releases: the saturating narrowing shifts
`VQSHRN`/`VQSHRUN`, which gas below 2.43 assembles as their rounding variants.
No kernel in this library uses them. The configure probe measures them alongside
the conversions and defines `ARM_NN_GAS_VQSHRN_BROKEN=1` when they are affected,
and a pre-commit lint refuses any use of those intrinsics in `Source/` or
`Include/` until there is a wrapper for them.
See [#437](https://github.com/AmbiqAI/ns-cmsis-nn/issues/437).

See [#427](https://github.com/AmbiqAI/ns-cmsis-nn/issues/427) for the analysis
this section summarizes.

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
