# Agent guidance for ns-cmsis-nn

Ambiq's optimized fork of Arm CMSIS-NN targeting Cortex-M (Apollo SoCs). Kernels
are C with optional Helium (MVE / M-Profile Vector Extension) SIMD paths.

## SIMD: this repo uses MVE (Helium), NOT NEON

Cortex-M55-class targets implement **MVE** (`arm_mve.h`), not Armv8-A **NEON**
(`arm_neon.h`). Many intrinsics look nearly identical between the two sets but
are NOT interchangeable — a NEON intrinsic will fail to compile for an MVE
target or, worse, silently compile under a host/test configuration and then
break the embedded build. When writing SIMD paths:

- Copy the idiom from an existing kernel in this repo (e.g.
  `Source/BasicMathFunctions/arm_elementwise_add_f32.c`,
  `Source/NNSupportFunctions/arm_nn_lstm_step_f32.c`) rather than writing
  intrinsics from memory.
- The MVE loop shape is **tail-predicated**: `vctp32q(n - i)` / `vctp16q(...)`
  produces an `mve_pred16_t`, loads/stores use predicated forms
  (`vld1q_z`, `vstrwq_p` / `vstrhq_p`), and the loop steps by 4 (f32) or
  8 (f16) with no scalar tail loop. NEON has none of this.
- MVE selection is `vpselq` on an `mve_pred16_t`; NEON's lookalike `vbslq`
  (bitwise select on a vector mask) does not exist in MVE. Comparisons like
  `vcmpgeq` return predicates in MVE, vectors in NEON.
- Guard macros: `ARM_MATH_MVEF` (f32), `ARM_MATH_MVE_FLOAT16` (f16),
  `ARM_MATH_MVEI` (integer) — always with `&& !defined(ARM_MATH_AUTOVECTORIZE)`.
- If an intrinsic you want isn't used anywhere in `Source/` or
  `Include/Internal/`, treat that as a red flag and verify it exists in
  `arm_mve.h` before using it.

## Adding a kernel: three build manifests + header

A new source file must be wired in **all** of these, or some consumer build
silently omits it (the unit-test build uses #2, Zephyr/NSX use #2, packs use #3):

1. `Source/<Group>Functions/CMakeLists.txt` — float files are listed
   explicitly under `if (ARM_NN_ENABLE_F32/F16)`; the globs only match
   `*_s8*/_s16*/...` patterns.
2. `cmake/ns_cmsis_nn.cmake` — the SSoT per-group manifest
   (`_ns_cmsis_nn_group_def`). This is the authoritative one.
3. `Ambiq.NS-CMSIS-NN.pdsc` — one `<file>` entry per source, alphabetical.
   `scripts/check_pdsc.py` validates it against `git ls-files`, so stage new
   files (`git add`) before running it.

Public float declarations go in `Include/arm_nnfunctions_flt.h` — the f32
section and f16 section are separate `#if ARM_NN_ENABLE_*` blocks; put each
declaration in the right one.

## Float unit tests

Float suites live in `Tests/UnitTest/TestCases/test_arm_<kernel>/` with
checked-in numpy-generated golden data (`<name>_data.h`), a test `.c`, a
`Unity/unity_test_*.c` wrapper, and a CMakeLists referencing a
`Unity/TestRunner/*_runner.c` that is generated at configure time by
`unittest_targets.py` (needs `pyserial` + `termcolor`). Register the suite in
`Tests/UnitTest/CMakeLists.txt` inside the matching `if(ARM_NN_ENABLE_F32/F16)`
block. Host-build with:

```
cmake -S Tests/UnitTest -B <build> -DBUILD_CMSIS_NN_UNIT=ON \
  -DCMSIS_PATH=<path with CMSIS/Core/Include> \
  -DARM_NN_ENABLE_F32=ON -DARM_NN_ENABLE_F16=ON
```

then build/run individual `test_arm_*` targets. (The integer suites do not
host-build — pre-existing `%ld` format errors; they target the Corstone-300
FVP.) Pick golden-data block sizes that are not multiples of the vector lane
count so the MVE tail-predication path is exercised.

## Formatting and checks

- `clang-format` 16.0.6 (the CI version) on `Source|Include` C/H files.
- `python3 scripts/check_pdsc.py` after any manifest change.
- License headers: new source files use the Ambiq SPDX header
  (`LicenseRef-Ambiq-Apollo-SDK`); generated test-data headers use Apache-2.0
  (matching existing precedent).
