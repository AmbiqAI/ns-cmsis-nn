# Build and test validation

The existing toolchain matrix contains two focused checks in addition to its
normal library builds and strict links.

- The GCC 15.3 Cortex-M55 cell compiles the complete library for
  `cortex-m55+nomve.fp` with `ARM_NN_ENABLE_F32=OFF` and
  `ARM_NN_ENABLE_F16=OFF`. This admits integer MVE without floating-point MVE.
- The ATfE Cortex-M4 cell builds the library with its default
  `ARM_NN_WORD_COPY` paths. A GNU 14.3 Cortex-M4 Unity harness links that archive
  and executes the `m4-word-copy` suites named in
  [check_atfe_unity_suites.py](../../scripts/check_atfe_unity_suites.py).
  QEMU's `mps3-an547` Cortex-M55 model executes the M4-compiled DSP code; this
  check does not qualify a physical Cortex-M4 board.
- The ATfE Cortex-M55 cell builds the shipped F32+F16 library. A GNU 14.3
  Cortex-M55 Unity harness links that archive and executes the `m55-float`
  suites: reduce min/max, argmax/argmin, softmax f16 and nearest-neighbor
  resize. These kernels returned different results under LLVM fast math
  (#852, #853, #854); against an archive built before those fixes, the reduce,
  softmax shapes and resize suites (9 of 14) fail. The argmax/argmin and basic
  softmax suites are guards with no known ATfE failure. The same cell rejects
  any MVE float compare in the reduce and argmax/argmin objects, which select
  on raw bits. The kernels are ATfE-compiled, but the harness, startup and libm
  (`floorf`, `roundf`) are GNU's, and QEMU is not a physical board.

For both ATfE executions, every case in the generated runner must report PASS
with its expected source file, name and line. Exactly one complete summary with
zero failures and zero ignored cases must follow.

The harness accepts an existing archive through `CMSIS_NN_TEST_LIBRARY` for
this cross-toolchain check. Ordinary builds compile the library from source.
The archive and harness must use compatible architecture and floating-point
ABI flags. The imported target carries the requested F32/F16 definitions
through the same feature-definition rule as source builds.

Unity is downloaded under the build directory's `generated/Unity`, and its
runners are generated under `generated/TestCases`. Independent build
directories have independent generated files. The tracked
`Tests/UnitTest/Unity/unityConfig.cmake` remains unchanged.

Integer suites supply bias metadata for a contiguous vector:
`{1, 1, 1, output_channels}`. SVDF uses one bias per output unit, so its vector
length is the feature count divided by the rank. The DS-CNN suites update the
channel count when moving between layers.

The existing stale-version reference tests rehearse a version bump in an
isolated tracked fixture tree. Both a test fixture and a help-string example
must reject the previous version after the bump and pass after their references
are updated. The live package version, tags and release assets are unaffected.

These checks extend build and harness coverage. Compile-only evidence remains
separate from execution, and model execution remains separate from physical
board qualification.
