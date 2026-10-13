# Build and test validation

The existing toolchain matrix contains two focused checks in addition to its
normal library builds and strict links.

- The GCC 15.3 Cortex-M55 cell compiles the complete library for
  `cortex-m55+nomve.fp` with `ARM_NN_ENABLE_F32=OFF` and
  `ARM_NN_ENABLE_F16=OFF`. This admits integer MVE without floating-point MVE.
- The ATfE Cortex-M4 cell builds the library with its default
  `ARM_NN_WORD_COPY` paths. A GNU 14.3 Cortex-M4 Unity harness links that archive
  and executes the eight suites named in
  [check_atfe_word_copy.py](../../scripts/check_atfe_word_copy.py).
  QEMU's `mps3-an547` Cortex-M55 model executes the M4-compiled DSP code; this
  check does not qualify a physical Cortex-M4 board. A suite must report at
  least one test and zero failures to pass.

The harness accepts an existing archive through `CMSIS_NN_TEST_LIBRARY` for
this cross-toolchain check. Ordinary builds compile the library from source.
The archive and harness must use compatible architecture and floating-point
ABI flags.

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
