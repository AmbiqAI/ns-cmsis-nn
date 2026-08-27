# CMSIS-Toolbox Float Unit Tests

This folder contains a small CMSIS-Toolbox setup for running selected float
unit tests from a local CMSIS-NN checkout directly on the Corstone-300 FVP.

The float test inputs and references are generated offline by the Python
generators under `Tests/UnitTest/`:

- raw reproducible samples are kept in `Tests/UnitTest/PregeneratedData/`
- generated C headers used by the tests are written to
  `Tests/UnitTest/TestCases/TestData/`

Regenerate those datasets before building if you update a float generator or
want fresh sampled inputs.

Current shared projects:

- `test_arm_reshape_flt`
- `test_arm_transpose_conv_flt`

## Unified Runner

For day-to-day float bring-up, you can use the shared runner script instead of
calling the generators, host `cmake`, `cbuild`, and FVP manually:

```bash
cd <repo-root>/Tests/UnitTest

# Default flow: generate + host build + host run
python3 run_float_unit_tests.py --tests reshape,transpose_conv --dtypes f16

# CMSIS build with a specific toolchain
python3 run_float_unit_tests.py \
  --tests reshape \
  --dtypes f16 \
  --build-cmsis \
  --toolchains GCC@15.2.1

# Full CMSIS + FVP flow
python3 run_float_unit_tests.py \
  --tests transpose_conv \
  --dtypes f16 \
  --build-cmsis \
  --run-fvp \
  --toolchains GCC@15.2.1 \
  --fvp-bin <fvp-bin>
```

Useful options:

- `--tests`: comma-separated family list such as `reshape,transpose_conv`
  (see `Current shared projects` above for the full list; `convolve` is
  still accepted but has no buildable suite in either dtype -- its F32
  half went with #256 and its F16 half, `test_arm_convolve_f16`, was
  deleted as unbuildable too; f16 convolution is covered by the
  helia-core-tester descriptors and `test_arm_convolve_packed_f16`)
- `--dtypes`: `f32`, `f16`, or `f32,f16`
- `--toolchains`: comma-separated `cbuild` toolchains such as `GCC@15.2.1,AC6@6.24.0`
- `--regenerate-input`: refresh pregenerated float samples before rebuilding

Run the full float matrix:

```bash
cd <repo-root>/Tests/UnitTest

# Host-only: all float families, both f32 and f16
python3 run_float_unit_tests.py --tests all --dtypes f32,f16

# Host + CMSIS build
python3 run_float_unit_tests.py \
  --tests all \
  --dtypes f32,f16 \
  --build-cmsis \
  --toolchains GCC@15.2.1

# Host + CMSIS + FVP
python3 run_float_unit_tests.py \
  --tests all \
  --dtypes f32,f16 \
  --build-cmsis \
  --run-fvp \
  --toolchains GCC@15.2.1 \
  --fvp-bin <fvp-bin>
```

`test_arm_reshape_flt` and `test_arm_transpose_conv_flt` each expose two
contexts:

- `.F32+Corstone-300-FVP`
- `.F16+Corstone-300-FVP`

## Prerequisites

1. Register the local NS-CMSIS-NN pack:

   ```bash
   cd <repo-root>
   cpackget add Ambiq.NS-CMSIS-NN.pdsc \
     -R <pack-root> \
     -F
   ```

2. Make sure the usual CMSIS packs used by the solution are already installed in
   the same pack root.

## Build

Example with GCC:

```bash
export PATH=<cmsis-toolbox-bin>:$PATH
export GCC_TOOLCHAIN_15_2_1=<gcc-toolchain-bin>

cd <repo-root>/Tests/UnitTest/cmsis

cbuild --update-rte \
  --context test_arm_transpose_conv_flt.F16+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j4 \
  --toolchain GCC@15.2.1

cbuild --context test_arm_transpose_conv_flt.F16+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j4 \
  --toolchain GCC@15.2.1
```

If you cleaned generated artifacts and removed `Tests/UnitTest/Unity`, restore the
local Unity checkout and regenerate the Unity runners before calling `cbuild`:

```bash
cd <repo-root>/Tests/UnitTest
python3 unittest_targets.py --download-and-generate-test-runners
```

You can replace `test_arm_transpose_conv_flt` with `test_arm_reshape_flt`,
and for either of them, replace `F16` with `F32` (or vice versa) — both
build-types are valid for both projects.

Example for a single context, one job:

```bash
# transpose conv f16
cbuild --update-rte \
  --context test_arm_transpose_conv_flt.F16+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j1 --toolchain GCC@15.2.1
cbuild \
  --context test_arm_transpose_conv_flt.F16+Corstone-300-FVP \
  cmsis_nn_unit_tests_flt.csolution.yml \
  -j1 --toolchain GCC@15.2.1
```

## Run on FVP

Example for the `f32` transpose-conv test built with GCC:

```bash
<fvp-bin> \
  test_arm_transpose_conv_flt.F32+Corstone-300-FVP-GCC/outdir/test_arm_transpose_conv_f32.elf \
  -C mps3_board.visualisation.disable-visualisation=1 \
  -C mps3_board.telnetterminal0.start_telnet=0 \
  -C mps3_board.uart0.out_file=- \
  -C mps3_board.uart0.unbuffered_output=1 \
  -C mps3_board.uart0.shutdown_on_eot=1
```

Example for the validated `f16` reshape test built with GCC:

```bash
<fvp-bin> \
  test_arm_reshape_flt.F16+Corstone-300-FVP-GCC/outdir/test_arm_reshape_f16.elf \
  -C mps3_board.visualisation.disable-visualisation=1 \
  -C mps3_board.telnetterminal0.start_telnet=0 \
  -C mps3_board.uart0.out_file=- \
  -C mps3_board.uart0.unbuffered_output=1 \
  -C mps3_board.uart0.shutdown_on_eot=1
```
