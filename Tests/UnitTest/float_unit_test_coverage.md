# Float Unit Test Coverage

This file summarizes the float unit-test families currently covered by the
fork-native test flow.

Source of truth:
- `Tests/UnitTest/run_float_unit_tests.py`
- `Tests/UnitTest/cmsis/cmsis_nn_unit_tests_flt.csolution.yml`
- `Tests/UnitTest/float_unit_test_coverage.yml`

Covered dtypes: `f32`, `f16`

## Family Summary

| Family | Generator | Host Targets | CMSIS Project |
| --- | --- | --- | --- |
| `convolve` | `conv_settings_flt.py` | none | none |
| `reshape` | `none` | `test_arm_reshape_f32`, `test_arm_reshape_f16` | `test_arm_reshape_flt` |
| `transpose_conv` | `transpose_conv_settings_flt.py` | `test_arm_transpose_conv_f32`, `test_arm_transpose_conv_f16` | `test_arm_transpose_conv_flt` |

## Coverage Details

### `convolve`

No Unity coverage in either dtype: both suites were deleted as unbuildable, so
nothing in this repo compiles or runs a float convolution case. The family is
listed only because `--tests convolve` still resolves and its generator is still
usable offline.

- Generator: `conv_settings_flt.py` (still usable offline; its output is
  not consumed by any Unity suite)
- Host targets: none (`test_arm_convolve_f32` deleted in #256,
  `test_arm_convolve_f16` deleted as unbuildable, never CI-built)
- CMSIS project: none (`test_arm_convolve_flt` deleted with its only suite)
- CMSIS contexts: none
- Covered cases: none
- f16 convolution elsewhere: still exercised by the helia-core-tester FP16
  convolve descriptors; a Unity suite for the packed f16 path
  (`test_arm_convolve_packed_f16`) is proposed in #325 and is not present
  in this repo.
- Generator cases, not compiled or executed: `conv_settings_flt.py` defines 35
  f32 shapes and mirrors every one into an f16 twin, so `--dataset all` writes
  70 datasets. 26 of the 35 are `conv_match_*` shapes that replay geometries
  from the integer convolution suites (7 basic/stride/activation, 6 dilation,
  5 1x1, 8 1xN). The remaining nine are:
  - Generic 3x3 case: input 1x6x9x11, out_ch=10, stride 1, no padding.
  - Generic 3x3 wrapper case with the same logical dimensions.
  - 1x1 NHWC stride case: input 1x3x17x11, out_ch=13, stride 1x2, activation clamp [-0.75, 0.75].
  - Optimized 1x3 case: input 1x8x1x21, out_ch=12.
  - Optimized 1x5 case: input 1x8x1x21, out_ch=12.
  - Tuned NHWC 1x3 case: input 1x1x21x16, out_ch=16.
  - Tuned NHWC 1x5 case: input 1x1x21x16, out_ch=16.
  - Common 2x2 case: input 1x4x6x7, out_ch=5.
  - Common 3x3 pad1 case: input 1x2x3x6, out_ch=4.

### `reshape`

Simple flat reshape/copy sanity coverage.

- Generator: `none`
- Host targets: `test_arm_reshape_f32`, `test_arm_reshape_f16`
- CMSIS project: `test_arm_reshape_flt`
- CMSIS contexts: `test_arm_reshape_flt.F32+Corstone-300-FVP`, `test_arm_reshape_flt.F16+Corstone-300-FVP`
- Covered cases:
  - Single contiguous buffer of length 6 copied input -> output.

### `transpose_conv`

Transpose-convolution coverage in NHWC direct and wrapper paths.

- Generator: `transpose_conv_settings_flt.py`
- Host targets: `test_arm_transpose_conv_f32`, `test_arm_transpose_conv_f16`
- CMSIS project: `test_arm_transpose_conv_flt`
- CMSIS contexts: `test_arm_transpose_conv_flt.F32+Corstone-300-FVP`, `test_arm_transpose_conv_flt.F16+Corstone-300-FVP`
- Covered cases:
  - Input 1x4x5x6, output_ch=6, kernel 3x3, stride 2x2, padding 1x1, output_padding 1x1.
  - Matching wrapper case with the same logical dimensions.

## Aliases

| Alias | Canonical Family |
| --- | --- |
| `conv` | `convolve` |
| `tconv` | `transpose_conv` |
| `transposeconv` | `transpose_conv` |
