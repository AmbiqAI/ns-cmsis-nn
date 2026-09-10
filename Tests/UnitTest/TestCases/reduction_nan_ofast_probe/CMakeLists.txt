#
# SPDX-FileCopyrightText: Copyright 2026 Ambiq
#
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
#
# Licensed under the Ambiq Apollo SDK License.
# See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
#

# Host -Ofast NaN/Inf probe for the f16 reduction kernels (#429). Same shape and
# same reasoning as TestCases/activation_nan_ofast_probe: deliberately NOT
# registered through add_cmsis_nn_unit_test_executable, because this is not a
# Unity suite and it must not inherit the harness's -fno-finite-math-only. The
# two reduction kernels are recompiled INTO this target so that they -- not just
# the probe TU -- are built at -Ofast -ffinite-math-only, the flag pair the
# shipped library carries; linking the harness's cmsis-nn library would test
# nothing. The target_compile_options below land after the directory-level
# `${CMSIS_OPTIMIZATION_LEVEL} -fno-finite-math-only` on the command line, so the
# last-wins rule gives this target the hostile pair. cmsis-nn is still linked for
# its PUBLIC include directories and the ARM_NN_ENABLE_F32/F16 definitions; the
# probe's own kernel objects resolve the two reduction symbols first, so no
# conflicting archive member is pulled in.
#
# On a host neither ARM_MATH_MVEF nor ARM_MATH_MVE_FLOAT16 is defined, so this
# probe exercises the SCALAR legs only. The MVE legs are covered on target.

add_executable(test_arm_reduction_nan_ofast_probe
    test_arm_reduction_nan_ofast_probe.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../Source/BasicMathFunctions/arm_nn_mean_f16.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../Source/BasicMathFunctions/arm_reduce_sum_f16.c)

target_compile_options(test_arm_reduction_nan_ofast_probe PRIVATE -Ofast -ffinite-math-only)
target_link_libraries(test_arm_reduction_nan_ofast_probe PRIVATE cmsis-nn m)

add_test(NAME test_arm_reduction_nan_ofast_probe COMMAND test_arm_reduction_nan_ofast_probe)
