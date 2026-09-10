#
# SPDX-FileCopyrightText: Copyright 2026 Ambiq
#
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
#
# Licensed under the Ambiq Apollo SDK License.
# See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
#

# Host -Ofast NaN probe for the float activation legs (#382). Same shape and
# same reasoning as TestCases/elementwise_nan_ofast_probe: deliberately NOT
# registered through add_cmsis_nn_unit_test_executable, because this is not a
# Unity suite and it must not inherit the harness's -fno-finite-math-only. The
# two activation kernels are recompiled INTO this target so that they -- not
# just the probe TU -- are built at -Ofast -ffinite-math-only, the shipped
# library's hostile flag pair; linking the harness's cmsis-nn library (built at
# the harness's optimization level) would test nothing. The
# target_compile_options below land after the directory-level
# `${CMSIS_OPTIMIZATION_LEVEL} -fno-finite-math-only` on the command line, so
# the last-wins rule gives this target the hostile pair. cmsis-nn is still
# linked for its PUBLIC include directories, the ARM_NN_ENABLE_F32/F16
# definitions, and the LUT tables the kernels reference from
# arm_nntables_flt.c; the probe's own kernel objects resolve the two activation
# symbols first, so no conflicting archive member is pulled in.
#
# On an x86 host neither ARM_MATH_MVEF nor ARM_MATH_MVE_FLOAT16 is defined, so
# this probe exercises the SCALAR activation legs only. The MVE RELU/RELU6 legs
# that #382 also changed are covered on target (QEMU mps3-an547), not here.

add_executable(test_arm_activation_nan_ofast_probe
    test_arm_activation_nan_ofast_probe.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../Source/ActivationFunctions/arm_nn_activation_f32.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../Source/ActivationFunctions/arm_nn_activation_f16.c)

target_compile_options(test_arm_activation_nan_ofast_probe PRIVATE -Ofast -ffinite-math-only)
target_link_libraries(test_arm_activation_nan_ofast_probe PRIVATE cmsis-nn m)

add_test(NAME test_arm_activation_nan_ofast_probe COMMAND test_arm_activation_nan_ofast_probe)
