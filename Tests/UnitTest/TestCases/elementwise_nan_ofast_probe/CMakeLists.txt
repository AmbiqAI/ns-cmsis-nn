#
# SPDX-FileCopyrightText: Copyright 2026 Ambiq
#
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
#
# Licensed under the Ambiq Apollo SDK License.
# See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
#

# Host -Ofast NaN probe (#334 item d). Deliberately NOT registered through
# add_cmsis_nn_unit_test_executable: this is not a Unity suite, and it must not
# inherit the harness's -fno-finite-math-only. The six elementwise kernels are
# recompiled INTO this target so that they -- not just the probe TU -- are built
# at -Ofast -ffinite-math-only, the shipped library's hostile flag pair; linking
# the harness's cmsis-nn library (built at the harness's optimization level)
# would test nothing. The target_compile_options below land after the
# directory-level `${CMSIS_OPTIMIZATION_LEVEL} -fno-finite-math-only` on the
# command line, so the last-wins rule gives this target the hostile pair.
# cmsis-nn is still linked, but only for its PUBLIC include directories and the
# ARM_NN_ENABLE_F32/F16 definitions; the probe's own kernel objects resolve the
# six symbols first, so no conflicting archive member is pulled in.

add_executable(test_arm_elementwise_nan_ofast_probe
    test_arm_elementwise_nan_ofast_probe.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../Source/BasicMathFunctions/arm_elementwise_add_f32.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../Source/BasicMathFunctions/arm_elementwise_sub_f32.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../Source/BasicMathFunctions/arm_elementwise_mul_f32.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../Source/BasicMathFunctions/arm_elementwise_add_f16.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../Source/BasicMathFunctions/arm_elementwise_sub_f16.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../Source/BasicMathFunctions/arm_elementwise_mul_f16.c)

target_compile_options(test_arm_elementwise_nan_ofast_probe PRIVATE -Ofast -ffinite-math-only)
target_link_libraries(test_arm_elementwise_nan_ofast_probe PRIVATE cmsis-nn m)

add_test(NAME test_arm_elementwise_nan_ofast_probe COMMAND test_arm_elementwise_nan_ofast_probe)
