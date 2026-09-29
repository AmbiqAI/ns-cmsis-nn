#
# SPDX-FileCopyrightText: Copyright 2026 Ambiq
#
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
#
# Licensed under the Ambiq Apollo SDK License.
# See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
#

# Check the six scalar elementwise kernels under shipped finite-math flags (#334).
# Rebuild the kernel objects in this standalone target; linking only cmsis-nn
# would exercise its different test-build flags. Private options below must
# follow the inherited -fno-finite-math-only so the final option takes effect.
# cmsis-nn supplies public includes/definitions; these objects supply the kernels.

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
