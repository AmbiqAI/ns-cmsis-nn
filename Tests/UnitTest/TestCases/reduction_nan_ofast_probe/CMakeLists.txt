#
# SPDX-FileCopyrightText: Copyright 2026 Ambiq
#
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
#
# Licensed under the Ambiq Apollo SDK License.
# See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
#

# Check the scalar f16 reduction kernels under shipped finite-math flags (#429).
# Rebuild the kernel objects here; private options must follow the inherited
# -fno-finite-math-only. cmsis-nn supplies public includes/definitions, while
# these objects supply the kernels. Native execution covers only scalar paths;
# MVE behavior requires target execution.

add_executable(test_arm_reduction_nan_ofast_probe
    test_arm_reduction_nan_ofast_probe.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../Source/BasicMathFunctions/arm_nn_mean_f16.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../Source/BasicMathFunctions/arm_reduce_sum_f16.c)

target_compile_options(test_arm_reduction_nan_ofast_probe PRIVATE -Ofast -ffinite-math-only)
target_link_libraries(test_arm_reduction_nan_ofast_probe PRIVATE cmsis-nn m)

add_test(NAME test_arm_reduction_nan_ofast_probe COMMAND test_arm_reduction_nan_ofast_probe)
