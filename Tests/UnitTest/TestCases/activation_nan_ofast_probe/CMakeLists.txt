#
# SPDX-FileCopyrightText: Copyright 2026 Ambiq
#
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
#
# Licensed under the Ambiq Apollo SDK License.
# See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
#

# Check the scalar RELU/RELU6 kernels under shipped finite-math flags (#382).
# Rebuild the kernel objects here; private options must follow the inherited
# -fno-finite-math-only. cmsis-nn supplies public includes/definitions and LUT
# tables, while these objects supply the kernels. Native execution covers only
# scalar paths; MVE behavior requires target execution.

add_executable(test_arm_activation_nan_ofast_probe
    test_arm_activation_nan_ofast_probe.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../Source/ActivationFunctions/arm_nn_activation_f32.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../Source/ActivationFunctions/arm_nn_activation_f16.c)

target_compile_options(test_arm_activation_nan_ofast_probe PRIVATE -Ofast -ffinite-math-only)
target_link_libraries(test_arm_activation_nan_ofast_probe PRIVATE cmsis-nn m)

add_test(NAME test_arm_activation_nan_ofast_probe COMMAND test_arm_activation_nan_ofast_probe)
