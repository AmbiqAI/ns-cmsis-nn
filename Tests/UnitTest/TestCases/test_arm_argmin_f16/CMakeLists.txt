# SPDX-FileCopyrightText: 2026 Ambiq
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK

add_cmsis_nn_unit_test_executable(test_arm_argmin_f16)
target_sources(test_arm_argmin_f16 PRIVATE
    Unity/unity_test_arm_argmin_f16.c
    Unity/TestRunner/unity_test_arm_argmin_f16_runner.c)
