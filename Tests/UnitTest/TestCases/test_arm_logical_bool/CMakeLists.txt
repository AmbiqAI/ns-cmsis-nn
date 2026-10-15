# SPDX-FileCopyrightText: 2026 Ambiq
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
add_cmsis_nn_unit_test_executable(test_arm_logical_bool)
target_sources(test_arm_logical_bool PRIVATE
    Unity/unity_test_arm_logical_bool.c
    Unity/TestRunner/unity_test_arm_logical_bool_runner.c)
