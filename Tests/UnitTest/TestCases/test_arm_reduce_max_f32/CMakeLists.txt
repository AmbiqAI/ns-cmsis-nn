# SPDX-FileCopyrightText: Copyright 2026 Ambiq
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK

add_cmsis_nn_unit_test_executable(test_arm_reduce_max_f32)
target_sources(test_arm_reduce_max_f32 PRIVATE
    Unity/unity_test_arm_reduce_max_f32.c
    Unity/TestRunner/unity_test_arm_reduce_max_f32_runner.c)
