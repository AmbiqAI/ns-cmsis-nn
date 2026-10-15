# SPDX-FileCopyrightText: Copyright 2026 Ambiq Micro, Inc.
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK

add_cmsis_nn_unit_test_executable(nn_accumulator_wrap_s8)

target_sources(nn_accumulator_wrap_s8 PRIVATE
    Unity/unity_test_arm_nn_accumulator_wrap_s8.c
    Unity/TestRunner/unity_test_arm_nn_accumulator_wrap_s8_runner.c)
