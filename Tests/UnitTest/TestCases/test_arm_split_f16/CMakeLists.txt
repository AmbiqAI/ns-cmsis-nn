#
# SPDX-FileCopyrightText: Copyright 2026 Ambiq
#
# SPDX-License-Identifier: Apache-2.0
#

add_cmsis_nn_unit_test_executable(test_arm_split_f16)

target_sources(test_arm_split_f16 PRIVATE
    Unity/unity_test_arm_split_f16.c
    Unity/TestRunner/unity_test_arm_split_f16_runner.c)
