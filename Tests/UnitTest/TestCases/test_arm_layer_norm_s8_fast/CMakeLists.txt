#
# SPDX-FileCopyrightText: 2026 Ambiq
#
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
#
# Licensed under the Ambiq Apollo SDK License.
# See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
#

add_cmsis_nn_unit_test_executable(test_arm_layer_norm_s8_fast)

target_sources(test_arm_layer_norm_s8_fast PRIVATE
    Unity/unity_test_arm_layer_norm_s8_fast.c
    Unity/TestRunner/unity_test_arm_layer_norm_s8_fast_runner.c)
