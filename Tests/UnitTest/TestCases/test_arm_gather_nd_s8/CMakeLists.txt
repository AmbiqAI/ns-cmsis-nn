#
# SPDX-FileCopyrightText: 2025 Ambiq
#
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
#
# Licensed under the Ambiq Apollo SDK License.
# See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
#

add_cmsis_nn_unit_test_executable(test_arm_gather_nd_s8)

target_sources(test_arm_gather_nd_s8 PRIVATE
    Unity/unity_test_arm_gather_nd_s8.c
    Unity/TestRunner/unity_test_arm_gather_nd_s8_runner.c)
