# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
add_cmsis_nn_unit_test_executable(test_arm_resize_bilinear_s8)
target_sources(test_arm_resize_bilinear_s8 PRIVATE
    Unity/unity_test_arm_resize_bilinear_s8.c
    Unity/TestRunner/unity_test_arm_resize_bilinear_s8_runner.c)
