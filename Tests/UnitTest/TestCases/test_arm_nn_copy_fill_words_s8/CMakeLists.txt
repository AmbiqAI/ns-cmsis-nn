#
# SPDX-FileCopyrightText: Copyright 2026 Ambiq
#
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
#
# Licensed under the Ambiq Apollo SDK License.
# See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
#

add_cmsis_nn_unit_test_executable(test_arm_nn_copy_fill_words_s8)

target_sources(test_arm_nn_copy_fill_words_s8 PRIVATE
    Unity/unity_test_arm_nn_copy_fill_words_s8.c
    Unity/TestRunner/unity_test_arm_nn_copy_fill_words_s8_runner.c)

# Select the word-at-a-time copy and fill on every toolchain, not only where the header selects it by default
target_compile_definitions(test_arm_nn_copy_fill_words_s8 PRIVATE ARM_NN_WORD_COPY)
