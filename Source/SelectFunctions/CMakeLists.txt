#
# SPDX-FileCopyrightText: 2025 Ambiq
#
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
#
# Licensed under the Ambiq Apollo SDK License.
# See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
#

file(GLOB SRC "*_s8.c" "*_s16.c" "*_s32.c")
list(APPEND SRC "${CMAKE_CURRENT_SOURCE_DIR}/arm_nn_select_word.c")
if(ARM_NN_ENABLE_F32)
  list(APPEND SRC "${CMAKE_CURRENT_SOURCE_DIR}/arm_select_v2_f32.c")
endif()
target_sources(cmsis-nn PRIVATE ${SRC})
