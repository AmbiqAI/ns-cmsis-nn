#
# SPDX-FileCopyrightText: Copyright 2024-2025
# SPDX-License-Identifier: Apache-2.0
#

# Numeric comparisons and Boolean logical entries.
file(GLOB SRC "*_s8.c" "*_s16.c" "*_s32.c" "*_bool.c")
if(ARM_NN_ENABLE_F32)
  list(APPEND SRC "${CMAKE_CURRENT_SOURCE_DIR}/arm_greater_f32.c")
endif()
target_sources(cmsis-nn PRIVATE ${SRC})
