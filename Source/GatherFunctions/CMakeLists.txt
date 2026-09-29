#
# SPDX-FileCopyrightText: 2025 Ambiq
#
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
#
# Licensed under the Ambiq Apollo SDK License.
# See LICENSE (root) or LICENSES/LicenseRef-Ambiq-Apollo-SDK.txt for the full text.
#

file(GLOB SRC "./*_s8.c" "./*_s16.c")
target_sources(cmsis-nn PRIVATE ${SRC})

if(ARM_NN_ENABLE_F32)
  target_sources(cmsis-nn PRIVATE arm_gather_f32.c arm_gather_nd_f32.c)
endif()

if(ARM_NN_ENABLE_F16)
  target_sources(cmsis-nn PRIVATE arm_gather_f16.c arm_gather_nd_f16.c)
endif()
