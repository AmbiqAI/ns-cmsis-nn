# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Minimal CMake toolchain file for GNU Arm Embedded (arm-none-eabi-gcc).
#
# Used by the release pipeline to cross-compile cmsis-nn as a static
# library for cortex-m0 / cortex-m4 / cortex-m55. NOT intended to be the
# only way to cross-build cmsis-nn — consumers with their own toolchain
# setup (Zephyr, NSX, …) are unaffected.
#
# Inputs:
#   NS_CMSIS_NN_TARGET_CPU   one of: cortex-m0 | cortex-m4 | cortex-m55
#
# Sets architecture flags to match the matrix in
# .github/workflows/staticlib.yml. Keep the two in sync.

set(CMAKE_SYSTEM_NAME      Generic)
set(CMAKE_SYSTEM_PROCESSOR ARM)

set(CMAKE_C_COMPILER   arm-none-eabi-gcc)
set(CMAKE_ASM_COMPILER arm-none-eabi-gcc)
set(CMAKE_AR           arm-none-eabi-ar)
set(CMAKE_RANLIB       arm-none-eabi-ranlib)

# We're building a static library — don't let CMake try to link a test
# executable during compiler detection (no startup files / linker
# script in this toolchain layout).
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

if(NOT NS_CMSIS_NN_TARGET_CPU)
  message(FATAL_ERROR
    "NS_CMSIS_NN_TARGET_CPU must be set "
    "(cortex-m0 | cortex-m4 | cortex-m55).")
endif()

if(NS_CMSIS_NN_TARGET_CPU STREQUAL "cortex-m0")
  set(_arch_flags "-mcpu=cortex-m0 -mthumb -mfloat-abi=soft")
elseif(NS_CMSIS_NN_TARGET_CPU STREQUAL "cortex-m4")
  set(_arch_flags "-mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard")
elseif(NS_CMSIS_NN_TARGET_CPU STREQUAL "cortex-m55")
  set(_arch_flags "-mcpu=cortex-m55 -mthumb -mfloat-abi=hard")
else()
  message(FATAL_ERROR
    "Unsupported NS_CMSIS_NN_TARGET_CPU '${NS_CMSIS_NN_TARGET_CPU}'. "
    "Expected cortex-m0 | cortex-m4 | cortex-m55.")
endif()

set(CMAKE_C_FLAGS_INIT          "${_arch_flags}")
set(CMAKE_ASM_FLAGS_INIT        "${_arch_flags}")
set(CMAKE_EXE_LINKER_FLAGS_INIT "${_arch_flags}")
