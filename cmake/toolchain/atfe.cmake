# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR ARM)

set(_root "${NS_CMSIS_NN_TOOLCHAIN_ROOT}")
if(NOT _root)
  set(_root "$ENV{NS_CMSIS_NN_TOOLCHAIN_ROOT}")
endif()
if(NOT _root)
  set(_root "$ENV{NS_CMSIS_NN_ATFE_ROOT}")
endif()
if(NOT _root)
  message(FATAL_ERROR "Set NS_CMSIS_NN_TOOLCHAIN_ROOT or NS_CMSIS_NN_ATFE_ROOT to the ATfE install root.")
endif()

set(CMAKE_C_COMPILER   "${_root}/bin/clang")
set(CMAKE_ASM_COMPILER "${_root}/bin/clang")
set(CMAKE_AR           "${_root}/bin/llvm-ar")
set(CMAKE_RANLIB       "${_root}/bin/llvm-ranlib")
set(CMAKE_STRIP        "${_root}/bin/llvm-strip")
set(CMAKE_C_COMPILER_TARGET   arm-none-eabi)
set(CMAKE_ASM_COMPILER_TARGET arm-none-eabi)

set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES NS_CMSIS_NN_TARGET_CPU NS_CMSIS_NN_TOOLCHAIN_ROOT)

if(NOT NS_CMSIS_NN_TARGET_CPU)
  message(FATAL_ERROR "NS_CMSIS_NN_TARGET_CPU must be set (cortex-m0 | cortex-m4 | cortex-m55).")
endif()

if(NS_CMSIS_NN_TARGET_CPU STREQUAL "cortex-m0")
  set(_arch_flags "-mcpu=cortex-m0 -mthumb -mfloat-abi=soft")
elseif(NS_CMSIS_NN_TARGET_CPU STREQUAL "cortex-m4")
  set(_arch_flags "-mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard")
elseif(NS_CMSIS_NN_TARGET_CPU STREQUAL "cortex-m55")
  set(_arch_flags "-mcpu=cortex-m55 -mthumb -mfloat-abi=hard")
else()
  message(FATAL_ERROR "Unsupported NS_CMSIS_NN_TARGET_CPU '${NS_CMSIS_NN_TARGET_CPU}'.")
endif()

set(CMAKE_C_FLAGS_INIT "${_arch_flags}")
set(CMAKE_ASM_FLAGS_INIT "${_arch_flags}")
set(CMAKE_EXE_LINKER_FLAGS_INIT "${_arch_flags}")
