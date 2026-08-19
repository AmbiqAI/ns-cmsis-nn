# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0

set(CMAKE_SYSTEM_NAME Generic)

# CMake's ARMClang module (Modules/Compiler/ARMClang.cmake) hard-errors unless
# CMAKE_SYSTEM_PROCESSOR names a processor it recognises -- the generic "ARM"
# that the GCC and ATfE toolchain files use is rejected. It is consumed during
# project() / compiler identification, so it has to be set before any of the
# NS_CMSIS_NN_TARGET_CPU validation further down. cortex-m0, cortex-m4 and
# cortex-m55 are all in the module's supported list.
if(NOT NS_CMSIS_NN_TARGET_CPU)
  set(NS_CMSIS_NN_TARGET_CPU "$ENV{NS_CMSIS_NN_TARGET_CPU}")
endif()
if(NS_CMSIS_NN_TARGET_CPU)
  set(CMAKE_SYSTEM_PROCESSOR "${NS_CMSIS_NN_TARGET_CPU}")
else()
  # Leave the real diagnostic to the validation block below, which names the
  # accepted values; a bare ARMClang error here would be far less useful.
  set(CMAKE_SYSTEM_PROCESSOR cortex-m4)
endif()

set(_root "${NS_CMSIS_NN_TOOLCHAIN_ROOT}")
if(NOT _root)
  set(_root "$ENV{NS_CMSIS_NN_TOOLCHAIN_ROOT}")
endif()
if(NOT _root)
  set(_root "$ENV{NS_CMSIS_NN_ARMCLANG_ROOT}")
endif()
if(NOT _root)
  message(FATAL_ERROR "Set NS_CMSIS_NN_TOOLCHAIN_ROOT or NS_CMSIS_NN_ARMCLANG_ROOT to the armclang install root.")
endif()

find_program(_llvm_ar llvm-ar)
find_program(_llvm_ranlib llvm-ranlib)
find_program(_llvm_strip llvm-strip)
if(NOT _llvm_ar OR NOT _llvm_ranlib)
  message(FATAL_ERROR "armclang release builds require llvm-ar and llvm-ranlib on PATH.")
endif()

set(CMAKE_C_COMPILER   "${_root}/bin/armclang")
set(CMAKE_ASM_COMPILER "${_root}/bin/armclang")
set(CMAKE_AR           "${_llvm_ar}")
set(CMAKE_RANLIB       "${_llvm_ranlib}")
if(_llvm_strip)
  set(CMAKE_STRIP "${_llvm_strip}")
endif()

set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES NS_CMSIS_NN_TARGET_CPU NS_CMSIS_NN_TOOLCHAIN_ROOT)

if(NOT NS_CMSIS_NN_TARGET_CPU)
  message(FATAL_ERROR "NS_CMSIS_NN_TARGET_CPU must be set (cortex-m0 | cortex-m4 | cortex-m55).")
endif()

if(NS_CMSIS_NN_TARGET_CPU STREQUAL "cortex-m0")
  set(_arch_flags "--target=arm-arm-none-eabi -mcpu=cortex-m0 -mthumb -mfloat-abi=soft")
elseif(NS_CMSIS_NN_TARGET_CPU STREQUAL "cortex-m4")
  set(_arch_flags "--target=arm-arm-none-eabi -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard")
elseif(NS_CMSIS_NN_TARGET_CPU STREQUAL "cortex-m55")
  set(_arch_flags "--target=arm-arm-none-eabi -mcpu=cortex-m55 -mthumb -mfloat-abi=hard")
else()
  message(FATAL_ERROR "Unsupported NS_CMSIS_NN_TARGET_CPU '${NS_CMSIS_NN_TARGET_CPU}'.")
endif()

set(CMAKE_C_FLAGS_INIT "${_arch_flags}")
set(CMAKE_ASM_FLAGS_INIT "${_arch_flags}")
set(CMAKE_EXE_LINKER_FLAGS_INIT "${_arch_flags}")
