# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0

set(CMAKE_SYSTEM_NAME Generic)

# CMAKE_SYSTEM_PROCESSOR names the concrete CPU rather than the generic "ARM"
# the GCC and ATfE toolchain files use: CMake's ARMClang module
# (Modules/Compiler/ARMClang.cmake) rejects a name absent from armclang's
# -mcpu=list. It is consumed during project() / compiler identification, so it
# has to be set before any of the NS_CMSIS_NN_TARGET_CPU validation further
# down. cortex-m0, cortex-m4 and cortex-m55 are all in the supported list.
# Under CMP0123 NEW -- set in the top-level CMakeLists.txt, since a
# cmake_policy() call in an include()d toolchain file is popped before
# project() sees it -- the module derives no flag from this variable; the
# arch flags below are the only source. see AmbiqAI/ns-cmsis-nn#292
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

# Arm Compiler ships its own archiver, `armar`, and CMake's ARMClang module
# hard-codes armar's dialect for the archive rule:
#   set(CMAKE_${lang}_ARCHIVE_CREATE "<CMAKE_AR> --create -cr <TARGET> ...")
# llvm-ar has no `--create`, so pointing CMAKE_AR at it fails with
# "llvm-ar: error: unknown option -" during the C compiler ABI check, before
# the project ever configures. Resolve armar from the toolchain root rather
# than PATH -- matching atfe.cmake and Tests/UnitTest/build_lib_variants.py's
# AC6 branch -- so nothing already on PATH can shadow it
# (AmbiqAI/ns-cmsis-nn#279).
if(NOT EXISTS "${_root}/bin/armar")
  message(FATAL_ERROR "armclang builds require armar at ${_root}/bin/armar.")
endif()

set(CMAKE_C_COMPILER   "${_root}/bin/armclang")
set(CMAKE_ASM_COMPILER "${_root}/bin/armclang")
# CMSISNN's top-level project() declares "C CXX", so CMake tests a C++
# compiler even though this project ships no C++ sources. Without this line
# it auto-detects the HOST compiler, which is both wrong for a cross build
# and fatal here: a non-ARMClang CXX compiler gets CMake's generic archive
# rule ("<CMAKE_AR> qc ..."), and armar rejects a dashless action with
# "L6839E: One of the actions -[dmpqrtx] must be specified". Naming armclang
# for CXX loads Compiler/ARMClang-CXX.cmake, whose __compiler_armclang(CXX)
# emits the same "--create -cr" form the C side already archives with.
set(CMAKE_CXX_COMPILER "${_root}/bin/armclang")
set(CMAKE_AR           "${_root}/bin/armar")

# armar writes the archive symbol table itself and Arm Compiler ships no
# ranlib. ARMClang.cmake overrides only ARCHIVE_CREATE, so CMake would fall
# back to its generic ARCHIVE_FINISH ("<CMAKE_RANLIB> <TARGET>") against an
# unset CMAKE_RANLIB. Empty the finish rule instead of inventing a ranlib.
set(CMAKE_C_ARCHIVE_FINISH   "")
set(CMAKE_CXX_ARCHIVE_FINISH "")
set(CMAKE_ASM_ARCHIVE_FINISH "")

# Stripping is done by scripts/build_staticlib.sh, which resolves llvm-strip
# off PATH for this toolchain; CMAKE_STRIP is unused for static libraries.
find_program(_llvm_strip llvm-strip)
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
