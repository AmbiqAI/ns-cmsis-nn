#
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
#
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
#
# Host reference build: the library compiled for the host CPU with no kernel
# path selection (no DSP, MVE or Helium code), used by helia-core-tester to
# produce test goldens.
#
#   cmake -S . -B build/host-ref -DNS_CMSIS_NN_HOST_REFERENCE=ON
#   cmake --build build/host-ref
#
#   ns_cmsis_nn_host_reference(<target>)
#       Fails the configure unless <target> can be a host reference, adds the
#       floating-point flags and the compiled guard, and writes the manifest
#       <binary dir>/ns_cmsis_nn_host_reference.json (schema
#       hct/host-reference/1) that consumers read instead of guessing flags.
#
# Every refusal is a FATAL_ERROR: a reference built with fast-math, a selected
# vector path or a missing float width would produce wrong goldens silently.
#

include_guard(GLOBAL)
include(CheckCSourceCompiles)

set(NS_CMSIS_NN_HOST_REFERENCE_SCHEMA "hct/host-reference/1")
set(NS_CMSIS_NN_HOST_REFERENCE_FP_OPTIONS -ffp-contract=off -fno-fast-math)
set(_NS_CMSIS_NN_HOST_REFERENCE_PATH_MACROS
    ARM_MATH_DSP ARM_MATH_MVEI ARM_MATH_MVEF ARM_MATH_HELIUM ARM_MATH_MVE_FLOAT16
    ARM_MATH_AUTOVECTORIZE ARM_MATH_NEON)
set(_NS_CMSIS_NN_HOST_REFERENCE_UNSAFE_FP
    "(^|[ ;])(-Ofast|-ffast-math|-funsafe-math-optimizations|-fassociative-math|-freciprocal-math|-ffinite-math-only|-ffp-contract=(fast|on))([ ;]|$)")

function(_ns_cmsis_nn_host_reference_fail reason)
  message(FATAL_ERROR "ns-cmsis-nn host reference: ${reason}")
endfunction()

function(_ns_cmsis_nn_host_reference_probe out_var source)
  set(CMAKE_REQUIRED_FLAGS "${CMSIS_OPTIMIZATION_LEVEL}")
  set(CMAKE_REQUIRED_INCLUDES "${NS_CMSIS_NN_ROOT}/Include")
  set(CMAKE_REQUIRED_DEFINITIONS -DARM_NN_ENABLE_F32=1 -DARM_NN_ENABLE_F16=1)
  set(CMAKE_REQUIRED_QUIET ON)
  # A cached answer would survive a change of CMAKE_C_FLAGS between configures.
  unset(${out_var} CACHE)
  check_c_source_compiles("${source}" ${out_var})
  set(${out_var} "${${out_var}}" PARENT_SCOPE)
  unset(${out_var} CACHE)
endfunction()

function(_ns_cmsis_nn_host_reference_check_settings target)
  if(NOT TARGET ${target})
    _ns_cmsis_nn_host_reference_fail("'${target}' is not a target")
  endif()
  if(CMAKE_CROSSCOMPILING)
    _ns_cmsis_nn_host_reference_fail(
      "it is built for the host CPU; configure it without a cross toolchain (CMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME})")
  endif()
  if(NOT CMAKE_C_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
    _ns_cmsis_nn_host_reference_fail(
      "needs GCC or Clang for its floating-point flags (got ${CMAKE_C_COMPILER_ID})")
  endif()
  if(NOT ARM_NN_ENABLE_F32 OR NOT ARM_NN_ENABLE_F16)
    _ns_cmsis_nn_host_reference_fail(
      "needs ARM_NN_ENABLE_F32=ON and ARM_NN_ENABLE_F16=ON so every kernel has a reference "
      "(got F32=${ARM_NN_ENABLE_F32} F16=${ARM_NN_ENABLE_F16})")
  endif()
  if(CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY)
    _ns_cmsis_nn_host_reference_fail("CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY selects Arm assembly; turn it OFF")
  endif()

  set(flag_vars CMSIS_OPTIMIZATION_LEVEL CMAKE_C_FLAGS)
  foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL ${CMAKE_CONFIGURATION_TYPES} ${CMAKE_BUILD_TYPE})
    string(TOUPPER "${config}" config)
    list(APPEND flag_vars CMAKE_C_FLAGS_${config})
  endforeach()
  list(REMOVE_DUPLICATES flag_vars)
  foreach(var IN LISTS flag_vars)
    if("${${var}}" MATCHES "${_NS_CMSIS_NN_HOST_REFERENCE_UNSAFE_FP}")
      _ns_cmsis_nn_host_reference_fail(
        "${var} contains '${CMAKE_MATCH_2}', which changes floating-point results; the reference is built "
        "with ${NS_CMSIS_NN_HOST_REFERENCE_FP_OPTIONS}")
    endif()
  endforeach()

  foreach(path IN ITEMS "${NS_CMSIS_NN_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_C_COMPILER}")
    if(path MATCHES "[\"\\\\]")
      _ns_cmsis_nn_host_reference_fail("path '${path}' contains a quote or backslash, which the manifest cannot carry")
    endif()
  endforeach()
endfunction()

function(_ns_cmsis_nn_host_reference_check_compiler out_float16_type)
  _ns_cmsis_nn_host_reference_probe(_ns_hr_float16 "
#include \"arm_nn_math_types.h\"
int main(void) { volatile float16_t h = (float16_t)1.0f; return (int)(float)h - 1; }")
  if(NOT _ns_hr_float16)
    _ns_cmsis_nn_host_reference_fail(
      "the host C compiler (${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}) has no IEEE float16 type "
      "(_Float16 or __fp16); use a compiler that does")
  endif()

  set(found "")
  foreach(macro IN LISTS _NS_CMSIS_NN_HOST_REFERENCE_PATH_MACROS)
    _ns_cmsis_nn_host_reference_probe(_ns_hr_no_${macro} "
#include \"arm_nn_math_types.h\"
#if defined(${macro})
#error ${macro}
#endif
int main(void) { return 0; }")
    if(NOT _ns_hr_no_${macro})
      list(APPEND found ${macro})
    endif()
  endforeach()
  if(found)
    string(REPLACE ";" ", " found "${found}")
    _ns_cmsis_nn_host_reference_fail(
      "the compile flags select a kernel path (${found}); the reference is the plain C build")
  endif()

  _ns_cmsis_nn_host_reference_probe(_ns_hr_fp16_storage "
#include \"arm_nn_math_types.h\"
#if !defined(__ARM_FP16_FORMAT_IEEE)
#error _Float16
#endif
int main(void) { return 0; }")
  if(_ns_hr_fp16_storage)
    set(${out_float16_type} "__fp16" PARENT_SCOPE)
  else()
    set(${out_float16_type} "_Float16" PARENT_SCOPE)
  endif()
endfunction()

function(ns_cmsis_nn_host_reference target)
  _ns_cmsis_nn_host_reference_check_settings(${target})
  _ns_cmsis_nn_host_reference_check_compiler(float16_type)

  target_compile_options(${target} PRIVATE ${NS_CMSIS_NN_HOST_REFERENCE_FP_OPTIONS})
  # Catches path macros added to the target itself, which the configure probes cannot see.
  target_sources(${target} PRIVATE "${NS_CMSIS_NN_ROOT}/cmake/host_reference/ns_cmsis_nn_host_reference_guard.c")

  set(options ${CMSIS_OPTIMIZATION_LEVEL} ${NS_CMSIS_NN_HOST_REFERENCE_FP_OPTIONS})
  string(REPLACE ";" "\", \"" options "${options}")
  set(includes "$<FILTER:$<REMOVE_DUPLICATES:$<TARGET_PROPERTY:${target},INTERFACE_INCLUDE_DIRECTORIES>>,INCLUDE,.+>")
  set(defines "$<FILTER:$<REMOVE_DUPLICATES:$<TARGET_PROPERTY:${target},INTERFACE_COMPILE_DEFINITIONS>>,INCLUDE,.+>")
  set(manifest "${CMAKE_CURRENT_BINARY_DIR}/ns_cmsis_nn_host_reference.json")
  file(GENERATE OUTPUT "${manifest}" CONTENT "{
  \"schema\": \"${NS_CMSIS_NN_HOST_REFERENCE_SCHEMA}\",
  \"library\": \"ns-cmsis-nn\",
  \"source_dir\": \"${NS_CMSIS_NN_ROOT}\",
  \"archive\": \"$<TARGET_FILE:${target}>\",
  \"include_dirs\": [\"$<JOIN:${includes},\", \">\"],
  \"compile_definitions\": [\"$<JOIN:${defines},\", \">\"],
  \"library_compile_options\": [\"${options}\"],
  \"link_libraries\": [\"m\"],
  \"float16_type\": \"${float16_type}\",
  \"c_compiler\": {
    \"path\": \"${CMAKE_C_COMPILER}\",
    \"id\": \"${CMAKE_C_COMPILER_ID}\",
    \"version\": \"${CMAKE_C_COMPILER_VERSION}\"
  }
}
")
  set_property(GLOBAL PROPERTY NS_CMSIS_NN_HOST_REFERENCE_MANIFEST "${manifest}")
  message(STATUS "ns-cmsis-nn host reference: ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}, "
                 "float16_t = ${float16_type}, manifest ${manifest}")
endfunction()
