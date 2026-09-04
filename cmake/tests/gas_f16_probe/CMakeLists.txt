#
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# Flag-view test for cmake/check_gas_mve_f16_cvt.cmake. Invoke as:
#
#     cmake -S cmake/tests/gas_f16_probe -B build/gas_f16_probe
#
# Runs on the host compiler; there is no arm-none-eabi toolchain in this job,
# so it cannot assemble the witness instruction and does not try to. What it
# pins down is which flags the probe measures. See AmbiqAI/ns-cmsis-nn#427.
#
# The arch flags sit on a target that is never attached, so the host compiler
# is never asked to accept a Cortex-M flag it has no idea about.
#
# The end-to-end assemble-and-compare is proved by the toolchain jobs:
# toolchain-matrix-strict-link.yml's gcc-f16-probe (refusal, escape hatch, and
# the -B recipe) and unity-f16-exec-gcc-floor.yml (execution).

cmake_minimum_required(VERSION 3.19)
# C is enabled so add_library(STATIC) works; nothing is compiled.
project(ns_cmsis_nn_gas_f16_probe C)

get_filename_component(_repo_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(ARM_NN_ENABLE_F16 ON)
include("${_repo_root}/cmake/ns_cmsis_nn.cmake")

set(_failures "")
function(_fail msg)
  set(_failures "${_failures}\n  - ${msg}" PARENT_SCOPE)
endfunction()

function(_expect_opt opts wanted why)
  if(NOT "${wanted}" IN_LIST opts)
    _fail("${why}: '${wanted}' missing from the measured flags (${opts})")
    set(_failures "${_failures}" PARENT_SCOPE)
  endif()
endfunction()

function(_expect_no_opt opts unwanted why)
  if("${unwanted}" IN_LIST opts)
    _fail("${why}: '${unwanted}' present in the measured flags (${opts})")
    set(_failures "${_failures}" PARENT_SCOPE)
  endif()
endfunction()

# Directory options, including the generator-expression shapes the ethos-u
# toolchain file uses: one self-contained, one spanning two list elements.
add_compile_options(
  -DNS_PROBE_DIR_OPT
  "$<$<CONFIG:Debug>:-DNS_PROBE_CONFIG_GENEX>"
  "$<$<COMPILE_LANGUAGE:CXX>:-DNS_PROBE_A;-DNS_PROBE_B>"
  -DNS_PROBE_DIR_AFTER_GENEX)

# A dependency of the dependency: usage requirements are transitive, so the
# probe has to walk the whole link closure and not just the first hop.
add_library(_probe_deep INTERFACE)
target_compile_options(_probe_deep INTERFACE -DNS_PROBE_DEEP_OPT)

add_library(_probe_flags INTERFACE)
target_compile_options(_probe_flags INTERFACE -DNS_PROBE_IFACE_OPT)
target_compile_definitions(_probe_flags INTERFACE NS_PROBE_IFACE_DEF=1)
# A board target that injects a C header into every translation unit. The
# assembler witness has no use for it, so the probe drops it and keeps the
# directory it came with.
target_compile_options(_probe_flags INTERFACE -include foo.h)
target_include_directories(_probe_flags INTERFACE "${CMAKE_CURRENT_LIST_DIR}/iface_inc")
target_link_libraries(_probe_flags INTERFACE _probe_deep)

add_library(_probe_lib STATIC)
ns_cmsis_nn_attach(_probe_lib GROUPS softmax)
target_compile_options(_probe_lib PRIVATE -DNS_PROBE_OWN_OPT)
target_compile_definitions(_probe_lib PRIVATE NS_PROBE_OWN_DEF=1)
target_compile_options(_probe_lib PRIVATE -imacrosbar.h)
target_include_directories(_probe_lib PRIVATE "${CMAKE_CURRENT_LIST_DIR}/own_inc")
# Linked after the attach on purpose: that is the order nsx/CMakeLists.txt
# uses, and reading the link closure during attach would see nothing.
target_link_libraries(_probe_lib PUBLIC _probe_flags)

_ns_cmsis_nn_gas_f16_flags(_probe_lib _opts _genex_skipped)

_expect_opt("${_opts}" "-DNS_PROBE_IFACE_OPT"
            "options on a linked INTERFACE target are invisible to the probe")
_expect_opt("${_opts}" "-DNS_PROBE_IFACE_DEF=1"
            "INTERFACE_COMPILE_DEFINITIONS are not measured")
_expect_opt("${_opts}" "-DNS_PROBE_DEEP_OPT"
            "usage requirements are not walked transitively")
_expect_opt("${_opts}" "-DNS_PROBE_OWN_OPT"
            "the target's own COMPILE_OPTIONS are not measured")
_expect_opt("${_opts}" "-DNS_PROBE_OWN_DEF=1"
            "the target's own COMPILE_DEFINITIONS are not measured")
_expect_opt("${_opts}" "-DNS_PROBE_DIR_OPT"
            "directory COMPILE_OPTIONS are not measured")
_expect_opt("${_opts}" "-DNS_PROBE_DIR_AFTER_GENEX"
            "the generator-expression filter does not recover its bracket depth")
_expect_opt("${_opts}" "-I${CMAKE_CURRENT_LIST_DIR}/iface_inc"
            "INTERFACE_INCLUDE_DIRECTORIES are not measured")
_expect_opt("${_opts}" "-I${CMAKE_CURRENT_LIST_DIR}/own_inc"
            "the target's own INCLUDE_DIRECTORIES are not measured")
_expect_no_opt("${_opts}" "-include"
               "a split -include reaches the assembler witness")
_expect_no_opt("${_opts}" "foo.h"
               "the header of a split -include reaches the assembler witness")
_expect_no_opt("${_opts}" "-imacrosbar.h"
               "a joined -imacros reaches the assembler witness")

foreach(_opt IN LISTS _opts)
  if(_opt MATCHES "\\$<")
    _fail("a generator expression reached the probe command line: '${_opt}'")
  endif()
endforeach()
if(NOT _genex_skipped)
  _fail("skipped generator expressions were not reported, so the probe cannot say its view of the flags is partial")
endif()

# Architecture flags carried on a linked INTERFACE target rather than in
# CMAKE_C_FLAGS. Not attached: this asserts what the probe would measure,
# without handing a Cortex-M flag to whatever host compiler is running the
# test.
add_library(_probe_cpu INTERFACE)
target_compile_options(_probe_cpu INTERFACE
                       -mcpu=cortex-m55 -mfloat-abi=hard -mfpu=auto
                       -include foo.h)
add_library(_probe_arch STATIC)
target_sources(_probe_arch PRIVATE "${_repo_root}/Source/SoftmaxFunctions/arm_softmax_f16.c")
target_link_libraries(_probe_arch PRIVATE _probe_cpu)

_ns_cmsis_nn_gas_f16_flags(_probe_arch _arch_opts _arch_genex_skipped)
foreach(_flag -mcpu=cortex-m55 -mfloat-abi=hard -mfpu=auto)
  _expect_opt("${_arch_opts}" "${_flag}"
              "arch flags on a linked INTERFACE target are invisible to the probe")
endforeach()
_expect_no_opt("${_arch_opts}" "-include"
               "an injected header is kept alongside the arch flags")

# A directory option lands in a target's COMPILE_OPTIONS only if the target
# already existed when it was added. The probe runs at the end of the
# directory, by which time the directory property carries flags that the
# earlier target will never compile with -- including, here, a Cortex-M flag
# the host compiler running this test cannot accept.
add_library(_probe_before STATIC)
target_sources(_probe_before PRIVATE "${_repo_root}/Source/SoftmaxFunctions/arm_softmax_f16.c")

add_compile_options(-mcpu=cortex-m55)

add_library(_probe_after STATIC)
target_sources(_probe_after PRIVATE "${_repo_root}/Source/SoftmaxFunctions/arm_softmax_f16.c")

_ns_cmsis_nn_gas_f16_flags(_probe_before _before_opts _before_genex_skipped)
_ns_cmsis_nn_gas_f16_flags(_probe_after _after_opts _after_genex_skipped)
_expect_no_opt("${_before_opts}" "-mcpu=cortex-m55"
               "a directory option added after the target was created is measured against it")
_expect_opt("${_after_opts}" "-mcpu=cortex-m55"
            "a directory option added before the target was created is not measured")

# Scheduling is gated on a GNU compiler, so this half only asserts where the
# host compiler is one. Everything above is compiler-independent.
if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
  get_property(_scheduled GLOBAL PROPERTY NS_GAS_MVE_F16_CVT_SCHEDULED)
  if(NOT "_probe_lib" IN_LIST _scheduled)
    _fail("ns_cmsis_nn_attach did not invoke the assembler probe for a target carrying float16 sources")
  endif()
else()
  message(STATUS
    "gas f16 probe test: host compiler is ${CMAKE_C_COMPILER_ID}, not GNU; "
    "the invocation assertion is skipped (the flag assertions are not).")
endif()

if(_failures)
  message(FATAL_ERROR "gas f16 probe flag-view test FAILED:${_failures}")
endif()

message(STATUS "gas f16 probe flag-view OK: ${_opts}")
message(STATUS "gas f16 probe arch flag-view OK: ${_arch_opts}")
