#
# SPDX-FileCopyrightText: Copyright 2024-2026 Ambiq <opensource@ambiq.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# Zephyr-integration wiring test for zephyr/CMakeLists.txt.
#
# Runs entirely during CMake configure (no Zephyr SDK, no cross-compiler, no
# west, no toolchain). It stubs every zephyr_* macro the real
# zephyr/CMakeLists.txt calls so we can record exactly what sources, include
# paths, and compile definitions would be handed to a real Zephyr build.
#
# Invoke as:
#
#     cmake -S cmake/tests/zephyr_wiring -B build/zephyr_wiring
#
# A green configure proves:
#   - the Kconfig -> SSoT-group translation table is complete (every one of
#     the 17 group ids is referenced) and uses the exact group spellings the
#     SSoT exposes;
#   - turning every CONFIG_NS_CMSIS_NN_<knob>=y on resolves to all 206
#     sources via ns_cmsis_nn_attach();
#   - Include/ is exposed via zephyr_include_directories();
#   - CONFIG_NS_CMSIS_NN_REQUANTIZE_INLINE_ASSEMBLY=y propagates as the
#     CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY define;
#   - the module is guarded on CONFIG_NS_CMSIS_NN.
#
cmake_minimum_required(VERSION 3.15)
# C is enabled so add_library(STATIC) (used internally by the stubs) works.
project(ns_cmsis_nn_zephyr_wiring C)

get_filename_component(_repo_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

# ----------------------------------------------------------------------------
# Test harness: collect failures, report them all at once.
# ----------------------------------------------------------------------------
set(_failures "")
function(_fail msg)
  set(_failures "${_failures}\n  - ${msg}" PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------
# Zephyr build-system stubs. Each macro records its arguments into a
# well-known cache so the assertions below can inspect them.
# ----------------------------------------------------------------------------
set(STUB_LIBRARY_NAME "")
set(STUB_COMPILE_OPTIONS "")
set(STUB_GLOBAL_INCLUDE_DIRS "")
set(STUB_LIBRARY_INCLUDE_DIRS "")
set(STUB_COMPILE_DEFINITIONS "")

macro(zephyr_library)
  set(STUB_LIBRARY_NAME "zephyr_stub_lib")
  add_library(${STUB_LIBRARY_NAME} STATIC)
  # Mirror Zephyr's real contract: subsequent zephyr_library_* calls operate
  # on this library, exposed as ZEPHYR_CURRENT_LIBRARY.
  set(ZEPHYR_CURRENT_LIBRARY "${STUB_LIBRARY_NAME}")
endmacro()

macro(zephyr_library_compile_options)
  list(APPEND STUB_COMPILE_OPTIONS ${ARGN})
endmacro()

macro(zephyr_include_directories)
  list(APPEND STUB_GLOBAL_INCLUDE_DIRS ${ARGN})
endmacro()

macro(zephyr_library_include_directories)
  list(APPEND STUB_LIBRARY_INCLUDE_DIRS ${ARGN})
endmacro()

macro(zephyr_compile_definitions)
  list(APPEND STUB_COMPILE_DEFINITIONS ${ARGN})
endmacro()

# ----------------------------------------------------------------------------
# Stub the Zephyr environment that zephyr/CMakeLists.txt references.
# ----------------------------------------------------------------------------
set(ZEPHYR_CURRENT_MODULE_DIR "${_repo_root}")
# The real Zephyr build sets ZEPHYR_CMSIS_MODULE_DIR to the cmsis module root.
# We don't care where it points for wiring purposes — only that the macro
# captures it. Use a sentinel so the assertions can identify it.
set(ZEPHYR_CMSIS_MODULE_DIR "${CMAKE_CURRENT_BINARY_DIR}/__fake_cmsis_module__")

# ----------------------------------------------------------------------------
# Case 1: CONFIG_NS_CMSIS_NN not set — module must be a no-op.
# ----------------------------------------------------------------------------
include("${_repo_root}/zephyr/CMakeLists.txt")
if(STUB_LIBRARY_NAME)
  _fail("guard regression: zephyr/CMakeLists.txt created a library even with CONFIG_NS_CMSIS_NN unset")
endif()

# ----------------------------------------------------------------------------
# Case 2: every CONFIG_NS_CMSIS_NN_<knob>=y. The renamed knobs only — the
# deprecated aliases are Kconfig-level select sugar and never appear as
# CONFIG_NS_CMSIS_NN_<DEPRECATED>=y in a real build (Kconfig rewrites them).
# ----------------------------------------------------------------------------
set(_renamed_knobs
    ACTIVATION
    BASICMATHSNN
    COMPARISON
    CONCATENATION
    CONVOLUTION
    FULLYCONNECTED
    GATHER
    LSTM
    NNSUPPORT
    PAD
    POOLING
    QUANTIZATION
    RESHAPE
    SOFTMAX
    STRIDEDSLICE
    SVDF
    TRANSPOSE
)

set(CONFIG_NS_CMSIS_NN 1)
set(CONFIG_NS_CMSIS_NN_REQUANTIZE_INLINE_ASSEMBLY 1)
foreach(_k IN LISTS _renamed_knobs)
  set(CONFIG_NS_CMSIS_NN_${_k} 1)
endforeach()

include("${_repo_root}/zephyr/CMakeLists.txt")

# ----------------------------------------------------------------------------
# Assertions on the captured wiring.
# ----------------------------------------------------------------------------

# Library was created.
if(NOT STUB_LIBRARY_NAME)
  _fail("zephyr_library() was never called; wiring not reached")
endif()

# All 206 sources attached.
if(TARGET ${STUB_LIBRARY_NAME})
  get_target_property(_srcs ${STUB_LIBRARY_NAME} SOURCES)
  if(NOT _srcs)
    _fail("no sources attached to the Zephyr library")
  else()
    list(LENGTH _srcs _nsrc)
    if(NOT _nsrc EQUAL 206)
      _fail("Zephyr wiring attached ${_nsrc} sources; expected 206 (SSoT ALL).\n      A Kconfig->group mapping typo will show up here.")
    endif()
  endif()
endif()

# Public Include/ exposed globally (consumers include arm_nn*.h directly).
list(FIND STUB_GLOBAL_INCLUDE_DIRS "${_repo_root}/Include" _inc_idx)
if(_inc_idx EQUAL -1)
  _fail("Include/ was not exposed via zephyr_include_directories(); got: ${STUB_GLOBAL_INCLUDE_DIRS}")
endif()

# CMSIS Core/Include forwarded to the library.
list(FIND STUB_LIBRARY_INCLUDE_DIRS "${ZEPHYR_CMSIS_MODULE_DIR}/CMSIS/Core/Include" _cmsis_idx)
if(_cmsis_idx EQUAL -1)
  _fail("CMSIS Core/Include not forwarded; got: ${STUB_LIBRARY_INCLUDE_DIRS}")
endif()

# Inline-asm requantize define propagated globally.
list(FIND STUB_COMPILE_DEFINITIONS "CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY" _def_idx)
if(_def_idx EQUAL -1)
  _fail("CONFIG_NS_CMSIS_NN_REQUANTIZE_INLINE_ASSEMBLY did not propagate as CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY")
endif()

# ----------------------------------------------------------------------------
# Translation-table coverage: every renamed Kconfig knob in this test must
# also appear in the real zephyr/CMakeLists.txt foreach() pair list. Read
# the file and grep for each knob to catch any typo (e.g. "FULLYCONNETED").
# ----------------------------------------------------------------------------
file(READ "${_repo_root}/zephyr/CMakeLists.txt" _zcml)
foreach(_k IN LISTS _renamed_knobs)
  string(FIND "${_zcml}" "\"${_k};" _hit)
  if(_hit EQUAL -1)
    _fail("translation table is missing Kconfig knob '${_k}' (or its spelling drifted)")
  endif()
endforeach()

# Coverage cross-check: the translation table must reference EVERY group the
# SSoT exposes. Pull the group list and verify each lowercase id appears as
# the value half of a "KNOB;group" pair.
include("${_repo_root}/cmake/ns_cmsis_nn.cmake")
ns_cmsis_nn_groups(_all_groups)
foreach(_g IN LISTS _all_groups)
  string(FIND "${_zcml}" ";${_g}\"" _hit)
  if(_hit EQUAL -1)
    _fail("SSoT group '${_g}' has no Kconfig knob in the Zephyr translation table")
  endif()
endforeach()

# ----------------------------------------------------------------------------
# Report.
# ----------------------------------------------------------------------------
if(_failures)
  message(FATAL_ERROR "Zephyr wiring test FAILED:${_failures}")
endif()

message(STATUS "Zephyr wiring OK: 17 knobs translated, 206 sources attached, inline-asm define propagated.")
